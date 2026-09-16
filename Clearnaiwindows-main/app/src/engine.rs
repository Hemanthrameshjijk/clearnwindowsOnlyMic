//! Real Windows audio engine: constructs every DSP stage up front and
//! wires the mic pipeline over `dsp_core` ring buffers, starting the real
//! WASAPI threads via `audio-io`.
//!
//! AEC and the hardware-output loopback tap that fed its reference signal
//! have been removed from this build - the loopback tap was a persistent,
//! unresolved source of instability (`AUDCLNT_E_DEVICE_INVALIDATED`/
//! `_IN_USE`, repeated event-wait timeouts) on real hardware, and AEC
//! defaulted off anyway per the original spec, so nothing depended on it
//! being present. `crates/aec` and `audio_io::loopback` still exist and are
//! still tested on their own; this module simply no longer wires either of
//! them into the running pipeline. Speaker-outbound (Pipeline 2) was
//! already removed earlier for the same "focus on the mic path, remove
//! unused instability" reasoning - see the removal note further down.
//!
//! This entire module only exists on Windows (`#![cfg(windows)]`) because
//! it is the only place in this crate that touches `audio_io::capture` and
//! `audio_io::render`, both of which are themselves Windows-only
//! (`#![cfg(windows)]` in `audio-io` itself). On any other platform,
//! `main.rs` uses an inert `EngineHandles` with no threads at all instead
//! of calling into this module - see `main.rs`.
//!
//! # What has and hasn't been verified
//!
//! This module type-checks against the real public APIs of every crate it
//! calls into (`cargo check --target x86_64-pc-windows-gnu`). It has never
//! been run: no real WASAPI device, no real `weya_nc.dll`, no real VB-Cable
//! install. Every claim about behavior below is a claim about what the code
//! is *written* to do, not what has been *observed* to happen.

#![cfg(windows)]

use crate::settings::Settings;
use crate::shared::{DeviceList, DeviceOption, EngineHandles, LiveDeviceSwitcher, NoOpStage};
use anyhow::{Context, Result};
use audio_io::capture::MicCapture;
use audio_io::devices;
use audio_io::render::HardwareRender;
use dsp_core::{make_ring_buffer, run_if_enabled, Consumer, Producer, Stage, FRAME_SAMPLES};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// Frame deadline in nanoseconds (10ms @ 48kHz), used to classify realtime
/// status against measured processing-time percentiles.
const DEADLINE_NS: u64 = (FRAME_SAMPLES as u64 * 1_000_000_000) / dsp_core::SAMPLE_RATE_HZ as u64;

/// One end of a ring buffer, shared between the pipeline thread that reads/
/// writes it every frame and `LiveAudioSwitcher`, which replaces the whole
/// `Option` when the user picks a different device for that role. `rtrb`'s
/// `Consumer`/`Producer` are consumed by move and cannot be "reused" after
/// their paired half is dropped (see module docs below) - a device switch
/// always builds a brand new ring buffer pair and swaps this pointer to the
/// new half, never mutates the old one in place.
///
/// Locking here is a deliberate, small departure from this project's
/// otherwise strict "no locks on the realtime audio path" rule: device
/// switching is a rare, user-initiated action (not per-frame data), and the
/// critical section is a single `Option::as_mut()` + `pop`/`push` per
/// sample - not spinning, not blocking on I/O, and never contended except
/// for the brief instant a switch is in flight. This mirrors how
/// `dsp_core::SharedPreset`/`StageToggle` already accept a small amount of
/// non-realtime-pure plumbing for rare live-control-surface actions.
type SwapConsumer = Arc<Mutex<Option<Consumer<f32>>>>;
type SwapProducer = Arc<Mutex<Option<Producer<f32>>>>;

fn swap_consumer(initial: Option<Consumer<f32>>) -> SwapConsumer {
    Arc::new(Mutex::new(initial))
}

fn swap_producer(initial: Option<Producer<f32>>) -> SwapProducer {
    Arc::new(Mutex::new(initial))
}

/// Owns every device-role live handle that a GUI device-picker change can
/// individually stop/restart, plus the swappable ring-buffer ends the
/// permanently-running mic/speaker pipeline threads read/write through.
///
/// # Why this replaces the old "just keep it alive in a Vec" approach
///
/// An earlier `LiveHandle`/`BootstrapOutcome::live` existed for the *non*-
/// switchable roles this app used to have (the loopback tap, the speaker
/// path's fixed hardware render) - both have since been removed, so that
/// bucket no longer exists at all (see this module's top-of-file doc
/// comment). What remains is only the four genuinely switchable roles
/// below. A device picker change needs a named, individually-replaceable
/// slot per
/// role. `Mutex<Option<T>>` per role (rather than, say, an enum-keyed map)
/// was chosen because there are exactly four fixed, statically-known roles
/// with different concrete types (`MicCapture` for two of them,
/// `HardwareRender` for the other two) - a map would need a trait object or
/// an enum anyway, with no benefit over four plain named fields.
///
/// # The ring-buffer wrinkle
///
/// The mic pipeline thread (`spawn_mic_pipeline`) runs for the entire
/// process lifetime regardless of whether any device is configured for its
/// role yet - the speaker-outbound pipeline has been removed for this
/// session (see `start()`) -
/// this is what makes "select a virtual mic target for the first time
/// without ever having configured one" and "switch it live later" the same
/// code path. They read/write through the `SwapConsumer`/`SwapProducer`
/// handles below instead of owning a `Consumer`/`Producer` directly, so a
/// device switch can point them at a freshly-created ring buffer half
/// without the pipeline thread itself needing to know a switch happened.
pub struct LiveAudioSwitcher {
    physical_mic: Mutex<Option<MicCapture>>,
    virtual_mic_render: Mutex<Option<HardwareRender>>,
    monitor_render: Mutex<Option<HardwareRender>>,

    mic_in_swap: SwapConsumer,
    mic_out_swap: SwapProducer,
    monitor_out_swap: SwapProducer,

    /// Shared with every WASAPI thread this switcher (re)starts, so overrun/
    /// underrun telemetry keeps accumulating into the same counters across a
    /// live device switch rather than resetting.
    counters: Arc<dsp_core::FrameCounters>,
}

impl LiveAudioSwitcher {
    fn stop_and_replace_render(slot: &Mutex<Option<HardwareRender>>, new: Option<HardwareRender>) {
        let old = std::mem::replace(&mut *slot.lock().unwrap(), new);
        if let Some(old) = old {
            old.stop();
        }
    }

    fn stop_and_replace_mic(slot: &Mutex<Option<MicCapture>>, new: Option<MicCapture>) {
        let old = std::mem::replace(&mut *slot.lock().unwrap(), new);
        if let Some(old) = old {
            old.stop();
        }
    }
}

impl LiveDeviceSwitcher for LiveAudioSwitcher {
    /// Restarts physical mic capture on `device_id` (or the system default
    /// if `None`), reconnecting the mic pipeline thread's input to a fresh
    /// ring buffer. The new capture is started (and its producer wired into
    /// `mic_in_swap`) *before* the old one is stopped, so the pipeline
    /// thread never sees a spurious "gone" gap wider than the time it takes
    /// WASAPI to open the new device.
    fn set_physical_mic(&self, device_id: Option<String>) {
        let (producer, consumer) = make_ring_buffer(2.0);
        let started = match device_id.as_deref() {
            Some(id) => MicCapture::start_on_device_id(id, producer, self.counters.clone()),
            None => MicCapture::start_default(producer, self.counters.clone()),
        };
        match started {
            Ok(capture) => {
                *self.mic_in_swap.lock().unwrap() = Some(consumer);
                Self::stop_and_replace_mic(&self.physical_mic, Some(capture));
            }
            Err(e) => {
                crate::log_error!(
                    "[clearnairt] failed to switch physical microphone live to {device_id:?} \
                     ({e:#}); the previous physical microphone (if any) keeps running."
                );
            }
        }
    }

    /// Restarts the virtual-mic-target render device, reconnecting the mic
    /// pipeline thread's `mic_out` output to a fresh ring buffer. `None`
    /// tears the render device down entirely (matching "no virtual mic
    /// target selected" at boot) - the mic pipeline thread keeps running
    /// (e.g. for the monitor path) but has nowhere configured to forward to.
    fn set_virtual_mic_target(&self, device_id: Option<String>) {
        match device_id {
            Some(id) => {
                let (producer, consumer) = make_ring_buffer(2.0);
                match HardwareRender::start_on_device_id(&id, consumer, self.counters.clone()) {
                    Ok(render) => {
                        *self.mic_out_swap.lock().unwrap() = Some(producer);
                        Self::stop_and_replace_render(&self.virtual_mic_render, Some(render));
                    }
                    Err(e) => {
                        crate::log_error!(
                            "[clearnairt] failed to switch virtual mic target live to '{id}' \
                             ({e:#}); the previous virtual mic target (if any) keeps running."
                        );
                    }
                }
            }
            None => {
                *self.mic_out_swap.lock().unwrap() = None;
                Self::stop_and_replace_render(&self.virtual_mic_render, None);
            }
        }
    }

    /// Speaker-outbound (Pipeline 2) is disabled for this session - see
    /// `start()`. Deliberately does not open the requested device at all
    /// (unlike before, when this would open a capture device that fed a
    /// pipeline thread which no longer exists) - an honest no-op instead of
    /// silently wasting a device handle on nothing.
    fn set_virtual_speaker_source(&self, _device_id: Option<String>) {}

    /// Restarts the monitor render device, reconnecting the mic pipeline
    /// thread's monitor output to a fresh ring buffer. `None` tears the
    /// monitor render device down entirely.
    fn set_monitor_device(&self, device_id: Option<String>) {
        match device_id {
            Some(id) => {
                let (producer, consumer) = make_ring_buffer(2.0);
                match HardwareRender::start_on_device_id(&id, consumer, self.counters.clone()) {
                    Ok(render) => {
                        *self.monitor_out_swap.lock().unwrap() = Some(producer);
                        Self::stop_and_replace_render(&self.monitor_render, Some(render));
                    }
                    Err(e) => {
                        crate::log_error!(
                            "[clearnairt] failed to switch monitor output live to '{id}' ({e:#}); \
                             the previous monitor output (if any) keeps running."
                        );
                    }
                }
            }
            None => {
                *self.monitor_out_swap.lock().unwrap() = None;
                Self::stop_and_replace_render(&self.monitor_render, None);
            }
        }
    }

    fn enumerate_devices(&self) -> DeviceList {
        enumerate_devices()
    }
}

/// Result of starting the engine: the shared toggle/metrics handles for the
/// GUI, the enumerated device lists for the two device pickers, and any
/// non-fatal startup warnings. Every real OS thread handle this engine
/// starts is already kept alive by `handles.device_switcher`'s own fields
/// (`LiveAudioSwitcher`'s `Mutex<Option<...>>`s) for the process lifetime -
/// there is no longer a separate "misc handles" bucket here (there used to
/// be one, `live: Vec<LiveHandle>`, for the loopback tap and the fixed
/// speaker-outbound hardware render; both were removed, so it would always
/// have been empty).
pub struct BootstrapOutcome {
    pub handles: EngineHandles,
    pub devices: DeviceList,
}

/// Enumerates render/capture devices for the GUI's device pickers. Returns
/// empty lists (never an error to the caller) if enumeration itself fails -
/// the GUI must still come up, just with no devices to pick from.
pub fn enumerate_devices() -> DeviceList {
    let render = devices::list_render_devices()
        .unwrap_or_else(|e| {
            crate::log_error!("[clearnairt] WARNING: failed to enumerate render devices: {e:#}");
            Vec::new()
        })
        .into_iter()
        .map(|(id, label)| DeviceOption { id, label })
        .collect();
    let capture = devices::list_capture_devices()
        .unwrap_or_else(|e| {
            crate::log_error!("[clearnairt] WARNING: failed to enumerate capture devices: {e:#}");
            Vec::new()
        })
        .into_iter()
        .map(|(id, label)| DeviceOption { id, label })
        .collect();
    DeviceList { render, capture }
}

/// Auto-selects the "virtual mic target" render device only if exactly one
/// render endpoint looks like a VB-Cable-style "CABLE Input" - see
/// `docs/VIRTUAL_DEVICES.md`. Ambiguous (zero or more than one match) means
/// no safe default exists and the user must choose explicitly.
pub fn default_virtual_mic_target(list: &DeviceList) -> Option<String> {
    let mut matches = list
        .render
        .iter()
        .filter(|d| devices::name_contains_cable_input(&d.label));
    let first = matches.next()?;
    if matches.next().is_some() {
        None
    } else {
        Some(first.id.clone())
    }
}

/// Auto-selects the "virtual speaker source" capture device only if exactly
/// one capture endpoint looks like a VB-Cable-style "CABLE Output". In the
/// common case of a single VB-Cable install, this is also the device the
/// mic-target render already paired with, which is exactly the collision
/// `docs/VIRTUAL_DEVICES.md` documents - callers should treat an
/// auto-selected speaker-source id that equals the auto-selected mic-target
/// id's paired endpoint as a signal to warn the user to install a *second*
/// virtual cable instance.
pub fn default_virtual_speaker_source(list: &DeviceList) -> Option<String> {
    let mut matches = list
        .capture
        .iter()
        .filter(|d| devices::name_contains_cable_output(&d.label));
    let first = matches.next()?;
    if matches.next().is_some() {
        None
    } else {
        Some(first.id.clone())
    }
}

/// Starts the full engine: loads BVC (or falls back to a real no-op stage,
/// honestly reflected in `EngineHandles::mic_bvc_available`/
/// `speaker_bvc_available`), constructs every
/// DSP stage up front (regardless of initial toggle state), wires the three
/// required pipelines, and starts their WASAPI threads.
///
/// `virtual_mic_target_id` / `virtual_speaker_source_id` are the
/// user-selected (or auto-detected) device ids for the two virtual-cable
/// roles described in `docs/VIRTUAL_DEVICES.md`. Either may be `None`, in
/// which case the corresponding half of the pipeline is not started and a
/// warning is added to `BootstrapOutcome`'s handles - this is not an error,
/// since a first-time user has not necessarily installed/configured virtual
/// cables yet.
pub fn start(
    bvc_assets_dir: &Path,
    settings: &Settings,
    devices_list: DeviceList,
    virtual_mic_target_id: Option<&str>,
    virtual_speaker_source_id: Option<&str>,
) -> Result<BootstrapOutcome> {
    let mut warnings = Vec::new();

    // --- BVC: one real load attempt per independent pipeline instance. ---
    // `bvc_assets_dir` is `setup::app_data_dir()` (`%LOCALAPPDATA%\ClearNAI`),
    // where `main.rs::bootstrap` has already self-extracted the embedded
    // `weya_nc.dll` + model bundle via `setup::ensure_bvc_assets_extracted`
    // before calling here - no more "next to the exe" file placement.
    // Mic-path BVC gates the GUI's single "BVC" toggle and its availability
    // flag; the speaker-path instance is wholly independent (its own DLL
    // session), matching the project's "always construct every stage, even
    // ones behind a toggle that starts off" rule, and its own honest
    // fallback if loading fails a second time (e.g. DLL present but a
    // transient session-creation failure) is a plain no-op, not a crash.
    let mic_bvc_load = bvc_hush::HushBvcStage::try_load(bvc_assets_dir);
    let (mic_bvc_available, mic_bvc_unavailable_reason) = match &mic_bvc_load {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    };
    // BVC calls into a closed-source DLL (`weya_nc.dll`) this crate does not
    // control. Real hardware finding: the mic pipeline thread can stop
    // draining its ring buffer entirely for a minute-plus with no error
    // from the WASAPI capture thread at all - i.e. something in the
    // pipeline blocked forever rather than erroring, and BVC's FFI call is
    // the only link in the chain that isn't our own pure-Rust code. Wrap it
    // in a `WatchdogStage` so a wedged call gets a bounded timeout (well
    // above the 10ms realtime deadline, since the DLL is allowed to be slow,
    // just not infinite) instead of freezing the whole pipeline thread
    // silently and permanently.
    const BVC_WATCHDOG_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);
    let mic_bvc_stage: Box<dyn Stage> = match mic_bvc_load {
        Ok(stage) => Box::new(dsp_core::WatchdogStage::new(
            "mic BVC",
            Box::new(stage),
            BVC_WATCHDOG_TIMEOUT,
            |msg| crate::log_error!("[clearnairt] {msg}"),
            || Box::new(audio_io::mmcss::elevate_current_thread()),
        )),
        Err(_) => Box::new(NoOpStage("BVC (unavailable)")),
    };
    // --- Speaker-outbound (Pipeline 2) has been deliberately disabled for
    // this session, per a scoped decision to focus exclusively on the mic
    // path: capture -> AEC/Noise/BVC/Studio -> virtual mic. Nothing below
    // loads a second BVC session, opens the virtual-speaker-source capture
    // device, or renders to real hardware for the outbound direction - see
    // the removed `spawn_speaker_pipeline` (kept only in git history) and
    // the honest "unavailable" status surfaced to the GUI here instead of
    // silently trying and failing. The mic-side BVC/AEC/loopback-tap code
    // above and below this block is untouched. ---
    let speaker_bvc_available = false;
    let speaker_bvc_unavailable_reason =
        Some("Speaker-outbound processing is disabled for this session (mic-only focus).".to_string());

    // --- Noise: RNNoise for both paths, per spec default. DeepFilter is a
    // real, supported `NoiseEngineKind` in `noise-engine`, but is not wired
    // to anything reachable from this GUI - see crate-level docs below for
    // why (no bundled model file). Not calling `build(DeepFilter, ...)`
    // anywhere is intentional, not an oversight. ---
    let mic_noise_stage: Box<dyn Stage> =
        noise_engine::build(noise_engine::NoiseEngineKind::RnNoise, None)
            .context("constructing mic-path RNNoise stage")?;

    // --- Studio: constructed from the persisted preset preference (default
    // `Natural`, the gentlest of the three active presets - see
    // `settings::default_studio_preset`). Both mic and speaker paths now get
    // their own independent Studio stage; live preset changes are handled by
    // each pipeline thread swapping in a freshly-constructed `StudioStage`
    // when its `SharedPreset` changes (see the mic/speaker pipeline loops
    // below) - `StudioStage` itself has no live "change preset" method. ---
    let mic_studio_stage: Box<dyn Stage> = Box::new(studio_dsp::StudioStage::new(
        crate::shared::studio_preset_from_index(settings.mic_studio_preset),
    ));

    let mut handles = EngineHandles::new_from_settings(
        settings,
        mic_bvc_available,
        mic_bvc_unavailable_reason,
        speaker_bvc_available,
        speaker_bvc_unavailable_reason,
        Vec::new(),
    );

    // --- Ring buffers. 2 seconds of headroom at 48kHz is generous relative
    // to the 10ms frame size; sized once here, never resized (a live device
    // switch later builds a brand new pair - see `LiveAudioSwitcher`). ---
    let (mic_in_p, mic_in_c) = make_ring_buffer(2.0);

    // --- Pipeline 1 (mic): real physical mic -> ring buffer. Uses the
    // user-selected physical capture device (`settings.physical_mic_device_id`,
    // e.g. a Bluetooth headset mic instead of a built-in mic) if one was
    // chosen, mirroring exactly how `virtual_mic_target_id`/
    // `default_virtual_mic_target` pick between "auto/default" (`None`) and
    // "a specific device" (`Some(id)`) elsewhere in this function. Picking a
    // *different* device later in the GUI now takes effect live, via
    // `LiveAudioSwitcher::set_physical_mic` - this initial start is only
    // "next launch" in the sense that the persisted *starting* selection is
    // read once, here. ---
    let physical_mic_id = settings.physical_mic_device_id.clone();
    let mic_capture = match physical_mic_id.as_deref() {
        Some(id) => MicCapture::start_on_device_id(id, mic_in_p, handles.counters.clone())
            .context("starting mic capture on the selected physical microphone device")?,
        None => MicCapture::start_default(mic_in_p, handles.counters.clone()).context("starting default mic capture")?,
    };

    // --- Monitor (diagnostic/testing-only): an optional second, independent
    // render path that receives a *copy* of the mic pipeline's final
    // processed frame, wholly separate from the virtual-mic-target pipeline.
    // Only started at all if the user has selected a monitor output device;
    // see `spawn_mic_pipeline` for where the frame is actually copied to
    // both downstream producers (the SPSC ring buffers used throughout this
    // crate cannot fan a single one out to two consumers, hence a wholly
    // separate ring buffer pair here). ---
    let (monitor_render, monitor_out_p) = match settings.monitor_device_id.as_deref() {
        Some(id) => {
            let (p, c) = make_ring_buffer(2.0);
            match HardwareRender::start_on_device_id(id, c, handles.counters.clone()) {
                Ok(render) => (Some(render), Some(p)),
                Err(e) => {
                    // Exactly as graceful as the "no virtual mic target
                    // selected" case below: a monitor device that fails to
                    // open must never take down the rest of the pipeline.
                    warnings.push(format!(
                        "Monitor output device could not be started ({e:#}); local monitoring \
                         is unavailable, but the rest of the pipeline is unaffected."
                    ));
                    (None, None)
                }
            }
        }
        None => (None, None),
    };

    // --- Pipeline 1 continued: engine -> user-selected virtual mic target,
    // and/or -> the monitor path started above. These two downstream
    // producers are independent of each other by construction (separate
    // ring buffer pairs). Unlike before, the mic pipeline thread is now
    // *always* spawned (see `spawn_mic_pipeline` below) regardless of
    // whether either downstream role is configured yet, so that selecting
    // one for the first time via the GUI (a live switch, not just a
    // next-launch preference) has a running thread to reconnect to. ---
    let (virtual_mic_render, mic_out_p) = match virtual_mic_target_id {
        Some(id) => {
            let (p, c) = make_ring_buffer(2.0);
            match HardwareRender::start_on_device_id(id, c, handles.counters.clone()) {
                Ok(render) => (Some(render), Some(p)),
                Err(e) => {
                    warnings.push(format!(
                        "Failed to start render into the selected virtual-mic-target device ({e:#}); \
                         the processed mic pipeline will keep running for the monitor path (if \
                         configured), but nothing is being forwarded to a virtual mic."
                    ));
                    (None, None)
                }
            }
        },
        None => {
            warnings.push(
                "No virtual mic target device selected - the processed mic signal is not being \
                 forwarded to a virtual microphone. Install a virtual audio cable and select its \
                 render endpoint in the GUI (see docs/VIRTUAL_DEVICES.md), or use the Monitor \
                 toggle to hear the processed signal directly without one."
                    .to_string(),
            );
            (None, None)
        }
    };

    let mic_in_swap = swap_consumer(Some(mic_in_c));
    let mic_out_swap = swap_producer(mic_out_p);
    let monitor_out_swap = swap_producer(monitor_out_p);

    // --- Driver-transport (see docs/DRIVER_PLAN.md): writes the mic
    // pipeline's final processed frame into a named shared-memory section,
    // ready for a future ClearNAI virtual audio driver to read - see
    // `driver_transport`'s crate docs for the exact wire format. No reader
    // exists yet (no driver has been built), so this is a one-sided write
    // for now; failure to create the section is non-fatal (same graceful-
    // degradation pattern as a missing virtual-mic-target device above) -
    // the mic pipeline must keep running with or without this. ---
    let driver_writer = match driver_transport::SharedRingWriter::create(
        driver_transport::MIC_OUTPUT_SECTION_NAME,
        driver_transport::DEFAULT_CAPACITY_FRAMES,
        FRAME_SAMPLES as u32,
    ) {
        Ok(writer) => Some(writer),
        Err(e) => {
            warnings.push(format!(
                "Driver-transport shared section could not be created ({e:#}); no effect on \
                 normal operation (no driver reads it yet - see docs/DRIVER_PLAN.md)."
            ));
            None
        }
    };

    spawn_mic_pipeline(
        mic_in_swap.clone(),
        mic_out_swap.clone(),
        monitor_out_swap.clone(),
        handles.clone(),
        mic_noise_stage,
        mic_bvc_stage,
        mic_studio_stage,
        driver_writer,
    )?;

    // --- Pipeline 2 (speaker outbound) is disabled for this session - see
    // the block above this function's BVC loading. No hardware render is
    // opened for it, no virtual-speaker-source capture device is opened
    // (even if one is configured in settings/GUI), and no pipeline thread
    // is spawned. `virtual_speaker_source_id` is intentionally unused here.
    let _ = virtual_speaker_source_id;

    handles.device_switcher = Arc::new(LiveAudioSwitcher {
        physical_mic: Mutex::new(Some(mic_capture)),
        virtual_mic_render: Mutex::new(virtual_mic_render),
        monitor_render: Mutex::new(monitor_render),
        mic_in_swap,
        mic_out_swap,
        monitor_out_swap,
        counters: handles.counters.clone(),
    });

    handles.warnings = std::sync::Arc::new(warnings);

    Ok(BootstrapOutcome {
        handles,
        devices: devices_list,
    })
}

/// Blocks (short-sleeping between polls - this is a plain consumer thread,
/// not the realtime WASAPI callback itself, so sleeping here is fine) until
/// `out` is fully filled from `consumer`.
///
/// Unlike the pre-live-switching version of this function, this never gives
/// up and returns `false`: `consumer` is a `SwapConsumer`, and its current
/// `None` state (no device configured for this role yet, or momentarily
/// mid-switch) is an ordinary, expected, *temporary* condition now - not a
/// sign the producer is gone forever - so the pipeline thread must keep
/// waiting rather than exit. A device permanently going away without a
/// replacement (e.g. unplugged) looks the same from here: silent underrun
/// forever, not a crash, matching this project's "never emit garbage,
/// silence on underrun" rule elsewhere.
/// How much buffered backlog to tolerate in `mic_in` before trimming it back
/// down - deliberately several frame periods (~40ms) above the nominal
/// single-frame 10ms budget, so ordinary scheduling jitter alone never
/// triggers a trim, but far short of `make_ring_buffer`'s full 2-second
/// capacity, which otherwise lets backlog grow unbounded.
const MAX_TOLERATED_BACKLOG_SAMPLES: usize = FRAME_SAMPLES * 4;

/// Clock-drift correction: the capture device (physical mic) and the
/// eventual render device downstream of this pipeline (virtual cable) run on
/// two independent hardware/driver clocks, which are never perfectly
/// matched. Left unchecked, any tiny rate mismatch between them accumulates
/// in `mic_in`'s ring buffer over the length of a whole conversation -
/// `pull_frame_blocking` drains strictly FIFO and has no notion of "too
/// old," so nothing here previously stopped it from ever queuing samples all
/// the way up to the full 2-second buffer capacity, which is exactly the
/// "delay that grows the longer I talk" symptom observed live. Trimming the
/// backlog back down to a small bounded target every time it grows past that
/// keeps end-to-end latency roughly constant instead of monotonically
/// growing, at the cost of a few dropped (skipped) samples whenever a trim
/// actually fires - inaudible compared to a multi-second delay.
fn trim_backlog(consumer: &SwapConsumer, counters: &dsp_core::FrameCounters) {
    let mut guard = consumer.lock().unwrap();
    let Some(c) = guard.as_mut() else { return };
    let backlog = c.slots();
    if backlog <= MAX_TOLERATED_BACKLOG_SAMPLES {
        return;
    }
    let to_drop = backlog - MAX_TOLERATED_BACKLOG_SAMPLES;
    let mut dropped = 0usize;
    for _ in 0..to_drop {
        if c.pop().is_err() {
            break;
        }
        dropped += 1;
    }
    if dropped > 0 {
        counters.dropped_frames.fetch_add(1, Ordering::Relaxed);
        crate::log_error!(
            "[clearnairt] mic input backlog exceeded {MAX_TOLERATED_BACKLOG_SAMPLES} samples \
             (clock drift between capture and render devices); trimmed {dropped} samples to keep \
             latency bounded"
        );
    }
}

fn pull_frame_blocking(consumer: &SwapConsumer, out: &mut [f32], counters: &dsp_core::FrameCounters) {
    trim_backlog(consumer, counters);

    let mut n = 0;
    while n < out.len() {
        let popped = {
            let mut guard = consumer.lock().unwrap();
            guard.as_mut().and_then(|c| c.pop().ok())
        };
        match popped {
            Some(sample) => {
                out[n] = sample;
                n += 1;
            }
            None => thread::sleep(std::time::Duration::from_micros(200)),
        }
    }
}

/// Pushes `frame` into `producer`. A `None` swap (no device configured for
/// this role) silently discards the frame - there is nowhere for it to go,
/// which is the expected steady state before a device is ever selected for
/// this role, not an error.
fn push_frame(producer: &SwapProducer, frame: &[f32]) {
    let mut guard = producer.lock().unwrap();
    let Some(p) = guard.as_mut() else { return };
    push_frame_to_fixed(p, frame);
}

/// Same as `push_frame` but for a plain (non-swappable) `Producer<f32>` -
/// used for the speaker path's fixed hardware render output, which is not
/// one of the four live-switchable device-picker roles.
fn push_frame_to_fixed(producer: &mut Producer<f32>, frame: &[f32]) {
    for &sample in frame {
        if producer.push(sample).is_err() {
            // Downstream (render thread) isn't keeping up; drop the rest of
            // this frame rather than block the processing thread.
            crate::log_error!("[clearnairt] output ring buffer full, dropping frame");
            return;
        }
    }
}

/// Pipeline 1: mic capture -> Noise -> BVC -> Studio -> virtual mic render.
/// Runs on its own thread for the life of the process. (AEC and its
/// loopback-tap reference signal were removed - see this module's top-of-
/// file doc comment.)
///
/// Returns `Err` if the OS itself refuses to create the thread (e.g. thread-
/// handle exhaustion after many rapid device switches). This is the one
/// thread-spawn in the whole engine that used to `.expect()` on that
/// outcome - with `panic = "abort"` set in the release profile, a bare
/// `.expect()` here would take the *entire process* down instantly and
/// silently, unlike every other spawn in this codebase (WASAPI worker
/// threads, `WatchdogStage`'s worker), which already surface a `Result`
/// instead.
#[allow(clippy::too_many_arguments)]
fn spawn_mic_pipeline(
    mic_in: SwapConsumer,
    mic_out: SwapProducer,
    monitor_out: SwapProducer,
    handles: EngineHandles,
    mut noise_stage: Box<dyn Stage>,
    mut bvc_stage: Box<dyn Stage>,
    mut studio_stage: Box<dyn Stage>,
    driver_writer: Option<driver_transport::SharedRingWriter>,
) -> Result<()> {
    thread::Builder::new()
        .name("clearnai-mic-pipeline".into())
        .spawn(move || {
            // Best-effort: see `audio_io::mmcss` - this thread runs the
            // Noise/BVC/Studio chain every 10ms and is exactly the kind of
            // realtime work MMCSS's "Pro Audio" category exists for.
            // Failure is silent and harmless (stays at normal priority).
            let _mmcss_guard = audio_io::mmcss::elevate_current_thread();
            let mut frame = vec![0.0f32; FRAME_SAMPLES];
            let silence = vec![0.0f32; FRAME_SAMPLES];
            let mut last_studio_preset = handles.mic_studio_preset.load();
            // Always-on, not a user toggle: corrects for quiet mic input
            // before it ever reaches RNNoise/BVC - see `dsp_core::AutoGainStage`
            // docs for the real, confirmed finding motivating this (RNNoise
            // suppressed voice alongside noise when input level was low;
            // this removes the dependency on the user having set Windows'
            // mic gain correctly).
            let mut auto_gain = dsp_core::AutoGainStage::new();
            loop {
                pull_frame_blocking(&mic_in, &mut frame, &handles.counters);
                let started = Instant::now();

                // Deliberate, narrow exception to "no allocation in the
                // realtime hot path": a preset change is a rare, deliberate
                // user action (not per-frame data), so once per frame we
                // cheaply load the shared atomic and only on an actual
                // change do we allocate a fresh `StudioStage` and swap it
                // into this thread-local variable. The common case (no
                // change) is a single relaxed atomic load, same cost as any
                // other toggle check in this loop.
                let current_preset = handles.mic_studio_preset.load();
                if current_preset != last_studio_preset {
                    studio_stage = Box::new(studio_dsp::StudioStage::new(
                        crate::shared::studio_preset_from_index(current_preset),
                    ));
                    last_studio_preset = current_preset;
                }

                auto_gain.process(&mut frame);
                run_if_enabled(noise_stage.as_mut(), &handles.mic_noise, &mut frame);
                run_if_enabled(bvc_stage.as_mut(), &handles.mic_bvc, &mut frame);
                run_if_enabled(studio_stage.as_mut(), &handles.mic_studio, &mut frame);

                let peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
                handles.level.store(peak);

                let elapsed_ns = started.elapsed().as_nanos() as u64;
                handles.latency_log.record_ns(elapsed_ns);
                handles.counters.frames_processed.fetch_add(1, Ordering::Relaxed);
                if elapsed_ns > DEADLINE_NS {
                    handles.counters.deadline_misses.fetch_add(1, Ordering::Relaxed);
                }

                // Hand a *copy* of the one final processed frame to each
                // independent downstream path that's actually active. This
                // is the fan-out `dsp_core`'s SPSC ring buffers can't do on
                // their own (one producer, one consumer per ring buffer,
                // per `RingBuffer::new`'s docs) - each path got its own
                // ring buffer pair back in `engine::start`, and here we just
                // push the same samples into whichever of the (up to two)
                // producers exist.
                push_frame(&mic_out, &frame);
                // Real, tested write into the driver-transport shared
                // section (see docs/DRIVER_PLAN.md) - a genuine second
                // consumer of the same final processed frame, exactly like
                // `mic_out`/`monitor_out` above. `push_frame` never blocks
                // (drops + counts on a full ring), matching every other
                // sink on this thread; `None` (section creation failed at
                // startup) is a silent no-op, same graceful-degradation
                // pattern as `mic_out`'s own `None` case.
                if let Some(writer) = driver_writer.as_ref() {
                    writer.push_frame(&frame);
                }
                // When the Monitor toggle is off, push real silence instead
                // of skipping the push entirely, so the monitor render
                // thread doesn't starve/underrun-log just because the user
                // wants it muted rather than torn down. If no monitor device
                // is configured at all, `push_frame` on a `None` swap is
                // already a no-op either way.
                if handles.monitor.is_on() {
                    push_frame(&monitor_out, &frame);
                } else {
                    push_frame(&monitor_out, &silence);
                }
            }
        })
        .context("spawning mic pipeline thread")?;
    Ok(())
}

// Pipeline 2 (speaker outbound: virtual-speaker-source capture -> Noise ->
// BVC -> Studio -> hardware speaker render) has been removed for this
// session - see the "disabled for this session" block in `start()`. The mic
// pipeline (`spawn_mic_pipeline` above) and the read-only hardware-output
// loopback tap (`LoopbackTap`, used as AEC's reference on the mic path) are
// untouched.
