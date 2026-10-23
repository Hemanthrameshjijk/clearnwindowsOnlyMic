//! Real Windows audio engine: constructs every DSP stage up front and
//! wires the mic pipeline over `dsp_core` ring buffers, starting the real
//! WASAPI threads via `audio-io`. Also constructs the speaker-reference-
//! enrollment subsystem and the BVC backend router (Off/Hush/TSE) - see
//! `docs/TSE_ENROLLMENT.md`.
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
//! This module compiles and its unit tests pass on a real Windows machine
//! with a real `x86_64-pc-windows-gnu` toolchain (not just cross-compile
//! type-checked from Linux, as earlier revisions of this module were). It
//! has still never been *run* as the actual `clearnairt.exe` binary against
//! real hardware in that environment: no real WASAPI device, no real
//! `weya_nc.dll` session, no real VB-Cable install were exercised there.
//! Every claim about behavior below is a claim about what the code is
//! *written* to do, verified by compilation and unit tests, not something
//! observed running end-to-end against real audio hardware.

#![cfg(windows)]

use crate::settings::Settings;
use crate::shared::{DeviceList, DeviceOption, EngineHandles, LiveDeviceSwitcher, NoOpStage};
use anyhow::{Context, Result};
use audio_io::capture::MicCapture;
use audio_io::devices;
use audio_io::render::HardwareRender;
use dsp_core::{make_ring_buffer, run_if_enabled, Consumer, LatencyLog, Producer, Stage, FRAME_SAMPLES};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Instant;

/// TSE's own watchdog timeout. Deliberately far tighter than Hush's 500ms
/// (`BVC_WATCHDOG_TIMEOUT` below) - see the task this subsystem was built
/// for: "a 500ms timeout is far too large to represent acceptable 10ms audio
/// processing latency." 20ms is still generous relative to the 10ms frame
/// deadline (allows for one frame's worth of scheduling jitter on top of the
/// deadline itself), but is small enough that a genuinely wedged TSE backend
/// is caught and bypassed within roughly 2 frame periods, not 50.
const TSE_WATCHDOG_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(20);

/// TEMPORARY, wider timeout for `tse_bsrnn::backend::RealBsrnnTseBackend`
/// specifically, while it is still far from real-time (measured at
/// RTF~21-31x - a single 10ms frame can take 200-300ms). With the real
/// `TSE_WATCHDOG_TIMEOUT` (20ms) this backend would time out on its very
/// first few calls and hit `dsp_core::watchdog`'s `MAX_CONSECUTIVE_TIMEOUTS`
/// (3), permanently bypassing itself within ~60ms of being selected - which
/// would make it impossible to actually hear/exercise the real model at all,
/// not just slow. This wider timeout is a deliberate, temporary trade: it
/// lets the real model run to completion (with real, audible extra latency)
/// so it can be tested end-to-end, at the cost of not enforcing the true
/// realtime deadline yet. Must be tightened back down to (or replaced by)
/// `TSE_WATCHDOG_TIMEOUT` once the backend is actually optimized to meet it -
/// this is not a production-ready value.
const TSE_REAL_BACKEND_TEMPORARY_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(500);

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
    speaker_capture: Mutex<Option<MicCapture>>,
    speaker_render: Mutex<Option<HardwareRender>>,

    mic_in_swap: SwapConsumer,
    mic_out_swap: SwapProducer,
    monitor_out_swap: SwapProducer,
    speaker_in_swap: SwapConsumer,
    speaker_out_swap: SwapProducer,

    /// Shared with every MIC-path WASAPI thread this switcher (re)starts, so
    /// overrun/underrun telemetry keeps accumulating into the same counters
    /// across a live device switch rather than resetting.
    counters: Arc<dsp_core::FrameCounters>,
    /// Same idea, but for the speaker-outbound path's two roles - a
    /// SEPARATE counters instance from `counters` above, so the GUI can
    /// show each pipeline's drop/overrun/underrun telemetry independently
    /// rather than one merged number that can't tell you which direction
    /// is unhealthy.
    speaker_counters: Arc<dsp_core::FrameCounters>,
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

    /// Restarts the speaker-outbound path's capture device, reconnecting the
    /// speaker pipeline thread's input to a fresh ring buffer. `None` tears
    /// the capture device down entirely (matching "no speaker source
    /// selected" at boot) - the speaker pipeline thread keeps running but
    /// has nothing to read from.
    fn set_virtual_speaker_source(&self, device_id: Option<String>) {
        match device_id {
            Some(id) => {
                let (producer, consumer) = make_ring_buffer(2.0);
                match MicCapture::start_on_device_id(&id, producer, self.speaker_counters.clone()) {
                    Ok(capture) => {
                        *self.speaker_in_swap.lock().unwrap() = Some(consumer);
                        Self::stop_and_replace_mic(&self.speaker_capture, Some(capture));
                    }
                    Err(e) => {
                        crate::log_error!(
                            "[clearnairt] failed to switch speaker-source capture live to '{id}' \
                             ({e:#}); the previous speaker-source device (if any) keeps running."
                        );
                    }
                }
            }
            None => {
                *self.speaker_in_swap.lock().unwrap() = None;
                Self::stop_and_replace_mic(&self.speaker_capture, None);
            }
        }
    }

    /// Restarts the speaker-outbound path's render device (the user's real
    /// physical speakers/headphones), reconnecting the speaker pipeline
    /// thread's output to a fresh ring buffer. `None` means "system default
    /// output device" (matching `physical_mic`'s own convention), not
    /// "disabled" - there's always a sensible real device to fall back to.
    fn set_real_speaker_output(&self, device_id: Option<String>) {
        let (producer, consumer) = make_ring_buffer(2.0);
        let started = match device_id.as_deref() {
            Some(id) => HardwareRender::start_on_device_id(id, consumer, self.speaker_counters.clone()),
            None => HardwareRender::start_default(consumer, self.speaker_counters.clone()),
        };
        match started {
            Ok(render) => {
                *self.speaker_out_swap.lock().unwrap() = Some(producer);
                Self::stop_and_replace_render(&self.speaker_render, Some(render));
            }
            Err(e) => {
                crate::log_error!(
                    "[clearnairt] failed to switch real speaker output live to {device_id:?} \
                     ({e:#}); the previous speaker output (if any) keeps running."
                );
            }
        }
    }

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
/// cables yet. `real_speaker_output_id` is the speaker-outbound path's
/// render target (the user's real physical speakers/headphones) - `None`
/// means the system default output device, NOT "disabled" (unlike the two
/// virtual-cable ids above, there's always a sensible real device to fall
/// back to).
pub fn start(
    bvc_assets_dir: &Path,
    settings: &Settings,
    devices_list: DeviceList,
    virtual_mic_target_id: Option<&str>,
    virtual_speaker_source_id: Option<&str>,
    real_speaker_output_id: Option<&str>,
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
    // in a `WatchdogStage` so a wedged call gets a bounded timeout instead
    // of freezing the whole pipeline thread silently and permanently.
    //
    // Real, second hardware finding (this session): 500ms was too generous
    // once a SECOND, independent Hush session (the speaker path's own
    // instance) exists and runs concurrently with the mic path's -
    // `WatchdogStage::process` blocks the CALLING pipeline thread for up to
    // this whole duration on every single frame while waiting on the
    // worker-thread round trip (see `watchdog.rs`'s docs), so a genuinely
    // struggling backend (two closed-source DLL sessions competing for CPU)
    // can stall EITHER pipeline thread for up to 500ms repeatedly - long
    // enough that it stops draining its own mic/speaker capture ring buffer
    // in time, which is exactly the "ring buffer full, dropping frame"
    // flood observed live on both paths simultaneously. Tightened to 50ms:
    // still 5x the 10ms budget (room for real, ordinary DLL slowness), but
    // `MAX_CONSECUTIVE_TIMEOUTS` (3) now trips after 150ms of real stalling
    // instead of 1.5 real seconds, and once tripped the backend is
    // permanently bypassed (no more worker-thread round trips at all) -
    // stopping the compounding pileup far sooner than before.
    const BVC_WATCHDOG_TIMEOUT: std::time::Duration = std::time::Duration::from_millis(50);
    // Wrapped in a `WatchdogStage` for the same real reason every other BVC
    // backend in this file is: this is a closed-source DLL call, and real
    // hardware testing found it can block the pipeline thread entirely, for
    // a minute-plus, with no error. Removing the watchdog entirely (an
    // earlier revision of this code briefly did) traded that bounded-time
    // guarantee away just to fix a real but separate problem: once tripped,
    // the watchdog bypasses the backend permanently and silently, with
    // nothing surfaced outside `dsp_core` - so the GUI's "BVC: on" toggle
    // kept reading "on" even after Hush had stopped actually running,
    // making a real-world listening test impossible to trust. The actual
    // fix is `mic_hush_status` below: a `dsp_core::WatchdogStatus` handle
    // captured from the real stage BEFORE it's boxed and moved into the
    // pipeline, read by the GUI (see `EngineHandles::mic_hush_bypassed`) so
    // a genuinely stalled-and-bypassed Hush is reported honestly instead of
    // silently misrepresented as still running.
    // `None` when no real backend loaded at all (nothing to ever bypass -
    // `mic_bvc_available` already covers that case honestly); `Some` only
    // when a real `HushBvcStage` is actually running behind the watchdog.
    let mut mic_hush_status: Option<dsp_core::WatchdogStatus> = None;
    let mic_hush_stage: Box<dyn Stage> = match mic_bvc_load {
        Ok(stage) => {
            let watchdog = dsp_core::WatchdogStage::new(
                "mic BVC",
                Box::new(stage),
                BVC_WATCHDOG_TIMEOUT,
                |msg| crate::log_error!("[clearnairt] {msg}"),
                || Box::new(audio_io::mmcss::elevate_current_thread()),
            );
            mic_hush_status = Some(watchdog.status());
            Box::new(watchdog)
        }
        Err(_) => Box::new(NoOpStage("BVC (unavailable)")),
    };

    // --- DeepFilterNet: real, official model (see
    // `noise_engine::load_bundled_deep_filter`'s docs for its verified
    // provenance), same generic noise/other-voice suppression category as
    // Hush, a different architecture/model. Loads a real ONNX graph via
    // `tract` (pure Rust, no external DLL) - construction can genuinely
    // fail (bad/missing bundled bytes, a `tract` runtime error), so this
    // follows the exact same "construct once at startup, honest fallback,
    // never crash the app over an optional backend" pattern as Hush/TSE
    // above and below.
    let (mic_deep_filter_available, mic_deep_filter_unavailable_reason, mic_deep_filter_stage): (bool, Option<String>, Box<dyn Stage>) =
        match noise_engine::load_bundled_deep_filter() {
            Ok(engine) => (
                true,
                None,
                Box::new(dsp_core::WatchdogStage::new(
                    "mic DeepFilterNet",
                    Box::new(engine),
                    BVC_WATCHDOG_TIMEOUT,
                    |msg| crate::log_error!("[clearnairt] {msg}"),
                    || Box::new(audio_io::mmcss::elevate_current_thread()),
                )),
            ),
            Err(e) => (
                false,
                Some(format!("DeepFilterNet failed to load ({e:#}); this BVC option is unavailable.")),
                Box::new(NoOpStage("BVC DeepFilterNet (unavailable)")),
            ),
        };

    // --- Hush (bpo_v1): a real fine-tune of Hush's own upstream checkpoint
    // (pulp-vision/Hush, `deployment/models/model_best.ckpt`), NOT a
    // separate architecture - see `bvc_hush::BPO_V1_MODEL_BUNDLE_FILE_NAME`'s
    // docs. Loaded through `HushBvcStage::try_load_with_model`, reusing the
    // SAME already-extracted `weya_nc.dll` as the default mic Hush instance
    // above but pointing `weya_nc_model_load_from_path` at the different
    // bundle. Safe to share the DLL module here (unlike the mic/speaker
    // split, which needs two distinct on-disk DLL copies) because this
    // instance and the default mic Hush instance never run concurrently -
    // the BVC router dispatches to exactly one of them per frame, on the
    // one mic pipeline thread. Added alongside, never replacing, the
    // default `Hush` option above - both independently selectable.
    let mic_hush_bpo_v1_load = bvc_hush::HushBvcStage::try_load_with_model(
        bvc_assets_dir,
        bvc_hush::DLL_FILE_NAME,
        bvc_hush::BPO_V1_MODEL_BUNDLE_FILE_NAME,
    );
    let (mic_bpo_v1_available, mic_bpo_v1_unavailable_reason) = match &mic_hush_bpo_v1_load {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    };
    // Same watchdog + observable-status pattern as `mic_hush_stage` above,
    // for the same closed-source-DLL reason.
    let mut mic_bpo_v1_status: Option<dsp_core::WatchdogStatus> = None;
    let mic_bpo_v1_stage: Box<dyn Stage> = match mic_hush_bpo_v1_load {
        Ok(stage) => {
            let watchdog = dsp_core::WatchdogStage::new(
                "mic Hush (bpo_v1)",
                Box::new(stage),
                BVC_WATCHDOG_TIMEOUT,
                |msg| crate::log_error!("[clearnairt] {msg}"),
                || Box::new(audio_io::mmcss::elevate_current_thread()),
            );
            mic_bpo_v1_status = Some(watchdog.status());
            Box::new(watchdog)
        }
        Err(_) => Box::new(NoOpStage("BVC Hush bpo_v1 (unavailable)")),
    };

    // --- Speaker-path BVC: Off/Hush/DeepFilterNet only (see
    // `tse_engine::SpeakerBvcBackend`'s docs - no TSE/Vanta, there's no way
    // to enroll an arbitrary incoming caller's voice ahead of time). Each
    // backend is its OWN independent instance from the mic path's - Hush
    // via `SPEAKER_DLL_FILE_NAME` (a second on-disk DLL copy, already
    // self-extracted by `setup::ensure_bvc_assets_extracted` specifically
    // for this), DeepFilterNet via a second `load_bundled_deep_filter()`
    // call (a fresh parse, no shared global state) - so a failure or a
    // wedged call on one path never affects the other.
    let speaker_bvc_load = bvc_hush::HushBvcStage::try_load_named(bvc_assets_dir, bvc_hush::SPEAKER_DLL_FILE_NAME);
    let (speaker_bvc_available, speaker_bvc_unavailable_reason) = match &speaker_bvc_load {
        Ok(_) => (true, None),
        Err(e) => (false, Some(e.to_string())),
    };
    let speaker_hush_stage: Box<dyn Stage> = match speaker_bvc_load {
        Ok(stage) => Box::new(dsp_core::WatchdogStage::new(
            "speaker BVC (Hush)",
            Box::new(stage),
            BVC_WATCHDOG_TIMEOUT,
            |msg| crate::log_error!("[clearnairt] {msg}"),
            || Box::new(audio_io::mmcss::elevate_current_thread()),
        )),
        Err(_) => Box::new(NoOpStage("Speaker BVC Hush (unavailable)")),
    };

    let (speaker_deep_filter_available, speaker_deep_filter_unavailable_reason, speaker_deep_filter_stage): (
        bool,
        Option<String>,
        Box<dyn Stage>,
    ) = match noise_engine::load_bundled_deep_filter() {
        Ok(engine) => (
            true,
            None,
            Box::new(dsp_core::WatchdogStage::new(
                "speaker BVC (DeepFilterNet)",
                Box::new(engine),
                BVC_WATCHDOG_TIMEOUT,
                |msg| crate::log_error!("[clearnairt] {msg}"),
                || Box::new(audio_io::mmcss::elevate_current_thread()),
            )),
        ),
        Err(e) => (
            false,
            Some(format!("DeepFilterNet failed to load ({e:#}); this BVC option is unavailable.")),
            Box::new(NoOpStage("Speaker BVC DeepFilterNet (unavailable)")),
        ),
    };

    // --- Speaker-reference-enrollment subsystem + the BVC backend router
    // (Off/Hush/DeepFilterNet/TSE/Vanta) - see docs/TSE_ENROLLMENT.md. `target_speaker_store`
    // (256-dim, WeSpeaker ResNet34) drives the GUI's enrollment/verification
    // display; `tse_speaker_store` (192-dim, ECAPA_TDNN_GLOB_c512) and
    // `vanta_speaker_store` (192-dim, SpeechBrain ECAPA - a DIFFERENT
    // embedding space despite the same dimension) are independent stores
    // populated from the SAME recording (see
    // `enrollment::EnrollmentSession::with_secondary_encoder`), read only by
    // their respective real backends below - see `tse_bsrnn::tse_speaker_encoder`'s
    // and `bvc_vanta::ecapa`'s module docs for why these are separate
    // stores, not one shared/bridged embedding. `enrollment_tap` is the
    // realtime-thread half, moved into `spawn_mic_pipeline` further down.
    let (enrollment_tap, target_speaker_store, enrollment_session, enrollment_warning, tse_speaker_store, vanta_speaker_store) =
        crate::shared::build_enrollment_subsystem();

    // --- Vanta: real, trained (if honestly weak - epoch 6, val SI-SDR
    // +5.911dB, real measured eval only +1.07dB average improvement with a
    // -7.10dB worst case, PESQ 0.84) separator + encoder/decoder weights,
    // all now bundled and loading - see `bvc_vanta`'s crate docs for the
    // full status. Same "construct once at startup, honest fallback,
    // never crash the app over an optional backend" pattern as every other
    // BVC backend above. Still labeled experimental in the GUI - a real,
    // running model is not the same as a production-quality one.
    let (mic_vanta_available, mic_vanta_unavailable_reason, mic_vanta_stage): (bool, Option<String>, Box<dyn Stage>) =
        match bvc_vanta::VantaBackend::try_load() {
            Ok(real) => {
                let adapter = tse_engine::TseStageAdapter::new(real, vanta_speaker_store.clone(), Arc::new(LatencyLog::new(256)));
                let stage = Box::new(dsp_core::WatchdogStage::new(
                    "mic Vanta",
                    Box::new(adapter),
                    TSE_REAL_BACKEND_TEMPORARY_TIMEOUT,
                    |msg| crate::log_error!("[clearnairt] {msg}"),
                    || Box::new(audio_io::mmcss::elevate_current_thread()),
                ));
                (
                    true,
                    Some(
                        "EXPERIMENTAL: real but weak model (measured +1.07dB average improvement, \
                         -7.10dB worst case on held-out eval) - may sometimes make audio worse, not better."
                            .to_string(),
                    ),
                    stage,
                )
            }
            Err(e) => {
                let adapter = tse_engine::TseStageAdapter::new(
                    tse_engine::UnavailableTse,
                    vanta_speaker_store.clone(),
                    Arc::new(LatencyLog::new(256)),
                );
                let stage = Box::new(dsp_core::WatchdogStage::new(
                    "mic Vanta",
                    Box::new(adapter),
                    TSE_WATCHDOG_TIMEOUT,
                    |msg| crate::log_error!("[clearnairt] {msg}"),
                    || Box::new(audio_io::mmcss::elevate_current_thread()),
                ));
                (false, Some(format!("Vanta backend failed to load ({e}); this BVC option is unavailable.")), stage)
            }
        };
    if let Some(w) = enrollment_warning {
        warnings.push(w);
    }
    // Real backend attempt, mirroring exactly how `bvc_hush::HushBvcStage::try_load`
    // gates Hush availability above: construct the real thing once, at
    // startup, falling back to the honest `UnavailableTse` stub (with a
    // specific reason surfaced to the GUI) if it doesn't load - never a
    // silent, partially-working state.
    //
    // KNOWN, NOT-YET-FIXED GAPS in this real backend (see
    // `tse_bsrnn::backend::RealBsrnnTseBackend`'s own docs for detail):
    // measured at roughly RTF~21-31x (i.e. NOT real-time - a 10ms frame can
    // take 200ms+ to process), and it allocates on the "realtime" frame
    // path, a real violation of `TseBackend`'s own no-allocation contract.
    // `TSE_WATCHDOG_TIMEOUT` below bounds the damage (a slow/failing call
    // passes the frame through unmodified) but does not make this fast.
    // Wired in now so the real model can be exercised/heard end-to-end;
    // performance work remains before this meets the realtime deadline.
    let (tse_backend_available, tse_unavailable_reason, tse_stage): (bool, Option<String>, Box<dyn Stage>) =
        match tse_bsrnn::backend::RealBsrnnTseBackend::try_load() {
            Ok(real) => {
                let adapter = tse_engine::TseStageAdapter::new(real, tse_speaker_store.clone(), Arc::new(LatencyLog::new(256)));
                // See TSE_REAL_BACKEND_TEMPORARY_TIMEOUT's docs: NOT the
                // real 20ms deadline yet, deliberately, until this backend
                // is optimized to actually meet it.
                let stage = Box::new(dsp_core::WatchdogStage::new(
                    "mic TSE",
                    Box::new(adapter),
                    TSE_REAL_BACKEND_TEMPORARY_TIMEOUT,
                    |msg| crate::log_error!("[clearnairt] {msg}"),
                    || Box::new(audio_io::mmcss::elevate_current_thread()),
                ));
                (true, None, stage)
            }
            Err(e) => {
                let adapter = tse_engine::TseStageAdapter::new(
                    tse_engine::UnavailableTse,
                    tse_speaker_store.clone(),
                    Arc::new(LatencyLog::new(256)),
                );
                let stage = Box::new(dsp_core::WatchdogStage::new(
                    "mic TSE",
                    Box::new(adapter),
                    TSE_WATCHDOG_TIMEOUT,
                    |msg| crate::log_error!("[clearnairt] {msg}"),
                    || Box::new(audio_io::mmcss::elevate_current_thread()),
                ));
                (false, Some(format!("Real TSE backend failed to load ({e}); TSE passes audio through unmodified.")), stage)
            }
        };
    let tse_available = tse_backend_available;
    // Owned by `handles.bvc_backend` (set below) and cloned into the
    // router - both sides must observe the same atomic for a live GUI
    // backend switch to actually take effect, mirroring exactly how
    // `mic_studio_preset` is shared between `EngineHandles` and the mic
    // pipeline thread's live preset-swap logic further down this file.
    let bvc_backend_selector = dsp_core::SharedPreset::new(settings.mic_bvc_backend);
    let mic_bvc_stage: Box<dyn Stage> = Box::new(tse_engine::BvcRouterStage::new(
        bvc_backend_selector.clone(),
        mic_hush_stage,
        mic_deep_filter_stage,
        tse_stage,
        mic_vanta_stage,
        mic_bpo_v1_stage,
    ));
    // --- Speaker-outbound (Pipeline 2) BVC router: Off/Hush/DeepFilterNet,
    // built from the two independent instances constructed above. ---
    let speaker_bvc_backend_selector = dsp_core::SharedPreset::new(settings.speaker_bvc_backend);
    let speaker_bvc_stage: Box<dyn Stage> = Box::new(tse_engine::SpeakerBvcRouterStage::new(
        speaker_bvc_backend_selector.clone(),
        speaker_hush_stage,
        speaker_deep_filter_stage,
    ));

    // --- Noise: RNNoise for both paths, each its own independent instance
    // (no shared state between the mic and speaker paths' stages). ---
    let mic_noise_stage: Box<dyn Stage> =
        noise_engine::build(noise_engine::NoiseEngineKind::RnNoise, None)
            .context("constructing mic-path RNNoise stage")?;
    let speaker_noise_stage: Box<dyn Stage> =
        noise_engine::build(noise_engine::NoiseEngineKind::RnNoise, None)
            .context("constructing speaker-path RNNoise stage")?;

    // --- Voice Gate: cheap, real-time volume-based background-voice
    // suppression - see `dsp_core::VoiceGateStage`'s docs. Must run FIRST
    // in the pipeline (before `AutoGainStage`, which always runs
    // unconditionally at the very top of `spawn_mic_pipeline`'s loop) - AGC
    // boosts any quiet frame, including quiet background chatter, up
    // toward its target level, which would erase the loudness contrast
    // this stage needs. ---
    let mic_voice_gate_stage: Box<dyn Stage> = Box::new(dsp_core::VoiceGateStage::new());

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
    let speaker_studio_stage: Box<dyn Stage> = Box::new(studio_dsp::StudioStage::new(
        crate::shared::studio_preset_from_index(settings.speaker_studio_preset),
    ));

    let mut handles = EngineHandles::new_from_settings(
        settings,
        mic_bvc_available,
        mic_bvc_unavailable_reason,
        speaker_bvc_available,
        speaker_bvc_unavailable_reason,
        speaker_deep_filter_available,
        speaker_deep_filter_unavailable_reason,
        Vec::new(),
        tse_available,
        tse_unavailable_reason,
        mic_deep_filter_available,
        mic_deep_filter_unavailable_reason,
        mic_vanta_available,
        mic_vanta_unavailable_reason,
        mic_bpo_v1_available,
        mic_bpo_v1_unavailable_reason,
        target_speaker_store,
        tse_speaker_store.clone(),
        vanta_speaker_store.clone(),
        enrollment_session,
    );
    // See `bvc_backend_selector`'s docs above - overwrite the freshly
    // constructed (otherwise-unread) `SharedPreset` `new_from_settings`
    // builds internally with the one actually shared with the router, the
    // same "construct inert, then wire the real thing" pattern already used
    // for `device_switcher` below.
    handles.bvc_backend = bvc_backend_selector;
    handles.speaker_bvc_backend = speaker_bvc_backend_selector;
    // See `mic_hush_status`/`mic_bpo_v1_status`'s docs - wire in the real
    // watchdog status handles captured when each `HushBvcStage` was
    // actually constructed above, same "construct inert, then wire the
    // real thing" pattern.
    handles.mic_hush_status = mic_hush_status;
    handles.mic_bpo_v1_status = mic_bpo_v1_status;

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
        mic_voice_gate_stage,
        mic_noise_stage,
        mic_bvc_stage,
        mic_studio_stage,
        driver_writer,
        enrollment_tap,
    )?;

    // --- Pipeline 2 (speaker outbound): capture from a second, independent
    // virtual-cable instance's capture endpoint (what a call app's output
    // gets redirected into) -> Noise/BVC/Studio -> real physical speakers/
    // headphones. Independent thread, independent failure mode from
    // Pipeline 1 - a speaker-pipeline construction failure must not take
    // down the mic path (see `spawn_speaker_pipeline`'s docs). Unlike the
    // two virtual-cable ids, `real_speaker_output_id` treats `None` as
    // "system default output device", not "disabled" - there's always a
    // sensible real device to render into.
    let (speaker_capture, speaker_in_p) = match virtual_speaker_source_id {
        Some(id) => {
            let (p, c) = make_ring_buffer(2.0);
            match MicCapture::start_on_device_id(id, p, handles.speaker_counters.clone()) {
                Ok(capture) => (Some(capture), Some(c)),
                Err(e) => {
                    warnings.push(format!(
                        "Failed to start capture from the selected speaker-source device ({e:#}); \
                         incoming-call audio cleanup is unavailable until a working device is selected."
                    ));
                    (None, None)
                }
            }
        }
        None => {
            warnings.push(
                "No speaker-source device selected - incoming-call audio is not being captured for \
                 cleanup. Install a second, independent virtual audio cable (e.g. VB-CABLE A+B) and \
                 select its capture endpoint in the GUI (see docs/VIRTUAL_DEVICES.md)."
                    .to_string(),
            );
            (None, None)
        }
    };
    let (speaker_render, speaker_out_c) = {
        let (p, c) = make_ring_buffer(2.0);
        let started = match real_speaker_output_id {
            Some(id) => HardwareRender::start_on_device_id(id, c, handles.speaker_counters.clone()),
            None => HardwareRender::start_default(c, handles.speaker_counters.clone()),
        };
        match started {
            Ok(render) => (Some(render), Some(p)),
            Err(e) => {
                warnings.push(format!(
                    "Failed to start the real speaker/headphone output device ({e:#}); incoming-call \
                     audio cleanup has nowhere to render to."
                ));
                (None, None)
            }
        }
    };

    let speaker_in_swap = swap_consumer(speaker_in_p);
    let speaker_out_swap = swap_producer(speaker_out_c);

    spawn_speaker_pipeline(
        speaker_in_swap.clone(),
        speaker_out_swap.clone(),
        handles.clone(),
        speaker_noise_stage,
        speaker_bvc_stage,
        speaker_studio_stage,
    )?;

    handles.device_switcher = Arc::new(LiveAudioSwitcher {
        physical_mic: Mutex::new(Some(mic_capture)),
        virtual_mic_render: Mutex::new(virtual_mic_render),
        monitor_render: Mutex::new(monitor_render),
        speaker_capture: Mutex::new(speaker_capture),
        speaker_render: Mutex::new(speaker_render),
        mic_in_swap,
        mic_out_swap,
        monitor_out_swap,
        speaker_in_swap,
        speaker_out_swap,
        counters: handles.counters.clone(),
        speaker_counters: handles.speaker_counters.clone(),
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
    mut voice_gate_stage: Box<dyn Stage>,
    mut noise_stage: Box<dyn Stage>,
    mut bvc_stage: Box<dyn Stage>,
    mut studio_stage: Box<dyn Stage>,
    driver_writer: Option<driver_transport::SharedRingWriter>,
    mut enrollment_tap: enrollment::EnrollmentTap,
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
            // Enrollment's own reference encoders expect audio built from
            // the pre-Noise/BVC/Studio signal, auto-gained the same way the
            // main pipeline always auto-gains it - see `enrollment`'s crate
            // docs. Once BVC could run BEFORE AutoGain (see the BVC-on
            // branch below), the main `frame` reaching
            // `enrollment_tap.push_frame_if_active` would sometimes already
            // be colored by BVC, which must never happen. A second,
            // independent `AutoGainStage` instance (its own gain-tracking
            // state, never shared with the main pipeline's `auto_gain`)
            // plus a preallocated scratch buffer let enrollment always see
            // a pre-BVC, auto-gained copy, regardless of which branch the
            // main frame takes - with no per-frame allocation.
            let mut enroll_gain = dsp_core::AutoGainStage::new();
            let mut enroll_frame = vec![0.0f32; FRAME_SAMPLES];
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

                // Must run BEFORE auto_gain - see `dsp_core::VoiceGateStage`'s
                // docs and `mic_voice_gate_stage`'s construction comment above.
                run_if_enabled(voice_gate_stage.as_mut(), &handles.mic_voice_gate, &mut frame);

                // Enrollment tap: ALWAYS reads a pre-Noise/BVC/Studio,
                // auto-gained copy of the signal - per `enrollment`'s own
                // docs, a real "clean voice reference" must never be built
                // from anything BVC/Studio touched, regardless of which
                // branch below the main `frame` takes. Taken here, before
                // the BVC-on/off split, so this is unconditionally correct
                // even when BVC runs first in the branch below.
                enroll_frame.copy_from_slice(&frame);
                enroll_gain.process(&mut enroll_frame);
                enrollment_tap.push_frame_if_active(&enroll_frame);

                // Real finding: when BVC (Hush/DeepFilterNet/etc, whichever
                // the router has selected) is on, running AutoGain and
                // RNNoise ahead of it was actively counterproductive -
                // AutoGain boosted quiet background voices (e.g. a
                // neighbour talking during the user's own silent pauses)
                // up toward the target level BEFORE BVC ever saw them,
                // undoing exactly the contrast BVC needs to tell "target
                // speaker" from "background" apart; and RNNoise had
                // already reshaped the signal (its own denoising) before
                // handing it to a backend that does its own denoising
                // internally, double-processing audio that was never
                // trained for that. So: when BVC is on, it runs FIRST,
                // directly on the raw post-voice-gate mic signal, RNNoise
                // is skipped entirely (BVC already suppresses noise), and
                // AutoGain runs AFTER BVC instead of before it. When BVC is
                // off, the original order (AutoGain -> RNNoise -> BVC
                // no-op) is unchanged.
                if handles.mic_bvc.is_on() {
                    bvc_stage.process(&mut frame);
                    auto_gain.process(&mut frame);
                } else {
                    auto_gain.process(&mut frame);
                    run_if_enabled(noise_stage.as_mut(), &handles.mic_noise, &mut frame);
                    // `handles.mic_bvc` is already known false here, so this
                    // is a no-op - kept as `run_if_enabled` (rather than
                    // skipped outright) only so a live toggle flip between
                    // this check and the next frame can't leave BVC
                    // permanently unrun; see `run_if_enabled`'s docs.
                    run_if_enabled(bvc_stage.as_mut(), &handles.mic_bvc, &mut frame);
                }
                run_if_enabled(studio_stage.as_mut(), &handles.mic_studio, &mut frame);

                let peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
                handles.level.store(peak);

                let elapsed_ns = started.elapsed().as_nanos() as u64;
                handles.latency_log.record_ns(elapsed_ns);
                handles.counters.frames_processed.fetch_add(1, Ordering::Relaxed);
                if elapsed_ns > DEADLINE_NS {
                    handles.counters.deadline_misses.fetch_add(1, Ordering::Relaxed);
                }

                // Master mute: unlike the individual Noise/BVC/Studio
                // toggles above (which only skip their OWN stage, leaving
                // the frame passing through untouched when off), this
                // genuinely silences the frame before it goes anywhere -
                // real gap this closes: turning every individual stage off
                // still forwarded raw, unprocessed mic audio, which a user
                // hit live ("everything is off, I can still hear audio").
                // Applied after metering (so the level meter still reflects
                // real mic activity while muted) but before every fan-out
                // below.
                if !handles.mic_master_enabled.is_on() {
                    frame.copy_from_slice(&silence);
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

/// Pipeline 2: speaker-source capture (a second, independent virtual-cable
/// instance's capture endpoint) -> Noise -> BVC (Off/Hush/DeepFilterNet
/// only, see `tse_engine::SpeakerBvcRouterStage`) -> Studio -> real
/// physical speaker/headphone render. Runs on its own dedicated thread for
/// the life of the process, entirely independent of `spawn_mic_pipeline` -
/// a construction failure or a slow/wedged stage here never blocks or takes
/// down the mic path (each BVC stage is already individually wrapped in its
/// own `WatchdogStage` before reaching this function, same as the mic
/// path).
///
/// Deliberately simpler than `spawn_mic_pipeline`: no Voice Gate (that
/// stage's asymmetric-attack/release "who's the loud one" logic was built
/// for suppressing a quieter background voice behind the MIC's speaker -
/// same reasoning `docs/TSE_ENROLLMENT.md` gives for excluding TSE/Vanta
/// here also applies: no natural "who is louder" contrast to key off of for
/// an arbitrary incoming call), no AutoGain (incoming call audio is
/// whatever level the remote party/call app already normalized it to - not
/// this project's mic-input-gain problem to fix), no driver-transport
/// write, no enrollment tap, and no second "monitor" fan-out (exactly one
/// render target, the real speaker output).
fn spawn_speaker_pipeline(
    speaker_in: SwapConsumer,
    speaker_out: SwapProducer,
    handles: EngineHandles,
    mut noise_stage: Box<dyn Stage>,
    mut bvc_stage: Box<dyn Stage>,
    mut studio_stage: Box<dyn Stage>,
) -> Result<()> {
    thread::Builder::new()
        .name("clearnai-speaker-pipeline".into())
        .spawn(move || {
            let _mmcss_guard = audio_io::mmcss::elevate_current_thread();
            let mut frame = vec![0.0f32; FRAME_SAMPLES];
            let silence = vec![0.0f32; FRAME_SAMPLES];
            let mut last_studio_preset = handles.speaker_studio_preset.load();
            loop {
                pull_frame_blocking(&speaker_in, &mut frame, &handles.speaker_counters);
                let started = Instant::now();

                let current_preset = handles.speaker_studio_preset.load();
                if current_preset != last_studio_preset {
                    studio_stage = Box::new(studio_dsp::StudioStage::new(
                        crate::shared::studio_preset_from_index(current_preset),
                    ));
                    last_studio_preset = current_preset;
                }

                run_if_enabled(noise_stage.as_mut(), &handles.speaker_noise, &mut frame);
                run_if_enabled(bvc_stage.as_mut(), &handles.speaker_bvc, &mut frame);
                run_if_enabled(studio_stage.as_mut(), &handles.speaker_studio, &mut frame);

                let elapsed_ns = started.elapsed().as_nanos() as u64;
                handles.speaker_latency_log.record_ns(elapsed_ns);
                handles.speaker_counters.frames_processed.fetch_add(1, Ordering::Relaxed);
                if elapsed_ns > DEADLINE_NS {
                    handles.speaker_counters.deadline_misses.fetch_add(1, Ordering::Relaxed);
                }

                // Master mute - see `spawn_mic_pipeline`'s matching comment
                // for why this exists as a genuine silence, separate from
                // the individual Noise/BVC/Studio toggles above.
                if handles.speaker_master_enabled.is_on() {
                    push_frame(&speaker_out, &frame);
                } else {
                    push_frame(&speaker_out, &silence);
                }
            }
        })
        .context("spawning speaker pipeline thread")?;
    Ok(())
}

use crate::shared::PlaybackHandle;

/// Plays `samples` (raw, at `dsp_core::SAMPLE_RATE_HZ`) through
/// `device_id` (or the system default if `None` - e.g. whatever the
/// user's Bluetooth headset happens to be, if currently the Windows
/// default; real, observed behavior, not hypothetical, which is exactly
/// why `device_id` exists as an explicit override rather than always
/// assuming default-is-fine) - backs the GUI's "Play Recording" button
/// and its Pause/Resume/Stop controls, letting the user hear exactly what
/// an enrollment attempt actually captured (useful for telling "too
/// quiet"/"wrong device" apart from a genuinely good recording). Spawns
/// its own background thread and returns a `PlaybackHandle` immediately;
/// failures opening the device are logged (via the same handle's
/// `Finished` status - the GUI just won't see progress advance), matching
/// this project's existing "rare, user-initiated action" idiom
/// (`spawn_device_switch`).
///
/// Feeds the ring buffer in small chunks (not the whole clip up front) so
/// `pause` can take effect quickly (stop feeding, existing buffered audio
/// finishes, then real silence - not a jump), and so `stop`/pause can be
/// checked between chunks without needing to interrupt a blocking write.
pub fn play_recorded_audio(samples: Vec<f32>, device_id: Option<String>) -> PlaybackHandle {
    let total_samples = samples.len();
    let stop_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let paused_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let finished_flag = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let played_samples = Arc::new(std::sync::atomic::AtomicUsize::new(0));

    let handle = PlaybackHandle {
        stop_flag: stop_flag.clone(),
        paused_flag: paused_flag.clone(),
        finished_flag: finished_flag.clone(),
        played_samples: played_samples.clone(),
        total_samples,
    };

    if samples.is_empty() {
        finished_flag.store(true, Ordering::Relaxed);
        return handle;
    }

    thread::spawn(move || {
        // A few seconds of headroom is enough now that feeding is chunked
        // (not "push the whole 12s clip up front") - the ring only ever
        // needs to hold a little more than the render thread drains per
        // wake-up, not the entire clip at once.
        let (mut producer, consumer) = make_ring_buffer(2.0);
        let counters = Arc::new(dsp_core::FrameCounters::default());
        let render = match &device_id {
            Some(id) => HardwareRender::start_on_device_id(id, consumer, counters),
            None => HardwareRender::start_default(consumer, counters),
        };
        let render = match render {
            Ok(r) => r,
            Err(e) => {
                crate::log_error!(
                    "[clearnairt] failed to open the selected playback device ({device_id:?}) for recording playback: {e:#}"
                );
                finished_flag.store(true, Ordering::Relaxed);
                return;
            }
        };

        // REAL BUG, FOUND AND FIXED HERE: an earlier version of this loop
        // fed samples into the ring as fast as the ring's backpressure
        // allowed (up to its full 2s capacity), then - once `pos` reached
        // `samples.len()` - waited only a fixed 300ms before calling
        // `render.stop()`. For anything longer than ~2.3s, that meant the
        // render thread could still have up to ~2 real seconds of
        // genuinely unplayed audio sitting in the ring when it was torn
        // down - the TAIL of the recording (often literally the last few
        // words spoken) was silently cut off. Observed and reported as
        // "missing the last few words" during real testing - a genuine
        // regression, not a hypothetical one.
        //
        // Fix: pace the feed loop itself to real time (never push more
        // than the clip's own elapsed real-world duration justifies), so
        // by the time this loop naturally finishes, real time has already
        // elapsed for the WHOLE clip and only a small, bounded amount
        // (the ring's own tiny remaining backlog + the WASAPI device
        // buffer) is left - which the short fixed sleep after the loop
        // can actually cover. `paused_accum` subtracts out any time spent
        // paused, so pausing doesn't make this pacing think more real
        // time has passed than actually has.
        const FEED_CHUNK: usize = 480; // one FRAME_SAMPLES' worth per iteration
        let mut pos = 0usize;
        let playback_clock_start = std::time::Instant::now();
        let mut paused_accum = std::time::Duration::ZERO;
        let mut pause_started: Option<std::time::Instant> = None;

        while pos < samples.len() {
            if stop_flag.load(Ordering::Relaxed) {
                break;
            }
            if paused_flag.load(Ordering::Relaxed) {
                if pause_started.is_none() {
                    pause_started = Some(std::time::Instant::now());
                }
                thread::sleep(std::time::Duration::from_millis(20));
                continue;
            }
            if let Some(p) = pause_started.take() {
                paused_accum += p.elapsed();
            }

            let target_elapsed = std::time::Duration::from_secs_f32(pos as f32 / dsp_core::SAMPLE_RATE_HZ as f32);
            let real_elapsed = playback_clock_start.elapsed().saturating_sub(paused_accum);
            if real_elapsed < target_elapsed {
                // Running ahead of real playback time - wait rather than
                // keep stuffing the ring, so this loop's own completion
                // tracks real elapsed playback time, not "how fast could
                // this thread possibly push samples."
                thread::sleep((target_elapsed - real_elapsed).min(std::time::Duration::from_millis(20)));
                continue;
            }

            let end = (pos + FEED_CHUNK).min(samples.len());
            for &sample in &samples[pos..end] {
                // A push failure means the ring is momentarily full (the
                // render thread hasn't drained yet) - back off briefly
                // and retry this same sample rather than dropping it.
                while producer.push(sample).is_err() {
                    if stop_flag.load(Ordering::Relaxed) {
                        break;
                    }
                    thread::sleep(std::time::Duration::from_millis(2));
                }
            }
            pos = end;
            played_samples.store(pos, Ordering::Relaxed);
        }

        // Only a small, bounded backlog should remain now (see the real-
        // time pacing above) - this covers it plus the WASAPI device
        // buffer, not a multi-second tail.
        thread::sleep(std::time::Duration::from_millis(300));
        render.stop();
        finished_flag.store(true, Ordering::Relaxed);
    });

    handle
}
