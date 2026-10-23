//! The single native window: mic-path and speaker-path stage togglers (each
//! path independently gating RNNoise/BVC/Studio, plus a live Studio preset
//! picker per path), a live realtime-status/
//! level indicator, and two device pickers for the virtual-cable roles
//! documented in `docs/VIRTUAL_DEVICES.md`.
//!
//! Built against `iced` 0.14's `iced::application(boot, update, view)`
//! builder API (confirmed current on docs.rs/crates.io on 2026-09-11 - see
//! the top-level report for the exact pages read; the older `Application`/
//! `Sandbox` trait API from pre-0.13 iced does not exist in this version).
//!
//! This module is portable: it only depends on `dsp_core` (for
//! `RealtimeStatus`/percentiles) and `crate::shared`/`crate::settings`
//! (also portable), never on `audio_io` or `engine` directly. `main.rs`
//! constructs the real (Windows) or inert (elsewhere) engine handles and
//! hands them in, so this file - and therefore the whole GUI layer - type
//! checks and its logic can in principle run on any platform `iced`
//! supports, even though the audio behind it is Windows-only.

use crate::setup::CompatibilityReport;
use crate::settings::Settings;
use crate::shared::{
    studio_preset_from_index, studio_preset_name, studio_preset_to_index, DeviceList, DeviceOption,
    EngineHandles, STUDIO_PRESET_NAMES,
};
use crate::status_ui::{status_color, status_label};
use dsp_core::{classify_status, compute_percentiles, RealtimeStatus};
use enrollment::EnrollmentState;
use tse_engine::BvcBackend;
use iced::widget::{button, column, container, pick_list, progress_bar, row, scrollable, text, toggler};
use iced::{Color, Element, Length, Subscription};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

/// Frame deadline in nanoseconds (10ms @ 48kHz), matching `dsp_core`'s frame
/// config - duplicated here (rather than importing a private constant) since
/// `dsp_core` only exposes `FRAME_SAMPLES`/`SAMPLE_RATE_HZ`, not a
/// precomputed deadline.
const DEADLINE_NS: u64 = (dsp_core::FRAME_SAMPLES as u64 * 1_000_000_000) / dsp_core::SAMPLE_RATE_HZ as u64;

/// Everything `gui::run` needs to boot the window, gathered in `main.rs`.
/// The audio engine (Windows) / inert stub (elsewhere) is always started up
/// front exactly as before `setup.rs` existed - a first-run setup screen
/// bolted in front of it never changes `engine.rs`'s bootstrap timing, it
/// only changes which screen is shown first.
pub struct Initial {
    pub settings: Settings,
    pub handles: EngineHandles,
    pub devices: DeviceList,
    /// Extra non-fatal warnings gathered by `main.rs` itself (e.g. "not
    /// running on Windows"), shown alongside any the engine collected.
    pub extra_warnings: Vec<String>,
    /// Kept alive only so the real engine's OS threads (mic capture,
    /// render, loopback tap) live exactly as long as this window does - the
    /// GUI never inspects it. `None` on a platform with no real engine.
    pub keepalive: Option<Box<dyn std::any::Any + Send>>,
    /// Fresh (fast, synchronous) system compatibility check, run in
    /// `main.rs` before the window is created.
    pub compat: CompatibilityReport,
    /// `Some(message)` if `main.rs::bootstrap` failed to self-extract the
    /// embedded BVC DLL/model into `%LOCALAPPDATA%` (disk full, permission
    /// denied, etc.) - `None` means extraction succeeded (or was already
    /// up to date), i.e. the ordinary, expected case that needs no mention
    /// on the setup screen at all.
    pub bvc_extraction_error: Option<String>,
}

struct MainState {
    settings: Settings,
    handles: EngineHandles,
    devices: DeviceList,
    warnings: Vec<String>,
    status: RealtimeStatus,
    level: f32,
    _keepalive: Option<Box<dyn std::any::Any + Send>>,
    device_switch_seqs: Arc<DeviceSwitchSequences>,
    /// Live drop/underrun telemetry, refreshed every `Message::Tick` from
    /// `handles.counters` (plain atomic loads - see
    /// `dsp_core::FrameCounters`). Snapshotted into plain `u64`s here rather
    /// than read directly in `view_main` so the view function stays a pure
    /// function of `MainState`, matching this module's existing pattern for
    /// `status`/`level`.
    io_counters_snapshot: IoCountersSnapshot,
    /// Same idea as `status`/`io_counters_snapshot` above, but for the
    /// speaker-outbound pipeline - its own, independent telemetry (see
    /// `EngineHandles::speaker_counters`/`speaker_latency_log`), so the two
    /// directions can be told apart instead of one merged status.
    speaker_status: RealtimeStatus,
    speaker_io_counters_snapshot: IoCountersSnapshot,
    /// How many *consecutive* ticks `status` has read `Overloaded` - reset
    /// to 0 the instant it isn't. A single blip (one slow frame under a
    /// momentary system hiccup) is normal and not worth interrupting the
    /// user about; `LOW_CPU_SUGGESTION_THRESHOLD_TICKS` consecutive ticks
    /// (a few real seconds, at the ~250ms tick rate `Subscription`/`main.rs`
    /// drives) means the overload is sustained, not transient.
    consecutive_overloaded_ticks: u32,
    /// Once true, stays true for the rest of the session (until the user
    /// actually clicks Low CPU mode) rather than flickering on/off as
    /// `status` bounces around Overloaded's boundary - a suggestion that
    /// appears and disappears every couple of seconds would be more
    /// annoying than useful.
    show_low_cpu_suggestion: bool,
    /// Point-in-time copy of `handles.enrollment.state()`, refreshed every
    /// `Message::Tick` - same "snapshot into plain state, view stays a pure
    /// function of `MainState`" pattern as `status`/`level`/
    /// `io_counters_snapshot` above.
    enrollment_state: enrollment::EnrollmentState,
    /// The in-progress (or just-finished) "Play Recording" playback, if
    /// any - `None` before the first click, or once the user has moved
    /// past it (a new recording, clearing the profile). Read directly in
    /// `view_main` via its own cheap atomic-load methods
    /// (`PlaybackHandle::status`/`progress`), not snapshotted into a
    /// separate field the way `enrollment_state` is - the same "read the
    /// cheap thing directly" pattern `target_speaker_store.has_profile()`
    /// already uses elsewhere in this same view.
    playback: Option<crate::shared::PlaybackHandle>,
}

/// See `MainState::consecutive_overloaded_ticks`'s docs. At the ~250ms tick
/// rate this is ~3 seconds of sustained overload before suggesting Low CPU
/// mode - long enough to filter out a single-frame hiccup, short enough to
/// still catch a real problem before the user gives up and starts digging
/// through logs themselves.
const LOW_CPU_SUGGESTION_THRESHOLD_TICKS: u32 = 12;

/// A point-in-time copy of `dsp_core::FrameCounters`'s fields relevant to
/// live diagnostics. Not every field in `FrameCounters` is shown - only the
/// ones a user/operator would actually act on (drops, over/underruns);
/// `frames_processed`/`deadline_misses` already drive the "Engine: ..."
/// status label above these via `classify_status`, so repeating them here
/// as raw counts would be redundant, not additional information.
#[derive(Debug, Clone, Copy, Default)]
struct IoCountersSnapshot {
    dropped_frames: u64,
    input_overruns: u64,
    output_underruns: u64,
}

impl IoCountersSnapshot {
    fn read(counters: &dsp_core::FrameCounters) -> Self {
        use std::sync::atomic::Ordering;
        Self {
            dropped_frames: counters.dropped_frames.load(Ordering::Relaxed),
            input_overruns: counters.input_overruns.load(Ordering::Relaxed),
            output_underruns: counters.output_underruns.load(Ordering::Relaxed),
        }
    }
}

/// One monotonic counter per device-picker role, used to make rapid/
/// double-clicked picks resolve to "whichever one the user picked *last*"
/// instead of "whichever background switch happened to finish last" (see
/// `spawn_device_switch`). `LiveAudioSwitcher::set_*` opens a device and
/// blocks joining the old worker thread, both of unpredictable duration, so
/// two in-flight switches for the same role can otherwise complete in
/// either order.
#[derive(Default)]
struct DeviceSwitchSequences {
    mic_target: Arc<AtomicU64>,
    physical_mic: Arc<AtomicU64>,
    monitor: Arc<AtomicU64>,
    speaker_source: Arc<AtomicU64>,
    real_speaker_output: Arc<AtomicU64>,
}

struct SetupState {
    settings: Settings,
    compat: CompatibilityReport,
    /// `Some(message)` if embedded-BVC-asset extraction failed; see
    /// `Initial::bvc_extraction_error`.
    bvc_extraction_error: Option<String>,
    // Kept only to hand off to `MainState` once the user continues past
    // setup - never read directly by the setup screen itself.
    handles: EngineHandles,
    devices: DeviceList,
    warnings: Vec<String>,
    keepalive: Option<Box<dyn std::any::Any + Send>>,
}

enum Screen {
    Setup(SetupState),
    Main(MainState),
}

#[derive(Debug, Clone)]
pub enum Message {
    Tick,
    ToggleMicNoise(bool),
    /// "Voice Gate (suppress quiet background voices)" toggled - see
    /// `dsp_core::VoiceGateStage`'s docs.
    ToggleMicVoiceGate(bool),
    ToggleMicStudio(bool),
    SelectMicStudioPreset(&'static str),
    SelectMicTarget(DeviceOption),
    SelectPhysicalMic(DeviceOption),
    ToggleMonitor(bool),
    SelectMonitorDevice(DeviceOption),
    /// Re-enumerates devices (e.g. after starting VoiceMeter/VB-Cable
    /// *after* ClearNAI) without restarting the app. See
    /// `LiveDeviceSwitcher::enumerate_devices`.
    RefreshDevices,
    /// Turns BVC and Studio off (RNNoise is left exactly as it
    /// was) - a one-shot action, not a persistent locked "mode": the user
    /// can turn either back on afterward via its own toggle. See
    /// `status_row`'s doc comment for why this exists (BVC's ONNX inference
    /// is the single most CPU-expensive stage; on weak hardware it's the
    /// first thing worth shedding).
    EnableLowCpuMode,
    /// Setup screen: "Continue" / "Continue anyway" clicked.
    ContinueFromSetup,
    /// Picks the BVC backend - Off/Hush/TSE (see `tse_engine::BvcBackend`).
    /// Selecting TSE is always allowed in the picker (its option is present,
    /// not hidden), but is shown as unavailable when `handles.tse_available`
    /// is `false` - matching how BVC itself already handles an unavailable
    /// backend (disabled toggle + reason text, never a silently-missing
    /// option).
    SelectBvcBackend(&'static str),
    /// "Record Voice Reference" clicked.
    StartEnrollment,
    /// "Stop recording" clicked (manual stop, alongside the automatic
    /// silence-based and max-duration stops `enrollment::EnrollmentSession`
    /// already handles on its own).
    StopEnrollment,
    /// "Re-record Reference" clicked - identical to `StartEnrollment`
    /// (starting a new recording naturally supersedes/replaces whatever was
    /// enrolled before once it succeeds), kept as its own message only so
    /// the GUI can offer a differently-labeled button once already enrolled.
    RestartEnrollment,
    /// "Clear Reference" clicked.
    ClearEnrollment,
    /// "Play Recording" clicked - plays back the most recent recording
    /// attempt's raw audio through the default playback device, so the
    /// user can hear exactly what was captured (see
    /// `enrollment::EnrollmentSession::last_recording`'s docs). Available
    /// even if that attempt was rejected/failed, not just after a
    /// successful enrollment - hearing a bad recording is exactly what
    /// helps diagnose why it was bad.
    PlayLastRecording,
    /// "Play Denoised" clicked - plays back the same recording run through
    /// the same RNNoise pass real enrollment applies before extracting a
    /// voice embedding (see
    /// `enrollment::EnrollmentSession::last_recording_denoised`'s docs),
    /// so the user can directly A/B it against the raw "Play Recording"
    /// version rather than trusting the denoising step blindly.
    PlayDenoisedRecording,
    /// "Test with TSE" clicked - runs the most recent recording through a
    /// real TSE backend OFFLINE (see `shared::run_tse_offline`'s docs on why
    /// this is the only way to hear full-quality, non-glitchy TSE output
    /// today, since the live real-time pipeline's `WatchdogStage` only
    /// bounds the *damage* of TSE currently running far slower than real
    /// time, not the slowness itself). Runs on a background thread
    /// (`shared::TseOfflineTest`) since this can legitimately take tens of
    /// seconds; `Message::Tick` picks up the result once ready.
    TestWithTse,
    /// "Pause"/"Resume" clicked during an active "Play Recording" playback.
    PausePlayback,
    ResumePlayback,
    /// "Stop" clicked during an active "Play Recording" playback.
    StopPlayback,
    /// Picks which output device "Play Recording" uses - see
    /// `Settings::recording_playback_device_id`'s docs on why this is
    /// remembered for the next click rather than driving a live switch.
    SelectPlaybackDevice(DeviceOption),
    /// Speaker-outbound (Pipeline 2) messages - see `tse_engine::SpeakerBvcBackend`'s
    /// docs on why this path's BVC picker is a smaller, separate enum from
    /// the mic path's.
    ToggleSpeakerNoise(bool),
    ToggleSpeakerStudio(bool),
    SelectSpeakerStudioPreset(&'static str),
    SelectSpeakerBvcBackend(&'static str),
    SelectSpeakerSource(DeviceOption),
    SelectRealSpeakerOutput(DeviceOption),
    /// Master mute for the whole mic pipeline - see
    /// `EngineHandles::mic_master_enabled`'s docs: unlike the individual
    /// Noise/BVC/Studio toggles, this genuinely silences the output rather
    /// than just skipping DSP processing.
    ToggleMicMasterEnabled(bool),
    /// Same as `ToggleMicMasterEnabled`, for the speaker path.
    ToggleSpeakerMasterEnabled(bool),
}

fn persist(settings: &Settings) {
    if let Err(e) = crate::settings::save(settings) {
        crate::log_error!("[clearnairt] failed to persist settings: {e:#}");
    }
}

/// Runs a `LiveDeviceSwitcher` call on a background thread rather than
/// inline in `update_main` (iced's UI thread). `LiveAudioSwitcher::set_*`
/// opens a fresh WASAPI device and then blocks joining the *old* worker
/// thread - real hardware log showed that join can take anywhere from the
/// event-wait timeout up to (with `audio-io`'s reconnect-on-failure loop)
/// however long the old thread's in-flight device-open attempt takes to
/// return, which can be much longer than a UI frame. Calling it inline froze
/// the whole window ("Not Responding") on every device switch. `switcher` is
/// `Arc`-shared and `Send + Sync`, so this is safe to run off-thread; any
/// failure is already logged by `LiveAudioSwitcher` itself.
///
/// `seq`/`this_seq`: since each call now races an unpredictable device-open
/// duration on its own thread, two rapid picks for the *same* role (a
/// double-click, or clicking a different device before the first switch
/// finished) could otherwise land in either order - whichever background
/// thread happens to finish last silently wins, even if it was the user's
/// *first* (now-stale) choice. `this_seq` is this call's position in
/// dispatch order; if `seq` has moved past it by the time this runs, a
/// newer request for the same role has already been queued (and will apply
/// its own choice right after), so this stale one is skipped rather than
/// briefly - or permanently - overriding it.
fn spawn_device_switch(
    switcher: Arc<dyn crate::shared::LiveDeviceSwitcher>,
    seq: Arc<AtomicU64>,
    this_seq: u64,
    call: impl FnOnce(&dyn crate::shared::LiveDeviceSwitcher) + Send + 'static,
) {
    std::thread::spawn(move || {
        if seq.load(Ordering::SeqCst) != this_seq {
            return;
        }
        call(switcher.as_ref())
    });
}

/// Bumps `seq` and returns the new value, for `spawn_device_switch`'s
/// `this_seq` argument. Must be called synchronously on the UI thread right
/// before dispatching, so dispatch order and sequence-number order match.
fn next_seq(seq: &AtomicU64) -> u64 {
    seq.fetch_add(1, Ordering::SeqCst) + 1
}

/// Top-level dispatch: routes to whichever screen is currently active, and
/// handles the one-way `Setup -> Main` transition when `update_setup`
/// reports the user is done with the setup screen.
fn update(screen: &mut Screen, message: Message) {
    let transition = match screen {
        Screen::Main(state) => {
            update_main(state, message);
            None
        }
        Screen::Setup(setup) => update_setup(setup, message),
    };
    if let Some(main_state) = transition {
        *screen = Screen::Main(main_state);
    }
}

/// Handles the setup screen's messages. There is no background work left to
/// poll here (BVC asset extraction is synchronous and already finished
/// before this screen is ever shown - see `main.rs::bootstrap`), so `Tick`
/// is a no-op on this screen; the only real transition is the user clicking
/// through. Returns `Some(MainState)` exactly then, signalling the
/// top-level `update` to switch screens.
fn update_setup(setup: &mut SetupState, message: Message) -> Option<MainState> {
    match message {
        Message::Tick => None,
        Message::ContinueFromSetup => {
            // Only the compatibility-warning half of the setup screen is
            // ever suppressed by this flag on future launches - a missing
            // model is independently re-checked fresh every launch (see
            // `Settings::setup_acknowledged`'s docs), so "Skip" here never
            // hides a real, currently-true problem, only an
            // already-acknowledged one.
            setup.settings.setup_acknowledged = true;
            persist(&setup.settings);
            Some(MainState {
                settings: setup.settings.clone(),
                handles: setup.handles.clone(),
                devices: setup.devices.clone(),
                warnings: setup.warnings.clone(),
                status: RealtimeStatus::Bypass,
                level: 0.0,
                _keepalive: setup.keepalive.take(),
                device_switch_seqs: Arc::new(DeviceSwitchSequences::default()),
                io_counters_snapshot: IoCountersSnapshot::default(),
                speaker_status: RealtimeStatus::Bypass,
                speaker_io_counters_snapshot: IoCountersSnapshot::default(),
                consecutive_overloaded_ticks: 0,
                show_low_cpu_suggestion: false,
                enrollment_state: enrollment::EnrollmentState::Idle,
                playback: None,
            })
        }
        // Every other message is a Main-screen-only toggle/picker action;
        // the setup screen's `view` never emits these, so this is
        // unreachable in practice, not a silently-swallowed real message.
        _ => None,
    }
}

fn update_main(state: &mut MainState, message: Message) {
    match message {
        Message::Tick => {
            let snapshot = state.handles.latency_log.snapshot();
            let percentiles = compute_percentiles(snapshot);
            // "Bypass" here means every mic-path DSP stage is currently
            // off, i.e. the mic pipeline is doing pure passthrough - a
            // reasonable, documented interpretation of `classify_status`'s
            // `bypass` flag, which `dsp_core` leaves for the caller to
            // define.
            let all_off = !state.handles.mic_noise.is_on()
                && !state.handles.mic_bvc.is_on()
                && !state.handles.mic_studio.is_on();
            state.status = classify_status(&percentiles, DEADLINE_NS, all_off);
            state.level = state.handles.level.load();
            state.io_counters_snapshot = IoCountersSnapshot::read(&state.handles.counters);

            let speaker_snapshot = state.handles.speaker_latency_log.snapshot();
            let speaker_percentiles = compute_percentiles(speaker_snapshot);
            let speaker_all_off = !state.handles.speaker_noise.is_on()
                && !state.handles.speaker_bvc.is_on()
                && !state.handles.speaker_studio.is_on();
            state.speaker_status = classify_status(&speaker_percentiles, DEADLINE_NS, speaker_all_off);
            state.speaker_io_counters_snapshot = IoCountersSnapshot::read(&state.handles.speaker_counters);

            state.enrollment_state = state.handles.enrollment.state();

            // Either pipeline being overloaded counts - a user hitting
            // Overloaded on the SPEAKER path deserves the same suggestion
            // as the mic path (see `Message::EnableLowCpuMode`, now updated
            // to relieve both paths at once).
            if state.status == RealtimeStatus::Overloaded || state.speaker_status == RealtimeStatus::Overloaded {
                state.consecutive_overloaded_ticks += 1;
            } else {
                state.consecutive_overloaded_ticks = 0;
            }
            if state.consecutive_overloaded_ticks >= LOW_CPU_SUGGESTION_THRESHOLD_TICKS {
                state.show_low_cpu_suggestion = true;
            }

            if let Some(handle) = state.handles.tse_offline_test.take_ready_playback() {
                state.playback = Some(handle);
            }
            if let Some(err) = state.handles.tse_offline_test.take_error() {
                crate::log_error!("[clearnairt] TSE offline test run failed: {err}");
            }
        }
        Message::ToggleMicNoise(on) => {
            state.handles.mic_noise.set(on);
            state.settings.mic_noise_on = on;
            persist(&state.settings);
        }
        Message::ToggleMicVoiceGate(on) => {
            state.handles.mic_voice_gate.set(on);
            state.settings.mic_voice_gate_on = on;
            persist(&state.settings);
        }
        Message::ToggleMicStudio(on) => {
            state.handles.mic_studio.set(on);
            state.settings.mic_studio_on = on;
            persist(&state.settings);
        }
        Message::SelectMicStudioPreset(name) => {
            if let Some(preset) = crate::shared::studio_preset_from_name(name) {
                let index = studio_preset_to_index(preset);
                state.handles.mic_studio_preset.store(index);
                state.settings.mic_studio_preset = index;
                persist(&state.settings);
            }
        }
        Message::SelectMicTarget(device) => {
            state.settings.virtual_mic_target_id = Some(device.id.clone());
            persist(&state.settings);
            // Optimistic clear, same reasoning as `SelectSpeakerSource`'s -
            // otherwise "No virtual mic target device selected"/"Failed to
            // start render into the selected virtual-mic-target device"
            // would stay on screen forever, even right after the user just
            // fixed exactly that.
            state.warnings.retain(|w| {
                !w.contains("virtual mic target device selected") && !w.contains("selected virtual-mic-target device")
            });
            // Live switch, not just a next-launch preference: tears down the
            // render thread on the previous virtual-mic-target device (if
            // any) and starts a fresh one on `device.id`, reconnecting the
            // mic pipeline thread's output to a new ring buffer. See
            // `engine::LiveAudioSwitcher::set_virtual_mic_target`. Run off
            // the UI thread - see `spawn_device_switch`.
            let seq = state.device_switch_seqs.mic_target.clone();
            let this_seq = next_seq(&seq);
            spawn_device_switch(state.handles.device_switcher.clone(), seq, this_seq, move |s| {
                s.set_virtual_mic_target(Some(device.id))
            });
        }
        Message::SelectPhysicalMic(device) => {
            state.settings.physical_mic_device_id = Some(device.id.clone());
            persist(&state.settings);
            let seq = state.device_switch_seqs.physical_mic.clone();
            let this_seq = next_seq(&seq);
            spawn_device_switch(state.handles.device_switcher.clone(), seq, this_seq, move |s| {
                s.set_physical_mic(Some(device.id))
            });
        }
        Message::ToggleMonitor(on) => {
            state.handles.monitor.set(on);
            state.settings.monitor_enabled = on;
            persist(&state.settings);
        }
        Message::SelectMonitorDevice(device) => {
            state.settings.monitor_device_id = Some(device.id.clone());
            persist(&state.settings);
            state.warnings.retain(|w| !w.contains("Monitor output device could not be started"));
            let seq = state.device_switch_seqs.monitor.clone();
            let this_seq = next_seq(&seq);
            spawn_device_switch(state.handles.device_switcher.clone(), seq, this_seq, move |s| {
                s.set_monitor_device(Some(device.id))
            });
        }
        Message::RefreshDevices => {
            // Enumeration itself (unlike a device switch) never opens a
            // device or joins a worker thread, so it's cheap and safe to run
            // inline on the UI thread rather than needing
            // `spawn_device_switch`'s background-thread treatment.
            state.devices = state.handles.device_switcher.enumerate_devices();
        }
        Message::EnableLowCpuMode => {
            // Plain `StageToggle::set` calls, same as the individual
            // `ToggleMicBvc`/`ToggleMicStudio` handlers - no pipeline
            // rebuild, glitch-free, matches this app's "every stage is
            // always constructed, toggles just gate whether `.process()` is
            // called" rule. Covers BOTH pipelines - either one can be the
            // actual overload source (see `speaker_status`), and turning
            // off a path the user doesn't currently need is harmless.
            if state.handles.mic_bvc_available {
                state.handles.mic_bvc.set(false);
                state.settings.mic_bvc_on = false;
            }
            state.handles.mic_studio.set(false);
            state.settings.mic_studio_on = false;
            if state.handles.speaker_bvc_available || state.handles.speaker_deep_filter_available {
                state.handles.speaker_bvc.set(false);
                state.settings.speaker_bvc_on = false;
            }
            state.handles.speaker_studio.set(false);
            state.settings.speaker_studio_on = false;
            persist(&state.settings);
            state.show_low_cpu_suggestion = false;
            state.consecutive_overloaded_ticks = 0;
        }
        Message::ToggleMicMasterEnabled(on) => {
            // Not persisted - a runtime-only mute (see `EngineHandles::
            // mic_master_enabled`'s docs), always starts back on at launch.
            state.handles.mic_master_enabled.set(on);
        }
        Message::ToggleSpeakerMasterEnabled(on) => {
            state.handles.speaker_master_enabled.set(on);
        }
        Message::SelectBvcBackend(name) => {
            if let Some(backend) = bvc_backend_from_name(name) {
                let index = backend.to_index();
                let wants_on = backend != BvcBackend::Off;
                // Same "never silently enable an unavailable backend"
                // rule `ToggleMicBvc` already enforces for Hush, extended
                // to TSE: picking a backend whose real implementation
                // couldn't be loaded/isn't integrated leaves the actual
                // gate off, even though the *selection* is remembered.
                let available = match backend {
                    BvcBackend::Off => true,
                    BvcBackend::Hush => state.handles.mic_bvc_available,
                    BvcBackend::Tse => state.handles.tse_available,
                    BvcBackend::DeepFilter => state.handles.mic_deep_filter_available,
                    BvcBackend::Vanta => state.handles.mic_vanta_available,
                    BvcBackend::HushBpoV1 => state.handles.mic_bpo_v1_available,
                };
                state.handles.bvc_backend.store(index);
                state.handles.mic_bvc.set(wants_on && available);
                state.settings.mic_bvc_backend = index;
                state.settings.mic_bvc_on = wants_on;
                persist(&state.settings);
            }
        }
        Message::ToggleSpeakerNoise(on) => {
            state.handles.speaker_noise.set(on);
            state.settings.speaker_noise_on = on;
            persist(&state.settings);
        }
        Message::ToggleSpeakerStudio(on) => {
            state.handles.speaker_studio.set(on);
            state.settings.speaker_studio_on = on;
            persist(&state.settings);
        }
        Message::SelectSpeakerStudioPreset(name) => {
            if let Some(preset) = crate::shared::studio_preset_from_name(name) {
                let index = studio_preset_to_index(preset);
                state.handles.speaker_studio_preset.store(index);
                state.settings.speaker_studio_preset = index;
                persist(&state.settings);
            }
        }
        Message::SelectSpeakerBvcBackend(name) => {
            if let Some(backend) = speaker_bvc_backend_from_name(name) {
                let index = backend.to_index();
                let wants_on = backend != tse_engine::SpeakerBvcBackend::Off;
                let available = match backend {
                    tse_engine::SpeakerBvcBackend::Off => true,
                    tse_engine::SpeakerBvcBackend::Hush => state.handles.speaker_bvc_available,
                    tse_engine::SpeakerBvcBackend::DeepFilter => state.handles.speaker_deep_filter_available,
                };
                state.handles.speaker_bvc_backend.store(index);
                state.handles.speaker_bvc.set(wants_on && available);
                state.settings.speaker_bvc_backend = index;
                state.settings.speaker_bvc_on = wants_on;
                persist(&state.settings);
            }
        }
        Message::SelectSpeakerSource(device) => {
            state.settings.virtual_speaker_source_id = Some(device.id.clone());
            persist(&state.settings);
            // Optimistic clear (same "assume the switch works, log on
            // failure elsewhere" trust the toggles already extend to the
            // rest of this app) - otherwise the startup "no speaker-source
            // device selected" warning would stay on screen forever, even
            // after the user just fixed exactly that.
            state.warnings.retain(|w| !w.contains("speaker-source device selected") && !w.contains("selected speaker-source device"));
            let seq = state.device_switch_seqs.speaker_source.clone();
            let this_seq = next_seq(&seq);
            spawn_device_switch(state.handles.device_switcher.clone(), seq, this_seq, move |s| {
                s.set_virtual_speaker_source(Some(device.id))
            });
        }
        Message::SelectRealSpeakerOutput(device) => {
            state.settings.real_speaker_output_device_id = Some(device.id.clone());
            persist(&state.settings);
            state.warnings.retain(|w| !w.contains("real speaker/headphone output device"));
            let seq = state.device_switch_seqs.real_speaker_output.clone();
            let this_seq = next_seq(&seq);
            spawn_device_switch(state.handles.device_switcher.clone(), seq, this_seq, move |s| {
                s.set_real_speaker_output(Some(device.id))
            });
        }
        Message::StartEnrollment | Message::RestartEnrollment => {
            state.handles.enrollment.start_recording(dsp_core::SAMPLE_RATE_HZ);
            state.enrollment_state = state.handles.enrollment.state();
        }
        Message::StopEnrollment => {
            state.handles.enrollment.stop_recording();
        }
        Message::ClearEnrollment => {
            state.handles.enrollment.clear_profile();
            state.enrollment_state = state.handles.enrollment.state();
        }
        Message::PlayLastRecording => {
            if let Some((samples, _sample_rate_hz)) = state.handles.enrollment.last_recording() {
                let device_id = state.settings.recording_playback_device_id.clone();
                state.playback = Some(crate::shared::play_recorded_audio(samples, device_id));
            }
        }
        Message::PlayDenoisedRecording => {
            if let Some((samples, _sample_rate_hz)) = state.handles.enrollment.last_recording_denoised() {
                let device_id = state.settings.recording_playback_device_id.clone();
                state.playback = Some(crate::shared::play_recorded_audio(samples, device_id));
            }
        }
        Message::TestWithTse => {
            if let (Some((samples, _sample_rate_hz)), Some(profile)) =
                (state.handles.enrollment.last_recording(), state.handles.tse_speaker_store.get_profile())
            {
                let device_id = state.settings.recording_playback_device_id.clone();
                state.handles.tse_offline_test.start(samples, profile, device_id);
            }
        }
        Message::SelectPlaybackDevice(device) => {
            // Not a live-switching device the way `SelectMonitorDevice` is
            // (no continuously-running stream to redirect) - just
            // remembered for the next "Play Recording" click, see
            // `Settings::recording_playback_device_id`'s docs.
            state.settings.recording_playback_device_id = Some(device.id);
            persist(&state.settings);
        }
        Message::PausePlayback => {
            if let Some(playback) = &state.playback {
                playback.pause();
            }
        }
        Message::ResumePlayback => {
            if let Some(playback) = &state.playback {
                playback.resume();
            }
        }
        Message::StopPlayback => {
            if let Some(playback) = &state.playback {
                playback.stop();
            }
        }
        // Setup-screen-only messages; unreachable once `Screen::Main` is
        // active (its `view` never emits them), same reasoning as the
        // catch-all in `update_setup`.
        Message::ContinueFromSetup => {}
    }
}

/// Maps a `pick_list`'s selected label back to `tse_engine::BvcBackend` -
/// mirrors `crate::shared::studio_preset_from_name`'s existing pattern for
/// the Studio preset picker.
fn bvc_backend_from_name(name: &str) -> Option<tse_engine::BvcBackend> {
    tse_engine::BvcBackend::ALL.into_iter().find(|b| b.name() == name)
}

/// Same as `bvc_backend_from_name` but for the speaker path's smaller
/// `SpeakerBvcBackend` enum.
fn speaker_bvc_backend_from_name(name: &str) -> Option<tse_engine::SpeakerBvcBackend> {
    tse_engine::SpeakerBvcBackend::ALL.into_iter().find(|b| b.name() == name)
}

fn status_row(state: &MainState) -> Element<'_, Message> {
    let color = status_color(state.status);
    let dot = container(text("●").size(20).color(Color::from_rgb(color.r, color.g, color.b)));
    let label = text(format!("Engine: {}", status_label(state.status))).size(16);
    let level_bar = progress_bar(0.0..=1.0, state.level.clamp(0.0, 1.0)).girth(Length::Fixed(10.0));

    // Real, live counts (not a guess from watching logs scroll by) - see
    // `IoCountersSnapshot`. Colored amber once any drop/underrun has ever
    // happened this session, so a perfectly healthy run stays visually
    // quiet (plain grey text) instead of drawing the eye to a "0" that
    // doesn't need attention.
    let io = state.io_counters_snapshot;
    let has_issues = io.dropped_frames > 0 || io.input_overruns > 0 || io.output_underruns > 0;
    let io_color = if has_issues {
        Color::from_rgb(0.90, 0.75, 0.15)
    } else {
        Color::from_rgb(0.55, 0.55, 0.55)
    };
    let io_row = text(format!(
        "Dropped frames: {}  |  Input overruns: {}  |  Output underruns: {}",
        io.dropped_frames, io.input_overruns, io.output_underruns
    ))
    .size(12)
    .color(io_color);

    column![
        row![dot, label].spacing(8).align_y(iced::Alignment::Center),
        row![text("Input level").size(12), level_bar].spacing(8).align_y(iced::Alignment::Center),
        io_row,
    ]
    .spacing(4)
    .into()
}

/// Same layout as `status_row`, for the speaker-outbound pipeline's own,
/// independent telemetry (`speaker_status`/`speaker_io_counters_snapshot`)
/// - no input-level meter here (that field only exists for the mic path).
fn speaker_status_row(state: &MainState) -> Element<'_, Message> {
    let color = status_color(state.speaker_status);
    let dot = container(text("●").size(20).color(Color::from_rgb(color.r, color.g, color.b)));
    let label = text(format!("Speaker pipeline: {}", status_label(state.speaker_status))).size(16);

    let io = state.speaker_io_counters_snapshot;
    let has_issues = io.dropped_frames > 0 || io.input_overruns > 0 || io.output_underruns > 0;
    let io_color = if has_issues {
        Color::from_rgb(0.90, 0.75, 0.15)
    } else {
        Color::from_rgb(0.55, 0.55, 0.55)
    };
    let io_row = text(format!(
        "Dropped frames: {}  |  Input overruns: {}  |  Output underruns: {}",
        io.dropped_frames, io.input_overruns, io.output_underruns
    ))
    .size(12)
    .color(io_color);

    column![
        row![dot, label].spacing(8).align_y(iced::Alignment::Center),
        io_row,
    ]
    .spacing(4)
    .into()
}

/// The "Record Voice Reference" / target-speaker-enrollment section shown
/// under the BVC backend picker - see this task's GUI mockup: a target-
/// speaker indicator, the reference sentence, a record/re-record/clear
/// button set, and a plain-language status line. Never exposes model
/// internals (no embedding dimension, no encoder name) to the user - only
/// `Ready`/`Recording`/`Processing`/`Error` - matching the task's "do not
/// expose model internals to the user" requirement.
fn enrollment_section(state: &MainState) -> Element<'_, Message> {
    let enrolled = state.handles.target_speaker_store.has_profile();
    let target_speaker_line = if enrolled {
        text("Target Speaker: \u{2713} Enrolled").size(13).color(Color::from_rgb(0.20, 0.75, 0.30))
    } else {
        text("Target Speaker: Not enrolled").size(13)
    };

    let (status_text, status_color_rgb) = match &state.enrollment_state {
        EnrollmentState::Idle => (
            if enrolled { "Ready".to_string() } else { "Not started".to_string() },
            Color::from_rgb(0.55, 0.55, 0.55),
        ),
        EnrollmentState::Preparing => ("Preparing...".to_string(), Color::from_rgb(0.55, 0.55, 0.55)),
        EnrollmentState::Recording => ("Recording - speak the sentence below".to_string(), Color::from_rgb(0.20, 0.55, 0.85)),
        EnrollmentState::Validating => ("Processing (checking recording quality)...".to_string(), Color::from_rgb(0.55, 0.55, 0.55)),
        EnrollmentState::ExtractingEmbedding => ("Processing (analyzing voice)...".to_string(), Color::from_rgb(0.55, 0.55, 0.55)),
        EnrollmentState::Ready => ("Enrolled".to_string(), Color::from_rgb(0.20, 0.75, 0.30)),
        EnrollmentState::Failed(e) => (format!("Error: {e}"), Color::from_rgb(0.85, 0.2, 0.2)),
    };

    let in_progress = state.enrollment_state.is_in_progress();

    let mut buttons = row![].spacing(8);
    if in_progress {
        buttons = buttons.push(button(text("Stop recording").size(13)).on_press(Message::StopEnrollment));
    } else if enrolled {
        buttons = buttons.push(button(text("Re-record Reference").size(13)).on_press(Message::RestartEnrollment));
        buttons = buttons.push(button(text("Clear Reference").size(13)).on_press(Message::ClearEnrollment));
    } else {
        buttons = buttons.push(button(text("Record Voice Reference").size(13)).on_press(Message::StartEnrollment));
    }
    // Playback transport - shown whenever a recording exists, even a
    // rejected/failed one (see `Message::PlayLastRecording`'s docs) - not
    // gated on `enrolled` or `in_progress` the way the buttons above are,
    // since "hear what I just recorded" is useful precisely when something
    // went wrong. While a playback is actually running/paused, this
    // replaces the single "Play Recording" button with Pause/Resume + Stop
    // rather than allowing a second overlapping playback to start.
    let playback_status = state.playback.as_ref().map(|p| p.status());
    let mut playback_progress_row: Option<Element<'_, Message>> = None;
    if !in_progress && state.handles.enrollment.has_last_recording() {
        match playback_status {
            Some(crate::shared::PlaybackStatus::Playing) => {
                buttons = buttons.push(button(text("Pause").size(13)).on_press(Message::PausePlayback));
                buttons = buttons.push(button(text("Stop").size(13)).on_press(Message::StopPlayback));
                if let Some(p) = &state.playback {
                    playback_progress_row = Some(
                        row![
                            text("Playing:").size(12).color(Color::from_rgb(0.55, 0.55, 0.55)),
                            progress_bar(0.0..=1.0, p.progress()).girth(Length::Fixed(8.0)),
                        ]
                        .spacing(6)
                        .align_y(iced::Alignment::Center)
                        .into(),
                    );
                }
            }
            Some(crate::shared::PlaybackStatus::Paused) => {
                buttons = buttons.push(button(text("Resume").size(13)).on_press(Message::ResumePlayback));
                buttons = buttons.push(button(text("Stop").size(13)).on_press(Message::StopPlayback));
                if let Some(p) = &state.playback {
                    playback_progress_row = Some(
                        row![
                            text("Paused:").size(12).color(Color::from_rgb(0.55, 0.55, 0.55)),
                            progress_bar(0.0..=1.0, p.progress()).girth(Length::Fixed(8.0)),
                        ]
                        .spacing(6)
                        .align_y(iced::Alignment::Center)
                        .into(),
                    );
                }
            }
            // No playback yet, or a previous one already finished/was
            // stopped - offer to start a fresh one, either the raw
            // capture or the same denoised version real enrollment
            // extracts the voice embedding from (see
            // `Message::PlayDenoisedRecording`'s docs) - both share this
            // one transport (Pause/Resume/Stop), only one can play at a
            // time.
            None | Some(crate::shared::PlaybackStatus::Finished) | Some(crate::shared::PlaybackStatus::Stopped) => {
                buttons = buttons.push(button(text("Play Recording").size(13)).on_press(Message::PlayLastRecording));
                buttons = buttons.push(button(text("Play Denoised").size(13)).on_press(Message::PlayDenoisedRecording));
                // Offline TSE test - see `Message::TestWithTse`'s docs on
                // why this is the only way to hear full-quality TSE output
                // today. Needs a TSE-specific (192-dim) enrolled embedding,
                // not just the primary 256-dim one `enrolled` above checks.
                if state.handles.tse_offline_test.is_in_progress() {
                    buttons = buttons.push(
                        text("Testing with TSE... (can take up to a minute)").size(12).color(Color::from_rgb(0.55, 0.55, 0.55)),
                    );
                } else if state.handles.tse_speaker_store.has_profile() {
                    buttons = buttons.push(button(text("Test with TSE").size(13)).on_press(Message::TestWithTse));
                }
            }
        }
    }

    let mut section = column![target_speaker_line, buttons,];
    if let Some(progress_row) = playback_progress_row {
        section = section.push(progress_row);
    }
    // Shown whenever the playback feature is at all relevant (a recording
    // to play back exists), not just while something is actively playing -
    // this is the device the *next* click will use, per
    // `Settings::recording_playback_device_id`'s docs.
    if state.handles.enrollment.has_last_recording() {
        section = section.push(device_picker(
            "Playback device (for \"Play Recording\")",
            &state.devices.render,
            &state.settings.recording_playback_device_id,
            Message::SelectPlaybackDevice,
        ));
    }

    section
    .push(text("Reference sentence:").size(12).color(Color::from_rgb(0.55, 0.55, 0.55)))
    .push(text(format!("\u{201C}{}\u{201D}", state.settings.reference_sentence)).size(13))
    .push(
        row![
            text("Status:").size(12).color(Color::from_rgb(0.55, 0.55, 0.55)),
            text(status_text).size(12).color(status_color_rgb),
        ]
        .spacing(6),
    )
    .spacing(8)
    .into()
}

fn device_picker<'a>(
    label: &'a str,
    options: &'a [DeviceOption],
    selected_id: &Option<String>,
    on_select: impl Fn(DeviceOption) -> Message + 'a,
) -> Element<'a, Message> {
    let selected = selected_id
        .as_ref()
        .and_then(|id| options.iter().find(|d| &d.id == id).cloned());
    column![
        text(label).size(14),
        pick_list(options, selected, on_select).placeholder("(not selected - applies immediately)"),
    ]
    .spacing(4)
    .into()
}

/// Every `tse_engine::BvcBackend` name, in `BvcBackend::ALL` order - a
/// plain `[&'static str; 6]` so `pick_list` can borrow it the same way
/// `STUDIO_PRESET_NAMES` already works for the Studio preset picker.
const BVC_BACKEND_NAMES: [&str; 6] = [
    BvcBackend::Off.name(),
    BvcBackend::Hush.name(),
    BvcBackend::DeepFilter.name(),
    BvcBackend::Tse.name(),
    BvcBackend::Vanta.name(),
    BvcBackend::HushBpoV1.name(),
];

/// Same as `BVC_BACKEND_NAMES` but for the speaker path's smaller
/// `SpeakerBvcBackend` enum - no TSE/Vanta (see that type's docs).
const SPEAKER_BVC_BACKEND_NAMES: [&str; 3] = [
    tse_engine::SpeakerBvcBackend::Off.name(),
    tse_engine::SpeakerBvcBackend::Hush.name(),
    tse_engine::SpeakerBvcBackend::DeepFilter.name(),
];

fn view_main(state: &MainState) -> Element<'_, Message> {
    let selected_backend = BvcBackend::from_index(state.handles.bvc_backend.load());
    let bvc_unavailable_note: Option<Element<'_, Message>> = match selected_backend {
        BvcBackend::Hush if !state.handles.mic_bvc_available => Some(
            text(format!(
                "Hush is unavailable: {}",
                state.handles.mic_bvc_unavailable_reason.as_deref().unwrap_or("weya_nc.dll not found")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        BvcBackend::Tse if !state.handles.tse_available => Some(
            text(format!(
                "TSE is unavailable: {}",
                state.handles.tse_unavailable_reason.as_deref().unwrap_or("no model bundled")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        BvcBackend::DeepFilter if !state.handles.mic_deep_filter_available => Some(
            text(format!(
                "DeepFilterNet is unavailable: {}",
                state.handles.mic_deep_filter_unavailable_reason.as_deref().unwrap_or("model failed to load")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        BvcBackend::DeepFilter => Some(
            text("Generic noise/other-voice suppression (DeepFilterNet) - like Hush, this doesn't target a specific speaker. For suppressing a specific person's voice, use Target Speaker (TSE).")
                .size(11)
                .color(Color::from_rgb(0.55, 0.55, 0.55))
                .into(),
        ),
        BvcBackend::Vanta if !state.handles.mic_vanta_available => Some(
            text(format!(
                "Vanta is unavailable: {}",
                state.handles.mic_vanta_unavailable_reason.as_deref().unwrap_or("model failed to load")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        BvcBackend::Vanta => Some(
            text(
                state
                    .handles
                    .mic_vanta_unavailable_reason
                    .as_deref()
                    .unwrap_or("EXPERIMENTAL: real but weak target-speaker model - may sometimes make audio worse, not better."),
            )
            .size(12)
            .color(Color::from_rgb(0.85, 0.6, 0.1))
            .into(),
        ),
        BvcBackend::HushBpoV1 if !state.handles.mic_bpo_v1_available => Some(
            text(format!(
                "bpo_v1 is unavailable: {}",
                state.handles.mic_bpo_v1_unavailable_reason.as_deref().unwrap_or("model failed to load")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        BvcBackend::HushBpoV1 => Some(
            text("A fine-tune of Hush's own checkpoint (bpo_v1) - same Hush engine and architecture as the default option above, just a different trained model. Does not replace the default Hush option, which remains separately selectable.")
                .size(11)
                .color(Color::from_rgb(0.55, 0.55, 0.55))
                .into(),
        ),
        _ => None,
    };

    let bvc_backend_row = row![
        text("Backend").size(12),
        pick_list(&BVC_BACKEND_NAMES[..], Some(selected_backend.name()), Message::SelectBvcBackend),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center);

    let mut bvc_section = column![text("Background Voice Cancellation (BVC)").size(14), bvc_backend_row].spacing(6);
    if let Some(note) = bvc_unavailable_note {
        bvc_section = bvc_section.push(note);
    }
    bvc_section = bvc_section.push(enrollment_section(state));

    let mic_studio_preset_row = row![
        text("Studio preset").size(12),
        pick_list(
            &STUDIO_PRESET_NAMES[..],
            Some(studio_preset_name(studio_preset_from_index(state.handles.mic_studio_preset.load()))),
            Message::SelectMicStudioPreset,
        ),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center);

    let monitor_section = column![
        text("Monitor - hear the processed mic locally, for testing (not part of the normal virtual-mic/speaker signal path)").size(13),
        toggler(state.handles.monitor.is_on())
            .label("Monitor (hear processed mic locally)")
            .on_toggle(Message::ToggleMonitor),
        device_picker(
            "Monitor output device (e.g. your headphones)",
            &state.devices.render,
            &state.settings.monitor_device_id,
            Message::SelectMonitorDevice,
        ),
    ]
    .spacing(8);

    let toggles = column![
        toggler(state.handles.mic_noise.is_on())
            .label("RNNoise (between Mic and Virtual Mic)")
            .on_toggle(Message::ToggleMicNoise),
        column![
            toggler(state.handles.mic_voice_gate.is_on())
                .label("Voice Gate (suppress quiet background voices)")
                .on_toggle(Message::ToggleMicVoiceGate),
            text("Attenuates background talk that's quieter than your own voice - a fast, real-time alternative to Target Speaker (TSE) for background voices at a similar or lower volume, not just noise. Won't help against a background voice speaking at the same volume as you - that needs TSE.")
                .size(11)
                .color(Color::from_rgb(0.55, 0.55, 0.55)),
        ]
        .spacing(2),
        toggler(state.handles.mic_studio.is_on())
            .label("Studio")
            .on_toggle(Message::ToggleMicStudio),
        mic_studio_preset_row,
    ]
    .spacing(10);

    // Turns off the two most CPU-expensive mic-path stages (BVC's ONNX
    // inference, then Studio's EQ/compressor chain) in one click - see
    // `Message::EnableLowCpuMode`'s doc comment. Placed right under the
    // toggles it affects rather than buried in a settings menu, since it's
    // meant to be reached for exactly when the "Dropped frames"/"Engine:
    // Overloaded" indicators above are already visibly bad.
    let low_cpu_row = row![
        button(text("Low CPU mode (turn off BVC + Studio, both paths)").size(13)).on_press(Message::EnableLowCpuMode),
        text("Keeps RNNoise on - use this if the Engine status above shows Degraded/Overloaded.").size(11),
    ]
    .spacing(12)
    .align_y(iced::Alignment::Center);

    let mut low_cpu_section = column![low_cpu_row].spacing(6);
    if state.show_low_cpu_suggestion {
        // Auto-detected (see `Message::Tick`'s `consecutive_overloaded_ticks`
        // handling) - this only ever *suggests* clicking the button above,
        // never flips the toggles itself. Silently degrading forever
        // without telling the user is exactly the failure mode this exists
        // to avoid.
        low_cpu_section = low_cpu_section.push(
            text("⚠ Sustained overload detected - processing is consistently missing its 10ms deadline. \
                  Consider clicking Low CPU mode above.")
                .size(12)
                .color(Color::from_rgb(0.85, 0.6, 0.1)),
        );
    }

    let devices_heading = row![
        text("Virtual devices (see docs/VIRTUAL_DEVICES.md - two separate cable instances required)").size(13),
        button(text("Refresh devices").size(12)).on_press(Message::RefreshDevices),
    ]
    .spacing(12)
    .align_y(iced::Alignment::Center);

    let devices = column![
        devices_heading,
        text("Started VoiceMeter/VB-Cable after ClearNAI? Click Refresh, then (re)select it below - devices are only scanned once at launch.").size(12),
        device_picker(
            "Virtual mic target (render into this device)",
            &state.devices.render,
            &state.settings.virtual_mic_target_id,
            Message::SelectMicTarget,
        ),
        device_picker(
            "Physical microphone (real input device)",
            &state.devices.capture,
            &state.settings.physical_mic_device_id,
            Message::SelectPhysicalMic,
        ),
    ]
    .spacing(8);

    // --- Speaker-outbound (Pipeline 2): incoming-call audio cleanup, own
    // BVC picker (Off/Hush/DeepFilterNet only - no TSE/Vanta, see
    // `tse_engine::SpeakerBvcBackend`'s docs), own toggles, own device pair
    // (a SECOND, independent virtual cable's capture endpoint + the user's
    // real speaker/headphone output). Entirely separate from the mic
    // section above - the two paths run on independent threads and can be
    // configured/toggled independently.
    let selected_speaker_backend = tse_engine::SpeakerBvcBackend::from_index(state.handles.speaker_bvc_backend.load());
    let speaker_bvc_unavailable_note: Option<Element<'_, Message>> = match selected_speaker_backend {
        tse_engine::SpeakerBvcBackend::Hush if !state.handles.speaker_bvc_available => Some(
            text(format!(
                "Hush is unavailable: {}",
                state.handles.speaker_bvc_unavailable_reason.as_deref().unwrap_or("weya_nc.dll not found")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        tse_engine::SpeakerBvcBackend::DeepFilter if !state.handles.speaker_deep_filter_available => Some(
            text(format!(
                "DeepFilterNet is unavailable: {}",
                state.handles.speaker_deep_filter_unavailable_reason.as_deref().unwrap_or("model failed to load")
            ))
            .size(12)
            .color(Color::from_rgb(0.85, 0.2, 0.2))
            .into(),
        ),
        _ => None,
    };

    let speaker_bvc_backend_row = row![
        text("Backend").size(12),
        pick_list(&SPEAKER_BVC_BACKEND_NAMES[..], Some(selected_speaker_backend.name()), Message::SelectSpeakerBvcBackend),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center);

    let mut speaker_bvc_section = column![
        text("Background Voice Cancellation (BVC) - Speaker").size(14),
        speaker_bvc_backend_row,
        text("No target-speaker (TSE) options here - there's no way to enroll an arbitrary caller's voice ahead of time.")
            .size(11)
            .color(Color::from_rgb(0.55, 0.55, 0.55)),
    ]
    .spacing(6);
    if let Some(note) = speaker_bvc_unavailable_note {
        speaker_bvc_section = speaker_bvc_section.push(note);
    }

    let speaker_studio_preset_row = row![
        text("Studio preset").size(12),
        pick_list(
            &STUDIO_PRESET_NAMES[..],
            Some(studio_preset_name(studio_preset_from_index(state.handles.speaker_studio_preset.load()))),
            Message::SelectSpeakerStudioPreset,
        ),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center);

    let speaker_toggles = column![
        toggler(state.handles.speaker_noise.is_on())
            .label("RNNoise (between Speaker Source and Physical Speaker)")
            .on_toggle(Message::ToggleSpeakerNoise),
        toggler(state.handles.speaker_studio.is_on())
            .label("Studio")
            .on_toggle(Message::ToggleSpeakerStudio),
        speaker_studio_preset_row,
    ]
    .spacing(10);

    let speaker_devices = column![
        text("Speaker-outbound devices (needs a SECOND, independent virtual cable - e.g. VB-CABLE A+B - separate from the mic path's)").size(13),
        device_picker(
            "Virtual speaker source (capture from this device - set your call app's output to its matching input)",
            &state.devices.capture,
            &state.settings.virtual_speaker_source_id,
            Message::SelectSpeakerSource,
        ),
        device_picker(
            "Physical speaker (real output device)",
            &state.devices.render,
            &state.settings.real_speaker_output_device_id,
            Message::SelectRealSpeakerOutput,
        ),
    ]
    .spacing(8);

    let mut warnings_col = column![].spacing(4);
    for w in &state.warnings {
        warnings_col = warnings_col.push(text(format!("\u{26A0} {w}")).size(12).color(Color::from_rgb(0.85, 0.6, 0.1)));
    }

    let mic_status_row = row![
        status_row(state),
        toggler(state.handles.mic_master_enabled.is_on()).label("Mic path enabled").on_toggle(Message::ToggleMicMasterEnabled),
    ]
    .spacing(16)
    .align_y(iced::Alignment::Center);
    let speaker_status_and_button = row![
        speaker_status_row(state),
        toggler(state.handles.speaker_master_enabled.is_on())
            .label("Speaker path enabled")
            .on_toggle(Message::ToggleSpeakerMasterEnabled),
    ]
    .spacing(16)
    .align_y(iced::Alignment::Center);

    let content = column![
        text("ClearNAI").size(24),
        mic_status_row,
        iced::widget::rule::horizontal(1),
        bvc_section,
        iced::widget::rule::horizontal(1),
        monitor_section,
        iced::widget::rule::horizontal(1),
        low_cpu_section,
        toggles,
        iced::widget::rule::horizontal(1),
        devices,
        iced::widget::rule::horizontal(1),
        speaker_status_and_button,
        speaker_bvc_section,
        speaker_toggles,
        speaker_devices,
        warnings_col,
    ]
    .spacing(16)
    .padding(20);

    scrollable(content).into()
}

/// One pass/fail line in the setup screen's compatibility list, reusing the
/// same green/red palette `status_ui::status_color` uses for
/// realtime/overloaded so the whole app reads as one consistent color
/// language rather than inventing a second one here.
fn compat_line<'a>(label: &'a str, ok: bool) -> Element<'a, Message> {
    let color = if ok { Color::from_rgb(0.20, 0.75, 0.30) } else { Color::from_rgb(0.85, 0.20, 0.20) };
    let mark = if ok { "\u{2713}" } else { "\u{2717}" };
    row![
        text(mark).size(16).color(color),
        text(label).size(14),
    ]
    .spacing(8)
    .align_y(iced::Alignment::Center)
    .into()
}

fn view_setup(setup: &SetupState) -> Element<'_, Message> {
    let compat_col = column![
        text("System compatibility").size(16),
        compat_line("Operating system (Windows 10/11)", setup.compat.os_ok),
        compat_line("CPU architecture (x86_64)", setup.compat.arch_ok),
        compat_line("Audio subsystem (COM)", setup.compat.com_ok),
    ]
    .spacing(6);

    let mut details_col = column![].spacing(2);
    for d in &setup.compat.details {
        details_col = details_col.push(text(d).size(11).color(Color::from_rgb(0.6, 0.6, 0.6)));
    }

    // BVC's DLL + model bundle are embedded in the binary and self-extract
    // synchronously before this screen is ever shown (see
    // `main.rs::bootstrap`) - there is no download/progress step left, only
    // a one-line confirmation, or a clear error if extraction itself
    // failed (disk full, unwritable %LOCALAPPDATA%, etc.).
    let model_col: Element<'_, Message> = match &setup.bvc_extraction_error {
        None => column![
            text("BVC files").size(16),
            compat_line("weya_nc.dll and the BVC model bundle ready", true),
        ]
        .spacing(6)
        .into(),
        Some(err) => column![
            text("BVC files").size(16),
            compat_line("failed to prepare weya_nc.dll / BVC model bundle", false),
            text(err.clone()).size(12).color(Color::from_rgb(0.85, 0.2, 0.2)),
        ]
        .spacing(8)
        .into(),
    };

    let content = column![
        text("ClearNAI - first-run setup").size(24),
        compat_col,
        details_col,
        iced::widget::rule::horizontal(1),
        model_col,
        iced::widget::rule::horizontal(1),
        text(
            "You can continue without BVC (background voice cleanup) - the rest of the \
             pipeline works normally and BVC will simply show as unavailable, exactly like \
             today's behavior when weya_nc.dll or the model file fail to load."
        )
        .size(12)
        .color(Color::from_rgb(0.6, 0.6, 0.6)),
        button(text("Continue")).on_press(Message::ContinueFromSetup),
    ]
    .spacing(16)
    .padding(20);

    scrollable(content).into()
}

fn view(screen: &Screen) -> Element<'_, Message> {
    match screen {
        Screen::Setup(setup) => view_setup(setup),
        Screen::Main(state) => view_main(state),
    }
}

fn subscription(_screen: &Screen) -> Subscription<Message> {
    iced::time::every(Duration::from_millis(250)).map(|_| Message::Tick)
}

/// Runs the GUI. Blocks until the window is closed, at which point the
/// whole process exits (no tray icon, no background daemon - per spec).
pub fn run(initial: Initial) -> iced::Result {
    let cell = RefCell::new(Some(initial));
    iced::application(
        move || {
            let initial = cell.borrow_mut().take().expect("boot is only called once");
            let mut warnings = (*initial.handles.warnings).clone();
            warnings.extend(initial.extra_warnings);

            // Show the setup screen when embedded-BVC-asset extraction
            // failed (always surfaced - never suppressed, since it's a
            // real, currently-true problem each time it happens), or when
            // a compatibility check is failing and the user hasn't already
            // acknowledged that on a previous launch (see
            // `Settings::setup_acknowledged`). Otherwise - the ordinary
            // case, extraction having succeeded instantly and compat
            // checks passing (or already acknowledged) - skip straight to
            // the main screen with no setup UI at all.
            let show_setup = initial.bvc_extraction_error.is_some()
                || (!initial.compat.all_ok() && !initial.settings.setup_acknowledged);

            if show_setup {
                Screen::Setup(SetupState {
                    settings: initial.settings,
                    compat: initial.compat,
                    bvc_extraction_error: initial.bvc_extraction_error,
                    handles: initial.handles,
                    devices: initial.devices,
                    warnings,
                    keepalive: initial.keepalive,
                })
            } else {
                Screen::Main(MainState {
                    settings: initial.settings,
                    handles: initial.handles,
                    devices: initial.devices,
                    warnings,
                    status: RealtimeStatus::Bypass,
                    level: 0.0,
                    _keepalive: initial.keepalive,
                    device_switch_seqs: Arc::new(DeviceSwitchSequences::default()),
                    io_counters_snapshot: IoCountersSnapshot::default(),
                    speaker_status: RealtimeStatus::Bypass,
                    speaker_io_counters_snapshot: IoCountersSnapshot::default(),
                    consecutive_overloaded_ticks: 0,
                    show_low_cpu_suggestion: false,
                    enrollment_state: enrollment::EnrollmentState::Idle,
                    playback: None,
                })
            }
        },
        update,
        view,
    )
    .title("ClearNAI")
    .subscription(subscription)
    .window_size(iced::Size::new(560.0, 720.0))
    .run()
}
