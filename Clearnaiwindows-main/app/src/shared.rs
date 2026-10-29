//! Types shared between the audio engine (real on Windows, inert stub
//! elsewhere - see `engine.rs`) and the GUI (`gui.rs`). Nothing in this file
//! touches an OS audio API, so it compiles and its tests run natively on
//! Linux as well as when cross-checked for Windows.

use dsp_core::{FrameCounters, LatencyLog, SharedPreset, Stage, StageToggle, WatchdogStatus};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Maps `studio_dsp::StudioPreset` to/from the plain `u8` index
/// `dsp_core::SharedPreset` carries (that crate can't depend on
/// `studio-dsp`'s real enum - wrong dependency direction). Kept in one place
/// so the app crate never encodes/decodes this mapping ad hoc.
pub const STUDIO_PRESET_NAMES: [&str; 4] = ["Off", "Natural", "Balanced", "Strong"];

pub fn studio_preset_to_index(preset: studio_dsp::StudioPreset) -> u8 {
    match preset {
        studio_dsp::StudioPreset::Off => 0,
        studio_dsp::StudioPreset::Natural => 1,
        studio_dsp::StudioPreset::Balanced => 2,
        studio_dsp::StudioPreset::Strong => 3,
    }
}

pub fn studio_preset_from_index(index: u8) -> studio_dsp::StudioPreset {
    match index {
        0 => studio_dsp::StudioPreset::Off,
        2 => studio_dsp::StudioPreset::Balanced,
        3 => studio_dsp::StudioPreset::Strong,
        // Anything else (including the expected `1`) - `Natural` is the
        // documented default, so an unrecognized index degrades to it
        // rather than panicking.
        _ => studio_dsp::StudioPreset::Natural,
    }
}

pub fn studio_preset_name(preset: studio_dsp::StudioPreset) -> &'static str {
    STUDIO_PRESET_NAMES[studio_preset_to_index(preset) as usize]
}

pub fn studio_preset_from_name(name: &str) -> Option<studio_dsp::StudioPreset> {
    match name {
        "Off" => Some(studio_dsp::StudioPreset::Off),
        "Natural" => Some(studio_dsp::StudioPreset::Natural),
        "Balanced" => Some(studio_dsp::StudioPreset::Balanced),
        "Strong" => Some(studio_dsp::StudioPreset::Strong),
        _ => None,
    }
}

/// Lock-free single-value f32 "mailbox", used to publish the most recent
/// processed frame's peak level from the realtime mic pipeline thread to the
/// GUI's periodic poll. Stored as raw bits in an `AtomicU32` so the store/
/// load are plain atomic ops with no lock, matching this project's realtime
/// rules (the writer side runs on the mic pipeline thread).
pub struct LevelMeter(AtomicU32);

impl LevelMeter {
    pub fn new() -> Self {
        Self(AtomicU32::new(0f32.to_bits()))
    }

    #[inline(always)]
    pub fn store(&self, value: f32) {
        self.0.store(value.to_bits(), Ordering::Relaxed);
    }

    #[inline(always)]
    pub fn load(&self) -> f32 {
        f32::from_bits(self.0.load(Ordering::Relaxed))
    }
}

impl Default for LevelMeter {
    fn default() -> Self {
        Self::new()
    }
}

/// A true no-op `Stage`, used in place of a real BVC (or, in principle, any
/// other stage) instance when it genuinely could not be loaded/constructed.
/// This exists so the pipeline never has to special-case "this stage might
/// be `None`" - every slot in the chain always holds a real `Box<dyn Stage>`,
/// constructed at startup, per the project's "always warm, never lazily
/// constructed" rule. Whether it does anything is controlled the normal way
/// (a `StageToggle`, forced off and disabled in the GUI when this is what
/// got constructed instead of the real thing) - never a silently-do-nothing
/// flag pretending to be a working feature.
pub struct NoOpStage(pub &'static str);

impl Stage for NoOpStage {
    fn name(&self) -> &'static str {
        self.0
    }

    #[inline(always)]
    fn process(&mut self, _frame: &mut [f32]) {}
}

/// Live device-switching interface for the four device-picker roles
/// ("Physical microphone", "Virtual mic target", "Virtual speaker source",
/// "Monitor output"). Implemented for real by the Windows engine
/// (`engine::LiveAudioSwitcher`, which actually stops/restarts the affected
/// WASAPI thread and reconnects it to a fresh ring buffer) and as a no-op
/// (`NoOpDeviceSwitcher`) everywhere else - defined here (portable) rather
/// than in `engine.rs` (Windows-only) so `gui.rs` can call it without
/// depending on `audio_io`/`engine` directly, matching this file's existing
/// "shared, portable types" role.
///
/// `device_id = None` means "system default" for the physical-mic role and
/// "not configured / disabled" for the other three - mirroring exactly how
/// `Settings`'s corresponding `Option<String>` fields are already
/// interpreted at next-launch time in `engine::start`.
pub trait LiveDeviceSwitcher: Send + Sync {
    fn set_physical_mic(&self, device_id: Option<String>);
    fn set_virtual_mic_target(&self, device_id: Option<String>);
    /// The speaker-outbound path's capture source - a second, independent
    /// virtual-cable instance's capture endpoint (what a call app's output
    /// gets redirected into, instead of real speakers) - see
    /// `spawn_speaker_pipeline`'s docs for the full topology.
    fn set_virtual_speaker_source(&self, device_id: Option<String>);
    /// The speaker-outbound path's render target - the user's real physical
    /// speakers/headphones, where the cleaned incoming-call audio actually
    /// ends up.
    fn set_real_speaker_output(&self, device_id: Option<String>);
    fn set_monitor_device(&self, device_id: Option<String>);

    /// Re-enumerates render/capture devices right now. Real hardware
    /// finding: devices are only enumerated once, at process startup
    /// (`main.rs`), so a virtual-cable app (e.g. VoiceMeter/VB-Cable)
    /// started *after* ClearNAI never appears in the device pickers and its
    /// role silently processes nothing (looks exactly like "NC/BVC isn't
    /// working") until the whole app is restarted. A GUI "Refresh devices"
    /// button calls this to pick up newly-available devices without a
    /// restart - selecting one still goes through the ordinary
    /// `set_*`/`Settings` live-switch path above.
    fn enumerate_devices(&self) -> DeviceList;
}

/// The non-Windows (and startup-failure) stand-in: every call is a no-op.
/// There is no real audio engine underneath in either case, so there is
/// nothing to switch.
pub struct NoOpDeviceSwitcher;

impl LiveDeviceSwitcher for NoOpDeviceSwitcher {
    fn set_physical_mic(&self, _device_id: Option<String>) {}
    fn set_virtual_mic_target(&self, _device_id: Option<String>) {}
    fn set_virtual_speaker_source(&self, _device_id: Option<String>) {}
    fn set_real_speaker_output(&self, _device_id: Option<String>) {}
    fn set_monitor_device(&self, _device_id: Option<String>) {}

    fn enumerate_devices(&self) -> DeviceList {
        DeviceList::empty()
    }
}

/// Live handles shared between the audio engine thread(s) and the GUI.
/// Constructed exactly once at startup and cloned (cheap: everything inside
/// is an `Arc`/`StageToggle`, itself an `Arc<AtomicBool>`) into the GUI's
/// `State`. Toggling a switch in the GUI flips the corresponding
/// `StageToggle` in place - the pipeline is never torn down or rebuilt.
#[derive(Clone)]
pub struct EngineHandles {
    /// Mic-path RNNoise, "RNNoise (between Mic and Virtual Mic)".
    pub mic_noise: StageToggle,
    /// Mic-path Voice Gate, "Voice Gate (suppress quiet background
    /// voices)" - see `dsp_core::VoiceGateStage`'s docs. Runs FIRST in the
    /// pipeline, before `AutoGainStage` - see that stage's docs for why
    /// the order matters.
    pub mic_voice_gate: StageToggle,
    /// Echo cancellation on the raw mic, using the call audio played by the speaker pipeline as the reference.
    pub echo_cancel: StageToggle,
    /// Mic-path BVC, "BVC (after Noise)". See `mic_bvc_available`.
    pub mic_bvc: StageToggle,
    /// Mic-path Studio stage, "Studio".
    pub mic_studio: StageToggle,
    /// Live preset selection for the mic-path Studio stage, encoded via
    /// `studio_preset_to_index`/`studio_preset_from_index`. See `engine.rs`'s
    /// mic pipeline loop for the once-per-frame load + swap-on-change
    /// pattern.
    pub mic_studio_preset: SharedPreset,
    /// Speaker-outbound path (Pipeline 2) - real, wired by `engine::start`'s
    /// `spawn_speaker_pipeline`. RNNoise for incoming-call audio, "Speaker
    /// Noise".
    pub speaker_noise: StageToggle,
    /// Speaker-path BVC, "Speaker BVC (after Noise)" - see
    /// `speaker_bvc_available`.
    pub speaker_bvc: StageToggle,
    pub speaker_studio: StageToggle,
    pub speaker_studio_preset: SharedPreset,

    /// Master mute for the WHOLE mic pipeline - unlike `mic_noise`/
    /// `mic_bvc`/`mic_studio` (which only gate whether their OWN stage's
    /// `.process()` runs, leaving the frame passing through untouched when
    /// off), this replaces the final frame with real silence before it's
    /// forwarded anywhere. Turning off every individual stage toggle still
    /// forwards raw, unprocessed audio - a real, confusing gap a user
    /// hit live ("everything is off, I can still hear audio") - this is
    /// the genuine "stop this pipeline" switch that gap needed. Defaults
    /// to enabled (`true`) and is NOT persisted - a runtime-only mute,
    /// always starting fresh/on at each launch.
    pub mic_master_enabled: StageToggle,
    /// Same as `mic_master_enabled`, for the speaker-outbound pipeline.
    pub speaker_master_enabled: StageToggle,

    /// "Monitor (hear processed mic locally)" - a diagnostic/testing-only
    /// path, wholly independent of the virtual-mic-target pipeline. When on
    /// (and a monitor output device has been configured - see
    /// `Settings::monitor_device_id`), the mic pipeline thread copies its
    /// fully-processed frame (after Noise/BVC/Studio) into a second
    /// render path so the user can hear the effect of those stages directly,
    /// without needing VB-Cable installed. When off, real silence is
    /// rendered to the monitor device instead of the processed signal (the
    /// render thread itself is never torn down/rebuilt on a toggle flip,
    /// matching every other `StageToggle` in this struct).
    pub monitor: StageToggle,

    /// Whether the mic-path `bvc_hush::HushBvcStage::try_load` succeeded at
    /// startup. When `false`, `mic_bvc` has been forced to `false` and the
    /// GUI must render its toggle disabled (never silently ignoring a
    /// user's attempt to turn it on). Independent of `speaker_bvc_available`
    /// - each path loads its own DLL session and can fail independently.
    pub mic_bvc_available: bool,
    /// Human-readable reason mic-path BVC is unavailable (e.g. "weya_nc.dll
    /// not found next to ..."), for a tooltip/label next to the disabled
    /// toggle. `None` when `mic_bvc_available` is `true`.
    pub mic_bvc_unavailable_reason: Option<String>,
    /// Whether the speaker-path Hush instance loaded successfully. See
    /// `mic_bvc_available` - its own independent flag, not shared with the
    /// mic path (a second, independent DLL session - see `bvc_hush`'s
    /// `speaker_dll_path`/`SPEAKER_DLL_FILE_NAME`).
    pub speaker_bvc_available: bool,
    /// Human-readable reason speaker-path Hush is unavailable. `None` when
    /// `speaker_bvc_available` is `true`.
    pub speaker_bvc_unavailable_reason: Option<String>,
    /// Whether the speaker-path DeepFilterNet instance loaded successfully -
    /// its own independent instance from the mic path's (see
    /// `noise_engine::load_bundled_deep_filter`).
    pub speaker_deep_filter_available: bool,
    /// Human-readable reason speaker-path DeepFilterNet is unavailable.
    pub speaker_deep_filter_unavailable_reason: Option<String>,

    /// Non-fatal startup warnings (e.g. "no virtual speaker source device
    /// selected; speaker outbound pipeline not started") surfaced once in
    /// the GUI.
    pub warnings: Arc<Vec<String>>,

    /// Live BVC backend selector - Off/Hush/DeepFilterNet/Hush-bpo_v1 (see `tse_engine::BvcBackend`).
    /// Wired to a real `tse_engine::BvcRouterStage` by the Windows engine
    /// (`engine::start`); on the portable/inert path this is a real,
    /// standalone `SharedPreset` that nothing reads, matching this file's
    /// existing "always a real, just possibly unread, handle" rule (see
    /// `device_switcher`'s docs).
    pub bvc_backend: SharedPreset,
    /// Live BVC backend selector for the speaker-outbound path - Off/Hush/
    /// DeepFilterNet only (see `tse_engine::SpeakerBvcBackend`).
    pub speaker_bvc_backend: SharedPreset,
    /// Whether `noise_engine::load_bundled_deep_filter` succeeded at
    /// startup - see that function's docs. Should essentially always be
    /// `true` (the model is embedded in the binary, not an external file
    /// that could be missing), but this is checked and surfaced honestly
    /// rather than assumed, matching `mic_bvc_available`.
    pub mic_deep_filter_available: bool,
    /// Human-readable reason DeepFilterNet is unavailable, for the GUI's
    /// disabled option label - mirrors `mic_bvc_unavailable_reason`.
    pub mic_deep_filter_unavailable_reason: Option<String>,
    /// Whether `bvc_hush::HushBvcStage::try_load_with_model` succeeded at
    /// startup for the `bpo_v1` checkpoint - see
    /// `tse_engine::BvcBackend::HushBpoV1`'s docs. A real fine-tune of
    /// Hush's own upstream checkpoint, added alongside (never replacing)
    /// the default `Hush` option - its own independent `HushBvcStage`
    /// instance and its own independent availability flag.
    pub mic_bpo_v1_available: bool,
    /// Mirrors `mic_bvc_unavailable_reason`.
    pub mic_bpo_v1_unavailable_reason: Option<String>,
    /// Honest, live status of the default mic Hush backend's watchdog -
    /// `None` when no real backend loaded at all (nothing to ever bypass;
    /// `mic_bvc_available` already covers that honestly), `Some` and
    /// checked via `WatchdogStatus::is_bypassed()` once a real
    /// `HushBvcStage` is actually running behind its `dsp_core::WatchdogStage`.
    /// Exists so the GUI can tell "Hush is on and running" apart from "Hush
    /// was auto-bypassed after repeated stalls and is now passing audio
    /// through unprocessed" - before this field existed, both looked
    /// identical from the GUI's side (the "on" toggle doesn't know the
    /// difference), which made a real listening test of Hush impossible to
    /// trust. Set post-construction by the real Windows engine
    /// (`engine::start`), same "construct inert, then wire the real thing"
    /// pattern as `device_switcher`/`bvc_backend` below.
    pub mic_hush_status: Option<WatchdogStatus>,
    /// Same as `mic_hush_status`, for the `Hush (bpo_v1)` backend's own,
    /// independent watchdog.
    pub mic_bpo_v1_status: Option<WatchdogStatus>,
    /// Accent conversion (voice choice + live status). Inert ("unavailable") until the real Windows engine
    /// overwrites it with the handle of the running worker - same "construct inert, then wire the real thing"
    /// pattern as `mic_hush_status`.
    pub accent: accent_onnx::pipeline::AccentHandle,
    /// Live transcript (lines + status). Inert until the real engine replaces it.
    pub transcript: asr_onnx::pipeline::TranscriptHandle,
    pub latency_log: Arc<LatencyLog>,
    /// Read by the real Windows engine's mic pipeline thread only
    /// (`engine.rs`); not yet surfaced in the GUI beyond the derived
    /// realtime-status indicator, so a non-Windows build (which never
    /// constructs `engine.rs`) sees this field as unread. That is expected,
    /// not a bug.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub counters: Arc<FrameCounters>,
    /// Speaker-outbound path's OWN, independent latency/drop telemetry -
    /// deliberately NOT shared with `latency_log`/`counters` above, so the
    /// GUI can show each pipeline's realtime status separately rather than
    /// one merged number that can't tell you which direction is unhealthy.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub speaker_latency_log: Arc<LatencyLog>,
    #[cfg_attr(not(windows), allow(dead_code))]
    pub speaker_counters: Arc<FrameCounters>,
    /// Peak of the processed frame (after the whole mic chain): the "After" meter.
    pub level: Arc<LevelMeter>,
    /// Peak of the raw mic frame (before any stage): the "Before" meter.
    pub input_level: Arc<LevelMeter>,

    /// Live device-switching handle for the four device pickers - see
    /// `LiveDeviceSwitcher`. Defaults to `NoOpDeviceSwitcher` in
    /// `new_from_settings`; the real Windows engine (`engine::start`)
    /// overwrites this with the real `Arc<engine::LiveAudioSwitcher>` after
    /// construction, once the actual WASAPI threads/ring buffers it needs to
    /// switch between exist.
    pub device_switcher: Arc<dyn LiveDeviceSwitcher>,
    /// `Some` only in the window process, where these handles are a mirror of the engine's (see `remote.rs`); `None`
    /// inside the engine itself.
    pub remote: Option<Arc<RemoteLive>>,
}

impl EngineHandles {
    /// Builds the toggle set from persisted `Settings`, honestly reflecting
    /// BVC availability per path (`mic_bvc_available`/`speaker_bvc_available`
    /// each force their own toggle off if that path's BVC could not be
    /// loaded, regardless of the persisted preference - the two paths load
    /// independent DLL sessions and can succeed/fail independently).
    #[allow(clippy::too_many_arguments)]
    pub fn new_from_settings(
        settings: &crate::settings::Settings,
        mic_bvc_available: bool,
        mic_bvc_unavailable_reason: Option<String>,
        speaker_bvc_available: bool,
        speaker_bvc_unavailable_reason: Option<String>,
        speaker_deep_filter_available: bool,
        speaker_deep_filter_unavailable_reason: Option<String>,
        warnings: Vec<String>,
        mic_deep_filter_available: bool,
        mic_deep_filter_unavailable_reason: Option<String>,
        mic_bpo_v1_available: bool,
        mic_bpo_v1_unavailable_reason: Option<String>,
    ) -> Self {
        Self {
            mic_noise: StageToggle::new(settings.mic_noise_on),
            mic_voice_gate: StageToggle::new(settings.mic_voice_gate_on),
            echo_cancel: StageToggle::new(settings.echo_cancel_on),
            mic_bvc: StageToggle::new(settings.mic_bvc_on && mic_bvc_available),
            mic_studio: StageToggle::new(settings.mic_studio_on),
            mic_studio_preset: SharedPreset::new(settings.mic_studio_preset),
            speaker_noise: StageToggle::new(settings.speaker_noise_on),
            speaker_bvc: StageToggle::new(settings.speaker_bvc_on && (speaker_bvc_available || speaker_deep_filter_available)),
            speaker_studio: StageToggle::new(settings.speaker_studio_on),
            speaker_studio_preset: SharedPreset::new(settings.speaker_studio_preset),
            mic_master_enabled: StageToggle::new(true),
            speaker_master_enabled: StageToggle::new(true),
            monitor: StageToggle::new(settings.monitor_enabled),
            mic_bvc_available,
            mic_bvc_unavailable_reason,
            speaker_bvc_available,
            speaker_bvc_unavailable_reason,
            speaker_deep_filter_available,
            speaker_deep_filter_unavailable_reason,
            bvc_backend: SharedPreset::new(settings.mic_bvc_backend),
            speaker_bvc_backend: SharedPreset::new(settings.speaker_bvc_backend),
            mic_deep_filter_available,
            mic_deep_filter_unavailable_reason,
            mic_bpo_v1_available,
            mic_bpo_v1_unavailable_reason,
            // Overwritten post-construction by the real Windows engine once
            // the real `WatchdogStage`s exist - see `mic_hush_status`'s docs.
            mic_hush_status: None,
            mic_bpo_v1_status: None,
            accent: accent_onnx::pipeline::AccentHandle::unavailable("accent engine not started"),
            transcript: asr_onnx::pipeline::TranscriptHandle::unavailable("transcript engine not started"),
            warnings: Arc::new(warnings),
            latency_log: Arc::new(LatencyLog::new(1024)),
            counters: Arc::new(FrameCounters::default()),
            speaker_latency_log: Arc::new(LatencyLog::new(1024)),
            speaker_counters: Arc::new(FrameCounters::default()),
            level: Arc::new(LevelMeter::new()),
            input_level: Arc::new(LevelMeter::new()),
            device_switcher: Arc::new(NoOpDeviceSwitcher),
            remote: None,
        }
    }

    /// Every on/off switch, in the fixed order used by `ipc::Status::toggles` / `ClientMsg::Toggle::key`.
    pub fn toggles(&self) -> [&StageToggle; crate::ipc::TOGGLE_COUNT] {
        [
            &self.mic_noise,
            &self.mic_voice_gate,
            &self.echo_cancel,
            &self.mic_bvc,
            &self.mic_studio,
            &self.speaker_noise,
            &self.speaker_bvc,
            &self.speaker_studio,
            &self.mic_master_enabled,
            &self.speaker_master_enabled,
            &self.monitor,
        ]
    }

    /// Every small-number selector, in the fixed order used by `ipc::Status::presets` / `ClientMsg::Preset::key`.
    pub fn presets(&self) -> [&SharedPreset; crate::ipc::PRESET_COUNT] {
        [&self.mic_studio_preset, &self.speaker_studio_preset, &self.bvc_backend, &self.speaker_bvc_backend]
    }
}

/// `FrameCounters` as a plain array (declaration order), for sending over the pipe.
pub fn counters_to_array(c: &FrameCounters) -> [u64; 8] {
    use std::sync::atomic::Ordering::Relaxed;
    [
        c.frames_processed.load(Relaxed),
        c.dropped_frames.load(Relaxed),
        c.deadline_misses.load(Relaxed),
        c.input_underruns.load(Relaxed),
        c.input_overruns.load(Relaxed),
        c.output_underruns.load(Relaxed),
        c.output_overruns.load(Relaxed),
        c.transcript_dropped.load(Relaxed),
    ]
}

/// Inverse of [`counters_to_array`] (the window's mirror counters).
pub fn counters_from_array(c: &FrameCounters, a: &[u64; 8]) {
    use std::sync::atomic::Ordering::Relaxed;
    c.frames_processed.store(a[0], Relaxed);
    c.dropped_frames.store(a[1], Relaxed);
    c.deadline_misses.store(a[2], Relaxed);
    c.input_underruns.store(a[3], Relaxed);
    c.input_overruns.store(a[4], Relaxed);
    c.output_underruns.store(a[5], Relaxed);
    c.output_overruns.store(a[6], Relaxed);
    c.transcript_dropped.store(a[7], Relaxed);
}

/// Live values the window gets from the engine that have no counterpart in the engine's own handles.
pub struct RemoteLive {
    pub mic_pct: std::sync::Mutex<dsp_core::Percentiles>,
    pub speaker_pct: std::sync::Mutex<dsp_core::Percentiles>,
    /// Engine CPU in tenths of a percent; `u32::MAX` = not measured yet.
    pub cpu_tenths: std::sync::atomic::AtomicU32,
    /// False while the pipe to the engine is down (the window shows a banner).
    pub connected: std::sync::atomic::AtomicBool,
    /// The engine said it is quitting.
    pub shutdown: std::sync::atomic::AtomicBool,
}

impl RemoteLive {
    pub fn new() -> Self {
        Self {
            mic_pct: Default::default(),
            speaker_pct: Default::default(),
            cpu_tenths: std::sync::atomic::AtomicU32::new(u32::MAX),
            connected: std::sync::atomic::AtomicBool::new(true),
            shutdown: std::sync::atomic::AtomicBool::new(false),
        }
    }
}

/// One enumerated WASAPI endpoint, as shown in a GUI device picker.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct DeviceOption {
    pub id: String,
    pub label: String,
}

impl std::fmt::Display for DeviceOption {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.label)
    }
}

/// Render/capture device lists as shown in the GUI's two device pickers.
/// Empty on any platform where enumeration isn't available (non-Windows, or
/// a real enumeration failure) - the GUI must degrade to "no devices
/// listed", never crash.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct DeviceList {
    pub render: Vec<DeviceOption>,
    pub capture: Vec<DeviceOption>,
}

impl DeviceList {
    #[cfg_attr(windows, allow(dead_code))]
    pub fn empty() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only convenience: builds `EngineHandles` with the remaining
    /// optional backends reported unavailable - the ordinary case for every
    /// test in this module.
    fn handles_with(
        settings: &crate::settings::Settings,
        mic_bvc_available: bool,
        mic_bvc_unavailable_reason: Option<String>,
        speaker_bvc_available: bool,
        speaker_bvc_unavailable_reason: Option<String>,
        warnings: Vec<String>,
    ) -> EngineHandles {
        EngineHandles::new_from_settings(
            settings,
            mic_bvc_available,
            mic_bvc_unavailable_reason,
            speaker_bvc_available,
            speaker_bvc_unavailable_reason,
            // DeepFilterNet availability isn't this helper's concern - keep
            // it unavailable so `speaker_bvc_available`'s effect on the
            // combined `speaker_bvc` toggle (see `new_from_settings`'s
            // `speaker_bvc_available || speaker_deep_filter_available`) is
            // exactly what each test below explicitly passes, not silently
            // overridden by a hardcoded `true`.
            false,
            None,
            warnings,
            true,
            None,
            // bpo_v1 availability isn't this helper's concern either - kept
            // unavailable for the same reason as DeepFilterNet above.
            false,
            Some("no bpo_v1 model bundled".to_string()),
        )
    }

    #[test]
    fn level_meter_round_trips_arbitrary_f32_via_bit_pattern() {
        let meter = LevelMeter::new();
        for v in [0.0f32, 1.0, -1.0, 0.5, 0.999_999, -0.000_1, f32::MIN_POSITIVE] {
            meter.store(v);
            assert_eq!(meter.load(), v);
        }
    }

    #[test]
    fn noop_stage_never_mutates_the_frame() {
        let mut stage = NoOpStage("bvc unavailable");
        let mut frame = [0.1, -0.2, 0.3, 0.0];
        let original = frame;
        stage.process(&mut frame);
        assert_eq!(frame, original);
        assert_eq!(stage.name(), "bvc unavailable");
    }

    #[test]
    fn bvc_unavailable_forces_toggle_off_regardless_of_saved_preference() {
        let mut settings = crate::settings::Settings::default();
        settings.mic_bvc_on = true; // user's saved preference is "on"
        settings.speaker_bvc_on = true;
        let handles = handles_with(
            &settings,
            false,
            Some("weya_nc.dll not found".into()),
            false,
            Some("weya_nc.dll not found".into()),
            Vec::new(),
        );
        assert!(!handles.mic_bvc_available);
        assert!(!handles.speaker_bvc_available);
        assert!(!handles.mic_bvc.is_on(), "BVC must be forced off when unavailable, never silently left on");
        assert!(!handles.speaker_bvc.is_on(), "BVC must be forced off when unavailable, never silently left on");
    }

    #[test]
    fn bvc_available_honors_saved_preference() {
        let mut settings = crate::settings::Settings::default();
        settings.mic_bvc_on = false;
        let handles = handles_with(&settings, true, None, true, None, Vec::new());
        assert!(handles.mic_bvc_available);
        assert!(handles.speaker_bvc_available);
        assert!(!handles.mic_bvc.is_on());
    }

    #[test]
    fn mic_and_speaker_bvc_availability_are_independent() {
        // The two paths load independent DLL sessions and can succeed/fail
        // independently - one path's failure must never force the other's
        // toggle off.
        let mut settings = crate::settings::Settings::default();
        settings.mic_bvc_on = true;
        settings.speaker_bvc_on = true;
        let handles = handles_with(
            &settings,
            true,
            None,
            false,
            Some("session limit reached".into()),
            Vec::new(),
        );
        assert!(handles.mic_bvc.is_on(), "mic BVC must stay on when only the speaker path failed to load");
        assert!(!handles.speaker_bvc.is_on(), "speaker BVC must be forced off when its own load failed");
    }

    #[test]
    fn device_list_empty_has_no_devices() {
        let list = DeviceList::empty();
        assert!(list.render.is_empty());
        assert!(list.capture.is_empty());
    }

    #[test]
    fn no_op_device_switcher_accepts_every_call_without_panicking() {
        // The non-Windows (and engine-start-failure) stand-in for
        // `LiveDeviceSwitcher` - there is no real engine underneath, so
        // every call must be inert, never panic, regardless of Some/None.
        let switcher = NoOpDeviceSwitcher;
        switcher.set_physical_mic(Some("device-1".to_string()));
        switcher.set_physical_mic(None);
        switcher.set_virtual_mic_target(Some("device-2".to_string()));
        switcher.set_virtual_mic_target(None);
        switcher.set_virtual_speaker_source(Some("device-3".to_string()));
        switcher.set_virtual_speaker_source(None);
        switcher.set_monitor_device(Some("device-4".to_string()));
        switcher.set_monitor_device(None);
    }

    #[test]
    fn engine_handles_default_to_a_no_op_device_switcher() {
        // `new_from_settings` is used both for the real Windows engine
        // (immediately overwritten with the real switcher afterward - see
        // `engine::start`) and for the non-Windows/engine-failure stubs
        // (never overwritten) - it must default to a real, inert
        // implementation, never a dangling/unset handle.
        let settings = crate::settings::Settings::default();
        let handles = handles_with(&settings, true, None, true, None, Vec::new());
        // Must not panic - proves a concrete, callable `LiveDeviceSwitcher`
        // is always present from construction.
        handles.device_switcher.set_physical_mic(None);
    }

    #[test]
    fn engine_handles_bvc_backend_selector_defaults_from_settings() {
        let mut settings = crate::settings::Settings::default();
        settings.mic_bvc_backend = tse_engine::BvcBackend::DeepFilter.to_index();
        let handles = handles_with(&settings, true, None, true, None, Vec::new());
        assert_eq!(handles.bvc_backend.load(), tse_engine::BvcBackend::DeepFilter.to_index());
    }
}
