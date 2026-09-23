//! Types shared between the audio engine (real on Windows, inert stub
//! elsewhere - see `engine.rs`) and the GUI (`gui.rs`). Nothing in this file
//! touches an OS audio API, so it compiles and its tests run natively on
//! Linux as well as when cross-checked for Windows.

use dsp_core::{FrameCounters, LatencyLog, SharedPreset, Stage, StageToggle};
use speaker_profile::TargetSpeakerStore;
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;

/// Builds a wholly independent, real (never-shared) enrollment subsystem
/// triple: the realtime-thread tap, the target-speaker store, and the
/// enrollment session that owns the tap's worker-thread half. Called once
/// by the real Windows engine (`engine::start`, whose caller keeps the
/// returned `EnrollmentTap` and moves it into the mic pipeline thread) and,
/// independently, by every portable/inert `EngineHandles` construction site
/// (`main.rs`'s non-Windows bootstrap and its engine-start-failure
/// fallback) - each gets its own real, functioning-but-never-fed tap/session
/// pair, exactly like `NoOpDeviceSwitcher` is its own real (just inert)
/// implementation rather than a shared placeholder.
///
/// The speaker encoder is the real, bundled
/// `speaker_encoder::WeSpeakerEncoder` (a WeSpeaker ResNet34 speaker-
/// verification model - see that crate's docs for its verified provenance,
/// including why CAM++/ECAPA-TDNN were considered and rejected first) if it
/// loads; if construction fails for any reason (should not happen in
/// practice - the model is embedded at compile time - but never assumed
/// infallible), this falls back to
/// `speaker_profile::UnavailableSpeakerEncoder`, mirroring
/// `bvc_hush::HushBvcStage::try_load`'s "always construct something real,
/// degrade honestly on failure" pattern. Model construction (~25MB of ONNX
/// parsing/optimization) happens here, once, at engine startup - never
/// per-enrollment-attempt and never on the realtime audio thread.
/// Second, TSE-specific target-speaker store, holding the 192-dim
/// ECAPA_TDNN_GLOB_c512 embedding `tse_bsrnn::tse_speaker_encoder::TseSpeakerEncoder`
/// produces - kept independent from the primary 256-dim
/// `TargetSpeakerStore` above (same enrollment recording, two different
/// embedding spaces - see `docs/TSE_INTEGRATION_PLAN.md` Step 3 and
/// `tse_bsrnn::tse_speaker_encoder`'s module docs for why this is a
/// second store rather than a shared/bridged one).
pub fn build_enrollment_subsystem() -> (
    enrollment::EnrollmentTap,
    TargetSpeakerStore,
    enrollment::EnrollmentSession,
    Option<String>,
    TargetSpeakerStore,
    TargetSpeakerStore,
) {
    let store = TargetSpeakerStore::new();
    let tse_store = TargetSpeakerStore::new();
    let vanta_store = TargetSpeakerStore::new();
    let (tap, consumer, active, _dropped_samples) = enrollment::new_pair(enrollment::MAX_RECORDING_SECS);
    let (encoder, warning): (Arc<dyn speaker_profile::SpeakerEncoder>, Option<String>) =
        match speaker_encoder::WeSpeakerEncoder::new() {
            Ok(real) => (Arc::new(real), None),
            Err(e) => (
                Arc::new(speaker_profile::UnavailableSpeakerEncoder),
                Some(format!(
                    "Speaker encoder (WeSpeaker ResNet34) failed to load ({e:#}); voice reference enrollment is unavailable."
                )),
            ),
        };
    let tse_encoder: Arc<dyn speaker_profile::SpeakerEncoder> = match tse_bsrnn::tse_speaker_encoder::TseSpeakerEncoder::new() {
        Ok(real) => Arc::new(real),
        // No separate warning pushed for this one: the primary 256-dim
        // encoder above already covers "enrollment is unavailable" in the
        // GUI; this second encoder failing only means the TSE backend
        // itself won't have a target speaker to condition on (already
        // covered by `tse_available`/`tse_unavailable_reason` in
        // `engine.rs`), not a second, redundant top-level warning.
        Err(_) => Arc::new(speaker_profile::UnavailableSpeakerEncoder),
    };
    // Vanta's OWN 192-dim ECAPA space (SpeechBrain's spkrec-ecapa-voxceleb
    // checkpoint) - same dimension as `tse_encoder` above but a DIFFERENT
    // trained model, not interchangeable (see `bvc_vanta::ecapa`'s docs).
    // `onnxruntime.dll` is embedded in the binary and self-extracted into
    // `setup::app_data_dir()` by `ensure_bvc_assets_extracted` (called
    // before this function, in `main.rs`'s `bootstrap`) - same directory
    // `weya_nc.dll` extracts into, NOT the exe's own directory. If
    // extraction hasn't run yet or failed, this degrades honestly to
    // `UnavailableSpeakerEncoder`, same as the TSE encoder above.
    let vanta_encoder: Arc<dyn speaker_profile::SpeakerEncoder> =
        match bvc_vanta::ecapa::VantaSpeakerEncoder::try_load(&crate::setup::app_data_dir()) {
            Ok(real) => Arc::new(real),
            Err(_) => Arc::new(speaker_profile::UnavailableSpeakerEncoder),
        };
    let session = enrollment::EnrollmentSession::new(consumer, active, encoder, store.clone())
        .with_secondary_encoder(tse_encoder, tse_store.clone())
        .with_secondary_encoder(vanta_encoder, vanta_store.clone());
    (tap, store, session, warning, tse_store, vanta_store)
}

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

/// Real, live state of a `play_recorded_audio` playback - not just a
/// number, so the GUI can show progress alongside Pause/Resume/Stop.
/// Portable (used by `gui.rs` on every platform) even though only the real
/// Windows implementation (`engine::play_recorded_audio`) ever produces
/// anything other than `Finished`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaybackStatus {
    Playing,
    Paused,
    /// Reached the end of the clip on its own, was stopped early, or (on a
    /// non-Windows build, or if the playback device failed to open) never
    /// really started - the GUI treats all of these the same way (remove
    /// the transport controls).
    Finished,
    Stopped,
}

/// Handle to a running (or finished) `play_recorded_audio` playback - lets
/// the GUI request pause/resume/stop and poll real progress, the same
/// "cheap `Arc`-backed handle, real background thread underneath" shape as
/// `enrollment::EnrollmentSession` itself. Fields are `pub(crate)` so only
/// `engine.rs`'s real Windows implementation constructs one with live
/// flags; everywhere else (including this file's own non-Windows stub)
/// only ever hands out an already-`Finished` handle.
#[derive(Clone)]
pub struct PlaybackHandle {
    pub(crate) stop_flag: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) paused_flag: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) finished_flag: Arc<std::sync::atomic::AtomicBool>,
    pub(crate) played_samples: Arc<std::sync::atomic::AtomicUsize>,
    pub(crate) total_samples: usize,
}

impl PlaybackHandle {
    /// An already-finished handle - for the non-Windows stub, and for
    /// `engine::play_recorded_audio`'s own empty-input/device-open-failure
    /// cases.
    pub(crate) fn finished() -> Self {
        Self {
            stop_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            paused_flag: Arc::new(std::sync::atomic::AtomicBool::new(false)),
            finished_flag: Arc::new(std::sync::atomic::AtomicBool::new(true)),
            played_samples: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            total_samples: 0,
        }
    }

    pub fn pause(&self) {
        self.paused_flag.store(true, Ordering::Relaxed);
    }

    pub fn resume(&self) {
        self.paused_flag.store(false, Ordering::Relaxed);
    }

    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Relaxed);
    }

    pub fn status(&self) -> PlaybackStatus {
        if self.finished_flag.load(Ordering::Relaxed) {
            if self.stop_flag.load(Ordering::Relaxed) {
                PlaybackStatus::Stopped
            } else {
                PlaybackStatus::Finished
            }
        } else if self.paused_flag.load(Ordering::Relaxed) {
            PlaybackStatus::Paused
        } else {
            PlaybackStatus::Playing
        }
    }

    /// `0.0..=1.0` - real progress through the clip (samples actually fed
    /// to the render device so far), for a progress bar. Not exact against
    /// what's audibly *finished* playing (WASAPI's own device buffer holds
    /// a small amount ahead), close enough for a UI indicator.
    pub fn progress(&self) -> f32 {
        if self.total_samples == 0 {
            return 1.0;
        }
        (self.played_samples.load(Ordering::Relaxed) as f32 / self.total_samples as f32).min(1.0)
    }
}

/// Portable wrapper around `engine::play_recorded_audio` (real on Windows,
/// an already-`Finished` handle plus a logged message elsewhere) - lets
/// `gui.rs` call this unconditionally without its own `#[cfg(windows)]`
/// split, the same reasoning `EngineHandles`/`build_enrollment_subsystem`
/// already follow for every other "real on Windows, inert stub elsewhere"
/// capability in this file.
#[cfg(windows)]
pub fn play_recorded_audio(samples: Vec<f32>, device_id: Option<String>) -> PlaybackHandle {
    crate::engine::play_recorded_audio(samples, device_id)
}

#[cfg(not(windows))]
pub fn play_recorded_audio(_samples: Vec<f32>, _device_id: Option<String>) -> PlaybackHandle {
    crate::log_error!("[clearnairt] recording playback is only implemented on Windows");
    PlaybackHandle::finished()
}

/// Runs `samples` (48kHz, matching `dsp_core::SAMPLE_RATE_HZ`) through a
/// freshly-loaded real TSE backend OFFLINE - not the live real-time mic
/// pipeline. `tse_bsrnn::backend::RealBsrnnTseBackend`'s own docs report an
/// honest RTF far above 1x, which the live pipeline's `dsp_core::WatchdogStage`
/// only bounds the *damage* of (timeouts/frame drops), it does not make the
/// backend fast - so live-pipeline TSE audio is expected to glitch. This
/// function has no per-frame deadline at all: it just runs the whole
/// recording through, however long that takes, and returns the finished,
/// correct result for playback - the only way to actually hear full-quality
/// TSE output today. Loads its own, separate `RealBsrnnTseBackend` (re-
/// parsing the ~100MB weight blob) rather than reaching into the live
/// pipeline's already-loaded one, since that one is owned by the mic
/// pipeline thread and not safely reachable from here - an explicit,
/// user-initiated test action can afford the extra ~1-2s load time.
pub fn run_tse_offline(mut samples: Vec<f32>, profile: Arc<speaker_profile::TargetSpeakerProfile>) -> Result<Vec<f32>, String> {
    use tse_engine::TseBackend;

    // Real finding: uncapped, this saturates every logical core for up to a
    // minute and measurably degrades live mic audio (added latency, frame
    // drops) via plain CPU contention - the live pipeline's MMCSS-elevated
    // threads don't fully protect it once literally every core is pegged.
    // Capping at half the cores (min 1) leaves real headroom; this is a
    // thread-local hint (see `tse_bsrnn::separator::set_max_parallelism_hint`'s
    // docs), so it only affects this background thread's own work, never
    // the live pipeline's.
    let half_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).max(2) / 2;
    tse_bsrnn::separator::set_max_parallelism_hint(Some(half_cores.max(1)));

    let mut backend = tse_bsrnn::backend::RealBsrnnTseBackend::try_load()?;
    backend.set_target_speaker(Some(profile));

    // Pad with trailing silence so the resampler/STFT's own internal group
    // delay gets fully flushed - otherwise the last real chunk of audio
    // would still be sitting in internal buffers when input runs out.
    const TAIL_PAD_SAMPLES: usize = 24_000; // 500ms @ 48kHz
    samples.extend(std::iter::repeat(0.0f32).take(TAIL_PAD_SAMPLES));

    const CHUNK: usize = 480; // dsp_core::FRAME_SAMPLES
    let mut output = Vec::with_capacity(samples.len());
    let mut i = 0;
    while i < samples.len() {
        let end = (i + CHUNK).min(samples.len());
        let real_len = end - i;
        let mut frame = vec![0.0f32; CHUNK];
        frame[..real_len].copy_from_slice(&samples[i..end]);
        // Errors here are just the real warm-up gap (frame is left
        // untouched, per `TseBackend::process_frame`'s contract) - the
        // same honest behavior the live pipeline already tolerates.
        let _ = backend.process_frame(&mut frame);
        output.extend_from_slice(&frame[..real_len]);
        i = end;
    }
    Ok(output)
}

#[derive(Default)]
struct TseOfflineTestInner {
    in_progress: bool,
    error: Option<String>,
    playback: Option<PlaybackHandle>,
}

/// Tracks one background `run_tse_offline` run, GUI-pollable the same
/// "background thread writes, `Message::Tick` reads" way
/// `enrollment::EnrollmentSession` already works - `update_main` cannot
/// mutate `MainState` from a spawned thread directly, so the thread stores
/// its result here instead of returning it.
#[derive(Clone, Default)]
pub struct TseOfflineTest(Arc<std::sync::Mutex<TseOfflineTestInner>>);

impl TseOfflineTest {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn is_in_progress(&self) -> bool {
        self.0.lock().unwrap().in_progress
    }

    /// Takes (clears) the error from the most recent run, if any - call
    /// once per `Tick` so the same error isn't logged repeatedly.
    pub fn take_error(&self) -> Option<String> {
        self.0.lock().unwrap().error.take()
    }

    /// Takes (clears) the `PlaybackHandle` for a just-finished run's
    /// already-started playback, if ready - call once per `Tick` so
    /// `MainState.playback` picks it up exactly once.
    pub fn take_ready_playback(&self) -> Option<PlaybackHandle> {
        self.0.lock().unwrap().playback.take()
    }

    /// Starts a background run; a no-op if one is already in progress
    /// (matches `enrollment::EnrollmentSession::start_recording`'s "ignore
    /// a redundant click" behavior rather than queuing or racing runs).
    /// Starts playback itself (via `play_recorded_audio`) as soon as
    /// processing finishes, so the GUI only needs to pick up the resulting
    /// handle on its next `Tick`, not orchestrate a second step.
    pub fn start(&self, samples: Vec<f32>, profile: Arc<speaker_profile::TargetSpeakerProfile>, device_id: Option<String>) {
        {
            let mut inner = self.0.lock().unwrap();
            if inner.in_progress {
                return;
            }
            inner.in_progress = true;
            inner.error = None;
        }

        let this = self.clone();
        std::thread::spawn(move || {
            #[cfg(windows)]
            audio_io::mmcss::lower_current_thread_priority();
            let result = run_tse_offline(samples, profile);
            match result {
                Ok(processed) => {
                    let handle = play_recorded_audio(processed, device_id);
                    let mut inner = this.0.lock().unwrap();
                    inner.in_progress = false;
                    inner.playback = Some(handle);
                }
                Err(e) => {
                    let mut inner = this.0.lock().unwrap();
                    inner.in_progress = false;
                    inner.error = Some(e);
                }
            }
        });
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
    /// Speaker-outbound (Pipeline 2) has been removed from the GUI and the
    /// real engine for this session (see `engine.rs`) - no caller invokes
    /// this any more, but the method is kept on the trait (rather than
    /// deleted) so re-enabling that pipeline later doesn't require touching
    /// this trait's shape again. `#[allow(dead_code)]` is the honest way to
    /// say "unused right now, not a mistake."
    #[allow(dead_code)]
    fn set_virtual_speaker_source(&self, device_id: Option<String>);
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
    /// Mic-path BVC, "BVC (after Noise)". See `mic_bvc_available`.
    pub mic_bvc: StageToggle,
    /// Mic-path Studio stage, "Studio".
    pub mic_studio: StageToggle,
    /// Live preset selection for the mic-path Studio stage, encoded via
    /// `studio_preset_to_index`/`studio_preset_from_index`. See `engine.rs`'s
    /// mic pipeline loop for the once-per-frame load + swap-on-change
    /// pattern.
    pub mic_studio_preset: SharedPreset,
    /// Speaker outbound path (Pipeline 2), removed from the GUI and the real
    /// engine for this session - see `engine.rs`. Kept on this struct
    /// (rather than deleted) so re-enabling that pipeline later doesn't
    /// require reshaping `EngineHandles` again; nothing reads these four
    /// fields right now, which is expected, not a bug.
    #[allow(dead_code)]
    pub speaker_noise: StageToggle,
    #[allow(dead_code)]
    pub speaker_bvc: StageToggle,
    #[allow(dead_code)]
    pub speaker_studio: StageToggle,
    #[allow(dead_code)]
    pub speaker_studio_preset: SharedPreset,

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
    /// Whether the speaker-path `bvc_hush::HushBvcStage::try_load` succeeded
    /// at startup. See `mic_bvc_available` - this is its own independent
    /// flag, not shared with the mic path. Unread now that Pipeline 2 is
    /// removed from the GUI/engine - see `speaker_noise`'s docs.
    #[allow(dead_code)]
    pub speaker_bvc_available: bool,
    /// Human-readable reason speaker-path BVC is unavailable. `None` when
    /// `speaker_bvc_available` is `true`.
    #[allow(dead_code)]
    pub speaker_bvc_unavailable_reason: Option<String>,

    /// Non-fatal startup warnings (e.g. "no virtual speaker source device
    /// selected; speaker outbound pipeline not started") surfaced once in
    /// the GUI.
    pub warnings: Arc<Vec<String>>,

    /// Live BVC backend selector - Off/Hush/TSE (see `tse_engine::BvcBackend`).
    /// Wired to a real `tse_engine::BvcRouterStage` by the Windows engine
    /// (`engine::start`); on the portable/inert path this is a real,
    /// standalone `SharedPreset` that nothing reads, matching this file's
    /// existing "always a real, just possibly unread, handle" rule (see
    /// `device_switcher`'s docs).
    pub bvc_backend: SharedPreset,
    /// Whether a real TSE backend is available. Always `false` today - see
    /// `tse_engine::UnavailableTse`'s docs on why no real streaming
    /// target-speaker-extraction model is integrated yet. Kept as an
    /// explicit flag (rather than hardcoding the GUI's TSE option to
    /// disabled) so wiring in a real backend later only means flipping this
    /// at construction time, exactly like `mic_bvc_available` already works
    /// for Hush.
    pub tse_available: bool,
    /// Human-readable reason TSE is unavailable, for the GUI's disabled
    /// option label - mirrors `mic_bvc_unavailable_reason`.
    pub tse_unavailable_reason: Option<String>,
    /// Whether `noise_engine::load_bundled_deep_filter` succeeded at
    /// startup - see that function's docs. Should essentially always be
    /// `true` (the model is embedded in the binary, not an external file
    /// that could be missing), but this is checked and surfaced honestly
    /// rather than assumed, matching `mic_bvc_available`/`tse_available`.
    pub mic_deep_filter_available: bool,
    /// Human-readable reason DeepFilterNet is unavailable, for the GUI's
    /// disabled option label - mirrors `mic_bvc_unavailable_reason`.
    pub mic_deep_filter_unavailable_reason: Option<String>,
    /// Whether `bvc_vanta::VantaBackend::try_load` succeeded at startup -
    /// see `tse_engine::BvcBackend::Vanta`'s docs. Real separator +
    /// encoder/decoder weights now exist and load, but this remains an
    /// EXPERIMENTAL, honestly weak model (real measured eval: +1.07dB
    /// average improvement, -7.10dB worst case, PESQ 0.84 - see
    /// `bvc_vanta`'s crate docs) - `true` here means "constructs and runs",
    /// not "production quality".
    pub mic_vanta_available: bool,
    /// Mirrors `mic_bvc_unavailable_reason`.
    pub mic_vanta_unavailable_reason: Option<String>,
    /// The enrolled target speaker (or none), read by the realtime TSE
    /// stage via `tse_engine::TseStageAdapter` and written by
    /// `enrollment`'s worker thread - never touched directly by the GUI or
    /// this struct itself, only shared so the GUI can ask
    /// `target_speaker_store.has_profile()` for its "Enrolled" indicator.
    pub target_speaker_store: TargetSpeakerStore,
    /// Second, TSE-specific target-speaker store (192-dim ECAPA), populated
    /// by the same enrollment recording alongside `target_speaker_store` -
    /// see `build_enrollment_subsystem`'s docs. Used by the GUI's
    /// "Test with TSE" offline-processing button (`tse_offline_test`) to
    /// fetch the embedding `run_tse_offline` needs.
    pub tse_speaker_store: TargetSpeakerStore,
    /// Third, Vanta-specific target-speaker store (192-dim SpeechBrain
    /// ECAPA - a DIFFERENT embedding space from `tse_speaker_store` despite
    /// the same dimension, see `bvc_vanta::ecapa`'s docs), populated by the
    /// same enrollment recording alongside the other two stores.
    pub vanta_speaker_store: TargetSpeakerStore,
    /// One-time speaker-reference-enrollment control path - see
    /// `enrollment`'s crate docs. Cheap to clone (every field inside is an
    /// `Arc`-backed handle), same as every other field on this struct.
    pub enrollment: enrollment::EnrollmentSession,
    /// Background-run tracker for the GUI's "Test with TSE" button - see
    /// `TseOfflineTest`'s docs.
    pub tse_offline_test: TseOfflineTest,
    pub latency_log: Arc<LatencyLog>,
    /// Read by the real Windows engine's mic pipeline thread only
    /// (`engine.rs`); not yet surfaced in the GUI beyond the derived
    /// realtime-status indicator, so a non-Windows build (which never
    /// constructs `engine.rs`) sees this field as unread. That is expected,
    /// not a bug.
    #[cfg_attr(not(windows), allow(dead_code))]
    pub counters: Arc<FrameCounters>,
    pub level: Arc<LevelMeter>,

    /// Live device-switching handle for the four device pickers - see
    /// `LiveDeviceSwitcher`. Defaults to `NoOpDeviceSwitcher` in
    /// `new_from_settings`; the real Windows engine (`engine::start`)
    /// overwrites this with the real `Arc<engine::LiveAudioSwitcher>` after
    /// construction, once the actual WASAPI threads/ring buffers it needs to
    /// switch between exist.
    pub device_switcher: Arc<dyn LiveDeviceSwitcher>,
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
        warnings: Vec<String>,
        tse_available: bool,
        tse_unavailable_reason: Option<String>,
        mic_deep_filter_available: bool,
        mic_deep_filter_unavailable_reason: Option<String>,
        mic_vanta_available: bool,
        mic_vanta_unavailable_reason: Option<String>,
        target_speaker_store: TargetSpeakerStore,
        tse_speaker_store: TargetSpeakerStore,
        vanta_speaker_store: TargetSpeakerStore,
        enrollment: enrollment::EnrollmentSession,
    ) -> Self {
        Self {
            mic_noise: StageToggle::new(settings.mic_noise_on),
            mic_voice_gate: StageToggle::new(settings.mic_voice_gate_on),
            mic_bvc: StageToggle::new(settings.mic_bvc_on && mic_bvc_available),
            mic_studio: StageToggle::new(settings.mic_studio_on),
            mic_studio_preset: SharedPreset::new(settings.mic_studio_preset),
            speaker_noise: StageToggle::new(settings.speaker_noise_on),
            speaker_bvc: StageToggle::new(settings.speaker_bvc_on && speaker_bvc_available),
            speaker_studio: StageToggle::new(settings.speaker_studio_on),
            speaker_studio_preset: SharedPreset::new(settings.speaker_studio_preset),
            monitor: StageToggle::new(settings.monitor_enabled),
            mic_bvc_available,
            mic_bvc_unavailable_reason,
            speaker_bvc_available,
            speaker_bvc_unavailable_reason,
            bvc_backend: SharedPreset::new(settings.mic_bvc_backend),
            tse_available,
            tse_unavailable_reason,
            mic_deep_filter_available,
            mic_deep_filter_unavailable_reason,
            mic_vanta_available,
            mic_vanta_unavailable_reason,
            target_speaker_store,
            tse_speaker_store,
            vanta_speaker_store,
            enrollment,
            tse_offline_test: TseOfflineTest::new(),
            warnings: Arc::new(warnings),
            latency_log: Arc::new(LatencyLog::new(1024)),
            counters: Arc::new(FrameCounters::default()),
            level: Arc::new(LevelMeter::new()),
            device_switcher: Arc::new(NoOpDeviceSwitcher),
        }
    }
}

/// One enumerated WASAPI endpoint, as shown in a GUI device picker.
#[derive(Debug, Clone, PartialEq, Eq)]
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
#[derive(Debug, Clone, Default)]
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

    /// Test-only convenience: builds `EngineHandles` with a fresh, real
    /// (independent) enrollment subsystem and TSE reported unavailable -
    /// the ordinary case for every test in this module that isn't itself
    /// about enrollment/TSE availability.
    fn handles_with(
        settings: &crate::settings::Settings,
        mic_bvc_available: bool,
        mic_bvc_unavailable_reason: Option<String>,
        speaker_bvc_available: bool,
        speaker_bvc_unavailable_reason: Option<String>,
        warnings: Vec<String>,
    ) -> EngineHandles {
        let (_tap, store, session, _warning, tse_store, vanta_store) = build_enrollment_subsystem();
        EngineHandles::new_from_settings(
            settings,
            mic_bvc_available,
            mic_bvc_unavailable_reason,
            speaker_bvc_available,
            speaker_bvc_unavailable_reason,
            warnings,
            false,
            Some("no TSE model bundled".to_string()),
            true,
            None,
            false,
            Some("no Vanta model bundled".to_string()),
            store,
            tse_store,
            vanta_store,
            session,
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
    fn build_enrollment_subsystem_starts_with_no_target_speaker_enrolled() {
        let (_tap, store, session, _warning, _tse_store, _vanta_store) = build_enrollment_subsystem();
        assert!(!store.has_profile());
        assert!(!session.has_target_speaker());
        assert_eq!(session.state(), enrollment::EnrollmentState::Idle);
    }

    #[test]
    fn engine_handles_carries_the_tse_availability_flag_through() {
        let settings = crate::settings::Settings::default();
        let (_tap, store, session, _warning, tse_store, vanta_store) = build_enrollment_subsystem();
        let handles = EngineHandles::new_from_settings(
            &settings,
            true,
            None,
            true,
            None,
            Vec::new(),
            false,
            Some("no TSE model bundled".to_string()),
            true,
            None,
            false,
            Some("no Vanta model bundled".to_string()),
            store,
            tse_store,
            vanta_store,
            session,
        );
        assert!(!handles.tse_available, "no real TSE backend is integrated yet - see tse_engine::UnavailableTse");
        assert_eq!(handles.tse_unavailable_reason.as_deref(), Some("no TSE model bundled"));
        assert!(!handles.target_speaker_store.has_profile());
    }

    #[test]
    fn engine_handles_bvc_backend_selector_defaults_from_settings() {
        let mut settings = crate::settings::Settings::default();
        settings.mic_bvc_backend = tse_engine::BvcBackend::Tse.to_index();
        let handles = handles_with(&settings, true, None, true, None, Vec::new());
        assert_eq!(handles.bvc_backend.load(), tse_engine::BvcBackend::Tse.to_index());
    }
}
