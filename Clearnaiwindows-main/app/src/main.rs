//! `clearnairt` - the single ClearNAI Windows binary: GUI and real-time
//! audio engine in one process (no separate daemon), per the project's hard
//! architecture requirement.
//!
//! See `docs/VIRTUAL_DEVICES.md` for a real, unresolved-by-this-codebase
//! user setup requirement (two separate virtual audio cable installs), and
//! `engine.rs` / `gui.rs` for the pipeline and GUI documentation
//! respectively.
//!
//! # Portable vs Windows-only
//!
//! `settings.rs`, `status_ui.rs`, `shared.rs`, and `gui.rs` are ordinary
//! portable Rust (plus `iced`, which itself supports Linux) and build/run
//! their unit tests on any platform. `engine.rs` is `#![cfg(windows)]` in
//! its entirety - it is the only module that touches `audio_io::capture`/
//! `render`/`loopback`/`devices`'s WASAPI-backed functions, which
//! themselves only exist on Windows. This file branches on `cfg(windows)`
//! to call into the real engine or, elsewhere, construct an inert
//! `EngineHandles` with no audio threads at all - the GUI layer, and this
//! file's own logic, are still fully exercised by a native (non-Windows)
//! `cargo check`/`cargo test`, just without any real audio behind them.

#![windows_subsystem = "windows"]

mod gui;
mod logging;
mod settings;
mod setup;
mod shared;
mod status_ui;

#[cfg(windows)]
mod engine;

#[cfg(windows)]
fn bootstrap(settings: &settings::Settings) -> gui::Initial {
    let devices = engine::enumerate_devices();

    let mic_target_auto = engine::default_virtual_mic_target(&devices);
    let speaker_source_auto = engine::default_virtual_speaker_source(&devices);
    let mic_target_was_auto_detected = settings.virtual_mic_target_id.is_none();
    let speaker_source_was_auto_detected = settings.virtual_speaker_source_id.is_none();
    let mic_target = settings.virtual_mic_target_id.clone().or_else(|| mic_target_auto.clone());
    let speaker_source = settings.virtual_speaker_source_id.clone().or_else(|| speaker_source_auto.clone());
    // `None` means "system default output device" here, unlike the two
    // virtual-cable roles above (which need an unambiguous match or stay
    // unset) - this is the user's real physical speakers/headphones, same
    // "None = system default" convention as `monitor_device_id`/
    // `recording_playback_device_id`.
    let real_speaker_output = settings.real_speaker_output_device_id.clone();

    // Real hazard, not hypothetical: `default_virtual_mic_target`/
    // `default_virtual_speaker_source`'s own docs flag that with only ONE
    // VB-Cable pair installed, auto-detection picks its two ends for BOTH
    // roles - meaning the speaker pipeline would capture the SAME signal
    // the call app already reads as its own microphone (the mic path's
    // own output), not real incoming call audio at all. Detected here
    // structurally (both roles resolved via auto-detection, meaning each
    // had exactly one unambiguous candidate - i.e. only one cable product
    // is installed) rather than by fragile device-name pattern matching -
    // a user who explicitly chose BOTH devices themselves (e.g. a real
    // VB-CABLE A+B install) never triggers this, since neither came from
    // auto-detection in that case.
    let mic_speaker_device_collision_warning =
        if mic_target_was_auto_detected && speaker_source_was_auto_detected && mic_target_auto.is_some() && speaker_source_auto.is_some() {
            Some(
                "Only one virtual audio cable was detected - the mic target and speaker source were \
                 both auto-selected from it, which means the speaker-outbound path may be capturing \
                 your OWN outgoing mic signal instead of real incoming call audio. Install a second, \
                 independent virtual cable (e.g. VB-CABLE A+B) and explicitly select its capture \
                 endpoint as the speaker source in the GUI."
                    .to_string(),
            )
        } else {
            None
        };

    // Fast, synchronous checks - both cheap enough to redo on every launch
    // rather than caching/skipping them (see `setup::CompatibilityReport`
    // and `Settings::setup_acknowledged` for why a real failure is never
    // permanently suppressed).
    let compat = setup::check_system_compatibility();

    // Self-extract the embedded BVC DLL + model into %LOCALAPPDATA% *before*
    // `engine::start` tries to load them - this used to be a manual/
    // download-based step; it is now synchronous and (bytes already being
    // resident in the binary) fast enough to never need a progress UI. A
    // real failure here (disk full, unwritable %LOCALAPPDATA%, etc.) is
    // surfaced to the setup screen rather than crashing - BVC simply stays
    // unavailable, same graceful-degradation pattern as a missing DLL
    // always used to be.
    let bvc_dir = setup::app_data_dir();
    let bvc_extraction_error = match setup::ensure_bvc_assets_extracted(&bvc_dir) {
        Ok(_) => None,
        Err(e) => {
            crate::log_error!("[clearnairt] WARNING: failed to extract embedded BVC assets: {e:#}");
            Some(format!("{e:#}"))
        }
    };

    match engine::start(
        &bvc_dir,
        settings,
        devices.clone(),
        mic_target.as_deref(),
        speaker_source.as_deref(),
        real_speaker_output.as_deref(),
    ) {
        Ok(outcome) => gui::Initial {
            settings: settings.clone(),
            handles: outcome.handles,
            devices: outcome.devices,
            extra_warnings: mic_speaker_device_collision_warning.into_iter().collect(),
            keepalive: None,
            compat,
            bvc_extraction_error,
        },
        Err(e) => {
            crate::log_error!("[clearnairt] FATAL: failed to start audio engine: {e:#}");
            // Still bring up the GUI rather than exiting silently: an inert
            // engine handle set makes the failure visible (status stays
            // Bypass, no audio flows) instead of the whole app vanishing
            // with no explanation.
            let (_tap, target_speaker_store, enrollment, _enrollment_warning, tse_speaker_store, vanta_speaker_store) =
                shared::build_enrollment_subsystem();
            let handles = shared::EngineHandles::new_from_settings(
                settings,
                false,
                Some("engine failed to start; see stderr log".to_string()),
                false,
                Some("engine failed to start; see stderr log".to_string()),
                false,
                Some("engine failed to start; see stderr log".to_string()),
                vec![format!("Audio engine failed to start: {e:#}")],
                false,
                Some("engine failed to start; see stderr log".to_string()),
                false,
                Some("engine failed to start; see stderr log".to_string()),
                false,
                Some("engine failed to start; see stderr log".to_string()),
                false,
                Some("engine failed to start; see stderr log".to_string()),
                target_speaker_store,
                tse_speaker_store,
                vanta_speaker_store,
                enrollment,
            );
            gui::Initial {
                settings: settings.clone(),
                handles,
                devices,
                extra_warnings: Vec::new(),
                keepalive: None,
                compat,
                bvc_extraction_error,
            }
        }
    }
}

#[cfg(not(windows))]
fn bootstrap(settings: &settings::Settings) -> gui::Initial {
    let compat = setup::check_system_compatibility();
    // Extraction is exercised here too (portable logic, see `setup.rs`) so
    // that a native `cargo check`/`cargo test` run still type-checks and
    // exercises this path, even though it's never real on this platform.
    let bvc_dir = setup::app_data_dir();
    let bvc_extraction_error = setup::ensure_bvc_assets_extracted(&bvc_dir).err().map(|e| format!("{e:#}"));
    let (_tap, target_speaker_store, enrollment, _enrollment_warning, tse_speaker_store, vanta_speaker_store) =
        shared::build_enrollment_subsystem();
    let handles = shared::EngineHandles::new_from_settings(
        settings,
        false,
        Some("not running on Windows".to_string()),
        false,
        Some("not running on Windows".to_string()),
        false,
        Some("not running on Windows".to_string()),
        Vec::new(),
        false,
        Some("not running on Windows".to_string()),
        false,
        Some("not running on Windows".to_string()),
        false,
        Some("not running on Windows".to_string()),
        false,
        Some("not running on Windows".to_string()),
        target_speaker_store,
        tse_speaker_store,
        vanta_speaker_store,
        enrollment,
    );
    gui::Initial {
        settings: settings.clone(),
        handles,
        devices: shared::DeviceList::empty(),
        extra_warnings: vec![
            "Not running on Windows: the real audio engine (WASAPI capture/render, virtual \
             cable I/O, BVC) is unavailable on this platform. This window is a portable-layer \
             type-check/preview only - see the project report for what has and hasn't been \
             verified."
                .to_string(),
        ],
        keepalive: None,
        compat,
        bvc_extraction_error,
    }
}

fn main() -> iced::Result {
    // Must happen before *anything* that touches WASAPI on this thread -
    // `bootstrap()` enumerates devices and runs each pipeline's "does a
    // matching device exist" pre-check on this very thread before spawning
    // any worker thread (each worker thread separately calls this itself
    // internally). Missing this produced a real, observed failure on actual
    // Windows hardware: "CoInitialize has not been called (0x800401F0)" on
    // both device enumeration and default mic capture startup.
    #[cfg(windows)]
    audio_io::initialize_com_for_this_thread();

    logging::init(&setup::app_data_dir());

    // `audio-io`'s capture/render/loopback worker threads used to report
    // their own fatal errors (a WASAPI device disappearing, an event wait
    // timing out, a write to a dead device) via a bare `eprintln!`, which
    // this `#![windows_subsystem = "windows"]` binary has no console to
    // ever show - the thread would silently die while its ring-buffer
    // partner kept logging generic "buffer full, dropping frame" errors
    // forever with no visible root cause. Redirect them into the same
    // file-backed logger everything else in this crate uses.
    audio_io::set_error_sink(|msg| crate::log_error!("{msg}"));

    let settings = settings::load();
    let initial = bootstrap(&settings);
    gui::run(initial)
}
