//! The background engine process (`clearnai-engine.exe`): audio pipelines, tray icon, mute hotkey, autostart,
//! settings, and the pipe the window (`clearnai.exe`) connects to. It has no window of its own, so closing the window
//! never interrupts the audio. See `ipc.rs` for the messages.

use std::fs::File;
use std::io::Write;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

use crate::ipc::{self, role, AccentWire, ClientMsg, Flags, Hello, PipeReader, Polled, ServerMsg, Status, TranscriptWire};
use crate::settings::Settings;
use crate::setup::CompatibilityReport;
use crate::shared::{DeviceList, EngineHandles};
use crate::{device_rules, engine, settings, setup, shared};

/// Everything the engine knows once it has started (or failed to start, in which case `handles` are inert).
pub struct Booted {
    pub settings: Settings,
    pub handles: EngineHandles,
    pub devices: DeviceList,
    pub compat: CompatibilityReport,
    pub bvc_extraction_error: Option<String>,
}

/// Builds the engine (audio threads, models, devices) from the saved settings. Never fails outright: if the engine
/// cannot start, inert handles are returned so the window can still say why.
pub fn bootstrap(settings: &settings::Settings) -> Booted {
    let devices = engine::enumerate_devices();

    // Check the saved device choices against what is installed right now: a wrong or unplugged device falls
    // back to automatic instead of silently misrouting audio (a physical mic saved as the call audio source
    // used to play the user's own voice out of the speakers).
    let resolved = device_rules::resolve(settings, &devices, engine::default_virtual_mic_target(&devices));
    for note in &resolved.notes {
        crate::log_info!("[clearnairt] devices: {note}");
    }
    if resolved.changed {
        if let Err(e) = settings::save(&resolved.settings) {
            crate::log_error!("[clearnairt] failed to save corrected device settings: {e:#}");
        }
    }
    let mic_target = resolved.mic_target.clone();
    // Like the Monitor, the speaker path is never auto-picked: with a single VB-Cable pair the only candidate is
    // "CABLE Output", the other end of the cable the mic path renders into, so the speaker path would capture
    // the user's own processed voice and play it back out of the speakers. It stays off until a separate cable
    // is picked (see docs/VIRTUAL_DEVICES.md).
    let speaker_source = resolved.speaker_source.clone();
    // `None` means "system default output device".
    let real_speaker_output = resolved.real_output.clone();
    let settings = &resolved.settings;

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
        Ok(outcome) => Booted {
            settings: settings.clone(),
            handles: outcome.handles,
            devices: outcome.devices,
            compat,
            bvc_extraction_error,
        },
        Err(e) => {
            crate::log_error!("[clearnairt] FATAL: failed to start audio engine: {e:#}");
            // Still bring up the GUI rather than exiting silently: an inert
            // engine handle set makes the failure visible (status stays
            // Bypass, no audio flows) instead of the whole app vanishing
            // with no explanation.
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
            );
            Booted {
                settings: settings.clone(),
                handles,
                devices,
                compat,
                bvc_extraction_error,
            }
        }
    }
}


/// A mutex named `name` stays alive as long as the process; if it already exists, another copy is running.
pub fn another_instance_running(name: &str) -> bool {
    #[link(name = "kernel32")]
    extern "system" {
        fn CreateMutexW(attrs: *const u8, initial_owner: i32, name: *const u16) -> isize;
        fn GetLastError() -> u32;
    }
    const ERROR_ALREADY_EXISTS: u32 = 183;
    let name: Vec<u16> = name.encode_utf16().chain(std::iter::once(0)).collect();
    // The handle is intentionally never closed: the OS releases it when the process ends.
    let handle = unsafe { CreateMutexW(std::ptr::null(), 0, name.as_ptr()) };
    handle != 0 && unsafe { GetLastError() } == ERROR_ALREADY_EXISTS
}

type SharedWriter = Arc<Mutex<File>>;

struct Host {
    handles: EngineHandles,
    settings: Mutex<Settings>,
    devices: DeviceList,
    compat: CompatibilityReport,
    bvc_extraction_error: Option<String>,
    /// Every connected window, so a quit can tell them all.
    writers: Mutex<Vec<SharedWriter>>,
    cpu_tenths: AtomicU32,
}

/// What this connection has already been told about the transcript.
#[derive(Default)]
struct TxState {
    sent_lines: usize,
}

impl Host {
    fn hello(&self) -> Hello {
        let h = &self.handles;
        Hello {
            version: ipc::PROTOCOL_VERSION,
            settings: self.settings.lock().unwrap().clone(),
            devices: self.devices.clone(),
            compat: self.compat.clone(),
            bvc_extraction_error: self.bvc_extraction_error.clone(),
            flags: Flags {
                mic_bvc_available: h.mic_bvc_available,
                mic_bvc_reason: h.mic_bvc_unavailable_reason.clone(),
                speaker_bvc_available: h.speaker_bvc_available,
                speaker_bvc_reason: h.speaker_bvc_unavailable_reason.clone(),
                speaker_deep_filter_available: h.speaker_deep_filter_available,
                speaker_deep_filter_reason: h.speaker_deep_filter_unavailable_reason.clone(),
                mic_deep_filter_available: h.mic_deep_filter_available,
                mic_deep_filter_reason: h.mic_deep_filter_unavailable_reason.clone(),
                mic_bpo_v1_available: h.mic_bpo_v1_available,
                mic_bpo_v1_reason: h.mic_bpo_v1_unavailable_reason.clone(),
                has_hush_status: h.mic_hush_status.is_some(),
                has_bpo_status: h.mic_bpo_v1_status.is_some(),
                warnings: (*h.warnings).clone(),
            },
        }
    }

    fn status(&self, tx: &mut TxState) -> Status {
        use accent_onnx::pipeline::AccentState;
        use asr_onnx::pipeline::TranscriptState;
        let h = &self.handles;
        let acc = h.accent.status();
        let ts = h.transcript.status();

        let mut transcript = TranscriptWire {
            state: match ts.state {
                TranscriptState::Off => 0,
                TranscriptState::Loading => 1,
                TranscriptState::Listening => 2,
                TranscriptState::Transcribing => 3,
                TranscriptState::Unavailable => 4,
            },
            enabled: h.transcript.enabled(),
            message: ts.message.clone(),
            rtf: ts.rtf,
            ..Default::default()
        };
        if ts.line_count < tx.sent_lines {
            // The transcript was cleared (or restarted): tell the window to start over.
            transcript.reset = true;
            tx.sent_lines = 0;
        }
        let (lines, partial) = h.transcript.snapshot(ts.line_count - tx.sent_lines);
        transcript.new_lines = lines.into_iter().map(|l| (l.at_secs, l.text)).collect();
        transcript.partial = partial;
        tx.sent_lines = ts.line_count;

        let cpu = self.cpu_tenths.load(Ordering::Relaxed);
        Status {
            toggles: h.toggles().iter().map(|t| t.is_on()).collect(),
            presets: h.presets().iter().map(|p| p.load()).collect(),
            level: h.level.load(),
            input_level: h.input_level.load(),
            mic_counters: shared::counters_to_array(&h.counters),
            speaker_counters: shared::counters_to_array(&h.speaker_counters),
            mic_pct: dsp_core::compute_percentiles(h.latency_log.snapshot()).into(),
            speaker_pct: dsp_core::compute_percentiles(h.speaker_latency_log.snapshot()).into(),
            hush_bypassed: h.mic_hush_status.as_ref().is_some_and(|s| s.is_bypassed()),
            bpo_bypassed: h.mic_bpo_v1_status.as_ref().is_some_and(|s| s.is_bypassed()),
            accent: AccentWire {
                voice: acc.voice,
                state: match acc.state {
                    AccentState::Off => 0,
                    AccentState::Loading => 1,
                    AccentState::Starting => 2,
                    AccentState::Active => 3,
                    AccentState::Paused => 4,
                    AccentState::Unavailable => 5,
                },
                rtf: acc.rtf,
                latency_ms: acc.latency_ms,
                underruns: acc.underruns,
                jitter_ms: acc.jitter_ms,
                dropped_input: acc.dropped_input,
                catchups: acc.catchups,
                message: acc.message,
            },
            transcript,
            cpu: (cpu != u32::MAX).then(|| cpu as f32 / 10.0),
        }
    }

    fn apply(self: &Arc<Self>, msg: ClientMsg, writer: &SharedWriter) {
        let h = &self.handles;
        match msg {
            ClientMsg::Toggle { key, on } => {
                if let Some(t) = h.toggles().get(key as usize) {
                    t.set(on);
                }
            }
            ClientMsg::Preset { key, val } => {
                if let Some(p) = h.presets().get(key as usize) {
                    p.store(val);
                }
            }
            ClientMsg::SaveSettings(new) => {
                if let Err(e) = crate::sysutil::set_autostart(new.start_with_windows) {
                    crate::log_error!("[clearnairt] {e}");
                }
                if let Err(e) = settings::save(&new) {
                    crate::log_error!("[clearnairt] failed to persist settings: {e:#}");
                }
                *self.settings.lock().unwrap() = *new;
            }
            ClientMsg::SetDevice { role: r, id } => {
                // Opening a device and joining the old worker can take a while: never on the pipe thread.
                let sw = h.device_switcher.clone();
                let _ = thread::Builder::new().name("clearnai-device-switch".into()).spawn(move || match r {
                    role::PHYSICAL_MIC => sw.set_physical_mic(id),
                    role::VIRTUAL_MIC_TARGET => sw.set_virtual_mic_target(id),
                    role::VIRTUAL_SPEAKER_SOURCE => sw.set_virtual_speaker_source(id),
                    role::REAL_SPEAKER_OUTPUT => sw.set_real_speaker_output(id),
                    role::MONITOR => sw.set_monitor_device(id),
                    _ => {}
                });
            }
            ClientMsg::RefreshDevices => {
                let this = self.clone();
                let writer = writer.clone();
                let _ = thread::Builder::new().name("clearnai-device-refresh".into()).spawn(move || {
                    let list = this.handles.device_switcher.enumerate_devices();
                    let _ = send(&writer, &ServerMsg::Devices(list));
                });
            }
            ClientMsg::AccentVoice(v) => h.accent.set_voice(v),
            ClientMsg::TranscriptEnabled(on) => h.transcript.set_enabled(on),
            ClientMsg::TranscriptClear => h.transcript.clear(),
            ClientMsg::Quit => self.quit(),
        }
    }

    /// Tells every window to close, then ends the process at once (see `sysutil::terminate_process`).
    fn quit(&self) -> ! {
        crate::log_info!("[clearnairt] engine quitting");
        for w in self.writers.lock().unwrap().iter() {
            let _ = send(w, &ServerMsg::Shutdown);
        }
        thread::sleep(Duration::from_millis(200));
        crate::sysutil::terminate_process();
    }
}

/// Writes one message with a single write call, so two threads (status ticks and replies) can never interleave
/// half-lines.
fn send(writer: &SharedWriter, msg: &ServerMsg) -> std::io::Result<()> {
    let mut bytes = Vec::new();
    ipc::write_msg(&mut bytes, msg)?;
    let mut w = writer.lock().unwrap();
    w.write_all(&bytes)?;
    w.flush()
}

/// Starts the window (`clearnai.exe` next to this exe). If it is already open, the new copy just brings the existing
/// window to the front and exits.
fn open_window(mode: ipc::ShowMode) {
    let Ok(me) = std::env::current_exe() else { return };
    let ui = me.with_file_name("clearnai.exe");
    let arg = if mode == ipc::ShowMode::Panel { "--panel" } else { "--full" };
    if let Err(e) = std::process::Command::new(&ui).arg(arg).spawn() {
        crate::log_error!("[clearnairt] could not open the window ({}): {e}", ui.display());
    }
}

fn serve(host: Arc<Host>, pipe: File) {
    let Ok(read_half) = pipe.try_clone() else { return };
    let writer: SharedWriter = Arc::new(Mutex::new(pipe));
    if send(&writer, &ServerMsg::Hello(Box::new(host.hello()))).is_err() {
        return;
    }
    host.writers.lock().unwrap().push(writer.clone());
    crate::log_info!("[clearnairt] window connected");

    let alive = Arc::new(AtomicBool::new(true));
    {
        let (host, writer, alive) = (host.clone(), writer.clone(), alive.clone());
        let _ = thread::Builder::new().name("clearnai-status".into()).spawn(move || {
            let mut tx = TxState::default();
            while alive.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_millis(250));
                let st = host.status(&mut tx);
                if send(&writer, &ServerMsg::Status(Box::new(st))).is_err() {
                    break;
                }
            }
            alive.store(false, Ordering::Relaxed);
        });
    }

    let mut reader = PipeReader::new(read_half);
    while alive.load(Ordering::Relaxed) {
        match reader.poll::<ClientMsg>() {
            Polled::Msg(m) => host.apply(m, &writer),
            Polled::Empty => thread::sleep(Duration::from_millis(20)),
            Polled::Closed => break,
        }
    }
    alive.store(false, Ordering::Relaxed);
    host.writers.lock().unwrap().retain(|w| !Arc::ptr_eq(w, &writer));
    crate::log_info!("[clearnairt] window disconnected");
}

fn toggle_mute(host: &Host) {
    let m = &host.handles.mic_master_enabled;
    m.set(!m.is_on());
}

/// Entry point of `clearnai-engine.exe`. Never returns.
pub fn run_engine() {
    if another_instance_running("Local\\ClearNAI-engine") {
        return;
    }
    // Must happen before anything that touches WASAPI on this thread (see `audio_io::initialize_com_for_this_thread`).
    audio_io::initialize_com_for_this_thread();
    crate::logging::init(&setup::app_data_dir());
    // audio-io reports everything through one channel; route its informational notes to INFO so the log's ERROR
    // lines are real errors only.
    audio_io::set_error_sink(|msg| match msg.split_once("] INFO: ") {
        Some((_, note)) => crate::log_info!("[audio-io] {note}"),
        None => crate::log_error!("{msg}"),
    });
    crate::log_info!("[clearnairt] engine starting");

    let saved = settings::load();
    let booted = bootstrap(&saved);
    if let Err(e) = crate::sysutil::set_autostart(booted.settings.start_with_windows) {
        crate::log_error!("[clearnairt] {e}");
    }
    crate::sysutil::start_mute_hotkey();
    crate::tray::start();

    let host = Arc::new(Host {
        handles: booted.handles,
        settings: Mutex::new(booted.settings),
        devices: booted.devices,
        compat: booted.compat,
        bvc_extraction_error: booted.bvc_extraction_error,
        writers: Mutex::new(Vec::new()),
        cpu_tenths: AtomicU32::new(u32::MAX),
    });

    // Tray menu, tray icon clicks and the Ctrl+Shift+M hotkey. They only touch the engine's own switches, so they work
    // with no window open.
    {
        let host = host.clone();
        let _ = thread::Builder::new().name("clearnai-engine-ui-events".into()).spawn(move || {
            let mut cpu = crate::sysutil::CpuSampler::new();
            let mut ticks = 0u32;
            loop {
                thread::sleep(Duration::from_millis(100));
                for action in crate::tray::take_actions() {
                    match action {
                        crate::tray::TrayAction::Open => open_window(ipc::ShowMode::Panel),
                        crate::tray::TrayAction::OpenFull => open_window(ipc::ShowMode::Full),
                        crate::tray::TrayAction::ToggleMute => toggle_mute(&host),
                        crate::tray::TrayAction::Quit => host.quit(),
                    }
                }
                if crate::sysutil::take_hotkey_pressed() {
                    toggle_mute(&host);
                }
                ticks += 1;
                if ticks % 10 == 0 {
                    if let Some(p) = cpu.sample() {
                        host.cpu_tenths.store((p * 10.0) as u32, Ordering::Relaxed);
                    }
                }
            }
        });
    }

    // Started by hand or by the installer (no --background): show the window once. Autostart at sign-in passes
    // --background and stays in the tray.
    if !std::env::args().any(|a| a == "--background") {
        open_window(ipc::ShowMode::Full);
    }

    loop {
        match ipc::accept() {
            Ok(pipe) => {
                let host = host.clone();
                let _ = thread::Builder::new().name("clearnai-client".into()).spawn(move || serve(host, pipe));
            }
            Err(e) => {
                crate::log_error!("[clearnairt] pipe accept failed: {e}");
                thread::sleep(Duration::from_secs(1));
            }
        }
    }
}
