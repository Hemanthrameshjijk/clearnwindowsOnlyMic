//! The window process (`clearnai.exe`): connects to the background engine over the pipe (see `ipc.rs`), and keeps a
//! local *mirror* of the engine's state in an ordinary `EngineHandles`, so the window code reads and flips switches
//! exactly as it did when the engine lived in the same process.
//!
//! * Engine -> window: every status message is copied into the mirror (levels, counters, accent and transcript state,
//!   switches, selectors).
//! * Window -> engine: the sync thread notices when the window changed a switch or selector and sends the command;
//!   `persist()` sends the whole `Settings`; device changes go through [`RemoteSwitcher`].
//! * If the pipe drops, the window shows a banner, restarts the engine if it is gone, and reconnects.

use std::fs::File;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Mutex, OnceLock};
use std::thread;
use std::time::{Duration, Instant};

use crate::ipc::{self, role, ClientMsg, Hello, PipeReader, Polled, ServerMsg, ShowMode, Status};
use crate::settings::Settings;
use crate::shared::{self, DeviceList, EngineHandles, LiveDeviceSwitcher, RemoteLive};

/// A change the window made is not overwritten by the engine's (older) value for this long, giving the command time
/// to arrive and be applied.
const PENDING_GRACE: Duration = Duration::from_millis(1500);

static LINK: OnceLock<Arc<Link>> = OnceLock::new();
/// 0 = nothing asked, 1 = full window, 2 = quick panel.
static SHOW_REQUESTED: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

/// The window's end of the pipe. The file is replaced on reconnect.
struct Link {
    writer: Mutex<Option<File>>,
    live: Arc<RemoteLive>,
    devices_tx: Sender<DeviceList>,
}

impl Link {
    fn send(&self, msg: &ClientMsg) {
        let mut guard = self.writer.lock().unwrap();
        let Some(file) = guard.as_mut() else { return };
        let mut bytes = Vec::new();
        if ipc::write_msg(&mut bytes, msg).is_err() {
            return;
        }
        use std::io::Write;
        if file.write_all(&bytes).and_then(|_| file.flush()).is_err() {
            // The reader thread notices the dead pipe and reconnects; the command is lost, which is fine for a
            // switch (the sync thread re-sends on the next difference).
            *guard = None;
        }
    }
}

/// Sends a command to the engine (no-op before the window has connected).
pub fn send(msg: ClientMsg) {
    if let Some(link) = LINK.get() {
        link.send(&msg);
    }
}

/// The window's settings changed: the engine saves them (it is the only writer of `settings.json`).
pub fn save_settings(settings: &Settings) {
    send(ClientMsg::SaveSettings(Box::new(settings.clone())));
}

/// Set once after another launch of the window asked this one to come to the front, and in which form.
pub fn take_show_request() -> Option<ShowMode> {
    match SHOW_REQUESTED.swap(0, Ordering::Relaxed) {
        1 => Some(ShowMode::Full),
        2 => Some(ShowMode::Panel),
        _ => None,
    }
}

/// `--panel` / `--full` on the command line (the engine passes one when the tray opens the window).
pub fn launch_mode() -> Option<ShowMode> {
    std::env::args().find_map(|a| match a.as_str() {
        "--panel" => Some(ShowMode::Panel),
        "--full" => Some(ShowMode::Full),
        _ => None,
    })
}

/// Device changes are commands to the engine; listing devices asks the engine and waits for its answer.
struct RemoteSwitcher {
    devices_rx: Mutex<Receiver<DeviceList>>,
    last: Mutex<DeviceList>,
}

impl RemoteSwitcher {
    fn set(&self, r: u8, id: Option<String>) {
        send(ClientMsg::SetDevice { role: r, id });
    }
}

impl LiveDeviceSwitcher for RemoteSwitcher {
    fn set_physical_mic(&self, id: Option<String>) {
        self.set(role::PHYSICAL_MIC, id);
    }
    fn set_virtual_mic_target(&self, id: Option<String>) {
        self.set(role::VIRTUAL_MIC_TARGET, id);
    }
    fn set_virtual_speaker_source(&self, id: Option<String>) {
        self.set(role::VIRTUAL_SPEAKER_SOURCE, id);
    }
    fn set_real_speaker_output(&self, id: Option<String>) {
        self.set(role::REAL_SPEAKER_OUTPUT, id);
    }
    fn set_monitor_device(&self, id: Option<String>) {
        self.set(role::MONITOR, id);
    }
    fn enumerate_devices(&self) -> DeviceList {
        let rx = self.devices_rx.lock().unwrap();
        while rx.try_recv().is_ok() {} // drop any stale answer
        send(ClientMsg::RefreshDevices);
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(list) => {
                *self.last.lock().unwrap() = list.clone();
                list
            }
            Err(_) => self.last.lock().unwrap().clone(),
        }
    }
}

fn engine_exe() -> Option<std::path::PathBuf> {
    std::env::current_exe().ok().map(|me| me.with_file_name("clearnai-engine.exe"))
}

fn start_engine() {
    if let Some(exe) = engine_exe() {
        if let Err(e) = std::process::Command::new(&exe).arg("--background").spawn() {
            crate::log_error!("[clearnairt] could not start the engine ({}): {e}", exe.display());
        }
    }
}

/// Connects to the engine, starting it first if it is not running. Gives up after `limit`.
fn connect_with_start(limit: Duration) -> Option<File> {
    if let Ok(f) = ipc::connect() {
        return Some(f);
    }
    start_engine();
    let t0 = Instant::now();
    while t0.elapsed() < limit {
        thread::sleep(Duration::from_millis(250));
        if let Ok(f) = ipc::connect() {
            return Some(f);
        }
    }
    None
}

/// Waits for the next message of the wanted kind.
fn wait_for<T>(reader: &mut PipeReader, limit: Duration, mut pick: impl FnMut(ServerMsg) -> Option<T>) -> Option<T> {
    let t0 = Instant::now();
    while t0.elapsed() < limit {
        match reader.poll::<ServerMsg>() {
            Polled::Msg(m) => {
                if let Some(v) = pick(m) {
                    return Some(v);
                }
            }
            Polled::Empty => thread::sleep(Duration::from_millis(20)),
            Polled::Closed => return None,
        }
    }
    None
}

fn build_mirror(hello: &Hello, live: Arc<RemoteLive>, switcher: Arc<RemoteSwitcher>) -> (EngineHandles, Option<Arc<AtomicBool>>, Option<Arc<AtomicBool>>) {
    let f = &hello.flags;
    let mut handles = EngineHandles::new_from_settings(
        &hello.settings,
        f.mic_bvc_available,
        f.mic_bvc_reason.clone(),
        f.speaker_bvc_available,
        f.speaker_bvc_reason.clone(),
        f.speaker_deep_filter_available,
        f.speaker_deep_filter_reason.clone(),
        f.warnings.clone(),
        f.mic_deep_filter_available,
        f.mic_deep_filter_reason.clone(),
        f.mic_bpo_v1_available,
        f.mic_bpo_v1_reason.clone(),
    );
    let mut hush_flag = None;
    let mut bpo_flag = None;
    if f.has_hush_status {
        let (st, flag) = dsp_core::WatchdogStatus::mirror();
        handles.mic_hush_status = Some(st);
        hush_flag = Some(flag);
    }
    if f.has_bpo_status {
        let (st, flag) = dsp_core::WatchdogStatus::mirror();
        handles.mic_bpo_v1_status = Some(st);
        bpo_flag = Some(flag);
    }
    handles.remote = Some(live);
    handles.device_switcher = switcher;
    (handles, hush_flag, bpo_flag)
}

/// Copies the parts of a status that are plain observations (not switches the window can also change).
fn apply_observed(handles: &EngineHandles, hush: &Option<Arc<AtomicBool>>, bpo: &Option<Arc<AtomicBool>>, st: &Status) {
    use accent_onnx::pipeline::{AccentState, AccentStatus};
    use asr_onnx::pipeline::{TranscriptLine, TranscriptState};
    handles.level.store(st.level);
    handles.input_level.store(st.input_level);
    shared::counters_from_array(&handles.counters, &st.mic_counters);
    shared::counters_from_array(&handles.speaker_counters, &st.speaker_counters);
    if let Some(live) = &handles.remote {
        *live.mic_pct.lock().unwrap() = (&st.mic_pct).into();
        *live.speaker_pct.lock().unwrap() = (&st.speaker_pct).into();
        live.cpu_tenths.store(st.cpu.map_or(u32::MAX, |c| (c * 10.0) as u32), Ordering::Relaxed);
    }
    if let Some(f) = hush {
        f.store(st.hush_bypassed, Ordering::Relaxed);
    }
    if let Some(f) = bpo {
        f.store(st.bpo_bypassed, Ordering::Relaxed);
    }
    let a = &st.accent;
    handles.accent.apply_remote(&AccentStatus {
        voice: a.voice,
        state: match a.state {
            1 => AccentState::Loading,
            2 => AccentState::Starting,
            3 => AccentState::Active,
            4 => AccentState::Paused,
            5 => AccentState::Unavailable,
            _ => AccentState::Off,
        },
        rtf: a.rtf,
        latency_ms: a.latency_ms,
        underruns: a.underruns,
        jitter_ms: a.jitter_ms,
        dropped_input: a.dropped_input,
        catchups: a.catchups,
        message: a.message.clone(),
    });
    let t = &st.transcript;
    handles.transcript.apply_remote(
        match t.state {
            1 => TranscriptState::Loading,
            2 => TranscriptState::Listening,
            3 => TranscriptState::Transcribing,
            4 => TranscriptState::Unavailable,
            _ => TranscriptState::Off,
        },
        &t.message,
        t.rtf,
        t.reset,
        t.new_lines.iter().map(|(at_secs, text)| TranscriptLine { at_secs: *at_secs, text: text.clone() }).collect(),
        &t.partial,
    );
}

/// Keeps the window's switches and selectors equal to the engine's, in both directions (see the module docs).
struct SwitchSync {
    known_toggles: Vec<bool>,
    known_presets: Vec<u8>,
    pend_toggles: Vec<Option<Instant>>,
    pend_presets: Vec<Option<Instant>>,
    known_voice: u8,
    pend_voice: Option<Instant>,
    known_enabled: bool,
    pend_enabled: Option<Instant>,
}

impl SwitchSync {
    fn new(h: &EngineHandles) -> Self {
        Self {
            known_toggles: h.toggles().iter().map(|t| t.is_on()).collect(),
            known_presets: h.presets().iter().map(|p| p.load()).collect(),
            pend_toggles: vec![None; ipc::TOGGLE_COUNT],
            pend_presets: vec![None; ipc::PRESET_COUNT],
            known_voice: h.accent.voice(),
            pend_voice: None,
            known_enabled: h.transcript.enabled(),
            pend_enabled: None,
        }
    }

    /// One pass: send what the window changed, adopt what the engine changed.
    fn step(&mut self, h: &EngineHandles, latest: Option<&Status>) {
        let now = Instant::now();
        // Generic helper over (local getter/setter, engine value, send).
        fn pass<T: PartialEq + Copy>(
            local: T,
            engine: Option<T>,
            known: &mut T,
            pend: &mut Option<Instant>,
            now: Instant,
            set_local: impl FnOnce(T),
            send_cmd: impl FnOnce(T),
        ) {
            if local != *known {
                // The window changed it: tell the engine and hold off adopting the engine's value for a moment.
                send_cmd(local);
                *known = local;
                *pend = Some(now);
                return;
            }
            let Some(engine) = engine else { return };
            match *pend {
                Some(t) if engine == local => {
                    let _ = t;
                    *pend = None;
                }
                Some(t) if now.duration_since(t) < PENDING_GRACE => {}
                _ => {
                    *pend = None;
                    if engine != local {
                        set_local(engine);
                        *known = engine;
                    }
                }
            }
        }

        for (i, t) in h.toggles().iter().enumerate() {
            let engine = latest.and_then(|s| s.toggles.get(i).copied());
            pass(
                t.is_on(),
                engine,
                &mut self.known_toggles[i],
                &mut self.pend_toggles[i],
                now,
                |v| t.set(v),
                |v| send(ClientMsg::Toggle { key: i as u8, on: v }),
            );
        }
        for (i, p) in h.presets().iter().enumerate() {
            let engine = latest.and_then(|s| s.presets.get(i).copied());
            pass(
                p.load(),
                engine,
                &mut self.known_presets[i],
                &mut self.pend_presets[i],
                now,
                |v| p.store(v),
                |v| send(ClientMsg::Preset { key: i as u8, val: v }),
            );
        }
        pass(
            h.accent.voice(),
            latest.map(|s| s.accent.voice),
            &mut self.known_voice,
            &mut self.pend_voice,
            now,
            |v| h.accent.set_voice(v),
            |v| send(ClientMsg::AccentVoice(v)),
        );
        pass(
            h.transcript.enabled(),
            latest.map(|s| s.transcript.enabled),
            &mut self.known_enabled,
            &mut self.pend_enabled,
            now,
            |v| h.transcript.set_enabled(v),
            |v| send(ClientMsg::TranscriptEnabled(v)),
        );
    }
}

/// Everything the window process needs after connecting.
struct Connected {
    initial: crate::gui::Initial,
}

fn connect_and_mirror() -> Option<Connected> {
    let file = connect_with_start(Duration::from_secs(90))?;
    let mut reader = PipeReader::new(file.try_clone().ok()?);
    let hello = wait_for(&mut reader, Duration::from_secs(60), |m| match m {
        ServerMsg::Hello(h) => Some(h),
        _ => None,
    })?;
    if hello.version != ipc::PROTOCOL_VERSION {
        crate::log_error!(
            "[clearnairt] engine speaks protocol {} but this window speaks {}; reinstall ClearNAI",
            hello.version,
            ipc::PROTOCOL_VERSION
        );
        return None;
    }

    let live = Arc::new(RemoteLive::new());
    let (devices_tx, devices_rx) = channel();
    let switcher = Arc::new(RemoteSwitcher { devices_rx: Mutex::new(devices_rx), last: Mutex::new(hello.devices.clone()) });
    let (handles, hush, bpo) = build_mirror(&hello, live.clone(), switcher);

    let link = Arc::new(Link { writer: Mutex::new(Some(file)), live: live.clone(), devices_tx });
    let _ = LINK.set(link.clone());

    // Adopt the engine's real switch positions before the window is first drawn.
    let first = wait_for(&mut reader, Duration::from_secs(3), |m| match m {
        ServerMsg::Status(s) => Some(s),
        _ => None,
    });
    let latest: Arc<Mutex<Option<Status>>> = Arc::new(Mutex::new(None));
    let mut sync = SwitchSync::new(&handles);
    if let Some(st) = first {
        apply_observed(&handles, &hush, &bpo, &st);
        for (i, t) in handles.toggles().iter().enumerate() {
            if let Some(v) = st.toggles.get(i) {
                t.set(*v);
            }
        }
        for (i, p) in handles.presets().iter().enumerate() {
            if let Some(v) = st.presets.get(i) {
                p.store(*v);
            }
        }
        handles.accent.set_voice(st.accent.voice);
        handles.transcript.set_enabled(st.transcript.enabled);
        sync = SwitchSync::new(&handles);
        *latest.lock().unwrap() = Some(*st);
    }

    // Reader thread: applies observations, answers device lists, reconnects when the pipe drops.
    {
        let (handles, latest, link) = (handles.clone(), latest.clone(), link.clone());
        let _ = thread::Builder::new().name("clearnai-engine-link".into()).spawn(move || {
            let mut reader = reader;
            loop {
                match reader.poll::<ServerMsg>() {
                    Polled::Msg(ServerMsg::Status(st)) => {
                        apply_observed(&handles, &hush, &bpo, &st);
                        *latest.lock().unwrap() = Some(*st);
                    }
                    Polled::Msg(ServerMsg::Devices(list)) => {
                        let _ = link.devices_tx.send(list);
                    }
                    Polled::Msg(ServerMsg::Shutdown) => link.live.shutdown.store(true, Ordering::Relaxed),
                    Polled::Msg(ServerMsg::Hello(_)) => {}
                    Polled::Empty => thread::sleep(Duration::from_millis(20)),
                    Polled::Closed => {
                        link.live.connected.store(false, Ordering::Relaxed);
                        *link.writer.lock().unwrap() = None;
                        if link.live.shutdown.load(Ordering::Relaxed) {
                            return;
                        }
                        crate::log_error!("[clearnairt] lost the connection to the engine; reconnecting");
                        // Loop until the engine (restarted if needed) answers again.
                        let (file, new_reader) = loop {
                            if let Some(f) = connect_with_start(Duration::from_secs(30)) {
                                if let Ok(rf) = f.try_clone() {
                                    let mut r = PipeReader::new(rf);
                                    if wait_for(&mut r, Duration::from_secs(60), |m| matches!(m, ServerMsg::Hello(_)).then_some(())).is_some() {
                                        break (f, r);
                                    }
                                }
                            }
                            thread::sleep(Duration::from_secs(2));
                        };
                        reader = new_reader;
                        // Forget what we thought the engine had, so its (possibly fresh) state is adopted.
                        *latest.lock().unwrap() = None;
                        *link.writer.lock().unwrap() = Some(file);
                        link.live.connected.store(true, Ordering::Relaxed);
                        crate::log_info!("[clearnairt] reconnected to the engine");
                    }
                }
            }
        });
    }

    // Sync thread: window -> engine for switches, engine -> window for switches changed elsewhere (tray, hotkey).
    {
        let (handles, latest) = (handles.clone(), latest.clone());
        let _ = thread::Builder::new().name("clearnai-switch-sync".into()).spawn(move || loop {
            thread::sleep(Duration::from_millis(50));
            let snapshot = latest.lock().unwrap().clone();
            sync.step(&handles, snapshot.as_ref());
        });
    }

    Some(Connected {
        initial: crate::gui::Initial {
            settings: hello.settings.clone(),
            handles,
            devices: hello.devices.clone(),
            extra_warnings: Vec::new(),
            keepalive: None,
            compat: hello.compat.clone(),
            bvc_extraction_error: hello.bvc_extraction_error.clone(),
            panel_at_start: false,
        },
    })
}

// ---- one window at a time ---------------------------------------------------------------------------------------

#[link(name = "kernel32")]
extern "system" {
    fn CreateEventW(attrs: *const u8, manual_reset: i32, initial: i32, name: *const u16) -> isize;
    fn OpenEventW(access: u32, inherit: i32, name: *const u16) -> isize;
    fn SetEvent(h: isize) -> i32;
    fn WaitForSingleObject(h: isize, ms: u32) -> u32;
}

fn show_event_name(mode: ShowMode) -> Vec<u16> {
    let name = if mode == ShowMode::Panel { "Local\\ClearNAI-window-show-panel" } else { "Local\\ClearNAI-window-show-full" };
    name.encode_utf16().chain(std::iter::once(0)).collect()
}

/// A second launch of the window: ask the first to come to the front.
fn signal_existing_window(mode: ShowMode) {
    const EVENT_MODIFY_STATE: u32 = 0x0002;
    let name = show_event_name(mode);
    let h = unsafe { OpenEventW(EVENT_MODIFY_STATE, 0, name.as_ptr()) };
    if h != 0 {
        unsafe { SetEvent(h) };
    }
}

/// The first window: listen for [`signal_existing_window`].
fn start_show_listener() {
    for (mode, code) in [(ShowMode::Full, 1u8), (ShowMode::Panel, 2u8)] {
        let name = show_event_name(mode);
        let h = unsafe { CreateEventW(std::ptr::null(), 0, 0, name.as_ptr()) };
        if h == 0 {
            continue;
        }
        let _ = thread::Builder::new().name("clearnai-window-show".into()).spawn(move || loop {
            if unsafe { WaitForSingleObject(h, 0xFFFF_FFFF) } == 0 {
                SHOW_REQUESTED.store(code, Ordering::Relaxed);
            }
        });
    }
}

/// Entry point of `clearnai.exe`.
pub fn run_window() -> iced::Result {
    if crate::host::another_instance_running("Local\\ClearNAI-window") {
        signal_existing_window(launch_mode().unwrap_or(ShowMode::Full));
        return Ok(());
    }
    crate::logging::init_named(&crate::setup::app_data_dir(), "clearnai-window.log");
    start_show_listener();

    let Some(connected) = connect_and_mirror() else {
        crate::log_error!("[clearnairt] could not reach the ClearNAI engine");
        return Ok(());
    };
    let mut initial = connected.initial;
    // The tray opens the quick panel; the Start menu / desktop icon and the tray menu open the full window. Either way
    // the choice is for this launch only (the Expand / Collapse buttons are what change the remembered mode).
    match launch_mode() {
        Some(ShowMode::Panel) => {
            initial.settings.compact_mode = true;
            initial.panel_at_start = true;
        }
        Some(ShowMode::Full) | None => initial.settings.compact_mode = false,
    }
    crate::gui::run(initial)
}
