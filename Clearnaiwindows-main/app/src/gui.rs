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
    studio_preset_to_index, DeviceList, DeviceOption,
    EngineHandles, STUDIO_PRESET_NAMES,
};
use dsp_core::{classify_status, compute_percentiles, RealtimeStatus};
use tse_engine::BvcBackend;
use crate::ui_logic::Tab;
use iced::widget::{button, column, row, scrollable, text};
use iced::{Color, Element, Subscription};
use std::cell::RefCell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;

mod icons;
mod style;
mod tabs;

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
    /// Opened from the tray as the quick panel: place it at the bottom-right of the screen on the first tick.
    pub panel_at_start: bool,
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
    /// Post-chain peak ("After") and raw mic peak ("Before"), refreshed every Tick.
    level: f32,
    input_level: f32,
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
    /// Selected tab of the full window. Not persisted (always opens on Home).
    tab: Tab,
    /// Transcript tab: the search box text and the result line of the last Export / Open folder / Copy.
    transcript_query: String,
    transcript_note: Option<String>,
    /// Advanced tab: result line of Export logs / Start with Windows / Reset, and whether the next Reset click
    /// is the confirming one (cleared when the tab changes).
    diag_note: Option<String>,
    reset_armed: bool,
    cpu_percent: Option<f32>,
    /// Problems to show as a banner (every tab and the collapsed bar); `logged_alerts` remembers which keys were
    /// already written to the log so each problem is logged once when it appears and once when it clears.
    alerts: Vec<crate::ui_logic::Alert>,
    logged_alerts: Vec<&'static str>,
    /// Opened from the tray as the quick panel: move it to the bottom-right of the screen on the first tick.
    place_panel: bool,
    /// A full window was just opened: shrink it to the screen if the default is too big.
    fit_full: bool,
    /// Whether the window currently has its title bar (the collapsed bar has none).
    decorated: bool,
}

impl MainState {
    fn new(
        settings: Settings,
        handles: EngineHandles,
        devices: DeviceList,
        warnings: Vec<String>,
        keepalive: Option<Box<dyn std::any::Any + Send>>,
    ) -> Self {
        Self {
            settings,
            handles,
            devices,
            warnings,
            status: RealtimeStatus::Bypass,
            level: 0.0,
            input_level: 0.0,
            _keepalive: keepalive,
            device_switch_seqs: Arc::new(DeviceSwitchSequences::default()),
            io_counters_snapshot: IoCountersSnapshot::default(),
            speaker_status: RealtimeStatus::Bypass,
            speaker_io_counters_snapshot: IoCountersSnapshot::default(),
            consecutive_overloaded_ticks: 0,
            show_low_cpu_suggestion: false,
            tab: Tab::Home,
            transcript_query: String::new(),
            transcript_note: None,
            diag_note: None,
            reset_armed: false,
            cpu_percent: None,
            alerts: Vec::new(),
            logged_alerts: Vec::new(),
            place_panel: false,
            fit_full: false,
            decorated: true,
        }
    }
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
    /// The window was closed: end the process at once (see `update`).
    WindowClosed,
    ToggleMicNoise(bool),
    /// "Voice Gate (suppress quiet background voices)" toggled - see
    /// `dsp_core::VoiceGateStage`'s docs.
    ToggleMicVoiceGate(bool),
    /// Echo cancellation (call audio from the speakers removed from the mic).
    ToggleEchoCancel(bool),
    ToggleMicStudio(bool),
    SelectMicStudioPreset(&'static str),
    /// Accent conversion on/off - its own switch, independent of BVC.
    ToggleAccent(bool),
    /// Accent voice picker (label from `accent_onnx::VOICE_CHOICES`).
    SelectAccentVoice(&'static str),
    /// Live transcript on/off.
    ToggleTranscript(bool),
    /// Empty the transcript box (the saved file is kept).
    ClearTranscript,
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
    /// Picks the BVC backend - Off/Hush/DeepFilterNet/Hush-bpo_v1 (see
    /// `tse_engine::BvcBackend`). An unavailable backend is still shown in
    /// the picker, with a reason line (never a silently-missing option).
    SelectBvcBackend(&'static str),
    /// Speaker-outbound (Pipeline 2) messages - see `tse_engine::SpeakerBvcBackend`'s
    /// docs on why this path's BVC picker is a smaller, separate enum from
    /// the mic path's.
    ToggleSpeakerNoise(bool),
    ToggleSpeakerStudio(bool),
    SelectSpeakerStudioPreset(&'static str),
    SelectSpeakerBvcBackend(&'static str),
    SelectSpeakerSource(DeviceOption),
    SelectRealSpeakerOutput(DeviceOption),
    ToggleSpeakerMasterEnabled(bool),
    /// Switches between the full window and the compact always-on-top bar (persisted).
    ToggleCompact,
    /// Press on the undecorated collapsed bar: drag the window.
    DragWindow,
    /// Open the full window on this tab (the collapsed bar's gear).
    OpenFullAt(Tab),
    SelectTab(Tab),
    /// Mute / unmute the mic path (flips `EngineHandles::mic_master_enabled`); also driven by Ctrl+Shift+M.
    ToggleMute,
    /// "Background voices" on/off without changing which backend is selected (see `update_main`).
    ToggleMicBvc(bool),
    /// Applies one of `ui_logic::PROFILES` through the individual control handlers.
    SelectProfile(&'static str),
    TranscriptSearch(String),
    CopyTranscript,
    ExportTranscript,
    OpenTranscriptFolder,
    ToggleDarkTheme(bool),
    ToggleStartWithWindows(bool),
    ExportLogs,
    /// First click arms, second click (without changing tab) resets processing settings to their defaults.
    ResetDefaults,
    /// Opens VB-Audio's page for the second virtual cable (call audio).
    OpenInstallGuide,
}

const FULL_WINDOW_SIZE: iced::Size = iced::Size::new(940.0, 620.0);

/// Smallest the full window can be dragged to (below ~760 px wide it switches to top tabs).
const MIN_WINDOW_SIZE: iced::Size = iced::Size::new(480.0, 400.0);
const COMPACT_WINDOW_SIZE: iced::Size = iced::Size::new(490.0, 84.0);

fn window_size_and_level(compact: bool) -> (iced::Size, iced::window::Level) {
    if compact {
        (COMPACT_WINDOW_SIZE, iced::window::Level::AlwaysOnTop)
    } else {
        (FULL_WINDOW_SIZE, iced::window::Level::Normal)
    }
}

/// Moves the window to the bottom-right of the screen, above the taskbar (where the tray icon is).
fn place_panel_task(size: iced::Size) -> iced::Task<Message> {
    iced::window::latest().and_then(move |id| {
        iced::window::monitor_size(id).and_then(move |screen| {
            let x = (screen.width - size.width - 16.0).max(0.0);
            let y = (screen.height - size.height - 64.0).max(0.0);
            iced::window::move_to(id, iced::Point::new(x, y))
        })
    })
}

/// Sizes the full window to FULL_WINDOW_SIZE but never larger than the screen allows (room is left for the title bar
/// and the taskbar), then centres it.
fn fit_full_task() -> iced::Task<Message> {
    iced::window::latest().and_then(move |id| {
        iced::window::monitor_size(id).and_then(move |screen| {
            let w = FULL_WINDOW_SIZE.width.min((screen.width - 60.0).max(MIN_WINDOW_SIZE.width));
            let h = FULL_WINDOW_SIZE.height.min((screen.height - 140.0).max(MIN_WINDOW_SIZE.height));
            let x = ((screen.width - w) / 2.0).max(0.0);
            let y = ((screen.height - h) / 2.0 - 24.0).max(0.0);
            iced::Task::batch([iced::window::resize(id, iced::Size::new(w, h)), iced::window::move_to(id, iced::Point::new(x, y))])
        })
    })
}

/// Everything that switches the window between the collapsed bar and the full window: size, always-on-top, title bar,
/// and where it sits (bottom-right by the tray for the bar, centre of the screen for the full window).
fn mode_tasks(state: &mut MainState, compact: bool) -> Vec<iced::Task<Message>> {
    let (size, level) = window_size_and_level(compact);
    let mut tasks = vec![iced::window::latest().and_then(move |id| {
        iced::Task::batch([iced::window::resize(id, size), iced::window::set_level(id, level)])
    })];
    if state.decorated == compact {
        state.decorated = !compact;
        tasks.push(iced::window::latest().and_then(iced::window::toggle_decorations));
    }
    tasks.push(if compact { place_panel_task(size) } else { fit_full_task() });
    tasks
}

/// The engine is the only writer of `settings.json`: send it the window's settings and it saves them.
fn persist(settings: &Settings) {
    crate::remote::save_settings(settings);
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
fn update(screen: &mut Screen, message: Message) -> iced::Task<Message> {
    // Another launch of the window (tray "Open", the Start menu) asks this one to come to the front.
    if matches!(message, Message::Tick) {
        if let Screen::Main(state) = screen {
            let mut tasks = Vec::new();
            // Opened from the tray as the quick panel: put it by the tray (bottom-right), once.
            if std::mem::take(&mut state.place_panel) {
                tasks.push(place_panel_task(COMPACT_WINDOW_SIZE));
            }
            // First tick of a full window: make sure it fits this screen.
            if std::mem::take(&mut state.fit_full) {
                tasks.push(fit_full_task());
            }
            // Another launch of the window (tray, Start menu) asks this one to come to the front, as the quick
            // panel or as the full window.
            if let Some(mode) = crate::remote::take_show_request() {
                let compact = mode == crate::ipc::ShowMode::Panel;
                if state.settings.compact_mode != compact {
                    state.settings.compact_mode = compact;
                    persist(&state.settings);
                    tasks.extend(mode_tasks(state, compact));
                } else if compact {
                    tasks.push(place_panel_task(COMPACT_WINDOW_SIZE));
                }
                tasks.push(iced::window::latest().and_then(|id| {
                    iced::Task::batch([
                        iced::window::set_mode(id, iced::window::Mode::Windowed),
                        iced::window::minimize(id, false),
                        iced::window::gain_focus(id),
                    ])
                }));
            }
            update_main(state, message);
            return iced::Task::batch(tasks);
        }
    }
    if matches!(message, Message::DragWindow) {
        return iced::window::latest().and_then(iced::window::drag);
    }
    if let Message::OpenFullAt(tab) = message {
        if let Screen::Main(state) = screen {
            state.tab = tab;
            state.reset_armed = false;
            if state.settings.compact_mode {
                state.settings.compact_mode = false;
                persist(&state.settings);
                return iced::Task::batch(mode_tasks(state, false));
            }
        }
        return iced::Task::none();
    }
    if matches!(message, Message::ToggleCompact) {
        if let Screen::Main(state) = screen {
            state.settings.compact_mode = !state.settings.compact_mode;
            persist(&state.settings);
            let compact = state.settings.compact_mode;
            return iced::Task::batch(mode_tasks(state, compact));
        }
        return iced::Task::none();
    }
    if matches!(message, Message::CopyTranscript) {
        if let Screen::Main(state) = screen {
            let (lines, _) = state.handles.transcript.snapshot(10_000);
            let all: Vec<(u32, String)> = lines.iter().map(|l| (l.at_secs, l.text.clone())).collect();
            let shown: Vec<(u32, String)> =
                crate::ui_logic::filter_lines(&all, &state.transcript_query).into_iter().cloned().collect();
            state.transcript_note = Some(if shown.is_empty() {
                "Nothing to copy yet.".to_string()
            } else {
                format!("Copied {} line(s) to the clipboard.", shown.len())
            });
            if !shown.is_empty() {
                return iced::clipboard::write(crate::ui_logic::format_transcript(&shown));
            }
        }
        return iced::Task::none();
    }
    update_inner(screen, message);
    iced::Task::none()
}

/// Closing the window ends only this process: the engine keeps the audio running in the background.
fn terminate_process() -> ! {
    crate::sysutil::terminate_process()
}

fn update_inner(screen: &mut Screen, message: Message) {
    if matches!(message, Message::WindowClosed) {
        crate::log_info!("[clearnairt] window closed");
        terminate_process();
    }
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
            Some(MainState::new(
                setup.settings.clone(),
                setup.handles.clone(),
                setup.devices.clone(),
                setup.warnings.clone(),
                setup.keepalive.take(),
            ))
        }
        // Every other message is a Main-screen-only toggle/picker action;
        // the setup screen's `view` never emits these, so this is
        // unreachable in practice, not a silently-swallowed real message.
        _ => None,
    }
}

/// Rebuilds `state.alerts` from the live engine status and logs each problem once when it appears (and once when
/// it clears), so the log and the window always tell the same story.
fn refresh_alerts(state: &mut MainState) {
    use accent_onnx::pipeline::AccentState;
    let h = &state.handles;
    let mut input = crate::ui_logic::AlertInput::default();
    if state.settings.accent_on {
        let st = h.accent.status();
        match st.state {
            AccentState::Paused => input.accent_paused_rtf = Some(st.rtf),
            AccentState::Unavailable => input.accent_unavailable = Some(st.message),
            _ => {}
        }
    }
    if h.mic_bvc.is_on() {
        match BvcBackend::from_index(h.bvc_backend.load()) {
            BvcBackend::Hush => {
                input.bvc_unavailable = !h.mic_bvc_available;
                input.bvc_stopped = h.mic_hush_status.as_ref().is_some_and(|s| s.is_bypassed());
            }
            BvcBackend::HushBpoV1 => {
                input.bvc_unavailable = !h.mic_bpo_v1_available;
                input.bvc_stopped = h.mic_bpo_v1_status.as_ref().is_some_and(|s| s.is_bypassed());
            }
            BvcBackend::DeepFilter => input.bvc_unavailable = !h.mic_deep_filter_available,
            BvcBackend::Off => {}
        }
    }
    input.cpu_struggling = state.show_low_cpu_suggestion;
    input.engine_disconnected = h.remote.as_ref().is_some_and(|r| !r.connected.load(Ordering::Relaxed));
    let alerts = crate::ui_logic::alerts(&input);
    for a in &alerts {
        if !state.logged_alerts.contains(&a.key) {
            crate::log_error!("[clearnairt] ALERT: {}", a.text);
            state.logged_alerts.push(a.key);
        }
    }
    let still: Vec<&'static str> = alerts.iter().map(|a| a.key).collect();
    state.logged_alerts.retain(|k| {
        let keep = still.contains(k);
        if !keep {
            crate::log_info!("[clearnairt] ALERT cleared: {k}");
        }
        keep
    });
    state.alerts = alerts;
}

fn update_main(state: &mut MainState, message: Message) {
    match message {
        Message::Tick => {
            refresh_alerts(state);
            // The window mirrors the engine: latency percentiles are computed there and sent over.
            let (percentiles, speaker_percentiles) = match &state.handles.remote {
                Some(live) => (*live.mic_pct.lock().unwrap(), *live.speaker_pct.lock().unwrap()),
                None => (
                    compute_percentiles(state.handles.latency_log.snapshot()),
                    compute_percentiles(state.handles.speaker_latency_log.snapshot()),
                ),
            };
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
            state.input_level = state.handles.input_level.load();
            state.io_counters_snapshot = IoCountersSnapshot::read(&state.handles.counters);

            let speaker_all_off = !state.handles.speaker_noise.is_on()
                && !state.handles.speaker_bvc.is_on()
                && !state.handles.speaker_studio.is_on();
            state.speaker_status = classify_status(&speaker_percentiles, DEADLINE_NS, speaker_all_off);
            state.speaker_io_counters_snapshot = IoCountersSnapshot::read(&state.handles.speaker_counters);

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
            if let Some(live) = &state.handles.remote {
                let tenths = live.cpu_tenths.load(Ordering::Relaxed);
                state.cpu_percent = (tenths != u32::MAX).then(|| tenths as f32 / 10.0);
                // The engine told us it is quitting (tray > Quit): close with it.
                if live.shutdown.load(Ordering::Relaxed) {
                    terminate_process();
                }
            }
        }
        Message::SelectTab(tab) => {
            state.tab = tab;
            state.reset_armed = false;
        }
        Message::ToggleMute => {
            let on = state.handles.mic_master_enabled.is_on();
            state.handles.mic_master_enabled.set(!on);
        }
        Message::ToggleMicBvc(on) => {
            if on {
                // Turning voices on with no backend selected picks Hush; otherwise keep the user's chosen backend.
                let backend = BvcBackend::from_index(state.handles.bvc_backend.load());
                let name = if backend == BvcBackend::Off { BvcBackend::Hush.name() } else { backend.name() };
                update_main(state, Message::SelectBvcBackend(name));
            } else {
                state.handles.mic_bvc.set(false);
                state.settings.mic_bvc_on = false;
                persist(&state.settings);
            }
        }
        Message::SelectProfile(name) => {
            if let Some(p) = crate::ui_logic::profile_by_name(name) {
                update_main(state, Message::ToggleMicNoise(p.noise));
                update_main(state, Message::ToggleMicVoiceGate(p.voice_gate));
                update_main(state, Message::ToggleMicStudio(p.studio));
                update_main(state, Message::SelectMicStudioPreset(p.studio_preset));
                update_main(state, Message::SelectBvcBackend(p.bvc_backend));
                state.settings.profile = p.name.to_string();
                persist(&state.settings);
            }
        }
        Message::OpenInstallGuide => {
            if let Err(e) = crate::sysutil::open_url("https://vb-audio.com/Cable/") {
                crate::log_error!("[clearnairt] {e}");
            }
        }
        Message::TranscriptSearch(query) => state.transcript_query = query,
        Message::ExportTranscript => {
            let (lines, _) = state.handles.transcript.snapshot(10_000);
            let all: Vec<(u32, String)> = lines.iter().map(|l| (l.at_secs, l.text.clone())).collect();
            let shown: Vec<(u32, String)> =
                crate::ui_logic::filter_lines(&all, &state.transcript_query).into_iter().cloned().collect();
            let dir = crate::setup::app_data_dir().join("transcripts");
            state.transcript_note = Some(if shown.is_empty() {
                "Nothing to export yet.".to_string()
            } else {
                match crate::sysutil::export_text(&dir, &crate::ui_logic::format_transcript(&shown)) {
                    Ok(path) => format!("Exported to {}", path.display()),
                    Err(e) => e,
                }
            });
        }
        Message::OpenTranscriptFolder => {
            let dir = crate::setup::app_data_dir().join("transcripts");
            state.transcript_note = crate::sysutil::open_folder(&dir).err();
        }
        Message::ToggleDarkTheme(on) => {
            state.settings.theme_dark = on;
            persist(&state.settings);
        }
        Message::ToggleStartWithWindows(on) => {
            // The engine owns the Windows sign-in entry and updates it when it saves the settings.
            state.settings.start_with_windows = on;
            state.diag_note = None;
            persist(&state.settings);
        }
        Message::ExportLogs => {
            let app_dir = crate::setup::app_data_dir();
            let dest_dir = crate::sysutil::export_dir(&app_dir);
            state.diag_note = Some(match crate::sysutil::export_log(&app_dir.join("clearnai.log"), &dest_dir) {
                Ok(path) => format!("Log exported to {}", path.display()),
                Err(e) => e,
            });
        }
        Message::ResetDefaults => {
            if !state.reset_armed {
                state.reset_armed = true;
            } else {
                reset_to_defaults(state);
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
        Message::ToggleEchoCancel(on) => {
            state.handles.echo_cancel.set(on);
            state.settings.echo_cancel_on = on;
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
        Message::ToggleAccent(on) => {
            state.settings.accent_on = on;
            state.handles.accent.set_voice(state.settings.accent_effective_voice());
            persist(&state.settings);
        }
        Message::ToggleTranscript(on) => {
            state.settings.transcript_on = on;
            state.handles.transcript.set_enabled(on);
            persist(&state.settings);
        }
        Message::ClearTranscript => {
            state.handles.transcript.clear();
            crate::remote::send(crate::ipc::ClientMsg::TranscriptClear);
        }
        Message::SelectAccentVoice(label) => {
            if let Some(i) = accent_onnx::VOICE_CHOICES.iter().position(|l| *l == label) {
                state.settings.accent_voice = i as u8 + 1;
                state.handles.accent.set_voice(state.settings.accent_effective_voice());
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
        Message::ToggleSpeakerMasterEnabled(on) => {
            state.handles.speaker_master_enabled.set(on);
        }
        Message::SelectBvcBackend(name) => {
            if let Some(backend) = bvc_backend_from_name(name) {
                let index = backend.to_index();
                let wants_on = backend != BvcBackend::Off;
                // Same "never silently enable an unavailable backend"
                // rule `ToggleMicBvc` already enforces for Hush, extended
                // to every backend: picking one whose real implementation
                // couldn't be loaded leaves the actual
                // gate off, even though the *selection* is remembered.
                let available = match backend {
                    BvcBackend::Off => true,
                    BvcBackend::Hush => state.handles.mic_bvc_available,
                    BvcBackend::DeepFilter => state.handles.mic_deep_filter_available,
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
        // Setup-screen-only messages; unreachable once `Screen::Main` is
        // active (its `view` never emits them), same reasoning as the
        // catch-all in `update_setup`.
        Message::ContinueFromSetup
        | Message::WindowClosed
        | Message::ToggleCompact
        | Message::CopyTranscript
        | Message::DragWindow
        | Message::OpenFullAt(_) => {}
    }
}

/// "Reset to defaults": puts every processing option back to `Settings::default()` live, through the same handlers
/// the controls use. Device choices, the setup acknowledgement and the window preferences (compact, theme) are kept
/// so a reset never breaks the virtual-cable setup.
fn reset_to_defaults(state: &mut MainState) {
    let d = Settings::default();
    let keep = state.settings.clone();
    let mut fresh = d.clone();
    fresh.virtual_mic_target_id = keep.virtual_mic_target_id;
    fresh.virtual_speaker_source_id = keep.virtual_speaker_source_id;
    fresh.physical_mic_device_id = keep.physical_mic_device_id;
    fresh.monitor_device_id = keep.monitor_device_id;
    fresh.real_speaker_output_device_id = keep.real_speaker_output_device_id;
    fresh.setup_acknowledged = keep.setup_acknowledged;
    fresh.compact_mode = keep.compact_mode;
    fresh.minimize_to_tray = keep.minimize_to_tray;
    fresh.theme_dark = keep.theme_dark;
    state.settings = fresh;
    state.handles.mic_master_enabled.set(true);
    state.handles.speaker_master_enabled.set(true);
    let preset = |i: u8| STUDIO_PRESET_NAMES[(i as usize).min(STUDIO_PRESET_NAMES.len() - 1)];
    update_main(state, Message::ToggleMicNoise(d.mic_noise_on));
    update_main(state, Message::ToggleMicVoiceGate(d.mic_voice_gate_on));
    update_main(state, Message::ToggleEchoCancel(d.echo_cancel_on));
    update_main(state, Message::ToggleMicStudio(d.mic_studio_on));
    update_main(state, Message::SelectMicStudioPreset(preset(d.mic_studio_preset)));
    update_main(state, Message::SelectBvcBackend(BvcBackend::from_index(d.mic_bvc_backend).name()));
    update_main(state, Message::ToggleAccent(d.accent_on));
    update_main(state, Message::SelectAccentVoice(accent_onnx::VOICE_CHOICES[(d.accent_voice_index() - 1) as usize]));
    update_main(state, Message::ToggleTranscript(d.transcript_on));
    update_main(state, Message::ToggleMonitor(d.monitor_enabled));
    update_main(state, Message::ToggleSpeakerNoise(d.speaker_noise_on));
    update_main(state, Message::ToggleSpeakerStudio(d.speaker_studio_on));
    update_main(state, Message::SelectSpeakerStudioPreset(preset(d.speaker_studio_preset)));
    update_main(
        state,
        Message::SelectSpeakerBvcBackend(tse_engine::SpeakerBvcBackend::from_index(d.speaker_bvc_backend).name()),
    );
    state.reset_armed = false;
    state.diag_note = Some("Settings reset to defaults. Your device choices were kept.".to_string());
    persist(&state.settings);
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

/// Every `tse_engine::BvcBackend` name, in `BvcBackend::ALL` order - a
/// plain `[&'static str; 4]` so `pick_list` can borrow it the same way
/// `STUDIO_PRESET_NAMES` already works for the Studio preset picker.
const BVC_BACKEND_NAMES: [&str; 4] = [
    BvcBackend::Off.name(),
    BvcBackend::Hush.name(),
    BvcBackend::DeepFilter.name(),
    BvcBackend::HushBpoV1.name(),
];

/// Same as `BVC_BACKEND_NAMES` but for the speaker path's smaller
/// `SpeakerBvcBackend` enum.
const SPEAKER_BVC_BACKEND_NAMES: [&str; 3] = [
    tse_engine::SpeakerBvcBackend::Off.name(),
    tse_engine::SpeakerBvcBackend::Hush.name(),
    tse_engine::SpeakerBvcBackend::DeepFilter.name(),
];

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
        compat_line("64-bit processor", setup.compat.arch_ok),
        compat_line("Windows audio", setup.compat.com_ok),
    ]
    .spacing(6);

    let mut details_col = column![].spacing(2);
    for d in &setup.compat.details {
        details_col = details_col.push(text(d).size(11).style(text::secondary));
    }

    // BVC's DLL + model bundle are embedded in the binary and self-extract
    // synchronously before this screen is ever shown (see
    // `main.rs::bootstrap`) - there is no download/progress step left, only
    // a one-line confirmation, or a clear error if extraction itself
    // failed (disk full, unwritable %LOCALAPPDATA%, etc.).
    let model_col: Element<'_, Message> = match &setup.bvc_extraction_error {
        None => column![
            text("Voice models").size(16),
            compat_line("Noise and voice models are ready", true),
        ]
        .spacing(6)
        .into(),
        Some(err) => column![
            text("Voice models").size(16),
            compat_line("Couldn't prepare the noise and voice models", false),
            text(err.clone()).size(12).color(Color::from_rgb(0.85, 0.2, 0.2)),
        ]
        .spacing(8)
        .into(),
    };

    let content = column![
        tabs::logo_with_wordmark(setup.settings.theme_dark, 36.0),
        text("A quick check before you start").size(18),
        compat_col,
        details_col,
        iced::widget::rule::horizontal(1),
        model_col,
        iced::widget::rule::horizontal(1),
        text("You can continue anyway. Everything else works normally, and background voice removal will show as unavailable.")
            .size(12)
            .style(text::secondary),
        button(text("Continue")).on_press(Message::ContinueFromSetup),
    ]
    .spacing(16)
    .padding(20);

    scrollable(content).into()
}

fn view(screen: &Screen) -> Element<'_, Message> {
    match screen {
        Screen::Setup(setup) => view_setup(setup),
        Screen::Main(state) if state.settings.compact_mode => tabs::view_compact(state),
        Screen::Main(state) => tabs::view_full(state),
    }
}

/// The title-bar / taskbar icon: 64x64 straight-alpha RGBA rendered from the logo (see `assets/clearnai.ico`).
fn window_icon() -> Option<iced::window::Icon> {
    const RGBA: &[u8] = include_bytes!("../assets/clearnai_icon_64.rgba");
    iced::window::icon::from_rgba(RGBA.to_vec(), 64, 64).ok()
}

fn theme(screen: &Screen) -> iced::Theme {
    let dark = match screen {
        Screen::Setup(setup) => setup.settings.theme_dark,
        Screen::Main(state) => state.settings.theme_dark,
    };
    style::theme(dark)
}

fn subscription(screen: &Screen) -> Subscription<Message> {
    let mut subs = vec![
        iced::time::every(Duration::from_millis(250)).map(|_| Message::Tick),
        iced::window::close_requests().map(|_| Message::WindowClosed),
    ];
    // The bar has no title bar and so no close button: Escape closes it (the engine keeps running).
    if matches!(screen, Screen::Main(state) if state.settings.compact_mode) {
        subs.push(iced::keyboard::listen().filter_map(|event| match event {
            iced::keyboard::Event::KeyPressed { key: iced::keyboard::Key::Named(iced::keyboard::key::Named::Escape), .. } => {
                Some(Message::WindowClosed)
            }
            _ => None,
        }));
    }
    Subscription::batch(subs)
}

/// Runs the window. Blocks until it is closed; the engine (a separate process) keeps running.
pub fn run(initial: Initial) -> iced::Result {
    let (start_size, start_level) = window_size_and_level(initial.settings.compact_mode);
    let start_compact = initial.settings.compact_mode;
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
                let mut main = MainState::new(initial.settings, initial.handles, initial.devices, warnings, initial.keepalive);
                main.place_panel = initial.panel_at_start;
                main.decorated = !main.settings.compact_mode;
                main.fit_full = !main.settings.compact_mode;
                Screen::Main(main)
            }
        },
        update,
        view,
    )
    .title("ClearNAI")
    .subscription(subscription)
    .theme(theme)
    .exit_on_close_request(false)
    .style(|screen: &Screen, theme: &iced::Theme| {
        let palette = theme.palette();
        let compact = matches!(screen, Screen::Main(state) if state.settings.compact_mode);
        iced::theme::Style {
            // The collapsed bar draws its own rounded background; the window behind it must be see-through.
            background_color: if compact { Color::TRANSPARENT } else { palette.background },
            text_color: palette.text,
        }
    })
    .window(iced::window::Settings {
        size: start_size,
        level: start_level,
        icon: window_icon(),
        decorations: !start_compact,
        transparent: true,
        min_size: Some(MIN_WINDOW_SIZE),
        position: if start_compact { iced::window::Position::Default } else { iced::window::Position::Centered },
        ..Default::default()
    })
    .run()
}
