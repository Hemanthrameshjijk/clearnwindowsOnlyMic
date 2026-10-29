//! "Minimize to tray": a notification-area icon with Open / Mute / Quit, shown while the setting is on. The icon
//! lives on its own thread (it needs a Win32 message loop; the GUI's loop belongs to iced), and its clicks reach the
//! GUI through a small queue polled from the 250 ms tick, like the mute hotkey. Never touches the audio path.

use std::collections::VecDeque;
use std::sync::Mutex;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayAction {
    Open,
    ToggleMute,
    Quit,
}

static QUEUE: Mutex<VecDeque<TrayAction>> = Mutex::new(VecDeque::new());

#[cfg_attr(not(windows), allow(dead_code))]
fn push(action: TrayAction) {
    if let Ok(mut q) = QUEUE.lock() {
        q.push_back(action);
    }
}

/// Everything the tray asked for since the last call (oldest first).
pub fn take_actions() -> Vec<TrayAction> {
    QUEUE.lock().map(|mut q| q.drain(..).collect()).unwrap_or_default()
}

#[cfg_attr(not(windows), allow(dead_code))]
const ID_OPEN: &str = "clearnai-open";
#[cfg_attr(not(windows), allow(dead_code))]
const ID_MUTE: &str = "clearnai-mute";
#[cfg_attr(not(windows), allow(dead_code))]
const ID_QUIT: &str = "clearnai-quit";

#[cfg_attr(not(windows), allow(dead_code))]
fn action_for_menu_id(id: &str) -> Option<TrayAction> {
    match id {
        ID_OPEN => Some(TrayAction::Open),
        ID_MUTE => Some(TrayAction::ToggleMute),
        ID_QUIT => Some(TrayAction::Quit),
        _ => None,
    }
}

/// A simple generated icon (no asset file): a filled blue disc with a white "C"-less ring, `size` x `size` RGBA.
#[cfg_attr(not(windows), allow(dead_code))]
fn icon_rgba(size: u32) -> Vec<u8> {
    let c = (size as f32 - 1.0) / 2.0;
    let (outer, inner) = (size as f32 * 0.48, size as f32 * 0.22);
    let mut px = Vec::with_capacity((size * size * 4) as usize);
    for y in 0..size {
        for x in 0..size {
            let d = ((x as f32 - c).powi(2) + (y as f32 - c).powi(2)).sqrt();
            if d <= inner {
                px.extend_from_slice(&[255, 255, 255, 255]);
            } else if d <= outer {
                px.extend_from_slice(&[10, 108, 240, 255]);
            } else {
                px.extend_from_slice(&[0, 0, 0, 0]);
            }
        }
    }
    px
}

/// Shows the tray icon (no-op if already shown). `muted` is only for the menu's first label.
#[cfg(windows)]
pub fn start() {
    win::start();
}

/// Removes the tray icon (no-op if not shown).
#[cfg(windows)]
pub fn stop() {
    win::stop();
}

#[cfg(not(windows))]
pub fn start() {}
#[cfg(not(windows))]
pub fn stop() {}

#[cfg(windows)]
mod win {
    use super::{action_for_menu_id, icon_rgba, push, TrayAction, ID_MUTE, ID_OPEN, ID_QUIT};
    use std::sync::atomic::{AtomicU32, Ordering};
    use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
    use tray_icon::{Icon, MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent};

    /// Id of the tray thread while it runs (0 = not running).
    static THREAD_ID: AtomicU32 = AtomicU32::new(0);

    #[repr(C)]
    struct Msg {
        hwnd: isize,
        message: u32,
        wparam: usize,
        lparam: isize,
        time: u32,
        pt_x: i32,
        pt_y: i32,
    }

    #[link(name = "user32")]
    extern "system" {
        fn GetMessageW(msg: *mut Msg, hwnd: isize, min: u32, max: u32) -> i32;
        fn TranslateMessage(msg: *const Msg) -> i32;
        fn DispatchMessageW(msg: *const Msg) -> isize;
        fn PostThreadMessageW(thread: u32, msg: u32, wparam: usize, lparam: isize) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThreadId() -> u32;
    }

    const WM_QUIT: u32 = 0x0012;

    pub fn start() {
        if THREAD_ID.load(Ordering::Acquire) != 0 {
            return;
        }
        // Reserve the slot so two quick toggles can't start two icons; the thread stores its real id once running.
        if THREAD_ID.compare_exchange(0, u32::MAX, Ordering::AcqRel, Ordering::Acquire).is_err() {
            return;
        }
        let spawned = std::thread::Builder::new().name("clearnai-tray".into()).spawn(|| unsafe {
            let menu = Menu::new();
            let open = MenuItem::with_id(ID_OPEN, "Open ClearNAI", true, None);
            let mute = MenuItem::with_id(ID_MUTE, "Mute / Unmute mic", true, None);
            let quit = MenuItem::with_id(ID_QUIT, "Quit", true, None);
            let built = menu
                .append_items(&[&open, &mute, &PredefinedMenuItem::separator(), &quit])
                .map_err(|e| e.to_string())
                .and_then(|_| Icon::from_rgba(icon_rgba(32), 32, 32).map_err(|e| e.to_string()))
                .and_then(|icon| {
                    TrayIconBuilder::new()
                        .with_menu(Box::new(menu))
                        .with_menu_on_left_click(false)
                        .with_tooltip("ClearNAI")
                        .with_icon(icon)
                        .build()
                        .map_err(|e| e.to_string())
                });
            let tray = match built {
                Ok(t) => t,
                Err(e) => {
                    crate::log_error!("[clearnairt] could not create the tray icon: {e}");
                    THREAD_ID.store(0, Ordering::Release);
                    return;
                }
            };
            MenuEvent::set_event_handler(Some(|e: MenuEvent| {
                if let Some(a) = action_for_menu_id(e.id().as_ref()) {
                    push(a);
                }
            }));
            TrayIconEvent::set_event_handler(Some(|e: TrayIconEvent| match e {
                TrayIconEvent::DoubleClick { .. } => push(TrayAction::Open),
                TrayIconEvent::Click { button: MouseButton::Left, button_state: MouseButtonState::Up, .. } => push(TrayAction::Open),
                _ => {}
            }));
            THREAD_ID.store(GetCurrentThreadId(), Ordering::Release);
            let mut msg: Msg = std::mem::zeroed();
            while GetMessageW(&mut msg, 0, 0, 0) > 0 {
                TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
            MenuEvent::set_event_handler(None::<fn(MenuEvent)>);
            TrayIconEvent::set_event_handler(None::<fn(TrayIconEvent)>);
            drop(tray);
            THREAD_ID.store(0, Ordering::Release);
        });
        if let Err(e) = spawned {
            crate::log_error!("[clearnairt] could not start the tray thread: {e}");
            THREAD_ID.store(0, Ordering::Release);
        }
    }

    pub fn stop() {
        let id = THREAD_ID.load(Ordering::Acquire);
        if id != 0 && id != u32::MAX {
            unsafe { PostThreadMessageW(id, WM_QUIT, 0, 0) };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn menu_ids_map_to_actions() {
        assert_eq!(action_for_menu_id(ID_OPEN), Some(TrayAction::Open));
        assert_eq!(action_for_menu_id(ID_MUTE), Some(TrayAction::ToggleMute));
        assert_eq!(action_for_menu_id(ID_QUIT), Some(TrayAction::Quit));
        assert_eq!(action_for_menu_id("other"), None);
    }

    #[test]
    fn icon_has_expected_size_and_visible_pixels() {
        let px = icon_rgba(32);
        assert_eq!(px.len(), 32 * 32 * 4);
        assert!(px.chunks(4).any(|p| p[3] == 255), "icon must not be fully transparent");
        assert_eq!(px[3], 0, "corner pixel is transparent");
    }

    #[test]
    fn queued_actions_are_drained_in_order() {
        let _ = take_actions();
        push(TrayAction::Open);
        push(TrayAction::Quit);
        assert_eq!(take_actions(), vec![TrayAction::Open, TrayAction::Quit]);
        assert!(take_actions().is_empty());
    }
}
