//! ClearNAI for Windows: one library, two programs.
//!
//! * `clearnai-engine.exe` (`bin/clearnai-engine.rs` -> [`host`]): the background engine. Runs the audio pipelines,
//!   the tray icon, the mute hotkey and autostart, owns `settings.json`, and serves the pipe described in [`ipc`].
//!   It has no window, so closing the window never interrupts the audio.
//! * `clearnai.exe` (`bin/clearnai.rs` -> [`remote`] + [`gui`]): the window. It connects to the engine (starting it if
//!   needed), shows what the engine reports and sends it commands.
//!
//! See `docs/VIRTUAL_DEVICES.md` for the virtual audio cable setup the audio path needs, and `engine.rs` / `gui.rs`
//! for the pipeline and the window.
//!
//! # Portable vs Windows-only
//!
//! `settings`, `status_ui`, `shared`, `ui_logic`, `ipc` (the messages) and `gui` are ordinary portable Rust and build
//! and test anywhere. `engine`, `host` and the pipe halves of `ipc`/`remote` only exist on Windows.

pub mod device_rules;
pub mod gui;
pub mod ipc;
pub mod logging;
pub mod settings;
pub mod setup;
pub mod shared;
pub mod status_ui;
pub mod sysutil;
pub mod tray;
pub mod ui_logic;

#[cfg(windows)]
pub mod engine;
#[cfg(windows)]
pub mod host;
#[cfg(windows)]
pub mod remote;

/// Non-Windows stand-in: there is no engine to talk to, so every call is a no-op.
#[cfg(not(windows))]
pub mod remote {
    pub fn send(_msg: crate::ipc::ClientMsg) {}
    pub fn save_settings(_settings: &crate::settings::Settings) {}
    pub fn take_show_request() -> Option<crate::ipc::ShowMode> {
        None
    }
    pub fn launch_mode() -> Option<crate::ipc::ShowMode> {
        None
    }
}
