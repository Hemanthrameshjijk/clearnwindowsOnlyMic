//! `clearnai-engine.exe` - the background engine (audio, tray, hotkey, settings). No window; see `clearnai_app::host`.

#![windows_subsystem = "windows"]

fn main() {
    #[cfg(windows)]
    clearnai_app::host::run_engine();
}
