//! `clearnai.exe` - the ClearNAI window. Talks to `clearnai-engine.exe` (see `clearnai_app::remote`).

#![windows_subsystem = "windows"]

fn main() -> iced::Result {
    #[cfg(windows)]
    return clearnai_app::remote::run_window();
    #[cfg(not(windows))]
    Ok(())
}
