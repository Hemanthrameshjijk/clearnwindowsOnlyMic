//! Small OS helpers for the window: the Ctrl+Shift+M mute hotkey, "Start with Windows", process CPU %, exporting
//! files and opening a folder. Raw FFI like `gui.rs`/`main.rs` already use (no new dependency). Everything is
//! best-effort and never touches the audio path; the Windows-only parts have inert stubs elsewhere.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

static HOTKEY_PRESSED: AtomicBool = AtomicBool::new(false);

/// True once per Ctrl+Shift+M press; polled from the GUI's 250 ms tick.
pub fn take_hotkey_pressed() -> bool {
    HOTKEY_PRESSED.swap(false, Ordering::Relaxed)
}

/// What Windows should launch at sign-in: the quoted engine exe path, started hidden in the background.
pub fn run_value(exe: &Path) -> String {
    format!("\"{}\" --background", exe.display())
}

/// Ends this process at once. A normal exit runs DLL detach for onnxruntime and the audio libraries, which was
/// observed to spin and hang ("Not responding" in the background), and dropping the audio engine joins threads that
/// can be stuck on an invalid device; TerminateProcess ends the process immediately.
pub fn terminate_process() -> ! {
    #[cfg(windows)]
    {
        #[link(name = "kernel32")]
        extern "system" {
            fn GetCurrentProcess() -> isize;
            fn TerminateProcess(h: isize, code: u32) -> i32;
        }
        unsafe { TerminateProcess(GetCurrentProcess(), 0) };
    }
    std::process::exit(0);
}

/// Process CPU use as a percentage of the whole machine (all cores), from CPU time and wall time over one interval,
/// both in 100 ns units.
pub fn cpu_percent(delta_cpu_100ns: u64, delta_wall_100ns: u64, cores: usize) -> f32 {
    if delta_wall_100ns == 0 || cores == 0 {
        return 0.0;
    }
    (delta_cpu_100ns as f64 / delta_wall_100ns as f64 / cores as f64 * 100.0).clamp(0.0, 100.0) as f32
}

pub struct CpuSampler {
    last_cpu: u64,
    last_wall: Instant,
    cores: usize,
}

impl CpuSampler {
    pub fn new() -> Self {
        Self {
            last_cpu: process_cpu_time_100ns().unwrap_or(0),
            last_wall: Instant::now(),
            cores: std::thread::available_parallelism().map_or(1, |n| n.get()),
        }
    }

    /// `Some(percent)` at most once a second (shorter intervals are too noisy), else `None`.
    pub fn sample(&mut self) -> Option<f32> {
        let wall = self.last_wall.elapsed();
        if wall < Duration::from_secs(1) {
            return None;
        }
        let cpu = process_cpu_time_100ns()?;
        let pct = cpu_percent(cpu.saturating_sub(self.last_cpu), (wall.as_nanos() / 100) as u64, self.cores);
        self.last_cpu = cpu;
        self.last_wall = Instant::now();
        Some(pct)
    }
}

pub fn unix_now() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_secs())
}

/// Where exported files go: the Desktop, else Downloads, else `fallback`.
pub fn export_dir(fallback: &Path) -> PathBuf {
    if let Some(home) = std::env::var_os("USERPROFILE") {
        for sub in ["Desktop", "Downloads"] {
            let p = PathBuf::from(&home).join(sub);
            if p.is_dir() {
                return p;
            }
        }
    }
    fallback.to_path_buf()
}

/// Copies `src` (the app log) to `<dir>\ClearNAI-log-<unix>.txt` and returns the new path.
pub fn export_log(src: &Path, dir: &Path) -> Result<PathBuf, String> {
    if !src.is_file() {
        return Err(format!("no log file yet at {}", src.display()));
    }
    let dest = dir.join(format!("ClearNAI-log-{}.txt", unix_now()));
    std::fs::copy(src, &dest).map_err(|e| format!("could not copy the log to {}: {e}", dest.display()))?;
    Ok(dest)
}

/// Writes `text` to `<dir>\transcript-export-<unix>.txt` (creating `dir`) and returns the path.
pub fn export_text(dir: &Path, text: &str) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    let dest = dir.join(format!("transcript-export-{}.txt", unix_now()));
    std::fs::write(&dest, text).map_err(|e| format!("could not write {}: {e}", dest.display()))?;
    Ok(dest)
}

#[cfg(windows)]
pub fn open_folder(dir: &Path) -> Result<(), String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("could not create {}: {e}", dir.display()))?;
    std::process::Command::new("explorer.exe")
        .arg(dir)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("could not open {}: {e}", dir.display()))
}

#[cfg(not(windows))]
pub fn open_folder(_dir: &Path) -> Result<(), String> {
    Err("opening a folder is only supported on Windows".to_string())
}

/// Opens a web page in the default browser (Windows `explorer.exe <url>`).
#[cfg(windows)]
pub fn open_url(url: &str) -> Result<(), String> {
    std::process::Command::new("explorer.exe")
        .arg(url)
        .spawn()
        .map(|_| ())
        .map_err(|e| format!("could not open {url}: {e}"))
}

#[cfg(not(windows))]
pub fn open_url(_url: &str) -> Result<(), String> {
    Err("opening a web page is only supported on Windows".to_string())
}

#[cfg(windows)]
mod win {
    use super::HOTKEY_PRESSED;
    use std::sync::atomic::Ordering;

    #[repr(C)]
    struct FileTime {
        low: u32,
        high: u32,
    }

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
        fn RegisterHotKey(hwnd: isize, id: i32, modifiers: u32, vk: u32) -> i32;
        fn GetMessageW(msg: *mut Msg, hwnd: isize, min: u32, max: u32) -> i32;
    }

    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(h: isize, creation: *mut FileTime, exit: *mut FileTime, kernel: *mut FileTime, user: *mut FileTime) -> i32;
    }

    #[link(name = "advapi32")]
    extern "system" {
        fn RegOpenKeyExW(hkey: isize, sub_key: *const u16, options: u32, sam: u32, out: *mut isize) -> i32;
        fn RegSetValueExW(hkey: isize, name: *const u16, reserved: u32, ty: u32, data: *const u8, len: u32) -> i32;
        fn RegDeleteValueW(hkey: isize, name: *const u16) -> i32;
        fn RegCloseKey(hkey: isize) -> i32;
    }

    fn wide(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    pub fn start_mute_hotkey() {
        const MOD_CONTROL: u32 = 0x0002;
        const MOD_SHIFT: u32 = 0x0004;
        const MOD_NOREPEAT: u32 = 0x4000;
        const WM_HOTKEY: u32 = 0x0312;
        let spawned = std::thread::Builder::new().name("clearnai-hotkey".into()).spawn(|| unsafe {
            // Registered on this thread, so the WM_HOTKEY message arrives in this thread's own queue.
            if RegisterHotKey(0, 1, MOD_CONTROL | MOD_SHIFT | MOD_NOREPEAT, b'M' as u32) == 0 {
                crate::log_error!("[clearnairt] could not register the Ctrl+Shift+M mute hotkey (another app may own it)");
                return;
            }
            let mut msg: Msg = std::mem::zeroed();
            while GetMessageW(&mut msg, 0, 0, 0) > 0 {
                if msg.message == WM_HOTKEY {
                    HOTKEY_PRESSED.store(true, Ordering::Relaxed);
                }
            }
        });
        if let Err(e) = spawned {
            crate::log_error!("[clearnairt] could not start the hotkey thread: {e}");
        }
    }

    pub fn process_cpu_time_100ns() -> Option<u64> {
        let mut c = FileTime { low: 0, high: 0 };
        let mut e = FileTime { low: 0, high: 0 };
        let mut k = FileTime { low: 0, high: 0 };
        let mut u = FileTime { low: 0, high: 0 };
        let ok = unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        if ok == 0 {
            return None;
        }
        let join = |t: &FileTime| ((t.high as u64) << 32) | t.low as u64;
        Some(join(&k) + join(&u))
    }

    pub fn set_autostart(enable: bool) -> Result<(), String> {
        const HKEY_CURRENT_USER: isize = 0x8000_0001u32 as i32 as isize;
        const KEY_SET_VALUE: u32 = 0x0002;
        const REG_SZ: u32 = 1;
        const ERROR_FILE_NOT_FOUND: i32 = 2;
        let sub = wide("Software\\Microsoft\\Windows\\CurrentVersion\\Run");
        let name = wide("ClearNAI");
        let mut key: isize = 0;
        let rc = unsafe { RegOpenKeyExW(HKEY_CURRENT_USER, sub.as_ptr(), 0, KEY_SET_VALUE, &mut key) };
        if rc != 0 {
            return Err(format!("could not open the Windows startup registry key (error {rc})"));
        }
        let result = if enable {
            match std::env::current_exe() {
                Ok(exe) => {
                    let value = wide(&super::run_value(&exe));
                    let rc = unsafe { RegSetValueExW(key, name.as_ptr(), 0, REG_SZ, value.as_ptr() as *const u8, (value.len() * 2) as u32) };
                    if rc == 0 { Ok(()) } else { Err(format!("could not register ClearNAI to start with Windows (error {rc})")) }
                }
                Err(e) => Err(format!("could not find the ClearNAI exe path: {e}")),
            }
        } else {
            let rc = unsafe { RegDeleteValueW(key, name.as_ptr()) };
            if rc == 0 || rc == ERROR_FILE_NOT_FOUND { Ok(()) } else { Err(format!("could not remove ClearNAI from Windows startup (error {rc})")) }
        };
        unsafe { RegCloseKey(key) };
        result
    }
}

#[cfg(windows)]
pub use win::{process_cpu_time_100ns, set_autostart, start_mute_hotkey};

#[cfg(not(windows))]
pub fn start_mute_hotkey() {}

#[cfg(not(windows))]
pub fn process_cpu_time_100ns() -> Option<u64> {
    None
}

#[cfg(not(windows))]
pub fn set_autostart(_enable: bool) -> Result<(), String> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn run_value_quotes_the_exe_path() {
        let v = run_value(Path::new(r"C:\Users\Dev eloper\AppData\Local\ClearNAI\clearnairt.exe"));
        assert!(v.starts_with('"') && v.ends_with("\" --background"));
        assert!(v.contains("Dev eloper"));
        assert!(v.contains("clearnairt.exe"));
    }

    #[test]
    fn cpu_percent_is_a_share_of_all_cores() {
        // 1 s of CPU in 1 s of wall time on 4 cores = 25 %.
        assert!((cpu_percent(10_000_000, 10_000_000, 4) - 25.0).abs() < 0.01);
        // 4 s of CPU in 1 s of wall time on 4 cores = 100 %.
        assert!((cpu_percent(40_000_000, 10_000_000, 4) - 100.0).abs() < 0.01);
        assert_eq!(cpu_percent(5, 0, 4), 0.0);
        assert_eq!(cpu_percent(5, 5, 0), 0.0);
        assert_eq!(cpu_percent(1_000_000_000, 10, 1), 100.0);
    }

    #[test]
    fn exports_write_and_copy_files() {
        let dir = std::env::temp_dir().join(format!("clearnai-sysutil-test-{}", unix_now()));
        std::fs::create_dir_all(&dir).unwrap();
        let log = dir.join("clearnai.log");
        assert!(export_log(&log, &dir).is_err());
        std::fs::write(&log, "hello").unwrap();
        let copied = export_log(&log, &dir).unwrap();
        assert_eq!(std::fs::read_to_string(copied).unwrap(), "hello");
        let t = export_text(&dir.join("transcripts"), "[00:01] hi\n").unwrap();
        assert_eq!(std::fs::read_to_string(t).unwrap(), "[00:01] hi\n");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
