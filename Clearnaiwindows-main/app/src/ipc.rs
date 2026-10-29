//! The link between the background engine (`clearnai-engine.exe`) and the window (`clearnai.exe`).
//!
//! One named pipe, newline-delimited JSON in both directions. The engine owns the audio, the tray, the hotkey,
//! autostart and `settings.json`; the window shows what the engine reports and sends it commands. Messages are
//! small (a status snapshot is well under 2 KB), so plain JSON lines are plenty and need no extra dependency.
//!
//! The pipe is created with `PIPE_REJECT_REMOTE_CLIENTS` and the default security descriptor, so only this
//! machine (and, for writing commands, the same user or an administrator) can talk to the engine.

use std::io::{self, BufRead, Write};

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::settings::Settings;
use crate::setup::CompatibilityReport;
use crate::shared::DeviceList;

/// Bumped when the messages change incompatibly, so an old window and a new engine (or the reverse) notice.
pub const PROTOCOL_VERSION: u32 = 1;

/// Order of the on/off switches in [`Status::toggles`] and [`ClientMsg::Toggle::key`]
/// (see `EngineHandles::toggles`).
pub const TOGGLE_COUNT: usize = 11;
/// Order of the small-number selectors in [`Status::presets`] and [`ClientMsg::Preset::key`]
/// (see `EngineHandles::presets`).
pub const PRESET_COUNT: usize = 4;

/// How the window should appear when it is opened or brought to the front.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShowMode {
    /// The full window.
    Full,
    /// The small always-on-top quick panel by the tray (toggles only; Expand opens the full window).
    Panel,
}

/// Which real device a [`ClientMsg::SetDevice`] changes.
pub mod role {
    pub const PHYSICAL_MIC: u8 = 0;
    pub const VIRTUAL_MIC_TARGET: u8 = 1;
    pub const VIRTUAL_SPEAKER_SOURCE: u8 = 2;
    pub const REAL_SPEAKER_OUTPUT: u8 = 3;
    pub const MONITOR: u8 = 4;
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct PctWire {
    pub p50: u64,
    pub p90: u64,
    pub p95: u64,
    pub p99: u64,
    pub max: u64,
    pub mean: u64,
    pub count: usize,
}

impl From<dsp_core::Percentiles> for PctWire {
    fn from(p: dsp_core::Percentiles) -> Self {
        Self { p50: p.p50_ns, p90: p.p90_ns, p95: p.p95_ns, p99: p.p99_ns, max: p.max_ns, mean: p.mean_ns, count: p.sample_count }
    }
}

impl From<&PctWire> for dsp_core::Percentiles {
    fn from(p: &PctWire) -> Self {
        Self { p50_ns: p.p50, p90_ns: p.p90, p95_ns: p.p95, p99_ns: p.p99, max_ns: p.max, mean_ns: p.mean, sample_count: p.count }
    }
}

/// What the engine could and could not load; fixed for the life of the engine process.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Flags {
    pub mic_bvc_available: bool,
    pub mic_bvc_reason: Option<String>,
    pub speaker_bvc_available: bool,
    pub speaker_bvc_reason: Option<String>,
    pub speaker_deep_filter_available: bool,
    pub speaker_deep_filter_reason: Option<String>,
    pub mic_deep_filter_available: bool,
    pub mic_deep_filter_reason: Option<String>,
    pub mic_bpo_v1_available: bool,
    pub mic_bpo_v1_reason: Option<String>,
    /// The Hush / bpo_v1 watchdogs exist (so "stopped responding" can be reported for them).
    pub has_hush_status: bool,
    pub has_bpo_status: bool,
    pub warnings: Vec<String>,
}

/// Sent once when the window connects.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Hello {
    pub version: u32,
    pub settings: Settings,
    pub devices: DeviceList,
    pub compat: CompatibilityReport,
    pub bvc_extraction_error: Option<String>,
    pub flags: Flags,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct AccentWire {
    pub voice: u8,
    /// 0 Off, 1 Loading, 2 Starting, 3 Active, 4 Paused, 5 Unavailable.
    pub state: u8,
    pub rtf: f32,
    pub latency_ms: f32,
    pub underruns: u32,
    pub jitter_ms: u32,
    pub dropped_input: u32,
    pub catchups: u32,
    pub message: String,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct TranscriptWire {
    /// 0 Off, 1 Loading, 2 Listening, 3 Transcribing, 4 Unavailable.
    pub state: u8,
    pub enabled: bool,
    pub message: String,
    pub rtf: f32,
    /// The engine cleared its transcript: drop what you have before adding `new_lines`.
    pub reset: bool,
    pub new_lines: Vec<(u32, String)>,
    pub partial: String,
}

/// Sent about every 250 ms: everything the window shows that changes while the app runs.
#[derive(Serialize, Deserialize, Clone, Debug, Default)]
pub struct Status {
    pub toggles: Vec<bool>,
    pub presets: Vec<u8>,
    pub level: f32,
    pub input_level: f32,
    /// `FrameCounters` fields in declaration order (see `shared::counters_to_array`).
    pub mic_counters: [u64; 8],
    pub speaker_counters: [u64; 8],
    pub mic_pct: PctWire,
    pub speaker_pct: PctWire,
    pub hush_bypassed: bool,
    pub bpo_bypassed: bool,
    pub accent: AccentWire,
    pub transcript: TranscriptWire,
    /// CPU used by the engine process, percent of the whole machine.
    pub cpu: Option<f32>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ServerMsg {
    Hello(Box<Hello>),
    Status(Box<Status>),
    /// Reply to [`ClientMsg::RefreshDevices`].
    Devices(DeviceList),
    /// The engine is quitting; the window should close too.
    Shutdown,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub enum ClientMsg {
    Toggle { key: u8, on: bool },
    Preset { key: u8, val: u8 },
    /// The window's current settings; the engine saves them (it is the only writer of `settings.json`).
    SaveSettings(Box<Settings>),
    SetDevice { role: u8, id: Option<String> },
    RefreshDevices,
    AccentVoice(u8),
    TranscriptEnabled(bool),
    TranscriptClear,
    /// Quit the engine (and so the whole app).
    Quit,
}

/// Writes one message as a JSON line and flushes.
pub fn write_msg<W: Write, T: Serialize>(w: &mut W, msg: &T) -> io::Result<()> {
    let mut line = serde_json::to_vec(msg).map_err(io::Error::other)?;
    line.push(b'\n');
    w.write_all(&line)?;
    w.flush()
}

/// Reads one message; `Ok(None)` when the other side closed the pipe cleanly.
pub fn read_msg<R: BufRead, T: DeserializeOwned>(r: &mut R) -> io::Result<Option<T>> {
    let mut line = String::new();
    if r.read_line(&mut line)? == 0 {
        return Ok(None);
    }
    serde_json::from_str(line.trim_end()).map(Some).map_err(io::Error::other)
}

pub fn pipe_name() -> String {
    // Overridable so tests never touch a real engine's pipe.
    if let Ok(name) = std::env::var("CLEARNAI_PIPE_NAME") {
        return name;
    }
    let user = std::env::var("USERNAME").unwrap_or_else(|_| "user".into());
    format!(r"\\.\pipe\ClearNAI-engine-{user}")
}

#[cfg(windows)]
mod win {
    use std::fs::{File, OpenOptions};
    use std::io;
    use std::os::windows::io::FromRawHandle;

    #[link(name = "kernel32")]
    extern "system" {
        fn PeekNamedPipe(handle: isize, buf: *mut u8, size: u32, read: *mut u32, avail: *mut u32, left: *mut u32) -> i32;
        fn CreateNamedPipeW(
            name: *const u16,
            open_mode: u32,
            pipe_mode: u32,
            max_instances: u32,
            out_buf: u32,
            in_buf: u32,
            default_timeout: u32,
            security: *const u8,
        ) -> isize;
        fn ConnectNamedPipe(handle: isize, overlapped: *mut u8) -> i32;
        fn GetLastError() -> u32;
        fn CloseHandle(handle: isize) -> i32;
    }

    const PIPE_ACCESS_DUPLEX: u32 = 3;
    const PIPE_TYPE_BYTE: u32 = 0;
    const PIPE_WAIT: u32 = 0;
    const PIPE_REJECT_REMOTE_CLIENTS: u32 = 8;
    const ERROR_PIPE_CONNECTED: u32 = 535;
    const INVALID_HANDLE_VALUE: isize = -1;

    /// Creates one instance of the engine's pipe and blocks until a window connects to it.
    pub fn accept() -> io::Result<File> {
        let name: Vec<u16> = super::pipe_name().encode_utf16().chain(std::iter::once(0)).collect();
        let handle = unsafe {
            CreateNamedPipeW(
                name.as_ptr(),
                PIPE_ACCESS_DUPLEX,
                PIPE_TYPE_BYTE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
                8,
                64 * 1024,
                64 * 1024,
                0,
                std::ptr::null(),
            )
        };
        if handle == INVALID_HANDLE_VALUE {
            return Err(io::Error::last_os_error());
        }
        let ok = unsafe { ConnectNamedPipe(handle, std::ptr::null_mut()) };
        // A client that connected between CreateNamedPipe and ConnectNamedPipe is reported as an "error".
        if ok == 0 && unsafe { GetLastError() } != ERROR_PIPE_CONNECTED {
            let err = io::Error::last_os_error();
            unsafe { CloseHandle(handle) };
            return Err(err);
        }
        Ok(unsafe { File::from_raw_handle(handle as *mut _) })
    }

    /// Bytes waiting to be read, without blocking. A blocking read would also block every write on the same pipe
    /// (synchronous handles take one lock per operation), so readers poll this first and only read what is there.
    /// `Err` once the other side has gone.
    pub fn available(file: &File) -> io::Result<u32> {
        use std::os::windows::io::AsRawHandle;
        let mut avail = 0u32;
        let ok = unsafe {
            PeekNamedPipe(file.as_raw_handle() as isize, std::ptr::null_mut(), 0, std::ptr::null_mut(), &mut avail, std::ptr::null_mut())
        };
        if ok == 0 {
            Err(io::Error::last_os_error())
        } else {
            Ok(avail)
        }
    }

    /// Opens the engine's pipe from the window side.
    pub fn connect() -> io::Result<File> {
        OpenOptions::new().read(true).write(true).open(super::pipe_name())
    }
}

#[cfg(windows)]
pub use win::{accept, connect};

/// What one poll of a [`PipeReader`] found.
#[cfg(windows)]
pub enum Polled<T> {
    Msg(T),
    /// Nothing complete yet; try again shortly.
    Empty,
    /// The other side closed the pipe (or it broke).
    Closed,
}

/// Reads JSON lines from a pipe without ever blocking (see `win::available`).
#[cfg(windows)]
pub struct PipeReader {
    file: std::fs::File,
    buf: Vec<u8>,
}

#[cfg(windows)]
impl PipeReader {
    pub fn new(file: std::fs::File) -> Self {
        Self { file, buf: Vec::new() }
    }

    pub fn poll<T: DeserializeOwned>(&mut self) -> Polled<T> {
        use std::io::Read;
        loop {
            if let Some(pos) = self.buf.iter().position(|&b| b == 10) {
                let line: Vec<u8> = self.buf.drain(..=pos).collect();
                return match serde_json::from_slice(&line[..line.len() - 1]) {
                    Ok(m) => Polled::Msg(m),
                    // A line we cannot understand (newer/older peer): skip it rather than drop the connection.
                    Err(_) => continue,
                };
            }
            match win::available(&self.file) {
                Err(_) => return Polled::Closed,
                Ok(0) => return Polled::Empty,
                Ok(n) => {
                    let mut tmp = vec![0u8; (n as usize).min(64 * 1024)];
                    match self.file.read(&mut tmp) {
                        Ok(0) | Err(_) => return Polled::Closed,
                        Ok(got) => self.buf.extend_from_slice(&tmp[..got]),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn messages_round_trip_through_json_lines() {
        let mut buf = Vec::new();
        write_msg(&mut buf, &ClientMsg::Toggle { key: 3, on: true }).unwrap();
        write_msg(&mut buf, &ClientMsg::SetDevice { role: role::MONITOR, id: Some("x".into()) }).unwrap();
        let mut r = Cursor::new(buf);
        match read_msg::<_, ClientMsg>(&mut r).unwrap().unwrap() {
            ClientMsg::Toggle { key, on } => assert!(key == 3 && on),
            other => panic!("wrong message {other:?}"),
        }
        assert!(matches!(read_msg::<_, ClientMsg>(&mut r).unwrap(), Some(ClientMsg::SetDevice { role: 4, .. })));
        assert!(read_msg::<_, ClientMsg>(&mut r).unwrap().is_none(), "end of stream is None, not an error");
    }

    #[test]
    fn status_survives_the_wire_and_stays_small() {
        let st = Status {
            toggles: vec![true; TOGGLE_COUNT],
            presets: vec![1; PRESET_COUNT],
            transcript: TranscriptWire { new_lines: vec![(5, "hello there".into())], ..Default::default() },
            ..Default::default()
        };
        let mut buf = Vec::new();
        write_msg(&mut buf, &ServerMsg::Status(Box::new(st))).unwrap();
        assert!(buf.len() < 2048, "status line is {} bytes", buf.len());
        let back = read_msg::<_, ServerMsg>(&mut Cursor::new(buf)).unwrap().unwrap();
        match back {
            ServerMsg::Status(s) => assert!(s.toggles.len() == TOGGLE_COUNT && s.transcript.new_lines[0].1 == "hello there"),
            other => panic!("wrong message {other:?}"),
        }
    }

    /// A real pipe: the server writes while the client's reader is idle and vice versa, both ways at once. A blocking
    /// read would stall every write on the same pipe; this proves the polling reader does not.
    #[cfg(windows)]
    #[test]
    fn real_pipe_carries_messages_both_ways_without_stalling() {
        use std::io::Write;
        use std::time::{Duration, Instant};
        std::env::set_var("CLEARNAI_PIPE_NAME", format!(r"\\.\pipe\ClearNAI-test-{}", std::process::id()));
        let server = std::thread::spawn(|| {
            let pipe = accept().unwrap();
            let mut reader = PipeReader::new(pipe.try_clone().unwrap());
            let mut writer = pipe;
            // Write first while our reader has nothing to read (it would block forever with a plain read).
            write_msg(&mut writer, &ServerMsg::Shutdown).unwrap();
            let t0 = Instant::now();
            loop {
                match reader.poll::<ClientMsg>() {
                    Polled::Msg(ClientMsg::Toggle { key, on }) => {
                        assert!(key == 7 && on);
                        return;
                    }
                    Polled::Msg(_) => {}
                    Polled::Empty => std::thread::sleep(Duration::from_millis(5)),
                    Polled::Closed => panic!("client went away"),
                }
                assert!(t0.elapsed() < Duration::from_secs(10), "timed out");
            }
        });
        // Client: retry until the server instance exists.
        let t0 = Instant::now();
        let file = loop {
            match connect() {
                Ok(f) => break f,
                Err(_) if t0.elapsed() < Duration::from_secs(10) => std::thread::sleep(Duration::from_millis(20)),
                Err(e) => panic!("could not connect: {e}"),
            }
        };
        let mut reader = PipeReader::new(file.try_clone().unwrap());
        let mut writer = file;
        let mut got = false;
        while !got {
            match reader.poll::<ServerMsg>() {
                Polled::Msg(ServerMsg::Shutdown) => got = true,
                Polled::Msg(_) => {}
                Polled::Empty => std::thread::sleep(Duration::from_millis(5)),
                Polled::Closed => panic!("server went away"),
            }
        }
        write_msg(&mut writer, &ClientMsg::Toggle { key: 7, on: true }).unwrap();
        writer.flush().unwrap();
        server.join().unwrap();
        // Server end dropped: the client's reader must report Closed, not hang.
        let t1 = Instant::now();
        loop {
            match reader.poll::<ServerMsg>() {
                Polled::Closed => break,
                _ => std::thread::sleep(Duration::from_millis(5)),
            }
            assert!(t1.elapsed() < Duration::from_secs(5), "client never noticed the closed pipe");
        }
    }

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        let mut r = Cursor::new(b"not json\n".to_vec());
        assert!(read_msg::<_, ClientMsg>(&mut r).is_err());
    }
}
