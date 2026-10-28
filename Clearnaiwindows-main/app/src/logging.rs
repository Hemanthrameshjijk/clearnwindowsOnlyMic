//! Minimal file-backed logger for this app crate only.
//!
//! With `#![windows_subsystem = "windows"]` (see `main.rs`) the process has
//! no console, so `eprintln!`/`println!` output is simply never seen by
//! anyone. This module replaces those calls (in `main.rs`, `gui.rs`,
//! `engine.rs`, `setup.rs`, `settings.rs` only - other crates' own
//! `eprintln!`s are a known, unfixed gap, see the top-level report) with a
//! tiny appender that writes timestamped lines to
//! `<app_data_dir>/clearnai.log`.
//!
//! Deliberately not the `log` crate + a custom `Log` impl: a couple of thin
//! wrapper functions plus two macros is less code than wiring up a facade
//! crate for a single-binary app with no library consumers of the log
//! output.

use std::collections::HashMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{sync_channel, Receiver, SyncSender};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};

static LOG_PATH: OnceLock<Option<PathBuf>> = OnceLock::new();

/// Past this size the file is truncated back to empty before the next line
/// is written, so it never grows past roughly this bound. (The writer
/// thread reopens the file with `truncate(true)`: `set_len(0)` on the old
/// append-mode handle fails with "Access is denied" on Windows, which is why
/// a previous version of this cap never worked - real logs reached 200+ MB.)
const MAX_LOG_BYTES: u64 = 20 * 1024 * 1024;

/// Lines waiting for the writer thread. When it is full (disk stalled) new
/// lines are dropped: a log line is never worth stalling an audio thread.
const QUEUE_LINES: usize = 1024;

/// The same message is written at most once per this window (see `repeat_count`).
const REPEAT_WINDOW: Duration = Duration::from_secs(1);

/// Bound on the number of distinct messages tracked for de-duplication.
const MAX_TRACKED_MESSAGES: usize = 128;

/// Real-hardware finding: some of this app's realtime audio threads (mic
/// capture, hardware render) call `log_error!` on *every* dropped/underrun
/// frame - if a stall starts, that's once every ~10ms, forever, with no
/// automatic recovery. Two earlier versions of this module made that
/// genuinely dangerous:
///
/// 1. Reopening the log file on every call, then (after the first fix)
///    writing to it synchronously under one global mutex. Disk I/O on a
///    realtime audio thread can take longer than the 10ms frame budget once
///    the file is large, which causes *more* drops, which logs more - a
///    runaway feedback loop that was measured to cause audible glitching
///    (the mic capture ring was ~95% full for two hours).
/// 2. De-duplicating only *consecutive* identical messages. Mic capture,
///    mic pipeline and speaker pipeline log different messages in
///    alternation, so nothing was ever "consecutive" and every line was
///    written.
///
/// Now `write_line` does only a short in-memory per-message de-duplication
/// and a non-blocking hand-off to a background writer thread, which does all
/// the file I/O. A realtime thread never touches the disk.
struct Logger {
    tx: SyncSender<String>,
    seen: Mutex<HashMap<String, (Instant, u64)>>,
}

static LOGGER: OnceLock<Option<Logger>> = OnceLock::new();

/// Must be called once, as the very first thing in `main()`, before any
/// other code in this crate might log. `dir` is `setup::app_data_dir()`.
pub fn init(dir: &Path) {
    let _ = LOG_PATH.set(Some(dir.join("clearnai.log")));
}

fn timestamp() -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("{}.{:03}", now.as_secs(), now.subsec_millis())
}

fn logger() -> Option<&'static Logger> {
    LOGGER
        .get_or_init(|| {
            let path = LOG_PATH.get().and_then(|p| p.clone())?;
            let (tx, rx) = sync_channel(QUEUE_LINES);
            std::thread::Builder::new()
                .name("clearnai-log-writer".into())
                .spawn(move || writer(&path, rx, MAX_LOG_BYTES))
                .ok()?;
            Some(Logger { tx, seen: Mutex::new(HashMap::new()) })
        })
        .as_ref()
}

/// `None`: this message was already written less than `REPEAT_WINDOW` ago
/// (counted, not written). `Some(n)`: write it now; `n` identical messages
/// were suppressed since the last time it was written.
fn repeat_count(seen: &mut HashMap<String, (Instant, u64)>, msg: &str, now: Instant) -> Option<u64> {
    match seen.get_mut(msg) {
        Some((last, suppressed)) if now.duration_since(*last) < REPEAT_WINDOW => {
            *suppressed += 1;
            None
        }
        Some((last, suppressed)) => {
            *last = now;
            Some(std::mem::take(suppressed))
        }
        None => {
            if seen.len() >= MAX_TRACKED_MESSAGES {
                seen.clear();
            }
            seen.insert(msg.to_string(), (now, 0));
            Some(0)
        }
    }
}

/// Owns the file. Everything queued is written in one `write_all`.
fn writer(path: &Path, rx: Receiver<String>, max_bytes: u64) {
    let open = |truncate: bool| {
        let mut o = OpenOptions::new();
        o.create(true);
        if truncate {
            o.write(true).truncate(true);
        } else {
            o.append(true);
        }
        o.open(path).ok()
    };
    let mut file = open(false);
    let mut size = file.as_ref().and_then(|f| f.metadata().ok()).map_or(0, |m| m.len());
    let mut batch = String::new();
    while let Ok(first) = rx.recv() {
        batch.clear();
        batch.push_str(&first);
        batch.push('\n');
        while let Ok(more) = rx.try_recv() {
            batch.push_str(&more);
            batch.push('\n');
        }
        if size >= max_bytes {
            file = open(true);
            size = 0;
            batch.insert_str(0, &format!("[{}] (log file exceeded {max_bytes} bytes; truncated)\n", timestamp()));
        }
        if let Some(f) = file.as_mut() {
            if f.write_all(batch.as_bytes()).is_ok() {
                size += batch.len() as u64;
            }
        }
    }
}

/// Queues one line for the log file, unless `init` was never called (then
/// this is a no-op). Never blocks on disk I/O and never waits for a full
/// queue - logging must never itself become a source of realtime-audio-thread
/// stalls. Repeats of the same message within a second are collapsed.
pub fn write_line(level: &str, msg: &str) {
    let Some(logger) = logger() else { return };
    let repeated = {
        let mut seen = logger.seen.lock().unwrap_or_else(|e| e.into_inner());
        match repeat_count(&mut seen, msg, Instant::now()) {
            Some(n) => n,
            None => return,
        }
    };
    let line = if repeated > 0 {
        format!("[{}] [{level}] {msg} (+{repeated} identical since the previous line)", timestamp())
    } else {
        format!("[{}] [{level}] {msg}", timestamp())
    };
    let _ = logger.tx.try_send(line);
}

#[macro_export]
macro_rules! log_info {
    ($($arg:tt)*) => {
        $crate::logging::write_line("INFO", &format!($($arg)*))
    };
}

#[macro_export]
macro_rules! log_error {
    ($($arg:tt)*) => {
        $crate::logging::write_line("ERROR", &format!($($arg)*))
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_messages_are_each_collapsed() {
        // The failure that defeated the old logger: A, B, A, B, ... from different threads.
        let mut seen = HashMap::new();
        let t0 = Instant::now();
        let mut written = 0;
        for i in 0..1000u32 {
            let msg = if i % 2 == 0 { "mic capture ring buffer full" } else { "output ring buffer full" };
            if repeat_count(&mut seen, msg, t0 + Duration::from_millis(i as u64)).is_some() {
                written += 1;
            }
        }
        assert_eq!(written, 2, "1 s of interleaved repeats must produce one line per message");
        // After the window the next occurrence is written with the suppressed count.
        assert_eq!(repeat_count(&mut seen, "output ring buffer full", t0 + Duration::from_millis(1500)), Some(499));
    }

    #[test]
    fn writer_truncates_at_the_cap_and_keeps_going() {
        let dir = std::env::temp_dir().join(format!("clearnai-log-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.log");
        std::fs::write(&path, vec![b'@'; 5000]).unwrap(); // already over the cap, like the 200 MB real log
        let (tx, rx) = sync_channel(QUEUE_LINES);
        let p = path.clone();
        let h = std::thread::spawn(move || writer(&p, rx, 1000));
        for i in 0..200 {
            tx.send(format!("line {i}")).unwrap();
        }
        drop(tx);
        h.join().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        assert!(text.len() < 2500, "file stayed bounded, got {} bytes", text.len());
        assert!(!text.contains('@'), "the old oversized content is gone");
        assert!(text.contains("line 199"), "the newest line is present");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
