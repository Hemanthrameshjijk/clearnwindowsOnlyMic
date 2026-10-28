//! Live transcript: the 10 ms mic thread only copies frames into a lock-free ring (never modifies the audio,
//! never allocates); a worker thread downsamples to 16 kHz, cuts utterances, runs the speech model and publishes
//! text lines to the GUI (and appends them to a transcript file).
//!
//! ```text
//!  mic thread: TranscriptStage::process --ring--> worker: 48k -> 16k -> Segmenter -> AsrModel -> lines
//! ```

use crate::model::{find_models_dir, AsrModel};
use crate::segmenter::{Event, Segmenter};
use dsp_core::{Consumer, Producer, RingBuffer, Stage, FRAME_SAMPLES};
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tse_bsrnn::causal_resampler::CausalDownsampler3x;

const RESAMPLER_TAPS: usize = 97;
const MAX_LINES: usize = 500;
const RING_SECONDS: usize = 30;

const S_OFF: u8 = 0;
const S_LOADING: u8 = 1;
const S_LISTENING: u8 = 2;
const S_TRANSCRIBING: u8 = 3;
const S_UNAVAILABLE: u8 = 4;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TranscriptState {
    Off,
    Loading,
    Listening,
    Transcribing,
    Unavailable,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TranscriptLine {
    /// Seconds since the app started.
    pub at_secs: u32,
    pub text: String,
}

#[derive(Clone, Debug)]
pub struct TranscriptStatus {
    pub state: TranscriptState,
    pub message: String,
    pub rtf: f32,
    pub line_count: usize,
}

impl TranscriptStatus {
    pub fn line(&self) -> String {
        match self.state {
            TranscriptState::Off => "Transcript: off".into(),
            TranscriptState::Loading => "Transcript: loading speech model...".into(),
            TranscriptState::Listening => "Transcript: listening".into(),
            TranscriptState::Transcribing => format!("Transcript: transcribing (RTF {:.2})", self.rtf),
            TranscriptState::Unavailable => format!("Transcript unavailable - {}", self.message),
        }
    }
}

pub struct TranscriptConfig {
    pub dll: PathBuf,
    pub models_dir: Option<PathBuf>,
    pub threads: usize,
    /// Directory for `transcript-<unix seconds>.txt`; `None` = do not save.
    pub save_dir: Option<PathBuf>,
    pub initial_enabled: bool,
}

impl TranscriptConfig {
    pub fn new(dll: PathBuf) -> Self {
        let threads = std::env::var("CLEARNAI_ASR_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(1);
        Self { dll, models_dir: None, threads, save_dir: None, initial_enabled: false }
    }
}

#[derive(Default)]
struct Data {
    lines: Vec<TranscriptLine>,
    partial: String,
    message: String,
    save_path: Option<PathBuf>,
}

struct Shared {
    enabled: AtomicBool,
    state: AtomicU8,
    stop: AtomicBool,
    rtf_milli: AtomicU32,
    version: AtomicU32,
    clear_requested: AtomicBool,
    started: Instant,
    data: Mutex<Data>,
}

/// GUI-side handle (cheap to clone).
#[derive(Clone)]
pub struct TranscriptHandle(Arc<Shared>);

impl TranscriptHandle {
    /// Inert handle for builds/paths where the engine is not running.
    pub fn unavailable(reason: &str) -> Self {
        let sh = new_shared(false);
        sh.state.store(S_UNAVAILABLE, Relaxed);
        sh.data.lock().unwrap().message = reason.to_string();
        Self(Arc::new(sh))
    }

    pub fn set_enabled(&self, on: bool) {
        self.0.enabled.store(on, Relaxed);
    }

    pub fn enabled(&self) -> bool {
        self.0.enabled.load(Relaxed)
    }

    pub fn status(&self) -> TranscriptStatus {
        let d = self.0.data.lock().unwrap();
        let state = match self.0.state.load(Relaxed) {
            S_LOADING => TranscriptState::Loading,
            S_LISTENING => TranscriptState::Listening,
            S_TRANSCRIBING => TranscriptState::Transcribing,
            S_UNAVAILABLE => TranscriptState::Unavailable,
            _ => TranscriptState::Off,
        };
        TranscriptStatus { state, message: d.message.clone(), rtf: self.0.rtf_milli.load(Relaxed) as f32 / 1000.0, line_count: d.lines.len() }
    }

    /// Last `max` finished lines plus the in-progress (partial) text.
    pub fn snapshot(&self, max: usize) -> (Vec<TranscriptLine>, String) {
        let d = self.0.data.lock().unwrap();
        let from = d.lines.len().saturating_sub(max);
        (d.lines[from..].to_vec(), d.partial.clone())
    }

    /// Whole transcript as plain text (for copy).
    pub fn full_text(&self) -> String {
        let d = self.0.data.lock().unwrap();
        d.lines.iter().map(|l| format!("[{}] {}", fmt_time(l.at_secs), l.text)).collect::<Vec<_>>().join("\n")
    }

    pub fn clear(&self) {
        self.0.clear_requested.store(true, Relaxed);
        let mut d = self.0.data.lock().unwrap();
        d.lines.clear();
        d.partial.clear();
        self.0.version.fetch_add(1, Relaxed);
    }

    /// Changes whenever lines or the partial text change (lets the GUI skip redundant work).
    pub fn version(&self) -> u32 {
        self.0.version.load(Relaxed)
    }

    pub fn save_path(&self) -> Option<PathBuf> {
        self.0.data.lock().unwrap().save_path.clone()
    }
}

pub fn fmt_time(secs: u32) -> String {
    format!("{:02}:{:02}", secs / 60, secs % 60)
}

fn new_shared(enabled: bool) -> Shared {
    Shared {
        enabled: AtomicBool::new(enabled),
        state: AtomicU8::new(S_OFF),
        stop: AtomicBool::new(false),
        rtf_milli: AtomicU32::new(0),
        version: AtomicU32::new(0),
        clear_requested: AtomicBool::new(false),
        started: Instant::now(),
        data: Mutex::new(Data::default()),
    }
}

/// Mic-thread end. A pure tap: the frame is left exactly as it is.
pub struct TranscriptStage {
    sh: Arc<Shared>,
    prod: Producer<f32>,
}

impl Drop for TranscriptStage {
    fn drop(&mut self) {
        self.sh.stop.store(true, Relaxed);
    }
}

impl Stage for TranscriptStage {
    fn name(&self) -> &'static str {
        "Transcript"
    }

    fn process(&mut self, frame: &mut [f32]) {
        if !self.sh.enabled.load(Relaxed) || frame.len() != FRAME_SAMPLES {
            return;
        }
        let st = self.sh.state.load(Relaxed);
        if st == S_LISTENING || st == S_TRANSCRIBING {
            for &s in frame.iter() {
                let _ = self.prod.push(s); // ring full = worker is busy; dropping is fine for a transcript
            }
        }
    }
}

pub fn start(cfg: TranscriptConfig) -> (TranscriptStage, TranscriptHandle) {
    let sh = Arc::new(new_shared(cfg.initial_enabled));
    let (prod, cons) = RingBuffer::<f32>::new(48_000 * RING_SECONDS);
    let w_sh = sh.clone();
    if let Err(e) = std::thread::Builder::new().name("clearnai-transcript-worker".into()).spawn(move || worker(w_sh, cons, cfg)) {
        sh.state.store(S_UNAVAILABLE, Relaxed);
        sh.data.lock().unwrap().message = format!("could not start worker thread: {e}");
    }
    (TranscriptStage { sh: sh.clone(), prod }, TranscriptHandle(sh))
}

fn fail(sh: &Shared, msg: String) {
    sh.data.lock().unwrap().message = msg;
    sh.state.store(S_UNAVAILABLE, Relaxed);
}

fn publish_line(sh: &Shared, text: String, save_dir: &Option<PathBuf>, session_start_unix: u64) {
    let at = sh.started.elapsed().as_secs() as u32;
    let mut d = sh.data.lock().unwrap();
    if let Some(dir) = save_dir {
        let path = d.save_path.clone().unwrap_or_else(|| dir.join(format!("transcript-{session_start_unix}.txt")));
        let _ = std::fs::create_dir_all(dir);
        if let Ok(mut f) = std::fs::OpenOptions::new().create(true).append(true).open(&path) {
            let _ = writeln!(f, "[{}] {}", fmt_time(at), text);
            d.save_path = Some(path);
        }
    }
    d.lines.push(TranscriptLine { at_secs: at, text });
    if d.lines.len() > MAX_LINES {
        d.lines.remove(0);
    }
    d.partial.clear();
    sh.version.fetch_add(1, Relaxed);
}

/// Below-normal priority: the transcript must never compete with the mic/BVC/Accent threads for a core.
#[cfg(windows)]
fn lower_thread_priority() {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(h: isize, p: i32) -> i32;
    }
    const THREAD_PRIORITY_BELOW_NORMAL: i32 = -1;
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_BELOW_NORMAL) };
}
#[cfg(not(windows))]
fn lower_thread_priority() {}

fn worker(sh: Arc<Shared>, mut cons: Consumer<f32>, cfg: TranscriptConfig) {
    lower_thread_priority();
    let mut model: Option<AsrModel> = None;
    let mut down = CausalDownsampler3x::new(RESAMPLER_TAPS);
    let mut seg = Segmenter::new();
    let session_start_unix = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0);
    let mut chunk = vec![0.0f32; FRAME_SAMPLES];
    let mut was_enabled = false;

    while !sh.stop.load(Relaxed) {
        if !sh.enabled.load(Relaxed) {
            if was_enabled {
                was_enabled = false;
                seg.reset();
                sh.data.lock().unwrap().partial.clear();
                sh.version.fetch_add(1, Relaxed);
            }
            if sh.state.load(Relaxed) != S_UNAVAILABLE {
                sh.state.store(S_OFF, Relaxed);
            }
            while cons.pop().is_ok() {}
            std::thread::sleep(std::time::Duration::from_millis(100));
            continue;
        }
        if sh.state.load(Relaxed) == S_UNAVAILABLE {
            std::thread::sleep(std::time::Duration::from_millis(200));
            continue;
        }
        if model.is_none() {
            sh.state.store(S_LOADING, Relaxed);
            let dir = cfg.models_dir.clone().or_else(find_models_dir);
            let Some(dir) = dir else {
                fail(&sh, "speech model not found (put asr_models next to the app, or set ASR_MODELS_DIR)".into());
                continue;
            };
            if let Err(e) = accent_onnx::ort_util::init_ort(&cfg.dll) {
                fail(&sh, format!("onnxruntime: {e}"));
                continue;
            }
            match AsrModel::load(&dir, cfg.threads) {
                Ok(m) => model = Some(m),
                Err(e) => {
                    fail(&sh, e);
                    continue;
                }
            }
            while cons.pop().is_ok() {}
        }
        if !was_enabled {
            was_enabled = true;
            seg.reset();
            down = CausalDownsampler3x::new(RESAMPLER_TAPS);
        }
        if sh.clear_requested.swap(false, Relaxed) {
            seg.reset();
        }
        sh.state.store(if seg.in_speech() { S_TRANSCRIBING } else { S_LISTENING }, Relaxed);

        if cons.slots() < FRAME_SAMPLES {
            std::thread::sleep(std::time::Duration::from_millis(10));
            continue;
        }
        for s in chunk.iter_mut() {
            *s = cons.pop().unwrap_or(0.0);
        }
        let x16 = down.push(&chunk);
        let events = seg.push(&x16);
        let n = events.len();
        for (i, ev) in events.into_iter().enumerate() {
            let (audio, is_final) = match ev {
                Event::Final(a) => (a, true),
                // A newer event for the same segment makes an older partial pointless.
                Event::Partial(a) if i + 1 == n => (a, false),
                Event::Partial(_) => continue,
            };
            let Some(m) = model.as_mut() else { continue };
            let t0 = Instant::now();
            match m.transcribe(&audio) {
                Ok(text) => {
                    let rtf = t0.elapsed().as_secs_f32() / (audio.len() as f32 / 16_000.0);
                    sh.rtf_milli.store((rtf * 1000.0) as u32, Relaxed);
                    if is_final {
                        if !text.is_empty() {
                            publish_line(&sh, text, &cfg.save_dir, session_start_unix);
                        } else {
                            sh.data.lock().unwrap().partial.clear();
                            sh.version.fetch_add(1, Relaxed);
                        }
                    } else {
                        sh.data.lock().unwrap().partial = text;
                        sh.version.fetch_add(1, Relaxed);
                    }
                }
                Err(e) => sh.data.lock().unwrap().message = e,
            }
        }
    }
}
