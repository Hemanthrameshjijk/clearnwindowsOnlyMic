//! Real-time placement of the accent engine: the 10 ms mic pipeline thread only touches two lock-free rings;
//! all model work happens on a dedicated worker thread.
//!
//! ```text
//!  mic thread (480 @ 48 kHz, no allocation)                worker thread (50 ms pushes)
//!  AccentStage::process --in ring--> 48k -> 16k (causal) -> leveler -> AccentStream (mel, converter,
//!        ^                                                              vocoder) -> 24k -> 48k (causal)
//!        +-- jitter buffer <--out ring-- 480 @ 48 kHz  <-----------------------------------+
//! ```
//!
//! Safety net: if the output ring runs dry, or the worker's real-time factor stays above 0.9 for 2 s, the stage
//! crossfades (20 ms) to the incoming (BVC-only) signal, keeps the engine running, and reports a "paused"
//! state; it returns to the accent output once the worker has caught up and the jitter buffer has refilled.

use crate::resample::CausalUpsampler2x;
use crate::{find_models_dir, init_runtime, AccentStream, VOICE_NAMES};
use dsp_core::{Consumer, Producer, RingBuffer, Stage, FRAME_SAMPLES};
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU8, Ordering::Relaxed};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use crate::causal_resampler::CausalDownsampler3x;

/// Worker push size: 100 ms of 48 kHz audio (-> 1600 samples at 16 kHz = 8 mel frames). 100 ms needs ~35% less CPU
/// than 50 ms (RTF 0.45 vs 0.69 at 4 threads) for +25 ms delay, which keeps it real-time when other apps load the PC.
const PUSH48: usize = 4800;
const PUSH16: usize = 1600;
const PUSH_MS: f32 = 100.0;
/// Input backlog (48 kHz samples) above which the worker skips ahead instead of falling further behind.
const MAX_BACKLOG48: usize = 4 * PUSH48;
const RING_SECONDS: usize = 1;
const RAMP_SAMPLES: f32 = 1920.0; // 40 ms crossfade
const REPRIME_HOLDOFF_FRAMES: u32 = 100; // 1 s of dry after an underrun before trusting the accent again
const SLOW_RTF: f32 = 0.9;
const SLOW_WINDOW: usize = 20; // pushes = 2 s
// Must sit clearly above the normal running RTF (0.55-0.8 on this PC) or a paused stage could never recover.
const RECOVER_RTF: f32 = 0.85;
const RECOVER_WINDOW: usize = 20; // pushes = 2 s
const LEVEL_TARGET_RMS: f32 = 0.06;
const RESAMPLER_TAPS: usize = 97;

/// 0 = off, 1..=6 = `VOICE_NAMES[voice - 1]`.
pub const ACCENT_OFF: u8 = 0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccentState {
    Off,
    Loading,
    Starting,
    Active,
    /// Crossfaded back to the BVC-only signal (worker too slow / output ran dry).
    Paused,
    Unavailable,
}

#[derive(Clone, Debug)]
pub struct AccentStatus {
    pub voice: u8,
    pub state: AccentState,
    /// Worker real-time factor (compute time / audio time), smoothed over ~2 s. 0 until measured.
    pub rtf: f32,
    /// Estimated total delay the accent stage adds, ms (model look-ahead + resamplers + push + jitter + compute).
    pub latency_ms: f32,
    pub underruns: u32,
    /// Current jitter buffer (grows after underruns, up to 300 ms).
    pub jitter_ms: u32,
    pub dropped_input: u32,
    pub catchups: u32,
    pub message: String,
}

impl AccentStatus {
    pub fn line(&self) -> String {
        match self.state {
            AccentState::Off => "Accent: off".into(),
            AccentState::Loading => "Accent: loading model...".into(),
            AccentState::Starting => "Accent: starting...".into(),
            AccentState::Active => {
                let mut l = format!("Accent: on | RTF {:.2} | +{:.0} ms", self.rtf, self.latency_ms);
                if self.underruns + self.catchups + self.dropped_input > 0 {
                    l += &format!(" | glitches: {} underruns, {} skips", self.underruns, self.catchups + self.dropped_input.min(1));
                }
                l
            }
            AccentState::Paused => format!("Accent paused - CPU too slow (RTF {:.2})", self.rtf),
            AccentState::Unavailable => format!("Accent unavailable: {}", self.message),
        }
    }
}

// worker states
const W_IDLE: u8 = 0;
const W_LOADING: u8 = 1;
const W_READY: u8 = 2;
const W_FAILED: u8 = 3;
// mic-side states
const M_IDLE: u8 = 0;
const M_STARTING: u8 = 1;
const M_ACTIVE: u8 = 2;
const M_PAUSED: u8 = 3;

struct Shared {
    voice: AtomicU8,
    worker: AtomicU8,
    mic: AtomicU8,
    generation: AtomicU32,
    slow: AtomicBool,
    stop: AtomicBool,
    rtf_milli: AtomicU32,
    rtf_conv_milli: AtomicU32,
    rtf_voc_milli: AtomicU32,
    latency_x10: AtomicU32,
    jitter_ms: AtomicU32,
    underruns: AtomicU32,
    dropped_input: AtomicU32,
    catchups: AtomicU32,
    error: Mutex<String>, // never locked on the mic thread
}

/// GUI-side handle: choose the voice, read the status.
#[derive(Clone)]
pub struct AccentHandle(Arc<Shared>);

impl AccentHandle {
    /// A handle for builds where accent conversion cannot run at all (reports `Unavailable` with `reason`).
    pub fn unavailable(reason: &str) -> Self {
        let (_stage, handle) = start_inner(None, reason);
        handle
    }
    pub fn set_voice(&self, voice: u8) {
        self.0.voice.store(voice.min(VOICE_NAMES.len() as u8), Relaxed);
    }
    pub fn voice(&self) -> u8 {
        self.0.voice.load(Relaxed)
    }
    pub fn status(&self) -> AccentStatus {
        let s = &self.0;
        let voice = s.voice.load(Relaxed);
        let w = s.worker.load(Relaxed);
        let state = if w == W_FAILED {
            AccentState::Unavailable
        } else if voice == ACCENT_OFF {
            AccentState::Off
        } else if w == W_LOADING || w == W_IDLE {
            AccentState::Loading
        } else {
            match s.mic.load(Relaxed) {
                M_ACTIVE => AccentState::Active,
                M_PAUSED => AccentState::Paused,
                _ => AccentState::Starting,
            }
        };
        AccentStatus {
            voice,
            state,
            rtf: s.rtf_milli.load(Relaxed) as f32 / 1000.0,
            latency_ms: s.latency_x10.load(Relaxed) as f32 / 10.0,
            underruns: s.underruns.load(Relaxed),
            jitter_ms: s.jitter_ms.load(Relaxed),
            dropped_input: s.dropped_input.load(Relaxed),
            catchups: s.catchups.load(Relaxed),
            message: s.error.lock().map(|m| m.clone()).unwrap_or_default(),
        }
    }
    /// (converter RTF, vocoder RTF) of the last measurement window.
    pub fn rtf_split(&self) -> (f32, f32) {
        (self.0.rtf_conv_milli.load(Relaxed) as f32 / 1000.0, self.0.rtf_voc_milli.load(Relaxed) as f32 / 1000.0)
    }
}

pub struct AccentConfig {
    /// `None`: search with [`find_models_dir`].
    pub models_dir: Option<PathBuf>,
    pub onnxruntime_dll: PathBuf,
    /// ORT intra-op threads for both sessions.
    pub threads: usize,
    /// Fixed jitter buffer on the output side.
    pub jitter_ms: u32,
    pub initial_voice: u8,
    /// Runs once on the worker thread right before the first model load (e.g. the app's "compare and install the
    /// bundle" step) and returns the directory to load from. Takes precedence over `models_dir`.
    pub prepare_models: Option<Box<dyn FnOnce() -> Result<PathBuf, String> + Send>>,
}

/// ORT intra-op threads for the accent sessions, chosen from the machine's logical core count so the model never
/// takes cores the audio threads need. Override: `CLEARNAI_ACCENT_THREADS`.
///
/// Above 8 cores this keeps the original rule, measured on a 12-thread PC with the full chain (Hush + Accent +
/// AutoGain, `accent_bench live`): 4 threads -> mic-thread p99 10.5 ms (over its 10 ms deadline), 2 underruns, +747 ms
/// added delay and an RTF close to the 0.9 pause limit; 8 threads -> p99 5.7 ms, 1 underrun, +477 ms. The smaller
/// tiers below are ESTIMATES (leave room for ~5 audio/worker threads), NOT measured on an i3 or any other small CPU.
pub fn accent_threads_for_cores(cores: usize) -> usize {
    match cores {
        0..=2 => 1,
        3..=4 => cores / 2,
        5..=8 => cores.saturating_sub(2).max(2),
        _ => cores.saturating_sub(4).clamp(4, 8),
    }
}

impl AccentConfig {
    pub fn new(onnxruntime_dll: PathBuf) -> Self {
        let cores = std::thread::available_parallelism().map_or(2, |n| n.get());
        let threads = std::env::var("CLEARNAI_ACCENT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| accent_threads_for_cores(cores));
        Self { models_dir: None, onnxruntime_dll, threads, jitter_ms: 100, initial_voice: ACCENT_OFF, prepare_models: None }
    }
}

/// The mic-thread end. Implements [`Stage`]: `process` is allocation-free and never blocks.
pub struct AccentStage {
    sh: Arc<Shared>,
    in_prod: Producer<f32>,
    out_cons: Consumer<f32>,
    wet: [f32; FRAME_SAMPLES],
    jitter: usize,
    base_jitter: usize,
    /// Frames since the last underrun (drives the jitter-buffer decay).
    stable: u32,
    primed: bool,
    mix: f32,
    was_on: bool,
    holdoff: u32,
}

/// Start the worker thread and return the mic-thread stage plus the GUI handle. The (large) model bundle is
/// loaded lazily the first time a voice is selected, so a user who leaves Accent off pays nothing.
pub fn start(cfg: AccentConfig) -> (AccentStage, AccentHandle) {
    start_inner(Some(cfg), "")
}

fn start_inner(cfg: Option<AccentConfig>, unavailable_reason: &str) -> (AccentStage, AccentHandle) {
    let initial_voice = cfg.as_ref().map_or(ACCENT_OFF, |c| c.initial_voice);
    let jitter_ms = cfg.as_ref().map_or(100, |c| c.jitter_ms);
    let sh = Arc::new(Shared {
        voice: AtomicU8::new(initial_voice.min(VOICE_NAMES.len() as u8)),
        worker: AtomicU8::new(W_IDLE),
        mic: AtomicU8::new(M_IDLE),
        generation: AtomicU32::new(0),
        slow: AtomicBool::new(false),
        stop: AtomicBool::new(false),
        rtf_milli: AtomicU32::new(0),
        rtf_conv_milli: AtomicU32::new(0),
        rtf_voc_milli: AtomicU32::new(0),
        latency_x10: AtomicU32::new(0),
        jitter_ms: AtomicU32::new(jitter_ms),
        underruns: AtomicU32::new(0),
        dropped_input: AtomicU32::new(0),
        catchups: AtomicU32::new(0),
        error: Mutex::new(String::new()),
    });
    let cap = 48_000 * RING_SECONDS;
    let (in_prod, in_cons) = RingBuffer::<f32>::new(cap);
    let (out_prod, out_cons) = RingBuffer::<f32>::new(cap);
    let jitter = (jitter_ms as usize * 48).max(FRAME_SAMPLES);
    match cfg {
        Some(cfg) => {
            let w_sh = sh.clone();
            let spawned = std::thread::Builder::new()
                .name("clearnai-accent-worker".into())
                .spawn(move || worker(w_sh, in_cons, out_prod, cfg));
            if let Err(e) = spawned {
                sh.worker.store(W_FAILED, Relaxed);
                set_error(&sh, format!("could not start worker thread: {e}"));
            }
        }
        None => {
            sh.worker.store(W_FAILED, Relaxed);
            set_error(&sh, unavailable_reason.to_string());
        }
    }
    let stage = AccentStage {
        sh: sh.clone(),
        in_prod,
        out_cons,
        wet: [0.0; FRAME_SAMPLES],
        jitter,
        base_jitter: jitter,
        stable: 0,
        primed: false,
        mix: 0.0,
        was_on: false,
        holdoff: 0,
    };
    (stage, AccentHandle(sh))
}

impl Drop for AccentStage {
    fn drop(&mut self) {
        self.sh.stop.store(true, Relaxed);
    }
}

impl Stage for AccentStage {
    fn name(&self) -> &'static str {
        "Accent"
    }

    fn process(&mut self, frame: &mut [f32]) {
        let sh = &*self.sh;
        if sh.voice.load(Relaxed) == ACCENT_OFF {
            if self.was_on {
                self.was_on = false;
                self.primed = false;
                self.mix = 0.0;
                sh.mic.store(M_IDLE, Relaxed);
            }
            return;
        }
        if !self.was_on {
            // Off -> on: drop stale audio, tell the worker to reset its streaming state.
            while self.out_cons.pop().is_ok() {}
            sh.generation.fetch_add(1, Relaxed);
            self.was_on = true;
            self.primed = false;
            self.mix = 0.0;
            self.holdoff = 0;
            sh.mic.store(M_STARTING, Relaxed);
        }
        if sh.worker.load(Relaxed) != W_READY || frame.len() != FRAME_SAMPLES {
            return; // loading or failed: pass the BVC signal through untouched
        }

        for &s in frame.iter() {
            if self.in_prod.push(s).is_err() {
                sh.dropped_input.fetch_add(1, Relaxed);
            }
        }

        let mut have = false;
        let avail = self.out_cons.slots();
        if !self.primed {
            // The worker delivers audio in bursts of one push (50 ms). Start with `jitter + one push` buffered so
            // the level then cycles between `jitter + push` (just after a burst) and `jitter` (just before the
            // next one): the jitter setting is the real safety margin, not a fraction of it.
            let target = self.jitter + PUSH48;
            if avail >= target + FRAME_SAMPLES {
                for _ in 0..avail - target - FRAME_SAMPLES {
                    let _ = self.out_cons.pop(); // skip audio that is older than the target level
                }
                self.primed = true;
            }
        }
        if self.primed {
            if self.out_cons.slots() >= FRAME_SAMPLES {
                for w in self.wet.iter_mut() {
                    *w = self.out_cons.pop().unwrap_or(0.0);
                }
                have = true;
            } else {
                self.primed = false;
                self.holdoff = REPRIME_HOLDOFF_FRAMES;
                self.stable = 0;
                sh.underruns.fetch_add(1, Relaxed);
                // Adaptive jitter buffer: every underrun asks for 40 ms more cushion (max 300 ms), so a
                // machine that cannot hold the first setting settles instead of flipping between the live
                // signal and the 300 ms-delayed accent signal (which is exactly what sounds like echo).
                self.jitter = (self.jitter + 40 * 48).min(300 * 48);
                sh.jitter_ms.store((self.jitter / 48) as u32, Relaxed);
            }
        }
        if !have {
            self.wet = [0.0; FRAME_SAMPLES];
        } else {
            // The cushion only ever grew after an underrun, so a rough patch left the delay permanently higher
            // (heard as "lag"). After 20 s without an underrun, give 20 ms back - but only while the converted
            // signal is silent, so the dropped samples are inaudible.
            self.stable = self.stable.saturating_add(1);
            if self.stable >= 2000
                && self.jitter > self.base_jitter
                && self.out_cons.slots() >= 2 * FRAME_SAMPLES
                && self.wet.iter().all(|v| v.abs() < 0.003)
            {
                for _ in 0..20 * 48 {
                    let _ = self.out_cons.pop();
                }
                self.jitter = self.jitter.saturating_sub(20 * 48).max(self.base_jitter);
                sh.jitter_ms.store((self.jitter / 48) as u32, Relaxed);
                self.stable = 0;
            }
        }
        let slow = sh.slow.load(Relaxed);
        let want_wet = have && !slow && self.holdoff == 0;
        if self.holdoff > 0 {
            self.holdoff -= 1;
        }
        let target = if want_wet { 1.0 } else { 0.0 };
        let step = 1.0 / RAMP_SAMPLES;
        for (d, &w) in frame.iter_mut().zip(self.wet.iter()) {
            if self.mix < target {
                self.mix = (self.mix + step).min(target);
            } else if self.mix > target {
                self.mix = (self.mix - step).max(target);
            }
            *d = *d * (1.0 - self.mix) + w * self.mix;
        }
        sh.mic.store(
            if want_wet && self.mix > 0.5 {
                M_ACTIVE
            } else if slow || (self.holdoff > 0 && self.was_on && sh.underruns.load(Relaxed) > 0) {
                M_PAUSED
            } else {
                M_STARTING
            },
            Relaxed,
        );
    }
}

/// Input conditioning for the model, on the 16 kHz chunk the worker is about to convert:
///
/// * **Speech gate.** The model was trained on clean speech and will happily "pronounce" low-level noise or
///   background babble that BVC lets through, which sounds like BVC not working. Below `GATE_OPEN` RMS (and after
///   a 300 ms hangover) the chunk is replaced by digital silence, so the model outputs silence.
/// * **Slow leveler.** The converter carries input loudness through to the output and was trained on speech at
///   about 0.06 RMS (the reference app applies a per-mic gain for the same reason). The gain only follows speech
///   chunks, falls fast (a sudden loud passage must not clip) and rises slowly.
/// * **Soft clip** instead of a hard clamp, so no gain choice can produce harsh distortion.
struct Leveler {
    est: f32,
    gain: f32,
    open: bool,
    hang: u32,
}

const GATE_OPEN_RMS: f32 = 0.008;
const GATE_HANG_CHUNKS: u32 = 6;

fn soft_clip(v: f32) -> f32 {
    let a = v.abs();
    if a <= 0.8 {
        v
    } else {
        v.signum() * (0.8 + 0.2 * ((a - 0.8) / 0.2).tanh())
    }
}

impl Leveler {
    fn new() -> Self {
        Self { est: LEVEL_TARGET_RMS, gain: 1.0, open: false, hang: 0 }
    }
    fn process(&mut self, x: &mut [f32]) {
        let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len().max(1) as f32).sqrt();
        if rms > GATE_OPEN_RMS {
            self.open = true;
            self.hang = GATE_HANG_CHUNKS;
            self.est += 0.05 * (rms - self.est);
        } else if self.open {
            self.hang = self.hang.saturating_sub(1);
            if self.hang == 0 {
                self.open = false;
            }
        }
        if !self.open {
            x.iter_mut().for_each(|v| *v = 0.0);
            return;
        }
        let target = (LEVEL_TARGET_RMS / self.est.max(1e-4)).clamp(0.1, 6.0);
        // fast down (attack), slow up (release)
        let k = if target < self.gain { 0.7 } else { 0.1 };
        let next = self.gain + k * (target - self.gain);
        let n = x.len() as f32;
        for (i, v) in x.iter_mut().enumerate() {
            let g = self.gain + (next - self.gain) * (i as f32 + 1.0) / n;
            *v = soft_clip(*v * g);
        }
        self.gain = next;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accent_thread_tiers_by_core_count() {
        let expect = [(0, 1), (1, 1), (2, 1), (3, 1), (4, 2), (5, 3), (6, 4), (8, 6), (9, 5), (12, 8), (16, 8), (64, 8)];
        for (cores, threads) in expect {
            assert_eq!(accent_threads_for_cores(cores), threads, "cores = {cores}");
        }
    }

    fn tone(amp: f32, n: usize) -> Vec<f32> {
        (0..n).map(|i| amp * (i as f32 * 0.05).sin()).collect()
    }

    #[test]
    fn gate_silences_noise_floor_but_passes_speech_with_hangover() {
        let mut l = Leveler::new();
        let mut noise = tone(0.004, 800); // ~ -51 dB: BVC residue
        l.process(&mut noise);
        assert!(noise.iter().all(|&v| v == 0.0), "noise below the gate must become digital silence");
        let mut speech = tone(0.1, 800);
        l.process(&mut speech);
        assert!(speech.iter().any(|&v| v.abs() > 0.05));
        // pause: stays open for the hangover (word gaps), then closes
        for _ in 0..GATE_HANG_CHUNKS - 1 {
            let mut quiet = tone(0.004, 800);
            l.process(&mut quiet);
            assert!(quiet.iter().any(|&v| v != 0.0), "hangover must keep the gate open through short pauses");
        }
        let mut quiet = tone(0.004, 800);
        l.process(&mut quiet);
        assert!(quiet.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn loud_input_after_quiet_speech_never_hard_clips() {
        let mut l = Leveler::new();
        for _ in 0..40 {
            let mut quiet_speech = tone(0.02, 800);
            l.process(&mut quiet_speech); // gain climbs towards its ceiling
        }
        let mut loud = tone(0.9, 800);
        l.process(&mut loud);
        let peak = loud.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak < 1.0, "soft clip must stay below full scale, got {peak}");
        // gain must have dropped quickly, not stay at the quiet-speech ceiling
        let mut loud2 = tone(0.9, 800);
        l.process(&mut loud2);
        assert!(l.gain < 1.0, "gain must attack down fast, got {}", l.gain);
    }

    #[test]
    fn soft_clip_is_transparent_below_the_knee_and_bounded_above() {
        assert_eq!(soft_clip(0.5), 0.5);
        assert!(soft_clip(5.0) <= 1.0 && soft_clip(-5.0) >= -1.0);
        assert!(soft_clip(0.9) > 0.8 && soft_clip(0.9) < 0.9);
    }
}

fn set_error(sh: &Shared, msg: String) {
    if let Ok(mut m) = sh.error.lock() {
        *m = msg;
    }
}

fn worker(sh: Arc<Shared>, mut in_cons: Consumer<f32>, mut out_prod: Producer<f32>, mut cfg: AccentConfig) {
    raise_thread_priority();
    let mut engine: Option<AccentStream> = None;
    let mut loaded_voice = 0u8;
    let mut seen_gen = sh.generation.load(Relaxed);
    let mut down = CausalDownsampler3x::new(RESAMPLER_TAPS);
    let mut up = CausalUpsampler2x::new(RESAMPLER_TAPS);
    let mut leveler = Leveler::new();
    let mut buf48 = vec![0.0f32; PUSH48];
    let mut x16: Vec<f32> = Vec::with_capacity(4 * PUSH16);
    let mut rtf_hist: VecDeque<f32> = VecDeque::with_capacity(SLOW_WINDOW + 1);
    let mut ok_streak = 0usize;
    let (mut conv_acc, mut voc_acc, mut total_acc, mut audio_acc) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, 0.0f32);

    while !sh.stop.load(Relaxed) {
        let voice = sh.voice.load(Relaxed);
        if voice == ACCENT_OFF || sh.worker.load(Relaxed) == W_FAILED {
            while in_cons.pop().is_ok() {}
            std::thread::sleep(Duration::from_millis(20));
            continue;
        }
        if engine.is_none() {
            sh.worker.store(W_LOADING, Relaxed);
            let loaded = (|| -> Result<AccentStream, String> {
                init_runtime(&cfg.onnxruntime_dll)?;
                let prepared = match cfg.prepare_models.take() {
                    Some(f) => Some(f()?),
                    None => cfg.models_dir.clone(),
                };
                let dir = prepared.or_else(find_models_dir).ok_or("model bundle (app_models) not found")?;
                AccentStream::load(&dir, cfg.threads, VOICE_NAMES[voice as usize - 1])
            })();
            match loaded {
                Ok(e) => {
                    engine = Some(e);
                    loaded_voice = voice;
                    seen_gen = sh.generation.load(Relaxed);
                    sh.worker.store(W_READY, Relaxed);
                }
                Err(e) => {
                    set_error(&sh, e);
                    sh.worker.store(W_FAILED, Relaxed);
                }
            }
            continue;
        }
        let eng = engine.as_mut().unwrap();
        if voice != loaded_voice {
            if eng.set_voice(VOICE_NAMES[voice as usize - 1]).is_ok() {
                loaded_voice = voice;
            }
        }
        let g = sh.generation.load(Relaxed);
        if g != seen_gen {
            seen_gen = g;
            eng.reset();
            down.reset();
            up.reset();
            leveler = Leveler::new();
            x16.clear();
            rtf_hist.clear();
            sh.slow.store(false, Relaxed);
            ok_streak = 0;
        }

        let avail = in_cons.slots();
        if avail < PUSH48 {
            std::thread::sleep(Duration::from_millis(2));
            continue;
        }
        if avail > MAX_BACKLOG48 {
            // Too far behind the mic: skip the oldest audio (a short gap beats an ever-growing delay).
            for _ in 0..avail - 2 * PUSH48 {
                let _ = in_cons.pop();
            }
            sh.catchups.fetch_add(1, Relaxed);
        }
        for s in buf48.iter_mut() {
            *s = in_cons.pop().unwrap_or(0.0);
        }

        let t0 = Instant::now();
        x16.extend(down.push(&buf48));
        while x16.len() >= PUSH16 {
            let mut chunk: Vec<f32> = x16.drain(..PUSH16).collect();
            leveler.process(&mut chunk);
            match eng.push(&chunk) {
                Ok(audio24) => {
                    let wet48 = up.push(&audio24);
                    for s in wet48 {
                        let _ = out_prod.push(s); // full ring: drop (the mic side will skip ahead anyway)
                    }
                }
                Err(e) => {
                    set_error(&sh, format!("inference failed: {e}"));
                    sh.worker.store(W_FAILED, Relaxed);
                    break;
                }
            }
            let (c, v) = eng.take_busy();
            conv_acc += c;
            voc_acc += v;
            audio_acc += PUSH_MS;
        }
        total_acc += t0.elapsed();

        // Real-time factor over this push.
        let rtf = t0.elapsed().as_secs_f32() * 1000.0 / PUSH_MS;
        if rtf_hist.len() == SLOW_WINDOW {
            rtf_hist.pop_front();
        }
        rtf_hist.push_back(rtf);
        let mean = rtf_hist.iter().sum::<f32>() / rtf_hist.len() as f32;
        sh.rtf_milli.store((mean * 1000.0) as u32, Relaxed);
        if audio_acc >= 1000.0 {
            sh.rtf_conv_milli.store((conv_acc.as_secs_f32() * 1000.0 / audio_acc * 1000.0) as u32, Relaxed);
            sh.rtf_voc_milli.store((voc_acc.as_secs_f32() * 1000.0 / audio_acc * 1000.0) as u32, Relaxed);
            conv_acc = Duration::ZERO;
            voc_acc = Duration::ZERO;
            total_acc = Duration::ZERO;
            audio_acc = 0.0;
        }
        let was_slow = sh.slow.load(Relaxed);
        if !was_slow && rtf_hist.len() == SLOW_WINDOW && mean > SLOW_RTF {
            sh.slow.store(true, Relaxed);
            ok_streak = 0;
        } else if was_slow {
            if mean < RECOVER_RTF {
                ok_streak += 1;
                if ok_streak >= RECOVER_WINDOW {
                    sh.slow.store(false, Relaxed);
                    ok_streak = 0;
                }
            } else {
                ok_streak = 0;
            }
        }
        // Added delay = model look-ahead + mel window reach + both resamplers + waiting for a full push
        // + compute + the fixed jitter buffer + one mic frame.
        let lat = eng.algorithmic_latency_ms()
            + 25.0
            + (down.group_delay_samples_48k + up.group_delay_samples_48k) as f32 / 48.0
            + PUSH_MS / 2.0
            + mean * PUSH_MS
            + sh.jitter_ms.load(Relaxed) as f32
            + PUSH_MS / 2.0 // average of the burst cycle on top of the jitter floor
            + 10.0;
        sh.latency_x10.store((lat * 10.0) as u32, Relaxed);
    }
}

/// Above-normal priority so the converter keeps its place when other apps load the CPU (no-op off Windows).
#[cfg(windows)]
fn raise_thread_priority() {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadPriority(h: isize, p: i32) -> i32;
    }
    const THREAD_PRIORITY_ABOVE_NORMAL: i32 = 1;
    unsafe { SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_ABOVE_NORMAL) };
}
#[cfg(not(windows))]
fn raise_thread_priority() {}
