//! Accent conversion benchmark on this PC (step 7 of the accent task).
//!
//!   cargo run --release -p clearnai-app --example accent_bench -- rtf  [models_dir]
//!       converter / vocoder real-time factor for 50/100/200 ms pushes x 1/2/4 ORT threads
//!   cargo run --release -p clearnai-app --example accent_bench -- live [models_dir] [threads] [seconds]
//!       the real mic-path chain (voice gate off) BVC (Hush, if extracted) -> Accent -> AutoGain fed at wall-clock
//!       pace in 10 ms frames: total process CPU %, mic-thread cost per frame, underruns, RTF, and the measured
//!       end-to-end delay added by Accent (BVC+Accent vs BVC only, by speech-envelope cross-correlation)
//!
//! Needs the accent model bundle (arg, `ACCENT_MODELS_DIR`, or %LOCALAPPDATA%\ClearNAI\app_models) and, for `live`,
//! the Hush assets the app extracts into %LOCALAPPDATA%\ClearNAI.

use accent_onnx::pipeline::{start, AccentConfig, AccentState};
use accent_onnx::{find_models_dir, init_runtime, AccentStream};
use dsp_core::Stage;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tse_bsrnn::causal_resampler::CausalUpsampler3x;

#[cfg(windows)]
mod cpu {
    #[link(name = "kernel32")]
    extern "system" {
        fn GetCurrentProcess() -> isize;
        fn GetProcessTimes(h: isize, create: *mut u64, exit: *mut u64, kernel: *mut u64, user: *mut u64) -> i32;
    }
    /// Total (kernel + user) CPU time of this process so far, seconds.
    pub fn process_seconds() -> f64 {
        let (mut c, mut e, mut k, mut u) = (0u64, 0u64, 0u64, 0u64);
        unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) };
        (k + u) as f64 * 1e-7
    }
}
#[cfg(not(windows))]
mod cpu {
    pub fn process_seconds() -> f64 {
        0.0
    }
}

fn dll() -> PathBuf {
    let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_default().join("ClearNAI").join("onnxruntime.dll");
    if local.exists() {
        local
    } else {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("assets/onnxruntime.dll")
    }
}

fn clip16() -> Vec<f32> {
    let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../crates/accent-onnx/tests/golden/input16.f32");
    std::fs::read(p).expect("golden input clip").chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn models(arg: Option<&String>) -> PathBuf {
    arg.filter(|s| !s.is_empty()).map(PathBuf::from).or_else(find_models_dir).expect("model bundle not found (pass a dir or set ACCENT_MODELS_DIR)")
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[((v.len() - 1) as f64 * p) as usize]
}

fn rtf_grid(dir: &Path) {
    init_runtime(&dll()).expect("onnxruntime.dll");
    let clip = clip16();
    let x: Vec<f32> = clip.iter().chain(clip.iter()).chain(clip.iter()).copied().collect(); // 15 s
    println!("| ORT threads | push | converter RTF | vocoder RTF | total RTF (incl. mel) | p95 per push (ms) | push budget (ms) |");
    println!("|---:|---:|---:|---:|---:|---:|---:|");
    for threads in [1usize, 2, 4] {
        let mut s = AccentStream::load(dir, threads, "american_male_1").expect("load");
        for push_ms in [50usize, 100, 200] {
            let n = push_ms * 16;
            s.reset();
            // warm-up: first second is not measured (ORT allocations, caches)
            for c in x[..16_000].chunks(n) {
                s.push(c).unwrap();
            }
            let _ = s.take_busy();
            let (mut conv, mut voc, mut total) = (0f64, 0f64, 0f64);
            let mut per_push = Vec::new();
            let rest = &x[16_000..];
            for c in rest.chunks_exact(n) {
                let t = Instant::now();
                s.push(c).unwrap();
                let dt = t.elapsed().as_secs_f64();
                total += dt;
                per_push.push(dt * 1000.0);
            }
            let (c, v) = s.take_busy();
            conv += c.as_secs_f64();
            voc += v.as_secs_f64();
            let audio = (rest.len() / n * n) as f64 / 16_000.0;
            println!(
                "| {threads} | {push_ms} ms | {:.3} | {:.3} | {:.3} | {:.1} | {push_ms} |",
                conv / audio,
                voc / audio,
                total / audio,
                pct(&mut per_push, 0.95)
            );
        }
    }
}

fn envelope(x: &[f32], win: usize) -> Vec<f32> {
    x.chunks(win).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt()).collect()
}

fn best_lag(out: &[f32], reference: &[f32], max_lag: usize) -> (usize, f32) {
    let norm = |v: &[f32]| v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
    let mut best = (0, -1.0f32);
    for lag in 0..max_lag {
        let n = out.len().saturating_sub(lag).min(reference.len());
        let (a, b) = (&out[lag..lag + n], &reference[..n]);
        let c = a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>() / (norm(a) * norm(b));
        if c > best.1 {
            best = (lag, c);
        }
    }
    best
}

/// Feeds `x48` through BVC (+ optionally Accent) + AutoGain at wall-clock pace; returns the output.
fn run_chain(
    x48: &[f32],
    bvc: &mut Option<Box<dyn Stage>>,
    accent: Option<&mut accent_onnx::pipeline::AccentStage>,
    mut transcript: Option<&mut asr_onnx::pipeline::TranscriptStage>,
    mic_ns: &mut Vec<f64>,
) -> Vec<f32> {
    let mut auto_gain = dsp_core::AutoGainStage::new();
    let mut accent = accent;
    let mut out = Vec::with_capacity(x48.len());
    let mut frame = vec![0.0f32; 480];
    let t0 = Instant::now();
    for (k, c) in x48.chunks_exact(480).enumerate() {
        frame.copy_from_slice(c);
        let t = Instant::now();
        if let Some(b) = bvc.as_mut() {
            b.process(&mut frame);
        }
        if let Some(t) = transcript.as_mut() {
            t.process(&mut frame);
        }
        if let Some(a) = accent.as_mut() {
            a.process(&mut frame);
        }
        auto_gain.process(&mut frame);
        mic_ns.push(t.elapsed().as_nanos() as f64);
        out.extend_from_slice(&frame);
        let due = t0 + Duration::from_millis(10 * (k as u64 + 1));
        if let Some(d) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }
    }
    out
}

fn live(dir: &Path, threads: usize, seconds: usize) {
    let local = std::env::var_os("LOCALAPPDATA").map(PathBuf::from).unwrap_or_default().join("ClearNAI");
    let mut bvc: Option<Box<dyn Stage>> = match bvc_hush::HushBvcStage::try_load(&local) {
        Ok(s) => Some(Box::new(s)),
        Err(e) => {
            println!("NOTE: Hush BVC not available ({e:#}); measuring without it.");
            None
        }
    };
    // 48 kHz test signal: 1.5 s silence, then the speech clip repeated (with gaps) up to `seconds`.
    let clip = clip16();
    let mut x16 = vec![0.0f32; 24_000];
    while x16.len() < seconds * 16_000 {
        x16.extend_from_slice(&clip);
        x16.extend(std::iter::repeat(0.0).take(8_000));
    }
    let mut up = CausalUpsampler3x::new(97);
    let mut x48 = up.push(&x16);
    x48.extend(up.push(&vec![0.0; 4_000]));
    let speech_start = 24_000 * 3;

    // Reference run: BVC only (its own delay is subtracted from the accent run).
    let mut ns_ref = Vec::new();
    let out_ref = run_chain(&x48, &mut bvc, None, None, &mut ns_ref);
    if let Some(b) = bvc.as_mut() {
        let mut f = vec![0.0f32; 480];
        for _ in 0..200 {
            b.process(&mut f); // let the BVC state settle between runs
        }
    }

    // Accent run.
    let mut cfg = AccentConfig::new(dll());
    cfg.models_dir = Some(dir.to_path_buf());
    cfg.threads = threads;
    cfg.initial_voice = 1;
    let (mut stage, handle) = start(cfg);
    let t = Instant::now();
    while handle.status().state == AccentState::Loading && t.elapsed() < Duration::from_secs(60) {
        let mut f = vec![0.0f32; 480];
        stage.process(&mut f);
        std::thread::sleep(Duration::from_millis(10));
    }
    println!("model load: {:.1} s (threads {threads})", t.elapsed().as_secs_f32());
    // Steady-state CPU window = the whole accent run minus the first 2 s.
    let mut ns = Vec::new();
    let cpu0 = cpu::process_seconds();
    let wall0 = Instant::now();
    // Optional live transcript running next to Accent (env WITH_TRANSCRIPT=1): does it hurt the accent chain?
    let mut tr = None;
    let mut tr_handle = None;
    if std::env::var_os("WITH_TRANSCRIPT").is_some() {
        let mut tcfg = asr_onnx::pipeline::TranscriptConfig::new(dll());
        tcfg.initial_enabled = true;
        let (ts, th) = asr_onnx::pipeline::start(tcfg);
        let t = Instant::now();
        while th.status().state != asr_onnx::pipeline::TranscriptState::Listening && t.elapsed() < Duration::from_secs(60) {
            std::thread::sleep(Duration::from_millis(50));
        }
        println!("transcript ready after {:.1} s: {}", t.elapsed().as_secs_f32(), th.status().line());
        tr = Some(ts);
        tr_handle = Some(th);
    }
    let out = run_chain(&x48, &mut bvc, Some(&mut stage), tr.as_mut(), &mut ns);
    if let Some(th) = &tr_handle {
        let (lines, partial) = th.snapshot(10);
        println!("transcript lines: {} | last partial: {partial:?} | {}", lines.len(), th.status().line());
        for l in lines.iter().take(4) {
            println!("   [{}] {}", asr_onnx::pipeline::fmt_time(l.at_secs), l.text);
        }
    }
    let (cpu1, wall) = (cpu::process_seconds(), wall0.elapsed().as_secs_f64());
    let st = handle.status();
    let (rc, rv) = handle.rtf_split();
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let cpu_s = cpu1 - cpu0;
    println!("{}", st.line());
    println!("converter RTF {rc:.3} | vocoder RTF {rv:.3} | underruns {} | dropped input {} | catch-ups {}", st.underruns, st.dropped_input, st.catchups);
    println!("process CPU while BVC + Accent run: {:.1}% of one core = {:.1}% of this PC ({cores} logical cores)", cpu_s / wall * 100.0, cpu_s / wall / cores as f64 * 100.0);
    println!(
        "mic-thread time per 10 ms frame (BVC + Accent stage + AutoGain): p50 {:.3} ms  p99 {:.3} ms  max {:.3} ms  (BVC only: p50 {:.3} ms)",
        pct(&mut ns.clone(), 0.5) / 1e6,
        pct(&mut ns.clone(), 0.99) / 1e6,
        pct(&mut ns.clone(), 1.0) / 1e6,
        pct(&mut ns_ref.clone(), 0.5) / 1e6
    );
    let win = 120;
    let ref_env = envelope(&x48[speech_start..], win);
    let (lag_ref, c_ref) = best_lag(&envelope(&out_ref[speech_start..], win), &ref_env, 400);
    let (lag_acc, c_acc) = best_lag(&envelope(&out[speech_start..], win), &ref_env, 400);
    println!(
        "delay input->output: BVC only {:.1} ms (corr {c_ref:.2}) | BVC + Accent {:.1} ms (corr {c_acc:.2}) | added by Accent {:.1} ms (status estimate {:.0} ms)",
        lag_ref as f32 * 2.5,
        lag_acc as f32 * 2.5,
        (lag_acc as f32 - lag_ref as f32) * 2.5,
        st.latency_ms
    );
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("rtf") => rtf_grid(&models(args.get(1))),
        Some("live") => live(
            &models(args.get(1)),
            args.get(2).and_then(|s| s.parse().ok()).unwrap_or(2),
            args.get(3).and_then(|s| s.parse().ok()).unwrap_or(20),
        ),
        _ => eprintln!("usage: accent_bench rtf [models_dir] | live [models_dir] [threads] [seconds]"),
    }
}
