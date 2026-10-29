//! Real-time integration test of `AccentStage` + worker thread: frames of 480 @ 48 kHz are fed at wall-clock
//! pace (10 ms each), exactly as the mic pipeline does. Skips (passes) without the model bundle.

use accent_onnx::pipeline::{start, AccentConfig, AccentState};
use accent_onnx::resample::CausalUpsampler2x;
use accent_onnx::{find_models_dir, init_runtime, AccentStream};
use dsp_core::Stage;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use accent_onnx::causal_resampler::{CausalDownsampler3x, CausalUpsampler3x};

fn dll() -> PathBuf {
    std::env::var("ORT_DLL").map(PathBuf::from).unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/assets/onnxruntime.dll"))
}

fn input16() -> Vec<f32> {
    let b = std::fs::read(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden/input16.f32")).unwrap();
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// 1.5 s of digital silence (lets the stage prime) then the 5 s clip, as 48 kHz.
fn input48() -> (Vec<f32>, usize) {
    let mut x16 = vec![0.0f32; 24_000];
    x16.extend(input16());
    let mut up = CausalUpsampler3x::new(97);
    let mut x48 = up.push(&x16);
    x48.extend(up.push(&vec![0.0; 4_000]));
    (x48, 24_000 * 3)
}

fn envelope(x: &[f32], win: usize) -> Vec<f32> {
    x.chunks(win).map(|c| (c.iter().map(|v| v * v).sum::<f32>() / c.len() as f32).sqrt()).collect()
}

/// Lag (in `win`-sample windows) at which `a[j]` best matches `b[j - lag]`.
fn best_lag(a: &[f32], b: &[f32], max_lag: usize) -> (usize, f32) {
    let norm = |v: &[f32]| (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-9);
    let mut best = (0, -1.0f32);
    for lag in 0..max_lag {
        let n = a.len().min(b.len() + lag).saturating_sub(lag);
        let (aa, bb) = (&a[lag..lag + n], &b[..n]);
        let c = aa.iter().zip(bb).map(|(x, y)| x * y).sum::<f32>() / (norm(aa) * norm(bb));
        if c > best.1 {
            best = (lag, c);
        }
    }
    best
}

#[test]
fn realtime_stage_runs_clean_and_reports_latency() {
    let Some(models) = find_models_dir() else {
        eprintln!("SKIPPED: model bundle not found (set ACCENT_MODELS_DIR)");
        return;
    };
    let (x48, speech_start) = input48();

    let mut cfg = AccentConfig::new(dll());
    cfg.models_dir = Some(models.clone());
    cfg.initial_voice = 1;
    let (mut stage, handle) = start(cfg);

    // Wait for the (lazy) model load without feeding audio the stage would drop anyway.
    let t = Instant::now();
    while handle.status().state == AccentState::Loading && t.elapsed() < Duration::from_secs(60) {
        let mut f = vec![0.0f32; 480];
        stage.process(&mut f);
        std::thread::sleep(Duration::from_millis(10));
    }
    eprintln!("model load: {:.1} s", t.elapsed().as_secs_f32());

    // Real-time feed.
    let mut out = Vec::with_capacity(x48.len());
    let mut frame = vec![0.0f32; 480];
    let t0 = Instant::now();
    let mut active_frames = 0usize;
    for (k, c) in x48.chunks_exact(480).enumerate() {
        frame.copy_from_slice(c);
        stage.process(&mut frame);
        out.extend_from_slice(&frame);
        if handle.status().state == AccentState::Active {
            active_frames += 1;
        }
        let due = t0 + Duration::from_millis(10 * (k as u64 + 1));
        if let Some(d) = due.checked_duration_since(Instant::now()) {
            std::thread::sleep(d);
        }
    }
    let st = handle.status();
    let (rc, rv) = handle.rtf_split();
    eprintln!("{} | converter RTF {rc:.2} vocoder RTF {rv:.2} | underruns {} dropped_in {} catchups {}", st.line(), st.underruns, st.dropped_input, st.catchups);
    assert_eq!(st.underruns, 0, "jitter buffer underran");
    assert_eq!(st.dropped_input, 0);
    assert!(active_frames > x48.len() / 480 / 2, "stage was Active for only {active_frames} frames");

    // Offline reference through the same chain (no leveler: only the envelope shape is compared).
    init_runtime(&dll()).unwrap();
    let mut eng = AccentStream::load(&models, 1, "american_male_1").unwrap();
    let mut down = CausalDownsampler3x::new(97);
    let mut x16 = down.push(&x48);
    x16.extend(vec![0.0; 800]);
    let mut wet24 = Vec::new();
    for c in x16.chunks(800) {
        wet24.extend(eng.push(c).unwrap());
    }
    wet24.extend(eng.flush().unwrap());
    let wet48 = CausalUpsampler2x::new(97).push(&wet24);

    let win = 120; // 2.5 ms
    let (ea, eb) = (envelope(&out[speech_start..], win), envelope(&wet48[speech_start.min(wet48.len())..], win));
    let (lag, corr) = best_lag(&ea, &eb, 400);
    let latency_ms = lag as f32 * 2.5;
    eprintln!("measured end-to-end added latency (envelope xcorr): {latency_ms:.1} ms (corr {corr:.3}); status estimate {:.0} ms", st.latency_ms);
    assert!(corr > 0.8, "output does not resemble the offline conversion (corr {corr})");
    assert!((100.0..600.0).contains(&latency_ms), "latency {latency_ms} ms out of range");
}

#[test]
fn off_is_a_bit_exact_passthrough() {
    let cfg = AccentConfig::new(dll());
    let (mut stage, handle) = start(cfg);
    assert_eq!(handle.status().state, AccentState::Off);
    let src: Vec<f32> = (0..480).map(|i| (i as f32 * 0.01).sin() * 0.3).collect();
    let mut f = src.clone();
    for _ in 0..100 {
        f.copy_from_slice(&src);
        stage.process(&mut f);
        assert_eq!(f, src);
    }
}

#[test]
fn missing_models_degrade_to_passthrough_with_a_message() {
    let mut cfg = AccentConfig::new(dll());
    cfg.models_dir = Some(PathBuf::from("Z:/definitely/not/here"));
    cfg.initial_voice = 2;
    let (mut stage, handle) = start(cfg);
    let src = vec![0.25f32; 480];
    let t = Instant::now();
    while handle.status().state != AccentState::Unavailable && t.elapsed() < Duration::from_secs(10) {
        let mut f = src.clone();
        stage.process(&mut f);
        assert_eq!(f, src, "audio must pass through untouched while the model is missing");
        std::thread::sleep(Duration::from_millis(10));
    }
    let st = handle.status();
    assert_eq!(st.state, AccentState::Unavailable);
    eprintln!("{}", st.line());
}
