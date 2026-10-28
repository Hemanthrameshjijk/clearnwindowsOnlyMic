//! Golden tests: the Rust port must reproduce the reference engine (tools/make_golden.py) on a 5 s clip.
//!
//! Needs the model bundle (`ACCENT_MODELS_DIR`, or app_models next to the exe / in %LOCALAPPDATA%\ClearNAI) and
//! `onnxruntime.dll` (`ORT_DLL`, default app/assets/onnxruntime.dll). Without the models the tests print a
//! SKIPPED line and pass, so a checkout without the 270 MB bundle still builds green.

use accent_onnx::{find_models_dir, init_runtime, AccentStream};
use std::path::{Path, PathBuf};
use std::sync::Once;

const MEL_TOL: f32 = 1e-4;
// Audio: the reference ran ORT 1.30 (pip), we run the shipped 1.19 DLL; different kernels give ~1e-4 peak noise
// through the vocoder while the converter mel still agrees to ~4e-6 (so the port itself is exact). The RMS bound
// is the tighter, more meaningful check.
const AUDIO_TOL: f32 = 2e-4;
const AUDIO_RMS_TOL: f32 = 2e-5;
// Different push sizes change matmul shapes inside ORT; the reference itself differs by 6.5e-5 in audio between
// 50 ms and 200 ms pushes (float noise through the vocoder), so chunk invariance is checked a bit looser.
const INVARIANCE_AUDIO_TOL: f32 = 3e-4;

fn golden() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

fn f32_file(name: &str) -> Vec<f32> {
    let b = std::fs::read(golden().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
    b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

static INIT: Once = Once::new();

fn stream(voice: &str, threads: usize) -> Option<AccentStream> {
    let Some(models) = find_models_dir() else {
        eprintln!("SKIPPED: model bundle not found (set ACCENT_MODELS_DIR)");
        return None;
    };
    INIT.call_once(|| {
        let dll = std::env::var("ORT_DLL")
            .map(PathBuf::from)
            .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/assets/onnxruntime.dll"));
        init_runtime(&dll).expect("load onnxruntime.dll");
    });
    Some(AccentStream::load(&models, threads, voice).expect("load models"))
}

struct Run {
    mel_in: Vec<f32>,
    mel_out: Vec<f32>,
    audio: Vec<f32>,
    counts: Vec<[usize; 3]>,
}

fn run(s: &mut AccentStream, x: &[f32], push: usize) -> Run {
    let mut r = Run { mel_in: vec![], mel_out: vec![], audio: vec![], counts: vec![] };
    let add = |r: &mut Run, o: accent_onnx::StepOut| {
        r.counts.push([o.mel_in.len() / 80, o.mel_out.len() / 80, o.audio.len()]);
        r.mel_in.extend(o.mel_in);
        r.mel_out.extend(o.mel_out);
        r.audio.extend(o.audio);
    };
    for c in x.chunks(push) {
        let o = s.push_stages(c).unwrap();
        add(&mut r, o);
    }
    let o = s.flush_stages().unwrap();
    add(&mut r, o);
    r
}

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    (a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>() / a.len() as f32).sqrt()
}

fn check_against_golden(name: &str, push: usize) -> Option<Run> {
    let mut s = stream("american_male_1", 1)?;
    let x = f32_file("input16.f32");
    let r = run(&mut s, &x, push);
    let (gi, go, ga) = (f32_file(&format!("{name}_mel_in.f32")), f32_file(&format!("{name}_mel_out.f32")), f32_file(&format!("{name}_audio24.f32")));
    let (di, dm, da) = (max_diff(&r.mel_in, &gi), max_diff(&r.mel_out, &go), max_diff(&r.audio, &ga));
    eprintln!("{name}: max abs diff  input mel {di:.3e}   converter mel {dm:.3e}   audio {da:.3e}");
    assert!(di < MEL_TOL, "input mel diff {di}");
    assert!(dm < MEL_TOL, "converter mel diff {dm}");
    let dr = rms_diff(&r.audio, &ga);
    eprintln!("{name}: audio rms diff {dr:.3e}");
    assert!(da < AUDIO_TOL, "audio diff {da}");
    assert!(dr < AUDIO_RMS_TOL, "audio rms diff {dr}");
    Some(r)
}

#[test]
fn golden_50ms_pushes() {
    let Some(r) = check_against_golden("p50ms", 800) else { return };
    // Per-push frame counts must match the reference exactly (same buffering behaviour).
    let meta: serde_json::Value = serde_json::from_slice(&std::fs::read(golden().join("golden.json")).unwrap()).unwrap();
    let want: Vec<[usize; 3]> = meta["runs"]["p50ms"]["per_push"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| [v[0].as_u64().unwrap() as usize, v[1].as_u64().unwrap() as usize, v[2].as_u64().unwrap() as usize])
        .collect();
    // The reference reports the vocoder flush as its own last record; flush_stages folds it into the final step.
    let mut want = want;
    let last = want.pop().unwrap();
    want.last_mut().unwrap()[2] += last[2];
    assert_eq!(r.counts, want, "per-push (mel_in, mel_out, audio) counts differ from the reference");
}

#[test]
fn golden_200ms_pushes() {
    check_against_golden("p200ms", 3200);
}

#[test]
fn push_size_invariance_50_vs_200ms() {
    let (Some(mut a), Some(mut b)) = (stream("american_male_1", 1), stream("american_male_1", 1)) else { return };
    let x = f32_file("input16.f32");
    let (r50, r200) = (run(&mut a, &x, 800), run(&mut b, &x, 3200));
    let (dm, da) = (max_diff(&r50.mel_out, &r200.mel_out), max_diff(&r50.audio, &r200.audio));
    eprintln!("50 ms vs 200 ms pushes: converter mel {dm:.3e}   audio {da:.3e}");
    assert!(dm < MEL_TOL && da < INVARIANCE_AUDIO_TOL, "chunking changes the output: mel {dm}, audio {da}");
}

#[test]
fn start_lag_is_dropped_and_length_is_exact() {
    let Some(mut s) = stream("american_male_1", 1) else { return };
    let x = f32_file("input16.f32");
    let mut audio = Vec::new();
    for c in x.chunks(800) {
        audio.extend(s.push(c).unwrap());
    }
    audio.extend(s.flush().unwrap());
    // 5 s in -> 401 mel frames -> 401*300 samples at 24 kHz, aligned with the input (lag dropped).
    assert_eq!(audio.len(), 401 * 300);
    let ga = f32_file("p50ms_audio24.f32");
    assert!(max_diff(&audio, &ga[600..]) < AUDIO_TOL);
}
