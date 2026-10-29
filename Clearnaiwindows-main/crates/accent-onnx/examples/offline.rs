//! Offline accent run on raw 48 kHz mono f32le, for tone comparisons against the Python reference.
//!
//!   offline <in48.f32> <out_prefix> [stage]
//!
//! Writes `<prefix>_24k.f32` (AccentStream output, vocoder lag dropped) and `<prefix>_48k.f32` (the app's chain:
//! CausalDownsampler3x -> AccentStream -> CausalUpsampler2x, no leveler). With `stage` it also feeds the real
//! `AccentStage` (leveler + worker + jitter buffer) 10 ms frames at wall-clock pace and writes `<prefix>_stage48k.f32`.
use accent_onnx::pipeline::{start, AccentConfig};
use accent_onnx::resample::CausalUpsampler2x;
use accent_onnx::{find_models_dir, init_runtime, AccentStream};
use dsp_core::Stage;
use std::path::PathBuf;
use accent_onnx::causal_resampler::CausalDownsampler3x;

fn read(p: &str) -> Vec<f32> {
    std::fs::read(p).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}
fn write(p: &str, x: &[f32]) {
    std::fs::write(p, x.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>()).unwrap();
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (inp, prefix) = (&a[1], &a[2]);
    let dll = PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/../../app/assets/onnxruntime.dll"));
    let models = find_models_dir().expect("model bundle");
    let x48 = read(inp);

    init_runtime(&dll).unwrap();
    let mut eng = AccentStream::load(&models, 4, "american_male_1").unwrap();
    let mut down = CausalDownsampler3x::new(97);
    let mut x16 = down.push(&x48);
    x16.extend(vec![0.0; 800]);
    let mut wet24 = Vec::new();
    for c in x16.chunks(800) {
        wet24.extend(eng.push(c).unwrap());
    }
    wet24.extend(eng.flush().unwrap());
    let wet48 = CausalUpsampler2x::new(97).push(&wet24);
    write(&format!("{prefix}_24k.f32"), &wet24);
    write(&format!("{prefix}_48k.f32"), &wet48);
    eprintln!("24k: {} samples, 48k: {} samples", wet24.len(), wet48.len());

    if a.get(3).map(String::as_str) == Some("stage") {
        let mut cfg = AccentConfig::new(dll);
        cfg.models_dir = Some(models);
        cfg.initial_voice = 1;
        let (mut stage, handle) = start(cfg);
        let mut inp48 = vec![0.0f32; 48_000 * 3]; // let it load + prime on silence
        inp48.extend(&x48);
        inp48.extend(vec![0.0f32; 48_000 * 2]);
        let mut out = Vec::with_capacity(inp48.len());
        let t0 = std::time::Instant::now();
        for (i, f) in inp48.chunks(480).enumerate() {
            let mut fr = f.to_vec();
            fr.resize(480, 0.0);
            stage.process(&mut fr);
            out.extend(fr);
            if i % 100 == 99 {
                eprintln!("t={:>4.1}s {}", (i + 1) as f32 / 100.0, handle.status().line());
            }
            let due = std::time::Duration::from_millis(10 * (i as u64 + 1));
            if let Some(d) = due.checked_sub(t0.elapsed()) {
                std::thread::sleep(d);
            }
        }
        eprintln!("stage: {}", handle.status().line());
        write(&format!("{prefix}_stage48k.f32"), &out);
    }
}
