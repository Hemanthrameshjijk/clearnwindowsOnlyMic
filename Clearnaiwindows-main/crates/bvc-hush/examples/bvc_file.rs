//! Test helper: runs a raw 48 kHz mono f32le file through Hush BVC the way the mic pipeline does
//! (480-sample frames). `cargo run --release -p bvc-hush --example bvc_file -- in.f32 out.f32 [hush|bpo_v1] [dll_dir]`
//! `dll_dir` defaults to %LOCALAPPDATA%\ClearNAI (where the app extracts weya_nc.dll and the model bundles).
use dsp_core::Stage;
use std::path::PathBuf;

const FRAME: usize = 480;
const TAIL_PAD: usize = 24_000; // 0.5 s of silence so Hush's own delay doesn't cut off the last word

fn main() {
    let args: Vec<String> = std::env::args().collect();
    assert!(args.len() >= 3, "usage: bvc_file <in.f32> <out.f32> [hush|bpo_v1] [dll_dir]");
    let which = args.get(3).map_or("hush", |s| s.as_str());
    let dir = args
        .get(4)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(std::env::var("LOCALAPPDATA").unwrap()).join("ClearNAI"));

    let mut stage = match which {
        "bpo_v1" => bvc_hush::HushBvcStage::try_load_with_model(&dir, bvc_hush::DLL_FILE_NAME, bvc_hush::BPO_V1_MODEL_BUNDLE_FILE_NAME),
        _ => bvc_hush::HushBvcStage::try_load(&dir),
    }
    .expect("load Hush");

    let mut samples: Vec<f32> =
        std::fs::read(&args[1]).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
    samples.resize(samples.len() + TAIL_PAD, 0.0);
    samples.resize(samples.len().div_ceil(FRAME) * FRAME, 0.0);
    for frame in samples.chunks_exact_mut(FRAME) {
        stage.process(frame);
    }
    let bytes: Vec<u8> = samples.iter().flat_map(|s| s.to_le_bytes()).collect();
    std::fs::write(&args[2], bytes).unwrap();
    println!("{which}: {} samples written to {}", samples.len(), args[2]);
}
