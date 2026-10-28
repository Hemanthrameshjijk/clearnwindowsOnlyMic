//! Transcribe raw 16 kHz mono f32le files: `cargo run --release -p asr-onnx --example transcribe_file -- a.f32 [b.f32 ...]`
//! One line per file: `<file name>\t<text>` (the model is loaded once).
use std::path::Path;

fn main() {
    let paths: Vec<String> = std::env::args().skip(1).collect();
    assert!(!paths.is_empty(), "usage: transcribe_file <16k mono f32le file>...");
    let dir = asr_onnx::find_models_dir().expect("speech model folder not found");
    let dll = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/assets/onnxruntime.dll");
    accent_onnx::ort_util::init_ort(&dll).expect("onnxruntime.dll");
    let mut m = asr_onnx::AsrModel::load(&dir, 2).expect("load");
    for path in &paths {
        let audio: Vec<f32> =
            std::fs::read(path).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
        let name = Path::new(path).file_name().unwrap().to_string_lossy();
        println!("{name}\t{}", m.transcribe(&audio).unwrap());
    }
}
