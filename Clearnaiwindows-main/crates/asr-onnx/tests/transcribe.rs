//! Golden check against the Python reference (`tools/make_golden.py`): log-mel features and the decoded text of the
//! 5 s accent test clip. The model test needs the model folder (ASR_MODELS_DIR, asr_models next to the exe, or
//! %LOCALAPPDATA%\ClearNAI\asr_models) and onnxruntime.dll; without them it skips with a message.

use asr_onnx::features::Features;
use std::path::Path;

fn read_f32(p: &Path) -> Vec<f32> {
    std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn golden(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(name)
}

fn clip() -> Vec<f32> {
    read_f32(&Path::new(env!("CARGO_MANIFEST_DIR")).join("../accent-onnx/tests/golden/input16.f32"))
}

#[test]
fn features_match_python() {
    let want = read_f32(&golden("asr_feat.f32"));
    let (got, frames) = Features::new().compute(&clip()).unwrap();
    assert_eq!(got.len(), want.len(), "frames {frames}");
    let max = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("max abs feature diff {max:.2e}");
    assert!(max < 2e-3, "features differ from the Python reference by {max}");
}

#[test]
fn text_matches_python() {
    let Some(dir) = asr_onnx::find_models_dir() else {
        eprintln!("SKIP: speech model folder not found");
        return;
    };
    let dll = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../app/assets/onnxruntime.dll");
    accent_onnx::ort_util::init_ort(&dll).expect("onnxruntime.dll");
    let mut m = asr_onnx::AsrModel::load(&dir, 2).expect("load");
    let want = std::fs::read_to_string(golden("asr_text.txt")).unwrap();
    let got = m.transcribe(&clip()).unwrap();
    println!("got:  {got}\nwant: {want}");
    assert_eq!(got, want);
    // long audio (3 clips, ~16 s) is decoded in windows: every clip must still come out ("insurance" x3)
    let mut long = Vec::new();
    for _ in 0..3 {
        long.extend(clip());
        long.extend(std::iter::repeat(0.0f32).take(4_800));
    }
    let text = m.transcribe(&long).unwrap();
    println!("long: {text}");
    assert!(text.matches("insurance").count() >= 3, "long audio lost words: {text}");
    // silence gives no text
    assert_eq!(m.transcribe(&vec![0.0; 16_000]).unwrap(), "");
}
