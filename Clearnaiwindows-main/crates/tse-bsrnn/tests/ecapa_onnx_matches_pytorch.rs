//! The critical Phase B exit criterion: does the exported ECAPA ONNX graph
//! actually load and run correctly under `tract-onnx` (not just
//! `onnxruntime`)? Uses the exact real fbank input and PyTorch-reference
//! embedding output saved by
//! `poc/real_tse/export/export_speaker_encoder_onnx.py` + `dump_fixture_as_raw.py`
//! from a real enrollment recording (`poc/real_tse/data/reference.wav`) -
//! not synthetic data.

use tse_bsrnn::{SpeakerEncoderOnnx, EMBEDDING_DIM, FBANK_DIM};

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    assert_eq!(bytes.len() % 4, 0, "fixture {path} is not a whole number of f32s");
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn ecapa_onnx_model_loads() {
    SpeakerEncoderOnnx::new().expect("bundled ECAPA ONNX model must load under tract-onnx");
}

#[test]
fn ecapa_onnx_matches_real_pytorch_reference_output() {
    let encoder = SpeakerEncoderOnnx::new().expect("bundled ECAPA ONNX model must load under tract-onnx");

    let fbank = read_f32_bin("tests/fixtures/ecapa_fbank.bin");
    let expected_embedding = read_f32_bin("tests/fixtures/ecapa_embedding.bin");
    assert_eq!(expected_embedding.len(), EMBEDDING_DIM);
    assert_eq!(fbank.len() % FBANK_DIM, 0, "fbank fixture length must be a multiple of FBANK_DIM");
    let num_frames = fbank.len() / FBANK_DIM;
    println!("real fixture: {num_frames} fbank frames, expecting {EMBEDDING_DIM}-dim embedding");

    let actual_embedding = encoder
        .embed_from_fbank(&fbank, num_frames)
        .expect("inference must succeed on the real fixture input");

    assert_eq!(actual_embedding.len(), EMBEDDING_DIM);

    let max_abs_diff = actual_embedding
        .iter()
        .zip(expected_embedding.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_abs_diff: f32 = actual_embedding
        .iter()
        .zip(expected_embedding.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / actual_embedding.len() as f32;

    println!("tract-onnx vs real PyTorch reference: max_abs_diff={max_abs_diff:.8}  mean_abs_diff={mean_abs_diff:.8}");
    assert!(
        max_abs_diff < 1e-2,
        "tract-onnx output diverges from the real PyTorch reference by more than expected: \
         max_abs_diff={max_abs_diff}, mean_abs_diff={mean_abs_diff}. This means tract-onnx handles \
         at least one op in this graph differently than PyTorch/onnxruntime did - do not proceed to \
         production wiring until this is root-caused (see speaker-encoder's CAM++ history for what \
         this kind of divergence looks like)."
    );
}
