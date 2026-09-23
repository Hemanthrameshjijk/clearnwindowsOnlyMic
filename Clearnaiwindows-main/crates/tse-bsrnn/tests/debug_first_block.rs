//! Isolates whether the separator's divergence from the PyTorch reference
//! (see separator_matches_pytorch.rs) originates in band_rnn or band_comm,
//! by checking each sub-step of repeat block 0 independently against a
//! real, targeted fixture (see poc/real_tse/export/debug_first_block.py).

use tse_bsrnn::separator::BsrnnSeparator;

const T_FRAMES: usize = 20;
const NBAND: usize = 32;
const FEATURE_DIM: usize = 128;

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn block0_band_rnn_then_band_comm_match_pytorch() {
    let sep = BsrnnSeparator::load().expect("weights must load");
    let subband_feature = read_f32_bin("tests/fixtures/separator_subband_feature.bin");
    assert_eq!(subband_feature.len(), NBAND * FEATURE_DIM * T_FRAMES);

    let expected_band_rnn_out = read_f32_bin("tests/fixtures/debug_band_rnn_out.bin");
    let expected_block0_output = read_f32_bin("tests/fixtures/debug_block0_output.bin");

    let mut state = sep.new_state();
    let band_rnn_out = sep.run_band_rnn_for_block(0, &subband_feature, &mut state[0], T_FRAMES);
    let d1 = max_abs_diff(&band_rnn_out, &expected_band_rnn_out);
    println!("band_rnn (block 0) only: max_abs_diff={d1:.8}");

    let block0_output = sep.run_band_comm_for_block(0, &band_rnn_out, T_FRAMES);
    let d2 = max_abs_diff(&block0_output, &expected_block0_output);
    println!("band_rnn + band_comm (block 0): max_abs_diff={d2:.8}");

    assert!(d1 < 1e-2, "band_rnn (block 0) diverges from PyTorch: max_abs_diff={d1}");
    assert!(d2 < 1e-2, "band_comm (block 0) diverges from PyTorch: max_abs_diff={d2}");
}
