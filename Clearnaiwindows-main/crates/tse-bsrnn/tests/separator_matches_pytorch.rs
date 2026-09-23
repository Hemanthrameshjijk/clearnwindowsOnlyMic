//! Verifies the hand-rolled Rust separator (band_split -> subband_norm ->
//! 6x repeat block -> band_masker) against a real forward pass through the
//! actual PyTorch modules, using the exact real weights from the
//! hash-verified checkpoint (see
//! `poc/real_tse/export/export_separator_weights.py`'s fixture-generation
//! code for exactly what was computed). No speaker-embedding fusion is
//! applied in this fixture (isolates separator correctness from the
//! separate, much simpler fusion multiply - see `separator.rs`'s
//! `apply_speaker_gate` and its own docs).

use tse_bsrnn::separator::BsrnnSeparator;

const T_FRAMES: usize = 20;
const F_TOTAL: usize = 257;

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn max_mean_abs_diff(a: &[f32], b: &[f32]) -> (f32, f32) {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    let max = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
    let mean = a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum::<f32>() / a.len() as f32;
    (max, mean)
}

#[test]
fn separator_pipeline_matches_real_pytorch_forward_pass() {
    let sep = BsrnnSeparator::load().expect("separator weights must load");
    assert_eq!(sep.nband(), 32);
    assert_eq!(sep.band_width().iter().sum::<usize>(), F_TOTAL);
    assert_eq!(sep.num_repeat(), 6);

    let spec_ri = read_f32_bin("tests/fixtures/separator_spec_RI.bin");
    assert_eq!(spec_ri.len(), 2 * F_TOTAL * T_FRAMES);
    let (spec_real, spec_imag) = spec_ri.split_at(F_TOTAL * T_FRAMES);

    let mix_real = read_f32_bin("tests/fixtures/separator_spec_complex_real.bin");
    let mix_imag = read_f32_bin("tests/fixtures/separator_spec_complex_imag.bin");

    let expected_subband_feature = read_f32_bin("tests/fixtures/separator_subband_feature.bin");
    let expected_sep_output = read_f32_bin("tests/fixtures/separator_sep_output.bin");
    let expected_est_spec_ri = read_f32_bin("tests/fixtures/separator_est_spec_RI.bin");

    // ---- Step 1: band_split + subband_norm ----
    let subband_feature = sep.band_split_and_subband_norm(spec_real, spec_imag, T_FRAMES);
    let (max_d, mean_d) = max_mean_abs_diff(&subband_feature, &expected_subband_feature);
    println!("subband_norm: max_abs_diff={max_d:.8} mean_abs_diff={mean_d:.8}");
    assert!(max_d < 1e-3, "subband_norm diverges from PyTorch reference: max_abs_diff={max_d}");

    // ---- Step 2: separator (6x repeat block), single call with T=20 ----
    let mut state = sep.new_state();
    let sep_output = sep.run_separator(&subband_feature, &mut state, T_FRAMES);
    let (max_d, mean_d) = max_mean_abs_diff(&sep_output, &expected_sep_output);
    println!("separator (6x repeat block): max_abs_diff={max_d:.8} mean_abs_diff={mean_d:.8}");
    assert!(max_d < 1e-2, "separator diverges from PyTorch reference: max_abs_diff={max_d}");

    // ---- Step 3: band_masker + complex mask application ----
    let (est_real, est_imag) = sep.run_band_masker(&sep_output, &mix_real, &mix_imag, T_FRAMES);
    let (expected_real, expected_imag) = expected_est_spec_ri.split_at(F_TOTAL * T_FRAMES);
    let (max_d_r, mean_d_r) = max_mean_abs_diff(&est_real, expected_real);
    let (max_d_i, mean_d_i) = max_mean_abs_diff(&est_imag, expected_imag);
    println!("band_masker real: max_abs_diff={max_d_r:.8} mean_abs_diff={mean_d_r:.8}");
    println!("band_masker imag: max_abs_diff={max_d_i:.8} mean_abs_diff={mean_d_i:.8}");
    assert!(max_d_r < 1e-2, "band_masker (real) diverges from PyTorch reference: max_abs_diff={max_d_r}");
    assert!(max_d_i < 1e-2, "band_masker (imag) diverges from PyTorch reference: max_abs_diff={max_d_i}");
}

#[test]
fn chunked_processing_with_carried_state_matches_single_call_processing() {
    // Self-consistency check: processing T=20 frames as 20 separate
    // single-timestep calls with band_rnn state carried across calls must
    // produce the same result as processing all 20 frames in one call -
    // this is the real streaming property this crate depends on, checked
    // directly rather than assumed (the same kind of check that caught
    // real state-handling bugs in the Python streaming engine this was
    // ported from).
    let sep = BsrnnSeparator::load().expect("separator weights must load");
    let spec_ri = read_f32_bin("tests/fixtures/separator_spec_RI.bin");
    let (spec_real, spec_imag) = spec_ri.split_at(F_TOTAL * T_FRAMES);

    let subband_feature_all = sep.band_split_and_subband_norm(spec_real, spec_imag, T_FRAMES);
    let mut state_single = sep.new_state();
    let sep_output_single_call = sep.run_separator(&subband_feature_all, &mut state_single, T_FRAMES);

    let mut state_chunked = sep.new_state();
    let mut sep_output_chunked = vec![0.0f32; sep.nband() * 128 * T_FRAMES];
    for t in 0..T_FRAMES {
        // Extract this single timestep's (nband, feature_dim, 1) slice.
        let mut one_frame_real = vec![0.0f32; F_TOTAL];
        let mut one_frame_imag = vec![0.0f32; F_TOTAL];
        for f in 0..F_TOTAL {
            one_frame_real[f] = spec_real[f * T_FRAMES + t];
            one_frame_imag[f] = spec_imag[f * T_FRAMES + t];
        }
        let subband_feature_one = sep.band_split_and_subband_norm(&one_frame_real, &one_frame_imag, 1);
        let out_one = sep.run_separator(&subband_feature_one, &mut state_chunked, 1);
        for band in 0..sep.nband() {
            for f in 0..128 {
                sep_output_chunked[(band * 128 + f) * T_FRAMES + t] = out_one[band * 128 + f];
            }
        }
    }

    let (max_d, mean_d) = max_mean_abs_diff(&sep_output_chunked, &sep_output_single_call);
    println!("chunked (20x T=1 calls, state carried) vs single-call (T=20): max_abs_diff={max_d:.8} mean_abs_diff={mean_d:.8}");
    assert!(
        max_d < 1e-4,
        "chunked processing with carried state must numerically match single-call processing: max_abs_diff={max_d}"
    );
}
