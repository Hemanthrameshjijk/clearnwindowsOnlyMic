//! Verifies the Rust causal_resampler.rs port against the already-fixed,
//! verified Python causal_resampler.py, chunked exactly as production
//! would (480-sample/10ms@48k chunks) - see
//! `poc/real_tse/export/export_resampler_fixture.py`.

use tse_bsrnn::causal_resampler::{CausalDownsampler3x, CausalUpsampler3x};

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn rust_resampler_matches_python_resampler_chunked() {
    let input = read_f32_bin("tests/fixtures/resampler_input_48k.bin");
    let expected_down = read_f32_bin("tests/fixtures/resampler_down_16k.bin");
    let expected_roundtrip = read_f32_bin("tests/fixtures/resampler_roundtrip_48k.bin");

    let mut down = CausalDownsampler3x::new(63);
    let mut up = CausalUpsampler3x::new(63);
    assert_eq!(down.group_delay_samples_48k, 31);
    assert_eq!(up.group_delay_samples_48k, 31);

    let chunk = 480; // 10ms @ 48kHz, matching production frame size
    let mut down_out = Vec::new();
    let mut roundtrip_out = Vec::new();
    let mut i = 0;
    while i < input.len() {
        let end = (i + chunk).min(input.len());
        let d = down.push(&input[i..end]);
        if !d.is_empty() {
            let u = up.push(&d);
            roundtrip_out.extend_from_slice(&u);
        }
        down_out.extend_from_slice(&d);
        i = end;
    }

    println!("Rust: {} downsampled samples (Python: {}), {} round-trip samples (Python: {})",
        down_out.len(), expected_down.len(), roundtrip_out.len(), expected_roundtrip.len());
    assert_eq!(down_out.len(), expected_down.len());
    assert_eq!(roundtrip_out.len(), expected_roundtrip.len());

    let d_down = max_abs_diff(&down_out, &expected_down);
    let d_roundtrip = max_abs_diff(&roundtrip_out, &expected_roundtrip);
    println!("downsampled max_abs_diff={d_down:.8}  round-trip max_abs_diff={d_roundtrip:.8}");
    assert!(d_down < 1e-4, "downsampled output diverges from Python reference: {d_down}");
    assert!(d_roundtrip < 1e-4, "round-trip output diverges from Python reference: {d_roundtrip}");
}
