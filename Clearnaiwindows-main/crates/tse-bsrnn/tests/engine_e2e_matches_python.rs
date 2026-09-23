//! The capstone Phase C verification: runs the assembled Rust
//! `tse_bsrnn::engine::StreamingTseEngine` on the exact same real
//! two-speaker mixture used throughout `poc/real_tse/`, at the same 10ms
//! chunk size, using the same real cached target-speaker embedding, and
//! compares against the already-verified Python `StreamingTseEngine`'s own
//! output (see `poc/real_tse/export/export_engine_e2e_fixture.py`) - which
//! itself was proven, in the original POC, to match the whole-utterance
//! reference model to SI-SDR 41-45dB / correlation 0.9999+ on real speech.
//! A match here means the Rust port inherits all of that real evidence.

use tse_bsrnn::engine::StreamingTseEngine;
use tse_bsrnn::separator::BsrnnSeparator;

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn correlation(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len()) as f32;
    let mean_a = a.iter().sum::<f32>() / n;
    let mean_b = b.iter().sum::<f32>() / n;
    let mut cov = 0.0f32;
    let mut var_a = 0.0f32;
    let mut var_b = 0.0f32;
    for i in 0..a.len().min(b.len()) {
        let da = a[i] - mean_a;
        let db = b[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    cov / (var_a.sqrt() * var_b.sqrt())
}

fn si_sdr(estimate: &[f32], reference: &[f32]) -> f32 {
    let n = estimate.len().min(reference.len());
    let (est, ref_) = (&estimate[..n], &reference[..n]);
    let dot: f32 = est.iter().zip(ref_).map(|(e, r)| e * r).sum();
    let ref_energy: f32 = ref_.iter().map(|r| r * r).sum::<f32>().max(1e-12);
    let scale = dot / ref_energy;
    let target: Vec<f32> = ref_.iter().map(|r| r * scale).collect();
    let noise: Vec<f32> = est.iter().zip(&target).map(|(e, t)| e - t).collect();
    let target_energy: f32 = target.iter().map(|t| t * t).sum::<f32>().max(1e-12);
    let noise_energy: f32 = noise.iter().map(|n| n * n).sum::<f32>().max(1e-12);
    10.0 * (target_energy / noise_energy).log10()
}

#[test]
fn rust_engine_10ms_chunks_matches_python_streaming_engine() {
    let mixture = read_f32_bin("tests/fixtures/e2e_mixture.bin");
    let embedding_vec = read_f32_bin("tests/fixtures/e2e_target_embedding.bin");
    let expected_output = read_f32_bin("tests/fixtures/e2e_output_10ms.bin");
    assert_eq!(embedding_vec.len(), 192);
    let mut embedding = [0.0f32; 192];
    embedding.copy_from_slice(&embedding_vec);

    let separator = BsrnnSeparator::load().expect("separator weights must load");
    let mut engine = StreamingTseEngine::new(separator, &embedding);

    let chunk_samples = 160; // 10ms @ 16kHz
    let mut output = Vec::new();
    let mut i = 0;
    while i < mixture.len() {
        let end = (i + chunk_samples).min(mixture.len());
        let out = engine.process_chunk(&mixture[i..end]);
        output.extend_from_slice(&out);
        i = end;
    }

    println!("Rust engine output: {} samples (Python streaming engine: {} samples)", output.len(), expected_output.len());

    let n = output.len().min(expected_output.len());
    let corr = correlation(&output[..n], &expected_output[..n]);
    let sdr = si_sdr(&output[..n], &expected_output[..n]);
    println!("Rust vs Python streaming engine (10ms chunks, real 2-speaker mixture): correlation={corr:.6}  SI-SDR={sdr:.2} dB (n={n})");

    assert!(corr > 0.999, "Rust engine output diverges from the Python streaming engine: correlation={corr}");
    assert!(sdr > 30.0, "Rust engine output diverges from the Python streaming engine: SI-SDR={sdr}dB");
}
