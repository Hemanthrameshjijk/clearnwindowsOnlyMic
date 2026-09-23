use tse_bsrnn::speaker_fusion::project_embedding;

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
fn fusion_projection_matches_real_pytorch_reference() {
    let embedding_vec = read_f32_bin("tests/fixtures/ecapa_embedding.bin");
    assert_eq!(embedding_vec.len(), 192);
    let mut embedding = [0.0f32; 192];
    embedding.copy_from_slice(&embedding_vec);

    let expected_gate = read_f32_bin("tests/fixtures/fusion_gate_expected.bin");
    assert_eq!(expected_gate.len(), 128);

    let gate = project_embedding(&embedding);
    let max_abs_diff = gate.iter().zip(&expected_gate).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("fusion projection: max_abs_diff={max_abs_diff:.8}");
    assert!(max_abs_diff < 1e-4, "fusion projection diverges from PyTorch reference: {max_abs_diff}");
}
