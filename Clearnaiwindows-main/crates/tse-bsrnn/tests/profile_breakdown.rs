//! Manual profiling harness (not a correctness test - `#[ignore]`d by
//! default). Runs the same real 12s two-speaker mixture used by
//! `engine_e2e_matches_python.rs`, at the same 10ms chunking, but times
//! each stage of `StreamingTseEngine::process_chunk` separately using
//! `std::time::Instant` instead of calling the assembled engine as a black
//! box - this is step 0 of the TSE optimization plan: find out where the
//! measured RTF~21x is actually going before changing any code.
//!
//! Run with: `cargo test --release -p tse-bsrnn --test profile_breakdown -- --ignored --nocapture`

use std::time::{Duration, Instant};
use tse_bsrnn::separator::BsrnnSeparator;
use tse_bsrnn::speaker_fusion::project_embedding;
use tse_bsrnn::stft::{StreamingIstft, StreamingStft};

fn read_f32_bin(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("failed to read fixture {path}: {e}"));
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

#[test]
#[ignore]
fn profile_stage_breakdown() {
    profile_with_chunk_size(160); // 10ms @ 16kHz - the real streaming call pattern today
}

#[test]
#[ignore]
fn profile_stage_breakdown_80ms_chunks() {
    profile_with_chunk_size(1280); // 80ms @ 16kHz - simulated larger-buffer call pattern
}

#[test]
#[ignore]
fn profile_stage_breakdown_300ms_chunks() {
    profile_with_chunk_size(4800); // 300ms @ 16kHz - simulated larger-buffer call pattern
}

fn profile_with_chunk_size(chunk_samples: usize) {
    let mixture = read_f32_bin("tests/fixtures/e2e_mixture.bin");
    let embedding_vec = read_f32_bin("tests/fixtures/e2e_target_embedding.bin");
    assert_eq!(embedding_vec.len(), 192);
    let mut embedding = [0.0f32; 192];
    embedding.copy_from_slice(&embedding_vec);

    let separator = BsrnnSeparator::load().expect("separator weights must load");
    let win = separator.win();
    let hop = separator.stride();
    let num_repeat = separator.num_repeat();
    let gate = project_embedding(&embedding);
    let mut state = separator.new_state();
    let mut stft = StreamingStft::new(win, hop);
    let mut istft = StreamingIstft::new(win, hop);

    let mut t_stft = Duration::ZERO;
    let mut t_band_split = Duration::ZERO;
    let mut t_gate = Duration::ZERO;
    let mut t_band_rnn = Duration::ZERO;
    let mut t_band_comm = Duration::ZERO;
    let mut t_masker = Duration::ZERO;
    let mut t_istft = Duration::ZERO;

    let mut i = 0;
    let mut total_frames = 0usize;
    let wall_start = Instant::now();

    while i < mixture.len() {
        let end = (i + chunk_samples).min(mixture.len());
        let chunk = &mixture[i..end];
        i = end;

        let t0 = Instant::now();
        let (spec_real, spec_imag, t) = stft.push(chunk);
        t_stft += t0.elapsed();
        if t == 0 {
            continue;
        }
        total_frames += t;

        let t0 = Instant::now();
        let mut x = separator.band_split_and_subband_norm(&spec_real, &spec_imag, t);
        t_band_split += t0.elapsed();

        let t0 = Instant::now();
        separator.apply_speaker_gate(&mut x, &gate, t);
        t_gate += t0.elapsed();

        for r in 0..num_repeat {
            let t0 = Instant::now();
            x = separator.run_band_rnn_for_block(r, &x, &mut state[r], t);
            t_band_rnn += t0.elapsed();

            let t0 = Instant::now();
            x = separator.run_band_comm_for_block(r, &x, t);
            t_band_comm += t0.elapsed();
        }

        let t0 = Instant::now();
        let (est_real, est_imag) = separator.run_band_masker(&x, &spec_real, &spec_imag, t);
        t_masker += t0.elapsed();

        let t0 = Instant::now();
        let _out = istft.push(&est_real, &est_imag, t);
        t_istft += t0.elapsed();
    }

    let wall = wall_start.elapsed();
    let audio_secs = mixture.len() as f64 / 16000.0;
    let rtf = wall.as_secs_f64() / audio_secs;

    let stages: [(&str, Duration); 7] = [
        ("stft", t_stft),
        ("band_split_and_subband_norm", t_band_split),
        ("apply_speaker_gate", t_gate),
        ("band_rnn (all 6 blocks)", t_band_rnn),
        ("band_comm (all 6 blocks)", t_band_comm),
        ("band_masker", t_masker),
        ("istft", t_istft),
    ];

    println!("\n=== TSE separator stage breakdown (chunk_samples={chunk_samples}, ~{:.0}ms/call) ===", chunk_samples as f64 / 16.0);
    println!("audio: {audio_secs:.2}s ({total_frames} STFT frames), wall time: {:.2}s, RTF: {rtf:.2}x", wall.as_secs_f64());
    for (name, d) in &stages {
        let pct = 100.0 * d.as_secs_f64() / wall.as_secs_f64();
        println!("  {name:32} {:>8.3}s  ({pct:5.1}%)", d.as_secs_f64());
    }
    let per_frame_comm_us = t_band_comm.as_secs_f64() * 1_000_000.0 / total_frames as f64 / num_repeat as f64;
    println!(
        "\nband_comm: {per_frame_comm_us:.1}us per (frame x repeat-block) = {:.1}us per single band_comm LSTM step (64 steps: 32 fwd + 32 bwd)",
        per_frame_comm_us / 64.0
    );
}
