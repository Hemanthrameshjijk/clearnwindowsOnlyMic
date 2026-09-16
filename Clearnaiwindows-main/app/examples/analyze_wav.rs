//! Diagnostic (NOT product code): reads a mono 48kHz f32 WAV file (e.g.
//! `out.wav` captured via `transport-reader`, the real processed output of
//! the live mic pipeline) and runs it through a *second*, fresh RNNoise
//! pass to check whether the file was already suppressed once.
//!
//! Rationale: if the pipeline's RNNoise stage really ran on this audio
//! already, a second pass should barely change its RMS (ratio close to
//! 1.0 - nothing left to remove). If the file is actually still raw/
//! unprocessed noise, a second pass will show the same strong reduction
//! `noise-engine`'s own correctness tests measure on synthetic noise
//! (ratio around ~0.6-0.8) - proving RNNoise never actually ran on it live.
//!
//! Run: cargo run --release -p clearnai-app --example analyze_wav -- out.wav

use dsp_core::FRAME_SAMPLES;

fn rms(xs: &[f32]) -> f32 {
    (xs.iter().map(|x| x * x).sum::<f32>() / xs.len() as f32).sqrt()
}

fn main() {
    let path = std::env::args().nth(1).unwrap_or_else(|| "out.wav".to_string());
    let mut reader = hound::WavReader::open(&path).unwrap_or_else(|e| panic!("opening {path}: {e}"));
    let spec = reader.spec();
    println!(
        "Loaded '{path}': {} Hz, {} channel(s), {:?}",
        spec.sample_rate, spec.channels, spec.sample_format
    );

    let samples: Vec<f32> = reader
        .samples::<f32>()
        .collect::<Result<Vec<_>, _>>()
        .expect("reading f32 samples");

    let overall_in_rms = rms(&samples);
    println!("Whole-file input RMS: {overall_in_rms:.6}");

    let mut engine = noise_engine::build(noise_engine::NoiseEngineKind::RnNoise, None)
        .expect("constructing a fresh RNNoise stage for the second pass");

    let mut in_sq_sum = 0.0f64;
    let mut out_sq_sum = 0.0f64;
    let mut counted = 0usize;
    let n_frames = samples.len() / FRAME_SAMPLES;
    let warmup_frames = 20.min(n_frames / 4);

    for i in 0..n_frames {
        let mut frame = samples[i * FRAME_SAMPLES..(i + 1) * FRAME_SAMPLES].to_vec();
        let this_in_rms = rms(&frame);
        engine.process(&mut frame);
        let this_out_rms = rms(&frame);
        if i >= warmup_frames {
            in_sq_sum += (this_in_rms as f64).powi(2);
            out_sq_sum += (this_out_rms as f64).powi(2);
            counted += 1;
        }
    }

    let in_rms = (in_sq_sum / counted as f64).sqrt();
    let out_rms = (out_sq_sum / counted as f64).sqrt();
    let ratio = out_rms / in_rms;

    println!("\nSecond-pass steady-state RMS (post-warmup, {counted} frames):");
    println!("  in_rms  = {in_rms:.6}");
    println!("  out_rms = {out_rms:.6}");
    println!("  ratio   = {ratio:.4}");
    println!();
    if ratio < 0.85 {
        println!(
            "==> ratio well below 1.0 (noise-engine's own tests see ~0.66-0.87 on real noise) - \
             this file still had real noise to suppress, meaning RNNoise likely did NOT run \
             effectively on it the first time through the live pipeline. Real bug, not just \
             'not enough suppression'."
        );
    } else {
        println!(
            "==> ratio close to 1.0 - a second RNNoise pass found little left to remove, meaning \
             the pipeline's first pass already did real work. If it still sounds noisy to you, \
             this is RNNoise's real, partial (not total) suppression ceiling on this specific \
             noise, not a bug."
        );
    }
}
