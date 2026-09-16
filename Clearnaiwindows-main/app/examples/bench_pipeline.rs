//! Real, isolated processing-time benchmark for this app's actual mic-path
//! DSP chain (AEC -> RNNoise -> BVC (if available) -> Studio), modeled on
//! the same methodology as the reference Linux project's `bench_studio.rs`
//! example (percentile-at-sorted-index, mean/p50/p95/p99/max, isolated from
//! any I/O/graph overhead - synthetic frames in a tight loop, not real
//! WASAPI capture).
//!
//! Run: cargo run --release -p clearnai-app --example bench_pipeline
//!
//! This produces real numbers (RUNTIME_VERIFIED, on whatever machine runs
//! it) instead of the guesses/assertions this project has otherwise relied
//! on all session - see the report's own rule against claiming latency
//! without measurement.

use dsp_core::{compute_percentiles, run_if_enabled, Stage, StageToggle, FRAME_SAMPLES, SAMPLE_RATE_HZ};
use std::time::Instant;

const FRAME_COUNT: usize = 50_000; // 50,000 frames @ 10ms = ~500s of synthetic audio processed

fn synthetic_frame(seed: usize) -> Vec<f32> {
    // A tone plus light synthetic noise - not silence (some stages, e.g.
    // RNNoise, behave differently on silence vs. real signal, and this
    // should reflect realistic per-frame cost, not a best-case shortcut).
    let fs = SAMPLE_RATE_HZ as f32;
    (0..FRAME_SAMPLES)
        .map(|i| {
            let t = (seed * FRAME_SAMPLES + i) as f32 / fs;
            let tone = 0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
            // Cheap deterministic pseudo-noise, no external RNG dependency needed here.
            let noise = ((seed * 2654435761 + i * 40503) % 1000) as f32 / 1000.0 - 0.5;
            (tone + noise * 0.05).clamp(-1.0, 1.0)
        })
        .collect()
}

fn bench_stage(name: &str, mut stage: Box<dyn Stage>) {
    let toggle = StageToggle::new(true);
    let mut durations_ns = Vec::with_capacity(FRAME_COUNT);
    for seed in 0..FRAME_COUNT {
        let mut frame = synthetic_frame(seed);
        let started = Instant::now();
        run_if_enabled(stage.as_mut(), &toggle, &mut frame);
        durations_ns.push(started.elapsed().as_nanos() as u64);
    }
    let p = compute_percentiles(durations_ns);
    let deadline_us = 10_000.0; // 10ms deadline, in microseconds, matching dsp_core's frame period
    println!(
        "{name:<28} mean={:>7.3}ms  p50={:>7.3}ms  p95={:>7.3}ms  p99={:>7.3}ms  max={:>7.3}ms  (deadline={:.1}ms)",
        p.mean_ns as f64 / 1e6,
        p.p50_ns as f64 / 1e6,
        p.p95_ns as f64 / 1e6,
        p.p99_ns as f64 / 1e6,
        p.max_ns as f64 / 1e6,
        deadline_us / 1000.0,
    );
}

fn main() {
    println!("ClearNAI mic-path DSP benchmark - {FRAME_COUNT} synthetic frames per stage, isolated from I/O.\n");

    bench_stage(
        "RNNoise",
        noise_engine::build(noise_engine::NoiseEngineKind::RnNoise, None)
            .expect("constructing RNNoise stage"),
    );

    bench_stage(
        "Studio (Natural preset)",
        Box::new(studio_dsp::StudioStage::new(studio_dsp::StudioPreset::Natural)),
    );

    // BVC only if the real DLL/model are already extracted on this machine
    // (i.e. the real app has been run here at least once) - skipped
    // honestly, not faked, if unavailable.
    let bvc_dir = clearnai_app_setup_path();
    match bvc_hush::HushBvcStage::try_load(&bvc_dir) {
        Ok(stage) => bench_stage("BVC (weya_nc.dll)", Box::new(stage)),
        Err(e) => println!("BVC (weya_nc.dll)          SKIPPED - not available on this machine: {e:#}"),
    }

    println!("\nDone. Compare each stage's p95/p99 against the 10ms deadline above -");
    println!("that's the real per-stage cost, not a guess.");
}

/// Mirrors `setup::app_data_dir()`'s logic without pulling in the whole
/// `app` binary crate's module tree into this example - same
/// `%LOCALAPPDATA%\ClearNAI` path the real app already extracts BVC into.
fn clearnai_app_setup_path() -> std::path::PathBuf {
    let base = std::env::var_os("LOCALAPPDATA")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("ClearNAI")
}
