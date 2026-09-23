//! Diagnostic: runs `shared::run_tse_offline`'s exact chunking logic
//! against a synthetic mixture, reporting where warm-up ends and whether
//! any mid-stream frame after warm-up still errors (raw passthrough
//! spliced into otherwise-processed audio - a real source of audible
//! "glitchy" discontinuities, not a hypothesis).
use speaker_profile::{NormalizationKind, TargetSpeakerProfile};
use std::sync::Arc;
use tse_engine::TseBackend;

fn main() {
    let half_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).max(2) / 2;
    tse_bsrnn::separator::set_max_parallelism_hint(Some(half_cores.max(1)));

    let mut backend = tse_bsrnn::backend::RealBsrnnTseBackend::try_load().expect("real backend must load");
    let embedding = vec![0.05f32; 192];
    let profile = TargetSpeakerProfile::new(embedding, "diag", 16_000, NormalizationKind::None).unwrap();
    backend.set_target_speaker(Some(Arc::new(profile)));

    // 3 seconds of a synthetic two-tone "mixture" @48kHz.
    let sr = 48_000;
    let secs = 3.0;
    let n = (sr as f32 * secs) as usize;
    let mut samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / sr as f32;
            0.3 * (2.0 * std::f32::consts::PI * 220.0 * t).sin() + 0.2 * (2.0 * std::f32::consts::PI * 440.0 * t).sin()
        })
        .collect();
    samples.extend(std::iter::repeat(0.0f32).take(24_000));

    const CHUNK: usize = 480;
    let mut i = 0;
    let mut frame_idx = 0usize;
    let mut warmup_end: Option<usize> = None;
    let mut errors_after_warmup = 0usize;
    let mut nan_count = 0usize;
    let mut prev_last_sample: Option<f32> = None;
    let mut max_jump_at_warmup_boundary = 0.0f32;

    while i < samples.len() {
        let end = (i + CHUNK).min(samples.len());
        let real_len = end - i;
        let mut frame = vec![0.0f32; CHUNK];
        frame[..real_len].copy_from_slice(&samples[i..end]);
        let result = backend.process_frame(&mut frame);

        match &result {
            Ok(()) => {
                if warmup_end.is_none() {
                    warmup_end = Some(frame_idx);
                    println!("Warm-up ended at frame {frame_idx} (~{:.1}ms of audio)", frame_idx as f32 * 10.0);
                    if let Some(prev) = prev_last_sample {
                        let jump = (frame[0] - prev).abs();
                        max_jump_at_warmup_boundary = jump;
                        println!("Sample discontinuity at warm-up boundary: {jump:.4}");
                    }
                }
            }
            Err(_) => {
                if warmup_end.is_some() {
                    errors_after_warmup += 1;
                }
            }
        }
        if frame.iter().any(|v| !v.is_finite()) {
            nan_count += 1;
        }
        prev_last_sample = frame.last().copied();
        i = end;
        frame_idx += 1;
    }

    println!("\nTotal frames: {frame_idx}");
    println!("Warm-up end: {warmup_end:?}");
    println!("Errors AFTER warm-up (mid-stream raw-passthrough splices): {errors_after_warmup}");
    println!("Frames with non-finite samples: {nan_count}");
    println!("Max sample jump at warm-up boundary: {max_jump_at_warmup_boundary:.4}");
}
