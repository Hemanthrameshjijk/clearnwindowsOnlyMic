//! Runs the REAL "Test with TSE" path (`shared::run_tse_offline`'s exact
//! logic) against real recorded speech (`poc/real_tse/data/`, the same
//! fixture this project's own TSE model selection/verification work
//! already used) - not synthetic tones. Reports real SI-SDR against the
//! known-clean target and writes the actual output WAV for listening.
use speaker_profile::SpeakerEncoder;
use std::sync::Arc;
use tse_engine::TseBackend;

fn read_wav_f32(path: &str) -> (Vec<f32>, u32) {
    let mut reader = hound::WavReader::open(path).unwrap_or_else(|e| panic!("open {path}: {e}"));
    let spec = reader.spec();
    let samples: Vec<f32> = reader.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    (samples, spec.sample_rate)
}

fn write_wav_f32(path: &str, samples: &[f32], sample_rate: u32) {
    let spec = hound::WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: hound::SampleFormat::Int };
    let mut writer = hound::WavWriter::create(path, spec).unwrap();
    for &s in samples {
        writer.write_sample((s.clamp(-1.0, 1.0) * 32767.0) as i16).unwrap();
    }
    writer.finalize().unwrap();
}

/// Real cross-correlation delay search - this project already hit the
/// exact trap of comparing at zero offset once before (the causal
/// resampler's round-trip test originally "failed" at -13.22dB SI-SDR
/// purely from an un-compensated group-delay scoring bug, not a real
/// signal bug - see `causal_resampler.rs`'s docs). The full pipeline here
/// (downsample + STFT win/2 + separator + ISTFT + upsample) has its own
/// real, nonzero group delay; comparing at offset 0 would repeat that
/// exact mistake. Searches a generous window and returns the offset (into
/// `estimate`) that maximizes correlation with `reference`.
fn correlation_at(estimate: &[f32], reference: &[f32], delay: usize, n: usize) -> f64 {
    if delay + n > estimate.len() {
        return f64::NEG_INFINITY;
    }
    estimate[delay..delay + n].iter().zip(reference[..n].iter()).map(|(&e, &r)| e as f64 * r as f64).sum()
}

/// Coarse-then-fine search: a naive full-resolution search over a 1s
/// window would be ~28 billion multiply-adds (too slow) - decimate both
/// signals 32x for a fast coarse pass, then refine at full resolution in a
/// small window around the coarse estimate.
fn find_best_delay(estimate: &[f32], reference: &[f32], max_delay: usize) -> usize {
    const DECIMATE: usize = 32;
    let dec = |s: &[f32]| -> Vec<f32> { s.iter().step_by(DECIMATE).copied().collect() };
    let est_dec = dec(estimate);
    let ref_dec = dec(reference);
    let n_dec = ref_dec.len().min(est_dec.len());
    let max_delay_dec = (max_delay / DECIMATE).max(1);

    let mut best_delay_dec = 0usize;
    let mut best_corr = f64::NEG_INFINITY;
    for delay in 0..max_delay_dec {
        let corr = correlation_at(&est_dec, &ref_dec, delay, n_dec.saturating_sub(max_delay_dec));
        if corr > best_corr {
            best_corr = corr;
            best_delay_dec = delay;
        }
    }

    let coarse_delay = best_delay_dec * DECIMATE;
    let refine_window = DECIMATE * 2;
    let refine_start = coarse_delay.saturating_sub(refine_window);
    let refine_end = (coarse_delay + refine_window).min(max_delay);
    let n_full = reference.len().min(estimate.len().saturating_sub(refine_end));

    let mut best_delay = coarse_delay;
    let mut best_corr_full = f64::NEG_INFINITY;
    for delay in refine_start..=refine_end {
        let corr = correlation_at(estimate, reference, delay, n_full);
        if corr > best_corr_full {
            best_corr_full = corr;
            best_delay = delay;
        }
    }
    best_delay
}

fn si_sdr_db(estimate: &[f32], reference: &[f32]) -> f32 {
    let n = estimate.len().min(reference.len());
    let est = &estimate[..n];
    let ref_ = &reference[..n];
    let dot: f64 = est.iter().zip(ref_.iter()).map(|(&e, &r)| e as f64 * r as f64).sum();
    let ref_energy: f64 = ref_.iter().map(|&r| (r as f64) * (r as f64)).sum();
    if ref_energy < 1e-12 {
        return f32::NEG_INFINITY;
    }
    let alpha = dot / ref_energy;
    let target: Vec<f64> = ref_.iter().map(|&r| alpha * r as f64).collect();
    let noise: Vec<f64> = est.iter().zip(target.iter()).map(|(&e, &t)| e as f64 - t).collect();
    let target_energy: f64 = target.iter().map(|&t| t * t).sum();
    let noise_energy: f64 = noise.iter().map(|&n| n * n).sum();
    if noise_energy < 1e-12 {
        return f32::INFINITY;
    }
    (10.0 * (target_energy / noise_energy).log10()) as f32
}

fn main() {
    let base = "poc/real_tse/data";

    // 1. Enroll: real 192-dim ECAPA embedding from the real target
    // reference clip, via the real encoder - exactly what enrollment does.
    let (reference_16k, reference_sr) = read_wav_f32(&format!("{base}/reference.wav"));
    println!("Loaded reference.wav: {} samples @ {}Hz", reference_16k.len(), reference_sr);

    let encoder = tse_bsrnn::tse_speaker_encoder::TseSpeakerEncoder::new().expect("ECAPA encoder must load");
    let profile = encoder.embed(&reference_16k, reference_sr).expect("embedding extraction must succeed");
    println!("Real embedding extracted: dim={}", profile.dimension());

    // 2. Run the REAL run_tse_offline logic against the real 48kHz mixture.
    let (mixture_48k, mix_sr) = read_wav_f32(&format!("{base}/mixture_48k.wav"));
    assert_eq!(mix_sr, 48_000);
    println!("Loaded mixture_48k.wav: {} samples @ {}Hz ({:.1}s)", mixture_48k.len(), mix_sr, mixture_48k.len() as f32 / 48000.0);

    let half_cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2).max(2) / 2;
    tse_bsrnn::separator::set_max_parallelism_hint(Some(half_cores.max(1)));

    let mut backend = tse_bsrnn::backend::RealBsrnnTseBackend::try_load().expect("real backend must load");
    backend.set_target_speaker(Some(Arc::new(profile)));

    let mut samples = mixture_48k.clone();
    const TAIL_PAD_SAMPLES: usize = 24_000;
    samples.extend(std::iter::repeat(0.0f32).take(TAIL_PAD_SAMPLES));

    const CHUNK: usize = 480;
    let mut output = Vec::with_capacity(samples.len());
    let mut i = 0;
    let mut raw_passthrough_frames = 0usize;
    let mut processed_frames = 0usize;
    while i < samples.len() {
        let end = (i + CHUNK).min(samples.len());
        let real_len = end - i;
        let mut frame = vec![0.0f32; CHUNK];
        frame[..real_len].copy_from_slice(&samples[i..end]);
        match backend.process_frame(&mut frame) {
            Ok(()) => processed_frames += 1,
            Err(_) => raw_passthrough_frames += 1,
        }
        output.extend_from_slice(&frame[..real_len]);
        i = end;
    }
    println!("\nFrames: {processed_frames} processed, {raw_passthrough_frames} raw passthrough (warm-up/failures)");

    write_wav_f32("tse_offline_output_48k.wav", &output, 48_000);
    println!("Wrote tse_offline_output_48k.wav ({} samples)", output.len());

    // 3. Real objective quality check: SI-SDR of the TSE output against
    // the known-clean target speaker audio, compared to the SI-SDR of the
    // UNPROCESSED mixture against the same target (the real "how much did
    // this actually help" baseline).
    let (target_clean_48k, _) = read_wav_f32(&format!("{base}/target_clean_48k.wav"));

    // Real delay search, not an assumed offset - "before" (raw mixture) has
    // zero pipeline delay (it's the direct 16k->48k upsample of the exact
    // same audio target_clean_48k was derived from), so search a small
    // window there just to confirm that; "after" (TSE output) has gone
    // through the resampler + STFT/ISTFT chain and needs a real search.
    let delay_before = find_best_delay(&mixture_48k, &target_clean_48k, 100);
    let delay_after = find_best_delay(&output, &target_clean_48k, 48_000); // search up to 1s
    println!("\nBest-alignment delay: before={delay_before} samples, after={delay_after} samples ({:.1}ms)", delay_after as f32 / 48.0);

    let sdr_before = si_sdr_db(&mixture_48k[delay_before..], &target_clean_48k);
    let sdr_after = si_sdr_db(&output[delay_after..], &target_clean_48k);
    println!("SI-SDR before (raw mixture vs. clean target): {sdr_before:.2} dB");
    println!("SI-SDR after  (TSE output vs. clean target, delay-compensated):  {sdr_after:.2} dB");
    println!("Improvement: {:.2} dB", sdr_after - sdr_before);

    // 4. Basic sanity: is the output amplitude blown up (clipping/scale
    // bug) relative to the input?
    let max_in = mixture_48k.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let max_out = output.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
    let rms_in = (mixture_48k.iter().map(|v| v * v).sum::<f32>() / mixture_48k.len() as f32).sqrt();
    let rms_out = (output.iter().map(|v| v * v).sum::<f32>() / output.len() as f32).sqrt();
    println!("\nPeak amplitude: input={max_in:.4} output={max_out:.4}");
    println!("RMS amplitude: input={rms_in:.4} output={rms_out:.4}");
    let clipped_samples = output.iter().filter(|&&v| v.abs() >= 0.999).count();
    println!("Output samples at/above clipping threshold: {clipped_samples} / {}", output.len());
}
