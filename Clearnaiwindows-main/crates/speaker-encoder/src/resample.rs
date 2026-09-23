//! Offline (whole-buffer, non-streaming) resampling to the fbank
//! extractor's required sample rate. Only ever called from
//! `CamPlusSpeakerEncoder::embed`, itself only ever called off the realtime
//! audio thread (see `speaker_profile::SpeakerEncoder::embed`'s docs) - this
//! is exactly the "perform conversion only inside the enrollment/model
//! worker path" placement the enrollment subsystem's design calls for.

use rubato::{FftFixedInOut, Resampler};

use crate::fbank::FBANK_SAMPLE_RATE_HZ;

/// Requested input chunk length hint passed to `FftFixedInOut::new`.
/// **Not** the actual per-call input length: per that constructor's own
/// docs ("desired length of input data in frames, actual value may be
/// different") it rounds this up internally based on
/// `lcm(sample_rate_in, sample_rate_out)` - confirmed by reading rubato
/// 0.14.1's real source (`FftFixedInOut::new` in `src/synchro.rs`), which
/// computes `fft_size_in = ceil(chunk_size_in / min_chunk_out) *
/// (sample_rate_in / gcd)`. For 48000->16000 specifically this rounds a
/// requested 1600 up to an actual 4800. The real, authoritative value is
/// always read back via `Resampler::input_frames_next()` after
/// construction below, rather than assumed from this hint.
const CHUNK_SIZE_HINT: usize = 1600;

/// Resamples `input` (mono `f32`, at `input_rate_hz`) to `FBANK_SAMPLE_RATE_HZ`.
/// A no-op copy if already at that rate.
///
/// Implementation note: input is zero-padded up to a whole number of the
/// resampler's actual required input-chunk length (see
/// `Resampler::input_frames_next`) before being fed through rubato's
/// FFT-based resampler, and the output is truncated back down to the exact
/// ratio-scaled expected length. This means the last fraction-of-a-chunk of
/// real audio is resampled together with some trailing silence padding
/// rather than in true isolation - for a several-second enrollment
/// recording this is an inaudible edge effect, not a correctness concern for
/// the embedding as a whole, but it is a real simplification worth stating
/// plainly rather than silently.
pub fn resample_to_fbank_rate(input: &[f32], input_rate_hz: u32) -> Result<Vec<f32>, String> {
    if input_rate_hz == FBANK_SAMPLE_RATE_HZ {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut resampler = FftFixedInOut::<f32>::new(
        input_rate_hz as usize,
        FBANK_SAMPLE_RATE_HZ as usize,
        CHUNK_SIZE_HINT,
        1, // mono
    )
    .map_err(|e| format!("failed to construct resampler: {e}"))?;
    let chunk_size_in = resampler.input_frames_next();

    let mut output = Vec::new();
    let mut offset = 0usize;
    while offset < input.len() {
        let end = (offset + chunk_size_in).min(input.len());
        let mut chunk = vec![0.0f32; chunk_size_in];
        chunk[..end - offset].copy_from_slice(&input[offset..end]);
        let out_chunks = resampler
            .process(&[chunk], None)
            .map_err(|e| format!("resample failed: {e}"))?;
        output.extend_from_slice(&out_chunks[0]);
        offset = end;
    }

    let expected_len =
        (input.len() as u64 * FBANK_SAMPLE_RATE_HZ as u64 / input_rate_hz as u64) as usize;
    output.truncate(expected_len.min(output.len()));
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(sample_rate: u32, secs: f32, amplitude: f32) -> Vec<f32> {
        (0..(sample_rate as f32 * secs) as usize)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn already_at_target_rate_is_an_identity_copy() {
        let input = tone(16_000, 1.0, 0.3);
        let output = resample_to_fbank_rate(&input, 16_000).unwrap();
        assert_eq!(input, output);
    }

    #[test]
    fn empty_input_produces_empty_output() {
        let output = resample_to_fbank_rate(&[], 48_000).unwrap();
        assert!(output.is_empty());
    }

    #[test]
    fn downsamples_48k_to_16k_with_the_expected_length_ratio() {
        let input = tone(48_000, 2.0, 0.3);
        let output = resample_to_fbank_rate(&input, 48_000).unwrap();
        let expected_len = input.len() / 3;
        // Slack of a few actual (post-rounding) input chunks' worth of
        // output, since the real per-call chunk size is rounded up from
        // `CHUNK_SIZE_HINT` - see that constant's docs.
        let slack = 4 * (CHUNK_SIZE_HINT / 3 + 1);
        assert!(
            (output.len() as i64 - expected_len as i64).abs() <= slack as i64,
            "output.len()={} expected~={}",
            output.len(),
            expected_len
        );
    }

    #[test]
    fn output_is_finite_and_bounded_in_amplitude() {
        let input = tone(48_000, 1.5, 0.5);
        let output = resample_to_fbank_rate(&input, 48_000).unwrap();
        assert!(output.iter().all(|v| v.is_finite()));
        // FFT resampling can slightly overshoot the input peak, but not by
        // an unbounded/nonsensical amount.
        assert!(output.iter().all(|&v| v.abs() < 2.0));
    }
}
