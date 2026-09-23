//! Pure, deterministic VAD/quality checks run on a completed reference
//! recording before it is ever handed to a speaker encoder. No ML model
//! here on purpose: these are cheap, explainable sanity checks (energy,
//! clipping, a simple fixed-threshold speech/silence split per short frame),
//! not a "real" voice-activity-detector - good enough to reject an empty/
//! silent/clipped recording with a specific, actionable reason, without
//! risking false rejections of an ordinary, slightly quiet microphone
//! recording (see this module's docs on why thresholds are deliberately
//! lenient).

use crate::error::EnrollmentError;

/// Tunable thresholds for `validate`. `Default` gives deliberately lenient
/// values - see each field's docs - because the task this crate was built
/// for explicitly warns against "overly aggressive rejection that makes
/// normal microphone recordings fail."
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ValidationThresholds {
    /// Minimum recording duration, in seconds, before it's even worth
    /// running the rest of the checks.
    pub min_duration_secs: f32,
    /// Minimum whole-recording RMS level (samples in `[-1.0, 1.0]`) below
    /// which the recording is considered too quiet to be a usable
    /// reference, even if some frames technically exceeded the per-frame
    /// speech-energy threshold below.
    pub min_rms: f32,
    /// A sample at or beyond this absolute value counts as clipped.
    pub clip_threshold: f32,
    /// Fraction of all samples that may be clipped before the recording is
    /// rejected outright.
    pub max_clipped_ratio: f32,
    /// Per-frame (see `speech_frame_ratio`'s `frame_len` parameter) energy
    /// threshold above which a frame counts as "speech-like."
    pub speech_energy_threshold: f32,
    /// Minimum fraction of frames that must count as speech-like.
    pub min_speech_frame_ratio: f32,
    /// Minimum acceptable gap (in dB) between the recording's loudest and
    /// quietest frames - see `estimate_snr_db`'s docs. Deliberately lenient
    /// (a real quiet room still has *some* gap between silence and speech;
    /// this only needs to catch a genuinely noisy background, not penalize
    /// an ordinary recording for not being anechoic).
    pub min_snr_db: f32,
}

impl Default for ValidationThresholds {
    fn default() -> Self {
        Self {
            min_duration_secs: 2.0,
            min_rms: 0.01,
            clip_threshold: 0.99,
            max_clipped_ratio: 0.01,
            speech_energy_threshold: 0.02,
            min_speech_frame_ratio: 0.15,
            // 6dB is deliberately lenient - a typical quiet-room recording
            // measures well above this (silence-vs-speech RMS gaps of
            // 15-30dB are common); this is meant to catch a genuinely loud,
            // steady background (TV, traffic, HVAC), not fail an ordinary
            // recording for having *some* ambient room tone.
            min_snr_db: 6.0,
        }
    }
}

/// Root-mean-square level of `audio`. `0.0` for an empty slice (callers must
/// check duration/emptiness separately - this function never divides by
/// zero).
pub fn rms(audio: &[f32]) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let sum_sq: f64 = audio.iter().map(|&s| (s as f64) * (s as f64)).sum();
    ((sum_sq / audio.len() as f64).sqrt()) as f32
}

/// Fraction of samples whose absolute value is at or beyond `threshold`.
pub fn clipping_ratio(audio: &[f32], threshold: f32) -> f32 {
    if audio.is_empty() {
        return 0.0;
    }
    let clipped = audio.iter().filter(|&&s| s.abs() >= threshold).count();
    clipped as f32 / audio.len() as f32
}

/// Splits `audio` into fixed-size (`frame_len` samples) frames and returns
/// the fraction whose RMS exceeds `energy_threshold` - a simple, fully
/// deterministic stand-in for a real VAD, sufficient to distinguish "mostly
/// silence" from "mostly speech-level energy" without any model/false-start
/// risk. The final partial frame (if any) is included as-is (not padded or
/// dropped), so a recording that isn't an exact multiple of `frame_len`
/// still contributes its trailing samples to the ratio.
pub fn speech_frame_ratio(audio: &[f32], frame_len: usize, energy_threshold: f32) -> f32 {
    if audio.is_empty() || frame_len == 0 {
        return 0.0;
    }
    let mut total = 0usize;
    let mut speech = 0usize;
    for chunk in audio.chunks(frame_len) {
        total += 1;
        if rms(chunk) > energy_threshold {
            speech += 1;
        }
    }
    speech as f32 / total as f32
}

/// Rough, explainable stand-in for signal-to-noise ratio:
/// splits `audio` into `frame_len`-sized frames and classifies each one
/// the exact same way `speech_frame_ratio` does (RMS above
/// `energy_threshold` = "speech-like", at/below it = "non-speech"), then
/// returns `20*log10(speech_mean / nonspeech_mean)` - the gap between how
/// loud the speech-classified frames are and how loud the non-speech
/// (background/silence) frames are.
///
/// Real world: this only means something when the recording actually
/// *has* non-speech frames to measure a background level from (any
/// realistic recording does - even continuous speech has micro-pauses,
/// and `session.rs`'s own auto-stop logic always leaves a genuine silent
/// tail). If every single frame classifies as "speech" (no non-speech
/// frames exist at all to measure a background from - e.g. a wall-to-wall
/// continuous tone with no gaps, which real recordings essentially never
/// are), there is no evidence of a noise floor one way or the other, so
/// this returns `f32::INFINITY` (never flagged as noisy) rather than
/// comparing two overlapping "loud" groups and manufacturing a false
/// "0dB, must be noisy" result - an earlier version of this function did
/// exactly that (quietest-N%-vs-loudest-N% of ALL frames, regardless of
/// classification) and it flagged a perfectly clean, continuous test tone
/// as "too noisy" purely because a gapless signal has no quiet frames to
/// compare against - a real, caught false positive, not a hypothetical
/// one.
///
/// This is NOT a rigorous SNR measurement (no spectral analysis) - a
/// cheap, deterministic, explainable check in the same spirit as this
/// module's other checks, good enough to catch a recording where a steady
/// background (noise, a TV, another conversation) is nearly as loud as
/// the speech itself. It does **not** reliably distinguish a second HUMAN
/// VOICE from the target speaker's own - a second talker registers as
/// "speech-like" just like the real speaker, so this check alone cannot
/// catch "someone else was also talking" (see `EnrollmentError::TooNoisy`'s
/// docs).
pub fn estimate_snr_db(audio: &[f32], frame_len: usize, energy_threshold: f32) -> f32 {
    if audio.is_empty() || frame_len == 0 {
        return f32::INFINITY;
    }

    let mut speech_sum = 0.0f64;
    let mut speech_count = 0usize;
    let mut nonspeech_sum = 0.0f64;
    let mut nonspeech_count = 0usize;

    for chunk in audio.chunks(frame_len) {
        let r = rms(chunk);
        if r > energy_threshold {
            speech_sum += r as f64;
            speech_count += 1;
        } else {
            nonspeech_sum += r as f64;
            nonspeech_count += 1;
        }
    }

    // No non-speech frames at all: no evidence of a noise floor, so this
    // cannot be flagged as noisy from this signal alone - see docs above.
    if nonspeech_count == 0 || speech_count == 0 {
        return f32::INFINITY;
    }

    let speech_mean = (speech_sum / speech_count as f64).max(1e-9);
    // A near-silent noise floor (true digital silence) would make this
    // ratio blow up to a meaningless huge number - floor it at a tiny
    // epsilon rather than divide by zero for an otherwise-clean recording.
    let nonspeech_mean = (nonspeech_sum / nonspeech_count as f64).max(1e-9);

    (20.0 * (speech_mean / nonspeech_mean).log10()) as f32
}

/// Runs every check in order, returning the first (most fundamental)
/// failure reason rather than all of them at once - matches the task's
/// requirement for "explicit failure reasons" without requiring a caller to
/// juggle multiple simultaneous errors for what is, in practice, always a
/// single "try again" action from the user.
///
/// Order matters: duration is checked first (a 200ms clip is not
/// meaningfully "too quiet" or "clipped", it's just too short to judge at
/// all), then overall level, then clipping, then speech presence - the last
/// check catches a recording that's a *reasonable* level throughout (so it
/// isn't "too quiet") but never actually contains speech-like energy bursts
/// (e.g. steady room tone/hum picked up while the user stayed silent).
pub fn validate(audio: &[f32], sample_rate_hz: u32, thresholds: &ValidationThresholds) -> Result<(), EnrollmentError> {
    if sample_rate_hz == 0 {
        return Err(EnrollmentError::RecordingError("sample rate is zero".to_string()));
    }
    let duration_secs = audio.len() as f32 / sample_rate_hz as f32;
    if duration_secs < thresholds.min_duration_secs {
        return Err(EnrollmentError::TooShort);
    }

    if rms(audio) < thresholds.min_rms {
        return Err(EnrollmentError::TooQuiet);
    }

    if clipping_ratio(audio, thresholds.clip_threshold) > thresholds.max_clipped_ratio {
        return Err(EnrollmentError::Clipped);
    }

    // 20ms frames at the given sample rate - fine-grained enough to catch a
    // genuinely all-silent recording without being so fine-grained that a
    // few noise-floor samples inside an otherwise-silent frame flip it.
    let frame_len = ((sample_rate_hz as f32) * 0.020).round().max(1.0) as usize;
    if speech_frame_ratio(audio, frame_len, thresholds.speech_energy_threshold) < thresholds.min_speech_frame_ratio {
        return Err(EnrollmentError::NoSpeech);
    }

    // Checked last: a recording can pass every check above (long enough,
    // loud enough, not clipped, has speech-like bursts) and still have a
    // steady, loud background noise underneath those bursts the earlier
    // checks can't see - see `estimate_snr_db`'s docs for exactly what
    // this does and doesn't catch.
    if estimate_snr_db(audio, frame_len, thresholds.speech_energy_threshold) < thresholds.min_snr_db {
        return Err(EnrollmentError::TooNoisy);
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silence(sample_rate: u32, secs: f32) -> Vec<f32> {
        vec![0.0f32; (sample_rate as f32 * secs) as usize]
    }

    fn tone(sample_rate: u32, secs: f32, amplitude: f32) -> Vec<f32> {
        (0..(sample_rate as f32 * secs) as usize)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
            .collect()
    }

    #[test]
    fn valid_speech_like_signal_passes() {
        let audio = tone(48_000, 3.0, 0.3);
        assert!(validate(&audio, 48_000, &ValidationThresholds::default()).is_ok());
    }

    /// Real speech is not gapless - alternates ON (speech tone + noise) and
    /// OFF (noise alone) in 200ms segments, standing in for natural pauses
    /// between words, with a steady "background noise" bed (a quieter
    /// continuous tone at a different frequency, standing in for hum/
    /// traffic/a TV) present throughout either way. During an OFF segment
    /// the only thing `estimate_snr_db` can see is the noise floor - this
    /// is exactly the scenario it's designed to catch (see its docs on why
    /// an EARLIER version of this test fixture, gapless on both sides,
    /// didn't actually exercise this - `estimate_snr_db` correctly found
    /// no evidence of noise in a gapless signal, so a realistic on/off
    /// pattern is required to test this at all).
    fn speech_with_pauses_plus_steady_background(sample_rate: u32, secs: f32, speech_amp: f32, noise_amp: f32) -> Vec<f32> {
        let n = (sample_rate as f32 * secs) as usize;
        let segment_len = (sample_rate as f32 * 0.2) as usize; // 200ms on/off segments
        (0..n)
            .map(|i| {
                let t = i as f32 / sample_rate as f32;
                let noise = noise_amp * (2.0 * std::f32::consts::PI * 60.0 * t).sin();
                let speech_on = (i / segment_len) % 2 == 0;
                let speech = if speech_on { speech_amp * (2.0 * std::f32::consts::PI * 220.0 * t).sin() } else { 0.0 };
                speech + noise
            })
            .collect()
    }

    #[test]
    fn loud_steady_background_noise_is_rejected_as_too_noisy() {
        // Background nearly as loud as the "speech" segments themselves -
        // not caught by NoSpeech/TooQuiet (there IS speech-like energy
        // present, and the overall RMS isn't too low), only by the gap
        // between speech-on and speech-off segments collapsing.
        // speech_amp=0.034, noise_amp=0.025: RMS during a speech-off
        // segment (~0.0177) still clears `min_rms`/stays under the
        // `speech_energy_threshold` (0.02) classification boundary, RMS
        // during a speech-on segment (~0.0298) clears it - real, hand-
        // computed gap ~4.5dB, comfortably under the 6dB default
        // threshold without being razor-thin against frame-to-frame RMS
        // variance.
        let audio = speech_with_pauses_plus_steady_background(48_000, 3.0, 0.034, 0.025);
        let err = validate(&audio, 48_000, &ValidationThresholds::default()).unwrap_err();
        assert_eq!(err, EnrollmentError::TooNoisy);
    }

    #[test]
    fn quiet_background_with_real_pauses_is_not_rejected_as_too_noisy() {
        // Same on/off speech pattern, but with a much quieter background -
        // a real, clean recording with natural pauses must NOT be
        // penalized just for having pauses at all.
        let audio = speech_with_pauses_plus_steady_background(48_000, 3.0, 0.3, 0.001);
        assert!(validate(&audio, 48_000, &ValidationThresholds::default()).is_ok());
    }

    #[test]
    fn estimate_snr_db_is_infinite_for_a_gapless_constant_level_signal() {
        // Every frame is above the energy threshold - there are no
        // "non-speech" frames at all to measure a background level from,
        // so this must NOT be flagged as noisy (no evidence either way).
        // An earlier version of this function compared the quietest-vs-
        // loudest percentile of ALL frames regardless of classification,
        // which wrongly returned ~0dB ("too noisy") for exactly this case
        // - a real false positive caught via `valid_speech_like_signal_passes`
        // (a continuous, gapless test tone) unexpectedly failing.
        let audio = vec![1.0f32; 48_000 * 3];
        let snr = estimate_snr_db(&audio, 960, 0.2);
        assert!(snr.is_infinite(), "expected no-evidence-of-noise (infinity), got {snr}");
    }

    #[test]
    fn estimate_snr_db_is_large_for_silence_then_a_loud_tone() {
        let mut audio = vec![0.0f32; 48_000]; // 1s silence
        audio.extend(tone(48_000, 1.0, 0.5)); // 1s loud tone
        let snr = estimate_snr_db(&audio, 960, 0.2);
        assert!(snr > 40.0, "expected a large SNR estimate, got {snr}");
    }

    #[test]
    fn pure_silence_is_rejected_as_too_quiet_or_no_speech() {
        let audio = silence(48_000, 3.0);
        let err = validate(&audio, 48_000, &ValidationThresholds::default()).unwrap_err();
        assert!(matches!(err, EnrollmentError::TooQuiet | EnrollmentError::NoSpeech));
    }

    #[test]
    fn very_short_audio_is_rejected_as_too_short() {
        let audio = tone(48_000, 0.5, 0.3);
        let err = validate(&audio, 48_000, &ValidationThresholds::default()).unwrap_err();
        assert_eq!(err, EnrollmentError::TooShort);
    }

    #[test]
    fn very_quiet_audio_is_rejected_as_too_quiet() {
        let audio = tone(48_000, 3.0, 0.001);
        let err = validate(&audio, 48_000, &ValidationThresholds::default()).unwrap_err();
        assert_eq!(err, EnrollmentError::TooQuiet);
    }

    #[test]
    fn heavily_clipped_audio_is_rejected_as_clipped() {
        let mut audio = tone(48_000, 3.0, 0.3);
        // Force most samples to the clipping ceiling.
        for s in audio.iter_mut() {
            *s = 1.0;
        }
        let err = validate(&audio, 48_000, &ValidationThresholds::default()).unwrap_err();
        assert_eq!(err, EnrollmentError::Clipped);
    }

    #[test]
    fn rms_of_empty_is_zero() {
        assert_eq!(rms(&[]), 0.0);
    }

    #[test]
    fn clipping_ratio_counts_correctly() {
        let audio = vec![0.0f32, 1.0, -1.0, 0.5];
        assert_eq!(clipping_ratio(&audio, 0.99), 0.5);
    }

    #[test]
    fn speech_frame_ratio_handles_a_trailing_partial_frame() {
        // 3 full frames of loud signal, then a partial quiet tail.
        let mut audio = tone(48_000, 0.06, 0.5); // 3 * 20ms frames at 48kHz
        audio.extend(std::iter::repeat(0.0f32).take(200)); // partial trailing frame
        let frame_len = 960; // 20ms @ 48kHz
        let ratio = speech_frame_ratio(&audio, frame_len, 0.02);
        assert!(ratio > 0.5 && ratio < 1.0, "ratio was {ratio}");
    }

    #[test]
    fn zero_sample_rate_is_a_recording_error_not_a_panic() {
        let audio = tone(48_000, 3.0, 0.3);
        let err = validate(&audio, 0, &ValidationThresholds::default()).unwrap_err();
        assert!(matches!(err, EnrollmentError::RecordingError(_)));
    }
}
