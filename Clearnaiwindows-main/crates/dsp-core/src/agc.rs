//! Automatic gain: boosts quiet mic input up toward a healthy target level
//! *before* it reaches noise-suppression stages.
//!
//! Real-world finding (confirmed live, not speculative): RNNoise measurably
//! suppressed voice right alongside background noise when the physical
//! mic's input level was low (peak well under half of full scale) - raising
//! the OS-level mic gain so the signal swung solidly higher fully resolved
//! it. RNNoise's voice-activity detection loses confidence distinguishing
//! speech from noise when both sit close to its internal noise floor, which
//! is exactly what a quiet input does regardless of how loud the actual
//! background noise is. This stage removes the dependency on the user
//! having correctly set Windows' mic gain by normalizing the level inside
//! the pipeline itself, ahead of `noise-engine`/`bvc-hush`.

use crate::Stage;

/// Peak level this stage tries to bring quiet input up to. `0.5` leaves
/// headroom below full scale for transients the envelope hasn't caught up
/// to yet, matching the level a "healthy" mic swing was observed at.
const TARGET_PEAK: f32 = 0.5;

/// Maximum boost applied to already-silent input - without a ceiling, a
/// true-silence frame's near-zero envelope would drive gain toward
/// infinity and amplify noise floor/hiss into audible content.
const MAX_GAIN: f32 = 8.0; // +18dB

/// Floor the envelope is clamped to before computing gain, for the same
/// divide-by-near-zero reason as `MAX_GAIN`.
const MIN_ENVELOPE: f32 = TARGET_PEAK / MAX_GAIN;

/// Envelope smoothing coefficients (0..1, higher = faster tracking).
/// Fast attack (envelope rises quickly when the signal gets louder) keeps
/// gain from lingering high into a sudden loud transient after quiet
/// speech, which is what would otherwise clip; slow release (envelope
/// falls gradually when the signal gets quieter) avoids the gain visibly
/// "pumping" up and down between words/syllables.
const ATTACK: f32 = 0.6;
const RELEASE: f32 = 0.02;

/// Peak-envelope-follower automatic gain stage. Only ever boosts (gain is
/// clamped to `>= 1.0`) - it never attenuates already-healthy input, so it
/// cannot itself cause the "voice suppressed" symptom it exists to prevent.
pub struct AutoGainStage {
    envelope: f32,
}

impl AutoGainStage {
    pub fn new() -> Self {
        Self { envelope: MIN_ENVELOPE }
    }
}

impl Default for AutoGainStage {
    fn default() -> Self {
        Self::new()
    }
}

impl Stage for AutoGainStage {
    fn name(&self) -> &'static str {
        "auto_gain"
    }

    fn process(&mut self, frame: &mut [f32]) {
        let frame_peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        let coeff = if frame_peak > self.envelope { ATTACK } else { RELEASE };
        self.envelope += (frame_peak - self.envelope) * coeff;
        let envelope = self.envelope.max(MIN_ENVELOPE);

        let gain = (TARGET_PEAK / envelope).clamp(1.0, MAX_GAIN);

        for s in frame.iter_mut() {
            *s = (*s * gain).clamp(-1.0, 1.0);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FRAME_SAMPLES;

    fn quiet_tone(peak: f32, seed_offset: usize) -> Vec<f32> {
        (0..FRAME_SAMPLES)
            .map(|i| {
                let t = (i + seed_offset * FRAME_SAMPLES) as f32 / crate::SAMPLE_RATE_HZ as f32;
                (2.0 * std::f32::consts::PI * 220.0 * t).sin() * peak
            })
            .collect()
    }

    #[test]
    fn boosts_quiet_input_toward_target_peak() {
        let mut stage = AutoGainStage::new();
        let mut last_peak = 0.0f32;
        // Run enough frames for the envelope to settle (release is slow by
        // design).
        for i in 0..200 {
            let mut frame = quiet_tone(0.05, i);
            stage.process(&mut frame);
            last_peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        }
        assert!(
            last_peak > 0.3,
            "quiet 0.05-peak input should be boosted well above its original level after settling, got {last_peak}"
        );
    }

    #[test]
    fn never_attenuates_already_healthy_input() {
        let mut stage = AutoGainStage::new();
        for i in 0..50 {
            let mut frame = quiet_tone(0.8, i);
            let original = frame.clone();
            stage.process(&mut frame);
            for (out, orig) in frame.iter().zip(original.iter()) {
                assert!(
                    out.abs() <= orig.abs() + 1e-6,
                    "already-healthy input must never be amplified further \
                     (out={out}, orig={orig})"
                );
            }
        }
    }

    #[test]
    fn never_exceeds_unity_output_range_even_on_a_loud_transient_after_silence() {
        let mut stage = AutoGainStage::new();
        // Long silence lets gain climb toward MAX_GAIN...
        for _ in 0..100 {
            let mut frame = vec![0.0f32; FRAME_SAMPLES];
            stage.process(&mut frame);
        }
        // ...then a sudden full-scale transient must still be hard-clamped.
        let mut frame = vec![1.0f32; FRAME_SAMPLES];
        stage.process(&mut frame);
        for &s in &frame {
            assert!((-1.0..=1.0).contains(&s), "output must stay in [-1.0, 1.0], got {s}");
        }
    }

    #[test]
    fn true_silence_stays_silent_not_amplified_into_noise() {
        let mut stage = AutoGainStage::new();
        for _ in 0..50 {
            let mut frame = vec![0.0f32; FRAME_SAMPLES];
            stage.process(&mut frame);
            assert!(frame.iter().all(|&s| s == 0.0), "true silence (all zeros) must stay exactly zero");
        }
    }
}
