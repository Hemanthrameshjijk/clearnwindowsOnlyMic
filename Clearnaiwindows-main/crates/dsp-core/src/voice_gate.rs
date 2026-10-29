//! Volume-based background-voice suppression ("Voice Gate"): attenuates
//! frames that are quiet RELATIVE to the user's own typical speaking level -
//! a cheap, genuinely real-time alternative to full speaker-targeted
//! extraction for the common case where unwanted voices are simply farther
//! from the mic than the primary speaker (a close-mic'd call-center agent
//! vs. other agents' chatter in the background, for example). This cannot
//! distinguish a background voice that is JUST AS LOUD as the primary
//! speaker - that would need true speaker-targeted extraction, which
//! this project no longer includes.
//!
//! MUST run BEFORE `AutoGainStage` in the pipeline, not after: `AutoGainStage`
//! boosts any quiet frame - including quiet background chatter when the
//! primary speaker is silent - up toward its target peak, which would erase
//! the very loudness contrast this stage relies on to tell foreground from
//! background. Gating first, on the true un-boosted signal, then letting
//! AGC normalize whatever is left afterward, preserves that contrast.
//!
//! Tuning note: the constants below are reasoned starting points (same
//! attack/release/envelope-follower shape as `AutoGainStage`, whose values
//! *were* tuned against a real confirmed finding), not yet validated
//! against real background-chatter recordings - expect to adjust
//! `RATIO_THRESHOLD`/`MIN_GAIN`/the attack-release pair once tested against
//! the real target environment (a BPO/call-center floor).

use crate::Stage;

/// How loud (as a fraction of the tracked "my normal speaking level"
/// reference) a frame must be to count as foreground speech and pass
/// through ungated. `0.35` means "at least 35% as loud as my typical
/// speaking level" - low enough to not clip the natural fade-out at the end
/// of a sentence, high enough to treat a meaningfully quieter/farther voice
/// as background.
const RATIO_THRESHOLD: f32 = 0.35;

/// Gain floor applied to fully-gated (background) audio. Not `0.0`
/// (full mute) deliberately: a hard gate to true digital silence is
/// perceptually harsher and more likely to sound like a dropout than a
/// smooth, still-audible-but-clearly-suppressed background - matches the
/// same "attenuate, don't obliterate" judgment call already made for BVC's
/// `atten_lim_db` (30dB, not unlimited) in `bvc-hush`. `0.12` is
/// approximately -18dB.
const MIN_GAIN: f32 = 0.12;

/// Reference-envelope tracking: how quickly the "my normal speaking level"
/// estimate rises when a louder frame appears, vs. how slowly it decays
/// during pauses/silence. Mirrors `AutoGainStage`'s attack/release shape
/// (fast attack so the reference reflects genuinely loud speech quickly,
/// slow release so it is NOT forgotten the moment the speaker pauses for
/// breath - forgetting it would make the very next quiet inter-word gap
/// look like "background", incorrectly gating the speaker's own voice).
const REF_ATTACK: f32 = 0.3;
const REF_RELEASE: f32 = 0.0015;

/// The gate's own open/close smoothing, independent of the reference
/// envelope above. Fast attack (open quickly once foreground speech is
/// detected, so word onsets are not clipped); moderate release (do not
/// snap shut/open on every syllable boundary, which would sound like
/// distracting "pumping").
const GATE_ATTACK: f32 = 0.5;
const GATE_RELEASE: f32 = 0.08;

/// Floor the reference envelope is clamped to, avoiding a divide-by-near-
/// zero blowup during startup/silence before any real speech has set a
/// reference level yet.
const MIN_REFERENCE: f32 = 0.01;

pub struct VoiceGateStage {
    reference_envelope: f32,
    gate_gain: f32,
}

impl VoiceGateStage {
    pub fn new() -> Self {
        Self { reference_envelope: MIN_REFERENCE, gate_gain: 1.0 }
    }
}

impl Default for VoiceGateStage {
    fn default() -> Self {
        Self::new()
    }
}

fn rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    (frame.iter().map(|&s| s * s).sum::<f32>() / frame.len() as f32).sqrt()
}

impl Stage for VoiceGateStage {
    fn name(&self) -> &'static str {
        "voice_gate"
    }

    fn process(&mut self, frame: &mut [f32]) {
        let level = rms(frame);

        // Track "my normal speaking level" - only ever informed by frames
        // that are already at or above the current reference (a quiet/
        // gated frame must never pull the reference down, or the gate would
        // decay its own threshold toward silence and stop gating anything).
        if level > self.reference_envelope {
            self.reference_envelope += (level - self.reference_envelope) * REF_ATTACK;
        } else {
            self.reference_envelope += (level - self.reference_envelope) * REF_RELEASE;
        }
        let reference = self.reference_envelope.max(MIN_REFERENCE);

        let target_gain = if level >= reference * RATIO_THRESHOLD { 1.0 } else { MIN_GAIN };
        let coeff = if target_gain > self.gate_gain { GATE_ATTACK } else { GATE_RELEASE };
        self.gate_gain += (target_gain - self.gate_gain) * coeff;

        for s in frame.iter_mut() {
            *s *= self.gate_gain;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::FRAME_SAMPLES;

    fn tone(peak: f32, seed_offset: usize) -> Vec<f32> {
        (0..FRAME_SAMPLES)
            .map(|i| {
                let t = (i + seed_offset * FRAME_SAMPLES) as f32 / crate::SAMPLE_RATE_HZ as f32;
                (2.0 * std::f32::consts::PI * 220.0 * t).sin() * peak
            })
            .collect()
    }

    /// Feeds a loud "foreground" tone long enough for the reference
    /// envelope to settle at that level, then returns the gate's steady-
    /// state output peak for a quieter probe tone at `probe_peak`.
    fn settled_probe_peak(foreground_peak: f32, probe_peak: f32) -> f32 {
        let mut stage = VoiceGateStage::new();
        for i in 0..200 {
            let mut frame = tone(foreground_peak, i);
            stage.process(&mut frame);
        }
        let mut last_peak = 0.0f32;
        for i in 0..100 {
            let mut frame = tone(probe_peak, 200 + i);
            stage.process(&mut frame);
            last_peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        }
        last_peak
    }

    #[test]
    fn loud_foreground_voice_passes_through_unattenuated() {
        let peak = settled_probe_peak(0.6, 0.6);
        assert!(peak > 0.55, "a loud, steady foreground voice must not be gated, got peak={peak}");
    }

    #[test]
    fn much_quieter_background_voice_is_attenuated() {
        // Background at ~15% of the foreground's level - well below
        // RATIO_THRESHOLD (0.35), must be gated down.
        let peak = settled_probe_peak(0.6, 0.6 * 0.15);
        let foreground_peak = 0.6 * 0.15;
        assert!(
            peak < foreground_peak * 0.8,
            "a background voice well below the foreground reference must be attenuated, got peak={peak} (ungated would be ~{foreground_peak})"
        );
    }

    #[test]
    fn gate_never_fully_mutes_background_audio() {
        let mut stage = VoiceGateStage::new();
        // Settle a foreground reference, then feed sustained quiet
        // background long enough for the gate to fully close.
        for i in 0..200 {
            let mut frame = tone(0.6, i);
            stage.process(&mut frame);
        }
        let mut last_peak = 1.0f32;
        for i in 0..200 {
            let mut frame = tone(0.6 * 0.1, 200 + i);
            stage.process(&mut frame);
            last_peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        }
        assert!(last_peak > 0.0, "gate must never fully mute to digital silence, got peak={last_peak}");
    }

    #[test]
    fn true_silence_stays_silent() {
        let mut stage = VoiceGateStage::new();
        for _ in 0..50 {
            let mut frame = vec![0.0f32; FRAME_SAMPLES];
            stage.process(&mut frame);
            assert!(frame.iter().all(|&s| s == 0.0), "true silence (all zeros) must stay exactly zero");
        }
    }

    #[test]
    fn quiet_speaker_with_no_loud_reference_yet_is_not_gated() {
        // Nothing loud has ever been seen, so there is no established
        // "normal speaking level" to gate relative to - a consistently
        // quiet speaker (soft-spoken, or just started talking) must not be
        // mistaken for background and suppressed.
        let mut stage = VoiceGateStage::new();
        let mut last_peak = 0.0f32;
        for i in 0..200 {
            let mut frame = tone(0.1, i);
            stage.process(&mut frame);
            last_peak = frame.iter().fold(0.0f32, |m, &s| m.max(s.abs()));
        }
        assert!(last_peak > 0.08, "a consistently quiet speaker (no louder reference established) must not be gated, got peak={last_peak}");
    }
}
