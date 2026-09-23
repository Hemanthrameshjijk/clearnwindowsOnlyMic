//! Real `tse_engine::TseBackend` implementation, wiring the encoder/
//! separator/decoder (`engine::VantaEngine`) into the production 48kHz/
//! 480-sample frame contract - direct structural mirror of
//! `tse_bsrnn::backend::RealBsrnnTseBackend`.
//!
//! Note this backend does NOT itself load the ECAPA speaker encoder - by
//! the time `set_target_speaker` is called, the 192-dim embedding has
//! already been computed once at enrollment time (via
//! `ecapa::VantaSpeakerEncoder::embed`, off the realtime thread) and is
//! just carried here inside the `TargetSpeakerProfile`. Same separation
//! `RealBsrnnTseBackend`/`TseSpeakerEncoder` already establish.
//!
//! # Known, honestly-reported gaps (not fixed in this pass)
//!
//! - **Allocates on the realtime path.** `process_frame` calls into
//!   `VantaEngine::process_chunk` and the causal resamplers, all of which
//!   allocate fresh `Vec`s per call - same documented, not-yet-fixed
//!   violation of `dsp_core::Stage`'s "must not allocate" rule that
//!   `RealBsrnnTseBackend` already carries.
//! - **Speed not yet measured for the REAL weights on the REAL pipeline.**
//!   The random-weight Rust port measured RTF ~0.8x (see
//!   `research/tse/candidates/vanta-causal-rs/src/main.rs`), which only
//!   proves the compute graph's *shape* is fast enough - real weight
//!   VALUES don't change FLOP count, so this should carry over, but it has
//!   not been re-measured inside the actual pipeline/`WatchdogStage` yet.
//! - **Cannot currently construct successfully at all** - see this crate's
//!   `lib.rs` docs: encoder/decoder weights are not yet exported.

use crate::engine::{VantaEngine, VantaEngineState};
use crate::weights::{load_encoder_decoder_weights, load_separator_weights, WeightLoadError};
use speaker_profile::TargetSpeakerProfile;
use std::collections::VecDeque;
use std::sync::Arc;
use tse_bsrnn::causal_resampler::{CausalDownsampler3x, CausalUpsampler3x};
use tse_engine::{TseBackend, TseError};

pub struct VantaBackend {
    engine: VantaEngine,
    state: Option<VantaEngineState>,
    target_embedding: Option<[f32; crate::separator::SPEAKER_DIM]>,
    downsampler: CausalDownsampler3x,
    upsampler: CausalUpsampler3x,
    output_queue: VecDeque<f32>,
}

impl VantaBackend {
    /// Fails loudly if either weight set isn't real (see this crate's
    /// `lib.rs` docs on the encoder/decoder gap) - never silently
    /// constructs against random/placeholder weights.
    pub fn try_load() -> Result<Self, WeightLoadError> {
        let separator = load_separator_weights()?;
        let (encoder, decoder) = load_encoder_decoder_weights()?;
        let engine = VantaEngine::new(encoder, decoder, separator);
        Ok(Self {
            engine,
            state: None,
            target_embedding: None,
            downsampler: CausalDownsampler3x::new(63),
            upsampler: CausalUpsampler3x::new(63),
            output_queue: VecDeque::new(),
        })
    }
}

impl TseBackend for VantaBackend {
    fn name(&self) -> &'static str {
        "Vanta (Conv-TasNet, experimental)"
    }

    fn set_target_speaker(&mut self, profile: Option<Arc<TargetSpeakerProfile>>) {
        match profile {
            Some(p) if p.dimension() == crate::separator::SPEAKER_DIM => {
                let mut embedding = [0.0f32; crate::separator::SPEAKER_DIM];
                embedding.copy_from_slice(p.embedding());
                self.target_embedding = Some(embedding);
                self.state = Some(self.engine.new_state());
                self.downsampler.reset();
                self.upsampler.reset();
                self.output_queue.clear();
            }
            _ => {
                // None, or a profile from the wrong encoder (e.g. the
                // 256-dim WeSpeaker one or the OTHER 192-dim ECAPA one
                // from tse-bsrnn - same dimension, different embedding
                // space, still wrong) - no usable target speaker.
                self.target_embedding = None;
                self.state = None;
                self.output_queue.clear();
            }
        }
    }

    fn process_frame(&mut self, frame: &mut [f32]) -> Result<(), TseError> {
        let state = self.state.as_mut().ok_or(TseError::NoTargetSpeaker)?;
        let embedding = self.target_embedding.as_ref().ok_or(TseError::NoTargetSpeaker)?;

        let down_16k = self.downsampler.push(frame);
        if !down_16k.is_empty() {
            let out_16k = self.engine.process_chunk(&down_16k, embedding, state);
            if !out_16k.is_empty() {
                let up_48k = self.upsampler.push(&out_16k);
                self.output_queue.extend(up_48k);
            }
        }

        if self.output_queue.len() < frame.len() {
            return Err(TseError::InferenceFailed("not enough buffered output yet (warm-up)".to_string()));
        }

        for sample in frame.iter_mut() {
            *sample = self.output_queue.pop_front().expect("length checked above");
        }
        Ok(())
    }

    fn reset_stream_state(&mut self) {
        if let Some(state) = self.state.as_mut() {
            self.engine.reset(state);
        }
        self.downsampler.reset();
        self.upsampler.reset();
        self.output_queue.clear();
    }

    fn latency_samples(&self) -> Option<usize> {
        // Same reasoning as RealBsrnnTseBackend: real algorithmic latency
        // exists but varies with output_queue's fill level - left honestly
        // unmeasured rather than guessed.
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speaker_profile::{NormalizationKind, TargetSpeakerProfile};

    #[test]
    fn try_load_succeeds_with_all_real_weights_bundled() {
        // The real end-to-end construction check: separator + encoder +
        // decoder weights must ALL be real now (see lib.rs's docs on the
        // two-part export) - this must succeed, not error.
        VantaBackend::try_load().expect("VantaBackend must construct with real bundled weights");
    }

    #[test]
    fn processes_real_48k_frames_without_panicking_or_nan() {
        let mut backend = VantaBackend::try_load().expect("construct");
        let embedding = vec![0.1f32; crate::separator::SPEAKER_DIM];
        let profile = TargetSpeakerProfile::new(embedding, "test", 16_000, NormalizationKind::None).unwrap();
        backend.set_target_speaker(Some(Arc::new(profile)));

        // Real pipeline frame size: 480 samples @48kHz (10ms).
        let mut saw_any_success = false;
        for i in 0..40 {
            let mut frame = [0.0f32; 480];
            for (idx, s) in frame.iter_mut().enumerate() {
                *s = ((i * 480 + idx) as f32 * 0.01).sin() * 0.3;
            }
            match backend.process_frame(&mut frame) {
                Ok(()) => {
                    saw_any_success = true;
                    assert!(frame.iter().all(|v| v.is_finite()), "output must be finite");
                }
                Err(TseError::InferenceFailed(msg)) => {
                    assert!(msg.contains("warm-up"), "only the warm-up gap is an expected error: {msg}");
                }
                Err(e) => panic!("unexpected error: {e:?}"),
            }
        }
        assert!(saw_any_success, "must produce real output within 40 real frames (400ms) of warm-up");
    }
}
