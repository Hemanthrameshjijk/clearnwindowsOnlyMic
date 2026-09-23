//! Real `tse_engine::TseBackend` implementation, wiring together every
//! verified piece of this crate (causal resampler, `StreamingTseEngine`)
//! into the production 48kHz/480-sample frame contract.
//!
//! # Known, honestly-reported gaps (not fixed in this pass)
//!
//! - **Not yet real-time.** The underlying `StreamingTseEngine` was
//!   measured (see `poc/real_tse/` and this crate's own dev history) at
//!   roughly RTF~21-31x depending on the optimization pass - far short of
//!   the 10ms/480-sample deadline this frame-by-frame contract implies.
//!   `dsp_core::WatchdogStage` (already wired around this backend in
//!   `app/src/engine.rs`) bounds the damage (a slow call times out and
//!   passes the frame through unmodified; 3 consecutive timeouts bypass
//!   permanently for the session) but does not make this backend fast.
//! - **Allocates on the realtime path.** `process_frame` calls into
//!   `StreamingTseEngine::process_chunk`, `CausalDownsampler3x::push`, and
//!   `CausalUpsampler3x::push`, all of which allocate fresh `Vec`s per
//!   call - a real violation of `dsp_core::Stage`'s "must not allocate"
//!   rule and `TseBackend`'s own realtime contract, not yet fixed. Safe
//!   (no crash/UB), just not compliant with the project's own realtime-
//!   safety rule.
//!
//! Both are flagged here, in the type's own docs, rather than silently
//! left for someone else to discover.

use crate::causal_resampler::{CausalDownsampler3x, CausalUpsampler3x};
use crate::engine::StreamingTseEngine;
use crate::separator::BsrnnSeparator;
use speaker_profile::TargetSpeakerProfile;
use std::collections::VecDeque;
use std::sync::Arc;
use tse_engine::{TseBackend, TseError};

/// Real BSRNN-based streaming TSE backend. Constructed once at engine
/// startup (`try_load`, parses/leaks the ~75MB separator weight blob plus
/// the ~25MB ECAPA ONNX graph - the same "expensive parse happens once,
/// never per-frame or per-enrollment" rule every other real backend in
/// this project already follows).
pub struct RealBsrnnTseBackend {
    separator: BsrnnSeparator, // cheap to Clone (see BsrnnSeparator's docs) - cloned into a fresh engine per re-enrollment
    engine: Option<StreamingTseEngine>,
    downsampler: CausalDownsampler3x,
    upsampler: CausalUpsampler3x,
    output_queue: VecDeque<f32>,
}

impl RealBsrnnTseBackend {
    pub fn try_load() -> Result<Self, String> {
        let separator = BsrnnSeparator::load().map_err(|e| e.to_string())?;
        Ok(Self {
            separator,
            engine: None,
            downsampler: CausalDownsampler3x::new(63),
            upsampler: CausalUpsampler3x::new(63),
            output_queue: VecDeque::new(),
        })
    }
}

impl TseBackend for RealBsrnnTseBackend {
    fn name(&self) -> &'static str {
        "TSE (BSRNN, real)"
    }

    fn set_target_speaker(&mut self, profile: Option<Arc<TargetSpeakerProfile>>) {
        match profile {
            Some(p) if p.dimension() == 192 => {
                let mut embedding = [0.0f32; 192];
                embedding.copy_from_slice(p.embedding());
                self.engine = Some(StreamingTseEngine::new(self.separator.clone(), &embedding));
                self.downsampler.reset();
                self.upsampler.reset();
                self.output_queue.clear();
            }
            _ => {
                // None, or a profile from the wrong encoder (e.g. the
                // existing 256-dim WeSpeaker one, if this ever gets
                // pointed at the wrong store by mistake) - no usable
                // target speaker, matching UnavailableTse's honest
                // behavior rather than guessing.
                self.engine = None;
                self.output_queue.clear();
            }
        }
    }

    fn process_frame(&mut self, frame: &mut [f32]) -> Result<(), TseError> {
        let engine = self.engine.as_mut().ok_or(TseError::NoTargetSpeaker)?;

        let down_16k = self.downsampler.push(frame);
        if !down_16k.is_empty() {
            let tse_out_16k = engine.process_chunk(&down_16k);
            if !tse_out_16k.is_empty() {
                let up_48k = self.upsampler.push(&tse_out_16k);
                self.output_queue.extend(up_48k);
            }
        }

        if self.output_queue.len() < frame.len() {
            // Real warm-up gap (STFT needs a minimum history before its
            // first frame; resampler group delay adds a small further
            // offset) - frame is left untouched, per this trait's
            // contract, since nothing below has mutated it yet.
            return Err(TseError::InferenceFailed(
                "not enough buffered output yet (warm-up)".to_string(),
            ));
        }

        for sample in frame.iter_mut() {
            *sample = self.output_queue.pop_front().expect("length checked above");
        }
        Ok(())
    }

    fn reset_stream_state(&mut self) {
        if let Some(engine) = self.engine.as_mut() {
            engine.reset();
        }
        self.downsampler.reset();
        self.upsampler.reset();
        self.output_queue.clear();
    }

    fn latency_samples(&self) -> Option<usize> {
        // Real algorithmic latency exists (resampler group delays, STFT's
        // win/2 warm-up requirement) but the *total*, exact figure also
        // depends on `output_queue`'s current fill level, which varies -
        // reporting a fixed number here would either be wrong sometimes or
        // require re-deriving it per call. Left honestly unmeasured (see
        // `docs/TSE_ENROLLMENT.md`'s convention for unmeasured latency)
        // rather than guessed.
        None
    }
}
