//! `TseBackend`: the trait a real streaming Target Speaker Extraction model
//! would implement, plus the one concrete implementation this project
//! currently ships (`UnavailableTse`, an honest "no model integrated yet"
//! stand-in - see this crate's top-level docs and `docs/TSE_ENROLLMENT.md`).

use speaker_profile::TargetSpeakerProfile;
use std::fmt;
use std::sync::Arc;

/// Everything that can go wrong processing one frame through a TSE backend.
/// Distinct from `speaker_profile::SpeakerEncoderError` (that one covers
/// turning reference audio into an embedding during enrollment; this one
/// covers the separate, later concern of running the live streaming model).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TseError {
    /// No real TSE backend/model is available (no bundled model, or the
    /// runtime/model failed to load). This is the only variant
    /// `UnavailableTse` ever returns.
    BackendUnavailable,
    /// `process_frame` was called with no target speaker ever set (or after
    /// `set_target_speaker(None)`). Distinguishable from
    /// `BackendUnavailable` so a caller/log can tell "no model" apart from
    /// "model present, but nobody enrolled yet" - both currently degrade to
    /// the same safe behavior (pass the frame through unmodified), but they
    /// are different situations worth telling apart in metrics/logs.
    NoTargetSpeaker,
    /// The backend ran but failed (a real runtime/inference error).
    InferenceFailed(String),
}

impl fmt::Display for TseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TseError::BackendUnavailable => write!(f, "TSE backend unavailable (no model/runtime bundled)"),
            TseError::NoTargetSpeaker => write!(f, "no target speaker profile set"),
            TseError::InferenceFailed(reason) => write!(f, "TSE inference failed: {reason}"),
        }
    }
}

impl std::error::Error for TseError {}

/// A streaming Target Speaker Extraction backend: given a live 10ms audio
/// frame and a previously-set target-speaker embedding, extracts (isolates)
/// that speaker's voice from the rest of the signal, maintaining whatever
/// persistent model state a real causal/streaming model needs across calls.
///
/// # Realtime contract
///
/// `process_frame` is called from the realtime mic pipeline thread (via
/// `TseStageAdapter`, wrapped in a `dsp_core::WatchdogStage` by whoever
/// constructs the pipeline - see `crate::adapter`). An implementation must
/// therefore, on every call:
/// - not allocate a new inference session, not reload a model, and not
///   extract a speaker embedding (all of that belongs in enrollment/backend
///   construction, which happen once, off this thread);
/// - not perform filesystem, network, or GUI operations;
/// - not take a lock that anything outside this call could ever hold across
///   a blocking operation;
/// - on failure, leave `frame` exactly as it was on entry (the caller/
///   watchdog is responsible for the "pass through unprocessed on failure"
///   policy - see `crate::adapter::TseStageAdapter` - but that policy only
///   works if a failing implementation itself never partially mutates the
///   buffer before erroring).
///
/// `set_target_speaker`/`reset_stream_state` are *not* necessarily called
/// from the realtime thread in every deployment, but in this project's
/// current wiring (`crate::adapter::TseStageAdapter`) they are, specifically
/// because they must run in strict frame-order relative to
/// `process_frame` (a re-enrollment mid-call must be picked up before the
/// very next frame, not racily). Both must therefore be realtime-safe too:
/// swapping a pointer/index, not doing any real work themselves.
pub trait TseBackend: Send {
    /// Human-readable name, used only for logging/GUI labels - mirrors
    /// `dsp_core::Stage::name`.
    fn name(&self) -> &'static str;

    /// Sets (or clears, via `None`) the active target speaker. Must be
    /// realtime-safe (see trait docs) - a real implementation stores the
    /// embedding for later conditioning steps, never runs inference here.
    fn set_target_speaker(&mut self, profile: Option<Arc<TargetSpeakerProfile>>);

    /// Processes one frame in place. Returns `Err` (leaving `frame`
    /// untouched) if there is no usable backend/target speaker or inference
    /// itself failed - see `TseError`. Must not allocate/lock/block for real
    /// - see trait docs.
    fn process_frame(&mut self, frame: &mut [f32]) -> Result<(), TseError>;

    /// Resets any persistent streaming state (recurrent/causal buffers),
    /// e.g. when the target speaker changes or a new call starts. Mirrors
    /// `bvc_hush::HushBvcStage::reset_stream_state`.
    fn reset_stream_state(&mut self);

    /// Algorithmic delay this backend introduces, in samples, if known.
    /// `None` if not (yet) measured/applicable - see
    /// `docs/TSE_ENROLLMENT.md`'s honesty notes on unmeasured latency.
    fn latency_samples(&self) -> Option<usize>;
}

/// The only `TseBackend` this project currently ships: a real, explicit
/// "no TSE model is integrated" implementation.
///
/// # Why this exists instead of a real model
///
/// The task that introduced this crate named a specific candidate model,
/// `penta2himajin/tse-conv-tasnet-48k`, for a causal/streaming/speaker-
/// conditioned TSE model at 48kHz. Before writing any inference code, that
/// repository was checked directly against the Hugging Face Hub API: a
/// direct fetch of `penta2himajin/tse-conv-tasnet-48k` returns an
/// authentication/not-found response, and a Hugging Face model search for
/// "tse-conv-tasnet" / "conv-tasnet" returns no matching repository by that
/// uploader or name. In other words, **this specific candidate model does
/// not exist publicly** at the time this was written, so there is nothing
/// real to download, inspect (ONNX input/output names, tensor shapes,
/// embedding dimension, state tensors), or wire up.
///
/// Rather than fabricate a plausible-looking integration against a model
/// that isn't real, or fake target-speaker extraction with plain noise
/// suppression (explicitly disallowed - see `docs/TSE_ENROLLMENT.md`), this
/// backend honestly reports `TseError::BackendUnavailable` on every call.
/// `crate::router::BvcRouterStage` treats that exactly like Hush reports
/// "BVC unavailable" today: the TSE option is present in the GUI, but
/// disabled/labelled unavailable, and selecting it (if forced) passes audio
/// through unmodified rather than corrupting it.
#[derive(Debug, Default)]
pub struct UnavailableTse;

impl TseBackend for UnavailableTse {
    fn name(&self) -> &'static str {
        "TSE (unavailable)"
    }

    fn set_target_speaker(&mut self, _profile: Option<Arc<TargetSpeakerProfile>>) {
        // Nothing to store - there is no model to condition.
    }

    fn process_frame(&mut self, _frame: &mut [f32]) -> Result<(), TseError> {
        Err(TseError::BackendUnavailable)
    }

    fn reset_stream_state(&mut self) {
        // No streaming state exists.
    }

    fn latency_samples(&self) -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unavailable_tse_always_errors_and_never_mutates_the_frame() {
        let mut backend = UnavailableTse;
        let mut frame = [0.1f32, 0.2, 0.3];
        let original = frame;
        let err = backend.process_frame(&mut frame).unwrap_err();
        assert_eq!(err, TseError::BackendUnavailable);
        assert_eq!(frame, original, "a failing backend must never partially mutate the frame");
    }

    #[test]
    fn unavailable_tse_reports_no_latency() {
        let backend = UnavailableTse;
        assert!(backend.latency_samples().is_none());
    }

    #[test]
    fn unavailable_tse_accepts_set_target_and_reset_without_panicking() {
        let mut backend = UnavailableTse;
        backend.set_target_speaker(None);
        backend.reset_stream_state();
    }
}
