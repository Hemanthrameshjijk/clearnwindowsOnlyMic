//! Portable target-speaker-profile abstraction: the data type produced by
//! enrollment (`TargetSpeakerProfile`), the trait a speaker encoder backend
//! implements to produce one (`SpeakerEncoder`), and the lock-free cell the
//! realtime TSE stage reads it from (`TargetSpeakerStore`).
//!
//! Nothing in this crate touches an OS API, an ONNX runtime, or any
//! concrete model - it builds and runs identically on Linux and Windows,
//! matching `dsp-core`'s "portable core, concrete backends live elsewhere"
//! shape. This crate depends on nothing from `dsp-core` (no `Stage`, no
//! `FRAME_SAMPLES`): a speaker profile is meaningful independent of the
//! 10ms audio pipeline that eventually consumes it, and keeping this crate
//! free of that dependency means `enrollment` (which needs both this crate
//! and, indirectly, capture-format constants) is the only place those two
//! concerns actually meet.
//!
//! # Honesty rule
//!
//! This crate only defines the shape a `SpeakerEncoder` backend uses - it
//! does not itself decide whether a real one is loaded. The real backend
//! (`speaker_encoder::WeSpeakerEncoder`, a bundled ONNX WeSpeaker ResNet34
//! speaker-verification model) lives in the separate `speaker-encoder` crate, kept
//! out of this one so this crate stays free of an ONNX runtime dependency.
//! `UnavailableSpeakerEncoder` in this crate is the honest fallback used
//! when that real backend fails to load - it always returns
//! `SpeakerEncoderError::ModelUnavailable` and must never fabricate a
//! profile (a random vector, an MFCC average, or anything else) just to
//! make enrollment appear to succeed. See `docs/TSE_ENROLLMENT.md`.

mod store;

pub use store::TargetSpeakerStore;

use std::fmt;
use std::sync::Arc;
use std::time::SystemTime;

/// How the embedding vector was normalized before being stored, so a TSE
/// backend consuming this profile knows exactly what it's receiving rather
/// than having to guess or re-derive it. `Unknown` exists only for
/// deserialization/compatibility with a future persisted format; nothing
/// in this codebase constructs it today - every real `SpeakerEncoder` must
/// report the normalization it actually applied.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormalizationKind {
    /// The embedding was not normalized at all - raw encoder output.
    None,
    /// The embedding was scaled to unit L2 norm (the common convention for
    /// cosine-similarity-based speaker verification/embedding models, e.g.
    /// ECAPA-TDNN-style encoders).
    L2,
}

impl fmt::Display for NormalizationKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NormalizationKind::None => write!(f, "none"),
            NormalizationKind::L2 => write!(f, "l2"),
        }
    }
}

/// An enrolled target speaker: the embedding vector plus enough metadata
/// that a TSE backend (or a future persistence format) can tell exactly
/// what it's holding without guessing.
///
/// Deliberately immutable once constructed (no `&mut` methods) - the
/// realtime-safe way to "change" the active profile is to build a new one
/// and hand it to `TargetSpeakerStore::set_profile`, which swaps the whole
/// `Arc` atomically. This also means a `TargetSpeakerProfile` can be freely
/// shared (`Arc`-wrapped) between the enrollment thread that built it and
/// any number of readers without synchronization beyond the store itself.
#[derive(Debug, Clone, PartialEq)]
pub struct TargetSpeakerProfile {
    embedding: Arc<[f32]>,
    encoder_id: String,
    sample_rate_hz: u32,
    normalization: NormalizationKind,
    /// Monotonically increasing across profiles created in this process
    /// (see `next_version`). Not a wall-clock timestamp: this project's
    /// realtime rules avoid relying on wall-clock reads for anything a
    /// correctness decision depends on, and "which of two profiles is
    /// newer" is exactly such a decision (used by a future TSE backend to
    /// notice a re-enrollment happened and reset its streaming state - see
    /// `tse_engine::TseBackend::set_target_speaker`).
    version: u64,
    /// Wall-clock creation time, for display/debugging only (e.g. a GUI
    /// "enrolled 2 minutes ago" label) - never used for ordering/equality
    /// logic, matching `version`'s docs above.
    created_at: SystemTime,
}

impl TargetSpeakerProfile {
    /// Builds a new profile. `embedding` must be non-empty; callers (in
    /// practice, only `SpeakerEncoder::embed` implementations) are expected
    /// to have already validated its length matches `encoder_id`'s known
    /// dimension - this constructor does not second-guess that, it only
    /// rejects the one case (empty) that can never be valid for any encoder.
    pub fn new(
        embedding: Vec<f32>,
        encoder_id: impl Into<String>,
        sample_rate_hz: u32,
        normalization: NormalizationKind,
    ) -> Result<Self, ProfileError> {
        if embedding.is_empty() {
            return Err(ProfileError::EmptyEmbedding);
        }
        if embedding.iter().any(|v| !v.is_finite()) {
            return Err(ProfileError::NonFiniteEmbedding);
        }
        Ok(Self {
            embedding: Arc::from(embedding),
            encoder_id: encoder_id.into(),
            sample_rate_hz,
            normalization,
            version: next_version(),
            created_at: SystemTime::now(),
        })
    }

    pub fn embedding(&self) -> &[f32] {
        &self.embedding
    }

    pub fn dimension(&self) -> usize {
        self.embedding.len()
    }

    pub fn encoder_id(&self) -> &str {
        &self.encoder_id
    }

    pub fn sample_rate_hz(&self) -> u32 {
        self.sample_rate_hz
    }

    pub fn normalization(&self) -> NormalizationKind {
        self.normalization
    }

    pub fn version(&self) -> u64 {
        self.version
    }

    pub fn created_at(&self) -> SystemTime {
        self.created_at
    }

    /// Seconds since this profile was created, for a GUI "enrolled Ns ago"
    /// label. Returns `0` if the clock has gone backwards since creation
    /// (must not panic on that - `SystemTime` is not guaranteed monotonic).
    pub fn age_seconds(&self) -> u64 {
        self.created_at
            .elapsed()
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProfileError {
    EmptyEmbedding,
    NonFiniteEmbedding,
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ProfileError::EmptyEmbedding => write!(f, "speaker embedding is empty"),
            ProfileError::NonFiniteEmbedding => write!(f, "speaker embedding contains a NaN/infinite value"),
        }
    }
}

impl std::error::Error for ProfileError {}

fn next_version() -> u64 {
    use std::sync::atomic::{AtomicU64, Ordering};
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_add(1, Ordering::Relaxed)
}

/// Everything that can go wrong turning reference audio into a speaker
/// embedding. Kept separate from `enrollment::EnrollmentError` (which also
/// covers recording/VAD/quality failures that never reach a `SpeakerEncoder`
/// at all) - this enum is only the encoder's own failure surface, so a
/// `SpeakerEncoder` implementation can be tested/used independent of the
/// enrollment state machine that happens to be this project's only current
/// caller.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpeakerEncoderError {
    /// No real encoder backend is available (no bundled model, or the
    /// runtime/model failed to load). This is the *only* variant
    /// `UnavailableSpeakerEncoder` ever returns, and it is a normal,
    /// honestly-reported outcome, not a bug - see this crate's docs.
    ModelUnavailable,
    /// `audio` was rejected before inference even ran (wrong length, wrong
    /// sample rate for what the encoder requires, etc.) - distinct from a
    /// VAD/quality rejection, which happens earlier, before the encoder is
    /// ever called (see `enrollment::validate`).
    InvalidAudio(String),
    /// The encoder ran but failed (a real runtime/inference error), carrying
    /// a human-readable reason.
    InferenceFailed(String),
}

impl fmt::Display for SpeakerEncoderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SpeakerEncoderError::ModelUnavailable => {
                write!(f, "speaker encoder backend unavailable (no model/runtime bundled)")
            }
            SpeakerEncoderError::InvalidAudio(reason) => write!(f, "invalid reference audio: {reason}"),
            SpeakerEncoderError::InferenceFailed(reason) => write!(f, "speaker encoder inference failed: {reason}"),
        }
    }
}

impl std::error::Error for SpeakerEncoderError {}

/// Converts reference speech into a `TargetSpeakerProfile`. Implemented by a
/// concrete backend - `speaker_encoder::WeSpeakerEncoder` (a real,
/// bundled ONNX WeSpeaker ResNet34 speaker-verification model - see that
/// crate's docs for why WeSpeaker rather than literal ECAPA-TDNN) is the real one this project
/// ships; `UnavailableSpeakerEncoder` below is the honest fallback used when
/// that backend fails to load.
///
/// `embed` must not be called from a realtime audio thread: a real
/// implementation runs model inference (plus resampling/feature extraction)
/// over several seconds of audio, which is arbitrarily slower than the 10ms
/// budget every stage in the live mic pipeline is held to.
/// `enrollment::EnrollmentSession` is responsible for only ever calling this
/// from its own dedicated worker thread.
pub trait SpeakerEncoder: Send + Sync {
    /// Stable identifier for the encoder that produced (or would produce) a
    /// profile, stored in `TargetSpeakerProfile::encoder_id` so a future TSE
    /// backend can refuse a profile it doesn't recognize instead of silently
    /// misinterpreting a foreign embedding format.
    fn encoder_id(&self) -> &'static str;

    /// The embedding dimension this encoder produces, if known without
    /// running inference (a real ONNX-backed encoder knows this from its
    /// model's output shape at load time). `None` for a backend that cannot
    /// answer this without a model loaded (in practice, only
    /// `UnavailableSpeakerEncoder`).
    fn embedding_dim(&self) -> Option<usize>;

    /// The sample rate this encoder's model actually operates at internally
    /// (e.g. 16000 for most speaker-verification models). Informational
    /// only - see `embed`'s docs: the caller does **not** need to resample to
    /// this rate first.
    fn expected_sample_rate_hz(&self) -> u32;

    /// Runs the encoder over `audio` (mono `f32`, at `input_sample_rate_hz` -
    /// in practice always `dsp_core::SAMPLE_RATE_HZ`, since
    /// `enrollment::EnrollmentSession` always records/validates at the mic
    /// pipeline's native rate) and returns a `TargetSpeakerProfile`.
    /// Resampling to whatever rate the concrete model actually needs
    /// internally (see `expected_sample_rate_hz`) is this implementation's
    /// own responsibility, not the caller's - this keeps `enrollment` fully
    /// decoupled from any specific encoder's sample-rate requirements. Must
    /// be called off the realtime audio thread - see trait docs.
    fn embed(&self, audio: &[f32], input_sample_rate_hz: u32) -> Result<TargetSpeakerProfile, SpeakerEncoderError>;
}

/// The fallback `SpeakerEncoder`: a real, explicit "no backend is
/// available" implementation, used when the real one
/// (`speaker_encoder::WeSpeakerEncoder`) fails to load (or on a build
/// that doesn't link it in at all). It exists so the rest of the enrollment
/// pipeline (state machine, GUI, tests) can always be built and exercised
/// against a real trait object, without ever pretending a model is present
/// when it isn't - see `docs/TSE_ENROLLMENT.md`'s "what's real vs. not"
/// section.
///
/// (A repository-wide search at the time this subsystem was first built
/// found no speaker-embedding model bundled anywhere in this project; a real
/// one - WeSpeaker ResNet34, not literal ECAPA-TDNN - has since been integrated in the
/// separate `speaker-encoder` crate. The streaming TSE model side is still
/// unavailable: the specific model suggested for it,
/// `penta2himajin/tse-conv-tasnet-48k`, does not exist on Hugging Face -
/// confirmed via a direct repo fetch and a Hub search, neither of which
/// found a match.)
#[derive(Debug, Default, Clone, Copy)]
pub struct UnavailableSpeakerEncoder;

impl SpeakerEncoder for UnavailableSpeakerEncoder {
    fn encoder_id(&self) -> &'static str {
        "unavailable"
    }

    fn embedding_dim(&self) -> Option<usize> {
        None
    }

    fn expected_sample_rate_hz(&self) -> u32 {
        // No real encoder is loaded, so there is no genuine "expected" rate
        // to report; 16000 is the most common convention for this class of
        // model (e.g. ECAPA-TDNN reference implementations) and is used here
        // only as a harmless, documented placeholder - callers must not
        // treat this as a confirmed requirement, since `embed` always fails
        // regardless of what rate the audio is actually at.
        16_000
    }

    fn embed(&self, _audio: &[f32], _input_sample_rate_hz: u32) -> Result<TargetSpeakerProfile, SpeakerEncoderError> {
        Err(SpeakerEncoderError::ModelUnavailable)
    }
}

/// L2-normalizes `embedding` in place. Exposed for a real `SpeakerEncoder`
/// implementation to call if (and only if) its model's documented
/// preprocessing expects unit-norm embeddings - see `NormalizationKind`'s
/// docs. Not called anywhere in this crate on its own initiative: whether to
/// normalize is a per-model decision, not something this portable crate
/// should decide for every future encoder.
pub fn l2_normalize(embedding: &mut [f32]) {
    let norm_sq: f32 = embedding.iter().map(|v| v * v).sum();
    if norm_sq <= 0.0 {
        return; // an all-zero vector has no meaningful direction; leave it as-is rather than divide by zero.
    }
    let norm = norm_sq.sqrt();
    for v in embedding.iter_mut() {
        *v /= norm;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn profile_rejects_empty_embedding() {
        let err = TargetSpeakerProfile::new(Vec::new(), "test-encoder", 16_000, NormalizationKind::L2)
            .unwrap_err();
        assert_eq!(err, ProfileError::EmptyEmbedding);
    }

    #[test]
    fn profile_rejects_non_finite_embedding() {
        let err = TargetSpeakerProfile::new(
            vec![1.0, f32::NAN, 0.5],
            "test-encoder",
            16_000,
            NormalizationKind::None,
        )
        .unwrap_err();
        assert_eq!(err, ProfileError::NonFiniteEmbedding);
    }

    #[test]
    fn profile_carries_metadata_through() {
        let profile = TargetSpeakerProfile::new(vec![0.1, 0.2, 0.3], "ecapa-tdnn-v1", 16_000, NormalizationKind::L2)
            .expect("valid embedding");
        assert_eq!(profile.dimension(), 3);
        assert_eq!(profile.encoder_id(), "ecapa-tdnn-v1");
        assert_eq!(profile.sample_rate_hz(), 16_000);
        assert_eq!(profile.normalization(), NormalizationKind::L2);
        assert_eq!(profile.embedding(), &[0.1, 0.2, 0.3]);
    }

    #[test]
    fn each_profile_gets_a_distinct_increasing_version() {
        let a = TargetSpeakerProfile::new(vec![1.0], "e", 16_000, NormalizationKind::None).unwrap();
        let b = TargetSpeakerProfile::new(vec![1.0], "e", 16_000, NormalizationKind::None).unwrap();
        assert!(b.version() > a.version(), "later profile must have a strictly greater version");
    }

    #[test]
    fn unavailable_encoder_always_reports_model_unavailable() {
        let encoder = UnavailableSpeakerEncoder;
        let audio = vec![0.0f32; 16_000 * 3];
        let err = encoder.embed(&audio, 48_000).unwrap_err();
        assert_eq!(err, SpeakerEncoderError::ModelUnavailable);
        assert!(encoder.embedding_dim().is_none());
    }

    #[test]
    fn l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "expected unit norm, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn l2_normalize_leaves_all_zero_vector_untouched() {
        let mut v = vec![0.0f32, 0.0, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0, 0.0]);
    }

    #[test]
    fn age_seconds_is_near_zero_for_a_freshly_created_profile() {
        let profile = TargetSpeakerProfile::new(vec![1.0], "e", 16_000, NormalizationKind::None).unwrap();
        assert!(profile.age_seconds() < 2);
    }
}
