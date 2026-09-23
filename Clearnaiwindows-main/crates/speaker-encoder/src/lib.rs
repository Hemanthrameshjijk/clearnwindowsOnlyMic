//! Real `speaker_profile::SpeakerEncoder` backend: a bundled ONNX WeSpeaker
//! ResNet34 (LM) speaker-verification model, run via `tract-onnx` (pure
//! Rust, CPU-only, no external DLL/dynamic loading - unlike `bvc-hush`'s
//! `weya_nc.dll`, this model is embedded directly in the binary via
//! `include_bytes!` and parsed entirely in memory).
//!
//! # Why WeSpeaker ResNet34, not ECAPA-TDNN or CAM++
//!
//! The task this crate was built for suggested ECAPA-TDNN specifically.
//! Before writing any code, real, currently-downloadable model options were
//! checked directly rather than assumed - and the path taken involved two
//! real, verified rejections before landing on the model actually bundled:
//!
//! 1. No literal "ECAPA-TDNN" `.onnx` asset exists in
//!    `k2-fsa/sherpa-onnx`'s speaker-recognition-models release (checked via
//!    the real GitHub Releases API). Informal ECAPA-TDNN ONNX re-uploads
//!    exist on Hugging Face from individual accounts with no stable
//!    maintenance signal - not adopted, per this project's "do not
//!    integrate something of unverified provenance" rule.
//! 2. `3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx` (CAM++, from
//!    Alibaba's 3D-Speaker/ModelScope project, packaged by `k2-fsa/sherpa-onnx`)
//!    was tried **first** and rejected after real, hands-on verification
//!    work, not by inspection alone: a throwaway `tract-onnx` probe binary
//!    was built and actually run against the real bundled file.
//!    - Under `tract-onnx` 0.19.16 (the version already in this workspace
//!      via `deep_filter`), model *loading* itself failed: two different
//!      dynamic-shape symbols on the same `BatchNormalization` node's
//!      input/output couldn't be unified, even after this model's known-1
//!      batch dimension was pinned to a concrete literal.
//!    - Under `tract-onnx` 0.21.18, loading succeeded, but actual
//!      *inference* failed inside CAM++'s "context-aware masking" TDNN
//!      layer: a `Slice` op computed an out-of-range index (an internal
//!      shape-propagation bug specific to that layer's dynamic-size split/
//!      slice pattern).
//!    Two different real failures across two different tract major versions,
//!    both against the actual bundled file (not simulated/assumed) - this
//!    was treated as a genuine incompatibility with this specific
//!    architecture/export, not something to keep patching around.
//! 3. `wespeaker_en_voxceleb_resnet34_LM.onnx` (WeSpeaker's own official
//!    ResNet34, "large-margin fine-tuned" checkpoint, packaged by the same
//!    `k2-fsa/sherpa-onnx` project, hosted at WeSpeaker's own official
//!    `wespeaker-...myqcloud.com` URL per its embedded metadata) is a plain
//!    ResNet architecture (Conv/BatchNorm/ReLU/pooling - the same class of
//!    ops the well-known `tract` mobilenet-v2 example already runs cleanly),
//!    with a consistently-named batch-size symbol between input and output
//!    (unlike the CAM++ export). Verified end-to-end, for real: the same
//!    probe binary, using `tract-onnx` 0.21.18, loaded this file,
//!    optimized it, ran inference on synthetic input, and produced a
//!    finite, correctly-shaped `[1, 256]` output.
//!
//! `speaker_profile::SpeakerEncoder` is architecture-agnostic (it doesn't
//! name ECAPA-TDNN anywhere in its trait shape), so this substitution is a
//! real, honestly-documented implementation choice, not a workaround.
//!
//! # Verified vs. not
//!
//! Verified by direct inspection/execution in this environment:
//! - The bundled file's real byte size, SHA-256, and embedded ONNX
//!   metadata (via raw byte inspection of the protobuf, including the exact
//!   `normalize_samples` byte value - see `fbank.rs`'s docs).
//! - That `tract-onnx` 0.21.18 actually loads this file, optimizes it, and
//!   runs inference end-to-end, reporting input `feats: [B, T, 80] f32` and
//!   producing a real `[1, 256] f32` output with finite values - run for
//!   real via a throwaway probe binary against the bundled file.
//! - The exact feature-extraction configuration and per-model quirks
//!   (`normalize_samples=0`, no `feature_normalize_type`), sourced from
//!   `sherpa-onnx`'s real, current C++ source plus this model's own
//!   metadata (see `fbank.rs`'s docs for the specific files/bytes read).
//!
//! **Not** verified: bit-exact numerical equivalence between this crate's
//! fbank/resampling/inference pipeline and the reference Python/C++
//! pipeline on real speech (no Python, no reference binary, and no real
//! multi-speaker speech corpus exist in this sandboxed environment). Nor is
//! this model's actual speaker-verification accuracy (target-preservation/
//! impostor-rejection) measured here - that requires real recordings of
//! real speakers, which this environment cannot provide. See
//! `docs/TSE_ENROLLMENT.md`.

mod fbank;
mod resample;

use speaker_profile::{NormalizationKind, SpeakerEncoder, SpeakerEncoderError, TargetSpeakerProfile};
use tract_onnx::prelude::*;

/// The exact bundled model file - see this crate's top-level docs for its
/// verified provenance. WeSpeaker's own official `voxceleb_resnet34_LM`
/// checkpoint, packaged as ONNX by `k2-fsa/sherpa-onnx`.
const MODEL_BYTES: &[u8] = include_bytes!("../assets/wespeaker_en_voxceleb_resnet34_LM.onnx");

/// Stable identifier stored in every `TargetSpeakerProfile` this encoder
/// produces - see `SpeakerEncoder::encoder_id`'s docs on why a future TSE
/// backend should check this before trusting a profile's embedding format.
pub const ENCODER_ID: &str = "wespeaker-resnet34-lm-en-voxceleb-16k";

/// The real speaker-embedding dimension this model produces (verified via
/// both its `output_dim` ONNX metadata and by actually running a
/// `tract-onnx` probe against the bundled file - see crate docs).
pub const EMBEDDING_DIM: usize = 256;

#[derive(Debug)]
pub enum LoadError {
    /// `tract-onnx` failed to parse or optimize the bundled model bytes.
    /// Should not happen in practice (the bytes are fixed at compile time
    /// and were verified to load once - see crate docs) but is surfaced as
    /// a typed error rather than a panic/`.expect()`, matching this
    /// project's "never crash on a load failure, report unavailable"
    /// convention (see `bvc_hush::HushLoadError`).
    ModelParseFailed(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::ModelParseFailed(reason) => write!(f, "failed to load bundled speaker-embedding ONNX model: {reason}"),
        }
    }
}

impl std::error::Error for LoadError {}

/// Real `SpeakerEncoder` backend - see crate docs for what "real" means
/// here and what remains unverified. Constructed once (parsing/optimizing
/// the ~25MB bundled ONNX model takes real, non-trivial time - this must
/// never happen on the realtime audio thread, or per-enrollment-attempt;
/// see `shared::build_enrollment_subsystem` for where construction happens,
/// once, at engine startup).
pub struct WeSpeakerEncoder {
    model: std::sync::Arc<TypedRunnableModel>,
}

impl WeSpeakerEncoder {
    pub fn new() -> Result<Self, LoadError> {
        let mut cursor = std::io::Cursor::new(MODEL_BYTES);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| LoadError::ModelParseFailed(format!("{e:#}")))?
            .into_optimized()
            .map_err(|e| LoadError::ModelParseFailed(format!("optimizing model: {e:#}")))?
            .into_runnable()
            .map_err(|e| LoadError::ModelParseFailed(format!("making model runnable: {e:#}")))?;
        Ok(Self { model })
    }
}

impl SpeakerEncoder for WeSpeakerEncoder {
    fn encoder_id(&self) -> &'static str {
        ENCODER_ID
    }

    fn embedding_dim(&self) -> Option<usize> {
        Some(EMBEDDING_DIM)
    }

    fn expected_sample_rate_hz(&self) -> u32 {
        fbank::FBANK_SAMPLE_RATE_HZ
    }

    fn embed(&self, audio: &[f32], input_sample_rate_hz: u32) -> Result<TargetSpeakerProfile, SpeakerEncoderError> {
        let resampled = resample::resample_to_fbank_rate(audio, input_sample_rate_hz)
            .map_err(SpeakerEncoderError::InvalidAudio)?;

        let (features, num_frames) =
            fbank::compute_fbank(&resampled).map_err(SpeakerEncoderError::InvalidAudio)?;

        let input: Tensor = tract_ndarray::Array3::from_shape_vec((1, num_frames, fbank::FEAT_DIM), features)
            .map_err(|e| SpeakerEncoderError::InferenceFailed(format!("building input tensor: {e}")))?
            .into();

        let outputs = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| SpeakerEncoderError::InferenceFailed(format!("{e:#}")))?;

        let embedding: Vec<f32> = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| SpeakerEncoderError::InferenceFailed(format!("reading output tensor: {e}")))?
            .iter()
            .copied()
            .collect();

        if embedding.len() != EMBEDDING_DIM {
            return Err(SpeakerEncoderError::InferenceFailed(format!(
                "model produced {} values, expected {EMBEDDING_DIM}",
                embedding.len()
            )));
        }

        let mut embedding = embedding;
        speaker_profile::l2_normalize(&mut embedding);

        TargetSpeakerProfile::new(embedding, ENCODER_ID, fbank::FBANK_SAMPLE_RATE_HZ, NormalizationKind::L2)
            .map_err(|e| SpeakerEncoderError::InferenceFailed(e.to_string()))
    }
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
    fn model_loads_without_error() {
        WeSpeakerEncoder::new().expect("bundled model must load");
    }

    #[test]
    fn embed_on_a_synthetic_tone_produces_a_256_dim_unit_norm_embedding() {
        let encoder = WeSpeakerEncoder::new().expect("bundled model must load");
        let audio = tone(48_000, 3.0, 0.3);
        let profile = encoder.embed(&audio, 48_000).expect("inference must succeed on valid audio");
        assert_eq!(profile.dimension(), EMBEDDING_DIM);
        assert_eq!(profile.encoder_id(), ENCODER_ID);
        assert_eq!(profile.normalization(), NormalizationKind::L2);
        assert!(profile.embedding().iter().all(|v| v.is_finite()));
        let norm: f32 = profile.embedding().iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-3, "expected L2-normalized output, norm was {norm}");
    }

    #[test]
    fn embed_works_directly_at_16khz_without_resampling() {
        let encoder = WeSpeakerEncoder::new().expect("bundled model must load");
        let audio = tone(16_000, 2.0, 0.3);
        let profile = encoder.embed(&audio, 16_000).expect("inference must succeed at the model's native rate");
        assert_eq!(profile.dimension(), EMBEDDING_DIM);
    }

    #[test]
    fn embedding_dim_and_expected_sample_rate_are_reported_before_any_inference() {
        let encoder = WeSpeakerEncoder::new().expect("bundled model must load");
        assert_eq!(encoder.embedding_dim(), Some(EMBEDDING_DIM));
        assert_eq!(encoder.expected_sample_rate_hz(), 16_000);
    }

    #[test]
    fn two_different_synthetic_tones_produce_different_embeddings() {
        // Not a speaker-verification correctness test (that requires real
        // speech from real speakers, which this environment cannot provide -
        // see crate docs) - only a sanity check that the model is not
        // degenerately returning a constant vector regardless of input.
        let encoder = WeSpeakerEncoder::new().expect("bundled model must load");
        let a = encoder.embed(&tone(48_000, 2.0, 0.2), 48_000).unwrap();
        let b = encoder.embed(&tone(48_000, 2.0, 0.6), 48_000).unwrap();
        let diff: f32 = a.embedding().iter().zip(b.embedding()).map(|(x, y)| (x - y).powi(2)).sum::<f32>().sqrt();
        assert!(diff > 1e-4, "expected different inputs to produce distinguishable embeddings");
    }
}
