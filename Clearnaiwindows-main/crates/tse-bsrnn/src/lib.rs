//! Real BSRNN-based streaming Target Speaker Extraction core, backing a
//! future `tse_engine::TseBackend` implementation. Ported from the
//! REAL-TSE/WeSep `spk_emb_causal_100` checkpoint, whose genuine causal
//! streaming behavior (cached embedding, persistent STFT/LSTM/iSTFT state,
//! no growing buffer) was proven correct in Python first, in
//! `poc/real_tse/` - see that directory's `README.md`/`results.md` for the
//! full real-measurement evidence (correlation 0.9999+/SI-SDR 41-45dB vs.
//! the whole-utterance reference across four chunk sizes; +12.3dB target
//! preservation / -28.9dB background suppression on real two-speaker
//! speech).
//!
//! # Runtime split (a deliberate choice, not an oversight)
//!
//! Reading the real model source
//! (`research/tse/candidates/wesep-real-tse/wesep/modules/separator/bsrnn.py`,
//! `.../speaker/encoder.py`, `.../common/norm.py`) showed the BSRNN
//! separator is built entirely from trivial, well-understood primitives:
//! band-splitting is pure slicing, sub-band normalization is plain
//! `GroupNorm`/`ChannelWiseLayerNorm` (ordinary per-timestep LayerNorm,
//! not cumulative), `band_rnn` is one single-layer unidirectional LSTM per
//! repeat block, `band_comm` is one single-layer bidirectional LSTM over
//! the (fixed, small) band axis only - with no time-recurrence at all -
//! and the mask head is `Conv1d` with `kernel_size=1` (a plain per-timestep
//! matmul) plus `Tanh`/`Sigmoid`. None of that needs an inference runtime.
//! The `ECAPA_TDNN_GLOB_c512` speaker encoder, by contrast, is a real conv
//! architecture (Res2Net/SE-blocks/attentive statistics pooling) - not
//! worth hand-rolling, and well precedented via ONNX (see
//! `crates/speaker-encoder`'s WeSpeaker ResNet34, run the same way).
//!
//! So: the speaker encoder runs through `tract-onnx` (this module,
//! [`SpeakerEncoderOnnx`]); the separator (band-split/sub-band norm/BSNet
//! repeat blocks/mask head) is hand-written Rust math over raw exported
//! weight tensors, added in a later pass - see `docs/TSE_INTEGRATION_PLAN.md`
//! and the approved integration plan for the full phased breakdown.
//!
//! # What's verified so far (this module only)
//!
//! The exact ONNX file bundled here
//! (`assets/ecapa_tdnn_glob_c512.onnx`) was exported from the real,
//! hash-verified checkpoint (see `poc/real_tse/export/export_speaker_encoder_onnx.py`)
//! and cross-checked against the PyTorch reference in `onnxruntime` before
//! ever reaching this crate (max abs diff 0.0000039 on real enrollment
//! audio). This crate's own test (`tests/ecapa_onnx_matches_pytorch.rs`)
//! re-verifies the SAME real fixture loads and matches under `tract-onnx`
//! specifically - the runtime that actually matters for production, since
//! `onnxruntime` succeeding does not guarantee `tract-onnx` handles the
//! same ops (two real, different CAM++ failures earlier in this project's
//! history are exactly why this is checked explicitly rather than assumed).

pub mod backend;
pub mod causal_resampler;
pub mod engine;
pub mod lstm;
pub mod matmul;
pub mod ops;
pub mod separator;
pub mod speaker_fusion;
pub mod stft;
pub mod tse_speaker_encoder;
pub mod weights;

use tract_onnx::prelude::*;

/// The exact ONNX export of `model.spk_ft.spkemb.encoder.spk_model`
/// (`ECAPA_TDNN_GLOB_c512`) from the real, hash-verified
/// `spk_emb_causal_100` checkpoint - see this crate's top-level docs.
const ECAPA_MODEL_BYTES: &[u8] = include_bytes!("../assets/ecapa_tdnn_glob_c512.onnx");

/// Real, verified embedding dimension (matches the checkpoint's own
/// `spk_args.embed_dim` in `config.yaml`, and the shape actually produced
/// by both the PyTorch reference and this ONNX export).
pub const EMBEDDING_DIM: usize = 192;

/// Real, verified fbank feature width this encoder expects (80 mel bins,
/// matching `Fbank_kaldi(num_mel_bins=80, ...)` in the real checkpoint's
/// config - see `poc/real_tse/export/export_speaker_encoder_onnx.py`).
pub const FBANK_DIM: usize = 80;

#[derive(Debug)]
pub enum LoadError {
    ModelParseFailed(String),
}

impl std::fmt::Display for LoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoadError::ModelParseFailed(reason) => {
                write!(f, "failed to load bundled ECAPA_TDNN_GLOB_c512 ONNX model: {reason}")
            }
        }
    }
}

impl std::error::Error for LoadError {}

#[derive(Debug)]
pub enum InferenceError {
    BadInputShape(String),
    RunFailed(String),
    BadOutputShape(String),
}

impl std::fmt::Display for InferenceError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InferenceError::BadInputShape(s) => write!(f, "bad input shape: {s}"),
            InferenceError::RunFailed(s) => write!(f, "inference failed: {s}"),
            InferenceError::BadOutputShape(s) => write!(f, "bad output shape: {s}"),
        }
    }
}

impl std::error::Error for InferenceError {}

/// Real `ECAPA_TDNN_GLOB_c512` speaker encoder, run via `tract-onnx` -
/// mirrors `speaker_encoder::WeSpeakerEncoder`'s exact construction pattern
/// (parse + optimize once, at startup; never on the realtime audio thread
/// or per-frame - this is only ever called once per enrollment).
pub struct SpeakerEncoderOnnx {
    model: std::sync::Arc<TypedRunnableModel>,
}

impl SpeakerEncoderOnnx {
    pub fn new() -> Result<Self, LoadError> {
        let mut cursor = std::io::Cursor::new(ECAPA_MODEL_BYTES);
        let model = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .map_err(|e| LoadError::ModelParseFailed(format!("{e:#}")))?
            .into_optimized()
            .map_err(|e| LoadError::ModelParseFailed(format!("optimizing model: {e:#}")))?
            .into_runnable()
            .map_err(|e| LoadError::ModelParseFailed(format!("making model runnable: {e:#}")))?;
        Ok(Self { model })
    }

    /// `fbank`: real Kaldi-style fbank features, `(num_frames * FBANK_DIM)`
    /// values, row-major `(num_frames, FBANK_DIM)`. Returns a real 192-dim
    /// embedding (not L2-normalized here - the reference model's own
    /// fusion layer, `SpeakerFuseLayer`, consumes the raw embedding
    /// directly; L2 normalization is specific to `speaker-profile`'s own,
    /// unrelated 256-dim profile type, not this one).
    pub fn embed_from_fbank(&self, fbank: &[f32], num_frames: usize) -> Result<Vec<f32>, InferenceError> {
        if fbank.len() != num_frames * FBANK_DIM {
            return Err(InferenceError::BadInputShape(format!(
                "expected {} values ({num_frames} frames x {FBANK_DIM} dim), got {}",
                num_frames * FBANK_DIM,
                fbank.len()
            )));
        }

        let input: Tensor = tract_ndarray::Array3::from_shape_vec((1, num_frames, FBANK_DIM), fbank.to_vec())
            .map_err(|e| InferenceError::BadInputShape(format!("{e}")))?
            .into();

        let outputs = self
            .model
            .run(tvec!(input.into()))
            .map_err(|e| InferenceError::RunFailed(format!("{e:#}")))?;

        let embedding: Vec<f32> = outputs[0]
            .to_plain_array_view::<f32>()
            .map_err(|e| InferenceError::BadOutputShape(format!("{e}")))?
            .iter()
            .copied()
            .collect();

        if embedding.len() != EMBEDDING_DIM {
            return Err(InferenceError::BadOutputShape(format!(
                "model produced {} values, expected {EMBEDDING_DIM}",
                embedding.len()
            )));
        }

        Ok(embedding)
    }
}
