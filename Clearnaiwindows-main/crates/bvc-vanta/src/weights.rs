//! Loads the real, trained separator weights bundled in
//! `../assets/separator_weights_{manifest.json,blob.bin}` - exported from
//! the actual (weak, epoch-6, val SI-SDR +5.911dB) Colab checkpoint - see
//! `docs/TSE_MODEL_COMPARISON.md`/this crate's `lib.rs` docs for the honest
//! quality caveat. Same name-indexed manifest + flat blob format already
//! used by `tse-bsrnn::weights` for the original model, and already
//! verified against these exact bytes by
//! `research/tse/candidates/vanta-causal-rs/src/bin/load_real.rs` (3,441,536
//! real params, param-count match confirmed).

use crate::separator::{
    AudioDecoderWeights, AudioEncoderWeights, CausalSeparatorWeights, CausalTcnBlockWeights, BLOCKS_PER_REPEAT,
    ENC_CHANNELS, ENC_KERNEL, KERNEL, REPEATS,
};
use serde::Deserialize;
use std::collections::HashMap;

const MANIFEST_JSON: &str = include_str!("../assets/separator_weights_manifest.json");
const BLOB_BYTES: &[u8] = include_bytes!("../assets/separator_weights_blob.bin");

/// Real encoder/decoder weights, once exported from the same checkpoint the
/// separator weights above came from (see this crate's `lib.rs` docs on why
/// this is a SEPARATE export step, and why it's still pending as of this
/// writing - a placeholder empty `{}`/zero-byte pair until then, which
/// `load_encoder_decoder_weights` below refuses to treat as real).
const ENC_DEC_MANIFEST_JSON: &str = include_str!("../assets/encoder_decoder_weights_manifest.json");
const ENC_DEC_BLOB_BYTES: &[u8] = include_bytes!("../assets/encoder_decoder_weights_blob.bin");

#[derive(Deserialize)]
struct TensorInfo {
    #[allow(dead_code)]
    shape: Vec<usize>,
    offset: usize,
    length: usize,
}

#[derive(Debug)]
pub struct WeightLoadError(pub String);

impl std::fmt::Display for WeightLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "vanta weight load error: {}", self.0)
    }
}
impl std::error::Error for WeightLoadError {}

struct Loader {
    manifest: HashMap<String, TensorInfo>,
}

impl Loader {
    fn get(&self, name: &str) -> Result<Vec<f32>, WeightLoadError> {
        let info = self.manifest.get(name).ok_or_else(|| WeightLoadError(format!("missing tensor {name}")))?;
        let start = info.offset;
        let end = start + info.length * 4;
        if end > BLOB_BYTES.len() {
            return Err(WeightLoadError(format!("tensor {name} out of range")));
        }
        Ok(BLOB_BYTES[start..end].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    }
}

/// Parses the bundled manifest+blob into real `CausalSeparatorWeights`.
/// Runs once at startup (this crate's `try_load`), never per-frame.
pub fn load_separator_weights() -> Result<CausalSeparatorWeights, WeightLoadError> {
    let manifest: HashMap<String, TensorInfo> =
        serde_json::from_str(MANIFEST_JSON).map_err(|e| WeightLoadError(format!("parsing manifest: {e}")))?;
    let loader = Loader { manifest };

    let mut blocks = Vec::with_capacity(REPEATS * BLOCKS_PER_REPEAT);
    let mut idx = 0usize;
    for _ in 0..REPEATS {
        for x in 0..BLOCKS_PER_REPEAT {
            let dilation = 1usize << x;
            let p = format!("blocks.{idx}.");
            blocks.push(CausalTcnBlockWeights {
                pointwise_in_w: loader.get(&format!("{p}pointwise_in.weight"))?,
                pointwise_in_b: loader.get(&format!("{p}pointwise_in.bias"))?,
                prelu1: loader.get(&format!("{p}prelu1.weight"))?,
                norm1_gamma: loader.get(&format!("{p}norm1.gamma"))?,
                norm1_beta: loader.get(&format!("{p}norm1.beta"))?,
                depthwise_w: loader.get(&format!("{p}depthwise.weight"))?,
                depthwise_b: loader.get(&format!("{p}depthwise.bias"))?,
                prelu2: loader.get(&format!("{p}prelu2.weight"))?,
                norm2_gamma: loader.get(&format!("{p}norm2.gamma"))?,
                norm2_beta: loader.get(&format!("{p}norm2.beta"))?,
                pointwise_out_w: loader.get(&format!("{p}pointwise_out.weight"))?,
                pointwise_out_b: loader.get(&format!("{p}pointwise_out.bias"))?,
                dilation,
                left_pad: (KERNEL - 1) * dilation,
            });
            idx += 1;
        }
    }

    Ok(CausalSeparatorWeights {
        in_norm_gamma: loader.get("in_norm.gamma")?,
        in_norm_beta: loader.get("in_norm.beta")?,
        in_proj_w: loader.get("in_proj.weight")?,
        in_proj_b: loader.get("in_proj.bias")?,
        speaker_proj_w: loader.get("speaker_proj.weight")?,
        speaker_proj_b: loader.get("speaker_proj.bias")?,
        blocks,
        out_prelu: loader.get("out_prelu.weight")?,
        out_proj_w: loader.get("out_proj.weight")?,
        out_proj_b: loader.get("out_proj.bias")?,
    })
}

/// Real encoder/decoder weights, loaded from `../assets/encoder_decoder_
/// weights_{manifest.json,blob.bin}`. Returns a clear error (never random
/// weights, never silently empty ones) if those assets are still the
/// zero-tensor placeholder - see this crate's `lib.rs` docs: running the
/// separator against an untrained encoder/decoder produces pure noise
/// regardless of separator quality, so this must fail loudly, not degrade
/// quietly.
pub fn load_encoder_decoder_weights() -> Result<(AudioEncoderWeights, AudioDecoderWeights), WeightLoadError> {
    let manifest: HashMap<String, TensorInfo> = serde_json::from_str(ENC_DEC_MANIFEST_JSON)
        .map_err(|e| WeightLoadError(format!("parsing encoder/decoder manifest: {e}")))?;
    if manifest.is_empty() {
        return Err(WeightLoadError(
            "encoder/decoder weights not yet exported/bundled (placeholder assets in use) - \
             see weights.rs::load_encoder_decoder_weights docs"
                .to_string(),
        ));
    }

    let get = |name: &str| -> Result<Vec<f32>, WeightLoadError> {
        let info = manifest.get(name).ok_or_else(|| WeightLoadError(format!("missing tensor {name}")))?;
        let start = info.offset;
        let end = start + info.length * 4;
        if end > ENC_DEC_BLOB_BYTES.len() {
            return Err(WeightLoadError(format!("tensor {name} out of range")));
        }
        Ok(ENC_DEC_BLOB_BYTES[start..end].chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect())
    };

    // Real key names, confirmed from the actual checkpoint's own printed
    // submodule prefixes (['audio_decoder', 'audio_encoder', 'separator',
    // 'speaker_encoder']) rather than guessed - the first guess
    // ("encoder."/"decoder.") was wrong and caught immediately by the
    // export cell's own "NOTHING MATCHED" safeguard.
    let encoder = AudioEncoderWeights { weight: get("audio_encoder.conv.weight")? };
    let decoder = AudioDecoderWeights { weight: get("audio_decoder.deconv.weight")? };

    if encoder.weight.len() != ENC_CHANNELS * ENC_KERNEL {
        return Err(WeightLoadError(format!(
            "encoder weight has {} floats, expected {}",
            encoder.weight.len(),
            ENC_CHANNELS * ENC_KERNEL
        )));
    }
    if decoder.weight.len() != ENC_CHANNELS * ENC_KERNEL {
        return Err(WeightLoadError(format!(
            "decoder weight has {} floats, expected {}",
            decoder.weight.len(),
            ENC_CHANNELS * ENC_KERNEL
        )));
    }

    Ok((encoder, decoder))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_real_weights_load_with_expected_param_count() {
        let w = load_separator_weights().expect("bundled weights must load");
        assert_eq!(w.param_count(), 3_441_536, "must match the real Python export's reported 13,766,144 / 4 bytes");
    }

    #[test]
    fn bundled_encoder_decoder_weights_load_with_expected_shape() {
        // Real export landed (audio_encoder.conv.weight / audio_decoder.
        // deconv.weight, (512, 1, 16) each - confirmed from the actual
        // checkpoint's own printed submodule prefixes, not guessed) -
        // this must now succeed, not fail.
        let (encoder, decoder) = load_encoder_decoder_weights().expect("real encoder/decoder weights must load");
        assert_eq!(encoder.weight.len(), ENC_CHANNELS * ENC_KERNEL);
        assert_eq!(decoder.weight.len(), ENC_CHANNELS * ENC_KERNEL);
    }
}
