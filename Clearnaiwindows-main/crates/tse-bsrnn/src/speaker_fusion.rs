//! The real speaker-conditioning fusion: `SpeakerFuseLayer` with
//! `fuse_type="multiply"` (`wesep/modules/fusion/speech.py`), confirmed by
//! reading the real source to be exactly: project the 192-dim ECAPA
//! embedding through a single `Linear(192, 128)` (`fc`), then broadcast
//! that 128-dim vector across every band and timestep and elementwise-
//! multiply it into `subband_feature`. The projection is a fixed function
//! of the (cached, once-per-enrollment) embedding alone - so it is
//! computed ONCE here (`project_embedding`), not per frame; only the
//! resulting 128-dim gate vector needs to be applied per chunk (see
//! `separator::BsrnnSeparator::apply_speaker_gate`).

const EMBED_DIM: usize = 192;
const FEATURE_DIM: usize = 128;

const FUSION_WEIGHT: &[u8] = include_bytes!("../assets/fusion_weight.bin"); // (128, 192) f32 LE
const FUSION_BIAS: &[u8] = include_bytes!("../assets/fusion_bias.bin"); // (128,) f32 LE

fn read_le_f32(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Projects a real 192-dim ECAPA embedding (from `SpeakerEncoderOnnx`) into
/// the 128-dim gate vector used by `BsrnnSeparator::apply_speaker_gate`.
/// Call this ONCE per enrollment (or target-speaker change), never per
/// frame - matches this crate's cached-embedding design throughout.
pub fn project_embedding(embedding: &[f32; EMBED_DIM]) -> [f32; FEATURE_DIM] {
    let weight = read_le_f32(FUSION_WEIGHT);
    let bias = read_le_f32(FUSION_BIAS);
    debug_assert_eq!(weight.len(), FEATURE_DIM * EMBED_DIM);
    debug_assert_eq!(bias.len(), FEATURE_DIM);

    let mut out = [0.0f32; FEATURE_DIM];
    for o in 0..FEATURE_DIM {
        let row = &weight[o * EMBED_DIM..(o + 1) * EMBED_DIM];
        let mut acc = bias[o];
        for i in 0..EMBED_DIM {
            acc += row[i] * embedding[i];
        }
        out[o] = acc;
    }
    out
}
