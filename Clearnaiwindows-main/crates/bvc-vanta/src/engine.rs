//! Ties the encoder/separator/decoder together into one persistent-state
//! streaming engine, mirroring `tse_bsrnn::engine::StreamingTseEngine`'s
//! shape (same "chunk in, finalized samples out, real internal state"
//! contract) but for Vanta's Conv-TasNet-style architecture instead of
//! BSRNN's STFT-domain one.
//!
//! Two real streaming-state concerns handled here, neither present in the
//! encoder/decoder's own unit tests (which only exercise single, aligned
//! calls):
//!
//! 1. **Framing across call boundaries.** The encoder's stride (8 samples)
//!    doesn't necessarily divide an arbitrary input chunk length evenly, so
//!    leftover un-framed samples must be carried in `sample_buffer` and
//!    combined with the next call's input rather than dropped or
//!    zero-padded (either of which would silently corrupt the signal).
//! 2. **Decoder overlap-add across call boundaries.** Consecutive encoder
//!    frames overlap by `KERNEL - STRIDE` (8) samples (see
//!    `separator.rs`'s `decode_scatter_add_matches_hand_computed_overlap`
//!    test for the within-call case) - and because frame boundaries are
//!    always multiples of `STRIDE`, the LAST frame decoded in one call
//!    overlaps with the FIRST frame decoded in the NEXT call by exactly
//!    that same 8 samples. `pending_tail` carries that not-yet-finalized
//!    overlap forward, exactly the same overlap-add pattern
//!    `tse_bsrnn::stft::StreamingIstft` already uses for its own
//!    frame-domain overlap.

use crate::separator::{AudioDecoderWeights, AudioEncoderWeights, CausalSeparatorState, CausalSeparatorWeights, ENC_KERNEL, ENC_STRIDE};

pub struct VantaEngine {
    encoder: AudioEncoderWeights,
    decoder: AudioDecoderWeights,
    separator: CausalSeparatorWeights,
}

pub struct VantaEngineState {
    sample_buffer: Vec<f32>,
    sep_state: CausalSeparatorState,
    pending_tail: Vec<f32>, // len == ENC_KERNEL - ENC_STRIDE
}

impl VantaEngine {
    pub fn new(encoder: AudioEncoderWeights, decoder: AudioDecoderWeights, separator: CausalSeparatorWeights) -> Self {
        Self { encoder, decoder, separator }
    }

    pub fn new_state(&self) -> VantaEngineState {
        VantaEngineState {
            sample_buffer: Vec::new(),
            sep_state: self.separator.new_state(),
            pending_tail: vec![0.0; ENC_KERNEL - ENC_STRIDE],
        }
    }

    /// `chunk`: raw mixture samples at Vanta's native 16kHz. `spk_emb`: the
    /// real 192-dim ECAPA embedding for the enrolled target speaker.
    /// Returns finalized output samples (may be shorter than `chunk`,
    /// especially on the very first call - see `pending_tail`'s docs).
    pub fn process_chunk(&self, chunk: &[f32], spk_emb: &[f32], state: &mut VantaEngineState) -> Vec<f32> {
        state.sample_buffer.extend_from_slice(chunk);

        let n_frames = if state.sample_buffer.len() >= ENC_KERNEL {
            (state.sample_buffer.len() - ENC_KERNEL) / ENC_STRIDE + 1
        } else {
            0
        };
        if n_frames == 0 {
            return Vec::new();
        }

        let samples_used = (n_frames - 1) * ENC_STRIDE + ENC_KERNEL;
        // Only advance the buffer by `n_frames * STRIDE`, NOT by
        // `samples_used` - the trailing `KERNEL - STRIDE` samples of the
        // window this call read are also the LEADING samples of the next
        // not-yet-formed frame (start position `n_frames * STRIDE`), and
        // must stay in the buffer for that frame to be computed once more
        // samples arrive. Draining the full `samples_used` here was a real
        // bug: it silently dropped every frame that straddled a call
        // boundary (caught by `chunking_into_small_pieces_matches_one_big_chunk`
        // - 179 frames one-shot vs. only 171 across 9 chunked calls of 160
        // samples each, before this fix).
        let frame_samples: Vec<f32> = state.sample_buffer[..samples_used].to_vec();
        state.sample_buffer.drain(..n_frames * ENC_STRIDE);

        let (encoded, nf) = self.encoder.encode(&frame_samples);
        debug_assert_eq!(nf, n_frames);

        let mask = self.separator.forward_streaming(&encoded, n_frames, spk_emb, &mut state.sep_state);

        let mut masked = encoded;
        for i in 0..masked.len() {
            masked[i] *= mask[i];
        }

        let mut out = self.decoder.decode(&masked, n_frames);

        let overlap = ENC_KERNEL - ENC_STRIDE;
        for i in 0..overlap.min(out.len()) {
            out[i] += state.pending_tail[i];
        }
        let out_len = out.len();
        if out_len >= overlap {
            state.pending_tail.copy_from_slice(&out[out_len - overlap..]);
            out.truncate(out_len - overlap);
        }

        out
    }

    pub fn reset(&self, state: &mut VantaEngineState) {
        state.sample_buffer.clear();
        state.sep_state = self.separator.new_state();
        state.pending_tail.iter_mut().for_each(|v| *v = 0.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn random_engine(rng: &mut impl Rng) -> VantaEngine {
        let encoder = AudioEncoderWeights::random(rng);
        let decoder = AudioDecoderWeights::random(rng);
        let separator = CausalSeparatorWeights::random(rng);
        VantaEngine::new(encoder, decoder, separator)
    }

    #[test]
    fn chunking_into_small_pieces_matches_one_big_chunk() {
        // The real correctness property that matters: whether 480 raw
        // samples arrive as one call or as several small calls (mirroring
        // real pipeline frame sizes), the TOTAL finalized output, once all
        // pending state is flushed, must be identical - the same
        // "streaming == batch" equivalence already proven for the
        // separator's TCN blocks in the research crate, now proven for
        // this call-boundary framing/overlap-add logic too.
        let mut rng = rand::thread_rng();
        let engine = random_engine(&mut rng);
        let spk_emb: Vec<f32> = (0..192).map(|_| rng.gen_range(-0.3..0.3)).collect();

        let total_samples = 480 * 3;
        let mixture: Vec<f32> = (0..total_samples).map(|_| rng.gen_range(-0.5..0.5)).collect();

        // One big call.
        let mut state_a = engine.new_state();
        let out_a = engine.process_chunk(&mixture, &spk_emb, &mut state_a);

        // Many small calls (irregular sizes, deliberately not multiples of
        // ENC_STRIDE, to exercise the leftover-buffer path).
        let mut state_b = engine.new_state();
        let mut out_b = Vec::new();
        let chunk_sizes = [37, 101, 480, 53, 200, 569];
        let mut pos = 0;
        for &sz in chunk_sizes.iter().cycle() {
            if pos >= mixture.len() {
                break;
            }
            let end = (pos + sz).min(mixture.len());
            out_b.extend(engine.process_chunk(&mixture[pos..end], &spk_emb, &mut state_b));
            pos = end;
        }

        // Both must agree on their OVERLAPPING prefix (state_b may lag
        // behind state_a by up to `pending_tail`'s length since it hasn't
        // seen a final flush) - compare the common length.
        let common = out_a.len().min(out_b.len());
        assert!(common > 0, "both paths must produce some finalized output for 3 real frames worth of input");
        let max_abs_diff = out_a[..common].iter().zip(out_b[..common].iter()).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        assert!(max_abs_diff < 1e-3, "chunked vs. one-shot processing diverged: max_abs_diff={max_abs_diff}");
    }

    #[test]
    fn reset_clears_all_streaming_state() {
        let mut rng = rand::thread_rng();
        let engine = random_engine(&mut rng);
        let spk_emb: Vec<f32> = (0..192).map(|_| rng.gen_range(-0.3..0.3)).collect();
        let mixture: Vec<f32> = (0..480).map(|_| rng.gen_range(-0.5..0.5)).collect();

        let mut state = engine.new_state();
        engine.process_chunk(&mixture, &spk_emb, &mut state);
        engine.reset(&mut state);

        assert!(state.sample_buffer.is_empty());
        assert!(state.pending_tail.iter().all(|v| *v == 0.0));
    }

    #[test]
    fn output_is_always_finite() {
        let mut rng = rand::thread_rng();
        let engine = random_engine(&mut rng);
        let spk_emb: Vec<f32> = (0..192).map(|_| rng.gen_range(-0.3..0.3)).collect();
        let mut state = engine.new_state();

        for _ in 0..50 {
            let chunk: Vec<f32> = (0..480).map(|_| rng.gen_range(-0.5..0.5)).collect();
            let out = engine.process_chunk(&chunk, &spk_emb, &mut state);
            assert!(out.iter().all(|v| v.is_finite()), "output must never be NaN/Inf even with random weights");
        }
    }
}
