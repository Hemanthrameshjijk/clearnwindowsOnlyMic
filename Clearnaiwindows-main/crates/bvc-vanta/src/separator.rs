//! Direct port of `research/tse/candidates/vanta-causal-rs/src/lib.rs` -
//! same architecture, numerically verified against the real Python
//! reference (see that crate's `src/bin/verify.rs`: max_abs_diff
//! 0.00009918 on random weights) and against the real trained checkpoint
//! (`src/bin/load_real.rs`: 3,441,536 real params load correctly, finite
//! non-degenerate output). Kept as an unmodified copy rather than a
//! `path` dependency so this crate has no dependency on the isolated
//! research workspace (see that crate's own `[workspace]` isolation
//! docs).

//! Real, honest Rust-level RTF measurement for the causal Vanta separator
//! (see `../vanta-causal/causal_separator.py`, the Python reference this
//! ports from - same architecture, same real Vanta hyperparameters,
//! random weights since nothing is trained yet - speed depends on the
//! compute graph's shape, not weight values).
//!
//! Standalone binary, not wired into anything - see this directory's
//! `Cargo.toml` docs for why.

use ndarray::ArrayView2;
use rand::Rng;

// Real Vanta hyperparameters (from the actual repo's separator.py/
// audio_encoder.py defaults) - not invented.
pub const ENC_CHANNELS: usize = 512; // N
pub const BOTTLENECK: usize = 128; // B_chan
pub const HIDDEN: usize = 512; // H
pub const KERNEL: usize = 3; // P
pub const BLOCKS_PER_REPEAT: usize = 8;
pub const REPEATS: usize = 3;
pub const SPEAKER_DIM: usize = 192;

// 10ms of audio @16kHz, through Vanta's AudioEncoder (kernel=16, stride=8)
// -> 160/8 = 20 encoded frames. Same reasoning as the Python benchmark.
pub const ENCODED_FRAMES_PER_10MS_CHUNK: usize = 20;
pub const SECONDS_TO_SIMULATE: f64 = 5.0;

pub fn rand_vec(n: usize, rng: &mut impl Rng) -> Vec<f32> {
    (0..n).map(|_| rng.gen_range(-0.5..0.5)).collect()
}

/// `Y = X @ W^T + b`. `x`: (rows, in_dim) row-major. `weight`: (out_dim,
/// in_dim) row-major (PyTorch `nn.Linear`/`Conv1d(k=1)` convention).
/// Returns (rows, out_dim) row-major. Same proven approach as this
/// project's real `tse-bsrnn::matmul::batched_linear` - batching the
/// per-timestep 1x1-conv/Linear ops across all timesteps in a chunk in
/// ONE matmul, instead of one tiny matvec per timestep.
pub fn batched_linear(x: &[f32], rows: usize, in_dim: usize, weight: &[f32], bias: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    let x_arr = ArrayView2::from_shape((rows, in_dim), x).unwrap();
    let w_arr = ArrayView2::from_shape((out_dim, in_dim), weight).unwrap();
    let y = x_arr.dot(&w_arr.t());
    let mut out = y.into_raw_vec();
    if let Some(b) = bias {
        for r in 0..rows {
            for o in 0..out_dim {
                out[r * out_dim + o] += b[o];
            }
        }
    }
    out
}

pub fn prelu(x: &mut [f32], slope: &[f32], channels: usize, rows: usize) {
    for r in 0..rows {
        for c in 0..channels {
            let v = x[r * channels + c];
            if v < 0.0 {
                x[r * channels + c] = v * slope[c];
            }
        }
    }
}

/// Scalar running (sum, sqsum, count) - the norm collapses BOTH channel
/// and time-so-far into one running statistic per batch item, matching
/// `CausalCumulativeLayerNorm`'s real Python semantics (`dim=(1,2)` in the
/// original `GlobalLayerNorm` this replaces).
#[derive(Clone, Copy, Default)]
pub struct NormState {
    pub sum: f32,
    pub sqsum: f32,
    pub count: f32,
}

/// In-place causal cumulative LayerNorm over a chunk: `x` is (rows,
/// channels) row-major. Mirrors `CausalCumulativeLayerNorm.forward_streaming`
/// exactly (same incremental-cumsum formulation, real not approximated).
pub fn cumulative_norm_streaming(x: &mut [f32], rows: usize, channels: usize, gamma: &[f32], beta: &[f32], eps: f32, state: &mut NormState) {
    for r in 0..rows {
        let row = &x[r * channels..(r + 1) * channels];
        let step_sum: f32 = row.iter().sum();
        let step_sqsum: f32 = row.iter().map(|v| v * v).sum();
        state.sum += step_sum;
        state.sqsum += step_sqsum;
        state.count += channels as f32;
        let mean = state.sum / state.count;
        let var = (state.sqsum / state.count - mean * mean).max(0.0);
        let inv_std = 1.0 / (var + eps).sqrt();
        for c in 0..channels {
            let idx = r * channels + c;
            x[idx] = (x[idx] - mean) * inv_std * gamma[c] + beta[c];
        }
    }
}

pub struct CausalTcnBlockWeights {
    pub pointwise_in_w: Vec<f32>, // (hidden, bottleneck)
    pub pointwise_in_b: Vec<f32>, // (hidden,)
    pub prelu1: Vec<f32>,         // (hidden,)
    pub norm1_gamma: Vec<f32>,
    pub norm1_beta: Vec<f32>,
    pub depthwise_w: Vec<f32>, // (hidden, kernel) - depthwise: one length-`kernel` filter per channel
    pub depthwise_b: Vec<f32>, // (hidden,)
    pub prelu2: Vec<f32>,
    pub norm2_gamma: Vec<f32>,
    pub norm2_beta: Vec<f32>,
    pub pointwise_out_w: Vec<f32>, // (bottleneck, hidden)
    pub pointwise_out_b: Vec<f32>, // (bottleneck,)
    pub dilation: usize,
    pub left_pad: usize,
}

pub struct CausalTcnBlockState {
    pub norm1: NormState,
    pub norm2: NormState,
    pub conv_history: Vec<f32>, // (left_pad, hidden) row-major
}

impl CausalTcnBlockWeights {
    pub fn random(rng: &mut impl Rng, dilation: usize) -> Self {
        Self {
            pointwise_in_w: rand_vec(HIDDEN * BOTTLENECK, rng),
            pointwise_in_b: rand_vec(HIDDEN, rng),
            prelu1: vec![0.25; HIDDEN],
            norm1_gamma: vec![1.0; HIDDEN],
            norm1_beta: vec![0.0; HIDDEN],
            depthwise_w: rand_vec(HIDDEN * KERNEL, rng),
            depthwise_b: rand_vec(HIDDEN, rng),
            prelu2: vec![0.25; HIDDEN],
            norm2_gamma: vec![1.0; HIDDEN],
            norm2_beta: vec![0.0; HIDDEN],
            pointwise_out_w: rand_vec(BOTTLENECK * HIDDEN, rng),
            pointwise_out_b: rand_vec(BOTTLENECK, rng),
            dilation,
            left_pad: (KERNEL - 1) * dilation,
        }
    }

    pub fn new_state(&self) -> CausalTcnBlockState {
        CausalTcnBlockState { norm1: NormState::default(), norm2: NormState::default(), conv_history: vec![0.0; self.left_pad * HIDDEN] }
    }

    /// `x_chunk`: (t, bottleneck) row-major. `spk_bias`: (bottleneck,).
    /// Returns (t, bottleneck) row-major - mirrors
    /// `CausalTCNBlock.forward_streaming` exactly.
    pub fn forward_streaming(&self, x_chunk: &[f32], t: usize, spk_bias: &[f32], state: &mut CausalTcnBlockState) -> Vec<f32> {
        let residual = x_chunk.to_vec();

        let mut h_in = vec![0.0f32; t * BOTTLENECK];
        for r in 0..t {
            for c in 0..BOTTLENECK {
                h_in[r * BOTTLENECK + c] = x_chunk[r * BOTTLENECK + c] + spk_bias[c];
            }
        }

        let mut h = batched_linear(&h_in, t, BOTTLENECK, &self.pointwise_in_w, Some(&self.pointwise_in_b), HIDDEN);
        prelu(&mut h, &self.prelu1, HIDDEN, t);
        cumulative_norm_streaming(&mut h, t, HIDDEN, &self.norm1_gamma, &self.norm1_beta, 1e-8, &mut state.norm1);

        // Prepend left-context history, then causal depthwise conv
        // (groups=hidden: each output channel only convolves with its OWN
        // channel's history + current values, over `KERNEL` taps at this
        // block's dilation).
        let total_len = self.left_pad + t;
        let mut h_with_history = vec![0.0f32; total_len * HIDDEN];
        h_with_history[..self.left_pad * HIDDEN].copy_from_slice(&state.conv_history);
        h_with_history[self.left_pad * HIDDEN..].copy_from_slice(&h);

        let mut conv_out = vec![0.0f32; t * HIDDEN];
        for out_t in 0..t {
            for c in 0..HIDDEN {
                let mut acc = self.depthwise_b[c];
                for k in 0..KERNEL {
                    // Position in h_with_history this tap reads from -
                    // dilation-spaced taps ending at (left_pad + out_t).
                    let src_t = self.left_pad + out_t - (KERNEL - 1 - k) * self.dilation;
                    acc += h_with_history[src_t * HIDDEN + c] * self.depthwise_w[c * KERNEL + k];
                }
                conv_out[out_t * HIDDEN + c] = acc;
            }
        }
        let new_history_start = total_len - self.left_pad;
        state.conv_history.copy_from_slice(&h_with_history[new_history_start * HIDDEN..]);

        let mut h2 = conv_out;
        prelu(&mut h2, &self.prelu2, HIDDEN, t);
        cumulative_norm_streaming(&mut h2, t, HIDDEN, &self.norm2_gamma, &self.norm2_beta, 1e-8, &mut state.norm2);

        let out_delta = batched_linear(&h2, t, HIDDEN, &self.pointwise_out_w, Some(&self.pointwise_out_b), BOTTLENECK);

        let mut out = residual;
        for i in 0..out.len() {
            out[i] += out_delta[i];
        }
        out
    }
}

pub struct CausalSeparatorWeights {
    pub in_norm_gamma: Vec<f32>,
    pub in_norm_beta: Vec<f32>,
    pub in_proj_w: Vec<f32>, // (bottleneck, enc_channels)
    pub in_proj_b: Vec<f32>,
    pub speaker_proj_w: Vec<f32>, // (bottleneck, speaker_dim)
    pub speaker_proj_b: Vec<f32>,
    pub blocks: Vec<CausalTcnBlockWeights>,
    pub out_prelu: Vec<f32>,
    pub out_proj_w: Vec<f32>, // (enc_channels, bottleneck)
    pub out_proj_b: Vec<f32>,
}

pub struct CausalSeparatorState {
    pub in_norm: NormState,
    pub blocks: Vec<CausalTcnBlockState>,
}

impl CausalSeparatorWeights {
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut blocks = Vec::with_capacity(REPEATS * BLOCKS_PER_REPEAT);
        for _ in 0..REPEATS {
            for x in 0..BLOCKS_PER_REPEAT {
                blocks.push(CausalTcnBlockWeights::random(rng, 1 << x));
            }
        }
        Self {
            in_norm_gamma: vec![1.0; ENC_CHANNELS],
            in_norm_beta: vec![0.0; ENC_CHANNELS],
            in_proj_w: rand_vec(BOTTLENECK * ENC_CHANNELS, rng),
            in_proj_b: rand_vec(BOTTLENECK, rng),
            speaker_proj_w: rand_vec(BOTTLENECK * SPEAKER_DIM, rng),
            speaker_proj_b: rand_vec(BOTTLENECK, rng),
            blocks,
            out_prelu: vec![0.25; BOTTLENECK],
            out_proj_w: rand_vec(ENC_CHANNELS * BOTTLENECK, rng),
            out_proj_b: rand_vec(ENC_CHANNELS, rng),
        }
    }

    pub fn new_state(&self) -> CausalSeparatorState {
        CausalSeparatorState { in_norm: NormState::default(), blocks: self.blocks.iter().map(|b| b.new_state()).collect() }
    }

    pub fn param_count(&self) -> usize {
        let mut n = self.in_norm_gamma.len() + self.in_norm_beta.len() + self.in_proj_w.len() + self.in_proj_b.len();
        n += self.speaker_proj_w.len() + self.speaker_proj_b.len();
        n += self.out_prelu.len() + self.out_proj_w.len() + self.out_proj_b.len();
        for b in &self.blocks {
            n += b.pointwise_in_w.len() + b.pointwise_in_b.len() + b.prelu1.len() + b.norm1_gamma.len() + b.norm1_beta.len();
            n += b.depthwise_w.len() + b.depthwise_b.len() + b.prelu2.len() + b.norm2_gamma.len() + b.norm2_beta.len();
            n += b.pointwise_out_w.len() + b.pointwise_out_b.len();
        }
        n
    }

    /// `enc_mix_chunk`: (t, enc_channels) row-major. `spk_emb`: (speaker_dim,).
    pub fn forward_streaming(&self, enc_mix_chunk: &[f32], t: usize, spk_emb: &[f32], state: &mut CausalSeparatorState) -> Vec<f32> {
        let mut h = enc_mix_chunk.to_vec();
        cumulative_norm_streaming(&mut h, t, ENC_CHANNELS, &self.in_norm_gamma, &self.in_norm_beta, 1e-8, &mut state.in_norm);
        let mut h = batched_linear(&h, t, ENC_CHANNELS, &self.in_proj_w, Some(&self.in_proj_b), BOTTLENECK);

        let spk_bias = batched_linear(spk_emb, 1, SPEAKER_DIM, &self.speaker_proj_w, Some(&self.speaker_proj_b), BOTTLENECK);

        for (block, block_state) in self.blocks.iter().zip(state.blocks.iter_mut()) {
            h = block.forward_streaming(&h, t, &spk_bias, block_state);
        }

        prelu(&mut h, &self.out_prelu, BOTTLENECK, t);
        let mut mask = batched_linear(&h, t, BOTTLENECK, &self.out_proj_w, Some(&self.out_proj_b), ENC_CHANNELS);
        for v in mask.iter_mut() {
            *v = v.max(0.0); // ReLU
        }
        mask
    }
}

// --- Audio encoder/decoder (Conv-TasNet style, from the real
// `vanta/models/audio_encoder.py`: plain strided Conv1d/ConvTranspose1d,
// no padding, no bias, kernel=16/stride=8 at the real Vanta hyperparameters
// - already causal by construction (stride-based framing looks only at
// past+current samples for the encoder, and each decoder output tap is a
// deterministic function of a bounded window of frames), so unlike the
// separator's TCN blocks there is no cumulative/global state to convert -
// this is a direct, un-modified line-for-line port. ---

pub const ENC_KERNEL: usize = 16;
pub const ENC_STRIDE: usize = 8;

pub struct AudioEncoderWeights {
    /// PyTorch `Conv1d(1, ENC_CHANNELS, kernel=16, stride=8, bias=False)`
    /// weight, shape (ENC_CHANNELS, 1, ENC_KERNEL) - flattened row-major,
    /// so `weight[c * ENC_KERNEL + k]` is channel `c`'s tap `k`.
    pub weight: Vec<f32>,
}

impl AudioEncoderWeights {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self { weight: rand_vec(ENC_CHANNELS * ENC_KERNEL, rng) }
    }

    pub fn param_count(&self) -> usize {
        self.weight.len()
    }

    /// `n_frames_for` from the real Python: `(samples - kernel) / stride + 1`.
    pub fn n_frames_for(samples: usize) -> usize {
        (samples - ENC_KERNEL) / ENC_STRIDE + 1
    }

    /// `samples`: raw waveform, any length >= ENC_KERNEL. Returns
    /// (n_frames * ENC_CHANNELS,) row-major (n_frames, ENC_CHANNELS) - i.e.
    /// already transposed to the (T, N) layout this crate's separator
    /// expects, rather than PyTorch's native (N, T). ReLU applied, matching
    /// `AudioEncoder.forward`.
    pub fn encode(&self, samples: &[f32]) -> (Vec<f32>, usize) {
        let n_frames = Self::n_frames_for(samples.len());
        let mut out = vec![0.0f32; n_frames * ENC_CHANNELS];
        for t in 0..n_frames {
            let base = t * ENC_STRIDE;
            for c in 0..ENC_CHANNELS {
                let mut acc = 0.0f32;
                for k in 0..ENC_KERNEL {
                    acc += samples[base + k] * self.weight[c * ENC_KERNEL + k];
                }
                out[t * ENC_CHANNELS + c] = acc.max(0.0); // ReLU
            }
        }
        (out, n_frames)
    }
}

pub struct AudioDecoderWeights {
    /// PyTorch `ConvTranspose1d(ENC_CHANNELS, 1, kernel=16, stride=8,
    /// bias=False)` weight, shape (ENC_CHANNELS, 1, ENC_KERNEL) - same
    /// flattened row-major layout as the encoder (PyTorch's native
    /// ConvTranspose1d weight layout is (in_channels, out_channels/groups,
    /// kernel), which for out_channels=1 collapses to the same shape as
    /// the encoder's).
    pub weight: Vec<f32>,
}

impl AudioDecoderWeights {
    pub fn random(rng: &mut impl Rng) -> Self {
        Self { weight: rand_vec(ENC_CHANNELS * ENC_KERNEL, rng) }
    }

    pub fn param_count(&self) -> usize {
        self.weight.len()
    }

    /// `expected_output_samples` from the real Python:
    /// `(n_frames - 1) * stride + kernel`.
    pub fn expected_output_samples(n_frames: usize) -> usize {
        (n_frames - 1) * ENC_STRIDE + ENC_KERNEL
    }

    /// `feat`: (n_frames * ENC_CHANNELS,) row-major (n_frames, ENC_CHANNELS)
    /// - the masked/separated feature map, same (T, N) layout `encode`
    /// returns. Returns raw waveform samples. This is the real
    /// ConvTranspose1d "scatter-add" formulation: each input frame `t`
    /// contributes to output samples `[t*stride, t*stride+kernel)`,
    /// weighted by the same per-channel kernel taps as the encoder (but
    /// summed over channels here instead of over input samples).
    pub fn decode(&self, feat: &[f32], n_frames: usize) -> Vec<f32> {
        let out_len = Self::expected_output_samples(n_frames);
        let mut out = vec![0.0f32; out_len];
        for t in 0..n_frames {
            let base = t * ENC_STRIDE;
            for c in 0..ENC_CHANNELS {
                let v = feat[t * ENC_CHANNELS + c];
                if v == 0.0 {
                    continue;
                }
                for k in 0..ENC_KERNEL {
                    out[base + k] += v * self.weight[c * ENC_KERNEL + k];
                }
            }
        }
        out
    }
}

#[cfg(test)]
mod encoder_decoder_tests {
    use super::*;

    #[test]
    fn n_frames_and_expected_samples_match_real_python_formulas() {
        // 10ms @16kHz = 160 samples -> (160-16)/8+1 = 19 frames (NOT 20 -
        // the 20-per-chunk constant elsewhere is a rounding convenience
        // for the separator benchmark, not this exact formula).
        assert_eq!(AudioEncoderWeights::n_frames_for(160), 19);
        assert_eq!(AudioDecoderWeights::expected_output_samples(19), (19 - 1) * 8 + 16);
    }

    #[test]
    fn encode_shape_matches_n_frames_times_channels() {
        let mut rng = rand::thread_rng();
        let enc = AudioEncoderWeights::random(&mut rng);
        let samples = rand_vec(1600, &mut rng); // 100ms
        let (feat, n_frames) = enc.encode(&samples);
        assert_eq!(n_frames, AudioEncoderWeights::n_frames_for(1600));
        assert_eq!(feat.len(), n_frames * ENC_CHANNELS);
        assert!(feat.iter().all(|v| *v >= 0.0), "ReLU must clamp all encoder output non-negative");
    }

    #[test]
    fn decode_shape_matches_expected_output_samples() {
        let mut rng = rand::thread_rng();
        let dec = AudioDecoderWeights::random(&mut rng);
        let n_frames = 19;
        let feat = rand_vec(n_frames * ENC_CHANNELS, &mut rng);
        let wav = dec.decode(&feat, n_frames);
        assert_eq!(wav.len(), AudioDecoderWeights::expected_output_samples(n_frames));
    }

    #[test]
    fn single_tap_single_channel_encode_matches_hand_computed_dot_product() {
        // Direct arithmetic check (not just shape): a single frame, all
        // channels zeroed except one, weight = known values -> output must
        // equal the exact dot product for that one channel, matching
        // Conv1d's definition (no bias, no padding).
        let mut weight = vec![0.0f32; ENC_CHANNELS * ENC_KERNEL];
        for k in 0..ENC_KERNEL {
            weight[0 * ENC_KERNEL + k] = (k + 1) as f32; // channel 0: taps 1..16
        }
        let enc = AudioEncoderWeights { weight };
        let samples: Vec<f32> = (0..ENC_KERNEL as i32).map(|i| i as f32).collect(); // exactly one frame's worth
        let (feat, n_frames) = enc.encode(&samples);
        assert_eq!(n_frames, 1);
        let expected: f32 = (0..ENC_KERNEL).map(|k| (k as f32) * (k as f32 + 1.0)).sum();
        assert!((feat[0] - expected.max(0.0)).abs() < 1e-4, "got {} expected {}", feat[0], expected);
        // Every other channel's weight is all-zero -> output must be exactly 0.
        assert!(feat[1..].iter().all(|v| *v == 0.0));
    }

    #[test]
    fn decode_scatter_add_matches_hand_computed_overlap() {
        // Two frames (stride=8, kernel=16) overlap in samples [8,16) - a
        // real ConvTranspose1d must SUM both frames' contributions there,
        // not overwrite. Use a weight of all-ones so each frame's
        // contribution to any output sample equals the sum of that
        // frame's active-channel values.
        let mut weight = vec![0.0f32; ENC_CHANNELS * ENC_KERNEL];
        for c in 0..2 {
            for k in 0..ENC_KERNEL {
                weight[c * ENC_KERNEL + k] = 1.0;
            }
        }
        let dec = AudioDecoderWeights { weight };
        let mut feat = vec![0.0f32; 2 * ENC_CHANNELS];
        feat[0] = 1.0; // frame 0, channel 0
        feat[1] = 2.0; // frame 0, channel 1
        feat[ENC_CHANNELS] = 3.0; // frame 1, channel 0
        feat[ENC_CHANNELS + 1] = 4.0; // frame 1, channel 1
        let wav = dec.decode(&feat, 2);
        // frame0 contributes (1+2)=3 to samples [0,16); frame1 contributes
        // (3+4)=7 to samples [8,24). Overlap region [8,16) must be 3+7=10.
        assert_eq!(wav.len(), 24);
        for i in 0..8 {
            assert!((wav[i] - 3.0).abs() < 1e-5, "sample {i}: {}", wav[i]);
        }
        for i in 8..16 {
            assert!((wav[i] - 10.0).abs() < 1e-5, "sample {i}: {}", wav[i]);
        }
        for i in 16..24 {
            assert!((wav[i] - 7.0).abs() < 1e-5, "sample {i}: {}", wav[i]);
        }
    }
}
