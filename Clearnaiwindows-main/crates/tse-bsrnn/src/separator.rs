//! Hand-rolled real BSRNN separator: band-split, sub-band normalization,
//! 6 causal repeat blocks (`band_rnn` time-recurrent per band, `band_comm`
//! stateless-per-timestep over the band axis), and the complex-mask head -
//! a direct, careful port of the real, verified module structure in
//! `wesep/modules/separator/bsrnn.py` (see that file's `BandSplit`,
//! `SubbandNorm`, `ResRNN`, `BSNet`, `BandMasker` classes), using the raw
//! weights `weights.rs` loads from the hash-verified checkpoint.
//!
//! Hardcodes `nspk=1` throughout (matches this checkpoint's real
//! `config.yaml`: `separator.nspk: 1`) rather than generalizing for a case
//! that doesn't exist in the real model this crate embeds.

use crate::lstm::{lstm_step, LstmWeights};
use crate::ops::{group_norm_1_inplace, layer_norm_inplace, linear_row};
use crate::weights::{WeightError, WeightStore};

thread_local! {
    static MAX_PARALLELISM_HINT: std::cell::Cell<Option<usize>> = const { std::cell::Cell::new(None) };
}

/// Caps how many worker threads `run_band_comm`/`run_band_masker` will
/// spawn, for calls made ON THE CALLING THREAD from here on (a thread-local
/// hint, not global - the live realtime pipeline's own worker thread is
/// unaffected unless it calls this itself). `None` (the default) means "use
/// every core", matching the original behavior.
///
/// Exists because raw `std::thread::available_parallelism()` has no way to
/// know a totally unrelated realtime workload (the live mic pipeline) might
/// be running concurrently on the same machine. Real finding: the GUI's
/// offline "Test with TSE" button, uncapped, saturated every logical core
/// for up to a minute and measurably degraded live mic audio (added
/// latency, frame drops) purely through CPU contention - Windows thread
/// priority does not fix this on its own, since `std::thread::scope`'s
/// spawned child threads do not inherit the parent thread's OS priority.
/// Call this (with a cap well below the core count) at the top of any
/// CPU-heavy, non-realtime background run that might overlap with live
/// realtime work elsewhere in the process.
pub fn set_max_parallelism_hint(n: Option<usize>) {
    MAX_PARALLELISM_HINT.with(|c| c.set(n));
}

fn effective_parallelism() -> usize {
    let avail = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    MAX_PARALLELISM_HINT.with(|c| c.get()).map(|cap| avail.min(cap.max(1))).unwrap_or(avail)
}

const FEATURE_DIM: usize = 128;
const HIDDEN: usize = 256; // feature_dim * 2, per ResRNN(feature_dim, feature_dim*2, ...)
const MASK_HIDDEN: usize = 512; // feature_dim * 4, per BandMasker's Conv1d(128,512,1)

#[derive(Clone)]
struct SubbandNormLayer {
    norm_weight: &'static [f32], // (BW*2,)
    norm_bias: &'static [f32],
    norm_eps: f32,
    conv_weight: &'static [f32], // (feature_dim, BW*2)
    conv_bias: &'static [f32],   // (feature_dim,)
    band_width: usize,
}

#[derive(Clone)]
struct ResRnnWeights {
    norm_weight: &'static [f32], // (feature_dim,)
    norm_bias: &'static [f32],
    norm_eps: f32,
    bidirectional: bool,
    lstm_fwd_ih: &'static [f32],
    lstm_fwd_hh: &'static [f32],
    lstm_fwd_bias_ih: &'static [f32],
    lstm_fwd_bias_hh: &'static [f32],
    lstm_bwd_ih: Option<&'static [f32]>,
    lstm_bwd_hh: Option<&'static [f32]>,
    lstm_bwd_bias_ih: Option<&'static [f32]>,
    lstm_bwd_bias_hh: Option<&'static [f32]>,
    proj_weight: &'static [f32], // (feature_dim, hidden*(1+bidirectional))
    proj_bias: &'static [f32],
    /// `[lstm_fwd_ih | lstm_fwd_hh]` concatenated row-wise into one
    /// `(4*hidden, feature_dim+hidden)` matrix, computed once at load time
    /// (see `resrnn_weights`) - lets `run_band_rnn` issue ONE batched
    /// matmul per timestep instead of two (input-contribution and hidden-
    /// contribution separately), halving per-call GEMM dispatch overhead
    /// for the same total FLOPs. Only used by `run_band_rnn` (band_comm's
    /// per-band-position `lstm_step` doesn't batch this way - see that
    /// function's own docs).
    combined_fwd_ih_hh: &'static [f32],
}

#[derive(Clone)]
struct BandMaskerLayer {
    norm_weight: &'static [f32],
    norm_bias: &'static [f32],
    norm_eps: f32,
    conv1_weight: &'static [f32], // (512, 128)
    conv1_bias: &'static [f32],
    conv2_weight: &'static [f32], // (512, 512)
    conv2_bias: &'static [f32],
    conv3_weight: &'static [f32], // (BW*4, 512)
    conv3_bias: &'static [f32],
    band_width: usize,
}

#[derive(Clone)]
struct RepeatBlockWeights {
    band_rnn: ResRnnWeights,
    band_comm: ResRnnWeights,
}

/// Persisted, real streaming state: ONLY `band_rnn` is time-recurrent (one
/// `(h,c)` pair per band, per repeat block - 6 repeat blocks x 32 bands
/// here); `band_comm` is stateless across time by construction (it only
/// ever recurs over the fixed 32-band axis, fresh every timestep) - see
/// this module's top doc and the real, already-verified Python streaming
/// engine's equivalent finding.
pub struct RepeatBlockState {
    band_rnn_h: Vec<f32>, // (nband, hidden)
    band_rnn_c: Vec<f32>, // (nband, hidden)
}

/// `Clone` is cheap: every heavy field is a `&'static` reference into the
/// weight blob already parsed once at `load()` time (see `weights.rs`'s
/// docs on why that's leaked, not re-parsed) - cloning only copies small
/// `Vec<usize>`/struct-of-references metadata, never the underlying
/// weight data. Production code (`RealBsrnnTseBackend`) relies on this to
/// build a fresh `StreamingTseEngine` on re-enrollment without re-leaking
/// another ~75MB weight blob each time.
#[derive(Clone)]
pub struct BsrnnSeparator {
    nband: usize,
    band_width: Vec<usize>,
    band_offset: Vec<usize>, // cumulative sum, for indexing into the full F=257 spectrum
    subband_norm: Vec<SubbandNormLayer>,
    repeat_blocks: Vec<RepeatBlockWeights>,
    band_masker: Vec<BandMaskerLayer>,
    win: usize,
    stride: usize,
}

fn resrnn_weights(w: &WeightStore, prefix: &str) -> Result<ResRnnWeights, WeightError> {
    let bidirectional = w.get_i64_vec(&format!("{prefix}.bidirectional"))?[0] != 0;
    let lstm_fwd_ih = w.get_f32(&format!("{prefix}.lstm.weight_ih_fwd"))?;
    let lstm_fwd_hh = w.get_f32(&format!("{prefix}.lstm.weight_hh_fwd"))?;

    // Concatenate row-wise: row g = [lstm_fwd_ih[g] (FEATURE_DIM vals),
    // lstm_fwd_hh[g] (HIDDEN vals)] - computed once here (weights load
    // once at startup), not per-timestep - see the field's doc.
    let in_dim = lstm_fwd_ih.len() / (4 * HIDDEN);
    let hid_dim = lstm_fwd_hh.len() / (4 * HIDDEN);
    let mut combined = Vec::with_capacity(4 * HIDDEN * (in_dim + hid_dim));
    for g in 0..4 * HIDDEN {
        combined.extend_from_slice(&lstm_fwd_ih[g * in_dim..(g + 1) * in_dim]);
        combined.extend_from_slice(&lstm_fwd_hh[g * hid_dim..(g + 1) * hid_dim]);
    }
    let combined_fwd_ih_hh: &'static [f32] = Box::leak(combined.into_boxed_slice());

    Ok(ResRnnWeights {
        norm_weight: w.get_f32(&format!("{prefix}.norm.weight"))?,
        norm_bias: w.get_f32(&format!("{prefix}.norm.bias"))?,
        norm_eps: w.get_f32(&format!("{prefix}.norm.eps"))?[0],
        bidirectional,
        lstm_fwd_ih,
        lstm_fwd_hh,
        lstm_fwd_bias_ih: w.get_f32(&format!("{prefix}.lstm.bias_ih_fwd"))?,
        lstm_fwd_bias_hh: w.get_f32(&format!("{prefix}.lstm.bias_hh_fwd"))?,
        lstm_bwd_ih: if bidirectional { Some(w.get_f32(&format!("{prefix}.lstm.weight_ih_bwd"))?) } else { None },
        lstm_bwd_hh: if bidirectional { Some(w.get_f32(&format!("{prefix}.lstm.weight_hh_bwd"))?) } else { None },
        lstm_bwd_bias_ih: if bidirectional { Some(w.get_f32(&format!("{prefix}.lstm.bias_ih_bwd"))?) } else { None },
        lstm_bwd_bias_hh: if bidirectional { Some(w.get_f32(&format!("{prefix}.lstm.bias_hh_bwd"))?) } else { None },
        proj_weight: w.get_f32(&format!("{prefix}.proj.weight"))?,
        proj_bias: w.get_f32(&format!("{prefix}.proj.bias"))?,
        combined_fwd_ih_hh,
    })
}

impl BsrnnSeparator {
    pub fn load() -> Result<Self, WeightError> {
        let w = WeightStore::load()?;
        let nband = w.get_i64_vec("meta_nband")?[0] as usize;
        let band_width: Vec<usize> = w.get_i64_vec("meta_band_width")?.into_iter().map(|v| v as usize).collect();
        let num_repeat = w.get_i64_vec("meta_num_repeat")?[0] as usize;
        let win = w.get_i64_vec("meta_win")?[0] as usize;
        let stride = w.get_i64_vec("meta_stride")?[0] as usize;
        assert_eq!(band_width.len(), nband);

        let mut band_offset = Vec::with_capacity(nband);
        let mut acc = 0usize;
        for &bw in &band_width {
            band_offset.push(acc);
            acc += bw;
        }

        let mut subband_norm = Vec::with_capacity(nband);
        for i in 0..nband {
            let bw = band_width[i];
            subband_norm.push(SubbandNormLayer {
                norm_weight: w.get_f32(&format!("subband_norm.{i}.norm.weight"))?,
                norm_bias: w.get_f32(&format!("subband_norm.{i}.norm.bias"))?,
                norm_eps: w.get_f32(&format!("subband_norm.{i}.norm.eps"))?[0],
                conv_weight: w.get_f32(&format!("subband_norm.{i}.conv.weight"))?,
                conv_bias: w.get_f32(&format!("subband_norm.{i}.conv.bias"))?,
                band_width: bw,
            });
        }

        let mut repeat_blocks = Vec::with_capacity(num_repeat);
        for r in 0..num_repeat {
            repeat_blocks.push(RepeatBlockWeights {
                band_rnn: resrnn_weights(&w, &format!("separator.{r}.band_rnn"))?,
                band_comm: resrnn_weights(&w, &format!("separator.{r}.band_comm"))?,
            });
        }

        let mut band_masker = Vec::with_capacity(nband);
        for i in 0..nband {
            let bw = band_width[i];
            band_masker.push(BandMaskerLayer {
                norm_weight: w.get_f32(&format!("band_masker.{i}.norm.weight"))?,
                norm_bias: w.get_f32(&format!("band_masker.{i}.norm.bias"))?,
                norm_eps: w.get_f32(&format!("band_masker.{i}.norm.eps"))?[0],
                conv1_weight: w.get_f32(&format!("band_masker.{i}.conv1.weight"))?,
                conv1_bias: w.get_f32(&format!("band_masker.{i}.conv1.bias"))?,
                conv2_weight: w.get_f32(&format!("band_masker.{i}.conv2.weight"))?,
                conv2_bias: w.get_f32(&format!("band_masker.{i}.conv2.bias"))?,
                conv3_weight: w.get_f32(&format!("band_masker.{i}.conv3.weight"))?,
                conv3_bias: w.get_f32(&format!("band_masker.{i}.conv3.bias"))?,
                band_width: bw,
            });
        }

        Ok(Self { nband, band_width, band_offset, subband_norm, repeat_blocks, band_masker, win, stride })
    }

    pub fn nband(&self) -> usize {
        self.nband
    }

    pub fn win(&self) -> usize {
        self.win
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn band_width(&self) -> &[usize] {
        &self.band_width
    }

    pub fn num_repeat(&self) -> usize {
        self.repeat_blocks.len()
    }

    pub fn new_state(&self) -> Vec<RepeatBlockState> {
        (0..self.repeat_blocks.len())
            .map(|_| RepeatBlockState { band_rnn_h: vec![0.0; self.nband * HIDDEN], band_rnn_c: vec![0.0; self.nband * HIDDEN] })
            .collect()
    }

    pub fn reset_state(&self, state: &mut [RepeatBlockState]) {
        for s in state {
            s.band_rnn_h.iter_mut().for_each(|v| *v = 0.0);
            s.band_rnn_c.iter_mut().for_each(|v| *v = 0.0);
        }
    }

    /// spec_real/spec_imag: (F=257, T) row-major (band axis implicit via
    /// band_offset/band_width). Returns subband_feature: (nband, feature_dim, T).
    pub fn band_split_and_subband_norm(&self, spec_real: &[f32], spec_imag: &[f32], t_frames: usize) -> Vec<f32> {
        let f_total: usize = self.band_width.iter().sum();
        debug_assert_eq!(spec_real.len(), f_total * t_frames);
        debug_assert_eq!(spec_imag.len(), f_total * t_frames);

        let mut out = vec![0.0f32; self.nband * FEATURE_DIM * t_frames];
        let mut scratch_in = vec![0.0f32; 0];
        for (band_idx, layer) in self.subband_norm.iter().enumerate() {
            let bw = layer.band_width;
            let off = self.band_offset[band_idx];
            scratch_in.resize(bw * 2, 0.0);
            for t in 0..t_frames {
                // Build (2, BW) real/imag block for this band/timestep, in
                // the same [real(BW); imag(BW)] channel order BandSplit's
                // (B,2,F,T) layout implies (dim=1 is the real/imag axis,
                // BEFORE the per-band Conv1d flattens it with BW via
                // `.view(B, BW*spec_dim, -1)` - real/imag stays outermost).
                for w in 0..bw {
                    scratch_in[w] = spec_real[(off + w) * t_frames + t];
                    scratch_in[bw + w] = spec_imag[(off + w) * t_frames + t];
                }
                layer_norm_inplace(&mut scratch_in, layer.norm_weight, layer.norm_bias, layer.norm_eps, bw * 2);
                let out_slice = &mut out[(band_idx * FEATURE_DIM * t_frames + t)..];
                // write feature_dim values strided by t_frames (since out
                // layout is (nband, feature_dim, T) row-major)
                let mut tmp = [0.0f32; FEATURE_DIM];
                linear_row(&scratch_in, layer.conv_weight, Some(layer.conv_bias), bw * 2, FEATURE_DIM, &mut tmp);
                for (f, val) in tmp.iter().enumerate() {
                    out_slice[f * t_frames] = *val;
                }
            }
        }
        out
    }

    /// In-place: `subband_feature` (nband, feature_dim, T) *= broadcast
    /// per-band-and-timestep gate (feature_dim,) - the real speaker-
    /// conditioning fusion (`SpeakerFuseLayer`, fuse_type="multiply"),
    /// computed once per enrollment from the cached 192-dim embedding via
    /// a separate `Linear(192,128)` (see `speaker_fusion.rs`), NOT
    /// recomputed here - this function only applies the (already
    /// projected) gate vector.
    pub fn apply_speaker_gate(&self, subband_feature: &mut [f32], gate: &[f32; FEATURE_DIM], t_frames: usize) {
        for band in 0..self.nband {
            for f in 0..FEATURE_DIM {
                let base = (band * FEATURE_DIM + f) * t_frames;
                for t in 0..t_frames {
                    subband_feature[base + t] *= gate[f];
                }
            }
        }
    }

    /// Runs all repeat blocks (band_rnn time-recurrent using `state`,
    /// band_comm fresh per timestep) over `subband_feature` (nband,
    /// feature_dim, T), returning the separator's output in the same shape.
    pub fn run_separator(&self, subband_feature: &[f32], state: &mut [RepeatBlockState], t_frames: usize) -> Vec<f32> {
        let mut x = subband_feature.to_vec();
        for (r, block) in self.repeat_blocks.iter().enumerate() {
            x = self.run_band_rnn(&block.band_rnn, &x, &mut state[r], t_frames);
            x = self.run_band_comm(&block.band_comm, &x, t_frames);
        }
        x
    }

    /// band_rnn: per band, time-recurrent over T using persisted state.
    /// input/output: (nband, feature_dim, T). `pub` (not just used
    /// internally) so tests can isolate this step from band_comm when
    /// debugging a divergence against the PyTorch reference.
    pub fn run_band_rnn_for_block(
        &self,
        block_idx: usize,
        input: &[f32],
        state: &mut RepeatBlockState,
        t_frames: usize,
    ) -> Vec<f32> {
        self.run_band_rnn(&self.repeat_blocks[block_idx].band_rnn, input, state, t_frames)
    }

    /// Same as above but band_comm - see `run_band_rnn_for_block`'s docs.
    pub fn run_band_comm_for_block(&self, block_idx: usize, input: &[f32], t_frames: usize) -> Vec<f32> {
        self.run_band_comm(&self.repeat_blocks[block_idx].band_comm, input, t_frames)
    }

    /// PERFORMANCE: band_rnn's 32 bands are mathematically independent per
    /// timestep (each band's own history through time is independent of
    /// other bands) but share the SAME weights - the real model itself
    /// computes this as one batched-over-bands LSTM call
    /// (`input.view(B*nband, feature_dim, -1)` in `bsrnn.py`'s
    /// `BSNet.forward`), not 32 separate ones. An earlier version of this
    /// function computed each band's LSTM step with a scalar per-band loop
    /// (`lstm::lstm_step`), which was correct (verified to ~1e-6 against
    /// the real reference) but measured at RTF~31 on a real 12s mixture -
    /// this batches the SAME real math across all bands via
    /// `matmul::batched_linear` (see that module's docs for the real
    /// op-count reasoning behind this fix), computed once per TIMESTEP
    /// (loop order is now time-outer/band-batched-inner, the reverse of
    /// the original band-outer/time-inner order - mathematically
    /// equivalent, since each band's own state is still threaded
    /// correctly across timesteps within this call and across calls via
    /// `RepeatBlockState`).
    fn run_band_rnn(&self, w: &ResRnnWeights, input: &[f32], state: &mut RepeatBlockState, t_frames: usize) -> Vec<f32> {
        debug_assert!(!w.bidirectional, "band_rnn must be causal/unidirectional in this checkpoint");
        let nband = self.nband;
        let mut out = vec![0.0f32; nband * FEATURE_DIM * t_frames];

        // bias_ih + bias_hh combined once (PyTorch adds both separately,
        // but summing them ahead of time and applying the sum only on the
        // combined matmul is equivalent and avoids double-adding).
        let combined_bias: Vec<f32> =
            w.lstm_fwd_bias_ih.iter().zip(w.lstm_fwd_bias_hh.iter()).map(|(a, b)| a + b).collect();

        let mut x_t = vec![0.0f32; nband * FEATURE_DIM];
        let mut residual_t = vec![0.0f32; nband * FEATURE_DIM];
        // [x_t | state.band_rnn_h] concatenated per band, feeding
        // `combined_fwd_ih_hh` in ONE matmul instead of two - see that
        // field's doc for why.
        let mut xh_t = vec![0.0f32; nband * (FEATURE_DIM + HIDDEN)];

        for t in 0..t_frames {
            for band in 0..nband {
                let slice = &mut x_t[band * FEATURE_DIM..(band + 1) * FEATURE_DIM];
                for f in 0..FEATURE_DIM {
                    let v = input[(band * FEATURE_DIM + f) * t_frames + t];
                    slice[f] = v;
                    residual_t[band * FEATURE_DIM + f] = v;
                }
                layer_norm_inplace(slice, w.norm_weight, w.norm_bias, w.norm_eps, FEATURE_DIM);

                let xh_base = band * (FEATURE_DIM + HIDDEN);
                xh_t[xh_base..xh_base + FEATURE_DIM].copy_from_slice(slice);
                xh_t[xh_base + FEATURE_DIM..xh_base + FEATURE_DIM + HIDDEN]
                    .copy_from_slice(&state.band_rnn_h[band * HIDDEN..(band + 1) * HIDDEN]);
            }

            // (nband, 4*HIDDEN) gate pre-activations, ONE batched matmul
            // across all bands AND both the input- and hidden-contribution
            // (see `combined_fwd_ih_hh`'s doc) instead of 32 separate
            // scalar LSTM steps or two separate batched matmuls.
            let gates_all = crate::matmul::batched_linear(
                &xh_t,
                nband,
                FEATURE_DIM + HIDDEN,
                w.combined_fwd_ih_hh,
                Some(&combined_bias),
                4 * HIDDEN,
            );

            let mut h_new_all = vec![0.0f32; nband * HIDDEN];
            let mut c_new_all = vec![0.0f32; nband * HIDDEN];
            for band in 0..nband {
                let base = band * 4 * HIDDEN;
                for j in 0..HIDDEN {
                    let g_i = crate::ops::sigmoid(gates_all[base + j]);
                    let g_f = crate::ops::sigmoid(gates_all[base + HIDDEN + j]);
                    let g_g = gates_all[base + 2 * HIDDEN + j].tanh();
                    let g_o = crate::ops::sigmoid(gates_all[base + 3 * HIDDEN + j]);
                    let c_prev = state.band_rnn_c[band * HIDDEN + j];
                    let c = g_f * c_prev + g_i * g_g;
                    c_new_all[band * HIDDEN + j] = c;
                    h_new_all[band * HIDDEN + j] = g_o * c.tanh();
                }
            }
            state.band_rnn_h.copy_from_slice(&h_new_all);
            state.band_rnn_c.copy_from_slice(&c_new_all);

            let proj_out_all = crate::matmul::batched_linear(&h_new_all, nband, HIDDEN, w.proj_weight, Some(w.proj_bias), FEATURE_DIM);
            for band in 0..nband {
                for f in 0..FEATURE_DIM {
                    out[(band * FEATURE_DIM + f) * t_frames + t] = residual_t[band * FEATURE_DIM + f] + proj_out_all[band * FEATURE_DIM + f];
                }
            }
        }
        out
    }

    /// band_comm: per timestep, bidirectional over the band axis, fresh
    /// (zero) state every call - no time-recurrence, matches the real
    /// model's own stateless-per-timestep behavior (see this module's top
    /// doc). input/output: (nband, feature_dim, T).
    ///
    /// PERFORMANCE: unlike band_rnn, band_comm's 32 steps within one
    /// direction are a TRUE sequential recurrence over the band axis (each
    /// position's hidden state depends on the previous position's), so
    /// they cannot be batched into one matmul the way band_rnn's
    /// independent-per-band steps can - unavoidable. The final `proj` step
    /// (applied independently per band, after both directions' full
    /// per-position hidden states are already known) is NOT sequential,
    /// though, and IS batched across all 32 bands as one matmul below,
    /// same reasoning as band_rnn's proj.
    ///
    /// PERFORMANCE: `band_comm` has NO recurrence across timesteps at all
    /// (see this function's top doc) - only *within* one direction's 32
    /// sequential band-positions. That means every (timestep, direction)
    /// pair is fully independent of every other one, so instead of
    /// spawning exactly 2 threads per timestep (the earlier fix - real,
    /// but capped at 2x no matter how many cores exist), this spreads ALL
    /// `t_frames * 2` (timestep, direction) work items across
    /// `std::thread::available_parallelism()` workers in ONE
    /// `thread::scope` call for the whole chunk. This only pays off when
    /// `t_frames` is large enough to give every core real work - at
    /// `t_frames=1` (the real streaming call pattern today) this is
    /// mathematically identical to the old 2-thread split; the actual won
    /// speedup requires the *caller* to buffer more audio per call (a
    /// latency-for-throughput tradeoff outside this function's scope -
    /// see `tests/profile_breakdown.rs` for real measurements at both
    /// `t_frames=1` and larger simulated chunk sizes).
    fn run_band_comm(&self, w: &ResRnnWeights, input: &[f32], t_frames: usize) -> Vec<f32> {
        debug_assert!(w.bidirectional, "band_comm must be bidirectional in this checkpoint");
        let nband = self.nband;
        let mut out = vec![0.0f32; nband * FEATURE_DIM * t_frames];
        let fwd_w = LstmWeights {
            weight_ih: w.lstm_fwd_ih,
            weight_hh: w.lstm_fwd_hh,
            bias_ih: w.lstm_fwd_bias_ih,
            bias_hh: w.lstm_fwd_bias_hh,
        };
        let bwd_w = LstmWeights {
            weight_ih: w.lstm_bwd_ih.unwrap(),
            weight_hh: w.lstm_bwd_hh.unwrap(),
            bias_ih: w.lstm_bwd_bias_ih.unwrap(),
            bias_hh: w.lstm_bwd_bias_hh.unwrap(),
        };

        // Step 1: compute (and norm) all T timesteps' band vectors upfront
        // - each timestep's GroupNorm is over that timestep's own (nband,
        // feature_dim) block only (see group_norm_1_inplace's docs), so
        // this loop has no cross-timestep dependency either.
        let mut normed_all = vec![0.0f32; t_frames * nband * FEATURE_DIM];
        let mut residual_all = vec![0.0f32; t_frames * nband * FEATURE_DIM];
        for t in 0..t_frames {
            let base = t * nband * FEATURE_DIM;
            for band in 0..nband {
                for f in 0..FEATURE_DIM {
                    let v = input[(band * FEATURE_DIM + f) * t_frames + t];
                    normed_all[base + band * FEATURE_DIM + f] = v;
                    residual_all[base + band * FEATURE_DIM + f] = v;
                }
            }
            group_norm_1_inplace(&mut normed_all[base..base + nband * FEATURE_DIM], w.norm_weight, w.norm_bias, w.norm_eps, nband, FEATURE_DIM);
        }

        // Step 2: run every (t, direction) LSTM sequence, spread across
        // all available cores instead of exactly 2 threads.
        let mut fwd_all = vec![0.0f32; t_frames * nband * HIDDEN];
        let mut bwd_all = vec![0.0f32; t_frames * nband * HIDDEN];
        let num_workers = effective_parallelism().min(2 * t_frames).max(1);

        std::thread::scope(|scope| {
            let normed_ref = &normed_all;
            let fwd_item_chunks: Vec<&mut [f32]> = fwd_all.chunks_mut(nband * HIDDEN).collect();
            let bwd_item_chunks: Vec<&mut [f32]> = bwd_all.chunks_mut(nband * HIDDEN).collect();

            // Round-robin-assign each (t, direction) work item (identified
            // by (t, is_bwd, its output slice)) to one of `num_workers`
            // groups, so thread-dispatch overhead is paid once per group,
            // not once per work item.
            let mut groups: Vec<Vec<(usize, bool, &mut [f32])>> = (0..num_workers).map(|_| Vec::new()).collect();
            for (t, slice) in fwd_item_chunks.into_iter().enumerate() {
                groups[t % num_workers].push((t, false, slice));
            }
            for (t, slice) in bwd_item_chunks.into_iter().enumerate() {
                groups[(t_frames + t) % num_workers].push((t, true, slice));
            }

            for group in groups {
                if group.is_empty() {
                    continue;
                }
                scope.spawn(move || {
                    let mut h = [0.0f32; HIDDEN];
                    let mut c = [0.0f32; HIDDEN];
                    let mut h_new = [0.0f32; HIDDEN];
                    let mut c_new = [0.0f32; HIDDEN];
                    let mut scratch = [0.0f32; 4 * HIDDEN];
                    for (t, is_bwd, out_slice) in group {
                        let base = t * nband * FEATURE_DIM;
                        let seq = &normed_ref[base..base + nband * FEATURE_DIM];
                        let w_local = if is_bwd { &bwd_w } else { &fwd_w };
                        h.iter_mut().for_each(|v| *v = 0.0);
                        c.iter_mut().for_each(|v| *v = 0.0);
                        let order: Vec<usize> = if is_bwd { (0..nband).rev().collect() } else { (0..nband).collect() };
                        for band in order {
                            let x = &seq[band * FEATURE_DIM..(band + 1) * FEATURE_DIM];
                            lstm_step(x, &h, &c, w_local, FEATURE_DIM, HIDDEN, &mut h_new, &mut c_new, &mut scratch);
                            h.copy_from_slice(&h_new);
                            c.copy_from_slice(&c_new);
                            out_slice[band * HIDDEN..(band + 1) * HIDDEN].copy_from_slice(&h_new);
                        }
                    }
                });
            }
        });

        // Step 3: proj step, batched across ALL T*nband rows at once (not
        // just per-timestep) - same reasoning as band_rnn's/band_comm's
        // existing proj batching, just extended to the T axis too.
        let mut proj_in_all = vec![0.0f32; t_frames * nband * 2 * HIDDEN];
        for t in 0..t_frames {
            for band in 0..nband {
                let row = t * nband + band;
                proj_in_all[row * 2 * HIDDEN..row * 2 * HIDDEN + HIDDEN].copy_from_slice(&fwd_all[row * HIDDEN..(row + 1) * HIDDEN]);
                proj_in_all[row * 2 * HIDDEN + HIDDEN..(row + 1) * 2 * HIDDEN]
                    .copy_from_slice(&bwd_all[row * HIDDEN..(row + 1) * HIDDEN]);
            }
        }
        let proj_out_all = crate::matmul::batched_linear(
            &proj_in_all,
            t_frames * nband,
            2 * HIDDEN,
            w.proj_weight,
            Some(w.proj_bias),
            FEATURE_DIM,
        );
        for t in 0..t_frames {
            for band in 0..nband {
                let row = t * nband + band;
                for f in 0..FEATURE_DIM {
                    out[(band * FEATURE_DIM + f) * t_frames + t] =
                        residual_all[row * FEATURE_DIM + f] + proj_out_all[row * FEATURE_DIM + f];
                }
            }
        }
        out
    }

    /// BandMasker + complex-mask application. `sep_output`: (nband,
    /// feature_dim, T). `mix_real`/`mix_imag`: (F=257, T). Returns
    /// (est_real, est_imag), each (F=257, T) - matches
    /// `wesep/modules/separator/bsrnn.py::BandMasker.forward`'s exact
    /// reshape/gate derivation (nspk=1 hardcoded - see module docs).
    /// PERFORMANCE: every band's masker head has its OWN weights (unlike
    /// `band_rnn`/`band_comm`, which share one set of weights across all
    /// 32 bands), so bands can't be batched into one matmul the way
    /// `run_band_rnn` batches its per-band steps. But unlike `band_comm`,
    /// bands here have ZERO dependency on each other (no recurrence at
    /// all, over either time or the band axis) - each band's output only
    /// depends on `sep_output` for that band. Split the 32 bands across
    /// `std::thread::available_parallelism()` worker threads (contiguous
    /// band-index ranges, which map to contiguous, disjoint (F,T) ranges
    /// of `est_real`/`est_imag` since `band_offset` accumulates
    /// `band_width` in increasing band-index order - no unsafe aliasing).
    pub fn run_band_masker(
        &self,
        sep_output: &[f32],
        mix_real: &[f32],
        mix_imag: &[f32],
        t_frames: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        let nband = self.nband;
        let f_total: usize = self.band_width.iter().sum();
        let mut est_real = vec![0.0f32; f_total * t_frames];
        let mut est_imag = vec![0.0f32; f_total * t_frames];

        let num_workers = effective_parallelism().min(nband).max(1);
        let mut group_bounds = Vec::with_capacity(num_workers + 1);
        group_bounds.push(0usize);
        for g in 1..num_workers {
            group_bounds.push((nband * g) / num_workers);
        }
        group_bounds.push(nband);

        std::thread::scope(|scope| {
            let mut real_rest: &mut [f32] = &mut est_real;
            let mut imag_rest: &mut [f32] = &mut est_imag;
            let mut prev_f_end = 0usize;
            for g in 0..num_workers {
                let band_start = group_bounds[g];
                let band_end = group_bounds[g + 1];
                let f_start = prev_f_end;
                let f_end =
                    if band_end == 0 { 0 } else { self.band_offset[band_end - 1] + self.band_width[band_end - 1] };
                let len = (f_end - f_start) * t_frames;
                let (real_slice, real_tail) = real_rest.split_at_mut(len);
                let (imag_slice, imag_tail) = imag_rest.split_at_mut(len);
                real_rest = real_tail;
                imag_rest = imag_tail;
                prev_f_end = f_end;

                scope.spawn(move || {
                    let mut feat = [0.0f32; FEATURE_DIM];
                    let mut h1 = [0.0f32; MASK_HIDDEN];
                    let mut h2 = [0.0f32; MASK_HIDDEN];
                    for band_idx in band_start..band_end {
                        let layer = &self.band_masker[band_idx];
                        let bw = layer.band_width;
                        let off = self.band_offset[band_idx];
                        let local_off = off - f_start;
                        let mut h3 = vec![0.0f32; bw * 4];
                        for t in 0..t_frames {
                            for f in 0..FEATURE_DIM {
                                feat[f] = sep_output[(band_idx * FEATURE_DIM + f) * t_frames + t];
                            }
                            layer_norm_inplace(&mut feat, layer.norm_weight, layer.norm_bias, layer.norm_eps, FEATURE_DIM);
                            linear_row(&feat, layer.conv1_weight, Some(layer.conv1_bias), FEATURE_DIM, MASK_HIDDEN, &mut h1);
                            h1.iter_mut().for_each(|v| *v = v.tanh());
                            linear_row(&h1, layer.conv2_weight, Some(layer.conv2_bias), MASK_HIDDEN, MASK_HIDDEN, &mut h2);
                            h2.iter_mut().for_each(|v| *v = v.tanh());
                            linear_row(&h2, layer.conv3_weight, Some(layer.conv3_bias), MASK_HIDDEN, bw * 4, &mut h3);

                            // Exact reshape/gate derivation (nspk=1): channel c in
                            // [0,4*BW) = a*2*BW + b*BW + w, a,b in {0,1}, w in [0,BW).
                            // mask_real[w] = h3[w]        * sigmoid(h3[2*BW+w])
                            // mask_imag[w] = h3[BW+w]     * sigmoid(h3[3*BW+w])
                            for w in 0..bw {
                                let mask_real = h3[w] * crate::ops::sigmoid(h3[2 * bw + w]);
                                let mask_imag = h3[bw + w] * crate::ops::sigmoid(h3[3 * bw + w]);
                                let mr = mix_real[(off + w) * t_frames + t];
                                let mi = mix_imag[(off + w) * t_frames + t];
                                real_slice[(local_off + w) * t_frames + t] = mr * mask_real - mi * mask_imag;
                                imag_slice[(local_off + w) * t_frames + t] = mr * mask_imag + mi * mask_real;
                            }
                        }
                    }
                });
            }
        });
        (est_real, est_imag)
    }
}
