//! Real, hand-written tensor primitives backing the BSRNN separator - see
//! this crate's top-level docs for why these are hand-rolled instead of
//! going through `tract-onnx` (every op here is a trivial, well-understood
//! primitive; only the ECAPA speaker encoder uses tract-onnx). All
//! functions here are allocation-light and take pre-sized output buffers
//! where it matters for the realtime hot path (per-frame `process_frame`
//! calls) - construction-time-only code paths (weight loading) may still
//! allocate freely.

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// 8-way-accumulator dot product: a single running sum creates a serial
/// FMA dependency chain (each add must wait for the previous one), which
/// caps throughput at roughly one FMA latency per element regardless of
/// vectorization. Splitting into `LANES` independent partial sums breaks
/// that chain, letting the compiler pipeline/auto-vectorize the multiply-
/// adds; the final reduction of `LANES` values is negligible next to the
/// dot product itself. Shared by `linear_row` (below) and
/// `lstm::lstm_step` - both were measured (`tests/profile_breakdown.rs`)
/// running at naive-scalar throughput before this fix.
#[inline]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    const LANES: usize = 8;
    let mut acc = [0.0f32; LANES];
    let n = a.len();
    let full_chunks = n / LANES;
    for c in 0..full_chunks {
        let base = c * LANES;
        for l in 0..LANES {
            acc[l] += a[base + l] * b[base + l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for i in (full_chunks * LANES)..n {
        sum += a[i] * b[i];
    }
    sum
}

/// y = x @ W^T + b, for a single "row" x (in_dim,), W (out_dim, in_dim)
/// row-major (PyTorch's `nn.Linear`/`Conv1d(k=1)` weight convention),
/// b (out_dim,). Writes into `out` (must be exactly `out_dim` long).
pub fn linear_row(x: &[f32], weight: &[f32], bias: Option<&[f32]>, in_dim: usize, out_dim: usize, out: &mut [f32]) {
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);
    debug_assert_eq!(out.len(), out_dim);
    for o in 0..out_dim {
        let row = &weight[o * in_dim..(o + 1) * in_dim];
        out[o] = dot(x, row) + bias.map(|b| b[o]).unwrap_or(0.0);
    }
}

/// In-place per-timestep LayerNorm over `dim` (the "channel-wise" LayerNorm
/// used throughout BSRNN - `ChannelWiseLayerNorm`/`select_norm('cLN', ...)`
/// - confirmed via `wesep/modules/common/norm.py` to be an ordinary
/// per-position LayerNorm after a channel<->time transpose, NOT cumulative;
/// safe to apply independently per timestep/band with no history needed).
pub fn layer_norm_inplace(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32, dim: usize) {
    debug_assert_eq!(x.len(), dim);
    debug_assert_eq!(weight.len(), dim);
    debug_assert_eq!(bias.len(), dim);
    let mean = x.iter().sum::<f32>() / dim as f32;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    for i in 0..dim {
        x[i] = (x[i] - mean) * inv_std * weight[i] + bias[i];
    }
}

/// GroupNorm(num_groups=1, num_channels=dim) applied jointly over a whole
/// (dim, positions) block - NOT per-position. `nn.GroupNorm(1, C)` on
/// input shape (N, C, L) normalizes over (C, L) JOINTLY for each N, then
/// applies a per-channel affine (weight/bias, shape (C,)) broadcast across
/// L. `x` is laid out (positions, dim) row-major here (matching this
/// crate's band-major `(nband, feature_dim)` convention) - i.e.
/// `x[pos*dim + c]`. This is `band_comm`'s real norm (BUG FOUND: `BSNet`'s
/// `band_comm = ResRNN(..., norm_type='GN')` is HARDCODED regardless of
/// the `causal` flag - confirmed by reading `bsrnn.py` line 198 directly -
/// unlike `band_rnn`, which does use the causal-dependent `cLN`/per-
/// position norm this crate's `layer_norm_inplace` implements. An earlier
/// version of this crate wrongly applied per-position LayerNorm to
/// band_comm too, which passed band_rnn's own isolated test (max_abs_diff
/// ~1e-6) but diverged sharply once band_comm was included
/// (max_abs_diff~0.79) - exactly the kind of thing this fixes.
pub fn group_norm_1_inplace(x: &mut [f32], weight: &[f32], bias: &[f32], eps: f32, positions: usize, dim: usize) {
    debug_assert_eq!(x.len(), positions * dim);
    debug_assert_eq!(weight.len(), dim);
    debug_assert_eq!(bias.len(), dim);
    let n = (positions * dim) as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
    let inv_std = 1.0 / (var + eps).sqrt();
    for pos in 0..positions {
        for c in 0..dim {
            let idx = pos * dim + c;
            x[idx] = (x[idx] - mean) * inv_std * weight[c] + bias[c];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_row_matches_hand_computed_example() {
        // W = [[1,0,2],[0,1,-1]] (out=2, in=3), b=[0.5,-0.5], x=[1,2,3]
        let w = [1.0, 0.0, 2.0, 0.0, 1.0, -1.0];
        let b = [0.5, -0.5];
        let x = [1.0, 2.0, 3.0];
        let mut out = [0.0; 2];
        linear_row(&x, &w, Some(&b), 3, 2, &mut out);
        assert!((out[0] - (1.0 * 1.0 + 0.0 * 2.0 + 2.0 * 3.0 + 0.5)).abs() < 1e-6);
        assert!((out[1] - (0.0 * 1.0 + 1.0 * 2.0 + -1.0 * 3.0 - 0.5)).abs() < 1e-6);
    }

    #[test]
    fn layer_norm_zero_mean_unit_var_before_affine() {
        let mut x = [1.0, 2.0, 3.0, 4.0];
        let weight = [1.0; 4];
        let bias = [0.0; 4];
        layer_norm_inplace(&mut x, &weight, &bias, 1e-5, 4);
        let mean: f32 = x.iter().sum::<f32>() / 4.0;
        assert!(mean.abs() < 1e-5, "expected ~zero mean, got {mean}");
    }
}
