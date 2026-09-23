//! Batched matmul, backing the separator's real performance fix.
//!
//! The first correctness-verified pass of `separator.rs` computed every
//! Linear/Conv1d(k=1)/LSTM-gate step as 32 independent per-band scalar
//! loops (`ops::linear_row`, called once per band). That is correct - it
//! matched the real PyTorch reference to ~1e-6 - but measured at RTF~31 on
//! a real 12s two-speaker mixture (370s wall-clock), because scalar,
//! unvectorized, per-band loops cannot use the CPU efficiently: a rough
//! real op-count for this model is ~260M multiply-adds per STFT frame
//! (~245M of that in the 6 repeat blocks' LSTM cells alone, ~15M in
//! band_masker), and a ~12s clip needs roughly 1500 frames, i.e. ~390
//! billion multiply-adds total - at naive scalar throughput (~1 GFLOP/s)
//! that is exactly the ~370s measured.
//!
//! `band_rnn`'s 32 bands are mathematically INDEPENDENT per timestep (each
//! band only depends on its own history through time, not on other
//! bands), so its LSTM gate computation for a given timestep is not truly
//! 32 separate operations - it is one (32, in_dim) x (in_dim, out_dim)
//! matmul that happens to have been computed as 32 separate matvecs.
//! `ndarray`'s default matmul (pure Rust, `matrixmultiply`-backed, no
//! external BLAS/system dependency) computes that batched form directly,
//! which is both mathematically identical (same real weights, same real
//! sums, only reordered/batched - re-verified against the same fixtures
//! that already passed at ~1e-6) and dramatically better for the CPU
//! (contiguous, cache-friendly, autovectorizable inner loops).

use ndarray::ArrayView2;

/// Batched `Y = X @ W^T + b`. `x`: `(batch, in_dim)` row-major. `weight`:
/// `(out_dim, in_dim)` row-major (PyTorch `nn.Linear`/`Conv1d(k=1)`
/// convention - unchanged from `ops::linear_row`). Returns `(batch,
/// out_dim)` row-major.
///
/// Uses `ArrayView2::from_shape` (borrows `x`/`weight` directly, no copy)
/// rather than `Array2::from_shape_vec` (which requires an owned `Vec`,
/// forcing a copy of the weight matrix on every single call - wasteful
/// since the weight never changes between calls). Only the output `y` is
/// freshly allocated, which is unavoidable.
pub fn batched_linear(x: &[f32], batch: usize, in_dim: usize, weight: &[f32], bias: Option<&[f32]>, out_dim: usize) -> Vec<f32> {
    debug_assert_eq!(x.len(), batch * in_dim);
    debug_assert_eq!(weight.len(), out_dim * in_dim);

    let x_arr = ArrayView2::from_shape((batch, in_dim), x).expect("shape matches by construction");
    let w_arr = ArrayView2::from_shape((out_dim, in_dim), weight).expect("shape matches by construction");
    // W^T view (in_dim, out_dim) - no copy, ndarray handles the transposed
    // stride directly.
    let y = x_arr.dot(&w_arr.t()); // (batch, out_dim), C-contiguous (Array2's dot always returns standard layout)
    let mut out = y.into_raw_vec();

    // Plain flat-index bias add (avoids depending on a specific ndarray
    // row-iteration method name across versions - out is guaranteed
    // (batch, out_dim) row-major since Array2::dot returns standard
    // (C-contiguous) layout).
    if let Some(b) = bias {
        debug_assert_eq!(b.len(), out_dim);
        for row in 0..batch {
            for o in 0..out_dim {
                out[row * out_dim + o] += b[o];
            }
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn batched_linear_matches_hand_computed_example() {
        // Same example as ops::linear_row's test, but batched (batch=2,
        // both rows identical input) to confirm the batched path agrees
        // with the already-verified per-row formula.
        let w = [1.0, 0.0, 2.0, 0.0, 1.0, -1.0]; // (2,3)
        let b = [0.5, -0.5];
        let x = [1.0, 2.0, 3.0, 1.0, 2.0, 3.0]; // (2,3): two identical rows
        let out = batched_linear(&x, 2, 3, &w, Some(&b), 2);
        assert_eq!(out.len(), 4);
        for row in 0..2 {
            assert!((out[row * 2] - (1.0 + 6.0 + 0.5)).abs() < 1e-6);
            assert!((out[row * 2 + 1] - (2.0 - 3.0 - 0.5)).abs() < 1e-6);
        }
    }
}
