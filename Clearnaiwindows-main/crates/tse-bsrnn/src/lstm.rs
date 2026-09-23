//! Single-timestep LSTM cell, matching PyTorch's `nn.LSTM` exactly:
//! `weight_ih`/`weight_hh` are `(4*hidden, input_or_hidden)` with gates
//! concatenated in PyTorch's fixed order `[input, forget, cell, output]` -
//! this is a hard, documented PyTorch convention, not assumed - preserved
//! as-is by `export_separator_weights.py` (no gate reordering there), so
//! the same order is used here.
//!
//! PERFORMANCE (read before "optimizing" this function again):
//! `tests/profile_breakdown.rs` found this cell accounts for ~89-92% of
//! total separator wall-clock time (called 64x per timestep x 6 repeat
//! blocks - band_comm is a true sequential recurrence over the 32 band
//! positions, see `separator.rs`'s `run_band_comm` docs, so it can't be
//! batched across bands the way `run_band_rnn` batches its independent
//! per-band steps into one matmul).
//!
//! TRIED AND REVERTED: routing this cell's two matvecs through
//! `crate::matmul::batched_linear` (batch=1) - the same `ndarray`/
//! `matrixmultiply`-backed GEMM path that gave `run_band_rnn` a real
//! ~20x speedup at batch=32. Measured *42% SLOWER* (341us -> 484us per
//! step) on the real 12s fixture: a GEMM kernel's fixed per-call
//! (blocking/dispatch) overhead doesn't amortize at M=1, and the two
//! `Vec` allocations per call (x ~575k calls for the 12s fixture) add
//! real cost on top. The plain scalar loop below, despite its low
//! measured throughput (~2.3 GFLOP/s, i.e. essentially unvectorized),
//! is still the faster real option found so far. Any future attempt at
//! this should measure with `tests/profile_breakdown.rs` before assuming
//! a "should be faster" abstraction actually is one at this scale.

use crate::ops::{dot, sigmoid};

#[derive(Copy, Clone)]
pub struct LstmWeights<'a> {
    pub weight_ih: &'a [f32], // (4*hidden, input)
    pub weight_hh: &'a [f32], // (4*hidden, hidden)
    pub bias_ih: &'a [f32],   // (4*hidden,)
    pub bias_hh: &'a [f32],   // (4*hidden,)
}

/// One LSTM timestep. `x`: (input_size,), `h_prev`/`c_prev`: (hidden,).
/// Writes `h_new`/`c_new` (each must be exactly `hidden` long) - no
/// allocation, safe for the realtime hot path once weights are already
/// loaded.
#[allow(clippy::too_many_arguments)]
pub fn lstm_step(
    x: &[f32],
    h_prev: &[f32],
    c_prev: &[f32],
    w: &LstmWeights,
    input_size: usize,
    hidden: usize,
    h_new: &mut [f32],
    c_new: &mut [f32],
    gates_scratch: &mut [f32], // caller-owned scratch, length 4*hidden
) {
    debug_assert_eq!(x.len(), input_size);
    debug_assert_eq!(h_prev.len(), hidden);
    debug_assert_eq!(c_prev.len(), hidden);
    debug_assert_eq!(w.weight_ih.len(), 4 * hidden * input_size);
    debug_assert_eq!(w.weight_hh.len(), 4 * hidden * hidden);
    debug_assert_eq!(gates_scratch.len(), 4 * hidden);
    debug_assert_eq!(h_new.len(), hidden);
    debug_assert_eq!(c_new.len(), hidden);

    for g in 0..4 * hidden {
        let row_ih = &w.weight_ih[g * input_size..(g + 1) * input_size];
        let row_hh = &w.weight_hh[g * hidden..(g + 1) * hidden];
        gates_scratch[g] = w.bias_ih[g] + w.bias_hh[g] + dot(row_ih, x) + dot(row_hh, h_prev);
    }

    for j in 0..hidden {
        let i_gate = sigmoid(gates_scratch[j]);
        let f_gate = sigmoid(gates_scratch[hidden + j]);
        let g_gate = gates_scratch[2 * hidden + j].tanh();
        let o_gate = sigmoid(gates_scratch[3 * hidden + j]);
        let c = f_gate * c_prev[j] + i_gate * g_gate;
        c_new[j] = c;
        h_new[j] = o_gate * c.tanh();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_weights_and_zero_state_produce_zero_output() {
        let hidden = 3;
        let input_size = 2;
        let weight_ih = vec![0.0; 4 * hidden * input_size];
        let weight_hh = vec![0.0; 4 * hidden * hidden];
        let bias_ih = vec![0.0; 4 * hidden];
        let bias_hh = vec![0.0; 4 * hidden];
        let w = LstmWeights { weight_ih: &weight_ih, weight_hh: &weight_hh, bias_ih: &bias_ih, bias_hh: &bias_hh };
        let x = vec![1.0, -2.0];
        let h_prev = vec![0.0; hidden];
        let c_prev = vec![0.0; hidden];
        let mut h_new = vec![0.0; hidden];
        let mut c_new = vec![0.0; hidden];
        let mut scratch = vec![0.0; 4 * hidden];
        lstm_step(&x, &h_prev, &c_prev, &w, input_size, hidden, &mut h_new, &mut c_new, &mut scratch);
        // sigmoid(0)=0.5, tanh(0)=0 => i=0.5,f=0.5,g=0,o=0.5 => c=0.5*0+0.5*0=0 => h=0.5*tanh(0)=0
        for v in &h_new {
            assert!(v.abs() < 1e-6);
        }
        for v in &c_new {
            assert!(v.abs() < 1e-6);
        }
    }
}
