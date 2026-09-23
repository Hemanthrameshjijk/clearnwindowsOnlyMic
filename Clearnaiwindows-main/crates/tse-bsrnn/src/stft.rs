//! Incremental STFT/iSTFT matching `wesep.modules.feature.speech.STFT`/
//! `iSTFT` (Hann window, win=512, hop=128, `torch.stft`/`istft`'s
//! `center=True` convention) - a direct port of the already-debugged
//! Python reference, `poc/real_tse/src/streaming_engine.py`'s
//! `StreamingSTFT`/`StreamingISTFT`. That file's docs record the exact
//! bugs found and fixed there (front-pad width must be `win/2`, not
//! `win-hop`; history must retain the whole unconsumed remainder, not a
//! fixed size; iSTFT must trim `win/2` samples off the front, persisted
//! across many small calls) - all preserved here, not re-derived from
//! scratch. Single-stream (no batch dimension) - production always
//! processes one mic channel.

use rustfft::{num_complex::Complex32, Fft, FftPlanner};
use std::sync::Arc;

fn hann_window(win: usize) -> Vec<f32> {
    // torch.hann_window(win, periodic=True) (the default): w[n] = 0.5 -
    // 0.5*cos(2*pi*n/win) for n=0..win-1 - denominator is `win`, NOT
    // `win-1` (that would be the "symmetric" variant, not used here).
    (0..win).map(|n| 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * n as f32) / win as f32).cos()).collect()
}

pub struct StreamingStft {
    win: usize,
    hop: usize,
    window: Vec<f32>,
    history: Vec<f32>,
    primed: bool,
    fft: Arc<dyn Fft<f32>>,
}

impl StreamingStft {
    pub fn new(win: usize, hop: usize) -> Self {
        let window = hann_window(win);
        let fft = FftPlanner::new().plan_fft_forward(win);
        let mut s = Self { win, hop, window, history: Vec::new(), primed: false, fft };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        let pad = self.win / 2;
        self.history = vec![0.0; pad];
        self.primed = false;
    }

    pub fn f_bins(&self) -> usize {
        self.win / 2 + 1
    }

    /// `new_samples`: raw audio for this call. Returns (real, imag) as
    /// flat `(F, T)` row-major buffers (F = win/2+1) and the frame count T
    /// (T may be 0 for small chunks during warm-up).
    pub fn push(&mut self, new_samples: &[f32]) -> (Vec<f32>, Vec<f32>, usize) {
        let pad = self.win / 2;
        if !self.primed {
            // Seed the history buffer the same way torch.stft's
            // center=True reflect-pad would at the very start of the whole
            // utterance: reverse of the first `win/2` samples.
            let seed_len = pad.min(new_samples.len());
            for i in 0..seed_len {
                self.history[pad - seed_len + i] = new_samples[seed_len - 1 - i];
            }
            self.primed = true;
        }

        let mut buf = std::mem::take(&mut self.history);
        buf.extend_from_slice(new_samples);
        let n_total = buf.len();
        let n_frames = if n_total >= self.win { (n_total - self.win) / self.hop + 1 } else { 0 };

        let f_bins = self.f_bins();
        let mut out_real = vec![0.0f32; f_bins * n_frames];
        let mut out_imag = vec![0.0f32; f_bins * n_frames];

        let mut scratch: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); self.win];
        for k in 0..n_frames {
            let start = k * self.hop;
            for i in 0..self.win {
                scratch[i] = Complex32::new(buf[start + i] * self.window[i], 0.0);
            }
            self.fft.process(&mut scratch);
            for f in 0..f_bins {
                out_real[f * n_frames + k] = scratch[f].re;
                out_imag[f * n_frames + k] = scratch[f].im;
            }
        }

        // Keep the entire unconsumed remainder as history (not a fixed
        // size) - see this module's top doc on why a fixed-size
        // truncation is a real, previously-found bug.
        let consumed = n_frames * self.hop;
        self.history = buf[consumed..].to_vec();

        (out_real, out_imag, n_frames)
    }
}

pub struct StreamingIstft {
    win: usize,
    hop: usize,
    window: Vec<f32>,
    win_sq: Vec<f32>,
    acc: Vec<f32>,
    norm_acc: Vec<f32>,
    pending_trim: usize,
    ifft: Arc<dyn Fft<f32>>,
}

impl StreamingIstft {
    pub fn new(win: usize, hop: usize) -> Self {
        let window = hann_window(win);
        let win_sq: Vec<f32> = window.iter().map(|w| w * w).collect();
        let ifft = FftPlanner::new().plan_fft_inverse(win);
        let mut s = Self { win, hop, window, win_sq, acc: Vec::new(), norm_acc: Vec::new(), pending_trim: 0, ifft };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.acc = vec![0.0; self.win];
        self.norm_acc = vec![0.0; self.win];
        // torch.istft's center=True trims win/2 samples off the front of
        // the full overlap-add reconstruction - see this module's top doc.
        self.pending_trim = self.win / 2;
    }

    /// `frames_real`/`frames_imag`: `(F, T)` row-major, F = win/2+1.
    /// Returns newly finalized real output samples (may be empty).
    pub fn push(&mut self, frames_real: &[f32], frames_imag: &[f32], t_frames: usize) -> Vec<f32> {
        if t_frames == 0 {
            return Vec::new();
        }
        let f_bins = self.win / 2 + 1;
        let mut out = Vec::with_capacity(t_frames * self.hop);
        let mut full = vec![Complex32::new(0.0, 0.0); self.win];
        let inv_n = 1.0 / self.win as f32;

        for t in 0..t_frames {
            for f in 0..f_bins {
                full[f] = Complex32::new(frames_real[f * t_frames + t], frames_imag[f * t_frames + t]);
            }
            // Hermitian symmetry to reconstruct the redundant half (this
            // is an inverse *real* FFT - matches torch.fft.irfft's
            // implicit assumption that the input is the non-redundant half
            // of a real signal's spectrum).
            for k in f_bins..self.win {
                full[k] = full[self.win - k].conj();
            }
            self.ifft.process(&mut full);

            self.acc.extend(std::iter::repeat(0.0).take(self.hop));
            self.norm_acc.extend(std::iter::repeat(0.0).take(self.hop));
            for i in 0..self.win {
                self.acc[i] += full[i].re * inv_n * self.window[i];
                self.norm_acc[i] += self.win_sq[i];
            }
            for i in 0..self.hop {
                let denom = self.norm_acc[i].max(1e-8);
                out.push(self.acc[i] / denom);
            }
            self.acc.drain(0..self.hop);
            self.norm_acc.drain(0..self.hop);
        }

        if self.pending_trim > 0 {
            let trim_now = self.pending_trim.min(out.len());
            out.drain(0..trim_now);
            self.pending_trim -= trim_now;
        }
        out
    }
}
