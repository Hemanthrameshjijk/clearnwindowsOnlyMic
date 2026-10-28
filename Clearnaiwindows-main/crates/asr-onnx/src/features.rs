//! Log-mel features for the speech model: 16 kHz, pre-emphasis 0.97, 25 ms Hann window in a 512-point FFT,
//! 10 ms hop, 80 Slaney mel bands, `ln(x + 2^-24)`, then per-feature mean/std normalisation over the segment.
//! (Matches fpConfig.json `audioModelCfg`/`mfccCfg` and the Python reference in `tools/make_golden.py`.)

use realfft::num_complex::Complex;
use realfft::{RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const SAMPLE_RATE: usize = 16_000;
pub const N_FFT: usize = 512;
pub const WIN: usize = 400;
pub const HOP: usize = 160;
pub const N_MELS: usize = 80;
const PREEMPH: f64 = 0.97;
const LOG_GUARD: f64 = 5.960_464_477_539_063e-8; // 2^-24

pub struct Features {
    fft: Arc<dyn RealToComplex<f64>>,
    window: Vec<f64>,
    /// N_MELS x (N_FFT/2 + 1), row-major.
    fb: Vec<f64>,
}

fn hz_to_mel(f: f64) -> f64 {
    let (lin, log_t, logstep) = (f / (200.0 / 3.0), 1000.0 / (200.0 / 3.0), 6.4f64.ln() / 27.0);
    if f >= 1000.0 {
        log_t + (f.max(1e-9) / 1000.0).ln() / logstep
    } else {
        lin
    }
}

fn mel_to_hz(m: f64) -> f64 {
    let (log_t, logstep) = (1000.0 / (200.0 / 3.0), 6.4f64.ln() / 27.0);
    if m >= log_t {
        1000.0 * (logstep * (m - log_t)).exp()
    } else {
        m * (200.0 / 3.0)
    }
}

impl Default for Features {
    fn default() -> Self {
        Self::new()
    }
}

impl Features {
    pub fn new() -> Self {
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(N_FFT);
        // np.hanning(WIN), centred inside the N_FFT frame.
        let left = (N_FFT - WIN) / 2;
        let mut window = vec![0.0; N_FFT];
        for n in 0..WIN {
            window[left + n] = 0.5 - 0.5 * (2.0 * std::f64::consts::PI * n as f64 / (WIN - 1) as f64).cos();
        }
        let bins = N_FFT / 2 + 1;
        let (lo, hi) = (hz_to_mel(0.0), hz_to_mel(SAMPLE_RATE as f64 / 2.0));
        let hz: Vec<f64> = (0..N_MELS + 2).map(|i| mel_to_hz(lo + (hi - lo) * i as f64 / (N_MELS + 1) as f64)).collect();
        let mut fb = vec![0.0; N_MELS * bins];
        for m in 0..N_MELS {
            let (l, c, h) = (hz[m], hz[m + 1], hz[m + 2]);
            for k in 0..bins {
                let f = k as f64 * (SAMPLE_RATE as f64 / 2.0) / (bins - 1) as f64;
                let w = ((f - l) / (c - l)).min((h - f) / (h - c)).max(0.0);
                fb[m * bins + k] = w * 2.0 / (h - l);
            }
        }
        Self { fft, window, fb }
    }

    /// Returns (mel-major data of shape [N_MELS, frames], frames); `None` when the audio is too short (< 513 samples).
    pub fn compute(&self, x: &[f32]) -> Option<(Vec<f32>, usize)> {
        let pad = N_FFT / 2;
        if x.len() <= pad {
            return None;
        }
        // pre-emphasis, then numpy-style reflect padding by N_FFT/2 on both sides
        let pre: Vec<f64> = (0..x.len()).map(|i| if i == 0 { x[0] as f64 } else { x[i] as f64 - PREEMPH * x[i - 1] as f64 }).collect();
        let mut xp = Vec::with_capacity(pre.len() + 2 * pad);
        for i in 0..pad {
            xp.push(pre[pad - i]);
        }
        xp.extend_from_slice(&pre);
        for j in 0..pad {
            xp.push(pre[pre.len() - 2 - j]);
        }
        let frames = 1 + (xp.len() - N_FFT) / HOP;
        let bins = N_FFT / 2 + 1;
        let mut mel = vec![0.0f64; N_MELS * frames];
        let mut buf = vec![0.0f64; N_FFT];
        let mut spec = vec![Complex::new(0.0, 0.0); bins];
        let mut power = vec![0.0f64; bins];
        for t in 0..frames {
            for (i, b) in buf.iter_mut().enumerate() {
                *b = xp[t * HOP + i] * self.window[i];
            }
            self.fft.process(&mut buf, &mut spec).ok()?;
            for (p, c) in power.iter_mut().zip(&spec) {
                *p = c.re * c.re + c.im * c.im;
            }
            for m in 0..N_MELS {
                let row = &self.fb[m * bins..(m + 1) * bins];
                let e: f64 = row.iter().zip(&power).map(|(a, b)| a * b).sum();
                mel[m * frames + t] = (e + LOG_GUARD).ln();
            }
        }
        let mut out = vec![0.0f32; N_MELS * frames];
        for m in 0..N_MELS {
            let r = &mel[m * frames..(m + 1) * frames];
            let mean = r.iter().sum::<f64>() / frames as f64;
            let var = r.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / frames as f64;
            let std = var.sqrt();
            for t in 0..frames {
                out[m * frames + t] = ((r[t] - mean) / (std + 1e-5)) as f32;
            }
        }
        Some((out, frames))
    }
}
