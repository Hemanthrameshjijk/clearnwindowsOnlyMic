//! Causal 24 kHz -> 48 kHz (exact 2:1) upsampler. Same recipe as `crate::causal_resampler::CausalUpsampler3x`
//! (zero-stuff, Hamming-windowed-sinc lowpass at 0.95 of Nyquist, gain-compensated, bounded history), which only
//! exists for 3:1. The 48k -> 16k direction reuses `causal_resampler::CausalDownsampler3x` directly.

pub struct CausalUpsampler2x {
    kernel: Vec<f32>, // unity-DC-gain lowpass, times 2 to compensate zero stuffing
    history: Vec<f32>,
    buf: Vec<f32>,
    pub group_delay_samples_48k: usize,
}

impl CausalUpsampler2x {
    pub fn new(num_taps: usize) -> Self {
        let cutoff = 0.5 * 0.95;
        let mut h: Vec<f32> = (0..num_taps)
            .map(|i| {
                let n = i as f32 - (num_taps as f32 - 1.0) / 2.0;
                let x = cutoff * n;
                let sinc = if x.abs() < 1e-8 { 1.0 } else { (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x) };
                let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (num_taps as f32 - 1.0)).cos();
                sinc * cutoff * w
            })
            .collect();
        let sum: f32 = h.iter().sum();
        for v in h.iter_mut() {
            *v = *v / sum * 2.0;
        }
        Self { kernel: h, history: vec![0.0; num_taps - 1], buf: Vec::new(), group_delay_samples_48k: (num_taps - 1) / 2 }
    }

    pub fn reset(&mut self) {
        self.history.iter_mut().for_each(|v| *v = 0.0);
    }

    /// `chunk_24k` -> exactly `2 * len` samples at 48 kHz, delayed by `group_delay_samples_48k`.
    pub fn push(&mut self, chunk_24k: &[f32]) -> Vec<f32> {
        let taps = self.kernel.len();
        let h = taps - 1;
        self.buf.clear();
        self.buf.extend_from_slice(&self.history);
        for &v in chunk_24k {
            self.buf.push(v);
            self.buf.push(0.0);
        }
        let mut out = Vec::with_capacity(2 * chunk_24k.len());
        for i in 0..2 * chunk_24k.len() {
            // y[i] = sum_k kernel[k] * buf[h + i - k]
            let mut acc = 0.0f32;
            for (k, &c) in self.kernel.iter().enumerate() {
                acc += c * self.buf[h + i - k];
            }
            out.push(acc);
        }
        let len = self.buf.len();
        self.history.copy_from_slice(&self.buf[len - h..]);
        out
    }
}
