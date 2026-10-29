//! Real causal, bounded-state 48kHz<->16kHz (exact 3:1) resampler for the
//! path between ClearNAI's native 48kHz mic pipeline and the 16kHz ONNX models. A direct port of the already-fixed, verified
//! `poc/real_tse/src/causal_resampler.py`:
//! - lowpass FIR (causal, windowed-sinc, Hamming window) then decimate by
//!   3 for downsampling; zero-stuff by 3 then lowpass FIR (gain-
//!   compensated x3) for upsampling.
//! - `CausalDownsampler3x`'s decimation-phase tracking across chunk-
//!   boundary calls uses an explicit absolute-sample-position counter
//!   (`abs_pos`, mod 3) - an earlier version's ad hoc phase formula was
//!   found and fixed for exactly this reason (see the Python file's own
//!   history).
//! - The round-trip quality FAIL first measured for this algorithm
//!   (-13.22dB SI-SDR) was root-caused to a TEST-SCORING bug (the
//!   combined group delay used to align the round trip wrongly multiplied
//!   the upsampler's own delay - already in 48kHz-domain units - by 3, as
//!   if it were still in 16kHz-domain units). The resampler's own DSP was
//!   never broken: corrected round-trip result is SI-SDR 34.25dB /
//!   correlation 0.9998 on real speech (see `poc/real_tse/results.md` §8).
//!   Both directions' `group_delay_samples_48k` fields here are already in
//!   48kHz-sample units - do not multiply either by 3 when combining them.

fn windowed_sinc_lowpass(num_taps: usize, cutoff_ratio: f32) -> Vec<f32> {
    let mut h: Vec<f32> = (0..num_taps)
        .map(|i| {
            let n = i as f32 - (num_taps as f32 - 1.0) / 2.0;
            let x = cutoff_ratio * n;
            let sinc = if x.abs() < 1e-8 { 1.0 } else { (std::f32::consts::PI * x).sin() / (std::f32::consts::PI * x) };
            sinc * cutoff_ratio
        })
        .collect();
    // np.hamming(M): w[n] = 0.54 - 0.46*cos(2*pi*n/(M-1))
    for (i, v) in h.iter_mut().enumerate() {
        let w = 0.54 - 0.46 * (2.0 * std::f32::consts::PI * i as f32 / (num_taps as f32 - 1.0)).cos();
        *v *= w;
    }
    let sum: f32 = h.iter().sum();
    for v in h.iter_mut() {
        *v /= sum; // unity DC gain
    }
    h
}

/// `y[n] = sum_k kernel[k] * buf[n - k]`, evaluated causally (valid mode,
/// no padding): output length = `buf.len() - kernel.len() + 1`, output[i]
/// corresponds to `n = i + kernel.len() - 1`. Matches
/// `torch.nn.functional.conv1d(buf, kernel.flip(0))`'s real math exactly
/// (see the Python file's own derivation).
fn convolve_causal(buf: &[f32], kernel: &[f32]) -> Vec<f32> {
    let num_taps = kernel.len();
    if buf.len() < num_taps {
        return Vec::new();
    }
    let n_valid = buf.len() - num_taps + 1;
    let mut out = vec![0.0f32; n_valid];
    for (i, out_i) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for (k, &h) in kernel.iter().enumerate() {
            acc += h * buf[i + num_taps - 1 - k];
        }
        *out_i = acc;
    }
    out
}

pub struct CausalDownsampler3x {
    num_taps: usize,
    kernel: Vec<f32>,
    history: Vec<f32>,
    abs_pos: usize, // mod 3 - see module docs
    pub group_delay_samples_48k: usize,
}

impl CausalDownsampler3x {
    pub fn new(num_taps: usize) -> Self {
        let kernel = windowed_sinc_lowpass(num_taps, (1.0 / 3.0) * 0.95);
        let group_delay_samples_48k = (num_taps - 1) / 2;
        let mut s = Self { num_taps, kernel, history: Vec::new(), abs_pos: 0, group_delay_samples_48k };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.history = vec![0.0; self.num_taps - 1];
        self.abs_pos = 0;
    }

    /// `chunk_48k` -> ~`chunk_48k.len()/3` samples at 16kHz (may be empty
    /// during warm-up).
    pub fn push(&mut self, chunk_48k: &[f32]) -> Vec<f32> {
        let mut buf = std::mem::take(&mut self.history);
        buf.extend_from_slice(chunk_48k);

        if buf.len() < self.num_taps {
            self.history = buf;
            return Vec::new();
        }

        let filtered = convolve_causal(&buf, &self.kernel);
        let n = filtered.len();
        // first_keep = (-abs_pos) % 3, computed for unsigned abs_pos.
        let first_keep = (3 - (self.abs_pos % 3)) % 3;
        let mut decimated = Vec::new();
        let mut i = first_keep;
        while i < n {
            decimated.push(filtered[i]);
            i += 3;
        }
        self.abs_pos = (self.abs_pos + n) % 3;
        self.history = buf[buf.len() - (self.num_taps - 1)..].to_vec();
        decimated
    }
}

pub struct CausalUpsampler3x {
    num_taps: usize,
    kernel: Vec<f32>, // gain-compensated (x3)
    history: Vec<f32>,
    pub group_delay_samples_48k: usize,
}

impl CausalUpsampler3x {
    pub fn new(num_taps: usize) -> Self {
        let mut kernel = windowed_sinc_lowpass(num_taps, (1.0 / 3.0) * 0.95);
        for v in kernel.iter_mut() {
            *v *= 3.0;
        }
        let group_delay_samples_48k = (num_taps - 1) / 2;
        let mut s = Self { num_taps, kernel, history: Vec::new(), group_delay_samples_48k };
        s.reset();
        s
    }

    pub fn reset(&mut self) {
        self.history = vec![0.0; self.num_taps - 1];
    }

    /// `chunk_16k` -> ~`chunk_16k.len()*3` samples at 48kHz.
    pub fn push(&mut self, chunk_16k: &[f32]) -> Vec<f32> {
        let n = chunk_16k.len();
        if n == 0 {
            return Vec::new();
        }
        let mut zero_stuffed = vec![0.0f32; n * 3];
        for (i, &v) in chunk_16k.iter().enumerate() {
            zero_stuffed[i * 3] = v;
        }
        let mut buf = std::mem::take(&mut self.history);
        buf.extend_from_slice(&zero_stuffed);
        let filtered = convolve_causal(&buf, &self.kernel);
        self.history = buf[buf.len() - (self.num_taps - 1)..].to_vec();
        filtered
    }
}
