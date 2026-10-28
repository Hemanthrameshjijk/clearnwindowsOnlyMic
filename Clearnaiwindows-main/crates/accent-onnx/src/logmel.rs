//! 16 kHz input log-mel, a port of `dsp.LogMel` / `dsp.StreamingLogMel` (reference: Accent-Converter/code/dsp.py).
//! Centred STFT (reflect padding), 1024-FFT with the 800-sample window centred inside it, magnitude floored at
//! sqrt(1e-10), Slaney mel filters from features.npz, floor 1e-10, log10. Computed in f64 like numpy.

use crate::npz::Npz;
use realfft::{num_complex::Complex, RealFftPlanner, RealToComplex};
use std::sync::Arc;

pub const N_MELS: usize = 80;

pub struct LogMel {
    fb: Vec<f64>, // [n_fft/2+1, 80]
    win: Vec<f64>,
    pub n_fft: usize,
    pub hop: usize,
    pub win_length: usize,
    fft: Arc<dyn RealToComplex<f64>>,
    inbuf: Vec<f64>,
    spec: Vec<Complex<f64>>,
    fft_scratch: Vec<Complex<f64>>,
    amp: Vec<f64>,
}

impl LogMel {
    pub fn from_npz(z: &Npz) -> Result<Self, String> {
        let fb = z.get("fb")?;
        let window = z.get("window")?.data.clone();
        let n_fft = z.scalar("n_fft")? as usize;
        let hop = z.scalar("hop")? as usize;
        if fb.shape != [n_fft / 2 + 1, N_MELS] {
            return Err(format!("features fb shape {:?}", fb.shape));
        }
        let left = (n_fft - window.len()) / 2;
        let mut win = vec![0.0; n_fft];
        win[left..left + window.len()].copy_from_slice(&window);
        let fft = RealFftPlanner::<f64>::new().plan_fft_forward(n_fft);
        Ok(Self {
            fb: fb.data.clone(),
            win,
            n_fft,
            hop,
            win_length: window.len(),
            inbuf: fft.make_input_vec(),
            spec: fft.make_output_vec(),
            fft_scratch: fft.make_scratch_vec(),
            amp: vec![0.0; n_fft / 2 + 1],
            fft,
        })
    }

    /// `n` frames from `seg` (frame i = seg[hop*i .. hop*i + n_fft]), appended to `out` as [n, 80].
    pub fn frames_to_mel(&mut self, seg: &[f32], n: usize, out: &mut Vec<f32>) {
        for i in 0..n {
            let fr = &seg[self.hop * i..self.hop * i + self.n_fft];
            for (k, v) in self.inbuf.iter_mut().enumerate() {
                *v = fr[k] as f64 * self.win[k];
            }
            self.fft.process_with_scratch(&mut self.inbuf, &mut self.spec, &mut self.fft_scratch).expect("fft");
            for (a, c) in self.amp.iter_mut().zip(&self.spec) {
                *a = (c.re * c.re + c.im * c.im).max(1e-10).sqrt();
            }
            let mut mel = [0.0f64; N_MELS];
            for (k, &a) in self.amp.iter().enumerate() {
                let row = &self.fb[k * N_MELS..(k + 1) * N_MELS];
                for (m, &f) in mel.iter_mut().zip(row) {
                    *m += a * f;
                }
            }
            out.extend(mel.iter().map(|&m| m.max(1e-10).log10() as f32));
        }
    }
}

/// Same frames as the whole-file LogMel, emitted as soon as the last sample a frame needs has arrived.
pub struct StreamingLogMel {
    mel: LogMel,
    half: usize,
    reach: usize,
    head: Vec<f32>,
    buf: Vec<f32>,
    buf_start: i64,
    n: usize,
    next_t: usize,
    seg: Vec<f32>,
}

impl StreamingLogMel {
    pub fn new(mel: LogMel) -> Self {
        let half = mel.n_fft / 2;
        let reach = mel.win_length / 2;
        Self { mel, half, reach, head: Vec::new(), buf: Vec::new(), buf_start: 0, n: 0, next_t: 0, seg: Vec::new() }
    }

    pub fn reset(&mut self) {
        self.head.clear();
        self.buf.clear();
        self.buf_start = 0;
        self.n = 0;
        self.next_t = 0;
    }

    fn get(&self, idx: usize) -> f32 {
        if idx < self.head.len() {
            self.head[idx]
        } else {
            self.buf[idx - self.buf_start as usize]
        }
    }

    /// samples [a, b) of the (reflect-padded) stream into `self.seg`; `final_` enables right reflection at the end.
    fn sample_range(&mut self, a: i64, b: i64, final_: bool) {
        self.seg.clear();
        for idx in a..b {
            let v = if idx < 0 {
                let m = (-idx) as usize;
                if m < self.n { self.get(m) } else { 0.0 }
            } else if (idx as usize) < self.n {
                self.get(idx as usize)
            } else if final_ {
                let m = 2 * (self.n as i64 - 1) - idx;
                if m >= 0 { self.get(m as usize) } else { 0.0 }
            } else {
                0.0
            };
            self.seg.push(v);
        }
    }

    fn frames(&mut self, t0: usize, t1: usize, final_: bool, out: &mut Vec<f32>) {
        if t1 <= t0 {
            return;
        }
        let hop = self.mel.hop as i64;
        self.sample_range(hop * t0 as i64 - self.half as i64, hop * (t1 as i64 - 1) + self.half as i64, final_);
        let seg = std::mem::take(&mut self.seg);
        self.mel.frames_to_mel(&seg, t1 - t0, out);
        self.seg = seg;
    }

    /// Feed 16 kHz samples; returns the new mel frames as [frames, 80] row-major.
    pub fn push(&mut self, x: &[f32]) -> Vec<f32> {
        if self.head.len() < self.half + 1 {
            let take = (self.half + 1 - self.head.len()).min(x.len());
            self.head.extend_from_slice(&x[..take]);
        }
        self.buf.extend_from_slice(x);
        self.n += x.len();
        let t1 = if self.n >= self.reach { (self.n - self.reach) / self.mel.hop + 1 } else { 0 };
        let mut out = Vec::new();
        self.frames(self.next_t, t1, false, &mut out);
        self.next_t = self.next_t.max(t1);
        let keep_from = (self.mel.hop * self.next_t) as i64 - self.half as i64;
        if keep_from > self.buf_start {
            self.buf.drain(..(keep_from - self.buf_start) as usize);
            self.buf_start = keep_from;
        }
        out
    }

    pub fn flush(&mut self) -> Vec<f32> {
        let t1 = self.n / self.mel.hop + 1;
        let mut out = Vec::new();
        self.frames(self.next_t, t1, true, &mut out);
        self.next_t = t1;
        out
    }
}
