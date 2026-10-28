//! Streaming converter, a line-for-line port of the reference `engine.ConverterV5` + `StreamingConverterV5`
//! (Accent-Converter/code/engine.py), split path: encoder half -> melody smoothing (here, in code) -> decoder
//! half, with explicit state tensors. Input: raw 16 kHz log-mel frames. Output: 24 kHz-domain mel frames
//! (de-normalised), already lined up with the input (the first `2*(delay+dec_delay)` frames are dropped).

use crate::logmel::N_MELS;
use crate::npz::Npz;
use crate::ort_util::{make_session, run, view, Feed, Tensor32};
use ort::session::Session;
use std::borrow::Cow;
use std::path::Path;
use std::time::{Duration, Instant};

pub struct ConverterParams {
    pub src_mean: Vec<f32>,
    pub src_std: Vec<f32>,
    pub tgt_mean: Vec<f32>,
    pub tgt_std: Vec<f32>,
    pub delay: usize,
    pub dec_delay: usize,
    pub left: usize,
    pub inv: Vec<f32>,
    pub silence_in: Vec<f32>,
    pub start_state: Vec<Tensor32>,
    pub n_enc: usize,
    pub pitch_smooth_ms: f64,
    pub pitch_detail: f64,
}

impl ConverterParams {
    pub fn from_npz(z: &Npz) -> Result<Self, String> {
        if z.scalar("version")? as i64 != 5 {
            return Err("converter npz is not version 5".into());
        }
        let n = z.scalar("n_state")? as usize;
        let pitch_smooth_ms = if z.has("pitch_smooth_ms") { z.scalar("pitch_smooth_ms")? } else { 0.0 };
        if pitch_smooth_ms <= 0.0 || z.has("pitch_target") || z.has("voice") {
            return Err("only the split (melody-smoothing, no fixed-voice, no pitch_target) converter is supported".into());
        }
        let mut start_state = Vec::with_capacity(n);
        for i in 0..n {
            let shape: Vec<usize> = z.get(&format!("shape{i}"))?.data.iter().map(|&v| v as usize).collect();
            start_state.push(Tensor32::zeros(&shape));
        }
        Ok(Self {
            src_mean: z.f32s("src_mean")?,
            src_std: z.f32s("src_std")?,
            tgt_mean: z.f32s("tgt_mean")?,
            tgt_std: z.f32s("tgt_std")?,
            delay: z.scalar("delay")? as usize,
            dec_delay: z.scalar("dec_delay")? as usize,
            left: z.scalar("left")? as usize,
            inv: z.f32s("inv")?,
            silence_in: z.f32s("silence_in")?,
            start_state,
            n_enc: z.scalar("n_enc_state")? as usize,
            pitch_smooth_ms,
            pitch_detail: if z.has("pitch_detail") { z.scalar("pitch_detail")? } else { 0.0 },
        })
    }
}

/// Melody smoothing state: (value, started). Port of `shape_pitch` for the smooth-only case.
#[derive(Clone, Copy, Default)]
struct PitchState {
    y: f32,
    started: bool,
}

/// `pros` is [T, 2] row-major (log-F0, voiced logit). Returns the shaped copy; updates `st`.
fn shape_pitch(pros: &[f32], st: &mut PitchState, smooth_ms: f64, detail: f32) -> Vec<f32> {
    let a = (1.0 - (-12.5 / smooth_ms).exp()) as f32;
    let mut out = pros.to_vec();
    let (mut y, mut started) = (st.y, st.started);
    for t in 0..pros.len() / 2 {
        let lf0 = pros[2 * t];
        if pros[2 * t + 1] > 0.0 {
            y = if !started { lf0 } else { y + a * (lf0 - y) };
            started = true;
        }
        out[2 * t] = y + detail * (lf0 - y);
    }
    *st = PitchState { y, started };
    out
}

/// Rotary angles and attention mask for one call: (q_cos, q_sin, k_cos, k_sin, mask), row-major
/// [S,64], [S,64], [L+S,64], [L+S,64], [S,L+S].
fn rotary_and_mask(inv: &[f32], l: usize, left: usize, steps: usize, nvalid: usize) -> [Vec<f32>; 5] {
    let ls = l + steps;
    let hd = inv.len() * 2;
    let (mut kc, mut ks) = (vec![0f32; ls * hd], vec![0f32; ls * hd]);
    for p in 0..ls {
        for (j, &iv) in inv.iter().enumerate() {
            let f = p as f32 * iv;
            let (c, s) = (f.cos(), f.sin());
            kc[p * hd + 2 * j] = c;
            kc[p * hd + 2 * j + 1] = c;
            ks[p * hd + 2 * j] = s;
            ks[p * hd + 2 * j + 1] = s;
        }
    }
    let qc = kc[l * hd..].to_vec();
    let qs = ks[l * hd..].to_vec();
    let mut mask = vec![-1e9f32; steps * ls];
    for r in 0..steps {
        let i = l + r;
        for j in 0..ls {
            if j <= i && j + left >= i && j + nvalid >= l {
                mask[r * ls + j] = 0.0;
            }
        }
    }
    [qc, qs, kc, ks, mask]
}

pub struct StreamingConverterV5 {
    p: ConverterParams,
    enc: Session,
    dec: Session,
    names: Vec<String>,
    spk: Vec<f32>,
    run_every: usize,
    skip: usize,
    // streaming state
    state: Vec<Tensor32>,
    xbuf: Tensor32,
    nvalid: usize,
    pitch: PitchState,
    pending: Vec<f32>,
    to_skip: usize,
    received: usize,
    emitted: usize,
    pub pitch_detail: f32,
    /// Wall-clock time spent inside the two ONNX sessions since the last [`take_busy`](Self::take_busy).
    busy: Duration,
}

impl StreamingConverterV5 {
    pub fn load(dir: &Path, npz_name: &str, onnx_name: &str, threads: usize, spk: &[f32]) -> Result<Self, String> {
        let p = ConverterParams::from_npz(&Npz::open(&dir.join(npz_name))?)?;
        let enc = make_session(&dir.join(onnx_name), threads)?;
        let dec_name = onnx_name.strip_suffix(".onnx").map(|s| format!("{s}.dec.onnx")).ok_or("converter onnx name")?;
        let dec = make_session(&dir.join(dec_name), threads)?;
        let names = (0..p.start_state.len()).map(|i| format!("s{i}")).collect();
        let skip = 2 * (p.delay + p.dec_delay);
        let pitch_detail = p.pitch_detail as f32;
        let mut s = Self {
            xbuf: Tensor32::zeros(&[1, 2 * p.delay, N_MELS]),
            state: Vec::new(),
            p,
            enc,
            dec,
            names,
            spk: spk.to_vec(),
            run_every: 2,
            skip,
            nvalid: 0,
            pitch: PitchState::default(),
            pending: Vec::new(),
            to_skip: skip,
            received: 0,
            emitted: 0,
            pitch_detail,
            busy: Duration::ZERO,
        };
        s.reset();
        Ok(s)
    }

    pub fn set_voice(&mut self, spk: &[f32]) {
        self.spk = spk.to_vec();
    }

    /// Delay the converter itself adds (frames dropped at start x 12.5 ms), like the reference `latency_ms`.
    pub fn latency_ms(&self) -> f32 {
        (self.skip + 1) as f32 * 12.5
    }

    pub fn take_busy(&mut self) -> Duration {
        std::mem::take(&mut self.busy)
    }

    pub fn reset(&mut self) {
        self.state = self.p.start_state.clone();
        self.xbuf = Tensor32 {
            shape: vec![1, 2 * self.p.delay as i64, N_MELS as i64],
            data: self.p.silence_in.iter().copied().cycle().take(2 * self.p.delay * N_MELS).collect(),
        };
        self.nvalid = 0;
        self.pitch = PitchState::default();
        self.pending.clear();
        self.to_skip = self.skip;
        self.received = 0;
        self.emitted = 0;
    }

    fn step(&mut self, x: &[f32], need: Option<usize>) -> Result<Vec<f32>, String> {
        self.pending.extend_from_slice(x);
        let n = self.pending.len() / N_MELS / 2 * 2;
        if n == 0 || n < need.unwrap_or(self.run_every) {
            return Ok(Vec::new());
        }
        let [qc, qs, kc, ks, mask] = rotary_and_mask(&self.p.inv, self.p.left, self.p.left, n / 2, self.nvalid);
        let (s, ls, hd) = (n / 2, self.p.left + n / 2, self.p.inv.len() * 2);
        let xs = &self.pending[..n * N_MELS];
        let (xsh, qsh, ksh, msh) = ([1, n as i64, N_MELS as i64], [s as i64, hd as i64], [ls as i64, hd as i64], [s as i64, ls as i64]);
        let ne = self.p.n_enc;

        let t0 = Instant::now();
        let mut feed: Feed = vec![
            (Cow::Borrowed("x"), view(&xsh, xs)?),
            (Cow::Borrowed("rot_q_cos"), view(&qsh, &qc)?),
            (Cow::Borrowed("rot_q_sin"), view(&qsh, &qs)?),
            (Cow::Borrowed("rot_k_cos"), view(&ksh, &kc)?),
            (Cow::Borrowed("rot_k_sin"), view(&ksh, &ks)?),
            (Cow::Borrowed("mask"), view(&msh, &mask)?),
        ];
        for i in 0..ne {
            feed.push((Cow::Borrowed(self.names[i].as_str()), self.state[i].view()?));
        }
        let mut enc = run(&mut self.enc, feed)?;
        let enc_states = enc.split_off(2);
        let pros_raw = enc.pop().unwrap(); // [1,n,2]
        let phone = enc.pop().unwrap(); // [1,n,73]

        let pros = shape_pitch(&pros_raw.data, &mut self.pitch, self.p.pitch_smooth_ms, self.pitch_detail);
        let prosh = pros_raw.shape.clone();
        let spkh = [1i64, self.spk.len() as i64];
        let mut feed: Feed = vec![
            (Cow::Borrowed("x"), view(&xsh, xs)?),
            (Cow::Borrowed("phone"), phone.view()?),
            (Cow::Borrowed("pros"), view(&prosh, &pros)?),
            (Cow::Borrowed("spk"), view(&spkh, &self.spk)?),
            (Cow::Borrowed("rot_q_cos"), view(&qsh, &qc)?),
            (Cow::Borrowed("rot_q_sin"), view(&qsh, &qs)?),
            (Cow::Borrowed("rot_k_cos"), view(&ksh, &kc)?),
            (Cow::Borrowed("rot_k_sin"), view(&ksh, &ks)?),
            (Cow::Borrowed("mask"), view(&msh, &mask)?),
            (Cow::Borrowed("xbuf"), self.xbuf.view()?),
        ];
        for i in ne..self.names.len() {
            feed.push((Cow::Borrowed(self.names[i].as_str()), self.state[i].view()?));
        }
        let mut dec = run(&mut self.dec, feed)?;
        self.busy += t0.elapsed();
        let dec_states = dec.split_off(2);
        self.xbuf = dec.pop().unwrap();
        let mel = dec.pop().unwrap(); // [1,n,80]

        self.state = enc_states.into_iter().chain(dec_states).collect();
        self.nvalid = self.p.left.min(self.nvalid + n / 2);
        self.pending.drain(..n * N_MELS);
        let frames = mel.data.len() / N_MELS;
        let drop = self.to_skip.min(frames);
        self.to_skip -= drop;
        let kept = &mel.data[drop * N_MELS..];
        self.emitted += frames - drop;
        let mut out = Vec::with_capacity(kept.len());
        for (i, &v) in kept.iter().enumerate() {
            let b = i % N_MELS;
            out.push(v * self.p.tgt_std[b] + self.p.tgt_mean[b]);
        }
        Ok(out)
    }

    /// Push raw (un-normalised) input mel frames [k, 80]; returns the output mel frames now ready [m, 80].
    pub fn push(&mut self, frames: &[f32]) -> Result<Vec<f32>, String> {
        let x: Vec<f32> = frames
            .iter()
            .enumerate()
            .map(|(i, &v)| (v - self.p.src_mean[i % N_MELS]) / self.p.src_std[i % N_MELS])
            .collect();
        self.received += x.len() / N_MELS;
        self.step(&x, None)
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, String> {
        if self.received <= self.emitted {
            return Ok(Vec::new());
        }
        let need = self.received - self.emitted;
        let n = self.skip + (self.received + self.skip) % 2;
        let sil: Vec<f32> = self.p.silence_in.iter().copied().cycle().take(n * N_MELS).collect();
        let mut y = self.step(&sil, Some(2))?;
        y.truncate(need * N_MELS);
        Ok(y)
    }
}
