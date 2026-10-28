//! Causal HiFi-GAN streamer, a port of the reference `engine.CausalVocoder` + `CausalStreamer`:
//! 80-band mel frames in, 24 kHz audio out (`hop` = 300 samples per frame), explicit state tensors.
//! The audio is `delay_frames * hop` samples behind the mel (see [`CausalStreamer::lag`]).

use crate::logmel::N_MELS;
use crate::npz::Npz;
use crate::ort_util::{make_session, run, view, Feed, Tensor32};
use ort::session::Session;
use std::borrow::Cow;
use std::path::Path;
use std::time::{Duration, Instant};

pub struct CausalStreamer {
    sess: Session,
    pub delay_frames: usize,
    pub hop: usize,
    silence: Vec<f32>,
    start_state: Vec<Tensor32>,
    names: Vec<String>,
    state: Vec<Tensor32>,
    busy: Duration,
}

impl CausalStreamer {
    pub fn load(dir: &Path, npz_name: &str, onnx_name: &str, threads: usize) -> Result<Self, String> {
        let z = Npz::open(&dir.join(npz_name))?;
        let n = z.scalar("n_state")? as usize;
        let mut start_state = Vec::with_capacity(n);
        for i in 0..n {
            let k = z.get(&format!("s{i}"))?;
            start_state.push(Tensor32 { shape: k.shape.iter().map(|&d| d as i64).collect(), data: k.f32s() });
        }
        let mut s = Self {
            sess: make_session(&dir.join(onnx_name), threads)?,
            delay_frames: z.scalar("delay_frames")? as usize,
            hop: z.scalar("hop")? as usize,
            silence: z.f32s("silence")?,
            names: (0..n).map(|i| format!("s{i}")).collect(),
            state: Vec::new(),
            start_state,
            busy: Duration::ZERO,
        };
        s.reset();
        Ok(s)
    }

    /// 24 kHz samples the output is behind the mel.
    pub fn lag(&self) -> usize {
        self.delay_frames * self.hop
    }

    pub fn take_busy(&mut self) -> Duration {
        std::mem::take(&mut self.busy)
    }

    pub fn reset(&mut self) {
        self.state = self.start_state.clone();
    }

    /// `mel` is [k, 80] row-major; returns k*hop samples at 24 kHz.
    pub fn push(&mut self, mel: &[f32]) -> Result<Vec<f32>, String> {
        if mel.is_empty() {
            return Ok(Vec::new());
        }
        let shape = [1, (mel.len() / N_MELS) as i64, N_MELS as i64];
        let t0 = Instant::now();
        let mut feed: Feed = vec![(Cow::Borrowed("mel"), view(&shape, mel)?)];
        for (n, s) in self.names.iter().zip(&self.state) {
            feed.push((Cow::Borrowed(n.as_str()), s.view()?));
        }
        let mut out = run(&mut self.sess, feed)?;
        self.busy += t0.elapsed();
        self.state = out.split_off(1);
        Ok(out.pop().unwrap().data)
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, String> {
        let sil: Vec<f32> = self.silence.iter().copied().cycle().take(self.delay_frames * N_MELS).collect();
        self.push(&sil)
    }
}
