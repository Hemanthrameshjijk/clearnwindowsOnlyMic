//! Real-time accent conversion (Indian -> American English) with ONNX Runtime: a Rust port of the reference
//! engine in github.com/KarthikGavini/Accent-Converter (code/engine.py + dsp.py). The math is ported 1:1 and
//! verified against fixtures produced by the reference (see `tests/golden.rs`, `tools/make_golden.py`).
//!
//! Signal path of [`AccentStream`]:  16 kHz mono -> [`logmel::StreamingLogMel`] -> [`converter::StreamingConverterV5`]
//! -> [`vocoder::CausalStreamer`] -> 24 kHz mono. Fixed American voices only (no ECAPA / own-voice yet).

pub mod converter;
pub mod logmel;
pub mod npz;
pub mod ort_util;
pub mod pipeline;
pub mod resample;
pub mod vocoder;

use converter::StreamingConverterV5;
use logmel::{LogMel, StreamingLogMel};
use npz::Npz;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Duration;
use vocoder::CausalStreamer;

pub const VOICE_NAMES: [&str; 6] =
    ["american_male_1", "american_male_2", "american_male_3", "american_female_1", "american_female_2", "american_female_3"];

/// GUI labels: index 0 is Off, index `i` (1..=6) selects `VOICE_NAMES[i - 1]`.
pub const VOICE_LABELS: [&str; 7] = [
    "Off",
    "American male 1",
    "American male 2",
    "American male 3",
    "American female 1",
    "American female 2",
    "American female 3",
];

/// Voice picker labels (without "Off"): choice `i` (0..6) is voice index `i + 1`.
pub const VOICE_CHOICES: [&str; 6] = [
    "American male 1",
    "American male 2",
    "American male 3",
    "American female 1",
    "American female 2",
    "American female 3",
];

/// Files that make up the model bundle (what `setup.rs` ships / compares). ECAPA and RNNoise are not needed.
pub const BUNDLE_FILES: [&str; 8] = [
    "catalog.json",
    "features.npz",
    "american_voices.npz",
    "converter_best_v6dmel.npz",
    "converter_best_v6dmel.onnx",
    "converter_best_v6dmel.dec.onnx",
    "vocoder_causal_hifigan_best.npz",
    "vocoder_causal_hifigan_best.onnx",
];

/// Load `onnxruntime.dll` once for the process.
pub fn init_runtime(dll: &Path) -> Result<(), String> {
    ort_util::init_ort(dll)
}

/// One step of the stream, stage by stage (used by the golden tests; `audio` still contains the vocoder lag).
pub struct StepOut {
    pub mel_in: Vec<f32>,
    pub mel_out: Vec<f32>,
    pub audio: Vec<f32>,
}

pub struct AccentStream {
    spec: StreamingLogMel,
    conv: StreamingConverterV5,
    voc: CausalStreamer,
    voices: BTreeMap<String, Vec<f32>>,
    /// Vocoder-lag samples still to drop from the start of the stream.
    lag_left: usize,
}

impl AccentStream {
    pub fn load(models: &Path, threads: usize, voice: &str) -> Result<Self, String> {
        let cat: serde_json::Value = serde_json::from_slice(&std::fs::read(models.join("catalog.json")).map_err(|e| format!("catalog.json: {e}"))?)
            .map_err(|e| format!("catalog.json: {e}"))?;
        let s = |v: &serde_json::Value, k: &str| -> Result<String, String> {
            v.get(k).and_then(|x| x.as_str()).map(str::to_string).ok_or_else(|| format!("catalog.json: '{k}' missing"))
        };
        let conv_entry = cat["converters"].get(0).ok_or("catalog.json: no converters")?;
        let mel = LogMel::from_npz(&Npz::open(&models.join(s(&cat, "features")?))?)?;
        let vz = Npz::open(&models.join("american_voices.npz"))?;
        let mut voices = BTreeMap::new();
        for name in VOICE_NAMES {
            voices.insert(name.to_string(), vz.f32s(name)?);
        }
        let spk = voices.get(voice).ok_or_else(|| format!("unknown voice '{voice}'"))?.clone();
        let conv = StreamingConverterV5::load(models, &s(conv_entry, "npz")?, &s(conv_entry, "onnx")?, threads, &spk)?;
        let voc = CausalStreamer::load(models, &s(&cat["vocoder"], "npz")?, &s(&cat["vocoder"], "onnx")?, threads)?;
        let lag_left = voc.lag();
        Ok(Self { spec: StreamingLogMel::new(mel), conv, voc, voices, lag_left })
    }

    pub fn set_voice(&mut self, voice: &str) -> Result<(), String> {
        let spk = self.voices.get(voice).ok_or_else(|| format!("unknown voice '{voice}'"))?;
        self.conv.set_voice(spk);
        Ok(())
    }

    pub fn reset(&mut self) {
        self.spec.reset();
        self.conv.reset();
        self.voc.reset();
        self.lag_left = self.voc.lag();
    }

    /// Algorithmic delay of the model chain in ms (converter look-ahead + vocoder look-ahead), excluding the
    /// 25 ms mel window reach and any compute / buffering time.
    pub fn algorithmic_latency_ms(&self) -> f32 {
        self.conv.latency_ms() + (self.voc.lag() as f32 / 24.0)
    }

    /// 16 kHz samples in, stage outputs out. `audio` is the raw vocoder output (lag not dropped).
    pub fn push_stages(&mut self, x16: &[f32]) -> Result<StepOut, String> {
        let mel_in = self.spec.push(x16);
        let mel_out = self.conv.push(&mel_in)?;
        let audio = self.voc.push(&mel_out)?;
        Ok(StepOut { mel_in, mel_out, audio })
    }

    pub fn flush_stages(&mut self) -> Result<StepOut, String> {
        let mel_in = self.spec.flush();
        let mut mel_out_all = self.conv.push(&mel_in)?;
        mel_out_all.extend(self.conv.flush()?);
        let mut audio = self.voc.push(&mel_out_all)?;
        audio.extend(self.voc.flush()?);
        Ok(StepOut { mel_in, mel_out: mel_out_all, audio })
    }

    fn drop_lag(&mut self, mut audio: Vec<f32>) -> Vec<f32> {
        let d = self.lag_left.min(audio.len());
        self.lag_left -= d;
        audio.drain(..d);
        audio
    }

    /// 16 kHz in -> 24 kHz audio ready to play (the first `lag` samples of junk are removed).
    pub fn push(&mut self, x16: &[f32]) -> Result<Vec<f32>, String> {
        let a = self.push_stages(x16)?.audio;
        Ok(self.drop_lag(a))
    }

    pub fn flush(&mut self) -> Result<Vec<f32>, String> {
        let a = self.flush_stages()?.audio;
        Ok(self.drop_lag(a))
    }

    /// Time spent inside ONNX Runtime (converter, vocoder) since the last call.
    pub fn take_busy(&mut self) -> (Duration, Duration) {
        (self.conv.take_busy(), self.voc.take_busy())
    }
}

/// Where the model bundle lives: `ACCENT_MODELS_DIR`, else `app_models` next to the exe, else
/// `%LOCALAPPDATA%\ClearNAI\app_models`.
pub fn find_models_dir() -> Option<PathBuf> {
    let ok = |p: &PathBuf| BUNDLE_FILES.iter().all(|f| p.join(f).exists());
    let mut cands: Vec<PathBuf> = Vec::new();
    if let Ok(p) = std::env::var("ACCENT_MODELS_DIR") {
        cands.push(p.into());
    }
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            cands.push(d.join("app_models"));
        }
    }
    if let Ok(l) = std::env::var("LOCALAPPDATA") {
        cands.push(PathBuf::from(l).join("ClearNAI").join("app_models"));
    }
    cands.into_iter().find(ok)
}
