//! `speaker_profile::SpeakerEncoder` implementation for Vanta's real,
//! pretrained ECAPA-TDNN speaker encoder (SpeechBrain's
//! `speechbrain/spkrec-ecapa-voxceleb`, 192-dim) - a THIRD independent
//! embedding space in this project, alongside `speaker-encoder`'s 256-dim
//! WeSpeaker and `tse-bsrnn`'s own 192-dim `ECAPA_TDNN_GLOB_c512` (same
//! dimension, but a DIFFERENT trained checkpoint - not interchangeable,
//! see this module's `ENCODER_ID`).
//!
//! # Why `ort` (onnxruntime), not `tract-onnx`
//!
//! Every other ONNX model in this project loads via `tract-onnx` (pure
//! Rust, no native DLL). This one genuinely cannot: real, hands-on testing
//! against the actual exported `assets/ecapa_embedding_model.onnx` failed
//! under THREE separate tract-onnx versions (0.19.16, 0.21.18, 0.23.7 -
//! the last being this workspace's own pinned version elsewhere), each
//! with a different real error rooted in how the attentive-statistics-
//! pooling layer's large pointwise `Conv1d` got decomposed by the ONNX
//! exporter (an `EinSum`/`PermuteAxes` shape-inference failure tract
//! cannot resolve, not a bug in the export itself - verified separately
//! against PyTorch/onnxruntime to max abs diff 0.0000496). This mirrors
//! this project's own documented precedent
//! (`speaker-encoder`'s CAM++ rejection): a genuine architecture/export
//! incompatibility, not something to keep patching around blindly.
//!
//! The `ort` crate (official onnxruntime Rust bindings) loads and runs
//! this exact file correctly, numerically verified to match Python
//! onnxruntime to max abs diff 0.0000496 on a fixed test input. Same
//! "not everything has to be pure-Rust tract" precedent this project
//! already established for Hush's native DLL - here it's onnxruntime's
//! DLL instead of Hush's, loaded the same "look in `dll_dir`, fail
//! cleanly (never silently) if missing" way as `bvc_hush::HushBvcStage`.
//!
//! # Fbank front-end - NOT numerically cross-verified (an honest gap)
//!
//! SpeechBrain's `Fbank` module is a torchaudio-based STFT+mel-filterbank
//! (Hann window, no dither, HTK-style mel scale, `n_fft=400`/`n_mels=80`/
//! `hop_length=10ms`/`win_length=25ms` at 16kHz - see this crate's build
//! notes for the exact `hyperparams.yaml` values read). This uses
//! `kaldi_native_fbank` (Kaldi-style: different mel-scale formula/window
//! convention) as an approximation, exactly the same category of honest,
//! flagged gap `tse_bsrnn::tse_speaker_encoder` already carries for its
//! own (different) ECAPA checkpoint - closing it would need a real
//! Python-vs-Rust fbank round trip this pass didn't include. `mean_var_norm`
//! itself (`norm_type=sentence, std_norm=False` - confirmed from the real
//! `hyperparams.yaml`) IS implemented exactly: per-utterance, per-mel-bin
//! mean-only subtraction, no variance normalization.

use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, FrameOptions, OnlineFeature};
use ort::session::Session;
use ort::value::Tensor;
use speaker_profile::{NormalizationKind, SpeakerEncoder, SpeakerEncoderError, TargetSpeakerProfile};
use std::path::Path;
use tse_bsrnn::causal_resampler::CausalDownsampler3x;

pub const FEAT_DIM: usize = 80;
pub const FBANK_SAMPLE_RATE_HZ: u32 = 16_000;
pub const EMBEDDING_DIM: usize = 192;
pub const ENCODER_ID: &str = "ecapa-tdnn-speechbrain-voxceleb-16k";
const SAMPLE_SCALE: f32 = 32768.0;
pub const DLL_FILE_NAME: &str = "onnxruntime.dll";
const ONNX_MODEL_BYTES: &[u8] = include_bytes!("../assets/ecapa_embedding_model.onnx");

#[derive(Debug)]
pub enum EcapaLoadError {
    DllNotFound(std::path::PathBuf),
    SessionInit(String),
    WriteModelTemp(String),
}

impl std::fmt::Display for EcapaLoadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::DllNotFound(p) => write!(f, "onnxruntime.dll not found at {}", p.display()),
            Self::SessionInit(e) => write!(f, "onnxruntime session init failed: {e}"),
            Self::WriteModelTemp(e) => write!(f, "failed to stage bundled ONNX model: {e}"),
        }
    }
}
impl std::error::Error for EcapaLoadError {}

fn fbank_options() -> FbankOptions {
    FbankOptions {
        frame_opts: FrameOptions {
            samp_freq: FBANK_SAMPLE_RATE_HZ as f32,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.0,
            preemph_coeff: 0.0,
            remove_dc_offset: true,
            window_type: "hann".to_string(),
            round_to_power_of_two: true,
            blackman_coeff: 0.42,
            snip_edges: false,
        },
        mel_opts: MelOptions { num_bins: FEAT_DIM, low_freq: 0.0, high_freq: 0.0, ..MelOptions::default() },
        use_energy: false,
        ..FbankOptions::default()
    }
}

fn compute_fbank_with_mean_norm(audio: &[f32]) -> Result<(Vec<f32>, usize), String> {
    let scaled: Vec<f32> = audio.iter().map(|&s| s * SAMPLE_SCALE).collect();

    let computer = FbankComputer::new(fbank_options())?;
    let mut online = OnlineFeature::new(FeatureComputer::Fbank(computer));
    online.accept_waveform(FBANK_SAMPLE_RATE_HZ as f32, &scaled);
    online.input_finished();

    let num_frames = online.num_frames_ready();
    if num_frames == 0 {
        return Err("no fbank frames were produced (input too short)".to_string());
    }

    let mut flat = vec![0.0f32; num_frames * FEAT_DIM];
    for (i, out_row) in flat.chunks_mut(FEAT_DIM).enumerate() {
        let frame = online.get_frame(i).expect("num_frames_ready() guarantees this index is populated");
        out_row.copy_from_slice(frame);
    }

    // mean_var_norm: norm_type="sentence", std_norm=False (confirmed from
    // the real hyperparams.yaml) - per-utterance, per-mel-bin mean-only
    // subtraction.
    let mut means = vec![0.0f64; FEAT_DIM];
    for row in flat.chunks(FEAT_DIM) {
        for (m, v) in means.iter_mut().zip(row.iter()) {
            *m += *v as f64;
        }
    }
    for m in means.iter_mut() {
        *m /= num_frames as f64;
    }
    for row in flat.chunks_mut(FEAT_DIM) {
        for (v, m) in row.iter_mut().zip(means.iter()) {
            *v -= *m as f32;
        }
    }

    Ok((flat, num_frames))
}

/// Real `speaker_profile::SpeakerEncoder` backend for Vanta's ECAPA-TDNN.
/// Construction (loading `onnxruntime.dll` + parsing the ~83MB ONNX graph)
/// happens once, at engine startup - never per-enrollment, never per-frame.
pub struct VantaSpeakerEncoder {
    session: std::sync::Mutex<Session>,
}

impl VantaSpeakerEncoder {
    /// `dll_dir`: directory containing `onnxruntime.dll` (shipped alongside
    /// the app the same way `weya_nc.dll` is - see `bvc_hush`'s docs - NOT
    /// embedded in the binary, since it's a large native library, not a
    /// small parseable graph).
    pub fn try_load(dll_dir: &Path) -> Result<Self, EcapaLoadError> {
        let dll_path = dll_dir.join(DLL_FILE_NAME);
        if !dll_path.exists() {
            return Err(EcapaLoadError::DllNotFound(dll_path));
        }
        let init_ok = ort::init_from(dll_path.to_string_lossy().to_string())
            .map_err(|e| EcapaLoadError::SessionInit(e.to_string()))?
            .commit();
        if !init_ok {
            return Err(EcapaLoadError::SessionInit("ort::init_from(...).commit() returned false".to_string()));
        }

        // `ort::Session::builder().commit_from_file` needs a real file path
        // (not in-memory bytes for this onnx-graph size reliably across
        // ort versions) - stage the bundled bytes to a temp file once,
        // exactly like `noise_engine::load_bundled_deep_filter` does for
        // its own bundled model archive.
        let temp_path = std::env::temp_dir().join("clearnai_vanta_ecapa.onnx");
        std::fs::write(&temp_path, ONNX_MODEL_BYTES).map_err(|e| EcapaLoadError::WriteModelTemp(e.to_string()))?;

        let session = Session::builder()
            .map_err(|e| EcapaLoadError::SessionInit(e.to_string()))?
            .commit_from_file(&temp_path)
            .map_err(|e| EcapaLoadError::SessionInit(e.to_string()))?;

        Ok(Self { session: std::sync::Mutex::new(session) })
    }

    fn run_embedding_model(&self, fbank: &[f32], num_frames: usize) -> Result<Vec<f32>, String> {
        let input = Tensor::<f32>::from_array(([1usize, num_frames, FEAT_DIM], fbank.to_vec())).map_err(|e| e.to_string())?;
        let mut session = self.session.lock().map_err(|_| "session mutex poisoned".to_string())?;
        let outputs = session.run(ort::inputs!["feats" => input]).map_err(|e| e.to_string())?;
        let (_, data) = outputs["embedding"].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        Ok(data.to_vec())
    }
}

impl SpeakerEncoder for VantaSpeakerEncoder {
    fn encoder_id(&self) -> &'static str {
        ENCODER_ID
    }

    fn embedding_dim(&self) -> Option<usize> {
        Some(EMBEDDING_DIM)
    }

    fn expected_sample_rate_hz(&self) -> u32 {
        FBANK_SAMPLE_RATE_HZ
    }

    fn embed(&self, audio: &[f32], input_sample_rate_hz: u32) -> Result<TargetSpeakerProfile, SpeakerEncoderError> {
        let resampled = if input_sample_rate_hz == FBANK_SAMPLE_RATE_HZ {
            audio.to_vec()
        } else {
            // Enrollment-only, off the realtime thread - a fresh
            // whole-buffer resampler per call is fine here (same
            // trade-off `tse_bsrnn::tse_speaker_encoder::resample_to_16k`
            // already documents), even though `CausalDownsampler3x` is
            // built for the streaming 48k->16k case elsewhere in this
            // crate.
            let mut down = CausalDownsampler3x::new(63);
            let mut out = down.push(audio);
            out.extend(down.push(&[]));
            out
        };

        let (fbank, num_frames) = compute_fbank_with_mean_norm(&resampled).map_err(SpeakerEncoderError::InvalidAudio)?;

        let embedding = self.run_embedding_model(&fbank, num_frames).map_err(SpeakerEncoderError::InferenceFailed)?;

        if embedding.len() != EMBEDDING_DIM {
            return Err(SpeakerEncoderError::InferenceFailed(format!(
                "expected {EMBEDDING_DIM}-dim embedding, got {}",
                embedding.len()
            )));
        }

        TargetSpeakerProfile::new(embedding, ENCODER_ID, FBANK_SAMPLE_RATE_HZ, NormalizationKind::None)
            .map_err(|e| SpeakerEncoderError::InferenceFailed(e.to_string()))
    }
}
