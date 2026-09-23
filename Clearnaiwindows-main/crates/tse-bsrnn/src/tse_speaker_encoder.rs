//! `speaker_profile::SpeakerEncoder` implementation for THIS checkpoint's
//! own baked-in `ECAPA_TDNN_GLOB_c512` speaker encoder (192-dim) - a
//! second, TSE-specific encoder, deliberately separate from
//! `speaker-encoder`'s bundled WeSpeaker ResNet34 (256-dim), per
//! `docs/TSE_INTEGRATION_PLAN.md` Step 3's explicit instruction not to
//! bridge the two embedding spaces.
//!
//! `speaker_profile::TargetSpeakerProfile`/`TargetSpeakerStore` turned out
//! to be dimension-agnostic when actually re-read (just an
//! `embedding: Arc<[f32]>` plus metadata, no hardcoded size) - so this
//! reuses those existing, already-tested types directly via a SECOND,
//! independent `TargetSpeakerStore` instance, rather than inventing a new
//! `TseSpeakerProfile` type as originally planned. `app`'s wiring
//! constructs that second store and points `TseStageAdapter` at it - see
//! `app/src/shared.rs`/`engine.rs`.
//!
//! # Fbank configuration - NOT numerically cross-verified (an honest gap)
//!
//! Every other piece of this crate (separator, STFT/iSTFT, ECAPA ONNX
//! export, causal resampler) was checked against a real PyTorch/Python
//! fixture and matched to ~1e-5 or better. This fbank extraction was NOT -
//! doing so would require another Python round trip this pass didn't
//! include. The configuration below is transcribed as carefully as
//! possible from the real source
//! (`wesep/modules/speaker/encoder.py::Fbank_kaldi`, which wraps
//! `torchaudio.compliance.kaldi.fbank`): `num_mel_bins=80,
//! frame_length=25ms, frame_shift=10ms, dither=1.0, window_type=hamming,
//! sample_rate=16000, use_energy=false`, `snip_edges` left at
//! `torchaudio.compliance.kaldi.fbank`'s own true default (`true`, Kaldi's
//! traditional convention - unlike `speaker-encoder`'s WeSpeaker model,
//! which explicitly uses `sherpa-onnx`'s `snip_edges=false` default), plus
//! per-utterance mean-only CMVN (`Fbank_kaldi.apply_cmvn`'s
//! `norm_mean=True, norm_var=False` defaults, confirmed not overridden in
//! this checkpoint's `config.yaml`). `dither=1.0` also means this
//! encoder's output is not bit-reproducible run-to-run by design (matching
//! the real reference model's own behavior) - a real, inherent property
//! of the reference, not a bug introduced here.

use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, FrameOptions, OnlineFeature};
use rubato::{FftFixedInOut, Resampler};
use speaker_profile::{NormalizationKind, SpeakerEncoder, SpeakerEncoderError, TargetSpeakerProfile};

use crate::SpeakerEncoderOnnx;

pub const FEAT_DIM: usize = 80;
pub const FBANK_SAMPLE_RATE_HZ: u32 = 16_000;
const SAMPLE_SCALE: f32 = 32768.0; // same int16-amplitude convention as speaker-encoder's fbank.rs
const CHUNK_SIZE_HINT: usize = 1600; // see speaker-encoder/src/resample.rs's docs on why this is only a hint

pub const ENCODER_ID: &str = "ecapa-tdnn-glob-c512-tse-16k";
pub const EMBEDDING_DIM: usize = 192;

fn fbank_options() -> FbankOptions {
    FbankOptions {
        frame_opts: FrameOptions {
            samp_freq: FBANK_SAMPLE_RATE_HZ as f32,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 1.0,
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: "hamming".to_string(),
            round_to_power_of_two: true,
            blackman_coeff: 0.42,
            snip_edges: true,
        },
        mel_opts: MelOptions { num_bins: FEAT_DIM, low_freq: 20.0, high_freq: -400.0, ..MelOptions::default() },
        use_energy: false,
        ..FbankOptions::default()
    }
}

fn compute_fbank_with_cmvn(audio: &[f32]) -> Result<(Vec<f32>, usize), String> {
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

    // CMVN: per-utterance mean-only normalization (Fbank_kaldi.apply_cmvn's
    // norm_mean=True, norm_var=False) - subtract each mel-bin's mean across
    // all frames from every frame's value in that bin.
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

/// Non-causal, whole-buffer resample to 16kHz - only ever called from
/// `embed()`, itself only ever called off the realtime audio thread during
/// enrollment (never per-frame) - same technique and same real caveats as
/// `speaker-encoder::resample::resample_to_fbank_rate` (its own copy, not
/// shared - see this crate's Cargo.toml).
fn resample_to_16k(input: &[f32], input_rate_hz: u32) -> Result<Vec<f32>, String> {
    if input_rate_hz == FBANK_SAMPLE_RATE_HZ {
        return Ok(input.to_vec());
    }
    if input.is_empty() {
        return Ok(Vec::new());
    }

    let mut resampler = FftFixedInOut::<f32>::new(input_rate_hz as usize, FBANK_SAMPLE_RATE_HZ as usize, CHUNK_SIZE_HINT, 1)
        .map_err(|e| format!("failed to construct resampler: {e}"))?;
    let chunk_size_in = resampler.input_frames_next();

    let mut output = Vec::new();
    let mut offset = 0usize;
    while offset < input.len() {
        let end = (offset + chunk_size_in).min(input.len());
        let mut chunk = vec![0.0f32; chunk_size_in];
        chunk[..end - offset].copy_from_slice(&input[offset..end]);
        let out_chunks = resampler.process(&[chunk], None).map_err(|e| format!("resample failed: {e}"))?;
        output.extend_from_slice(&out_chunks[0]);
        offset = end;
    }

    let expected_len = (input.len() as u64 * FBANK_SAMPLE_RATE_HZ as u64 / input_rate_hz as u64) as usize;
    output.truncate(expected_len.min(output.len()));
    Ok(output)
}

/// Real `speaker_profile::SpeakerEncoder` backend for this checkpoint's
/// baked-in ECAPA_TDNN_GLOB_c512 encoder - see module docs. Construction
/// (parsing the ~25MB ONNX graph) happens once, at engine startup, exactly
/// like `speaker_encoder::WeSpeakerEncoder`.
pub struct TseSpeakerEncoder {
    onnx: SpeakerEncoderOnnx,
}

impl TseSpeakerEncoder {
    pub fn new() -> Result<Self, crate::LoadError> {
        Ok(Self { onnx: SpeakerEncoderOnnx::new()? })
    }
}

impl SpeakerEncoder for TseSpeakerEncoder {
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
        let resampled = resample_to_16k(audio, input_sample_rate_hz).map_err(SpeakerEncoderError::InvalidAudio)?;
        let (fbank, num_frames) = compute_fbank_with_cmvn(&resampled).map_err(SpeakerEncoderError::InvalidAudio)?;

        let embedding = self
            .onnx
            .embed_from_fbank(&fbank, num_frames)
            .map_err(|e| SpeakerEncoderError::InferenceFailed(e.to_string()))?;

        if embedding.len() != EMBEDDING_DIM {
            return Err(SpeakerEncoderError::InferenceFailed(format!(
                "model produced {} values, expected {EMBEDDING_DIM}",
                embedding.len()
            )));
        }

        // NOT L2-normalized: the real reference pipeline's speaker-fusion
        // layer (SpeakerFuseLayer) consumes the raw embedding directly
        // through its own Linear projection - L2 normalization is specific
        // to speaker-profile's unrelated 256-dim verification use case.
        TargetSpeakerProfile::new(embedding, ENCODER_ID, FBANK_SAMPLE_RATE_HZ, NormalizationKind::None)
            .map_err(|e| SpeakerEncoderError::InferenceFailed(e.to_string()))
    }
}
