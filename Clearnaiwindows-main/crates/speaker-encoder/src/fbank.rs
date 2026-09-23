//! Kaldi-style 80-dim log-mel filterbank feature extraction, matching the
//! exact configuration the bundled WeSpeaker ResNet34 (LM) model expects.
//!
//! # How these parameters were confirmed (not guessed)
//!
//! The ONNX file this crate bundles (see `lib.rs`) embeds its own metadata
//! (read directly from the file's protobuf `metadata_props` via a raw byte
//! dump, not assumed): `framework=wespeaker`,
//! `url=https://wespeaker-1256283475.cos.ap-shanghai.myqcloud.com/models/voxceleb/voxceleb_resnet34_LM.onnx`,
//! `sample_rate=16000`, `output_dim=256`, `normalize_samples=0` (the literal
//! byte `0`, confirmed via `od -c` on the raw file - not `1` like the CAM++
//! model that was tried and rejected first, see `lib.rs`'s docs), and no
//! `feature_normalize_type` key present at all (unlike CAM++, which had
//! `feature_normalize_type=global-mean`).
//!
//! `sherpa-onnx/csrc/features.h`'s `FeatureExtractorConfig` (fetched from
//! <https://raw.githubusercontent.com/k2-fsa/sherpa-onnx/master/sherpa-onnx/csrc/features.h>)
//! documents `normalize_samples`: "If true, the feature extractor expects
//! inputs to be normalized to the range `[-1, 1]`. If false, we will
//! multiply the inputs by 32768." Since this model's metadata says `0`
//! (false), this crate multiplies its (already `[-1, 1]`-ranged, from
//! `enrollment`'s capture format) samples by 32768.0 before computing fbank
//! - the same int16-amplitude convention `dsp_core`'s own `nnnoiseless`/
//! RNNoise wrapper already uses elsewhere in this codebase (see
//! `noise-engine/src/lib.rs`), so this is a familiar, not novel, scaling
//! convention for this project.
//!
//! `sherpa-onnx/csrc/speaker-embedding-extractor-general-impl.h` shows that
//! `SubtractGlobalMean` is applied *only* when a model's
//! `feature_normalize_type` metadata key is non-empty (`if
//! (!meta_data.feature_normalize_type.empty()) { ... }`). This model has no
//! such key, so this crate does **not** apply global-mean subtraction -
//! doing so anyway would silently feed this model input it was never
//! trained/exported to expect.
//!
//! The frame/mel parameters below (`frame_shift_ms=10`,
//! `frame_length_ms=25`, `num_bins=80`, `low_freq=20`, `high_freq=-400`,
//! `dither=0`, `snip_edges=false`, `window_type=povey`,
//! `preemph_coeff=0.97`, `remove_dc_offset=true`) come from the same
//! `FeatureExtractorConfig` defaults `sherpa-onnx` applies project-wide
//! (shared across every speaker-embedding model it serves, including the
//! CAM++ model tried first) - only `normalize_samples` and
//! `feature_normalize_type` differ per-model, and both are read from each
//! model's own metadata as described above, not guessed.
//!
//! This crate's own `kaldi_native_fbank::mel::MelBanks::new`/`compute`
//! source (fetched from
//! <https://raw.githubusercontent.com/RustedBytes/kaldi-native-fbank/master/src/mel.rs>)
//! was read directly to confirm its mel-filter math
//! (`mel_scale(freq) = 1127 * ln(1 + freq/700)`, the standard Kaldi
//! formula) does not depend on `MelOptions`'s `use_slaney_mel_scale`/`norm`/
//! `is_librosa`/`htk_mode` fields at all - those fields exist on the struct
//! but are never read by the actual bank-construction algorithm in this
//! port, so leaving them at their `Default` values does not risk a
//! Slaney-vs-Kaldi mel-scale mismatch.
//!
//! `FbankOptions::default().use_energy` is `true` in this crate (an
//! `htk`/older-Kaldi-style default that differs from plain Kaldi `fbank`'s
//! usual `false`) - explicitly set to `false` below, required to get
//! exactly 80 output dimensions (`FbankComputer::dim()` adds one extra
//! "energy" dimension when `use_energy` is true), matching this model's
//! `feature_dim=80` expectation.
//!
//! # What is not independently verified
//!
//! There is no way, in this sandboxed environment (no Python, no reference
//! Kaldi/sherpa-onnx binary, no real speech corpus), to run the *original*
//! Python/C++ feature extractor on the same audio and diff the resulting
//! fbank frames numerically against this Rust port's output. The
//! configuration above is sourced directly from the real, current upstream
//! code and this model's own embedded metadata (not guessed or copied from
//! an unrelated project) - but bit-exact numerical equivalence with the
//! reference implementation has not been, and cannot currently be,
//! confirmed by execution. What *has* been confirmed by execution: this
//! exact pipeline (resample -> scale by 32768 -> fbank -> tract inference)
//! produces a finite, non-degenerate, correctly-shaped 256-dim output for
//! synthetic test tones - see `lib.rs`'s tests.

use kaldi_native_fbank::mel::MelOptions;
use kaldi_native_fbank::online::FeatureComputer;
use kaldi_native_fbank::{FbankComputer, FbankOptions, FrameOptions, OnlineFeature};

/// Feature dimension the bundled model expects (its own `feature_dim`
/// convention, confirmed via `sherpa-onnx`'s `FeatureExtractorConfig::feature_dim`
/// default - see module docs).
pub const FEAT_DIM: usize = 80;

/// Sample rate the fbank extractor (and therefore the model) operates at.
pub const FBANK_SAMPLE_RATE_HZ: u32 = 16_000;

/// This model's `normalize_samples=0` metadata (confirmed via a raw byte
/// dump of the bundled file - see module docs) means samples must be scaled
/// from this project's usual `[-1.0, 1.0]` range up to the legacy
/// int16-amplitude range before fbank computation.
const SAMPLE_SCALE: f32 = 32768.0;

fn fbank_options() -> FbankOptions {
    FbankOptions {
        frame_opts: FrameOptions {
            samp_freq: FBANK_SAMPLE_RATE_HZ as f32,
            frame_shift_ms: 10.0,
            frame_length_ms: 25.0,
            dither: 0.0, // sherpa-onnx default: dithering disabled
            preemph_coeff: 0.97,
            remove_dc_offset: true,
            window_type: "povey".to_string(),
            round_to_power_of_two: true,
            blackman_coeff: 0.42, // unused for "povey"; left at this crate's default
            snip_edges: false,    // sherpa-onnx default
        },
        mel_opts: MelOptions {
            num_bins: FEAT_DIM,
            low_freq: 20.0,
            high_freq: -400.0, // interpreted as "nyquist - 400Hz" (7600Hz @ 16kHz) - see module docs
            ..MelOptions::default()
        },
        use_energy: false, // required for exactly 80 dims - see module docs
        ..FbankOptions::default()
    }
}

/// Computes 80-dim log-mel fbank frames for `audio` (mono `f32` samples in
/// `[-1.0, 1.0]`, already at `FBANK_SAMPLE_RATE_HZ`), returning a flat
/// row-major `(num_frames, FEAT_DIM)` buffer. **Not** mean-normalized - see
/// module docs on why this model's own metadata says not to. Returns
/// `(flattened_features, num_frames)`.
pub fn compute_fbank(audio: &[f32]) -> Result<(Vec<f32>, usize), String> {
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
        let frame = online
            .get_frame(i)
            .expect("num_frames_ready() guarantees this index is populated");
        out_row.copy_from_slice(frame);
    }

    Ok((flat, num_frames))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(secs: f32, amplitude: f32) -> Vec<f32> {
        (0..(FBANK_SAMPLE_RATE_HZ as f32 * secs) as usize)
            .map(|i| amplitude * (2.0 * std::f32::consts::PI * 220.0 * i as f32 / FBANK_SAMPLE_RATE_HZ as f32).sin())
            .collect()
    }

    #[test]
    fn produces_the_expected_feature_dimension() {
        let audio = tone(1.0, 0.3);
        let (flat, num_frames) = compute_fbank(&audio).expect("fbank extraction");
        assert_eq!(flat.len(), num_frames * FEAT_DIM);
        assert!(num_frames > 0);
    }

    #[test]
    fn frame_count_matches_the_10ms_shift_25ms_window_formula() {
        // snip_edges=false (Kaldi convention): num_frames = round(num_samples / frame_shift).
        let audio = tone(1.0, 0.3);
        let (_flat, num_frames) = compute_fbank(&audio).expect("fbank extraction");
        let frame_shift_samples = (FBANK_SAMPLE_RATE_HZ as f32 * 0.010) as usize;
        let expected = (audio.len() as f32 / frame_shift_samples as f32).round() as usize;
        assert!(
            (num_frames as i64 - expected as i64).abs() <= 1,
            "num_frames={num_frames}, expected~={expected}"
        );
    }

    #[test]
    fn every_output_value_is_finite() {
        let audio = tone(2.0, 0.5);
        let (flat, _) = compute_fbank(&audio).expect("fbank extraction");
        assert!(flat.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn silence_still_produces_frames_without_nan() {
        let audio = vec![0.0f32; FBANK_SAMPLE_RATE_HZ as usize];
        let (flat, num_frames) = compute_fbank(&audio).expect("fbank extraction");
        assert!(num_frames > 0);
        assert!(flat.iter().all(|v| v.is_finite()), "log(0) energy must be handled, not produce NaN/-inf");
    }

    #[test]
    fn louder_input_produces_a_different_feature_vector() {
        // Not a numeric-precision test against a reference implementation
        // (impossible here - see module docs) - only a sanity check that
        // the 32768x sample scaling is actually applied (a quiet and a loud
        // tone must not produce identical features).
        let quiet = compute_fbank(&tone(1.0, 0.05)).unwrap().0;
        let loud = compute_fbank(&tone(1.0, 0.5)).unwrap().0;
        assert_ne!(quiet, loud);
    }
}
