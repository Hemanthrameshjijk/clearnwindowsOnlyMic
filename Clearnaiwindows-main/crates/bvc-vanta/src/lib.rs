//! Real Vanta (Conv-TasNet + ECAPA-TDNN) target-speaker-extraction backend
//! - `Komalpreet2809/Vanta`'s architecture, converted to a causal/streaming
//! form (see `research/tse/candidates/vanta-causal{,-rs}/`), now wired into
//! the real pipeline behind `tse_engine::TseBackend`.
//!
//! # Honest status as of this writing - READ BEFORE ENABLING
//!
//! - **Separator weights: real, trained.** Exported from an actual (if
//!   weak - epoch 6, val SI-SDR +5.911dB, real measured eval showed only
//!   +1.07dB average improvement with a -7.10dB worst case, PESQ 0.84)
//!   Colab checkpoint. 3,441,536 real params, confirmed loading correctly.
//! - **Encoder/decoder weights: real, trained.** A second export pass
//!   (after the first only dumped the separator submodule) pulled
//!   `audio_encoder.conv.weight`/`audio_decoder.deconv.weight` from the
//!   same checkpoint - confirmed from the checkpoint's own printed
//!   submodule prefixes (`['audio_decoder', 'audio_encoder', 'separator',
//!   'speaker_encoder']`), not guessed; the first guess (`encoder.`/
//!   `decoder.`) was wrong and caught immediately by the export cell's own
//!   "NOTHING MATCHED" safeguard rather than silently exporting nothing.
//!   Both tensors are (512, 1, 16), matching `ENC_CHANNELS * ENC_KERNEL`
//!   exactly. `VantaBackend::try_load` now constructs successfully.
//! - **License**: based on `Komalpreet2809/Vanta`, which has no LICENSE
//!   file. The user has represented that they contacted the author
//!   directly and received personal approval for this demo/personal use -
//!   that does not extend to distributing this backend or its weights to
//!   other ClearNAI users; resolve that separately before any such
//!   distribution.
//! - **ECAPA embedding runtime**: real, verified (`ort`/onnxruntime.dll,
//!   not tract-onnx - see `ecapa.rs`'s docs for why), but requires
//!   `onnxruntime.dll` to be present alongside the app (same "external DLL,
//!   fail cleanly if missing" pattern as `bvc-hush`'s `weya_nc.dll`) - not
//!   embedded in the binary.
//! - **Fbank front-end**: approximated (Kaldi-style via
//!   `kaldi_native_fbank`, not SpeechBrain's actual torchaudio-based
//!   `Fbank`) - not numerically cross-verified against Python. See
//!   `ecapa.rs`'s docs.
//!
//! Given the above, `app`'s wiring must keep this backend's `BvcBackend`
//! entry honestly labeled "experimental" and must NOT present it as
//! production-quality even once it does construct successfully.

pub mod backend;
pub mod ecapa;
pub mod engine;
pub mod separator;
pub mod weights;

pub use backend::VantaBackend;
