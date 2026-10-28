//! Speech-to-text for ClearNAI's live transcript: log-mel features, the quantised FastConformer model through
//! ONNX Runtime (the shipped `onnxruntime.dll`), greedy decoding, and the real-time worker/GUI plumbing.
//! The model files are NOT part of the repository or the installer (see docs/TRANSCRIPT.md).

pub mod features;
pub mod model;
pub mod pipeline;
pub mod segmenter;

pub use model::{find_models_dir, AsrModel, BUNDLE_FILES};
