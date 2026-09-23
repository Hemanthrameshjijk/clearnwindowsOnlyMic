//! Loads the BSRNN separator's real weight tensors, exported from the
//! hash-verified `spk_emb_causal_100` checkpoint by
//! `poc/real_tse/export/export_separator_weights.py` and
//! `dump_separator_weights_as_raw.py` (see that script's docstring for the
//! exact module-to-tensor-name mapping, confirmed by reading the real
//! `bsrnn.py` source, not assumed).

use serde::Deserialize;
use std::collections::HashMap;

const WEIGHTS_BLOB: &[u8] = include_bytes!("../assets/separator_weights.bin");
const MANIFEST_JSON: &str = include_str!("../assets/separator_weights_manifest.json");

#[derive(Deserialize)]
struct TensorEntry {
    shape: Vec<usize>,
    dtype: String,
    offset: usize,
    length: usize,
}

pub struct WeightStore {
    manifest: HashMap<String, TensorEntry>,
}

#[derive(Debug)]
pub struct WeightError(pub String);

impl std::fmt::Display for WeightError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "weight loading error: {}", self.0)
    }
}
impl std::error::Error for WeightError {}

impl WeightStore {
    pub fn load() -> Result<Self, WeightError> {
        let manifest: HashMap<String, TensorEntry> =
            serde_json::from_str(MANIFEST_JSON).map_err(|e| WeightError(format!("parsing manifest: {e}")))?;
        Ok(Self { manifest })
    }

    fn entry(&self, name: &str) -> Result<&TensorEntry, WeightError> {
        self.manifest.get(name).ok_or_else(|| WeightError(format!("missing tensor: {name}")))
    }

    /// Real tensor data, as a flat row-major (C-contiguous) f32 slice -
    /// same layout numpy/PyTorch used when exporting. Caller is expected to
    /// already know the expected shape (it's documented at each call site
    /// against the real architecture) - `shape()` is available for
    /// assertions in tests.
    pub fn get_f32(&self, name: &str) -> Result<&'static [f32], WeightError> {
        let e = self.entry(name)?;
        if e.dtype != "f32" {
            return Err(WeightError(format!("{name} is not f32 (is {})", e.dtype)));
        }
        let byte_start = e.offset;
        let byte_len = e.length * 4;
        let bytes = WEIGHTS_BLOB
            .get(byte_start..byte_start + byte_len)
            .ok_or_else(|| WeightError(format!("{name}: byte range out of bounds")))?;
        // SAFETY: `bytes` is a `'static` slice into the embedded asset,
        // 4-byte-aligned-independent read via `from_le_bytes` below would
        // be the fully-safe alternative; using bytemuck-free direct cast
        // here is avoided in favor of an explicit, obviously-correct copy
        // path (see `read_le_f32s`) - no unsafe code in this crate.
        Ok(read_le_f32s_leaked(bytes))
    }

    pub fn shape(&self, name: &str) -> Result<&[usize], WeightError> {
        Ok(&self.entry(name)?.shape)
    }

    pub fn get_i64_vec(&self, name: &str) -> Result<Vec<i64>, WeightError> {
        let e = self.entry(name)?;
        if e.dtype != "i64" {
            return Err(WeightError(format!("{name} is not i64 (is {})", e.dtype)));
        }
        let byte_start = e.offset;
        let byte_len = e.length * 8;
        let bytes = WEIGHTS_BLOB
            .get(byte_start..byte_start + byte_len)
            .ok_or_else(|| WeightError(format!("{name}: byte range out of bounds")))?;
        Ok(bytes.chunks_exact(8).map(|c| i64::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// Converts a raw little-endian byte slice into an owned `Vec<f32>`, then
/// leaks it to get a `'static` slice - acceptable here because every
/// weight tensor is loaded exactly once, at process startup, and lives for
/// the lifetime of the process (identical in spirit to `include_bytes!`
/// itself being `'static`; this just adds the necessary endian-safe
/// reinterpretation without unsafe transmutes).
fn read_le_f32s_leaked(bytes: &[u8]) -> &'static [f32] {
    let v: Vec<f32> = bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
    Box::leak(v.into_boxed_slice())
}
