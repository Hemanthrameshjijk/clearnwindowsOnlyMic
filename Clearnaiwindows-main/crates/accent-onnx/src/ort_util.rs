//! Thin helpers over `ort` (load-dynamic): one-time init from the shipped `onnxruntime.dll`, session creation
//! with the same options as the reference engine (`_session`: ENABLE_ALL, no spinning), and f32 tensor in/out.

use ort::session::builder::GraphOptimizationLevel;
use ort::session::{Session, SessionInputValue};
use ort::value::TensorRef;
use std::borrow::Cow;
use std::path::Path;

/// Owned f32 tensor (shape + row-major data). Used for every explicit model state.
#[derive(Clone, Debug)]
pub struct Tensor32 {
    pub shape: Vec<i64>,
    pub data: Vec<f32>,
}

impl Tensor32 {
    pub fn zeros(shape: &[usize]) -> Self {
        Self { shape: shape.iter().map(|&d| d as i64).collect(), data: vec![0.0; shape.iter().product()] }
    }
    pub fn view(&self) -> Result<SessionInputValue<'_>, String> {
        view(&self.shape, &self.data)
    }
}

pub type Feed<'a> = Vec<(Cow<'a, str>, SessionInputValue<'a>)>;

pub fn view<'a>(shape: &[i64], data: &'a [f32]) -> Result<SessionInputValue<'a>, String> {
    let t = TensorRef::from_array_view((shape.to_vec(), data)).map_err(|e| e.to_string())?;
    Ok(SessionInputValue::from(t))
}

/// Load `onnxruntime.dll` (once per process; a second call, e.g. from bvc-vanta, is harmless).
pub fn init_ort(dll: &Path) -> Result<(), String> {
    ort::init_from(dll.to_string_lossy().to_string()).map_err(|e| e.to_string())?.commit();
    Ok(())
}

pub fn make_session(path: &Path, threads: usize) -> Result<Session, String> {
    fn s<T: std::fmt::Display>(path: &Path) -> impl Fn(T) -> String + '_ {
        move |e| format!("{}: {e}", path.display())
    }
    let mut b = Session::builder().map_err(s(path))?;
    b = b.with_optimization_level(GraphOptimizationLevel::All).map_err(s(path))?;
    b = b.with_intra_threads(threads.max(1)).map_err(s(path))?;
    b = b.with_inter_threads(1).map_err(s(path))?;
    b = b.with_intra_op_spinning(false).map_err(s(path))?;
    b = b.with_inter_op_spinning(false).map_err(s(path))?;
    b.commit_from_file(path).map_err(s(path))
}

/// Run and copy every output out as an owned f32 tensor (outputs are positional, like the reference's `out[i]`).
pub fn run(sess: &mut Session, feed: Feed<'_>) -> Result<Vec<Tensor32>, String> {
    let out = sess.run(feed).map_err(|e| e.to_string())?;
    let mut res = Vec::with_capacity(out.len());
    for i in 0..out.len() {
        let (shape, data) = out[i].try_extract_tensor::<f32>().map_err(|e| e.to_string())?;
        res.push(Tensor32 { shape: shape.to_vec(), data: data.to_vec() });
    }
    Ok(res)
}
