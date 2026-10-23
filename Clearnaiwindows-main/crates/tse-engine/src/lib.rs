//! BVC backend abstraction (`BvcBackend`, `BvcRouterStage`) and the Target
//! Speaker Extraction backend interface (`TseBackend`) plus its
//! `dsp_core::Stage` bridge (`TseStageAdapter`).
//!
//! This crate does not itself depend on `bvc-hush` - `BvcRouterStage` takes
//! `hush`/`tse` as already-constructed, already-isolated (watchdog-wrapped)
//! `Box<dyn dsp_core::Stage>` values, so it has no reason to know Hush's
//! concrete type. The app crate (`engine.rs`) is what wires a real
//! `bvc_hush::HushBvcStage` and a real (or, currently, `UnavailableTse`)
//! `TseBackend` into one `BvcRouterStage` - see `docs/TSE_ENROLLMENT.md`.

mod adapter;
mod backend;
mod router;
mod speaker_router;

pub use adapter::{TseMetrics, TseStageAdapter};
pub use backend::{TseBackend, TseError, UnavailableTse};
pub use router::{BvcBackend, BvcRouterStage};
pub use speaker_router::{SpeakerBvcBackend, SpeakerBvcRouterStage};
