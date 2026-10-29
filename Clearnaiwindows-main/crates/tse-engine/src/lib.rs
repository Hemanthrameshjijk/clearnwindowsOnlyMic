//! BVC backend routers: `BvcBackend`/`BvcRouterStage` for the mic path and
//! `SpeakerBvcBackend`/`SpeakerBvcRouterStage` for the speaker path.
//!
//! The crate name is historical (it used to also hold the Target Speaker
//! Extraction backend interface, since removed). It does not itself depend on
//! `bvc-hush` - the routers take already-constructed, already-isolated
//! (watchdog-wrapped) `Box<dyn dsp_core::Stage>` values, so they have no
//! reason to know Hush's concrete type. The app crate (`engine.rs`) wires the
//! real stages into them.

mod router;
mod speaker_router;

pub use router::{BvcBackend, BvcRouterStage};
pub use speaker_router::{SpeakerBvcBackend, SpeakerBvcRouterStage};
