//! One-time target-speaker reference enrollment: a separate control path
//! from the always-on mic pipeline (see `dsp_core`/`bvc_hush`/`tse_engine`),
//! not a permanent `dsp_core::Stage`. The user records a short reference
//! sentence, this crate validates it (VAD/quality checks, no ML), runs it
//! through a `speaker_profile::SpeakerEncoder`, and publishes the resulting
//! `speaker_profile::TargetSpeakerProfile` to a
//! `speaker_profile::TargetSpeakerStore` that the realtime TSE stage
//! (`tse_engine::TseStageAdapter`) reads from - never the other way around.
//!
//! # Where this fits
//!
//! ```text
//!                     ClearNAI
//!                        |
//!           +------------+------------+
//!           |                         |
//!    Enrollment Mode              Call Mode
//!           |                         |
//!    Physical Mic (tap)          Physical Mic
//!           |                         |
//!    Reference Recording         AutoGain -> RNNoise -> BVC/TSE -> Studio
//!           |                         |
//!    VAD / Quality (this crate)       |
//!           |                         |
//!    Speaker Encoder                  |
//!           |                         |
//!    TargetSpeakerProfile ----------> (read by the TSE backend)
//! ```
//!
//! `tap::EnrollmentTap` is the only piece of this crate that runs on the
//! realtime mic pipeline thread, and it does the least possible amount of
//! work there (a toggle check plus, when active, a plain non-blocking ring-
//! buffer push) - see its docs for why no lock is needed. Everything else
//! (`session::EnrollmentSession`'s recording loop, VAD, and the call into
//! `SpeakerEncoder::embed`) runs on a dedicated background thread spawned
//! per recording attempt, never on the audio thread.

mod error;
mod metrics;
mod sentence;
mod session;
mod state;
mod tap;
mod validate;

pub use error::EnrollmentError;
pub use metrics::EnrollmentMetrics;
pub use sentence::DEFAULT_REFERENCE_SENTENCE;
pub use session::{EnrollmentSession, MAX_RECORDING_SECS};
pub use state::EnrollmentState;
pub use tap::{new_pair, EnrollmentTap};
pub use validate::{clipping_ratio, rms, speech_frame_ratio, validate, ValidationThresholds};
