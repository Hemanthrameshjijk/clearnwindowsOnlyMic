use std::sync::atomic::AtomicU64;

/// Benchmarking hooks for the enrollment path (section 23 of the task this
/// crate was built for: "do not claim the system is realtime until actual
/// measurements are collected"). These are wall-clock durations of
/// inherently non-realtime, one-shot operations (a whole recording session,
/// one embedding extraction) - not a per-frame P50/P90/P95/P99 series like
/// `dsp_core::LatencyLog` (that's what `tse_engine::TseStageAdapter` uses
/// for the actual per-10ms-frame TSE cost, a different and much more
/// frequent measurement). Plain atomics are enough here since enrollment
/// happens at most a few times per session, not every frame.
#[derive(Default)]
pub struct EnrollmentMetrics {
    pub attempts: AtomicU64,
    pub successes: AtomicU64,
    pub failures: AtomicU64,
    /// Wall-clock duration of the most recently completed recording, in
    /// milliseconds (from `start_recording` actually beginning to pull
    /// samples to the recording ending, before validation).
    pub last_recording_duration_ms: AtomicU64,
    /// Wall-clock duration of the most recent `SpeakerEncoder::embed` call,
    /// in milliseconds. `0` if validation itself failed before the encoder
    /// was ever called.
    pub last_embedding_extraction_ms: AtomicU64,
}
