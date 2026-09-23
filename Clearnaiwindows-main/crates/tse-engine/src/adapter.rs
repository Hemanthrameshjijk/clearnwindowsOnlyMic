//! Bridges any `TseBackend` into a `dsp_core::Stage`, so it can be wrapped
//! by the same `dsp_core::WatchdogStage` primitive already used for Hush
//! (see `bvc-hush`), and slotted into `crate::router::BvcRouterStage`
//! alongside it.
//!
//! This is also where target-speaker bookkeeping lives: reading the current
//! profile from a `speaker_profile::TargetSpeakerStore`, noticing when it
//! changed (by `TargetSpeakerProfile::version`, not by wall-clock time or by
//! comparing embeddings - see that type's docs on why version comparison is
//! the realtime-safe way to detect this), and resetting the backend's
//! streaming state exactly then - never on every frame, and never missed
//! when it does change.

use dsp_core::{LatencyLog, Stage};
use speaker_profile::TargetSpeakerStore;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use crate::backend::{TseBackend, TseError};

/// Realtime-safe counters for the TSE path, mirroring
/// `dsp_core::FrameCounters`'s "plain atomic increments only" contract.
/// Exposed separately from `dsp_core::FrameCounters` (rather than added to
/// it) because these are TSE-specific outcomes that only make sense once a
/// TSE backend exists at all - adding them to the pipeline-wide counters
/// would make every non-TSE build/config carry meaningless always-zero
/// fields.
#[derive(Default)]
pub struct TseMetrics {
    /// Frames where the backend actually ran and returned `Ok`.
    pub frames_processed: AtomicU64,
    /// Frames skipped (passed through unmodified) because no target speaker
    /// profile was enrolled at the time.
    pub bypassed_no_profile: AtomicU64,
    /// Frames where the backend returned `Err` (backend unavailable or a
    /// real inference failure) - the frame was passed through unmodified.
    pub failures: AtomicU64,
}

/// Adapts a `TseBackend` into a `dsp_core::Stage`. Construct one, then wrap
/// it in a `dsp_core::WatchdogStage` with a timeout appropriate for a 10ms
/// realtime deadline (see this crate's top-level docs: *do not* reuse
/// Hush's 500ms watchdog timeout for this - that value is far too large to
/// represent an acceptable stall in a 10ms-budget stage; pick something
/// tight, e.g. tens of milliseconds, and treat even that as a last-resort
/// safety net, not a target).
pub struct TseStageAdapter<B: TseBackend> {
    backend: B,
    store: TargetSpeakerStore,
    last_profile_version: u64,
    metrics: Arc<TseMetrics>,
    /// Per-frame *inference* latency (only recorded for frames where the
    /// backend actually ran - a bypassed frame due to "no profile" isn't a
    /// meaningful latency sample for the model itself). Separate from the
    /// whole-pipeline `dsp_core::LatencyLog` already recorded once per frame
    /// in `engine.rs`'s mic pipeline loop - this one isolates just the TSE
    /// backend's own cost, which is what a P50/P90/P95/P99 claim about "TSE
    /// frame processing time" (see `docs/TSE_ENROLLMENT.md`) must be based
    /// on to mean anything.
    latency_log: Arc<LatencyLog>,
}

impl<B: TseBackend> TseStageAdapter<B> {
    pub fn new(backend: B, store: TargetSpeakerStore, latency_log: Arc<LatencyLog>) -> Self {
        Self {
            backend,
            store,
            last_profile_version: 0,
            metrics: Arc::new(TseMetrics::default()),
            latency_log,
        }
    }

    pub fn metrics(&self) -> Arc<TseMetrics> {
        self.metrics.clone()
    }
}

impl<B: TseBackend> Stage for TseStageAdapter<B> {
    fn name(&self) -> &'static str {
        self.backend.name()
    }

    fn process(&mut self, frame: &mut [f32]) {
        let profile = self.store.get_profile();
        let version = profile.as_ref().map(|p| p.version()).unwrap_or(0);
        if version != self.last_profile_version {
            // Target speaker changed (enrolled, re-enrolled, or cleared):
            // reset streaming state *before* handing over the new profile,
            // so the backend never runs even one frame conditioned on a
            // stale profile with leftover state from the previous one.
            self.backend.reset_stream_state();
            self.backend.set_target_speaker(profile.clone());
            self.last_profile_version = version;
        }

        if profile.is_none() {
            // Deterministic, honest bypass - never pretend to extract a
            // target speaker that was never enrolled (see crate docs / the
            // task's explicit "do not fake" requirement).
            self.metrics.bypassed_no_profile.fetch_add(1, Ordering::Relaxed);
            return;
        }

        let started = Instant::now();
        match self.backend.process_frame(frame) {
            Ok(()) => {
                self.metrics.frames_processed.fetch_add(1, Ordering::Relaxed);
                self.latency_log.record_ns(started.elapsed().as_nanos() as u64);
            }
            Err(TseError::NoTargetSpeaker) | Err(TseError::BackendUnavailable) => {
                self.metrics.bypassed_no_profile.fetch_add(1, Ordering::Relaxed);
            }
            Err(TseError::InferenceFailed(_)) => {
                self.metrics.failures.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use speaker_profile::{NormalizationKind, TargetSpeakerProfile};

    /// A fake backend for testing the adapter's bookkeeping, independent of
    /// whether any real model exists. Mutates the frame by adding a fixed
    /// offset only while it has a target speaker, and counts resets.
    struct FakeBackend {
        has_target: bool,
        reset_count: u32,
        fail_next: bool,
    }

    impl TseBackend for FakeBackend {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn set_target_speaker(&mut self, profile: Option<Arc<TargetSpeakerProfile>>) {
            self.has_target = profile.is_some();
        }
        fn process_frame(&mut self, frame: &mut [f32]) -> Result<(), TseError> {
            if self.fail_next {
                self.fail_next = false;
                return Err(TseError::InferenceFailed("synthetic".into()));
            }
            if !self.has_target {
                return Err(TseError::NoTargetSpeaker);
            }
            for s in frame.iter_mut() {
                *s += 1.0;
            }
            Ok(())
        }
        fn reset_stream_state(&mut self) {
            self.reset_count += 1;
        }
        fn latency_samples(&self) -> Option<usize> {
            Some(0)
        }
    }

    fn profile(tag: f32) -> TargetSpeakerProfile {
        TargetSpeakerProfile::new(vec![tag], "enc", 16_000, NormalizationKind::None).unwrap()
    }

    #[test]
    fn no_profile_bypasses_and_leaves_frame_untouched() {
        let store = TargetSpeakerStore::new();
        let backend = FakeBackend { has_target: false, reset_count: 0, fail_next: false };
        let mut adapter = TseStageAdapter::new(backend, store, Arc::new(LatencyLog::new(8)));
        let mut frame = [1.0f32, 2.0, 3.0];
        adapter.process(&mut frame);
        assert_eq!(frame, [1.0, 2.0, 3.0]);
        assert_eq!(adapter.metrics().bypassed_no_profile.load(Ordering::Relaxed), 1);
        assert_eq!(adapter.metrics().frames_processed.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn enrolling_a_profile_resets_state_and_enables_processing() {
        let store = TargetSpeakerStore::new();
        let backend = FakeBackend { has_target: false, reset_count: 0, fail_next: false };
        let mut adapter = TseStageAdapter::new(backend, store.clone(), Arc::new(LatencyLog::new(8)));

        store.set_profile(profile(1.0));
        let mut frame = [1.0f32, 2.0];
        adapter.process(&mut frame);
        assert_eq!(frame, [2.0, 3.0], "backend must actually run once a profile is set");
        assert_eq!(adapter.metrics().frames_processed.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn reenrollment_resets_stream_state_exactly_once_per_change() {
        let store = TargetSpeakerStore::new();
        let backend = FakeBackend { has_target: false, reset_count: 0, fail_next: false };
        let mut adapter = TseStageAdapter::new(backend, store.clone(), Arc::new(LatencyLog::new(8)));

        store.set_profile(profile(1.0));
        let mut frame = [0.0f32];
        adapter.process(&mut frame); // picks up profile v1, resets once
        adapter.process(&mut frame); // same profile, no reset
        adapter.process(&mut frame); // same profile, no reset

        store.set_profile(profile(2.0)); // re-enrollment: new version
        adapter.process(&mut frame); // must reset again

        // 2 resets total: once for the initial enrollment, once for the
        // re-enrollment. (The very first construction has version 0 == "no
        // profile", so picking up v1 counts as a change.)
        assert_eq!(adapter.backend.reset_count, 2);
    }

    #[test]
    fn clearing_the_profile_falls_back_to_bypass() {
        let store = TargetSpeakerStore::new();
        let backend = FakeBackend { has_target: false, reset_count: 0, fail_next: false };
        let mut adapter = TseStageAdapter::new(backend, store.clone(), Arc::new(LatencyLog::new(8)));

        store.set_profile(profile(1.0));
        let mut frame = [1.0f32];
        adapter.process(&mut frame);
        assert_eq!(frame, [2.0]);

        store.clear_profile();
        let mut frame2 = [1.0f32];
        adapter.process(&mut frame2);
        assert_eq!(frame2, [1.0], "clearing the profile must fall back to pass-through, not keep using stale state");
    }

    #[test]
    fn inference_failure_is_counted_and_leaves_frame_untouched() {
        let store = TargetSpeakerStore::new();
        store.set_profile(profile(1.0));
        let backend = FakeBackend { has_target: true, reset_count: 0, fail_next: true };
        let mut adapter = TseStageAdapter::new(backend, store, Arc::new(LatencyLog::new(8)));
        let mut frame = [5.0f32];
        adapter.process(&mut frame);
        assert_eq!(frame, [5.0]);
        assert_eq!(adapter.metrics().failures.load(Ordering::Relaxed), 1);
    }
}
