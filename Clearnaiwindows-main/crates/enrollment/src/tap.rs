//! The realtime-side half of enrollment recording: a bounded tap the mic
//! pipeline thread feeds every frame, active only while a recording is in
//! progress.
//!
//! # Why this doesn't need a lock
//!
//! Unlike `engine::SwapProducer` (which the app crate uses for the
//! *device-switchable* roles, because a live device switch replaces the
//! whole ring buffer), the enrollment tap's ring buffer is created exactly
//! once, at the same time as every other stage in the pipeline (matching
//! this project's "everything is always constructed, a toggle just gates
//! whether it's used" rule - see `dsp_core::run_if_enabled`'s docs). The
//! `Producer<f32>` half is therefore owned outright by the realtime thread
//! for the entire process lifetime, with no `Option`/`Mutex` wrapper needed
//! at all: "is a recording in progress" is a separate, lock-free
//! `StageToggle` check, and pushing (when active) is a plain, uncontended
//! `Producer::push` call - strictly cheaper than even the narrow
//! `Mutex`-per-sample pattern `engine.rs` already accepts for device
//! switching.

use dsp_core::{make_ring_buffer, Consumer, Producer, StageToggle};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// Realtime-thread-owned half of the enrollment tap. Call
/// `push_frame_if_active` once per processed frame from the mic pipeline
/// loop, in the same place `driver_writer.push_frame(&frame)` is already
/// called in `engine.rs` - this is exactly one more optional sink for the
/// same final frame, not a new capture device.
pub struct EnrollmentTap {
    active: StageToggle,
    producer: Producer<f32>,
    dropped_samples: Arc<AtomicU64>,
}

impl EnrollmentTap {
    /// Pushes `frame` into the bounded recording buffer only while a
    /// recording is currently active (`EnrollmentSession::start_recording`
    /// having flipped the shared toggle on). Never blocks: once the bounded
    /// ring buffer (sized for `max_duration_secs` - see `new_pair`) fills up
    /// because the enrollment worker thread hasn't drained it fast enough,
    /// the remainder of the frame is dropped and counted, exactly like every
    /// other non-blocking sink in this project's mic pipeline
    /// (`engine.rs`'s `push_frame`/`push_frame_to_fixed`).
    #[inline]
    pub fn push_frame_if_active(&mut self, frame: &[f32]) {
        if !self.active.is_on() {
            return;
        }
        for &sample in frame {
            if self.producer.push(sample).is_err() {
                self.dropped_samples.fetch_add(1, Ordering::Relaxed);
                return;
            }
        }
    }
}

/// Builds one enrollment tap: the realtime-thread half (`EnrollmentTap`,
/// move into the mic pipeline closure), the worker-thread half
/// (`Consumer<f32>`, kept by `EnrollmentSession`), the shared `StageToggle`
/// both `EnrollmentTap` and `EnrollmentSession` hold a clone of, and a
/// shared dropped-sample counter for diagnostics/tests.
///
/// `max_duration_secs` bounds the ring buffer's capacity - this is the "hard
/// cap on enrollment duration" and the "no unbounded memory growth"
/// guarantee in one place: even if the worker thread never drained a single
/// sample, the realtime thread can never push more than this many seconds
/// of audio into it before every further sample is simply dropped (not
/// queued, not reallocated).
pub fn new_pair(max_duration_secs: f32) -> (EnrollmentTap, Consumer<f32>, StageToggle, Arc<AtomicU64>) {
    let (producer, consumer) = make_ring_buffer(max_duration_secs);
    let active = StageToggle::new(false);
    let dropped_samples = Arc::new(AtomicU64::new(0));
    let tap = EnrollmentTap {
        active: active.clone(),
        producer,
        dropped_samples: dropped_samples.clone(),
    };
    (tap, consumer, active, dropped_samples)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inactive_tap_drops_everything_silently() {
        let (mut tap, mut consumer, _active, dropped) = new_pair(1.0);
        tap.push_frame_if_active(&[1.0, 2.0, 3.0]);
        assert!(consumer.pop().is_err(), "nothing should have been pushed while inactive");
        assert_eq!(dropped.load(Ordering::Relaxed), 0, "an inactive tap doesn't count this as a drop");
    }

    #[test]
    fn active_tap_forwards_samples() {
        let (mut tap, mut consumer, active, _dropped) = new_pair(1.0);
        active.set(true);
        tap.push_frame_if_active(&[1.0, 2.0, 3.0]);
        assert_eq!(consumer.pop().unwrap(), 1.0);
        assert_eq!(consumer.pop().unwrap(), 2.0);
        assert_eq!(consumer.pop().unwrap(), 3.0);
    }

    #[test]
    fn full_ring_drops_remainder_and_counts_it_never_blocking() {
        let (mut tap, _consumer, active, dropped) = new_pair(0.0001); // tiny capacity, rounds up to 1 frame
        active.set(true);
        // Push far more samples than the ring can hold; must return promptly
        // (this test itself would hang if `push_frame_if_active` ever
        // blocked instead of dropping).
        let big_frame = vec![0.5f32; 100_000];
        tap.push_frame_if_active(&big_frame);
        assert!(dropped.load(Ordering::Relaxed) > 0, "overflow must be counted, not silently discarded");
    }

    #[test]
    fn toggling_active_off_again_stops_forwarding() {
        let (mut tap, mut consumer, active, _dropped) = new_pair(1.0);
        active.set(true);
        tap.push_frame_if_active(&[1.0]);
        active.set(false);
        tap.push_frame_if_active(&[2.0]);
        assert_eq!(consumer.pop().unwrap(), 1.0);
        assert!(consumer.pop().is_err(), "no second sample should have been pushed once inactive");
    }
}
