//! Lock-free "current target speaker" cell.
//!
//! # Why not a `Mutex<Option<TargetSpeakerProfile>>`
//!
//! The realtime TSE stage (`tse_engine`) reads whatever profile is
//! currently active on every 10ms audio frame. This project's realtime rule
//! (already followed by `dsp_core::StageToggle`/`SharedPreset`) is that a
//! rarely-changed value published from a non-realtime thread (here: the
//! enrollment worker thread, finishing a multi-second embedding extraction)
//! to a realtime one must be read via a lock-free atomic swap, never a
//! `Mutex` - a `Mutex` guarantees mutual exclusion, not bounded wait time,
//! and the audio thread cannot tolerate blocking on *any* other thread for
//! an unbounded duration, however "usually fast" that other thread's
//! critical section is in practice.
//!
//! `TargetSpeakerStore` wraps `arc_swap::ArcSwapOption`, which gives
//! wait-free reads (`load` never blocks, ever) and an write side that is
//! itself lock-free (an atomic pointer swap, not a lock) - a strictly
//! stronger guarantee than `StageToggle`'s plain `AtomicBool` needs, because
//! the payload here is a whole profile, not a single bool/u8.

use crate::TargetSpeakerProfile;
use arc_swap::ArcSwapOption;
use std::sync::Arc;

/// Shared, clonable handle to the currently-enrolled target speaker (or
/// none). Cloning is cheap (an `Arc` clone) - the enrollment subsystem, the
/// realtime TSE stage, and the GUI all hold their own clone of the same
/// underlying cell, exactly like `dsp_core::StageToggle`'s existing
/// `Arc`-backed-handle pattern.
#[derive(Clone)]
pub struct TargetSpeakerStore(Arc<ArcSwapOption<TargetSpeakerProfile>>);

impl TargetSpeakerStore {
    /// Starts with no target speaker enrolled - the ordinary, expected state
    /// before the user has ever recorded a reference (or after `clear_profile`).
    pub fn new() -> Self {
        Self(Arc::new(ArcSwapOption::from(None)))
    }

    /// Replaces the active profile. Called from the enrollment worker thread
    /// once embedding extraction finishes - never from the realtime audio
    /// thread. Old readers that already loaded the previous `Arc` keep a
    /// valid (now-stale) reference to it until they drop it; this is a
    /// normal, harmless snapshot-read race, not a bug (matches
    /// `dsp_core::StageToggle`'s "audible-frame-granularity staleness is
    /// fine" tolerance).
    pub fn set_profile(&self, profile: TargetSpeakerProfile) {
        self.0.store(Some(Arc::new(profile)));
    }

    /// Wait-free: safe to call from the realtime audio thread every frame.
    /// Returns `None` if no target speaker is currently enrolled.
    #[inline]
    pub fn get_profile(&self) -> Option<Arc<TargetSpeakerProfile>> {
        self.0.load_full()
    }

    /// Removes the active profile (if any), e.g. "Clear Reference" in the GUI.
    pub fn clear_profile(&self) {
        self.0.store(None);
    }

    /// `true` if a target speaker is currently enrolled. Equivalent to
    /// `get_profile().is_some()` but avoids cloning the `Arc` when the
    /// caller only needs the boolean (e.g. the GUI's "Enrolled" indicator).
    #[inline]
    pub fn has_profile(&self) -> bool {
        self.0.load().is_some()
    }
}

impl Default for TargetSpeakerStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NormalizationKind;

    fn make_profile(tag: f32) -> TargetSpeakerProfile {
        TargetSpeakerProfile::new(vec![tag, tag, tag], "test-encoder", 16_000, NormalizationKind::L2)
            .expect("valid embedding")
    }

    #[test]
    fn starts_with_no_profile() {
        let store = TargetSpeakerStore::new();
        assert!(!store.has_profile());
        assert!(store.get_profile().is_none());
    }

    #[test]
    fn set_then_get_round_trips() {
        let store = TargetSpeakerStore::new();
        store.set_profile(make_profile(1.0));
        assert!(store.has_profile());
        let profile = store.get_profile().expect("profile was just set");
        assert_eq!(profile.embedding(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn set_replaces_the_previous_profile_atomically() {
        let store = TargetSpeakerStore::new();
        store.set_profile(make_profile(1.0));
        let first = store.get_profile().unwrap();
        store.set_profile(make_profile(2.0));
        let second = store.get_profile().unwrap();
        assert_eq!(first.embedding(), &[1.0, 1.0, 1.0], "previously-loaded Arc must remain valid/unchanged");
        assert_eq!(second.embedding(), &[2.0, 2.0, 2.0]);
        assert!(second.version() > first.version());
    }

    #[test]
    fn clear_removes_the_profile() {
        let store = TargetSpeakerStore::new();
        store.set_profile(make_profile(1.0));
        assert!(store.has_profile());
        store.clear_profile();
        assert!(!store.has_profile());
        assert!(store.get_profile().is_none());
    }

    #[test]
    fn clones_share_the_same_underlying_cell() {
        let a = TargetSpeakerStore::new();
        let b = a.clone();
        a.set_profile(make_profile(5.0));
        assert!(b.has_profile(), "a clone must observe writes made through another clone");
    }
}
