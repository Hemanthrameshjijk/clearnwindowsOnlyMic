//! `SpeakerBvcRouterStage`: the speaker-outbound path's own BVC backend
//! selector - deliberately a SEPARATE, smaller type from `BvcRouterStage`/
//! `BvcBackend`, not a reuse of them. It offers just generic
//! noise/background-voice suppression (Hush, DeepFilterNet). Same lock-free
//! `SharedPreset` dispatch pattern as `BvcRouterStage` - see that type's docs
//! for the general reasoning.

use dsp_core::{SharedPreset, Stage};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpeakerBvcBackend {
    Off,
    Hush,
    DeepFilter,
}

impl SpeakerBvcBackend {
    pub const ALL: [SpeakerBvcBackend; 3] = [SpeakerBvcBackend::Off, SpeakerBvcBackend::Hush, SpeakerBvcBackend::DeepFilter];

    pub const fn to_index(self) -> u8 {
        match self {
            SpeakerBvcBackend::Off => 0,
            SpeakerBvcBackend::Hush => 1,
            SpeakerBvcBackend::DeepFilter => 2,
        }
    }

    pub const fn from_index(index: u8) -> SpeakerBvcBackend {
        match index {
            1 => SpeakerBvcBackend::Hush,
            2 => SpeakerBvcBackend::DeepFilter,
            // Anything else (including the expected 0) degrades to Off,
            // matching `BvcBackend::from_index`'s same rule.
            _ => SpeakerBvcBackend::Off,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            SpeakerBvcBackend::Off => "Off",
            SpeakerBvcBackend::Hush => "Hush",
            SpeakerBvcBackend::DeepFilter => "DeepFilterNet",
        }
    }
}

/// Routes each frame to exactly one of two already-constructed stages (Off
/// is implicitly "call none of them"), selected by a live, lock-free
/// `SharedPreset`. Both backing stages are constructed up front at startup
/// regardless of which one is initially selected, same "every stage always
/// warm" rule `BvcRouterStage` follows.
pub struct SpeakerBvcRouterStage {
    backend: SharedPreset,
    hush: Box<dyn Stage>,
    deep_filter: Box<dyn Stage>,
}

impl SpeakerBvcRouterStage {
    /// `hush`/`deep_filter` are expected to already be wrapped in whatever
    /// isolation each backend needs (e.g. `dsp_core::WatchdogStage`) - this
    /// type only handles *which* one runs, not how each one is made safe to
    /// call. `selector` is owned by the caller (typically `EngineHandles`)
    /// and only cloned here, so the GUI's write side and this stage's read
    /// side observe the same atomic.
    pub fn new(selector: SharedPreset, hush: Box<dyn Stage>, deep_filter: Box<dyn Stage>) -> Self {
        Self { backend: selector, hush, deep_filter }
    }
}

impl Stage for SpeakerBvcRouterStage {
    fn name(&self) -> &'static str {
        "Speaker BVC (Off/Hush/DeepFilterNet router)"
    }

    fn process(&mut self, frame: &mut [f32]) {
        match SpeakerBvcBackend::from_index(self.backend.load()) {
            SpeakerBvcBackend::Off => {}
            SpeakerBvcBackend::Hush => self.hush.process(frame),
            SpeakerBvcBackend::DeepFilter => self.deep_filter.process(frame),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct AddOne;
    impl Stage for AddOne {
        fn name(&self) -> &'static str {
            "add-one"
        }
        fn process(&mut self, frame: &mut [f32]) {
            for s in frame.iter_mut() {
                *s += 1.0;
            }
        }
    }

    struct AddHundred;
    impl Stage for AddHundred {
        fn name(&self) -> &'static str {
            "add-hundred"
        }
        fn process(&mut self, frame: &mut [f32]) {
            for s in frame.iter_mut() {
                *s += 100.0;
            }
        }
    }

    #[test]
    fn backend_index_round_trips() {
        for backend in SpeakerBvcBackend::ALL {
            assert_eq!(SpeakerBvcBackend::from_index(backend.to_index()), backend);
        }
    }

    #[test]
    fn unrecognized_index_falls_back_to_off() {
        assert_eq!(SpeakerBvcBackend::from_index(200), SpeakerBvcBackend::Off);
    }

    #[test]
    fn off_leaves_frame_untouched() {
        let selector = SharedPreset::new(SpeakerBvcBackend::Off.to_index());
        let mut router = SpeakerBvcRouterStage::new(selector, Box::new(AddOne), Box::new(AddHundred));
        let mut frame = [1.0f32, 2.0];
        router.process(&mut frame);
        assert_eq!(frame, [1.0, 2.0]);
    }

    #[test]
    fn hush_runs_the_hush_stage_only() {
        let selector = SharedPreset::new(SpeakerBvcBackend::Hush.to_index());
        let mut router = SpeakerBvcRouterStage::new(selector, Box::new(AddOne), Box::new(AddHundred));
        let mut frame = [1.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [2.0]);
    }

    #[test]
    fn deep_filter_runs_the_deep_filter_stage_only() {
        let selector = SharedPreset::new(SpeakerBvcBackend::DeepFilter.to_index());
        let mut router = SpeakerBvcRouterStage::new(selector, Box::new(AddOne), Box::new(AddHundred));
        let mut frame = [1.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [101.0]);
    }

    #[test]
    fn switching_backend_live_via_the_selector_takes_effect_immediately() {
        let selector = SharedPreset::new(SpeakerBvcBackend::Off.to_index());
        let mut router = SpeakerBvcRouterStage::new(selector.clone(), Box::new(AddOne), Box::new(AddHundred));
        let mut frame = [0.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [0.0]);

        selector.store(SpeakerBvcBackend::Hush.to_index());
        router.process(&mut frame);
        assert_eq!(frame, [1.0]);

        selector.store(SpeakerBvcBackend::DeepFilter.to_index());
        router.process(&mut frame);
        assert_eq!(frame, [101.0]);
    }
}
