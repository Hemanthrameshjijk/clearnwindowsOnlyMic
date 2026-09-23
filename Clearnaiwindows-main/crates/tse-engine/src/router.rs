//! `BvcRouterStage`: the generic BVC backend-selection abstraction the
//! project did not previously have (see the earlier architecture audit's
//! §8 finding: `HushBvcStage` was used directly as a `Box<dyn Stage>`, with
//! no registry allowing more than one backend). This is a `dsp_core::Stage`
//! itself, so it drops into `engine.rs`'s pipeline exactly where the old
//! `mic_bvc_stage: Box<dyn Stage>` used to sit, with zero changes required
//! to `spawn_mic_pipeline`'s loop.
//!
//! Dispatch is a single `Relaxed` atomic load (`dsp_core::SharedPreset`,
//! the same primitive already used for the Studio-preset selector) plus a
//! call into whichever already-constructed stage is selected - no lock, no
//! allocation, matching every other realtime dispatch point in this
//! project.

use dsp_core::{SharedPreset, Stage};

/// Which BVC backend is currently selected. Mirrors `studio_dsp::StudioPreset`
/// / `SharedPreset`'s existing "small enum encoded as a `u8` index" pattern
/// (see `app::shared::studio_preset_to_index`) so the GUI/settings layer can
/// keep using the same plain-integer persistence approach it already does
/// for the Studio preset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BvcBackend {
    Off,
    Hush,
    /// Real, official DeepFilterNet3 (via the `deep_filter` crate - see
    /// `noise_engine::load_bundled_deep_filter`'s docs for its verified
    /// provenance: the upstream project's own bundled ONNX export, not a
    /// third-party's unattributed copy). Generic noise/other-voice
    /// suppression, same category as Hush - it has no speaker-targeting of
    /// its own, just a different model/architecture.
    DeepFilter,
    Tse,
    /// Speaker-targeted extraction backend based on `Komalpreet2809/Vanta`
    /// (Conv-TasNet + ECAPA-TDNN), converted to a causal/streaming form and
    /// REAL-Rust-verified for correctness and speed (RTF ~0.8x measured) -
    /// see `research/tse/candidates/vanta-causal{,-rs}/` and
    /// `bvc_vanta`'s crate docs. Real, trained separator + encoder/decoder
    /// weights are now bundled and load successfully, but the model is
    /// honestly weak (real measured eval: +1.07dB average improvement,
    /// -7.10dB worst case, PESQ 0.84) - construction/availability
    /// (`app`'s `mic_vanta_available`) reflects whether the backend loads
    /// and runs, not whether it's production quality. Falls back to a
    /// no-op, exactly like Hush/DeepFilter/Tse do, if construction ever
    /// fails (e.g. `onnxruntime.dll` missing for the ECAPA encoder).
    Vanta,
}

impl BvcBackend {
    pub const ALL: [BvcBackend; 5] =
        [BvcBackend::Off, BvcBackend::Hush, BvcBackend::DeepFilter, BvcBackend::Tse, BvcBackend::Vanta];

    pub const fn to_index(self) -> u8 {
        match self {
            BvcBackend::Off => 0,
            BvcBackend::Hush => 1,
            BvcBackend::Tse => 2,
            BvcBackend::DeepFilter => 3,
            BvcBackend::Vanta => 4,
        }
    }

    pub const fn from_index(index: u8) -> BvcBackend {
        match index {
            1 => BvcBackend::Hush,
            2 => BvcBackend::Tse,
            3 => BvcBackend::DeepFilter,
            4 => BvcBackend::Vanta,
            // Anything else (including the expected 0) degrades to Off,
            // matching `studio_preset_from_index`'s "unrecognized index
            // falls back to a safe default rather than panicking" rule -
            // here the safe default is "do not run any BVC backend."
            _ => BvcBackend::Off,
        }
    }

    pub const fn name(self) -> &'static str {
        match self {
            BvcBackend::Off => "Off",
            BvcBackend::Hush => "Hush",
            BvcBackend::DeepFilter => "DeepFilterNet",
            BvcBackend::Tse => "Target Speaker (TSE)",
            BvcBackend::Vanta => "Vanta (experimental)",
        }
    }
}

/// Routes each frame to exactly one of three already-constructed stages
/// (Off is implicitly "call none of them"), selected by a live, lock-free
/// `SharedPreset`. All three backing stages are constructed up front at
/// startup regardless of which one is initially selected - matching this
/// project's "every stage always warm" rule (see `dsp_core::run_if_enabled`'s
/// docs) - so switching backends live is just a pointer/index flip, never a
/// pipeline rebuild.
pub struct BvcRouterStage {
    backend: SharedPreset,
    hush: Box<dyn Stage>,
    deep_filter: Box<dyn Stage>,
    tse: Box<dyn Stage>,
    /// Always a `NoOpStage` in practice today - see `BvcBackend::Vanta`'s
    /// docs on why this must never run real (untrained) inference.
    vanta: Box<dyn Stage>,
}

impl BvcRouterStage {
    /// `hush`, `deep_filter`, `tse`, and `vanta` are expected to already be
    /// wrapped in whatever isolation each backend needs (e.g.
    /// `dsp_core::WatchdogStage` with a timeout appropriate to that backend
    /// - 500ms for Hush's closed-source DLL call, a much tighter bound for
    /// TSE - see `crate::adapter`'s docs). This type only handles *which*
    /// one runs, not how each one is made safe to call.
    ///
    /// `selector` is owned by the caller (typically `EngineHandles`, which
    /// the GUI writes to live) and only cloned here - constructing this
    /// stage from an *existing* `SharedPreset` rather than building a fresh
    /// one internally is what lets the GUI's write side and this stage's
    /// read side observe the same atomic, exactly like
    /// `mic_studio_preset`/`StudioStage`'s existing live-preset-swap wiring
    /// in `engine.rs`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        selector: SharedPreset,
        hush: Box<dyn Stage>,
        deep_filter: Box<dyn Stage>,
        tse: Box<dyn Stage>,
        vanta: Box<dyn Stage>,
    ) -> Self {
        Self {
            backend: selector,
            hush,
            deep_filter,
            tse,
            vanta,
        }
    }
}

impl Stage for BvcRouterStage {
    fn name(&self) -> &'static str {
        "BVC (Off/Hush/DeepFilterNet/TSE/Vanta router)"
    }

    fn process(&mut self, frame: &mut [f32]) {
        match BvcBackend::from_index(self.backend.load()) {
            BvcBackend::Off => {}
            BvcBackend::Hush => self.hush.process(frame),
            BvcBackend::DeepFilter => self.deep_filter.process(frame),
            BvcBackend::Tse => self.tse.process(frame),
            BvcBackend::Vanta => self.vanta.process(frame),
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

    struct AddTen;
    impl Stage for AddTen {
        fn name(&self) -> &'static str {
            "add-ten"
        }
        fn process(&mut self, frame: &mut [f32]) {
            for s in frame.iter_mut() {
                *s += 10.0;
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

    struct AddThousand;
    impl Stage for AddThousand {
        fn name(&self) -> &'static str {
            "add-thousand"
        }
        fn process(&mut self, frame: &mut [f32]) {
            for s in frame.iter_mut() {
                *s += 1000.0;
            }
        }
    }

    #[test]
    fn backend_index_round_trips() {
        for backend in BvcBackend::ALL {
            assert_eq!(BvcBackend::from_index(backend.to_index()), backend);
        }
    }

    #[test]
    fn unrecognized_index_falls_back_to_off() {
        assert_eq!(BvcBackend::from_index(200), BvcBackend::Off);
    }

    #[test]
    fn off_leaves_frame_untouched() {
        let selector = SharedPreset::new(BvcBackend::Off.to_index());
        let mut router = BvcRouterStage::new(
            selector,
            Box::new(AddOne),
            Box::new(AddHundred),
            Box::new(AddTen),
            Box::new(AddThousand),
        );
        let mut frame = [1.0f32, 2.0];
        router.process(&mut frame);
        assert_eq!(frame, [1.0, 2.0]);
    }

    #[test]
    fn hush_runs_the_hush_stage_only() {
        let selector = SharedPreset::new(BvcBackend::Hush.to_index());
        let mut router = BvcRouterStage::new(
            selector,
            Box::new(AddOne),
            Box::new(AddHundred),
            Box::new(AddTen),
            Box::new(AddThousand),
        );
        let mut frame = [1.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [2.0]);
    }

    #[test]
    fn tse_runs_the_tse_stage_only() {
        let selector = SharedPreset::new(BvcBackend::Tse.to_index());
        let mut router = BvcRouterStage::new(
            selector,
            Box::new(AddOne),
            Box::new(AddHundred),
            Box::new(AddTen),
            Box::new(AddThousand),
        );
        let mut frame = [1.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [11.0]);
    }

    #[test]
    fn deep_filter_runs_the_deep_filter_stage_only() {
        let selector = SharedPreset::new(BvcBackend::DeepFilter.to_index());
        let mut router = BvcRouterStage::new(
            selector,
            Box::new(AddOne),
            Box::new(AddHundred),
            Box::new(AddTen),
            Box::new(AddThousand),
        );
        let mut frame = [1.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [101.0]);
    }

    #[test]
    fn vanta_runs_the_vanta_stage_only() {
        let selector = SharedPreset::new(BvcBackend::Vanta.to_index());
        let mut router = BvcRouterStage::new(
            selector,
            Box::new(AddOne),
            Box::new(AddHundred),
            Box::new(AddTen),
            Box::new(AddThousand),
        );
        let mut frame = [1.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [1001.0]);
    }

    #[test]
    fn switching_backend_live_via_the_selector_takes_effect_immediately() {
        let selector = SharedPreset::new(BvcBackend::Off.to_index());
        let mut router = BvcRouterStage::new(
            selector.clone(),
            Box::new(AddOne),
            Box::new(AddHundred),
            Box::new(AddTen),
            Box::new(AddThousand),
        );
        let mut frame = [0.0f32];
        router.process(&mut frame);
        assert_eq!(frame, [0.0]);

        selector.store(BvcBackend::Hush.to_index());
        router.process(&mut frame);
        assert_eq!(frame, [1.0]);

        selector.store(BvcBackend::DeepFilter.to_index());
        router.process(&mut frame);
        assert_eq!(frame, [101.0]);

        selector.store(BvcBackend::Tse.to_index());
        router.process(&mut frame);
        assert_eq!(frame, [111.0]);

        selector.store(BvcBackend::Vanta.to_index());
        router.process(&mut frame);
        assert_eq!(frame, [1111.0]);
    }
}
