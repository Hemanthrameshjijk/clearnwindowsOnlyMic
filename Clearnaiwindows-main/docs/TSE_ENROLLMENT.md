# Target-speaker reference enrollment and TSE architecture

This documents the speaker-reference-enrollment subsystem and the BVC
backend router (Off / Hush / Target Speaker Extraction). Like this
project's other docs, it says plainly what's real, what's wired but
inert, and what does not exist yet.

**Enrollment itself, including a real speaker-embedding model, is fully
real and verified end-to-end.** Target Speaker Extraction (the actual
live streaming separation of a target speaker from a mixture) is *not* -
no real streaming TSE model was found, so `UnavailableTse` remains the
only backend and TSE always passes audio through unmodified. Nothing
below should be read as a claim that live TSE works.

## What's real vs. not, in one table

| Piece | Status |
|---|---|
| `speaker-profile` crate (`TargetSpeakerProfile`, `TargetSpeakerStore`, `SpeakerEncoder` trait) | Real, tested, portable. |
| `enrollment` crate (recording tap, VAD/quality checks, state machine, worker thread) | Real, tested, portable. |
| `speaker-encoder` crate (`WeSpeakerEncoder`) | **Real.** A bundled ONNX WeSpeaker ResNet34 (LM) speaker-verification model, run via `tract-onnx`, verified end-to-end (see below). |
| `tse-engine` crate (`BvcRouterStage`, `TseBackend` trait, `TseStageAdapter`) | Real, tested, portable. |
| Streaming TSE model | **Not integrated.** `tse_engine::UnavailableTse` is the only backend shipped; it always returns `TseError::BackendUnavailable`. |
| GUI (backend picker, enrollment controls) | Real, wired to the above. Selecting "Target Speaker (TSE)" is a real, live option - it just always results in pass-through audio today, honestly reported via the "unavailable" note in the BVC section. |
| Realtime pipeline order (`AutoGain -> RNNoise -> BVC router -> Studio`) | Unchanged. The router replaces the old direct `HushBvcStage` slot with zero change to `spawn_mic_pipeline`'s loop shape. |

## The speaker encoder: what it is and how it was chosen

The task that introduced this subsystem suggested ECAPA-TDNN specifically.
Three real candidates were tried, in order, each verified by actually
building and running code against the real file - not by reading a model
card and assuming it would work:

1. **Literal ECAPA-TDNN**: no `.onnx` asset by that name exists in
   `k2-fsa/sherpa-onnx`'s speaker-recognition-models release (checked via
   the real GitHub Releases API). Informal ECAPA-TDNN ONNX re-uploads exist
   on Hugging Face from individual accounts with no stable maintenance
   signal - not adopted, per this project's "do not integrate something of
   unverified provenance" rule.
2. **CAM++** (`3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx`, from
   Alibaba's 3D-Speaker/ModelScope project, packaged by `k2-fsa/sherpa-onnx`)
   - real, stably hosted, downloaded and inspected. Rejected only after
   real, hands-on verification: a throwaway `tract-onnx` probe binary was
   built and run against the actual file. Under `tract-onnx` 0.19.16,
   model *loading* failed (two dynamic-shape symbols on the same
   `BatchNormalization` node couldn't be unified, even after pinning the
   known-1 batch dimension to a literal). Under `tract-onnx` 0.21.18,
   loading succeeded but *inference* failed inside CAM++'s
   context-aware-masking TDNN layer (an internal `Slice` op computed an
   out-of-range index). Two different real failures across two tract
   versions, against the real file - a genuine architecture/tooling
   incompatibility, not a config mistake.
3. **WeSpeaker ResNet34 (LM)** (`wespeaker_en_voxceleb_resnet34_LM.onnx`,
   WeSpeaker's own official checkpoint, packaged by the same
   `k2-fsa/sherpa-onnx` project) - a plain ResNet architecture. Verified
   end-to-end for real: downloaded, its embedded ONNX metadata read
   directly (`framework=wespeaker`, `sample_rate=16000`, `output_dim=256`,
   `normalize_samples=0`, no `feature_normalize_type`), loaded and run
   through `tract-onnx` via a probe binary, producing a finite,
   correctly-shaped `[1, 256]` embedding. **This is the model bundled
   today**, as `speaker_encoder::WeSpeakerEncoder`.

`speaker_profile::SpeakerEncoder` is architecture-agnostic (nothing in its
trait shape names ECAPA-TDNN), so this substitution needed no changes
outside the one new `speaker-encoder` crate.

### A real, separate `tract` version-resolution problem (and its fix)

`tract-onnx` 0.21.18 (needed to run WeSpeaker correctly - 0.19.16, already
in this workspace via `noise-engine`'s `deep_filter` dependency, fails on
this model's `Reduce<Prod>` op) pins `half = "=2.4.1"` *exactly*. Once
`speaker-encoder` became a real dependency of `app` (which also depends on
`iced`, whose `wgpu`/`naga` chain needs `half ^2.5`), `cargo` could not
resolve the workspace at all - a real, reproducible deadlock, not a
hypothetical one. `tract-onnx` 0.23.7 relaxes this to `half = ">=2.4,<3.0"`
(confirmed against its real `Cargo.toml`) and was re-verified (not
assumed) to load/run the exact same bundled file with byte-identical
output to 0.21.18. `speaker-encoder` uses 0.23.7; `deep_filter`'s
(currently dead-code, unreachable) path still resolves its own 0.19.16
independently, coexisting in the same workspace without collision.

## No streaming TSE model is bundled

Before writing any inference code, the specific model this task named as a
candidate - `penta2himajin/tse-conv-tasnet-48k` - was checked directly
against the Hugging Face Hub: a direct fetch of that repository returns an
authentication/not-found response, and a Hub search for "tse-conv-tasnet" /
"conv-tasnet" returns no match for that uploader or name. **The named
candidate does not exist publicly.** No other real, causal, streaming,
publicly-downloadable TSE checkpoint was found either - unlike speaker
embedding (a mature, widely-served model class), streaming TSE with a
verified pretrained ONNX export does not appear to exist in a form this
project could safely bundle. There is therefore nothing real to download,
inspect (ONNX input/output names, tensor shapes, embedding dimension, state
tensors, required pre/post-processing), or wire up for the TSE side.
`tse_engine::UnavailableTse` remains the only backend, honestly reporting
`TseError::BackendUnavailable` rather than faking separation with plain
noise suppression.

**Bottom line: enrollment runs fully end-to-end today - recording, VAD,
quality checks, a real speaker-embedding model, the state machine, the
store, the GUI.** Only the live TSE step (using that enrolled embedding to
actually separate a target speaker from a mixture, frame by frame) remains
unavailable, for lack of a real model to run. Wiring one in later means
writing one new `TseBackend` implementation and passing it into
`engine.rs`'s TSE-stage construction - no other code in this subsystem
needs to change.

## Architecture

```text
                    ClearNAI
                       |
          +------------+------------+
          |                         |
   Enrollment Mode              Call Mode
          |                         |
   Physical Mic (tap)           Physical Mic
          |                         |
   enrollment::EnrollmentTap    AutoGain
   (realtime thread, lock-free)     |
          |                      RNNoise
   enrollment::EnrollmentSession    |
   (background thread)          tse_engine::BvcRouterStage
          |                       (selected by handles.bvc_backend)
   VAD / quality (validate.rs)      |-- Off:  pass-through
          |                         |-- Hush: bvc_hush::HushBvcStage
   speaker_encoder::WeSpeakerEncoder|         (watchdog, 500ms)
   (real ONNX model, via tract)     |-- TSE:  tse_engine::TseStageAdapter<UnavailableTse>
          |                         |         (watchdog, 20ms - always bypasses, no real model)
          |                            |
   TargetSpeakerProfile                Studio
          |                            |
   speaker_profile::TargetSpeakerStore  Virtual Microphone
          `------------------------------> (read every frame by the TSE
                                             stage when backend = TSE)
```

### Realtime-safety notes

- `EnrollmentTap::push_frame_if_active` runs on the mic pipeline thread. It
  is a single `Relaxed` atomic load (a `dsp_core::StageToggle`) plus,
  only while a recording is active, a plain non-blocking `rtrb::Producer`
  push into a ring buffer sized once at startup (`enrollment::MAX_RECORDING_SECS`,
  12 seconds) - never a lock, never an allocation, never a blocking call.
  Everything else (the recording loop, VAD, the encoder call) runs on a
  background thread spawned per recording attempt by
  `enrollment::EnrollmentSession::start_recording`.
- `tse_engine::TseStageAdapter` reads the active `TargetSpeakerProfile` via
  `speaker_profile::TargetSpeakerStore::get_profile`, which is an
  `arc_swap::ArcSwapOption` - a wait-free read, never a `Mutex`. A
  re-enrollment mid-call is picked up on the very next frame (compared by
  `TargetSpeakerProfile::version`, a plain integer, not by comparing
  embeddings or touching a wall clock) and triggers exactly one
  `TseBackend::reset_stream_state` call before the new profile is used.
- The TSE backend gets its own `dsp_core::WatchdogStage`, but with a 20ms
  timeout, not Hush's 500ms - see `engine.rs`'s `TSE_WATCHDOG_TIMEOUT`
  constant. A 500ms tolerance was calibrated for a closed-source DLL of
  unknown latency; a 10ms-budget realtime stage should never be given that
  much rope by default.

## Enrollment state machine

```text
Idle -> Preparing -> Recording -> Validating -> ExtractingEmbedding -> Ready
                          |             |                |
                          `-------------+----------------+--> Failed(EnrollmentError)
```

Recording stops on whichever comes first:
1. A manual stop (`EnrollmentSession::stop_recording`, the GUI's "Stop
   recording" button).
2. `enrollment::session::AUTO_STOP_TRAILING_SILENCE_SECS` (1.5s) of
   near-silence after at least a second of speech-level energy has already
   been seen - a plain per-sample energy threshold, not a real VAD model.
3. `enrollment::MAX_RECORDING_SECS` (12 seconds) - the hard ceiling that
   also bounds the tap's ring-buffer memory.

`EnrollmentError` variants: `NoSpeech`, `TooShort`, `TooQuiet`, `Clipped`,
`InvalidEmbedding`, `ModelUnavailable`, `RecordingError`. See
`crates/enrollment/src/validate.rs` for the exact (deliberately lenient)
thresholds and the order failures are checked in.

## BVC backend selection

`tse_engine::BvcBackend` is `Off | Hush | Tse`, persisted as
`Settings::mic_bvc_backend` (a plain `u8` index, same convention as
`mic_studio_preset`). The GUI's single "Backend" picker in the BVC section
drives both the backend selection and the existing `mic_bvc` on/off gate:
picking "Off" turns the gate off; picking "Hush" or "TSE" turns it on *only
if that backend is actually available* (`mic_bvc_available`/
`tse_available` respectively) - selecting an unavailable backend is
remembered but never silently treated as "on."

## Configuration added

- `Settings::mic_bvc_backend: u8` - Off/Hush/TSE selection, default Hush
  (matches this project's pre-TSE behavior).
- `Settings::reference_sentence: String` - default
  `enrollment::DEFAULT_REFERENCE_SENTENCE`.

Neither the raw reference recording nor the speaker embedding itself is
persisted to disk - `TargetSpeakerStore` lives in memory for the current
process only (see this task's explicit "do not automatically persist
biometric speaker embeddings permanently" instruction). Re-enrollment is
required after every restart. `TargetSpeakerStore`'s
`set_profile`/`get_profile`/`clear_profile`/`has_profile` shape already
supports adding real persistence later without changing its API.

## Testing

Every new crate (`speaker-profile`, `speaker-encoder`, `enrollment`,
`tse-engine`) has unit tests that run with plain `cargo test` on any
platform - none require real audio hardware or a real DLL.
`speaker-encoder`'s tests *do* exercise a real ONNX model and a real
`tract-onnx` inference run (no mocking) - the only thing not exercised by
any test is the model's actual speaker-verification accuracy on real human
voices, which needs a real speech corpus this environment doesn't have.
Notably:
- `speaker-encoder`'s tests load the real bundled model and run real
  inference on synthetic tones, checking output shape (256-dim), unit L2
  norm, finiteness, and that different inputs produce different embeddings
  (not a correctness/accuracy check - see crate docs on why that's not
  possible here).
- `enrollment`'s tests drive the *real* `EnrollmentSession` state machine
  end-to-end with synthetic tone/silence signals pushed through the real
  `EnrollmentTap`, including a real background thread and a real (fake, but
  trait-object-based) `SpeakerEncoder` - not a mock of the session itself.
- `tse-engine`'s adapter tests verify the realtime-thread-facing contract
  directly: no-profile bypass, re-enrollment causing exactly one
  `reset_stream_state` call, a cleared profile falling back to pass-through,
  and a failing backend never partially mutating the frame.
- No test claims to measure real TSE latency, CPU%, or background-speaker
  leakage - those require a real streaming TSE model and real hardware,
  neither of which exist in this build. See the "Benchmarking" section
  below.

## Benchmarking hooks (and what they do not yet prove)

- `enrollment::EnrollmentMetrics`: `attempts`/`successes`/`failures` counts,
  plus the most recent recording's wall-clock duration and the most recent
  `SpeakerEncoder::embed` call's wall-clock duration, in milliseconds.
- `tse_engine::TseMetrics`: `frames_processed`/`bypassed_no_profile`/
  `failures` counts.
- `tse_engine::TseStageAdapter` also feeds a dedicated
  `dsp_core::LatencyLog` (separate from the whole-pipeline one already
  recorded in `engine.rs`) with the TSE backend's own per-frame processing
  time, so a real backend's P50/P90/P95/P99/max can be computed with the
  existing `dsp_core::compute_percentiles` the moment a real model exists.

**None of this has been run against real hardware or a real model.** Since
the only TSE backend that exists always returns `BackendUnavailable`
instantly, any latency numbers it would produce today are meaningless (they
measure a no-op, not inference) and are deliberately not reported anywhere
as if they were real performance data.

## What still needs real microphone/model testing before this is a working feature

1. A real, verified streaming/causal TSE model with confirmed chunk size,
   state-tensor shapes, and embedding format, integrated as a new
   `TseBackend` implementation - none was found to exist publicly (see
   above); this is the single biggest remaining gap.
2. Real-microphone testing of the enrollment recording flow and the speaker
   encoder together (does the auto-stop silence threshold feel right for a
   real room; does WeSpeaker's embedding actually separate real speakers
   well enough for a downstream TSE model to use, once one exists - the
   *inference pipeline* is verified, the *embedding quality on real voices*
   is not, since that requires a real speech corpus this environment
   doesn't have).
3. Real measurement of TSE frame-processing latency on the actual target
   Windows CPU, once a real streaming TSE backend exists - only then does
   `TSE_WATCHDOG_TIMEOUT` (currently a design choice, not a measured value)
   deserve to be tuned against real data. (The speaker encoder's own
   inference latency, being an off-realtime-thread, once-per-enrollment
   operation, is not latency-critical the same way and has not been
   benchmarked either.)
4. Real target-speaker-preservation / background-speaker-leakage
   measurement with actual recordings of a target speaker plus one or more
   background speakers - impossible without a real TSE model (item 1) to
   measure in the first place.
