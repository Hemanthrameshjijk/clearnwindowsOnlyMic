# TSE Integration Plan (Not Yet Executed)

This describes the concrete engineering work required to move from "a
real checkpoint that loads and runs whole-utterance CPU inference in a
Python research environment" (verified, see `docs/TSE_SELECTED_MODEL.md`)
to "a real `tse_engine::TseBackend` in ClearNAI's production Rust
pipeline" (not yet done). No production code has been changed as part of
this research phase, per the brief's explicit instruction to research
before integrating.

## Step 0 — License resolution (blocking, do first)

Contact the REAL-TSE/wesep-real-tse repository owners (or the Real-TSE
Challenge organizers) to get an explicit, written answer on whether
`spk_emb_causal_100`'s checkpoint weights may be used in a commercial
product. Everything below is engineering effort that would be wasted if
the answer is no.

## Step 1 — Real chunked/stateful re-implementation (the biggest gap)

The reference `TSE_BSRNN_SPK.forward()` (real source, read in this
session - see `research/tse/candidates/wesep-real-tse/wesep/models/tse_bsrnn_spk.py`)
STFTs the whole mixture, runs one separator pass over the whole
time axis, and ISTFTs the whole thing back. To get genuine 10ms/480-sample
(or its 16kHz-native 160-sample) streaming:

**Step 1a — decouple the speaker embedding from every call (confirmed real
issue, not hypothetical):** `forward(self, mix, enroll)` takes the *raw
enrollment waveform* as an argument, and internally calls
`self.spk_ft.spkemb.compute(wav_enroll)` (the full ECAPA-TDNN forward pass)
**on every single call** - there is no cached-embedding code path in the
reference implementation at all. This was confirmed directly by
`research/tse/benchmarks/benchmark_streaming.py`'s real measurement: even
the smallest processable buffer (40ms) took ~108ms wall-clock on this
CPU (single thread) - far more than 40ms of audio should cost for a
25M-param causal separator alone, consistent with a full ECAPA-TDNN
re-run happening on every call. Any real streaming re-implementation
**must** split this into (1) a one-time `compute_target_embedding(enroll)
-> [192]` call, done once at enrollment (exactly matching ClearNAI's
existing `speaker-encoder` pattern), and (2) a per-chunk
`process_frame(frame, cached_embedding, state) -> (output, next_state)`
call that never touches raw enrollment audio again. This is a concrete,
measured finding, not a generic streaming-engineering platitude.

1. Decompose the BSRNN separator's causal Conv1d/RNN layers so each can
   accept a small time-chunk plus its own carried-forward state and
   produce the next chunk plus updated state - this requires reading
   `wesep/modules/separator/bsrnn.py` in full (only skimmed in this
   research phase) and identifying every stateful sub-layer.
2. Verify the re-implementation is numerically equivalent to the reference
   whole-utterance forward pass on the same input (e.g. feed a synthetic
   signal both ways and confirm the outputs match to within floating-point
   tolerance) - this is the correctness check that must pass before any
   latency number from the chunked version can be trusted.
3. Only then measure real per-chunk P50/P95/P99 latency (this is what
   `research/tse/benchmarks/benchmark_streaming.py`'s current *naive*
   growing-buffer probe is a rough stand-in for - see that script's own
   header for why it is explicitly not this real measurement).

This is genuine ML-engineering work, not a config change, and is the
single largest risk item in this whole plan.

## Step 2 — Sample-rate path

ClearNAI's mic pipeline is 48kHz/480-sample native; this model is 16kHz.
Per the research brief: evaluate PATH A (native 48kHz model - not
available for this candidate) vs. PATH B (causal resampler in, 16kHz TSE,
causal resampler out). Concretely:

1. Implement (or reuse - `speaker-encoder`'s `resample.rs` already exists
   as a *non-causal*, whole-buffer FFT resampler for enrollment; a
   *causal*, low-latency resampler is a different, new component) a
   bounded-latency 48k→16k and 16k→48k resampling stage.
2. Measure its own added latency and CPU cost in isolation before
   combining it with the TSE model - do not assume it's negligible.

## Step 3 — Speaker embedding path (real, not a random projection)

This model needs a 192-dim ECAPA_TDNN_GLOB_c512 embedding, computed by the
submodule already confirmed loadable inside this exact checkpoint
(`wesep/modules/speaker/encoder.py::SpeakerEncoder`, wrapping WeSpeaker's
`get_speaker_model("ECAPA_TDNN_GLOB_c512")`). Two real options, in
preference order:

1. **Export just this submodule** (not the whole TSE model) to ONNX or
   TorchScript, and run it once per enrollment (exactly like ClearNAI's
   existing `speaker-encoder` crate already does for its WeSpeaker-ResNet34
   encoder) to produce a 192-dim embedding, cached in a *new*,
   TSE-specific profile type - **not** a reuse of the existing 256-dim
   `TargetSpeakerProfile`, since they are different embedding spaces for
   different purposes (existing: ClearNAI's own enrollment
   verification/UI: WeSpeaker-ResNet34-256; new: this TSE model's own
   conditioning: ECAPA-192).
2. Alternatively, separately source the standalone WeSpeaker
   `voxceleb_ECAPA512` checkpoint this model's training config references
   (`./wespeaker_models/voxceleb_ECAPA512/avg_model.pt`) and confirm its
   weights match what's baked into `avg_model.pt` (they should, if the
   encoder was frozen during TSE training - not yet confirmed either way
   in this session).

Do **not** attempt to bridge ClearNAI's existing 256-dim embedding to this
model's 192-dim space with a learned or random projection unless a
properly supervised, validated adapter is built with real training data -
per the brief's explicit instruction, this is not a first-prototype
option.

## Step 4 — Runtime: ONNX export or LibTorch binding

The project's own C++ runtime uses LibTorch (TorchScript `.zip`), not
ONNX. Two real paths, both unattempted so far:

1. **Attempt an ONNX export** (`torch.onnx.export`) of the chunked/stateful
   re-implementation from Step 1, with explicit `state_in_*`/`state_out_*`
   graph inputs/outputs. This preserves ClearNAI's existing `tract-onnx`-
   based Rust integration pattern (same as `speaker-encoder`'s
   `WeSpeakerEncoder`) - **verify the exported graph actually loads and
   runs correctly in `tract-onnx`**, the same way `speaker-encoder`'s CAM++
   attempt was verified to fail and WeSpeaker ResNet34 was verified to
   succeed, before assuming it works.
2. If ONNX export of the BSRNN/STFT/complex-mask operations proves
   impractical, the alternative is a Rust LibTorch binding (e.g. `tch-rs`)
   - a new, heavier runtime dependency this project does not currently
   have, and its own separate integration/vetting effort.

## Step 5 — `tse-engine` production wiring

Only after Steps 1-4 produce a working, measured, licensed, real-time
component:

1. Add a real `TseBackend` implementation (e.g. `RealTseBackend`) in
   `crates/tse-engine`, replacing `UnavailableTse` as the default when the
   model/runtime loads successfully - falling back to `UnavailableTse`
   (with an honest, specific unavailability reason surfaced to the GUI,
   exactly as the existing pattern already does for Hush/BVC) if it does
   not.
2. Wire a new TSE-specific enrollment path (Step 3) alongside, not instead
   of, the existing WeSpeaker-ResNet34 enrollment (which remains useful
   for ClearNAI's own profile verification/UI, per the brief).
3. Model/session construction happens once, at engine startup - never
   inside `process_frame()`. No allocations, no locks, no disk I/O, no
   logging on every frame in the realtime path - exactly the existing
   `tse_engine::TseStageAdapter`/`dsp_core::WatchdogStage` conventions
   already in place; this plan does not change that contract, only fills
   in a real backend behind it.
4. Re-run and extend the existing `tse-engine` test suite (initialization,
   missing model, invalid model, wrong sample rate, wrong embedding size,
   state init/progression/reset, profile replacement, bounded processing,
   no NaNs/infs, backend selection, unavailable-backend reporting) against
   the real backend, in addition to the existing tests against
   `UnavailableTse`.

## Step 6 — Real quality benchmarking

Before any claim that this "works," construct real test mixtures (real
speech, not synthetic tones) covering the scenarios listed in the research
brief (target only; target + 1/2 background speakers; target + noise;
overlapping speech; target/background start-stop; silence; near/far-end;
reverberant if available) and measure SI-SDR, SI-SDRi, target-speaker
preservation, and background-speaker suppression for real. This requires
sourcing or recording a real multi-speaker speech corpus, which was not
available in this research session.

## Honest summary of remaining work

Every step above is real, scoped, unstarted engineering work - not a
checklist that's "basically done." The research phase's job was to find
out whether there is anything real worth doing this work for. The answer:
yes, conditionally (pending Step 0's license answer) - REAL-TSE/WeSep
`spk_emb_causal_100` is a real, correctly-loading, CPU-runnable causal TSE
checkpoint, which is more than any other candidate investigated offered.
Whether it can actually meet ClearNAI's 10ms/48kHz/2-hour-call/commercial-
license bar is not yet known and will only be known after Steps 0-6 above
are actually carried out.
