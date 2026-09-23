# Model: REAL-TSE / WeSep — `spk_emb_causal_100`

Repository: https://github.com/REAL-TSE/wesep-real-tse (cloned to `research/tse/candidates/wesep-real-tse`)
Checkpoint: Google Drive bundle `real-tse-ckpt.zip`, linked from the repo's own README
Paper: Wang et al., "WeSep: A Scalable and Flexible Toolkit Towards Generalizable Target Speaker Extraction," Interspeech 2024
License: **Ambiguous — see Licensing section**

## Availability

Can checkpoint actually be downloaded? **YES — actually downloaded in this session.**

- Exact URL: `https://drive.google.com/uc?export=download&id=1M4UqK2A2EeHmQ0pCevYqBgaYn3RvklgC` (redirects through Google's large-file virus-scan-warning confirm flow to `https://drive.usercontent.google.com/download`)
- Exact file: `real-tse-ckpt.zip`, containing four checkpoint directories: `spk_emb_100/`, `spk_emb_causal_100/`, `tfmap_context_100/`, `tfmap_context_causal_100/`, each with `avg_model.pt` + `config.yaml`.
- File sizes (actually measured): zip = 1,048,798,335 bytes (~1.00 GB); `spk_emb_causal_100/avg_model.pt` = 251,856,016 bytes; `spk_emb_causal_100/config.yaml` = 2,487 bytes.
- SHA-256 (computed in this session, `certutil -hashfile ... SHA256`):
  - `real-tse-ckpt.zip`: `0a7e44050b1235db209181d7cda29ac93487fbb981ec768d3062b3243c49ce98`
  - `spk_emb_causal_100/avg_model.pt`: `42b73219eefefcbba4b5da0dcb89dce292359c9f93b1a951c7b6d196da2155b3`
  - `spk_emb_causal_100/config.yaml`: `80820eb78de6639160f7124d09cbacbc8397e321d8075721e89edb7cec50a0fa`

## Architecture

Sample rate: **16000 Hz** (config `separator.sr: 16000`; not native 48kHz)

Input: mixture waveform `[B, 1, T]` + enrollment waveform `[B, 1, T_e]` (raw PCM; the model computes its own speaker embedding internally from `T_e`)

Output: extracted target-speaker waveform `[B, 1, T]` (same length as mixture)

Chunk size: **none provided** — reference code (`evaluate.py`, `wesep/cli/extractor.py`, `runtime/separate/separate_engine.cc`) all process the *entire* mixture/enrollment file in one forward call. No chunked/streaming call pattern is provided anywhere in the repository.

Frame size / hop (STFT front-end inside the BSRNN separator, verified from `wesep/models/tse_bsrnn_spk.py` defaults and `config.yaml`): `win=512` samples (32ms @ 16kHz), `stride=128` samples (8ms hop).

Lookahead: **none at the architecture level** — `causal: true` in `config.yaml`, and the model's own `check_causal()` self-test (in `tse_bsrnn_spk.py`) verifies zero-lookahead causality by construction (output before time *i* is provably unaffected by input changes after time *i*). This is an architectural/mathematical property, not a demonstrated low-latency streaming *implementation* (see Streaming section).

State: internally, the BSRNN separator's causal Conv1d/RNN layers carry state *within* one forward call; the reference code never persists or re-injects that state across separate calls (no `state_in`/`state_out` tensors are exposed anywhere in the provided Python or C++ code).

Speaker embedding: **not a separate/external component in this repository's usage pattern** — computed internally by a `SpeakerEncoder` submodule (`wesep/modules/speaker/encoder.py`), which wraps WeSpeaker's `get_speaker_model()` factory and is trained/saved jointly inside the same `avg_model.pt`. Confirmed by config: `speaker_encoder.model: ECAPA_TDNN_GLOB_c512`, `pretrained: ./wespeaker_models/voxceleb_ECAPA512/avg_model.pt` (this path is only used at *training* time to initialize from a public WeSpeaker checkpoint; `wesep/cli/extractor.py`'s `disable_pretrained()` zeroes this field at inference time since the fine-tuned copy is already inside `avg_model.pt`).

Embedding dimension: **192** (`spk_args.embed_dim: 192`, ASTP pooling — confirmed in `config.yaml`, both for `spk_emb_causal_100` and `tfmap_context_causal_100`).

## Streaming

Causal: **YES** (architecturally, verified via config + the repo's own causality self-test).

Stateful: **NO reference implementation provided.** The model's internal layers are stateful mathematically, but no code in this repository exposes/persists that state across calls — every provided entry point (Python `evaluate.py`, C++ `separate_engine.cc`) re-runs a full forward pass over the whole buffer each time.

Future context: **NO** (consistent with `causal: true` and the self-test).

Algorithmic latency: not separately measurable from the provided code — bounded above by the causal architecture's own receptive field, which was not decomposed/measured in this session.

Buffering latency: **N/A as provided** — the reference pipeline requires the whole utterance before returning anything.

Total expected latency: **not the 10ms/480-sample-frame latency ClearNAI needs, as shipped.** Turning this into a genuine low-latency streaming component requires new engineering (see `docs/TSE_INTEGRATION_PLAN.md`) — this is not a "drop-in" streaming model despite the causal architecture.

## Runtime

CPU: **YES**, confirmed by actually running it on CPU in this session (no GPU used or required).

GPU required: NO (CPU inference verified).

ONNX: **not provided by the project, not attempted in this session.** The project's own C++ runtime uses **LibTorch (TorchScript)**, not ONNX (`runtime/separate/separate_engine.cc` calls `torch::jit::load(model_path)` on a `.zip` TorchScript export — no ONNX export tooling exists in the repo).

Rust: not directly usable as-is; would require either (a) a LibTorch-based Rust binding (e.g. `tch-rs`, a new, heavy runtime dependency this project doesn't currently have) or (b) a custom ONNX export attempted by us (not done in this session — nontrivial given the model's internal STFT/complex-mask operations and the speaker encoder submodule).

ONNX Runtime: N/A (no ONNX export exists yet).

Estimated RTF: not applicable — see Measured RTF (a real measurement was taken instead of estimating).

**Measured RTF (whole-utterance, single CPU thread, this session's hardware, 5 runs per duration):**

| Duration | Median forward time | RTF |
|---|---|---|
| 1.0s | 1016.8 ms | 1.037 |
| 3.0s | 2829.3 ms | 0.946 |
| 5.0s | 4908.7 ms | 0.973 |

(See `research/tse/scripts/benchmark_whole_utterance.py` for the exact script; raw run-by-run numbers are printed by the script, not hand-picked.)

**This is whole-utterance batch RTF, not 10ms-chunk streaming RTF** — it says the model can process N seconds of audio in roughly N seconds when given the whole buffer at once, on a single CPU thread, unoptimized PyTorch eager mode. It does **not** by itself confirm that a properly re-engineered 480-sample/10ms streaming version would meet the 10ms deadline — per-call fixed overhead, lack of incremental state reuse, and PyTorch eager-mode dispatch cost all become proportionally larger at a 480-sample granularity than at whole-utterance granularity. This must be measured after a real chunked/stateful re-implementation exists, not assumed from this number.

Measured P50/P95/P99: a **naive** growing-buffer probe was run (`research/tse/benchmarks/benchmark_streaming.py`) — it re-processes the entire buffer from scratch on every simulated "chunk" (O(n²), not real incremental streaming) and is explicitly documented in its own header as a rough ceiling, not a true streaming measurement. Real result (single CPU thread, 16kHz, 10ms/160-sample nominal chunk, 50 chunks = 500ms of audio): **P50 = 310.8 ms, P90 = 477.3 ms, P95 = 506.6 ms, P99 = 539.9 ms, MAX = 539.9 ms.** Two concrete, real findings from this run (not estimates):
1. **The model's internal STFT (512-sample window) rejects any buffer shorter than 512 samples** — a real `RuntimeError` from `torch.stft` was hit and confirmed for chunks 1–3 (buffers of 160/320/480 samples); the probe only starts once the buffer exceeds the window (chunk 4, 640 samples/40ms).
2. **Even the smallest processable buffer (40ms) took ~108ms wall-clock** — because `forward()` recomputes the full ECAPA-TDNN speaker embedding from raw enrollment audio on *every call* (confirmed directly in source — `enroll` is a raw waveform argument, not a cached embedding). See `docs/TSE_INTEGRATION_PLAN.md` Step 1a for the concrete fix this implies.

Peak latency: 539.9 ms in the naive probe above (again, an O(n²) artifact ceiling, not a real chunked implementation's expected peak).

RAM: process RSS measured at ~945 MB before model construction growing to ~995 MB after loading `spk_emb_causal_100` (Python interpreter + PyTorch + wespeaker + this ~251MB checkpoint all loaded in-process — this is a whole-process figure, not an isolated "just the model" figure, and is not representative of what a compiled Rust/tract or LibTorch-C++ deployment would use).

## Quality

**Real human-speech quality was not measured in this session.** No real multi-speaker speech corpus was available in this sandboxed environment to construct genuine mixture/enrollment test pairs, and the paper's own reported benchmark numbers (on Libri2Mix/VoxCeleb-derived test sets) were not independently re-verified here. Reporting the paper's numbers as this session's own measurement would violate the "do not invent latency/accuracy" instruction — they are not repeated here.

A **synthetic-tone sanity probe** was run instead (`research/tse/benchmarks/benchmark_tse.py`, 11 scenarios: target-only, +1/+2 background tones, +noise, overlapping, start/stop, silence, near/far-end; reverberation skipped — no RIR data available). Real, measured SI-SDR-improvement results: mostly **negative** (mean −29.4 dB across scenarios; individual scenarios ranged from −49.7 dB to +22.6 dB). **This must not be read as "the model performs badly on real speech."** Pure sine tones are wildly outside a speech-trained model's input distribution (no formants, no harmonics-of-a-fundamental-pitch pattern, no pauses) — both the baked-in ECAPA speaker encoder and the BSRNN separator were trained exclusively on real speech statistics, so their behavior on synthetic tones is not informative about real-speech performance in either direction. This result is reported only because the research brief asked not to hide any measurement taken, not as a quality verdict. Real quality testing (per `docs/TSE_INTEGRATION_PLAN.md` Step 6) requires real speech.

## Licensing

Commercial use / Modification / Redistribution: **ambiguous, not cleanly determined.**

- The GitHub API reports `"license": null` for the repository (no license auto-detected).
- A direct fetch of `LICENSE` at the repo root returns **404 (does not exist)**.
- However, multiple individual source files carry an explicit SPDX/Apache header, e.g. `runtime/separate/separate_engine.cc`: `"Licensed under the Apache License, Version 2.0"`, and `wesep/models/tse_bsrnn_spk.py`: `"SPDX-License-Identifier: Apache-2.0"`. This is the same license convention used by the same team's other projects (WeNet, WeSpeaker), both genuinely Apache-2.0.
- **This is a real, unresolved gap, not a settled "Apache-2.0, safe to use" answer.** Without a repository-level `LICENSE` file, the project's overall licensing terms (and in particular the **checkpoint weights**, which are a separate artifact from the source code and are not necessarily covered by a source-file SPDX header at all) are not conclusively established. The checkpoint is also explicitly framed as belonging to an active academic "Real-TSE Challenge," which commonly imposes non-commercial/research-only use restrictions on challenge baseline weights that would not necessarily be documented in the code repository at all.

Model weights license: **not documented anywhere found in this session.** Do not assume Apache-2.0 coverage extends to `avg_model.pt`.

Code license: Apache-2.0 per per-file SPDX headers (not a repo-level LICENSE file).

Potential restrictions: possible non-commercial/research-only restriction on the checkpoint weights specifically, given the "Real-TSE Challenge" framing; **must be confirmed directly with the repository owners/challenge organizers before any commercial use of the weights.**

## ClearNAI Compatibility

48 kHz: **NO** — native 16kHz. Would require the resample-in/resample-out path from the task brief (48k → causal resampler → 16k TSE → causal resampler → 48k), with its own added latency/CPU cost, not yet measured.

10 ms / 480 samples: **NO, not as provided.** Native STFT hop is 8ms (128 samples @ 16kHz) internally; no 10ms/480-sample (48kHz) chunked entry point exists.

CPU: **YES** (verified, see Runtime).

Windows: not separately verified (PyTorch/CPU inference itself has no Windows-specific blocker; this session's Python research environment *was* run on Windows successfully).

Rust: **NO, not directly** — no ONNX export exists; would need either a LibTorch Rust binding or a from-scratch ONNX export effort.

ONNX: **NO** (not provided; not attempted).

Persistent state: **NO reference implementation** - would need new engineering to expose/carry state across 480-sample calls.

Enrollment compatibility: **Requires a different speaker encoder than ClearNAI's existing one.** ClearNAI's `speaker-encoder` crate bundles WeSpeaker ResNet34-LM (256-dim). This checkpoint's internal encoder is WeSpeaker **ECAPA_TDNN_GLOB_c512 (192-dim, ASTP pooling)** — a different architecture and dimension, jointly fine-tuned inside this specific checkpoint. **Per the task's explicit instruction, these must not be bridged with a random projection.** The existing WeSpeaker ResNet34 encoder can remain for ClearNAI's own enrollment-verification/UI purposes; this TSE model would need its own, separate enrollment-embedding computation (either by isolating and reusing its own baked-in `SpeakerEncoder` submodule - confirmed loadable, verified in this session - or by separately sourcing the standalone WeSpeaker `voxceleb_ECAPA512` checkpoint referenced in its training config).

Production suitability: **Not yet — real, verified architecture and real, verified checkpoint, but requires substantial additional engineering (streaming re-implementation, resampling path, Rust/ONNX or LibTorch integration, real quality benchmarking, license clarification) before this could ship.** See `docs/TSE_INTEGRATION_PLAN.md`.

## What was actually done in this session (for reproducibility)

1. Cloned `REAL-TSE/wesep-real-tse` (real repo, confirmed via GitHub API).
2. Downloaded the real `real-tse-ckpt.zip` (1.00 GB) via the actual Google Drive large-file confirm-download flow (not a placeholder/assumed link).
3. Extracted all four checkpoint variants; confirmed `spk_emb_causal_100/{avg_model.pt,config.yaml}` exist with the sizes/hashes above.
4. Installed a real Python 3.11 + PyTorch 2.14 (CPU) + `wespeaker` (pinned commit from the repo's own README) research environment (`research/tse/venv`).
5. Loaded `TSE_BSRNN_SPK` from the repo's own source, with the real `config.yaml`, and loaded `avg_model.pt`'s actual weights - **0 missing keys, 0 unexpected keys** (`strict=True` succeeded) after correctly locating the state dict at `checkpoint["models"][0]` (the file is a full training-loop checkpoint dict, not a bare state_dict - discovered by inspection, not assumed).
6. Ran real forward passes on synthetic audio (not real speech - no real multi-speaker corpus was available), confirmed finite, correctly-shaped output, and measured real wall-clock timing (table above).

See `research/tse/scripts/load_real_tse.py` and `research/tse/scripts/benchmark_whole_utterance.py` for the exact, runnable code.
