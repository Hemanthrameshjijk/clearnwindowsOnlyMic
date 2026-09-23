# TSE Model Research — Overview

This document summarizes the research phase for finding a real Target
Speaker Extraction (TSE) model for ClearNAI, per the research brief. It
does not modify ClearNAI's production architecture - see
`docs/TSE_INTEGRATION_PLAN.md` for what would need to change, and only
after further engineering, not yet.

**Scope note on speed vs. exhaustiveness:** this research was explicitly
asked to be done "as fast as possible." Real, hands-on verification (repo
inspection, checkpoint download, model loading, real inference, real
timing) was prioritized for the most promising, most concretely-specified
candidate (REAL-TSE/WeSep `spk_emb_causal_100`). Candidates that showed an
early, real disqualifying signal (SpeakerBeam-SS's public reimplementation
reporting negative SI-SNR; `penta2himajin/tse-conv-tasnet-48k` not existing
at all) were not pursued to the same depth once that signal was found -
this is a deliberate triage decision, stated explicitly rather than hidden.

## Candidates investigated

| Candidate | Real repo? | Real checkpoint? | Actually downloaded? | Actually run? | Detail doc |
|---|---|---|---|---|---|
| REAL-TSE/WeSep `spk_emb_causal_100` | Yes | Yes | **Yes (1.00 GB)** | **Yes - loads with 0 missing/unexpected keys, real CPU inference confirmed** | `docs/tse-research/wesep-real-tse-spk-emb-causal.md` |
| REAL-TSE/WeSep `tfmap_context_causal_100` | Yes (same repo/bundle) | Yes | Yes (same bundle) | No (deprioritized - same architecture family, frame-level conditioning is a worse fit for ClearNAI's "enroll once" model - see below) | folded into the doc above |
| SpeakerBeam-SS (official paper) | N/A | **No official release found** | No | No | `docs/tse-research/speakerbeam-ss-openspeakerbeam.md` |
| OpenSpeakerBeam-SS (independent reimplementation) | Yes | Claimed (HF) | No (deprioritized after quality red flag) | No | same doc |
| `penta2himajin/tse-conv-tasnet-48k` | **No - does not exist** | N/A | N/A | N/A | `docs/tse-research/penta2himajin-tse-conv-tasnet-48k.md` |

No other candidate was found with (a) a real implementation, (b) a real,
downloadable checkpoint, (c) an identifiable license, and (d) reproducible
inference, all at once - the bar the research brief set for inclusion.
Broader searches for "causal target speech extraction," "streaming
SpeakerBeam," "BSRNN TSE," and "low-latency target speaker extraction"
surfaced only academic papers (e.g. 3S-TSE, SpEx+/SpEx family, Waveformer
for target *sound* rather than target *speaker* extraction) with no
publicly released, downloadable pretrained checkpoint identified in the
time available - these are not included as full candidates because there
was nothing concrete to verify.

## Why `tfmap_context_causal_100` was not run

Its `config.yaml` (real, inspected in this session) shows the same causal
BSRNN separator and the same `ECAPA_TDNN_GLOB_c512` (192-dim) speaker
encoder as `spk_emb_causal_100`, but conditions on a frame-level "TF-Map"
speaker cue (`tfmap.enabled: true`) rather than a single per-utterance
embedding (`spkemb.enabled: true`). ClearNAI's existing enrollment
architecture (`speaker-profile::TargetSpeakerProfile`,
`speaker_profile::TargetSpeakerStore`) is built around "compute one
embedding at enrollment time, cache it, reuse it for the whole call" -
`spk_emb_causal_100`'s single-vector conditioning is the direct
architectural match; `tfmap_context_causal_100` would need its own
investigation into whether its frame-level cue can also be reduced to a
single cached representation, which was not done given time constraints
and the already-strong fit of the `spkemb` variant.

## What was actually verified (not assumed) for the selected candidate

See `docs/tse-research/wesep-real-tse-spk-emb-causal.md` for full detail.
In summary, real and reproducible in this session:

- The GitHub repository exists and its README/source code were read directly.
- The checkpoint bundle was downloaded for real (1.00 GB, real SHA-256 recorded).
- A real Python 3.11 + PyTorch 2.14 (CPU) research environment was built
  (`research/tse/venv`), isolated from ClearNAI's Rust production crates.
- The model was constructed from the repo's own real source code and the
  checkpoint's real weights were loaded with **zero missing and zero
  unexpected keys** under `strict=True` - proof the architecture
  understanding and the checkpoint match exactly.
- Real forward passes were run on synthetic audio (finite, correctly-shaped
  output confirmed) and real wall-clock CPU timing was measured (not
  estimated) across three input durations, each averaged over 5 runs.

Not verified (stated plainly, not glossed over):

- Real-speech quality (SI-SDR, target preservation, background suppression,
  speaker similarity, WER) - no real multi-speaker speech corpus was
  available in this sandboxed environment.
- True 10ms/480-sample chunked streaming latency - the checkpoint is
  architecturally causal but the reference code only provides whole-
  utterance batch inference; a real chunked/stateful re-implementation
  would be needed to measure this honestly.
- ONNX export / Rust integration - not attempted; the project's own runtime
  path is LibTorch (TorchScript), not ONNX.
- License clarity for the checkpoint weights specifically (see the model
  doc's Licensing section - a real, unresolved gap).

## Research environment

```
research/tse/
├── candidates/
│   ├── wesep-real-tse/           # git clone of REAL-TSE/wesep-real-tse
│   └── OpenSpeakerBeam-SS/       # git clone of the independent reimplementation
├── checkpoints/                  # (real checkpoints are large; kept outside
│                                    the repo under the OS temp dir - see
│                                    scripts for the exact path used)
├── scripts/
│   ├── load_real_tse.py          # loads spk_emb_causal_100, runs 1 inference, verifies output
│   └── benchmark_whole_utterance.py  # 5-run timing + RSS measurement (real numbers, see model doc)
├── benchmarks/
│   ├── benchmark_tse.py          # multi-scenario synthetic-signal sanity benchmark (see script header for honesty caveats)
│   └── benchmark_streaming.py    # naive growing-buffer per-chunk latency probe (explicitly NOT a true streaming measurement - see script header)
└── results/                      # raw output captured from the scripts above
```

Python/PyTorch/wespeaker dependencies live only under `research/tse/venv` -
they are not, and must never become, a dependency of the production Rust
crates. See `docs/TSE_INTEGRATION_PLAN.md` for the real path from here to a
production Rust integration.
