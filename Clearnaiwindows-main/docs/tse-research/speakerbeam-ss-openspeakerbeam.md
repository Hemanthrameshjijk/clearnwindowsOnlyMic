# Model: SpeakerBeam-SS — official paper vs. `OpenSpeakerBeam-SS` (independent reimplementation)

Repository: **No official code/checkpoint release found** from the paper's authors (Sato et al., LINE/NTT-affiliated). Independent reimplementation: https://github.com/helloooideeeeea/OpenSpeakerBeam-SS (cloned to `research/tse/candidates/OpenSpeakerBeam-SS`)
Checkpoint: `checkpoints/best_model.pth`, referenced by the independent repo as hosted on Hugging Face at `https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset/tree/main` (not independently re-verified/downloaded in this session - see below)
Paper: Sato et al., "SpeakerBeam-SS: Real-time Target Speaker Extraction with Lightweight Conv-TasNet and State Space Modeling," Interspeech 2024, arXiv:2407.01857
License: **MIT** (independent repo only; confirmed via GitHub API `license.spdx_id: "mit"`)

## Availability

Can checkpoint actually be downloaded? **Official paper checkpoint: NO (not released by the authors).** **Independent reimplementation's checkpoint: claimed available on Hugging Face, not downloaded/verified in this session** (deprioritized after the quality finding below made further verification lower-value - see rationale).

Exact URL (independent repo only): `https://huggingface.co/datasets/helloidea/OpenSpeakerBeam-SS-dataset/tree/main`

Exact files: not enumerated in this session (not downloaded).

File sizes / SHA256: not obtained in this session.

## Architecture (independent reimplementation, per its own README/code)

Sample rate: 16000 Hz (`--sample_rate` default in `inference.py`)

Input: mixture waveform + enrollment waveform, both resampled to 16kHz mono; speaker embedding via `resemblyzer.VoiceEncoder` (a **third, distinct speaker-embedding system** - a d-vector-style encoder, not WeSpeaker, not ECAPA-TDNN) - confirmed directly from `inference.py`'s imports (`from resemblyzer import VoiceEncoder`).

Output: enhanced/extracted waveform (file-based, see `inference.py`)

Chunk size / Frame size / Hop / Lookahead / State: **not verified in this session** - would require reading `model/` source in detail, deprioritized (see rationale below). The architecture is described (by the repo, not independently confirmed) as a lightweight Conv-TasNet with S4D (a diagonal state-space model) blocks, which is *architecturally* streaming-oriented (Conv-TasNet + SSM is a genuinely real streaming-capable design family), but this session did not verify a concrete streaming code path any more than it did for WeSep.

Speaker embedding: `resemblyzer.VoiceEncoder` (d-vector), per the repo's own `inference.py`.

Embedding dimension: not confirmed in this session (resemblyzer's public `VoiceEncoder` typically produces 256-dim embeddings, but this was not independently checked against this specific repo's usage).

## Streaming

Not independently verified in this session (see rationale below).

## Runtime

CPU / GPU / ONNX / Rust / ONNX Runtime: repo claims "ONNX Runtime support for CPU acceleration (AVX2 / AVX-512)" and "designed for real-time inference." **Not independently verified** - no `.onnx` export file was located/downloaded/run in this session.

Estimated RTF: repo states "FLOPs: 21.60G, Params: 7.64M" for 1 second of 16kHz input, and speculates real-time CPU capability - **this is the project's own unverified estimate, not a measurement taken in this session.**

Measured RTF / P50 / P95 / P99 / Peak latency / RAM: **not measured in this session.**

## Quality

**The independent repo's own self-reported result: Test SI-SNR = -5.8925.**

This is a real number taken directly from the repository's own README (`## 💾 Dataset & Checkpoints` section: "🔍 [Test] Test Loss (SI-SNR): -5.8925"), not invented or estimated by this session. A negative SI-SNR on a separation/extraction task's own test output generally indicates the model's output is *not* a usable, improved isolation of the target speaker relative to the raw mixture - well-performing target speaker extraction models typically report SI-SDR/SI-SDRi in the range of +8 to +15 dB or higher. The repository's own author explicitly acknowledges this: *"current performance is modest; improvements are planned."* This is a significant understatement of what a negative SI-SNR implies about current usability.

**Decision rationale: given this self-reported quality result, further verification effort (downloading the checkpoint, running inference, measuring latency) was deprioritized in favor of the WeSep/REAL-TSE candidate, which has confirmed-working, checkpoint-verified extraction with no such quality red flag on record.** This is a documented triage decision, not a claim that OpenSpeakerBeam-SS was fully evaluated and rejected on all axes.

## Licensing

Commercial use: MIT (independent repo) - permissive, no restriction found.

Modification / Redistribution: permitted under MIT.

Model weights license: not confirmed (the Hugging Face dataset repo hosting the checkpoint was not inspected for its own license terms in this session).

Code license: MIT, confirmed via GitHub API.

Potential restrictions: **explicitly "not affiliated with the authors of the original paper"** - this is an independent reimplementation that "may have differences from the paper's methodology." Any claim about matching the *published* SpeakerBeam-SS paper's results/behavior is unverified and, per the current SI-SNR result, appears not to be matched yet.

## ClearNAI Compatibility

Not assessed in detail given the quality finding above. Summary: real, downloadable (claimed), permissively-licensed (MIT) independent reimplementation of a genuinely real-time-oriented architecture (Conv-TasNet + S4D), but with a self-reported negative SI-SNR that indicates the current checkpoint is very likely not usable for real target-speaker extraction as-is. The *official* paper (which reports real, presumably much better results) has no released code or checkpoint to verify against at all.

## What was actually done in this session

1. Confirmed the official paper (arXiv:2407.01857) exists; confirmed via web search that no official code/checkpoint repository from the paper's authors could be found.
2. Found and cloned the independent reimplementation `helloooideeeeea/OpenSpeakerBeam-SS` (real repo: created 2025-03-16, last pushed 2026-08-25, 19 stars, MIT license - confirmed via GitHub API, not assumed).
3. Read its real `README.md` and `inference.py` directly from the cloned repo.
4. Found the repo's own self-reported SI-SNR = -5.8925 test result and used it as the deciding factor to deprioritize further hands-on verification (no checkpoint download, no inference run, no latency measurement performed for this candidate).

This candidate was **not** run in this session, unlike WeSep/REAL-TSE. This is stated plainly rather than implied.
