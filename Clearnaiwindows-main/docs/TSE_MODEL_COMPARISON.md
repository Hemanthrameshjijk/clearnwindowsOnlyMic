# TSE Model Comparison

Factual compatibility matrix. "—" means not applicable/no code exists to
measure; "not measured" means real code/checkpoint exists but this session
did not run that specific measurement (stated explicitly, never guessed).

| | REAL-TSE/WeSep `spk_emb_causal_100` | OpenSpeakerBeam-SS (unofficial) | `penta2himajin/tse-conv-tasnet-48k` |
|---|---|---|---|
| Sample Rate | 16000 Hz | 16000 Hz (claimed) | — (does not exist) |
| Chunk Size | none (whole-utterance only, as provided) | not verified | — |
| Causal | **Yes** (verified: config + repo's own self-test) | claimed (Conv-TasNet+S4D is architecturally streaming-capable) | — |
| Stateful (as provided) | **No** (no state I/O in any provided code path) | not verified | — |
| Lookahead | **None** (verified) | not verified | — |
| Speaker Encoder | WeSpeaker **ECAPA_TDNN_GLOB_c512**, baked into checkpoint | `resemblyzer.VoiceEncoder` (d-vector) | — |
| Embedding Dim | **192** (verified in config) | not verified (resemblyzer is typically 256) | — |
| ONNX | **No** (project uses LibTorch/TorchScript) | claimed, not verified | — |
| CPU | **Yes (verified: ran real inference on CPU)** | claimed | — |
| Rust | No (would need LibTorch binding or new ONNX export) | No (same) | — |
| Measured P50 | not measured (no chunked impl. exists) | not measured | — |
| Measured P95 | not measured | not measured | — |
| Measured P99 | not measured | not measured | — |
| RTF (whole-utterance, measured) | **0.95–1.04**, single CPU thread (5-run median at 1/3/5s) | not measured | — |
| RAM | ~995 MB process RSS (whole Python/PyTorch process, not isolated) | not measured | — |
| SI-SDR | not measured (no real multi-speaker test corpus available) | **self-reported by author: -5.89 SI-SNR** (real number, quoted from their README) | — |
| Background Suppression | not measured | not measured | — |
| License | **Ambiguous** — Apache-2.0 per source-file headers, but no repo-level LICENSE file; checkpoint weights license undocumented; "Real-TSE Challenge" framing suggests possible non-commercial restriction on weights | **MIT** (repo confirmed via GitHub API) | — |
| Checkpoint Available | **Yes — actually downloaded (1.00 GB), hash recorded** | Claimed (HF), not downloaded in this session | **No — repository does not exist** |
| ClearNAI Integration Work | Large: 48k↔16k resampling path, real chunked/stateful re-implementation, ONNX export or LibTorch Rust binding, new enrollment-embedding path (ECAPA-192, not ClearNAI's existing ResNet34-256), license clarification, real quality benchmarking | Would be similarly large, moot given the quality result | N/A |

## Why each candidate does or does not fit ClearNAI

### REAL-TSE/WeSep `spk_emb_causal_100`
**Fits:** it is the only candidate that is a real, downloaded, correctly-loading
(0 missing/unexpected keys), CPU-runnable, *architecturally causal* target
speaker extraction model found in this research. Its single-vector speaker
conditioning (`spkemb`) maps cleanly onto ClearNAI's existing "enroll once,
cache the embedding" design.

**Does not fit, as shipped:** wrong native sample rate (16kHz, not 48kHz);
no chunked/streaming reference implementation at all (whole-utterance
only); no ONNX export (LibTorch/TorchScript only); needs a *different*
speaker encoder (ECAPA-192) than ClearNAI's current WeSpeaker-ResNet34-256,
which per the task's own explicit rule must not be bridged with a random
projection; checkpoint license is not cleanly established for commercial
use. All of this is real, addressable engineering work, not fatal - see
`docs/TSE_INTEGRATION_PLAN.md` - but it is real work, not a drop-in.

### OpenSpeakerBeam-SS (unofficial SpeakerBeam-SS reimplementation)
**Fits:** genuinely real-time-oriented architecture (Conv-TasNet + S4D
state-space blocks is a legitimate streaming-capable design family); MIT
license is the cleanest of any candidate found; small (7.64M params, per
its own FLOP count).

**Does not fit:** the checkpoint's own self-reported test SI-SNR (-5.89) is
a strong, real signal that its current trained weights do not perform
usable target speaker extraction - a negative SI-SNR generally means the
output is *worse* than doing nothing. It is also an unofficial
reimplementation explicitly disclaimed as possibly diverging from the
actual published SpeakerBeam-SS results, which themselves have no released
checkpoint to fall back on.

### `penta2himajin/tse-conv-tasnet-48k`
Does not exist. Re-verified directly in this session. Cannot fit or not
fit anything - there is nothing there.

## Final selection criteria answers

- **A. Best immediately integrable candidate:** REAL-TSE/WeSep
  `spk_emb_causal_100` - it is the only candidate with a real, verified,
  loading checkpoint today.
- **B. Best quality candidate:** unknown - no candidate's real-speech
  quality was measured in this session (WeSep's paper claims real
  Libri2Mix/VoxCeleb results but those weren't independently re-verified
  here; OpenSpeakerBeam-SS's own self-reported number is bad).
- **C. Best low-latency candidate:** unknown until a real chunked/stateful
  implementation of WeSep's causal architecture is built and measured -
  its whole-utterance RTF (~0.95-1.0 on one CPU thread) is a promising but
  not conclusive signal.
- **D. Best 48-kHz candidate:** none - every real candidate found operates
  natively at 16kHz.
- **E. Best long-term production candidate:** REAL-TSE/WeSep
  `spk_emb_causal_100`, conditional on resolving its license ambiguity for
  the checkpoint weights, building a real streaming re-implementation, and
  running real quality benchmarks - see `docs/TSE_SELECTED_MODEL.md`.

No candidate satisfies every requirement today. This is reported plainly,
not smoothed over with a confident-sounding recommendation the
measurements don't support.
