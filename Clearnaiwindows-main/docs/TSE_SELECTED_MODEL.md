# TSE Selected Model (Conditional)

## Selection

**REAL-TSE/WeSep, `spk_emb_causal_100` checkpoint** (BSRNN causal separator
+ baked-in WeSpeaker ECAPA_TDNN_GLOB_c512 speaker encoder).

This is a **conditional** selection for further engineering, not a claim
that target speaker extraction works in ClearNAI today. Per the research
brief's own instruction: *"Do not claim completion until the selected TSE
performs real target-speaker extraction on real mixed speech."* That has
not happened yet. What has happened: this is the only candidate found in
this research with a real, downloadable checkpoint that actually loads
(0 missing/unexpected keys) and actually runs inference on CPU, producing
finite output, with a real, measured (not estimated) whole-utterance RTF
of 0.95-1.04 on this machine's CPU. See
`docs/tse-research/wesep-real-tse-spk-emb-causal.md` for every verified
detail and every open gap.

## Why this one, concretely

- Real repository, real checkpoint, real download (1.00 GB, hash recorded).
- Real architecture match to ClearNAI's enrollment model: a single,
  per-utterance speaker embedding computed once and reused, not a
  per-frame feature requiring continuous re-computation.
- Architecturally causal (zero lookahead), confirmed by the project's own
  source-level self-test, not just a marketing claim.
- CPU-only, verified by actually running it.
- The only candidate in this research where model architecture, checkpoint
  weights, and inference code were all confirmed to agree with each other
  (0 missing/unexpected keys is a strong, specific, real signal that
  nothing was silently wrong).

## What is explicitly NOT yet true

- It does not run in real time at 480-sample/10ms granularity - no
  streaming/chunked/stateful implementation exists yet (see integration
  plan).
- It does not operate at 48kHz - a resampling path would be needed, with
  its own unmeasured latency/CPU cost.
- It cannot use ClearNAI's existing enrolled speaker embedding
  (WeSpeaker-ResNet34, 256-dim) - it needs its own ECAPA-192 embedding,
  computed via its own baked-in encoder submodule (confirmed loadable) or a
  separately-sourced standalone WeSpeaker ECAPA512 checkpoint.
- Its real-speech separation quality (SI-SDR, background-speaker
  suppression, target-speaker preservation) has not been measured against
  real speech in this session - there was no real multi-speaker speech
  corpus available in this sandboxed environment.
- Its checkpoint weights' commercial-use license is not cleanly
  established (Apache-2.0 source headers exist; no repo-level LICENSE
  file; "Real-TSE Challenge" framing raises a real possibility of a
  research-only restriction on the weights specifically). **This must be
  resolved - directly with the repository owners/challenge organizers -
  before shipping this checkpoint in a commercial product.**

## What would disqualify this selection

Any of the following, if confirmed during the next engineering phase,
would be grounds to not ship this model:

1. The license question resolves to "research/challenge use only, no
   commercial redistribution."
2. A real chunked/stateful re-implementation cannot meet the 10ms deadline
   on ClearNAI's target CPU once actually built and measured.
3. Real-speech quality testing (once a real speech corpus is available)
   shows poor target-speaker preservation or background-speaker
   suppression.

None of these has been ruled out yet - they are exactly what the
integration plan's early steps are for.

## If this selection is disqualified

Per the research brief's own fallback instruction, the next paths (in
rough order of effort) would be:

1. Contact the REAL-TSE/wesep-real-tse maintainers to clarify the
   checkpoint's actual license terms - if they confirm commercial use is
   fine, the disqualification (1) above is resolved with no new
   engineering.
2. Investigate whether the official (non-independent) SpeakerBeam-SS
   authors have since released code/checkpoints, or whether
   OpenSpeakerBeam-SS's quality improves in a future revision.
3. As a last resort, fine-tune a causal TSE architecture (e.g. this same
   BSRNN-causal design, or a from-scratch causal Conv-TasNet) on
   commercially-licensed or self-collected training data - a
   materially larger undertaking than integrating an existing checkpoint,
   and not recommended as a first step.
