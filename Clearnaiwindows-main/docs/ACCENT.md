# Accent conversion (Indian -> American English)

Crate `crates/accent-onnx`, a Rust port of the reference engine in github.com/KarthikGavini/Accent-Converter
(`code/engine.py`, `code/dsp.py`), model `best_v6dmel` + causal HiFi-GAN, fixed American voices only.

## Where it sits

```
mic -> voice gate -> BVC (Hush/...) -> ACCENT -> auto-gain -> studio
```

Accent only runs on the BVC-on path (the BVC-off path would feed it RNNoise-processed audio, which it must not get).
The GUI says "waiting - turn BVC on" in that case. Off = one atomic load, frame untouched (bit-exact, tested).

The model never runs on the 10 ms mic thread: the stage pushes 480-sample frames into a ring, a worker thread does
48k->16k (tse-bsrnn `CausalDownsampler3x`) -> input leveler -> log-mel -> converter -> vocoder -> 24k->48k
(`CausalUpsampler2x`, same recipe as the existing 3x one) and the mic thread pulls 480 samples per frame through a
60 ms jitter buffer. No allocation on the mic path.

Safety net: output ring empty, or worker RTF > 0.9 for 2 s -> 20 ms crossfade to the BVC-only signal, engine keeps
running, GUI shows "Accent paused - CPU too slow"; it returns after RTF < 0.75 for 3 s and the jitter buffer refills.

## Verification

Golden fixtures come from the reference engine (`tools/make_golden.py`, 5 s clip, voice `american_male_1`).
`cargo test -p accent-onnx --release` (needs `ACCENT_MODELS_DIR`; skips with a message without the bundle):

| check | max abs diff |
|---|---|
| input log-mel, 50 ms and 200 ms pushes | 0 |
| converter mel, 50 ms / 200 ms pushes | 3.6e-6 / 4.1e-6 |
| 24 kHz audio, 200 ms pushes | 5.9e-5 (RMS 1.5e-6) |
| 24 kHz audio, 50 ms pushes | 1.1e-4 (RMS 1.7e-6) |
| Rust 50 ms vs Rust 200 ms pushes: mel / audio | 3.5e-6 / 1.2e-4 |

The audio peak difference is float noise through the vocoder between the reference's ORT 1.30 and the shipped ORT 1.19
DLL (the reference itself differs by 6.5e-5 between its own 50 and 200 ms runs); the converter mel agreeing to ~4e-6
shows the port is exact. Per-push frame counts match the reference exactly.

## Benchmarks (this PC: 12 logical cores, release build)

Real-time factor = compute time / audio time (lower is better; must stay < 1, the safety net trips at 0.9).
`cargo run --release -p clearnai-app --example accent_bench -- rtf`

| ORT threads | push | converter RTF | vocoder RTF | total RTF (incl. mel) | p95 per push (ms) |
|---:|---:|---:|---:|---:|---:|
| 1 | 50 ms | 0.440 | 0.605 | **1.048** | 58.1 |
| 1 | 100 ms | 0.238 | 0.554 | 0.795 | 91.3 |
| 1 | 200 ms | 0.164 | 0.569 | 0.736 | 170.1 |
| 2 | 50 ms | 0.370 | 0.421 | 0.795 | 43.1 |
| 2 | 100 ms | 0.201 | 0.360 | 0.564 | 64.4 |
| 2 | 200 ms | 0.118 | 0.324 | 0.444 | 93.6 |
| 4 | 50 ms | 0.349 | 0.341 | 0.694 | 38.5 |
| 4 | 100 ms | 0.186 | 0.261 | 0.450 | 49.9 |
| 4 | 200 ms | 0.109 | 0.235 | 0.347 | 81.1 |

Live chain, 50 ms pushes, Hush BVC + Accent + AutoGain fed at wall-clock pace for 20 s
(`accent_bench -- live "" <threads> 20`):

| ORT threads | worker RTF | process CPU (one core = 100%) | of this PC | mic-thread p99 / max per 10 ms frame | underruns | added delay (measured) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | > 0.9 -> paused | 109% | 9.1% | 2.4 / 4.5 ms | 59 (safety net engaged) | n/a (BVC-only output) |
| 2 | 0.78 | 142% | 11.8% | 2.2 / 3.3 ms | 1 | 300 ms |
| 4 | 0.70 | 182% | 15.1% | 2.5 / 4.7 ms | 0 | 297 ms |

The BVC-only mic-thread p50 is 0.92-0.98 ms, so Accent adds ~0.2 ms to the 10 ms thread.
Measured delay = envelope cross-correlation of the output against the input, minus the BVC-only delay (10 ms);
the GUI status shows a slightly pessimistic estimate (~345 ms). Delay budget: converter 162.5 ms + vocoder 25 ms +
mel window 25 ms + 25 ms average push wait + compute + 60 ms jitter buffer.

**Reading**: with 50 ms pushes, 1 thread cannot keep up on this machine; 2 threads is marginal (RTF 0.78-0.80 against a
0.9 trip point); 4 threads is comfortable. Default is 2 threads / 50 ms; 100 ms pushes at 2 threads (RTF 0.56) buy a lot of
headroom for +25 ms delay. Neither knob is exposed in the GUI yet.

## Packaging

The bundle is ~270 MB (fp32 ONNX), too big to commit (GitHub 100 MB file limit) or sensibly `include_bytes!`. It ships as
`app_models/` next to the exe (`app/installer.nsi` installs it; during development put it in `app/assets/app_models/` or set
`ACCENT_MODELS_SRC`). On the first time a voice is selected the worker calls `setup::ensure_accent_models`, which installs
into `%LOCALAPPDATA%\ClearNAI\app_models`, comparing FULL file contents (streaming) and copying only files that differ.
Needed files: `accent_onnx::BUNDLE_FILES` (8 files; `ecapa.onnx`/`librnnoise.dylib` are not used). `app_models/` is gitignored.

## Not done

- Own-voice embedding (ECAPA) and fine-tuning; only the six fixed American voices.
- Accent while BVC is off.
- GUI controls for ORT threads / push size / jitter buffer.
- Worker thread runs at normal priority (no MMCSS), which matters most under heavy CPU load.
- Models are loaded lazily on first use (~1 s here), not at launch.
