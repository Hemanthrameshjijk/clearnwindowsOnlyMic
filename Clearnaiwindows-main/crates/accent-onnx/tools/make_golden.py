"""Golden fixtures for the Rust accent-onnx port.

Runs the REFERENCE engine (github.com/KarthikGavini/Accent-Converter, code/engine.py + dsp.py) on a 16 kHz
mono wav and writes input / per-push outputs / final audio as raw little-endian f32 .bin plus golden.json.

    pip install numpy scipy soundfile onnxruntime
    python -I tools/make_golden.py --ref-code <clone>/code --models <app_models> --wav in.wav --out tests/golden

(-I is fine: the reference code directory is added to sys.path explicitly below.)
"""
import argparse, json, sys
from pathlib import Path

import numpy as np
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("--ref-code", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--wav", required=True, help="16 kHz mono wav (5 s)")
ap.add_argument("--out", required=True)
ap.add_argument("--voice", default="american_male_1")
ap.add_argument("--threads", type=int, default=1)
a = ap.parse_args()
sys.path.insert(0, a.ref_code)

import engine as E  # noqa: E402
from dsp import LogMel, StreamingLogMel  # noqa: E402

models, out = Path(a.models), Path(a.out)
out.mkdir(parents=True, exist_ok=True)
cat = E.load_catalog(models)
entry = cat["converters"][0]
conv = E.ConverterV5(models / entry["onnx"], models / entry["npz"], a.threads)
voc = E.CausalVocoder(models, cat["vocoder"], a.threads)
logmel = LogMel.from_npz(models / cat["features"])
spk = np.load(models / "american_voices.npz")[a.voice].astype(np.float32)

x, sr = sf.read(a.wav, dtype="float32")
assert sr == 16000 and x.ndim == 1, "need 16 kHz mono"


def run(push):
    """Streaming: samples -> mel -> converter -> vocoder, `push` samples at a time. Returns per-push records."""
    spec, sc, st = StreamingLogMel(logmel), E.StreamingConverterV5(conv, spk), voc.streamer()
    in_mel, out_mel, audio, counts = [], [], [], []
    for i in range(0, len(x), push):
        m_in = spec.push(x[i:i + push])
        m_out = sc.push(m_in)
        w = st.push(m_out)
        in_mel.append(m_in); out_mel.append(m_out); audio.append(w)
        counts.append([len(m_in), len(m_out), len(w)])
    m_in = spec.flush()
    m_out = np.concatenate([sc.push(m_in), sc.flush()])
    w = st.push(m_out)
    in_mel.append(m_in); out_mel.append(m_out); audio.append(w)
    counts.append([len(m_in), len(m_out), len(w)])
    w = st.flush()
    audio.append(w)
    counts.append([0, 0, len(w)])
    cat_ = lambda L, d: np.concatenate(L).astype(np.float32) if sum(len(v) for v in L) else np.zeros((0, d), np.float32)
    return cat_(in_mel, 80), cat_(out_mel, 80), np.concatenate(audio).astype(np.float32), counts


meta = {"voice": a.voice, "input_samples": int(len(x)), "lag_samples_24k": voc.delay_frames * voc.hop, "runs": {}}
x.tofile(out / "input16.f32")
for name, push in (("p50ms", 800), ("p200ms", 3200)):
    m_in, m_out, w, counts = run(push)
    m_in.tofile(out / f"{name}_mel_in.f32")
    m_out.tofile(out / f"{name}_mel_out.f32")
    w.tofile(out / f"{name}_audio24.f32")
    meta["runs"][name] = {"push_samples": push, "mel_in_frames": len(m_in), "mel_out_frames": len(m_out),
                          "audio_samples": len(w), "per_push": counts}
    print(name, m_in.shape, m_out.shape, w.shape, "peak", float(np.abs(w).max()))
# whole-file cross-check of the reference itself (input mel only; the converter is chunk-invariant by design)
whole = logmel(x)
a50 = np.fromfile(out / "p50ms_mel_in.f32", np.float32).reshape(-1, 80)
print("ref whole-file vs streaming input mel max diff:", float(np.abs(whole - a50).max()), whole.shape, a50.shape)
a200 = np.fromfile(out / "p200ms_mel_out.f32", np.float32); a50o = np.fromfile(out / "p50ms_mel_out.f32", np.float32)
print("ref 50ms vs 200ms converter mel max diff:", float(np.abs(a200 - a50o).max()))
(out / "golden.json").write_text(json.dumps(meta, indent=1))
