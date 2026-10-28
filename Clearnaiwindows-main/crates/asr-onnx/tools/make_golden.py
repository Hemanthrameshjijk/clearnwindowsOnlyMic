"""Writes the speech-model golden files used by tests/transcribe.rs (run with the model folder as argv[1]).

  python make_golden.py <extracted asr folder> <out dir>

Produces asr_feat.f32 (80 x T log-mel, mel-major) and asr_text.txt (greedy decode of the 5 s accent test clip).
"""
import sys
import numpy as np
import onnxruntime as ort

root, out = sys.argv[1], sys.argv[2]
clip = sys.argv[3] if len(sys.argv) > 3 else out + '/../../../accent-onnx/tests/golden/input16.f32'
vocab = [l.rstrip('\n') for l in open(root + '/fp/lm/vocab.txt', encoding='utf-8')]
x = np.fromfile(clip, dtype='<f4')


def hz2mel(f):
    return f / (200.0 / 3) if f < 1000.0 else 1000.0 / (200.0 / 3) + np.log(f / 1000.0) / (np.log(6.4) / 27.0)


def mel2hz(m):
    t = 1000.0 / (200.0 / 3)
    return m * (200.0 / 3) if m < t else 1000.0 * np.exp((np.log(6.4) / 27.0) * (m - t))


def mel_filters(sr=16000, n_fft=512, n_mels=80):
    mels = np.linspace(hz2mel(0.0), hz2mel(sr / 2), n_mels + 2)
    hz = np.array([mel2hz(m) for m in mels])
    fftf = np.linspace(0, sr / 2, n_fft // 2 + 1)
    fb = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        lo, ce, hi = hz[i], hz[i + 1], hz[i + 2]
        fb[i] = np.maximum(0, np.minimum((fftf - lo) / (ce - lo), (hi - fftf) / (hi - ce))) * 2.0 / (hi - lo)
    return fb


def features(x, preemph=0.97, win=400, hop=160, n_fft=512):
    x = np.concatenate([x[:1], x[1:] - preemph * x[:-1]])
    xp = np.pad(x, (n_fft // 2, n_fft // 2), mode='reflect')
    w = np.pad(np.hanning(win), ((n_fft - win) // 2, (n_fft - win) - (n_fft - win) // 2))
    n = 1 + (len(xp) - n_fft) // hop
    frames = np.stack([xp[i * hop:i * hop + n_fft] * w for i in range(n)])
    m = (np.abs(np.fft.rfft(frames, axis=1)) ** 2) @ mel_filters().T
    m = np.log(m + 2 ** -24)
    m = (m - m.mean(0)) / (m.std(0) + 1e-5)
    return m.T.astype(np.float32)


f = features(x)
f.tofile(out + '/asr_feat.f32')
s = ort.InferenceSession(root + '/ie/audio/fast_conformer_with_laughter_MatMul_qu8.ort', providers=['CPUExecutionProvider'])
_, o2 = s.run(None, {'audio_signal': f[None], 'length': np.array([f.shape[1]], dtype=np.int64)})
ids, prev, txt = o2[0].argmax(1), -1, ''
for i in ids:
    if i != prev and i != 1024 and i != 0:
        t = vocab[i]
        txt += t[2:] if t.startswith('##') else (' ' if txt else '') + t
    prev = i
open(out + '/asr_text.txt', 'w').write(txt)
print(f.shape, repr(txt))
