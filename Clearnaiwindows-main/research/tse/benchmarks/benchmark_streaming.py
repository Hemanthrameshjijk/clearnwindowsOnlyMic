"""Naive "streaming" latency probe for the REAL-TSE `spk_emb_causal_100`
checkpoint.

IMPORTANT HONESTY NOTE (read before trusting any number this script
prints): the reference model provides NO real incremental/stateful
streaming API (see docs/tse-research/wesep-real-tse-spk-emb-causal.md).
This script does NOT implement genuine streaming inference. Instead, to
get *some* real, measured signal about per-chunk cost, it feeds audio in
480-sample (10ms @ 48kHz-equivalent scaled to this model's 16kHz rate =
160 samples) increments by re-running the model on the GROWING buffer
every time - i.e. chunk N's cost includes recomputing everything from
chunk 0 again. This is O(n^2) over a call and gets slower and slower as
the buffer grows - it is a rough, honest "ceiling probe," not a
representative measurement of what a real, correctly-implemented
incremental/stateful re-engineering (docs/TSE_INTEGRATION_PLAN.md Step 1)
would cost. Do not quote these P50/P95/P99 numbers as "the model's
streaming latency" - they are this naive approach's latency, which a real
implementation should be far below.

Not part of the ClearNAI production build.
"""
import sys
import time
import types
import yaml
import torch
import torchaudio

REPO = r"C:\Users\Developer\clearnwindowsOnlyMic\Clearnaiwindows-main\research\tse\candidates\wesep-real-tse"
CKPT_DIR = r"C:\Users\DEVELO~1\AppData\Local\Temp\models\real-tse\extracted\spk_emb_causal_100"
sys.path.insert(0, REPO)

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *a, **k: None
_fake_s3prl_nn = types.ModuleType("s3prl.nn")
_fake_s3prl_nn.Featurizer = object
_fake_s3prl_nn.S3PRLUpstream = object
_fake_s3prl = types.ModuleType("s3prl")
_fake_s3prl.nn = _fake_s3prl_nn
sys.modules["s3prl"] = _fake_s3prl
sys.modules["s3prl.nn"] = _fake_s3prl_nn

from wesep.models.tse_bsrnn_spk import TSE_BSRNN_SPK  # noqa: E402

SR = 16000
CHUNK_MS = 10
CHUNK_SAMPLES = SR * CHUNK_MS // 1000  # 160 samples @ 16kHz for a 10ms chunk
NUM_CHUNKS = 50  # 500ms of audio - kept short given O(n^2) cost


def disable_pretrained(cfg):
    if isinstance(cfg, dict):
        return {k: (None if k == "pretrained" else disable_pretrained(v)) for k, v in cfg.items()}
    return cfg


def load_model():
    with open(f"{CKPT_DIR}/config.yaml", "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = disable_pretrained(configs)
    model = TSE_BSRNN_SPK(configs["model_args"]["tse_model"])
    ckpt = torch.load(f"{CKPT_DIR}/avg_model.pt", map_location="cpu")
    model.load_state_dict(ckpt["models"][0], strict=True)
    model.eval()
    return model


def percentile(sorted_vals, p):
    idx = min(int(round(p * (len(sorted_vals) - 1))), len(sorted_vals) - 1)
    return sorted_vals[idx]


def main():
    torch.set_num_threads(1)
    model = load_model()
    enroll = (0.4 * torch.sin(2 * torch.pi * 220.0 * torch.arange(SR * 2, dtype=torch.float32) / SR)).unsqueeze(0)

    full_signal = torch.randn(1, CHUNK_SAMPLES * NUM_CHUNKS) * 0.1
    chunk_times_ms = []

    print(f"Naive growing-buffer probe: {NUM_CHUNKS} chunks x {CHUNK_MS}ms "
          f"({CHUNK_SAMPLES} samples @ {SR}Hz) = {NUM_CHUNKS * CHUNK_MS}ms total audio")
    print("(O(n^2): each 'chunk' re-processes the entire buffer so far - see module docstring)\n")

    # Real finding (not assumed): the model's internal STFT (win=512 @
    # 16kHz = 32ms) rejects any input shorter than its window - a bare
    # single 10ms/160-sample chunk (i=1..3, buffer < 512 samples) throws
    # "Padding size should be less than the corresponding input dimension"
    # from torch.stft itself. This is a genuine, concrete confirmation that
    # this model, as provided, cannot process sub-window buffers in
    # isolation - a real constraint any streaming re-implementation
    # (docs/TSE_INTEGRATION_PLAN.md Step 1) must design around (e.g. an
    # internal overlap-buffer feeding the STFT, decoupled from the
    # externally-visible 10ms chunk size). The probe below starts once the
    # buffer first exceeds the STFT window, and reports this constraint
    # explicitly rather than silently starting later.
    min_chunks_for_window = -(-513 // CHUNK_SAMPLES)  # ceil(513 / chunk) - first buffer > 512 samples
    print(f"Note: buffers shorter than the {512}-sample STFT window are rejected by "
          f"torch.stft (confirmed by an actual RuntimeError on chunk 1-{min_chunks_for_window - 1}); "
          f"probe starts at chunk {min_chunks_for_window} "
          f"(buffer={min_chunks_for_window * CHUNK_SAMPLES} samples = "
          f"{min_chunks_for_window * CHUNK_MS}ms).\n")

    with torch.no_grad():
        for i in range(min_chunks_for_window, NUM_CHUNKS + 1):
            buf = full_signal[:, : i * CHUNK_SAMPLES]
            t0 = time.perf_counter()
            _ = model(buf, enroll)
            t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000
            chunk_times_ms.append(elapsed_ms)
            if i in (min_chunks_for_window, 10, 25, 50):
                print(f"  chunk {i:3d} (buffer={i * CHUNK_MS:4d}ms): {elapsed_ms:8.2f} ms")

    sorted_times = sorted(chunk_times_ms)
    print(f"\nPer-call latency across {NUM_CHUNKS} calls (naive growing-buffer, NOT true streaming):")
    print(f"  P50:  {percentile(sorted_times, 0.50):8.2f} ms")
    print(f"  P90:  {percentile(sorted_times, 0.90):8.2f} ms")
    print(f"  P95:  {percentile(sorted_times, 0.95):8.2f} ms")
    print(f"  P99:  {percentile(sorted_times, 0.99):8.2f} ms")
    print(f"  MAX:  {max(chunk_times_ms):8.2f} ms")
    print(f"\n10ms deadline: the very first (smallest-buffer) call took "
          f"{chunk_times_ms[0]:.2f} ms; by chunk {NUM_CHUNKS} (buffer={NUM_CHUNKS * CHUNK_MS}ms) "
          f"the naive re-processing cost had grown to {chunk_times_ms[-1]:.2f} ms. "
          "This growth is an artifact of the naive O(n^2) probe, not evidence about a real "
          "incremental implementation's cost - see module docstring.")


if __name__ == "__main__":
    main()
