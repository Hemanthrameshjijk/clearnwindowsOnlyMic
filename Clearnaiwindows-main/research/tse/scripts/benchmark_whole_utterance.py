"""Research-only: repeated whole-utterance timing + peak RSS for the
REAL-TSE spk_emb_causal_100 checkpoint on CPU. Not part of the ClearNAI
production build. This measures whole-buffer (non-chunked) forward-pass
cost only - see load_real_tse.py's docs on why a true 10ms-chunk streaming
measurement requires further engineering this script does not attempt.
"""
import sys
import time
import types
import yaml
import torch
import torchaudio
import psutil
import os

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


def disable_pretrained(cfg):
    if isinstance(cfg, dict):
        return {k: (None if k == "pretrained" else disable_pretrained(v)) for k, v in cfg.items()}
    return cfg


def main():
    torch.set_num_threads(1)  # single-thread, to get a comparable RTF baseline
    with open(f"{CKPT_DIR}/config.yaml", "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = disable_pretrained(configs)
    model_args = configs["model_args"]["tse_model"]

    model = TSE_BSRNN_SPK(model_args)
    ckpt = torch.load(f"{CKPT_DIR}/avg_model.pt", map_location="cpu")
    state = ckpt["models"][0]
    model.load_state_dict(state, strict=True)
    model.eval()

    proc = psutil.Process(os.getpid())
    rss_before = proc.memory_info().rss / 1e6

    sr = 16000
    durations_s = [1.0, 3.0, 5.0]
    enroll = torch.randn(1, sr * 3) * 0.1

    for dur in durations_s:
        mix = torch.randn(1, int(sr * dur)) * 0.1
        times = []
        with torch.no_grad():
            for _ in range(5):
                t0 = time.perf_counter()
                out = model(mix, enroll)
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)
        times.sort()
        rtf = (sum(times) / len(times)) / (dur * 1000)
        print(
            f"dur={dur:.1f}s  runs_ms={[round(t, 1) for t in times]}  "
            f"median_ms={times[len(times)//2]:.1f}  rtf={rtf:.3f}  "
            f"output_finite={torch.isfinite(out).all().item()}"
        )

    rss_after = proc.memory_info().rss / 1e6
    print(f"RSS before: {rss_before:.1f} MB, after: {rss_after:.1f} MB")


if __name__ == "__main__":
    main()
