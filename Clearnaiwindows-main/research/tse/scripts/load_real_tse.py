"""Research-only script: load the REAL-TSE (WeSep) spk_emb_causal_100
checkpoint and run one inference pass on synthetic audio to verify it
actually loads and runs. Not part of the ClearNAI production build.
"""
import sys
import time
import yaml
import torch
import torchaudio

REPO = r"C:\Users\Developer\clearnwindowsOnlyMic\Clearnaiwindows-main\research\tse\candidates\wesep-real-tse"
CKPT_DIR = r"C:\Users\DEVELO~1\AppData\Local\Temp\models\real-tse\extracted\spk_emb_causal_100"
sys.path.insert(0, REPO)

# Shim: wespeaker's package __init__ eagerly imports its optional s3prl
# (self-supervised upstream) frontend, whose own dependency chain (s3prl's
# BYOL-S/mos_prediction hub entries) references torchaudio APIs removed in
# modern torchaudio (set_audio_backend, sox_effects). Our model config uses
# ECAPA_TDNN_GLOB_c512 directly and never touches this s3prl frontend at
# all, so short-circuit that entire unused import chain with an empty stub
# module rather than trying to satisfy every transitive incompatibility in
# a toolkit feature we don't use. Research-environment compatibility shim
# only - never shipped to production.
import types  # noqa: E402

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
    with open(f"{CKPT_DIR}/config.yaml", "r") as f:
        configs = yaml.load(f, Loader=yaml.FullLoader)
    configs = disable_pretrained(configs)
    model_args = configs["model_args"]["tse_model"]
    print("model_args:", model_args)

    model = TSE_BSRNN_SPK(model_args)
    ckpt = torch.load(f"{CKPT_DIR}/avg_model.pt", map_location="cpu")
    # Real checkpoint structure (verified by inspection, not assumed):
    # top-level {"models": [...], "optimizers": [...], "schedulers": [...],
    # "scaler": ...} - a full training-loop checkpoint, not a bare
    # state_dict. "models" is a length-1 list holding the actual
    # OrderedDict state_dict for the (single) tse_model.
    state = ckpt["models"][0]
    missing, unexpected = model.load_state_dict(state, strict=False)
    print("missing keys:", len(missing), missing[:5])
    print("unexpected keys:", len(unexpected), unexpected[:5])
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total params: {n_params / 1e6:.2f}M")

    sr = 16000
    mix = torch.randn(1, sr * 3) * 0.1
    enroll = torch.randn(1, sr * 2) * 0.1

    with torch.no_grad():
        t0 = time.time()
        out = model(mix, enroll)
        t1 = time.time()
    print("output shape:", out.shape)
    print("output finite:", torch.isfinite(out).all().item())
    print(f"whole-utterance (3s) forward time: {(t1 - t0) * 1000:.1f} ms")


if __name__ == "__main__":
    main()
