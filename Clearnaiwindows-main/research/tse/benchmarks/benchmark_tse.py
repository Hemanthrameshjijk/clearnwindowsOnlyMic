"""Multi-scenario sanity benchmark for the REAL-TSE `spk_emb_causal_100`
checkpoint.

IMPORTANT HONESTY NOTE (read before trusting any number this script
prints): this environment has no real multi-speaker speech corpus
available, so every "speaker" here is a synthetic sine/noise signal, not
real human speech. These are SANITY CHECKS (does the model do *something*
resembling source separation on trivially-separable synthetic signals),
NOT quality/accuracy measurements. Do not quote the SI-SDR numbers this
script prints as if they were measured on real speech - they are not. Real
quality benchmarking (per docs/TSE_INTEGRATION_PLAN.md Step 6) requires a
real speech corpus this session does not have.

Not part of the ClearNAI production build.
"""
import sys
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


def tone(freq, secs, amp=0.3, sr=SR):
    n = int(sr * secs)
    t = torch.arange(n, dtype=torch.float32) / sr
    return (amp * torch.sin(2 * torch.pi * freq * t)).unsqueeze(0)


def noise(secs, amp=0.1, sr=SR):
    return amp * torch.randn(1, int(sr * secs))


def si_sdr(estimate: torch.Tensor, reference: torch.Tensor, eps=1e-8) -> float:
    """Standard scale-invariant SDR, in dB. Plain implementation (no extra
    dependency) - matches the well-known SI-SDR formula."""
    estimate = estimate.flatten()
    reference = reference.flatten()
    n = min(estimate.numel(), reference.numel())
    estimate, reference = estimate[:n], reference[:n]
    reference = reference - reference.mean()
    estimate = estimate - estimate.mean()
    alpha = (estimate @ reference) / (reference @ reference + eps)
    target = alpha * reference
    noise_ = estimate - target
    return 10 * torch.log10((target @ target + eps) / (noise_ @ noise_ + eps)).item()


def run_scenario(model, name, target, mixture, enroll):
    """target: the clean target-only reference (for SI-SDR comparison).
    mixture: what's fed to the model as the mic input.
    enroll: the enrollment clip."""
    with torch.no_grad():
        out = model(mixture.unsqueeze(0), enroll.unsqueeze(0)).squeeze(0)
    input_sdr = si_sdr(mixture, target)
    output_sdr = si_sdr(out, target)
    print(f"[{name:32s}] input_SI-SDR={input_sdr:7.2f} dB  output_SI-SDR={output_sdr:7.2f} dB  "
          f"delta={output_sdr - input_sdr:+6.2f} dB  finite={torch.isfinite(out).all().item()}")
    return output_sdr - input_sdr


def main():
    model = load_model()
    target_freq, bg_freq_1, bg_freq_2 = 220.0, 330.0, 415.0

    enroll = tone(target_freq, 3.0, amp=0.4)

    scenarios = []

    # 1. target speaker only
    target = tone(target_freq, 4.0)
    scenarios.append(("target_only", target, target.clone()))

    # 2. target + one background "speaker"
    target = tone(target_freq, 4.0)
    bg = tone(bg_freq_1, 4.0)
    scenarios.append(("target_plus_one_background", target, target + bg))

    # 3. target + two background "speakers"
    target = tone(target_freq, 4.0)
    bg = tone(bg_freq_1, 4.0) + tone(bg_freq_2, 4.0)
    scenarios.append(("target_plus_two_background", target, target + bg))

    # 4. target + background noise
    target = tone(target_freq, 4.0)
    scenarios.append(("target_plus_noise", target, target + noise(4.0)))

    # 5. target + noise + background speaker
    target = tone(target_freq, 4.0)
    bg = tone(bg_freq_1, 4.0)
    scenarios.append(("target_plus_noise_plus_bg", target, target + bg + noise(4.0, amp=0.05)))

    # 6. overlapping speech (both active the whole time - same as #2, kept
    #    separate to match the brief's scenario list explicitly)
    target = tone(target_freq, 4.0)
    bg = tone(bg_freq_1, 4.0)
    scenarios.append(("overlapping_speech", target, target + bg))

    # 7. target starts/stops (silent for first and last second)
    target = tone(target_freq, 4.0)
    target[:, : SR] = 0.0
    target[:, -SR:] = 0.0
    bg = tone(bg_freq_1, 4.0)
    scenarios.append(("target_starts_stops", target, target + bg))

    # 8. background starts/stops
    target = tone(target_freq, 4.0)
    bg = tone(bg_freq_1, 4.0)
    bg[:, : SR] = 0.0
    bg[:, -SR:] = 0.0
    scenarios.append(("background_starts_stops", target, target + bg))

    # 9. silence
    target = torch.zeros(1, int(SR * 4.0))
    scenarios.append(("silence", target, target.clone()))

    # 10. near-end target (loud)
    target = tone(target_freq, 4.0, amp=0.8)
    bg = tone(bg_freq_1, 4.0, amp=0.1)
    scenarios.append(("near_end_target", target, target + bg))

    # 11. far-end target (quiet)
    target = tone(target_freq, 4.0, amp=0.05)
    bg = tone(bg_freq_1, 4.0, amp=0.4)
    scenarios.append(("far_end_target", target, target + bg))

    # 12. reverberation: not attempted - no RIR (room impulse response) data
    # available in this environment; explicitly skipped, not faked.
    print("Scenario 'reverberation' skipped: no RIR/reverb data available in this environment.\n")

    print("=== Synthetic-signal sanity benchmark (NOT a real-speech quality measurement) ===")
    deltas = []
    for name, target, mixture in scenarios:
        d = run_scenario(model, name, target, mixture, enroll)
        deltas.append(d)
    print(f"\nmean SI-SDR improvement across scenarios: {sum(deltas) / len(deltas):+.2f} dB "
          "(synthetic tones only - see module docstring)")


if __name__ == "__main__":
    main()
