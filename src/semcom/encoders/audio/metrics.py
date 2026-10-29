import numpy as np
import torch
from pystoi import stoi as _stoi

try:
    from pesq import NoUtterancesError, pesq as _pesq
except ImportError:  # optional: C extension may be unavailable
    _pesq = None


def si_sdr(est: torch.Tensor, ref: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Scale-invariant SDR in dB for (B, N) waveforms."""
    est = est - est.mean(dim=-1, keepdim=True)
    ref = ref - ref.mean(dim=-1, keepdim=True)
    alpha = (est * ref).sum(-1, keepdim=True) / (ref.pow(2).sum(-1, keepdim=True) + eps)
    target = alpha * ref
    noise = est - target
    return 10 * torch.log10(target.pow(2).sum(-1).clamp_min(eps) / noise.pow(2).sum(-1).clamp_min(eps))


def stoi_mean(est: torch.Tensor, ref: torch.Tensor, sr: int) -> float:
    return float(np.mean([_stoi(r, e, sr, extended=False) for e, r in zip(est.numpy(), ref.numpy())]))


def pesq_mean(est: torch.Tensor, ref: torch.Tensor, sr: int) -> float | None:
    """Mean wideband PESQ over clips PESQ can score; None if unavailable."""
    if _pesq is None:
        return None
    scores = []
    for e, r in zip(est.numpy(), ref.numpy()):
        try:
            scores.append(_pesq(sr, r, e, "wb"))
        except NoUtterancesError:
            continue
    return float(np.mean(scores)) if scores else None


def log_spectral_distance(est: torch.Tensor, ref: torch.Tensor, n_fft: int = 512, hop: int = 128, eps: float = 1e-8,
                          silence_floor_db: float | None = -60.0) -> torch.Tensor:
    """Per-clip log-spectral distance in dB for (B, N) waveforms: RMS over frequency of the difference of log power spectra,
    averaged over frames. 0 is a perfect match. Used for non-speech audio where STOI is not meaningful.

    Frames whose reference energy is more than |silence_floor_db| below the clip's loudest frame are skipped: ESC-50 clips are
    padded with digital silence, where any tiny non-zero reconstruction gives an enormous log-spectral difference that says
    nothing about audible quality. silence_floor_db=None keeps every frame."""
    window = torch.hann_window(n_fft, device=est.device)

    def spec(x):
        return torch.stft(x, n_fft, hop, window=window, return_complex=True).abs().pow(2)

    p_ref, p_est = spec(ref), spec(est)
    d = 10 * torch.log10(p_est + eps) - 10 * torch.log10(p_ref + eps)  # (B, F, T)
    frame_rms = d.pow(2).mean(dim=1).sqrt()  # (B, T)
    energy = p_ref.sum(dim=1)
    keep = torch.ones_like(energy) if silence_floor_db is None else (energy > energy.amax(dim=-1, keepdim=True) * 10 ** (silence_floor_db / 10)).float()
    return (frame_rms * keep).sum(-1) / keep.sum(-1).clamp_min(1.0)
