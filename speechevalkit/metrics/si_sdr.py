from __future__ import annotations

import numpy as np


def si_sdr(ref: np.ndarray, pred: np.ndarray, sr: int | None = None) -> float:
    """
    Compute Scale-Invariant Signal-to-Distortion Ratio.

    Parameters
    ----------
    ref:
        Reference mono waveform.
    pred:
        Predicted mono waveform.
    sr:
        Unused, kept for unified metric signature.

    Returns
    -------
    float
        SI-SDR in dB.
    """
    _ = sr

    if ref.size == 0 or pred.size == 0:
        return float("nan")

    eps = 1e-10

    ref = ref.astype(np.float64)
    pred = pred.astype(np.float64)

    ref = ref - np.mean(ref)
    pred = pred - np.mean(pred)

    ref_energy = np.sum(ref**2) + eps
    scale = np.sum(pred * ref) / ref_energy

    target = scale * ref
    noise = pred - target

    ratio = (np.sum(target**2) + eps) / (np.sum(noise**2) + eps)
    return float(10.0 * np.log10(ratio + eps))
