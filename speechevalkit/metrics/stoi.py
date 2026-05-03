from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from scipy.signal import stft


def _fallback_stoi_like(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    """
    Lightweight STOI-style intelligibility proxy.

    This fallback computes short-time spectral-envelope correlation and
    returns a score in [0, 1]. If the optional `pystoi` package is installed,
    `stoi_score` uses the real STOI algorithm.
    """
    _ = sr

    if ref.size < 256 or pred.size < 256:
        return float("nan")

    _, _, ref_spec = stft(ref, nperseg=384, noverlap=192)
    _, _, pred_spec = stft(pred, nperseg=384, noverlap=192)

    ref_env = np.log1p(np.abs(ref_spec))
    pred_env = np.log1p(np.abs(pred_spec))

    min_t = min(ref_env.shape[1], pred_env.shape[1])
    ref_env = ref_env[:, :min_t]
    pred_env = pred_env[:, :min_t]

    if min_t == 0:
        return float("nan")

    correlations = []
    for i in range(min_t):
        x = ref_env[:, i]
        y = pred_env[:, i]

        if np.std(x) < 1e-10 or np.std(y) < 1e-10:
            continue

        corr = float(np.corrcoef(x, y)[0, 1])
        correlations.append(corr)

    if not correlations:
        return float("nan")

    score = (float(np.mean(correlations)) + 1.0) / 2.0
    return float(np.clip(score, 0.0, 1.0))


def stoi_score(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    """
    Compute STOI intelligibility score.

    If the optional third-party `pystoi` package is available, this function
    uses the real STOI implementation. Otherwise, it uses a dependency-safe
    STOI-like fallback based on spectral-envelope correlation.

    Parameters
    ----------
    ref:
        Reference mono waveform.
    pred:
        Predicted mono waveform.
    sr:
        Sample rate.

    Returns
    -------
    float
        STOI or STOI-like score in [0, 1].
    """
    if ref.size == 0 or pred.size == 0:
        return float("nan")

    if importlib.util.find_spec("pystoi") is not None:
        try:
            from pystoi.stoi import stoi as real_stoi  # type: ignore

            return float(real_stoi(ref, pred, sr, extended=False))
        except Exception as exc:
            warnings.warn(
                f"Real STOI failed, using fallback STOI-like proxy: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return _fallback_stoi_like(ref, pred, sr)
