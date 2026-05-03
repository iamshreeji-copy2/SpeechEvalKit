from __future__ import annotations

import importlib.util
import warnings

import numpy as np
from scipy.signal import stft


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    value = float(np.corrcoef(x, y)[0, 1])
    return float(np.clip(value, -1.0, 1.0))


def _fallback_pesq_like(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    """
    Dependency-safe PESQ-style proxy.

    This is not the ITU-T PESQ algorithm. If the optional external
    `pesq` package is installed, `pesq_score` uses the real PESQ score.
    Otherwise this fallback returns a bounded perceptual proxy on a
    PESQ-like 1.0 to 4.5 range using waveform correlation, spectral
    distance, and energy consistency.
    """
    eps = 1e-10

    ref = ref.astype(np.float64)
    pred = pred.astype(np.float64)

    corr = (_safe_corrcoef(ref, pred) + 1.0) / 2.0

    _, _, ref_spec = stft(ref, fs=sr, nperseg=512, noverlap=256)
    _, _, pred_spec = stft(pred, fs=sr, nperseg=512, noverlap=256)

    ref_mag = np.log1p(np.abs(ref_spec))
    pred_mag = np.log1p(np.abs(pred_spec))

    min_t = min(ref_mag.shape[1], pred_mag.shape[1])
    ref_mag = ref_mag[:, :min_t]
    pred_mag = pred_mag[:, :min_t]

    spectral_rmse = float(np.sqrt(np.mean((ref_mag - pred_mag) ** 2)))
    spectral_score = float(np.exp(-spectral_rmse))

    ref_energy = float(np.mean(ref**2) + eps)
    pred_energy = float(np.mean(pred**2) + eps)
    energy_ratio_db = abs(10.0 * np.log10(pred_energy / ref_energy))
    energy_score = float(np.exp(-energy_ratio_db / 10.0))

    quality = 0.50 * corr + 0.35 * spectral_score + 0.15 * energy_score
    return float(np.clip(1.0 + 3.5 * quality, 1.0, 4.5))


def pesq_score(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    """
    Compute PESQ score.

    If the optional third-party `pesq` package is available, this function
    uses the real ITU-T PESQ implementation. Otherwise, it falls back to a
    lightweight PESQ-like proxy so SpeechEvalKit remains runnable with the
    core dependency set.

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
        PESQ score or PESQ-like score.
    """
    if ref.size == 0 or pred.size == 0:
        return float("nan")

    if importlib.util.find_spec("pesq") is not None:
        try:
            from pesq import pesq as real_pesq  # type: ignore

            mode = "wb" if sr >= 16000 else "nb"
            valid_sr = 16000 if mode == "wb" else 8000

            if sr != valid_sr:
                warnings.warn(
                    f"Real PESQ expects {valid_sr} Hz for mode={mode}. "
                    "Use evaluate(sample_rate=16000) or evaluate(sample_rate=8000).",
                    RuntimeWarning,
                    stacklevel=2,
                )

            return float(real_pesq(sr, ref, pred, mode))
        except Exception as exc:
            warnings.warn(
                f"Real PESQ failed, using fallback PESQ-like proxy: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

    return _fallback_pesq_like(ref, pred, sr)
