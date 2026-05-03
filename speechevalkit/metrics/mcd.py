from __future__ import annotations

import numpy as np
import scipy.spatial.distance
from scipy.fftpack import dct
from scipy.signal import stft


def _log_mel_spectrogram(
    audio: np.ndarray,
    sr: int,
    n_fft: int = 1024,
    hop_length: int = 256,
    n_mels: int = 40,
) -> np.ndarray:
    try:
        import librosa

        mel = librosa.feature.melspectrogram(
            y=audio.astype(np.float32),
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        return np.log(np.maximum(mel, 1e-10)).T
    except Exception:
        _, _, spec = stft(
            audio,
            fs=sr,
            nperseg=n_fft,
            noverlap=n_fft - hop_length,
            nfft=n_fft,
        )
        power = np.abs(spec) ** 2
        power = power[:n_mels, :]
        return np.log(np.maximum(power, 1e-10)).T


def _mfcc_like(audio: np.ndarray, sr: int, n_mfcc: int = 24) -> np.ndarray:
    log_mel = _log_mel_spectrogram(audio, sr)
    coeffs = dct(log_mel, type=2, axis=1, norm="ortho")
    return coeffs[:, 1 : n_mfcc + 1]


def _dtw_path_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute average Euclidean distance along a DTW path.

    This uses a memory-conscious dynamic programming implementation with
    backpointers suitable for typical utterance-level speech files.
    """
    n, m = x.shape[0], y.shape[0]

    if n == 0 or m == 0:
        return float("nan")

    dist = scipy.spatial.distance.cdist(x, y, metric="euclidean")

    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    cost[0, 0] = 0.0

    back = np.zeros((n, m), dtype=np.uint8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            choices = (
                cost[i - 1, j],
                cost[i, j - 1],
                cost[i - 1, j - 1],
            )
            arg = int(np.argmin(choices))
            cost[i, j] = dist[i - 1, j - 1] + choices[arg]
            back[i - 1, j - 1] = arg

    i, j = n - 1, m - 1
    path_distances = []

    while i >= 0 and j >= 0:
        path_distances.append(dist[i, j])
        step = back[i, j]

        if step == 0:
            i -= 1
        elif step == 1:
            j -= 1
        else:
            i -= 1
            j -= 1

    if not path_distances:
        return float("nan")

    return float(np.mean(path_distances))


def mcd(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    """
    Compute Mel Cepstral Distortion.

    Lower is better.

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
        MCD in dB.
    """
    if ref.size == 0 or pred.size == 0:
        return float("nan")

    ref_mfcc = _mfcc_like(ref, sr)
    pred_mfcc = _mfcc_like(pred, sr)

    if ref_mfcc.size == 0 or pred_mfcc.size == 0:
        return float("nan")

    avg_dist = _dtw_path_distance(ref_mfcc, pred_mfcc)

    if avg_dist != avg_dist:
        return float("nan")

    factor = 10.0 / np.log(10.0) * np.sqrt(2.0)
    return float(factor * avg_dist)
