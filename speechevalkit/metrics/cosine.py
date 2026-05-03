from __future__ import annotations

import numpy as np
from scipy.fftpack import dct
from scipy.signal import stft


def _mfcc_embedding(audio: np.ndarray, sr: int) -> np.ndarray:
    try:
        import librosa

        mfcc = librosa.feature.mfcc(
            y=audio.astype(np.float32),
            sr=sr,
            n_mfcc=20,
            n_fft=1024,
            hop_length=256,
        )
        delta = librosa.feature.delta(mfcc)
        delta2 = librosa.feature.delta(mfcc, order=2)
        features = np.concatenate([mfcc, delta, delta2], axis=0)
    except Exception:
        _, _, spec = stft(audio, fs=sr, nperseg=1024, noverlap=768)
        log_spec = np.log1p(np.abs(spec))
        mfcc = dct(log_spec.T, type=2, axis=1, norm="ortho")[:, :20].T
        features = mfcc

    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    return np.concatenate([mean, std], axis=0).astype(np.float64)


def _torch_cosine(a: np.ndarray, b: np.ndarray) -> float | None:
    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        ta = torch.tensor(a, dtype=torch.float32, device=device)
        tb = torch.tensor(b, dtype=torch.float32, device=device)
        score = torch.nn.functional.cosine_similarity(ta, tb, dim=0)
        return float(score.detach().cpu().item())
    except Exception:
        return None


def cosine_similarity(ref: np.ndarray, pred: np.ndarray, sr: int) -> float:
    """
    Compute embedding-style cosine similarity.

    The default embedding is a compact MFCC-statistics representation.
    If PyTorch is installed, cosine computation may use GPU automatically.

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
        Cosine similarity in [-1, 1].
    """
    if ref.size == 0 or pred.size == 0:
        return float("nan")

    ref_emb = _mfcc_embedding(ref, sr)
    pred_emb = _mfcc_embedding(pred, sr)

    torch_score = _torch_cosine(ref_emb, pred_emb)
    if torch_score is not None:
        return float(np.clip(torch_score, -1.0, 1.0))

    denom = np.linalg.norm(ref_emb) * np.linalg.norm(pred_emb)
    if denom < 1e-12:
        return float("nan")

    score = float(np.dot(ref_emb, pred_emb) / denom)
    return float(np.clip(score, -1.0, 1.0))
