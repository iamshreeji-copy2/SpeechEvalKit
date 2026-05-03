from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly


def _to_float32(audio: np.ndarray) -> np.ndarray:
    if audio.dtype == np.float32:
        out = audio
    elif audio.dtype == np.float64:
        out = audio.astype(np.float32)
    elif np.issubdtype(audio.dtype, np.integer):
        max_value = np.iinfo(audio.dtype).max
        out = audio.astype(np.float32) / float(max_value)
    else:
        out = audio.astype(np.float32)

    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(out, -1.0, 1.0)


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return np.mean(audio, axis=1)
    raise ValueError(f"Unsupported audio shape: {audio.shape}")


def _resample_scipy(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio.astype(np.float32)

    gcd = np.gcd(orig_sr, target_sr)
    up = target_sr // gcd
    down = orig_sr // gcd
    return resample_poly(audio, up, down).astype(np.float32)


def load_audio(
    path: str | Path,
    target_sr: int = 16000,
    mono: bool = True,
) -> tuple[np.ndarray, int]:
    """
    Load audio, convert to float32, optionally mono, and resample.

    Parameters
    ----------
    path:
        Audio file path.
    target_sr:
        Target sample rate.
    mono:
        If True, converts multichannel audio to mono.

    Returns
    -------
    tuple[np.ndarray, int]
        Waveform and target sample rate.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(path)

    try:
        import librosa

        audio, sr = librosa.load(
            str(path),
            sr=target_sr,
            mono=mono,
        )
        audio = _to_float32(audio)
        return audio, target_sr
    except Exception:
        pass

    try:
        sr, audio = wavfile.read(str(path))
    except Exception as exc:
        raise RuntimeError(f"Could not read audio file {path}: {exc}") from exc

    audio = _to_float32(audio)

    if mono:
        audio = _mono(audio)

    if sr != target_sr:
        audio = _resample_scipy(audio, sr, target_sr)

    if audio.size == 0:
        raise ValueError(f"Loaded empty audio file: {path}")

    return audio.astype(np.float32), target_sr
