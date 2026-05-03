from __future__ import annotations

import numpy as np


def trim_or_pad(audio: np.ndarray, target_length: int) -> np.ndarray:
    """
    Trim or zero-pad audio to a target length.
    """
    if target_length <= 0:
        return np.asarray([], dtype=np.float32)

    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == target_length:
        return audio

    if audio.size > target_length:
        return audio[:target_length]

    padded = np.zeros(target_length, dtype=np.float32)
    padded[: audio.size] = audio
    return padded


def align_pair(
    ref: np.ndarray,
    pred: np.ndarray,
    strategy: str = "min",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align two waveforms by trimming or padding.

    Parameters
    ----------
    ref:
        Reference waveform.
    pred:
        Predicted waveform.
    strategy:
        "min" trims both to the shorter length.
        "max" pads both to the longer length.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Aligned reference and prediction.
    """
    ref = np.asarray(ref, dtype=np.float32).flatten()
    pred = np.asarray(pred, dtype=np.float32).flatten()

    if ref.size == 0 or pred.size == 0:
        raise ValueError("Cannot align empty audio arrays.")

    if strategy == "min":
        length = min(ref.size, pred.size)
    elif strategy == "max":
        length = max(ref.size, pred.size)
    else:
        raise ValueError("strategy must be either 'min' or 'max'")

    return trim_or_pad(ref, length), trim_or_pad(pred, length)
