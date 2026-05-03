from __future__ import annotations

import re
from difflib import SequenceMatcher
from pathlib import Path

from speechevalkit.utils.io import AUDIO_EXTENSIONS


def normalize_name(name: str) -> str:
    """
    Normalize filename for fuzzy matching.
    """
    stem = Path(name).stem.lower()

    remove_tokens = [
        "generated",
        "synth",
        "synthetic",
        "prediction",
        "pred",
        "enhanced",
        "cleaned",
        "output",
        "converted",
        "vc",
        "tts",
    ]

    for token in remove_tokens:
        stem = stem.replace(token, "")

    stem = re.sub(r"[^a-z0-9]+", "", stem)
    return stem


def similarity(a: str, b: str) -> float:
    """
    Return filename similarity in [0, 1].
    """
    return SequenceMatcher(None, normalize_name(a), normalize_name(b)).ratio()


def collect_audio_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    directory = Path(directory)
    pattern = "**/*" if recursive else "*"
    return sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def fuzzy_match_audio_files(
    ref_dir: str | Path,
    pred_dir: str | Path,
    recursive: bool = False,
    threshold: float = 0.72,
) -> tuple[list[tuple[Path, Path, float]], list[Path]]:
    """
    Fuzzy-match reference and prediction audio files.

    Parameters
    ----------
    ref_dir:
        Reference audio directory.
    pred_dir:
        Prediction audio directory.
    recursive:
        Whether to search recursively.
    threshold:
        Minimum similarity ratio.

    Returns
    -------
    matched:
        List of (ref_path, pred_path, score).
    unmatched:
        Reference files with no sufficiently similar prediction.
    """
    refs = collect_audio_files(ref_dir, recursive=recursive)
    preds = collect_audio_files(pred_dir, recursive=recursive)

    unused_preds = set(preds)
    matched: list[tuple[Path, Path, float]] = []
    unmatched: list[Path] = []

    for ref in refs:
        best_pred = None
        best_score = -1.0

        for pred in list(unused_preds):
            score = similarity(ref.name, pred.name)
            if score > best_score:
                best_score = score
                best_pred = pred

        if best_pred is not None and best_score >= threshold:
            matched.append((ref, best_pred, best_score))
            unused_preds.remove(best_pred)
        else:
            unmatched.append(ref)

    return matched, unmatched
