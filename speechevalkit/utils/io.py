from __future__ import annotations

import json
from pathlib import Path
from typing import Any

AUDIO_EXTENSIONS = {
    ".wav",
    ".flac",
    ".mp3",
    ".ogg",
    ".m4a",
    ".aac",
    ".aiff",
    ".aif",
}


def _collect_audio_files(directory: Path, recursive: bool = False) -> list[Path]:
    pattern = "**/*" if recursive else "*"
    files = [
        path
        for path in directory.glob(pattern)
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]
    return sorted(files)


def _relative_key(path: Path, root: Path, recursive: bool) -> str:
    if recursive:
        return str(path.relative_to(root).with_suffix(""))
    return path.stem


def find_matched_audio_files(
    ref_dir: Path,
    pred_dir: Path,
    recursive: bool = False,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    """
    Match reference and prediction files by filename stem.

    Parameters
    ----------
    ref_dir:
        Reference directory.
    pred_dir:
        Prediction directory.
    recursive:
        If True, match by relative path without suffix.

    Returns
    -------
    tuple
        Matched pairs and reference files with missing predictions.
    """
    ref_files = _collect_audio_files(ref_dir, recursive=recursive)
    pred_files = _collect_audio_files(pred_dir, recursive=recursive)

    pred_map: dict[str, Path] = {}
    for pred in pred_files:
        key = _relative_key(pred, pred_dir, recursive=recursive)
        pred_map[key] = pred

    matched: list[tuple[Path, Path]] = []
    missing: list[Path] = []

    for ref in ref_files:
        key = _relative_key(ref, ref_dir, recursive=recursive)
        pred = pred_map.get(key)
        if pred is None:
            missing.append(ref)
        else:
            matched.append((ref, pred))

    return matched, missing


def _json_safe(obj: Any) -> Any:
    try:
        import numpy as np

        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
    except Exception:
        pass

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, float):
        if obj != obj:
            return None
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"

    return obj


def save_results_json(results: dict[str, Any], path: str | Path) -> None:
    """
    Save evaluation results to JSON.
    """
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(
            results,
            f,
            indent=2,
            ensure_ascii=False,
            default=_json_safe,
        )
