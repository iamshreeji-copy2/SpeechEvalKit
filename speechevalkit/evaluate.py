from __future__ import annotations

import csv
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from speechevalkit.metrics.cosine import cosine_similarity
from speechevalkit.metrics.mcd import mcd
from speechevalkit.metrics.pesq import pesq_score
from speechevalkit.metrics.si_sdr import si_sdr
from speechevalkit.metrics.stoi import stoi_score
from speechevalkit.utils.alignment import align_pair
from speechevalkit.utils.audio import load_audio
from speechevalkit.utils.io import find_matched_audio_files, save_results_json

LOGGER = logging.getLogger(__name__)

AudioMetricFn = Callable[[np.ndarray, np.ndarray, int], float]

SUPPORTED_AUDIO_METRICS: dict[str, AudioMetricFn] = {
    "pesq": pesq_score,
    "stoi": stoi_score,
    "si_sdr": si_sdr,
    "mcd": mcd,
    "cosine": cosine_similarity,
}

SUPPORTED_TEXT_METRICS = {
    "wer",
    "cer",
    "mer",
}

SUPPORTED_SINGLE_AUDIO_METRICS = {
    "nisqa_mos",
    "dnsmos",
}


def _setup_logging(verbose: bool = True) -> None:
    level = logging.INFO if verbose else logging.WARNING

    if logging.getLogger().handlers:
        logging.getLogger().setLevel(level)
        return

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def _normalize_metric_name(metric: str) -> str:
    return metric.strip().lower().replace("-", "_")


def _normalize_metrics(metrics: list[str] | tuple[str, ...] | None) -> list[str]:
    if metrics is None:
        return ["pesq", "stoi", "mcd", "si_sdr", "cosine"]

    normalized = [_normalize_metric_name(m) for m in metrics]

    supported = (
        set(SUPPORTED_AUDIO_METRICS)
        | SUPPORTED_TEXT_METRICS
        | SUPPORTED_SINGLE_AUDIO_METRICS
    )

    unknown = [m for m in normalized if m not in supported]
    if unknown:
        raise ValueError(
            f"Unsupported metric(s): {unknown}. "
            f"Supported metrics: {sorted(supported)}"
        )

    return normalized


def _nanmean(values: list[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)

    if arr.size == 0 or np.all(np.isnan(arr)):
        return float("nan")

    return float(np.nanmean(arr))


def _resolve_num_workers(num_workers: int | str | None) -> int:
    import os

    cpu_count = os.cpu_count() or 1

    if num_workers is None:
        return 1

    if isinstance(num_workers, str):
        value = num_workers.strip().lower()

        if value in {"auto", "cpu", "max"}:
            return max(1, cpu_count - 1)

        try:
            parsed = int(value)
        except ValueError as exc:
            raise ValueError(
                "num_workers must be an integer, 'auto', or None."
            ) from exc

        return max(1, min(parsed, cpu_count))

    return max(1, min(int(num_workers), cpu_count))


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _save_results_csv(results: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    metrics = results["metadata"]["metrics"]

    fieldnames = [
        "file",
        "ref_path",
        "pred_path",
        "pair_score",
        *metrics,
    ]

    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for item in results["per_file"]:
            row = {
                "file": item.get("file", ""),
                "ref_path": item.get("ref_path", ""),
                "pred_path": item.get("pred_path", ""),
                "pair_score": item.get("pair_score", ""),
            }

            for metric in metrics:
                row[metric] = item.get("metrics", {}).get(metric, "")

            writer.writerow(row)


def _read_text_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore").strip()


def _collect_text_files(directory: str | Path, recursive: bool = False) -> list[Path]:
    directory = Path(directory)
    pattern = "**/*.txt" if recursive else "*.txt"
    return sorted(p for p in directory.glob(pattern) if p.is_file())


def _match_text_files(
    ref_text_dir: str | Path,
    pred_text_dir: str | Path,
    recursive: bool = False,
) -> tuple[list[tuple[Path, Path]], list[Path]]:
    ref_dir = Path(ref_text_dir)
    pred_dir = Path(pred_text_dir)

    ref_files = _collect_text_files(ref_dir, recursive=recursive)
    pred_files = _collect_text_files(pred_dir, recursive=recursive)

    pred_map: dict[str, Path] = {}

    for pred in pred_files:
        key = str(pred.relative_to(pred_dir).with_suffix("")) if recursive else pred.stem
        pred_map[key] = pred

    matched: list[tuple[Path, Path]] = []
    missing: list[Path] = []

    for ref in ref_files:
        key = str(ref.relative_to(ref_dir).with_suffix("")) if recursive else ref.stem
        pred = pred_map.get(key)

        if pred is None:
            missing.append(ref)
        else:
            matched.append((ref, pred))

    return matched, missing


def _compute_text_metric(metric: str, reference: str, hypothesis: str) -> float:
    if metric == "wer":
        from speechevalkit.metrics.wer import wer_score

        return float(wer_score(reference, hypothesis))

    if metric == "cer":
        from speechevalkit.metrics.cer import cer_score

        return float(cer_score(reference, hypothesis))

    if metric == "mer":
        from speechevalkit.metrics.mer import mer_score

        return float(mer_score(reference, hypothesis))

    raise ValueError(f"Unsupported text metric: {metric}")


def _compute_single_audio_metric(
    metric: str,
    pred_audio: np.ndarray,
    sample_rate: int,
) -> Any:
    if metric == "nisqa_mos":
        from speechevalkit.metrics.nisqa import nisqa_mos_score

        return float(nisqa_mos_score(pred_audio, sample_rate))

    if metric == "dnsmos":
        from speechevalkit.metrics.dnsmos import dnsmos_score

        value = dnsmos_score(pred_audio, sample_rate)

        if isinstance(value, dict):
            return {k: float(v) for k, v in value.items()}

        return float(value)

    raise ValueError(f"Unsupported single-audio metric: {metric}")


def _get_audio_pairs(
    ref_path: Path,
    pred_path: Path,
    recursive: bool,
    fuzzy_match: bool,
    match_threshold: float,
) -> tuple[list[tuple[Path, Path, float | None]], list[Path]]:
    if fuzzy_match:
        try:
            from speechevalkit.utils.pairing import fuzzy_match_audio_files
        except ImportError as exc:
            raise ImportError(
                "Fuzzy matching requires speechevalkit.utils.pairing.py. "
                "Please add the pairing.py file first."
            ) from exc

        fuzzy_pairs, missing = fuzzy_match_audio_files(
            ref_path,
            pred_path,
            recursive=recursive,
            threshold=match_threshold,
        )

        pairs = [(ref, pred, score) for ref, pred, score in fuzzy_pairs]
        return pairs, missing

    matched_pairs, missing = find_matched_audio_files(
        ref_path,
        pred_path,
        recursive=recursive,
    )

    pairs = [(ref, pred, None) for ref, pred in matched_pairs]
    return pairs, missing


def _evaluate_audio_pair_worker(args: dict[str, Any]) -> dict[str, Any]:
    ref_file: Path = args["ref_file"]
    pred_file: Path = args["pred_file"]
    pair_score: float | None = args["pair_score"]
    selected_metrics: list[str] = args["metrics"]
    sample_rate: int = args["sample_rate"]

    file_result: dict[str, Any] = {
        "file": ref_file.name,
        "ref_path": str(ref_file),
        "pred_path": str(pred_file),
        "pair_score": pair_score,
        "metrics": {},
        "error": None,
    }

    try:
        ref_audio, sr = load_audio(ref_file, target_sr=sample_rate, mono=True)
        pred_audio, _ = load_audio(pred_file, target_sr=sample_rate, mono=True)
        ref_audio, pred_audio = align_pair(ref_audio, pred_audio)
    except Exception as exc:
        file_result["error"] = f"audio loading/alignment failed: {exc}"
        return file_result

    for metric_name in selected_metrics:
        try:
            if metric_name in SUPPORTED_AUDIO_METRICS:
                metric_fn = SUPPORTED_AUDIO_METRICS[metric_name]
                score = metric_fn(ref_audio, pred_audio, sr)
                file_result["metrics"][metric_name] = float(score)

            elif metric_name in SUPPORTED_SINGLE_AUDIO_METRICS:
                score = _compute_single_audio_metric(metric_name, pred_audio, sr)

                if isinstance(score, dict):
                    for key, value in score.items():
                        file_result["metrics"][key] = float(value)
                else:
                    file_result["metrics"][metric_name] = float(score)

            else:
                file_result["metrics"][metric_name] = float("nan")

        except Exception as exc:
            file_result["metrics"][metric_name] = float("nan")
            file_result.setdefault("metric_errors", {})[metric_name] = str(exc)

    return file_result


def evaluate_text(
    ref_text_dir: str | Path,
    pred_text_dir: str | Path,
    metrics: list[str] | tuple[str, ...] | None = None,
    recursive: bool = False,
    show_progress: bool = True,
    save_json: str | Path | None = None,
    save_csv: str | Path | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Evaluate text transcripts using WER, CER, and MER.

    Parameters
    ----------
    ref_text_dir:
        Directory containing reference .txt transcripts.
    pred_text_dir:
        Directory containing hypothesis .txt transcripts.
    metrics:
        Text metrics to compute. Supported: wer, cer, mer.
    recursive:
        Whether to match transcript files recursively.
    show_progress:
        Whether to show tqdm progress bar.
    save_json:
        Optional JSON output path.
    save_csv:
        Optional CSV output path.
    verbose:
        Whether to log information.

    Returns
    -------
    dict
        Evaluation results.
    """
    _setup_logging(verbose=verbose)

    selected_metrics = _normalize_metrics(metrics)
    selected_metrics = [m for m in selected_metrics if m in SUPPORTED_TEXT_METRICS]

    if not selected_metrics:
        raise ValueError("No text metrics selected. Use: wer, cer, mer.")

    ref_dir = Path(ref_text_dir)
    pred_dir = Path(pred_text_dir)

    if not ref_dir.exists():
        raise FileNotFoundError(f"Reference text directory does not exist: {ref_dir}")
    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction text directory does not exist: {pred_dir}")

    matched_pairs, missing_predictions = _match_text_files(
        ref_dir,
        pred_dir,
        recursive=recursive,
    )

    if not matched_pairs:
        raise RuntimeError(
            f"No matching transcript files found between {ref_dir} and {pred_dir}"
        )

    per_file: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    iterator = tqdm(
        matched_pairs,
        desc="Evaluating text",
        unit="file",
        disable=not show_progress,
    )

    for ref_file, pred_file in iterator:
        item: dict[str, Any] = {
            "file": ref_file.name,
            "ref_path": str(ref_file),
            "pred_path": str(pred_file),
            "pair_score": None,
            "metrics": {},
        }

        try:
            reference = _read_text_file(ref_file)
            hypothesis = _read_text_file(pred_file)
        except Exception as exc:
            skipped.append(
                {
                    "file": ref_file.name,
                    "reason": f"text loading failed: {exc}",
                }
            )
            continue

        for metric in selected_metrics:
            try:
                item["metrics"][metric] = float(
                    _compute_text_metric(metric, reference, hypothesis)
                )
            except Exception as exc:
                item["metrics"][metric] = float("nan")
                item.setdefault("metric_errors", {})[metric] = str(exc)

        per_file.append(item)

    summary = {}
    for metric in selected_metrics:
        values = [item["metrics"].get(metric, float("nan")) for item in per_file]
        summary[metric] = _nanmean(values)

    results: dict[str, Any] = {
        "summary": summary,
        "per_file": per_file,
        "skipped": skipped,
        "missing_predictions": [str(path) for path in missing_predictions],
        "metadata": {
            "task": "text",
            "ref_text_dir": str(ref_dir),
            "pred_text_dir": str(pred_dir),
            "metrics": selected_metrics,
            "num_matched": len(matched_pairs),
            "num_evaluated": len(per_file),
            "num_skipped": len(skipped),
            "recursive": recursive,
        },
    }

    if save_json is not None:
        save_results_json(results, save_json)

    if save_csv is not None:
        _save_results_csv(results, save_csv)

    return results


def evaluate(
    ref_dir: str | Path,
    pred_dir: str | Path,
    metrics: list[str] | tuple[str, ...] | None = None,
    sample_rate: int = 16000,
    recursive: bool = False,
    show_progress: bool = True,
    save_json: str | Path | None = None,
    save_csv: str | Path | None = None,
    verbose: bool = True,
    fuzzy_match: bool = False,
    match_threshold: float = 0.72,
    num_workers: int | str | None = 1,
) -> dict[str, Any]:
    """
    Evaluate generated speech against reference speech.

    Parameters
    ----------
    ref_dir:
        Directory containing reference audio.
    pred_dir:
        Directory containing predicted/generated audio.
    metrics:
        Metrics to compute.

        Supported paired-audio metrics:
            pesq, stoi, mcd, si_sdr, cosine

        Supported optional non-intrusive metrics:
            nisqa_mos, dnsmos

        Text metrics such as wer/cer/mer should use evaluate_text().
    sample_rate:
        Target sample rate used for loading and metric computation.
    recursive:
        If True, recursively searches subdirectories.
    show_progress:
        If True, displays tqdm progress bar.
    save_json:
        Optional path to save detailed results as JSON.
    save_csv:
        Optional path to save per-file results as CSV.
    verbose:
        If True, logs informational messages.
    fuzzy_match:
        If True, match files using filename similarity instead of exact stems.
    match_threshold:
        Minimum fuzzy filename similarity in [0, 1].
    num_workers:
        Number of parallel CPU workers.
        Use 1 for serial execution.
        Use "auto" to use CPU count - 1.

    Returns
    -------
    dict
        Dictionary with summary averages, per-file scores, skipped files,
        missing files, and metadata.
    """
    _setup_logging(verbose=verbose)

    ref_path = Path(ref_dir)
    pred_path = Path(pred_dir)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference directory does not exist: {ref_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_path}")
    if not ref_path.is_dir():
        raise NotADirectoryError(f"Reference path is not a directory: {ref_path}")
    if not pred_path.is_dir():
        raise NotADirectoryError(f"Prediction path is not a directory: {pred_path}")

    selected_metrics = _normalize_metrics(metrics)

    text_metrics = [m for m in selected_metrics if m in SUPPORTED_TEXT_METRICS]
    if text_metrics:
        raise ValueError(
            f"Text metrics {text_metrics} require transcripts. "
            "Use evaluate_text(ref_text_dir=..., pred_text_dir=..., metrics=...)."
        )

    if not 0.0 <= match_threshold <= 1.0:
        raise ValueError("match_threshold must be between 0.0 and 1.0.")

    matched_pairs, missing_predictions = _get_audio_pairs(
        ref_path=ref_path,
        pred_path=pred_path,
        recursive=recursive,
        fuzzy_match=fuzzy_match,
        match_threshold=match_threshold,
    )

    if not matched_pairs:
        raise RuntimeError(
            f"No matching audio files found between {ref_path} and {pred_path}"
        )

    LOGGER.info("Found %d matched files", len(matched_pairs))

    if missing_predictions:
        LOGGER.warning(
            "Missing predictions for %d reference files",
            len(missing_predictions),
        )

    workers = _resolve_num_workers(num_workers)

    tasks = [
        {
            "ref_file": ref_file,
            "pred_file": pred_file,
            "pair_score": pair_score,
            "metrics": selected_metrics,
            "sample_rate": sample_rate,
        }
        for ref_file, pred_file, pair_score in matched_pairs
    ]

    per_file: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    if workers == 1:
        iterator = tqdm(
            tasks,
            desc="Evaluating",
            unit="file",
            disable=not show_progress,
        )

        for task in iterator:
            result = _evaluate_audio_pair_worker(task)

            if result.get("error"):
                LOGGER.warning("Skipping %s: %s", result["file"], result["error"])
                skipped.append(
                    {
                        "file": result["file"],
                        "reason": result["error"],
                    }
                )
                continue

            per_file.append(result)

    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_evaluate_audio_pair_worker, task) for task in tasks]

            iterator = tqdm(
                as_completed(futures),
                total=len(futures),
                desc=f"Evaluating ({workers} workers)",
                unit="file",
                disable=not show_progress,
            )

            for future in iterator:
                try:
                    result = future.result()
                except Exception as exc:
                    skipped.append(
                        {
                            "file": "unknown",
                            "reason": f"worker failed: {exc}",
                        }
                    )
                    continue

                if result.get("error"):
                    LOGGER.warning("Skipping %s: %s", result["file"], result["error"])
                    skipped.append(
                        {
                            "file": result["file"],
                            "reason": result["error"],
                        }
                    )
                    continue

                per_file.append(result)

    summary_metrics: set[str] = set(selected_metrics)

    for item in per_file:
        for metric_name in item.get("metrics", {}):
            summary_metrics.add(metric_name)

    summary: dict[str, float] = {}

    for metric_name in sorted(summary_metrics):
        values = [
            _safe_float(item.get("metrics", {}).get(metric_name, float("nan")))
            for item in per_file
        ]
        summary[metric_name] = _nanmean(values)

    result: dict[str, Any] = {
        "summary": summary,
        "per_file": per_file,
        "skipped": skipped,
        "missing_predictions": [str(path) for path in missing_predictions],
        "metadata": {
            "task": "audio",
            "ref_dir": str(ref_path),
            "pred_dir": str(pred_path),
            "sample_rate": sample_rate,
            "metrics": selected_metrics,
            "num_matched": len(matched_pairs),
            "num_evaluated": len(per_file),
            "num_skipped": len(skipped),
            "recursive": recursive,
            "fuzzy_match": fuzzy_match,
            "match_threshold": match_threshold,
            "num_workers": workers,
        },
    }

    if save_json is not None:
        save_results_json(result, save_json)
        LOGGER.info("Saved JSON results to %s", save_json)

    if save_csv is not None:
        _save_results_csv(result, save_csv)
        LOGGER.info("Saved CSV results to %s", save_csv)

    return result