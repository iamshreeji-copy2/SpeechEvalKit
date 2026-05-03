from __future__ import annotations

import csv
import math
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import wavfile
from tqdm import tqdm


def mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(var))


def confidence_interval_95(values: list[float]) -> tuple[float, float, float]:
    """
    Compute approximate 95% confidence interval.

    Returns
    -------
    tuple
        mean, lower_bound, upper_bound
    """
    if not values:
        return float("nan"), float("nan"), float("nan")

    mu = mean(values)

    if len(values) == 1:
        return mu, mu, mu

    std = sample_std(values)
    margin = 1.96 * std / math.sqrt(len(values))

    return float(mu), float(mu - margin), float(mu + margin)


def nisqa_mos_score(
    audio: np.ndarray,
    sr: int,
    model_path: str | Path | None = None,
) -> float:
    """
    Predict NISQA MOS for a single waveform.

    This function expects an external NISQA command-line tool to be installed.

    Parameters
    ----------
    audio:
        Mono waveform.
    sr:
        Sample rate.
    model_path:
        Optional NISQA model path.

    Returns
    -------
    float
        Predicted MOS score.
    """
    audio = np.asarray(audio, dtype=np.float32)

    if audio.size == 0:
        return float("nan")

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        wav_path = tmpdir_path / "sample.wav"
        out_dir = tmpdir_path / "nisqa_out"
        out_dir.mkdir(parents=True, exist_ok=True)

        wavfile.write(str(wav_path), sr, audio)

        cmd = [
            "nisqa",
            "--mode",
            "predict_file",
            "--deg",
            str(wav_path),
            "--output_dir",
            str(out_dir),
        ]

        if model_path is not None:
            cmd.extend(["--pretrained_model", str(model_path)])

        try:
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise ImportError(
                "NISQA command not found. Install NISQA separately and make "
                "sure the 'nisqa' command is available in PATH."
            ) from exc
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                "NISQA failed.\n\n"
                f"Command:\n{' '.join(cmd)}\n\n"
                f"STDOUT:\n{exc.stdout}\n\n"
                f"STDERR:\n{exc.stderr}"
            ) from exc

        csv_files = sorted(out_dir.glob("*.csv"))

        if not csv_files:
            raise RuntimeError("NISQA did not produce a CSV output file.")

        with csv_files[0].open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)

        mos_keys = [
            "mos_pred",
            "MOS",
            "mos",
            "NISQA_MOS",
            "overall_mos",
            "predicted_mos",
        ]

        for key in mos_keys:
            if key in row:
                return float(row[key])

        raise RuntimeError(
            f"Could not find MOS column in NISQA output. "
            f"Available columns: {list(row.keys())}"
        )


def evaluate_nisqa_directory(
    pred_dir: str | Path,
    sample_rate: int = 16000,
    model_path: str | Path | None = None,
    output_csv: str | Path = "nisqa_mos_results.csv",
    recursive: bool = False,
) -> dict[str, Any]:
    """
    Evaluate NISQA MOS for all WAV files in a directory.

    This is a convenience function for testing NISQA independently from
    paired reference evaluation.

    Parameters
    ----------
    pred_dir:
        Directory containing generated/synthesized WAV files.
    sample_rate:
        Sample rate used when writing temporary audio.
    model_path:
        Optional NISQA model path.
    output_csv:
        Output CSV file.
    recursive:
        If True, scan subdirectories recursively.

    Returns
    -------
    dict
        Summary and per-file NISQA scores.
    """
    from speechevalkit.utils.audio import load_audio
    from speechevalkit.utils.io import AUDIO_EXTENSIONS

    pred_dir = Path(pred_dir)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory does not exist: {pred_dir}")

    pattern = "**/*" if recursive else "*"

    audio_files = sorted(
        p
        for p in pred_dir.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )

    if not audio_files:
        raise RuntimeError(f"No audio files found in {pred_dir}")

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    scores: list[float] = []
    per_file: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "nisqa_mos", "status", "error"])

        for audio_path in tqdm(
            audio_files,
            desc="NISQA MOS",
            unit="file",
        ):
            try:
                audio, sr = load_audio(
                    audio_path,
                    target_sr=sample_rate,
                    mono=True,
                )

                score = nisqa_mos_score(
                    audio=audio,
                    sr=sr,
                    model_path=model_path,
                )

                score = float(score)
                scores.append(score)

                writer.writerow([str(audio_path), score, "ok", ""])

                per_file.append(
                    {
                        "file": str(audio_path),
                        "nisqa_mos": score,
                        "status": "ok",
                        "error": "",
                    }
                )

            except Exception as exc:
                writer.writerow([str(audio_path), "", "skipped", str(exc)])

                skipped.append(
                    {
                        "file": str(audio_path),
                        "reason": str(exc),
                    }
                )

                per_file.append(
                    {
                        "file": str(audio_path),
                        "nisqa_mos": float("nan"),
                        "status": "skipped",
                        "error": str(exc),
                    }
                )

    mu, ci_low, ci_high = confidence_interval_95(scores)

    summary_csv = output_csv.with_name(f"{output_csv.stem}_summary.csv")

    with summary_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "metric",
                "num_files",
                "num_success",
                "num_skipped",
                "mean",
                "ci95_low",
                "ci95_high",
            ]
        )
        writer.writerow(
            [
                "nisqa_mos",
                len(audio_files),
                len(scores),
                len(skipped),
                mu,
                ci_low,
                ci_high,
            ]
        )

    return {
        "summary": {
            "nisqa_mos": mu,
            "nisqa_mos_ci95_low": ci_low,
            "nisqa_mos_ci95_high": ci_high,
        },
        "per_file": per_file,
        "skipped": skipped,
        "metadata": {
            "pred_dir": str(pred_dir),
            "sample_rate": sample_rate,
            "num_files": len(audio_files),
            "num_success": len(scores),
            "num_skipped": len(skipped),
            "output_csv": str(output_csv),
            "summary_csv": str(summary_csv),
        },
    }