from __future__ import annotations

from datetime import datetime
from pathlib import Path


def create_experiment_dir(
    root_dir: str | Path = "results/experiments",
    task_name: str = "experiment",
) -> Path:
    """
    Create an auto-increment experiment directory with date and time.

    Example:
        results/experiments/exp_001_2026-05-03_11-42-10_mos
    """
    root = Path(root_dir)
    root.mkdir(parents=True, exist_ok=True)

    existing = sorted(
        p for p in root.iterdir()
        if p.is_dir() and p.name.startswith("exp_")
    )

    max_id = 0
    for path in existing:
        parts = path.name.split("_")
        if len(parts) >= 2:
            try:
                max_id = max(max_id, int(parts[1]))
            except ValueError:
                pass

    exp_id = max_id + 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    safe_task = task_name.lower().replace(" ", "_").replace("-", "_")

    exp_name = f"exp_{exp_id:03d}_{timestamp}_{safe_task}"
    exp_dir = root / exp_name
    exp_dir.mkdir(parents=True, exist_ok=False)

    return exp_dir


def build_experiment_csv_paths(
    task_name: str,
    root_dir: str | Path = "results/experiments",
    base_filename: str | None = None,
) -> tuple[Path, Path, Path]:
    """
    Create experiment directory and return:
        experiment_dir, result_csv_path, summary_csv_path
    """
    exp_dir = create_experiment_dir(root_dir=root_dir, task_name=task_name)

    if base_filename is None:
        base_filename = f"{task_name}_results.csv"

    result_csv = exp_dir / base_filename
    summary_csv = exp_dir / base_filename.replace(".csv", "_summary.csv")

    return exp_dir, result_csv, summary_csv
