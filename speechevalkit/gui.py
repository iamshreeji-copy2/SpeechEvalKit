from __future__ import annotations

import csv
import json
import math
import mimetypes
import random
import socket
import threading
import uuid
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tqdm import tqdm

from speechevalkit.utils.io import AUDIO_EXTENSIONS

try:
    from speechevalkit.utils.experiment import create_experiment_dir
except Exception:
    create_experiment_dir = None


ADMIN_USERNAME = "ravindra"
ADMIN_PASSWORD = "speechevalkit"


# ---------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0

    mu = _mean(values)
    variance = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return float(math.sqrt(variance))


def _ci95(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")

    mu = _mean(values)

    if len(values) == 1:
        return mu, mu, mu

    sd = _std(values)
    margin = 1.96 * sd / math.sqrt(len(values))
    return float(mu), float(mu - margin), float(mu + margin)


def _fmt(value: float) -> str:
    try:
        if value != value:
            return "N/A"
        return f"{value:.3f}"
    except Exception:
        return "N/A"


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------


def _collect_audio(directory: str | Path, recursive: bool = False) -> list[Path]:
    directory = Path(directory).expanduser().resolve()

    if not directory.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {directory}")

    pattern = "**/*" if recursive else "*"

    return sorted(
        p
        for p in directory.glob(pattern)
        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
    )


def _safe_relpath(path: Path | None, root: Path | None) -> str:
    if path is None:
        return ""

    if root is None:
        return path.name

    try:
        return str(path.relative_to(root))
    except Exception:
        return path.name


def _safe_name(text: str) -> str:
    keep: list[str] = []

    for ch in text.strip().lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in {" ", "-", "_"}:
            keep.append("_")

    name = "".join(keep).strip("_")
    return name or "subject"


def _get_local_ip() -> str:
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.settimeout(0.5)
        sock.connect(("8.8.8.8", 80))
        ip = sock.getsockname()[0]
        sock.close()
        return ip
    except Exception:
        return "127.0.0.1"


def _create_fresh_experiment_dir(
    experiment_root: str | Path,
    mode: str,
) -> Path:
    root = Path(experiment_root)
    root.mkdir(parents=True, exist_ok=True)

    if create_experiment_dir is not None:
        return create_experiment_dir(root_dir=root, task_name=mode)

    existing = sorted(
        p for p in root.iterdir() if p.is_dir() and p.name.startswith("exp_")
    )

    max_id = 0
    for item in existing:
        parts = item.name.split("_")
        if len(parts) >= 2:
            try:
                max_id = max(max_id, int(parts[1]))
            except ValueError:
                pass

    exp_id = max_id + 1
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    exp_dir = root / f"exp_{exp_id:03d}_{timestamp}_{mode}"
    exp_dir.mkdir(parents=True, exist_ok=False)
    return exp_dir


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default

    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


# ---------------------------------------------------------------------
# Study state
# ---------------------------------------------------------------------


@dataclass
class SubjectInfo:
    subject_no: int
    subject_id: str
    name: str
    age: str
    gender: str
    organization: str
    started_at: str
    completed_at: str = ""
    remark: str = ""


@dataclass
class WebStudyConfig:
    audio_dir: Path
    ref_dir: Path | None
    mode: str
    recursive: bool
    experiment_dir: Path
    ratings_csv: Path
    subjects_csv: Path
    summary_csv: Path
    global_csv: Path
    settings_json: Path
    shuffle_default: bool
    allow_refresh_retake_default: bool
    copyright_text: str


class WebStudyState:
    def __init__(self, config: WebStudyConfig, audio_files: list[Path]) -> None:
        self.config = config
        self.original_audio_files = audio_files
        self.lock = threading.Lock()

        self.subject_counter = 0
        self.subjects: dict[str, SubjectInfo] = {}
        self.sessions: dict[str, dict[str, Any]] = {}

        self.pbar = tqdm(
            total=0,
            desc=f"{config.mode.upper()} ratings",
            unit="rating",
        )

        self._ensure_files()
        self._ensure_settings()

    def _ensure_files(self) -> None:
        self.config.experiment_dir.mkdir(parents=True, exist_ok=True)

        if not self.config.ratings_csv.exists():
            with self.config.ratings_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                if self.config.mode == "mos":
                    writer.writerow(
                        [
                            "experiment_id",
                            "subject_no",
                            "subject_id",
                            "subject_name",
                            "age",
                            "gender",
                            "organization",
                            "file_index",
                            "file",
                            "score",
                            "timestamp",
                        ]
                    )
                else:
                    writer.writerow(
                        [
                            "experiment_id",
                            "subject_no",
                            "subject_id",
                            "subject_name",
                            "age",
                            "gender",
                            "organization",
                            "file_index",
                            "file",
                            "reference_file",
                            "score",
                            "timestamp",
                        ]
                    )

        if not self.config.subjects_csv.exists():
            with self.config.subjects_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "experiment_id",
                        "subject_no",
                        "subject_id",
                        "name",
                        "age",
                        "gender",
                        "organization",
                        "started_at",
                        "completed_at",
                        "remark",
                    ]
                )

        if not self.config.global_csv.exists():
            with self.config.global_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "experiment_id",
                        "mode",
                        "subject_no",
                        "subject_id",
                        "name",
                        "age",
                        "gender",
                        "organization",
                        "num_files",
                        "num_scores",
                        "overall_score",
                        "std_deviation",
                        "mean_minus_std",
                        "mean_plus_std",
                        "ci95_low",
                        "ci95_high",
                        "remark",
                        "completed_at",
                    ]
                )

    def _ensure_settings(self) -> None:
        if self.config.settings_json.exists():
            return

        settings = {
            "shuffle": self.config.shuffle_default,
            "allow_refresh_retake": self.config.allow_refresh_retake_default,
            "show_subject_file_navigation": True,
            "require_all_files_before_submit": True,
        }
        _write_json(self.config.settings_json, settings)

    def settings(self) -> dict[str, Any]:
        return _read_json(
            self.config.settings_json,
            {
                "shuffle": self.config.shuffle_default,
                "allow_refresh_retake": self.config.allow_refresh_retake_default,
                "show_subject_file_navigation": True,
                "require_all_files_before_submit": True,
            },
        )

    def update_settings(self, settings: dict[str, Any]) -> None:
        settings["require_all_files_before_submit"] = True
        _write_json(self.config.settings_json, settings)

    def create_subject(
        self,
        name: str,
        age: str,
        gender: str,
        organization: str,
    ) -> str:
        with self.lock:
            self.subject_counter += 1
            subject_no = self.subject_counter

            subject_id = f"subject_{subject_no:03d}_{_safe_name(name)}_{uuid.uuid4().hex[:8]}"
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            subject = SubjectInfo(
                subject_no=subject_no,
                subject_id=subject_id,
                name=name.strip(),
                age=age.strip(),
                gender=gender.strip(),
                organization=organization.strip(),
                started_at=now,
            )

            settings = self.settings()
            use_shuffle = bool(settings.get("shuffle", self.config.shuffle_default))

            order = list(range(len(self.original_audio_files)))
            if use_shuffle:
                random.shuffle(order)

            self.subjects[subject_id] = subject
            self.sessions[subject_id] = {
                "order": order,
                "current_pos": 0,
                "ratings": {},
                "started_at": now,
                "finished": False,
                "submitted": False,
            }

            with self.config.subjects_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.config.experiment_dir.name,
                        subject.subject_no,
                        subject.subject_id,
                        subject.name,
                        subject.age,
                        subject.gender,
                        subject.organization,
                        subject.started_at,
                        "",
                        "",
                    ]
                )

            return subject_id

    def get_subject(self, subject_id: str) -> SubjectInfo | None:
        return self.subjects.get(subject_id)

    def get_session(self, subject_id: str) -> dict[str, Any] | None:
        return self.sessions.get(subject_id)

    def current_audio(self, subject_id: str) -> tuple[int, Path] | None:
        session = self.get_session(subject_id)
        if session is None:
            return None

        order = session["order"]
        current_pos = int(session["current_pos"])

        if current_pos >= len(order):
            return None

        file_index = int(order[current_pos])
        return file_index, self.original_audio_files[file_index]

    def reference_for(self, audio_path: Path) -> Path | None:
        if self.config.ref_dir is None:
            return None

        candidate = self.config.ref_dir / audio_path.name
        if candidate.exists():
            return candidate

        if self.config.recursive:
            try:
                rel = audio_path.relative_to(self.config.audio_dir)
                candidate = self.config.ref_dir / rel
                if candidate.exists():
                    return candidate
            except Exception:
                pass

        return None

    def save_rating(self, subject_id: str, score: int) -> None:
        with self.lock:
            subject = self.get_subject(subject_id)
            session = self.get_session(subject_id)

            if subject is None or session is None:
                return

            if session.get("submitted"):
                return

            current = self.current_audio(subject_id)
            if current is None:
                session["finished"] = True
                return

            file_index, audio_path = current
            ref_path = self.reference_for(audio_path)

            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            score = int(score)

            session["ratings"][str(file_index)] = {
                "score": score,
                "timestamp": timestamp,
            }

            with self.config.ratings_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                base_row = [
                    self.config.experiment_dir.name,
                    subject.subject_no,
                    subject.subject_id,
                    subject.name,
                    subject.age,
                    subject.gender,
                    subject.organization,
                    file_index,
                    _safe_relpath(audio_path, self.config.audio_dir),
                ]

                if self.config.mode == "mos":
                    writer.writerow(base_row + [score, timestamp])
                else:
                    writer.writerow(
                        base_row
                        + [
                            _safe_relpath(ref_path, self.config.ref_dir),
                            score,
                            timestamp,
                        ]
                    )

            session["current_pos"] = int(session["current_pos"]) + 1
            self.pbar.total += 1
            self.pbar.update(1)

            if int(session["current_pos"]) >= len(session["order"]):
                session["finished"] = True

    def go_to(self, subject_id: str, pos: int) -> None:
        session = self.get_session(subject_id)
        if session is None:
            return

        if session.get("submitted"):
            return

        settings = self.settings()
        if not settings.get("show_subject_file_navigation", True):
            return

        pos = max(0, min(pos, len(session["order"]) - 1))
        session["current_pos"] = pos

    def subject_scores(self, subject_id: str) -> list[int]:
        session = self.get_session(subject_id)
        if session is None:
            return []

        ratings = session.get("ratings", {})
        return [int(item["score"]) for item in ratings.values()]

    def subject_completed_all_files(self, subject_id: str) -> bool:
        session = self.get_session(subject_id)
        if session is None:
            return False

        ratings = session.get("ratings", {})
        return len(ratings) >= len(self.original_audio_files)

    def first_unrated_position(self, subject_id: str) -> int:
        session = self.get_session(subject_id)
        if session is None:
            return 0

        ratings = session.get("ratings", {})
        order = session.get("order", [])

        for pos, file_index in enumerate(order):
            if str(file_index) not in ratings:
                return pos

        return len(order)

    def complete_subject(self, subject_id: str, remark: str) -> bool:
        with self.lock:
            subject = self.get_subject(subject_id)
            session = self.get_session(subject_id)

            if subject is None or session is None:
                return False

            if not self.subject_completed_all_files(subject_id):
                session["current_pos"] = self.first_unrated_position(subject_id)
                session["finished"] = False
                return False

            completed_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            subject.completed_at = completed_at
            subject.remark = remark.strip()
            session["submitted"] = True
            session["finished"] = True

            scores = self.subject_scores(subject_id)
            score_values = [float(x) for x in scores]

            mu = _mean(score_values)
            sd = _std(score_values)
            _, ci_low, ci_high = _ci95(score_values)

            with self.config.global_csv.open("a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        self.config.experiment_dir.name,
                        self.config.mode,
                        subject.subject_no,
                        subject.subject_id,
                        subject.name,
                        subject.age,
                        subject.gender,
                        subject.organization,
                        len(self.original_audio_files),
                        len(scores),
                        mu,
                        sd,
                        mu - sd if scores else float("nan"),
                        mu + sd if scores else float("nan"),
                        ci_low,
                        ci_high,
                        subject.remark,
                        completed_at,
                    ]
                )

            self.write_summary()
            return True

    def all_scores(self) -> list[int]:
        scores: list[int] = []

        for subject_id in self.subjects:
            scores.extend(self.subject_scores(subject_id))

        return scores

    def subject_file_rows(self, subject_id: str) -> list[dict[str, Any]]:
        subject = self.get_subject(subject_id)
        session = self.get_session(subject_id)

        if subject is None or session is None:
            return []

        rows: list[dict[str, Any]] = []
        ratings = session.get("ratings", {})

        for order_pos, file_index in enumerate(session["order"]):
            audio_path = self.original_audio_files[int(file_index)]
            item = ratings.get(str(file_index), {})
            score = item.get("score", "")
            timestamp = item.get("timestamp", "")

            rows.append(
                {
                    "subject_no": subject.subject_no,
                    "subject_id": subject.subject_id,
                    "subject_name": subject.name,
                    "order_pos": order_pos + 1,
                    "file_index": int(file_index),
                    "file": _safe_relpath(audio_path, self.config.audio_dir),
                    "score": score,
                    "timestamp": timestamp,
                }
            )

        return rows

    def subject_dashboard_data(self, subject_id: str) -> dict[str, Any]:
        subject = self.get_subject(subject_id)
        session = self.get_session(subject_id)

        if subject is None or session is None:
            return {}

        scores = [float(x) for x in self.subject_scores(subject_id)]
        mu = _mean(scores)
        sd = _std(scores)
        _, ci_low, ci_high = _ci95(scores)

        score_counts = {str(i): 0 for i in range(1, 6)}
        for score in scores:
            key = str(int(score))
            score_counts[key] = score_counts.get(key, 0) + 1

        return {
            "subject": subject,
            "num_files": len(self.original_audio_files),
            "num_scores": len(scores),
            "mean": mu,
            "std": sd,
            "mean_minus_std": mu - sd if scores else float("nan"),
            "mean_plus_std": mu + sd if scores else float("nan"),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "score_counts": score_counts,
            "file_rows": self.subject_file_rows(subject_id),
        }

    def write_subject_csv(self, subject_id: str) -> Path:
        subject = self.get_subject(subject_id)

        if subject is None:
            raise ValueError(f"Unknown subject: {subject_id}")

        safe_subject = f"subject_{subject.subject_no:03d}_{_safe_name(subject.name)}"
        output_path = self.config.experiment_dir / f"{safe_subject}_ratings.csv"

        rows = self.subject_file_rows(subject_id)

        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "subject_no",
                    "subject_id",
                    "subject_name",
                    "order_pos",
                    "file_index",
                    "file",
                    "score",
                    "timestamp",
                ],
            )
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

        return output_path

    def per_file_summary(self) -> list[dict[str, Any]]:
        bucket: dict[int, list[int]] = {}

        for session in self.sessions.values():
            for file_index, item in session.get("ratings", {}).items():
                idx = int(file_index)
                bucket.setdefault(idx, []).append(int(item["score"]))

        rows: list[dict[str, Any]] = []

        for idx, audio_path in enumerate(self.original_audio_files):
            values = bucket.get(idx, [])
            float_values = [float(v) for v in values]

            mu = _mean(float_values)
            sd = _std(float_values)
            _, ci_low, ci_high = _ci95(float_values)

            rows.append(
                {
                    "file_index": idx,
                    "file": _safe_relpath(audio_path, self.config.audio_dir),
                    "num_scores": len(values),
                    "mean": mu,
                    "std": sd,
                    "mean_minus_std": mu - sd if values else float("nan"),
                    "mean_plus_std": mu + sd if values else float("nan"),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                }
            )

        return rows

    def write_summary(self) -> None:
        all_scores = [float(x) for x in self.all_scores()]

        mu = _mean(all_scores)
        sd = _std(all_scores)
        _, ci_low, ci_high = _ci95(all_scores)

        with self.config.summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "experiment_id",
                    "mode",
                    "num_subjects",
                    "num_files",
                    "num_total_scores",
                    "overall_mean",
                    "std_deviation",
                    "mean_minus_std",
                    "mean_plus_std",
                    "ci95_low",
                    "ci95_high",
                ]
            )
            writer.writerow(
                [
                    self.config.experiment_dir.name,
                    self.config.mode,
                    len(self.subjects),
                    len(self.original_audio_files),
                    len(all_scores),
                    mu,
                    sd,
                    mu - sd if all_scores else float("nan"),
                    mu + sd if all_scores else float("nan"),
                    ci_low,
                    ci_high,
                ]
            )

        file_summary_csv = self.config.experiment_dir / "wav_file_score_summary.csv"

        with file_summary_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "file_index",
                    "file",
                    "num_scores",
                    "mean",
                    "std",
                    "mean_minus_std",
                    "mean_plus_std",
                    "ci95_low",
                    "ci95_high",
                ],
            )
            writer.writeheader()

            for row in self.per_file_summary():
                writer.writerow(row)

    def dashboard_data(self) -> dict[str, Any]:
        self.write_summary()

        all_scores = [float(x) for x in self.all_scores()]

        mu = _mean(all_scores)
        sd = _std(all_scores)
        _, ci_low, ci_high = _ci95(all_scores)

        gender_counts: dict[str, int] = {}
        org_counts: dict[str, int] = {}

        for subject in self.subjects.values():
            gender = subject.gender or "Unknown"
            org = subject.organization or "Unknown"

            gender_counts[gender] = gender_counts.get(gender, 0) + 1
            org_counts[org] = org_counts.get(org, 0) + 1

        score_counts = {str(i): 0 for i in range(1, 6)}

        for score in all_scores:
            key = str(int(score))
            score_counts[key] = score_counts.get(key, 0) + 1

        return {
            "experiment_id": self.config.experiment_dir.name,
            "mode": self.config.mode,
            "num_subjects": len(self.subjects),
            "num_files": len(self.original_audio_files),
            "num_scores": len(all_scores),
            "mean": mu,
            "std": sd,
            "mean_minus_std": mu - sd if all_scores else float("nan"),
            "mean_plus_std": mu + sd if all_scores else float("nan"),
            "ci95_low": ci_low,
            "ci95_high": ci_high,
            "gender_counts": gender_counts,
            "organization_counts": org_counts,
            "score_counts": score_counts,
            "file_summary": self.per_file_summary(),
            "ratings_csv": str(self.config.ratings_csv),
            "subjects_csv": str(self.config.subjects_csv),
            "summary_csv": str(self.config.summary_csv),
            "global_csv": str(self.config.global_csv),
            "settings": self.settings(),
        }


# ---------------------------------------------------------------------
# HTML helpers
# ---------------------------------------------------------------------


def _escape(text: Any) -> str:
    import html

    return html.escape(str(text))


def _base_css() -> str:
    return """
<style>
:root {
    --saffron: #ff9933;
    --saffron-dark: #e87500;
    --green: #138808;
    --blue: #2563eb;
    --cyan: #06b6d4;
    --red: #dc2626;
    --purple: #7c3aed;
    --slate: #0f172a;
    --soft: #f8fafc;
    --muted: #64748b;
}

* {
    box-sizing: border-box;
}

html {
    -webkit-text-size-adjust: 100%;
}

body {
    margin: 0;
    min-height: 100vh;
    font-family: "Inter", "Segoe UI", "DejaVu Sans", Arial, sans-serif;
    background:
        radial-gradient(circle at top left, rgba(255,153,51,0.45), transparent 28%),
        radial-gradient(circle at top right, rgba(19,136,8,0.35), transparent 28%),
        linear-gradient(135deg, #020617, #0f172a 48%, #111827);
    color: #f8fafc;
}

@keyframes floatIn {
    from {
        opacity: 0;
        transform: translateY(24px) scale(0.98);
    }
    to {
        opacity: 1;
        transform: translateY(0) scale(1);
    }
}

@keyframes glow {
    0%, 100% {
        box-shadow: 0 0 18px rgba(255,153,51,0.35);
    }
    50% {
        box-shadow: 0 0 38px rgba(255,153,51,0.72);
    }
}

.topbar {
    padding: 20px 28px;
    background: rgba(15, 23, 42, 0.92);
    display: flex;
    align-items: center;
    justify-content: space-between;
    border-bottom: 3px solid var(--saffron);
    gap: 16px;
    position: sticky;
    top: 0;
    z-index: 10;
    backdrop-filter: blur(12px);
}

.brand {
    font-size: clamp(22px, 4vw, 30px);
    font-weight: 950;
    letter-spacing: -0.04em;
    white-space: nowrap;
}

.brand span {
    color: var(--saffron);
}

.nav {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    justify-content: flex-end;
}

.nav a {
    text-decoration: none;
    color: white;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    padding: 10px 14px;
    border-radius: 999px;
    font-weight: 800;
    font-size: 14px;
}

.nav a:hover {
    background: var(--saffron);
    color: #111827;
}

.container {
    width: min(1280px, 100%);
    margin: 0 auto;
    padding: clamp(14px, 3vw, 36px);
}

.card {
    background: rgba(248, 250, 252, 0.98);
    color: #0f172a;
    border-radius: clamp(18px, 3vw, 28px);
    padding: clamp(18px, 3vw, 34px);
    animation: floatIn 0.55s ease both;
    box-shadow: 0 28px 80px rgba(0,0,0,0.38);
    border: 3px solid rgba(255,153,51,0.55);
    overflow-x: auto;
}

.card h1 {
    margin-top: 0;
    font-size: clamp(26px, 5vw, 36px);
    letter-spacing: -0.04em;
    line-height: 1.1;
}

.card h2 {
    color: #075985;
    font-size: clamp(20px, 4vw, 28px);
}

.card h3 {
    font-size: clamp(18px, 3.5vw, 22px);
}

.input-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
}

label {
    display: block;
    font-weight: 900;
    color: #334155;
    margin-bottom: 8px;
}

input, select, textarea {
    width: 100%;
    border: 2px solid #cbd5e1;
    border-radius: 16px;
    padding: 14px 16px;
    font-size: 16px;
    outline: none;
    background: white;
}

textarea {
    min-height: 130px;
}

input:focus, select:focus, textarea:focus {
    border-color: var(--saffron);
    box-shadow: 0 0 0 4px rgba(255,153,51,0.18);
}

.btn {
    border: 0;
    border-radius: 18px;
    padding: 15px 22px;
    font-weight: 950;
    font-size: 16px;
    cursor: pointer;
    text-decoration: none;
    display: inline-block;
    color: white;
    background: linear-gradient(135deg, var(--saffron), #ef4444);
    box-shadow: 0 16px 36px rgba(255,153,51,0.28);
    text-align: center;
}

.btn:hover {
    filter: brightness(1.06);
    transform: translateY(-2px);
}

.btn-blue {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
}

.btn-green {
    background: linear-gradient(135deg, #138808, #22c55e);
}

.btn-red {
    background: linear-gradient(135deg, #dc2626, #991b1b);
}

.btn-purple {
    background: linear-gradient(135deg, #7c3aed, #ec4899);
}

.progress-track {
    width: 100%;
    height: 24px;
    border-radius: 999px;
    overflow: hidden;
    background: #e2e8f0;
    margin: 14px 0 20px;
}

.progress-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--saffron), #ffffff, var(--green));
    transition: width 0.25s ease;
}

.audio-grid {
    display: grid;
    grid-template-columns: repeat(2, minmax(0, 1fr));
    gap: 18px;
    margin: 24px 0;
}

.audio-card {
    color: white;
    border-radius: 24px;
    padding: 22px;
    animation: glow 2.8s ease-in-out infinite;
    min-width: 0;
}

.audio-card.generated {
    background: linear-gradient(135deg, #138808, #22c55e);
}

.audio-card.reference {
    background: linear-gradient(135deg, #2563eb, #0ea5e9);
}

.audio-card.missing {
    background: linear-gradient(135deg, #f97316, #dc2626);
}

.audio-card h3 {
    margin-top: 0;
}

.audio-card p {
    word-break: break-word;
    font-size: 14px;
}

.reference-only-wrap {
    margin: 24px 0;
}

.reference-only-wrap:empty {
    display: none;
}

audio {
    width: 100%;
}

.score-buttons {
    display: grid;
    grid-template-columns: repeat(5, minmax(58px, 1fr));
    gap: clamp(8px, 2vw, 16px);
    margin: 24px auto;
    max-width: 620px;
}

.score-buttons form {
    margin: 0;
}

.score-btn {
    width: 100%;
    aspect-ratio: 1 / 0.88;
    min-height: 64px;
    border: 0;
    border-radius: 22px;
    color: white;
    font-size: clamp(22px, 5vw, 32px);
    font-weight: 950;
    cursor: pointer;
    transition: transform 0.15s ease;
}

.score-btn:hover {
    transform: translateY(-5px) scale(1.05);
}

.grid-4 {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 16px;
}

.stat {
    color: white;
    border-radius: 22px;
    padding: 20px;
}

.stat:nth-child(1) {
    background: linear-gradient(135deg, #ff9933, #ef4444);
}

.stat:nth-child(2) {
    background: linear-gradient(135deg, #2563eb, #06b6d4);
}

.stat:nth-child(3) {
    background: linear-gradient(135deg, #138808, #22c55e);
}

.stat:nth-child(4) {
    background: linear-gradient(135deg, #7c3aed, #ec4899);
}

.stat-label {
    font-size: 13px;
    opacity: 0.88;
    font-weight: 800;
}

.stat-value {
    font-size: clamp(20px, 4vw, 28px);
    font-weight: 950;
    margin-top: 6px;
    word-break: break-word;
}

.table-wrap {
    overflow-x: auto;
    width: 100%;
    border-radius: 16px;
}

table {
    width: 100%;
    border-collapse: collapse;
    background: white;
    border-radius: 16px;
    overflow: hidden;
    min-width: 760px;
}

th {
    background: #0f172a;
    color: white;
    text-align: left;
    padding: 12px;
}

td {
    border-bottom: 1px solid #e2e8f0;
    padding: 11px 12px;
    color: #334155;
}

.file-nav {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    margin: 18px 0;
}

.file-nav a {
    text-decoration: none;
    padding: 9px 12px;
    border-radius: 999px;
    color: #0f172a;
    background: #e2e8f0;
    font-weight: 900;
}

.file-nav a.done {
    background: #bbf7d0;
}

.file-nav a.active {
    background: var(--saffron);
}

.footer {
    margin-top: 24px;
    color: #cbd5e1;
    text-align: center;
    font-size: 14px;
}

.notice {
    padding: 16px;
    border-radius: 18px;
    background: #fff7ed;
    border: 2px solid var(--saffron);
    color: #7c2d12;
    font-weight: 800;
}

.error-notice {
    padding: 16px;
    border-radius: 18px;
    background: #fef2f2;
    border: 2px solid #ef4444;
    color: #991b1b;
    font-weight: 900;
}

.plot-box {
    background: white;
    border-radius: 22px;
    padding: clamp(12px, 2vw, 20px);
    margin: 18px 0;
    overflow-x: auto;
}

canvas {
    max-width: 100%;
}

.mobile-stack {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    align-items: center;
}

@media (max-width: 900px) {
    .input-grid, .audio-grid, .grid-4 {
        grid-template-columns: 1fr;
    }

    .topbar {
        flex-direction: column;
        align-items: flex-start;
    }

    .nav {
        justify-content: flex-start;
    }

    .score-buttons {
        grid-template-columns: repeat(5, 1fr);
    }
}

@media (max-width: 560px) {
    .container {
        padding: 12px;
    }

    .topbar {
        padding: 16px;
    }

    .nav a {
        font-size: 13px;
        padding: 8px 10px;
    }

    .card {
        padding: 16px;
        border-radius: 18px;
    }

    .score-buttons {
        gap: 7px;
    }

    .score-btn {
        border-radius: 14px;
        min-height: 54px;
    }

    .audio-card {
        padding: 16px;
        border-radius: 18px;
    }

    .btn {
        width: 100%;
        margin-top: 8px;
    }

    .mobile-stack {
        flex-direction: column;
        align-items: stretch;
    }
}
</style>
"""


def _layout(
    title: str,
    body: str,
    admin: bool = False,
    copyright_text: str = "",
) -> str:
    admin_links = ""

    if admin:
        admin_links = """
        <a href="/admin/dashboard">Dashboard</a>
        <a href="/admin/settings">Settings</a>
        <a href="/admin/export/ratings">Export Ratings</a>
        <a href="/admin/export/global">Export Global</a>
        <a href="/admin/logout">Logout</a>
        """

    return f"""
<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8">
    <title>{_escape(title)}</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    {_base_css()}
</head>
<body>
    <div class="topbar">
        <div class="brand">🎧 Speech<span>EvalKit</span></div>
        <div class="nav">
            <a href="/">Subject Entry</a>
            <a href="/admin/login">Admin</a>
            {admin_links}
        </div>
    </div>

    <div class="container">
        {body}
        <div class="footer">{_escape(copyright_text)}</div>
    </div>
</body>
</html>
"""


# ---------------------------------------------------------------------
# Subject pages
# ---------------------------------------------------------------------


def _subject_form(state: WebStudyState) -> str:
    body = """
    <div class="card">
        <h1>Subject Information</h1>
        <p class="notice">
            Please enter your details before starting the listening evaluation.
            You must rate every WAV file before final submission.
        </p>

        <form method="post" action="/start">
            <div class="input-grid">
                <div>
                    <label>Name</label>
                    <input name="name" required placeholder="Enter subject name">
                </div>
                <div>
                    <label>Age</label>
                    <input name="age" required type="number" min="1" max="120" placeholder="Enter age">
                </div>
                <div>
                    <label>Gender</label>
                    <select name="gender" required>
                        <option value="">Select</option>
                        <option>Male</option>
                        <option>Female</option>
                        <option>Other</option>
                        <option>Prefer not to say</option>
                    </select>
                </div>
                <div>
                    <label>Organization</label>
                    <input name="organization" required placeholder="University / Company / Lab">
                </div>
            </div>

            <div style="margin-top:28px;">
                <button class="btn" type="submit">Start Listening Test</button>
            </div>
        </form>

        <p class="notice" style="margin-top:24px;">
            WAV order randomization is controlled by the administrator only.
        </p>
    </div>
    """

    return _layout(
        title="SpeechEvalKit Subject Entry",
        body=body,
        copyright_text=state.config.copyright_text,
    )


def _waveform_script(gen_url: str) -> str:
    return f"""
<div class="plot-box">
    <p class="notice">
        Click anywhere on the waveform to play from that timestamp. Click again at another point to jump there.
        The blue segment shows played audio; light grey shows remaining audio.
    </p>

    <canvas
        id="waveformCanvas"
        width="1600"
        height="300"
        style="width:100%; background:#f8fafc; border-radius:18px; border:2px solid #cbd5e1; cursor:pointer;">
    </canvas>

    <div style="display:flex; justify-content:space-between; color:#334155; font-weight:900; margin-top:8px;">
        <span id="currentTimeLabel">0.000 s</span>
        <span id="durationLabel">0.000 s</span>
    </div>
</div>

<script>
const canvas = document.getElementById("waveformCanvas");
const ctx = canvas.getContext("2d");
const audio = document.getElementById("generatedAudioPlayer");

let waveformData = null;
let audioDuration = 0;
let animationId = null;
let cachedWidth = 0;
let cachedSamples = null;

// Minimum target: 1000 visual steps per 5 seconds = 200 steps per second.
// Browser drawing uses requestAnimationFrame for smooth playback movement.
const STEPS_PER_SECOND = 200;

function resizeCanvasForDPR() {{
    const rect = canvas.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;

    const displayWidth = Math.max(600, Math.floor(rect.width));
    const displayHeight = 300;

    canvas.width = Math.floor(displayWidth * dpr);
    canvas.height = Math.floor(displayHeight * dpr);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}

function formatTime(seconds) {{
    if (!isFinite(seconds)) return "0.000 s";
    return seconds.toFixed(3) + " s";
}}

function downsampleWaveform(data, width) {{
    const samples = new Array(width);

    for (let x = 0; x < width; x++) {{
        const start = Math.floor((x / width) * data.length);
        const end = Math.max(start + 1, Math.floor(((x + 1) / width) * data.length));

        let min = 1.0;
        let max = -1.0;
        let rms = 0.0;
        let count = 0;

        for (let i = start; i < end; i++) {{
            const v = data[i] || 0;
            if (v < min) min = v;
            if (v > max) max = v;
            rms += v * v;
            count += 1;
        }}

        rms = Math.sqrt(rms / Math.max(1, count));

        samples[x] = {{
            min: min,
            max: max,
            rms: rms
        }};
    }}

    return samples;
}}

function drawWaveform(progressRatio = 0) {{
    if (!waveformData) return;

    const rect = canvas.getBoundingClientRect();
    const width = Math.max(600, Math.floor(rect.width));
    const height = 300;
    const mid = height / 2;
    const playedX = Math.max(0, Math.min(width, Math.floor(width * progressRatio)));

    if (width !== cachedWidth || cachedSamples === null) {{
        cachedWidth = width;
        cachedSamples = downsampleWaveform(waveformData, width);
    }}

    ctx.clearRect(0, 0, width, height);

    const bg = ctx.createLinearGradient(0, 0, width, height);
    bg.addColorStop(0, "#ffffff");
    bg.addColorStop(1, "#f1f5f9");
    ctx.fillStyle = bg;
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = "#e2e8f0";
    ctx.lineWidth = 1;

    for (let i = 0; i <= 10; i++) {{
        const x = (i / 10) * width;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, height);
        ctx.stroke();
    }}

    for (let i = 0; i <= 4; i++) {{
        const y = (i / 4) * height;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(width, y);
        ctx.stroke();
    }}

    for (let x = 0; x < width; x++) {{
        const s = cachedSamples[x];
        if (!s) continue;

        const ampBoost = 0.92;
        const yMin = mid + s.min * mid * ampBoost;
        const yMax = mid + s.max * mid * ampBoost;

        ctx.strokeStyle = x <= playedX ? "#2563eb" : "#cbd5e1";
        ctx.lineWidth = x <= playedX ? 1.8 : 1.0;

        ctx.beginPath();
        ctx.moveTo(x, yMin);
        ctx.lineTo(x, yMax);
        ctx.stroke();
    }}

    ctx.strokeStyle = "#0f172a";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(0, mid);
    ctx.lineTo(width, mid);
    ctx.stroke();

    ctx.fillStyle = "#1d4ed8";
    ctx.fillRect(Math.max(0, playedX - 2), 0, 4, height);

    ctx.beginPath();
    ctx.arc(playedX, 18, 8, 0, Math.PI * 2);
    ctx.fillStyle = "#1d4ed8";
    ctx.fill();

    document.getElementById("currentTimeLabel").innerText = formatTime(audio.currentTime || 0);
    document.getElementById("durationLabel").innerText = formatTime(audioDuration || audio.duration || 0);
}}

function smoothAnimationLoop() {{
    if (!audio || !waveformData) return;

    const duration = audio.duration || audioDuration || 0;
    const progress = duration > 0 ? audio.currentTime / duration : 0;

    drawWaveform(progress);

    if (!audio.paused && !audio.ended) {{
        animationId = requestAnimationFrame(smoothAnimationLoop);
    }}
}}

function startSmoothAnimation() {{
    if (animationId !== null) {{
        cancelAnimationFrame(animationId);
    }}
    animationId = requestAnimationFrame(smoothAnimationLoop);
}}

async function loadWaveform() {{
    try {{
        resizeCanvasForDPR();

        const response = await fetch("{gen_url}");
        const arrayBuffer = await response.arrayBuffer();
        const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        const audioBuffer = await audioCtx.decodeAudioData(arrayBuffer);

        waveformData = audioBuffer.getChannelData(0);
        audioDuration = audioBuffer.duration;

        document.getElementById("durationLabel").innerText = formatTime(audioDuration);
        drawWaveform(0);
    }} catch (err) {{
        console.error("Waveform loading failed:", err);

        const rect = canvas.getBoundingClientRect();
        ctx.fillStyle = "#fee2e2";
        ctx.fillRect(0, 0, rect.width, 300);
        ctx.fillStyle = "#991b1b";
        ctx.font = "18px sans-serif";
        ctx.fillText("Waveform loading failed. Please check audio format.", 24, 50);
    }}
}}

function seekAndPlayFromClick(event) {{
    if (!audio || !waveformData) return;

    const rect = canvas.getBoundingClientRect();
    const x = event.clientX - rect.left;
    const ratio = Math.max(0, Math.min(1, x / rect.width));
    const targetTime = ratio * (audio.duration || audioDuration || 0);

    audio.currentTime = targetTime;
    drawWaveform(ratio);

    const playPromise = audio.play();
    if (playPromise !== undefined) {{
        playPromise
            .then(() => {{
                startSmoothAnimation();
            }})
            .catch((err) => {{
                console.error("Audio play failed:", err);
            }});
    }} else {{
        startSmoothAnimation();
    }}
}}

window.addEventListener("load", () => {{
    loadWaveform();

    if (audio) {{
        audio.addEventListener("loadedmetadata", () => {{
            audioDuration = audio.duration || audioDuration;
            document.getElementById("durationLabel").innerText = formatTime(audioDuration);
        }});

        audio.addEventListener("play", () => {{
            startSmoothAnimation();
        }});

        audio.addEventListener("pause", () => {{
            if (animationId !== null) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            const duration = audio.duration || audioDuration || 0;
            drawWaveform(duration > 0 ? audio.currentTime / duration : 0);
        }});

        audio.addEventListener("ended", () => {{
            if (animationId !== null) {{
                cancelAnimationFrame(animationId);
                animationId = null;
            }}
            drawWaveform(1.0);
        }});

        audio.addEventListener("seeked", () => {{
            const duration = audio.duration || audioDuration || 0;
            drawWaveform(duration > 0 ? audio.currentTime / duration : 0);
            if (!audio.paused) {{
                startSmoothAnimation();
            }}
        }});
    }}

    canvas.addEventListener("click", seekAndPlayFromClick);

    window.addEventListener("resize", () => {{
        cachedWidth = 0;
        cachedSamples = null;
        resizeCanvasForDPR();

        const duration = audio.duration || audioDuration || 0;
        drawWaveform(duration > 0 ? audio.currentTime / duration : 0);
    }});
}});
</script>
"""


def _subject_rating_page(
    state: WebStudyState,
    subject_id: str,
    error_message: str = "",
) -> str:
    subject = state.get_subject(subject_id)
    session = state.get_session(subject_id)

    if subject is None or session is None:
        return _layout(
            "Invalid Session",
            '<div class="card"><h1>Invalid session</h1><a class="btn" href="/">Start again</a></div>',
            copyright_text=state.config.copyright_text,
        )

    if not state.subject_completed_all_files(subject_id):
        session["current_pos"] = state.first_unrated_position(subject_id)
        session["finished"] = False

    current = state.current_audio(subject_id)
    settings = state.settings()

    if current is None and state.subject_completed_all_files(subject_id):
        return _remark_page(state, subject_id)

    if current is None:
        session["current_pos"] = state.first_unrated_position(subject_id)
        current = state.current_audio(subject_id)

    if current is None:
        return _layout(
            "No Audio",
            '<div class="card"><h1>No audio available</h1></div>',
            copyright_text=state.config.copyright_text,
        )

    file_index, audio_path = current
    ref_path = state.reference_for(audio_path)

    current_pos = int(session["current_pos"])
    total = len(session["order"])
    ratings = session.get("ratings", {})
    completed_count = len(ratings)
    percent = (completed_count / total) * 100 if total else 0

    rel_audio = _safe_relpath(audio_path, state.config.audio_dir)
    gen_url = f"/audio/generated/{quote(rel_audio)}"

    ref_card = ""

    if state.config.mode == "smos":
        if ref_path is not None:
            rel_ref = _safe_relpath(ref_path, state.config.ref_dir)
            ref_url = f"/audio/reference/{quote(rel_ref)}"
            ref_card = f"""
            <div class="audio-card reference">
                <h3>Reference Audio</h3>
                <audio controls preload="metadata" src="{ref_url}"></audio>
                <p>{_escape(rel_ref)}</p>
            </div>
            """
        else:
            ref_card = """
            <div class="audio-card missing">
                <h3>Reference Audio</h3>
                <p>⚠ No matching reference file found.</p>
            </div>
            """

    scale = (
        "1 = Bad · 2 = Poor · 3 = Fair · 4 = Good · 5 = Excellent"
        if state.config.mode == "mos"
        else "1 = Completely Different · 2 = Different · 3 = Somewhat Similar · 4 = Similar · 5 = Same Speaker"
    )

    score_buttons = ""

    for score, color in [
        (1, "#ef4444"),
        (2, "#f97316"),
        (3, "#eab308"),
        (4, "#22c55e"),
        (5, "#0ea5e9"),
    ]:
        score_buttons += f"""
        <form method="post" action="/score/{_escape(subject_id)}">
            <input type="hidden" name="score" value="{score}">
            <button class="score-btn" style="background:{color};" type="submit">{score}</button>
        </form>
        """

    file_nav = ""

    if settings.get("show_subject_file_navigation", True):
        links = []

        for pos, idx in enumerate(session["order"]):
            cls = []

            if str(idx) in ratings:
                cls.append("done")

            if pos == current_pos:
                cls.append("active")

            cls_text = " ".join(cls)

            links.append(
                f'<a class="{cls_text}" href="/goto/{_escape(subject_id)}/{pos}">{pos + 1}</a>'
            )

        file_nav = f"""
        <h3>WAV File Navigation</h3>
        <p class="notice">
            Green pages are already rated. You must rate all WAV files before final submission.
        </p>
        <div class="file-nav">{''.join(links)}</div>
        """

    error_html = ""
    if error_message:
        error_html = f'<p class="error-notice">{_escape(error_message)}</p>'

    waveform_html = _waveform_script(gen_url)

    body = f"""
    <div class="card">
        <h1>{state.config.mode.upper()} Listening Test</h1>

        {error_html}

        <p class="notice">
            Subject #{subject.subject_no}: {_escape(subject.name)}
            · Age: {_escape(subject.age)}
            · Gender: {_escape(subject.gender)}
            · Organization: {_escape(subject.organization)}
        </p>

        <h2>Current File: {current_pos + 1} of {total}</h2>
        <p style="font-weight:900; color:#334155;">
            Rated files: {completed_count}/{total}. Final submission is locked until every WAV file is rated.
        </p>

        <div class="progress-track">
            <div class="progress-fill" style="width:{percent:.2f}%;"></div>
        </div>

        <div style="display:none;">
            <audio id="generatedAudioPlayer" preload="metadata" src="{gen_url}"></audio>
        </div>

        <div class="reference-only-wrap">
            {ref_card}
        </div>

        {waveform_html}

        <h2 style="text-align:center;">Give Your Rating</h2>
        <div class="score-buttons">{score_buttons}</div>
        <p style="text-align:center; font-weight:900; color:#334155;">{_escape(scale)}</p>

        {file_nav}

        <p class="notice">
            Subjects only provide ratings. Mean, standard deviation, confidence interval,
            and global statistics are visible only in the admin dashboard.
        </p>
    </div>
    """

    return _layout(
        title="SpeechEvalKit Rating",
        body=body,
        copyright_text=state.config.copyright_text,
    )


def _remark_page(
    state: WebStudyState,
    subject_id: str,
    error_message: str = "",
) -> str:
    subject = state.get_subject(subject_id)

    if subject is None:
        return _layout(
            "Invalid Session",
            '<div class="card"><h1>Invalid session</h1><a class="btn" href="/">Start again</a></div>',
            copyright_text=state.config.copyright_text,
        )

    if not state.subject_completed_all_files(subject_id):
        return _subject_rating_page(
            state,
            subject_id,
            error_message="You must rate every WAV file before final submission.",
        )

    scores = state.subject_scores(subject_id)
    total = len(state.original_audio_files)

    error_html = ""
    if error_message:
        error_html = f'<p class="error-notice">{_escape(error_message)}</p>'

    body = f"""
    <div class="card">
        <h1>Final Remark</h1>

        {error_html}

        <p class="notice">
            Thank you, {_escape(subject.name)}. You rated all {len(scores)}/{total} WAV files.
            Please add any optional remark before final submission.
        </p>

        <form method="post" action="/submit/{_escape(subject_id)}">
            <label>Remark / Comment</label>
            <textarea name="remark" placeholder="Write your remark here..."></textarea>

            <div style="margin-top:24px;" class="mobile-stack">
                <button class="btn btn-green" type="submit">Submit Examination</button>
                <a class="btn btn-blue" href="/test/{_escape(subject_id)}">Review Rated Files</a>
            </div>
        </form>
    </div>
    """

    return _layout(
        title="SpeechEvalKit Final Remark",
        body=body,
        copyright_text=state.config.copyright_text,
    )


def _submit_done_page(state: WebStudyState, subject_id: str) -> str:
    subject = state.get_subject(subject_id)
    name = subject.name if subject else "Subject"

    body = f"""
    <div class="card">
        <h1>Submission Complete</h1>
        <p class="notice">
            Thank you, {_escape(name)}. Your responses have been saved successfully.
        </p>

        <p>
            The experiment administrator can view the overall result, mean ± standard deviation,
            confidence interval, demographic summary, file-wise summary, subject-wise summary,
            and export data from the admin dashboard.
        </p>

        <a class="btn" href="/">Start Another Subject Session</a>
    </div>
    """

    return _layout(
        title="Submission Complete",
        body=body,
        copyright_text=state.config.copyright_text,
    )


# ---------------------------------------------------------------------
# Admin pages
# ---------------------------------------------------------------------


def _admin_login_page(state: WebStudyState, error: str = "") -> str:
    err = f'<p class="error-notice">{_escape(error)}</p>' if error else ""

    body = f"""
    <div class="card">
        <h1>Admin Login</h1>
        {err}
        <form method="post" action="/admin/login">
            <div class="input-grid">
                <div>
                    <label>Username</label>
                    <input name="username" required placeholder="Username">
                </div>
                <div>
                    <label>Password</label>
                    <input name="password" type="password" required placeholder="Password">
                </div>
            </div>
            <div style="margin-top:24px;">
                <button class="btn btn-blue" type="submit">Login</button>
            </div>
        </form>

        <p class="notice" style="margin-top:20px;">
            Default admin username: ravindra<br>
            Default admin password: speechevalkit
        </p>
    </div>
    """

    return _layout(
        title="Admin Login",
        body=body,
        copyright_text=state.config.copyright_text,
    )


def _admin_dashboard_page(
    state: WebStudyState,
    selected_subject_id: str | None = None,
) -> str:
    data = state.dashboard_data()

    subject_options = """
    <option value="">Overall Experiment Dashboard</option>
    """

    for subject in sorted(state.subjects.values(), key=lambda s: s.subject_no):
        selected = "selected" if selected_subject_id == subject.subject_id else ""
        subject_options += f"""
        <option value="{_escape(subject.subject_id)}" {selected}>
            {subject.subject_no}. {_escape(subject.name)} | Age: {_escape(subject.age)} | {_escape(subject.gender)}
        </option>
        """

    subject_panel = ""

    if selected_subject_id:
        subject_data = state.subject_dashboard_data(selected_subject_id)

        if subject_data:
            subject = subject_data["subject"]
            subject_score_counts_json = json.dumps(subject_data["score_counts"])
            subject_file_rows_json = json.dumps(subject_data["file_rows"])

            subject_rows_html = ""
            for row in subject_data["file_rows"]:
                subject_rows_html += f"""
                <tr>
                    <td>{row["order_pos"]}</td>
                    <td>{row["file_index"]}</td>
                    <td>{_escape(row["file"])}</td>
                    <td>{_escape(row["score"])}</td>
                    <td>{_escape(row["timestamp"])}</td>
                </tr>
                """

            subject_panel = f"""
            <h2>Subject-Specific Dashboard</h2>

            <p class="notice">
                Subject #{subject.subject_no}: {_escape(subject.name)}
                · Age: {_escape(subject.age)}
                · Gender: {_escape(subject.gender)}
                · Organization: {_escape(subject.organization)}
            </p>

            <div class="grid-4">
                <div class="stat">
                    <div class="stat-label">Subject Mean</div>
                    <div class="stat-value">{_fmt(subject_data["mean"])}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Subject Std</div>
                    <div class="stat-value">{_fmt(subject_data["std"])}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Mean ± Std</div>
                    <div class="stat-value">{_fmt(subject_data["mean_minus_std"])} / {_fmt(subject_data["mean_plus_std"])}</div>
                </div>
                <div class="stat">
                    <div class="stat-label">Scores</div>
                    <div class="stat-value">{subject_data["num_scores"]}/{subject_data["num_files"]}</div>
                </div>
            </div>

            <p>
                <a class="btn btn-green" href="/admin/export/subject/{_escape(subject.subject_id)}">
                    Download This Subject CSV
                </a>
            </p>

            <div class="plot-box">
                <div id="subjectScorePlot"></div>
            </div>

            <div class="plot-box">
                <div id="subjectFilePlot"></div>
            </div>

            <h2>Subject File-wise Ratings</h2>
            <div class="table-wrap">
            <table>
                <thead>
                    <tr>
                        <th>Order</th>
                        <th>File Index</th>
                        <th>File</th>
                        <th>Score</th>
                        <th>Timestamp</th>
                    </tr>
                </thead>
                <tbody>
                    {subject_rows_html}
                </tbody>
            </table>
            </div>

            <script>
                const subjectScoreCounts = {subject_score_counts_json};
                const subjectFileRows = {subject_file_rows_json};

                Plotly.newPlot("subjectScorePlot", [{{
                    x: Object.keys(subjectScoreCounts),
                    y: Object.values(subjectScoreCounts),
                    type: "bar",
                    marker: {{
                        color: ["#ef4444", "#f97316", "#eab308", "#22c55e", "#0ea5e9"]
                    }}
                }}], {{
                    title: "Selected Subject Score Distribution",
                    xaxis: {{title: "Score"}},
                    yaxis: {{title: "Count"}}
                }}, {{responsive: true}});

                Plotly.newPlot("subjectFilePlot", [{{
                    x: subjectFileRows.map(r => r.order_pos),
                    y: subjectFileRows.map(r => r.score === "" ? null : Number(r.score)),
                    type: "scatter",
                    mode: "lines+markers",
                    line: {{color: "#2563eb"}},
                    marker: {{size: 9}}
                }}], {{
                    title: "Selected Subject File-wise Scores",
                    xaxis: {{title: "Listening Order"}},
                    yaxis: {{title: "Score", range: [0, 5]}}
                }}, {{responsive: true}});
            </script>
            """

    file_rows_html = ""
    for row in data["file_summary"]:
        file_rows_html += f"""
        <tr>
            <td>{row["file_index"]}</td>
            <td>{_escape(row["file"])}</td>
            <td>{row["num_scores"]}</td>
            <td>{_fmt(row["mean"])}</td>
            <td>{_fmt(row["std"])}</td>
            <td>{_fmt(row["mean_minus_std"])} to {_fmt(row["mean_plus_std"])}</td>
            <td>{_fmt(row["ci95_low"])} to {_fmt(row["ci95_high"])}</td>
        </tr>
        """

    score_counts_json = json.dumps(data["score_counts"])
    gender_counts_json = json.dumps(data["gender_counts"])
    org_counts_json = json.dumps(data["organization_counts"])
    file_summary_json = json.dumps(data["file_summary"])

    body = f"""
    <div class="card">
        <h1>Admin Dashboard</h1>

        <form method="get" action="/admin/dashboard" style="margin-bottom:24px;">
            <label>Select Subject</label>
            <select name="subject_id" onchange="this.form.submit()">
                {subject_options}
            </select>
        </form>

        {subject_panel}

        <h2>Overall Experiment Dashboard</h2>

        <div class="grid-4">
            <div class="stat">
                <div class="stat-label">Overall Score</div>
                <div class="stat-value">{_fmt(data["mean"])}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Std Deviation</div>
                <div class="stat-value">{_fmt(data["std"])}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Mean ± Std</div>
                <div class="stat-value">{_fmt(data["mean_minus_std"])} / {_fmt(data["mean_plus_std"])}</div>
            </div>
            <div class="stat">
                <div class="stat-label">95% CI</div>
                <div class="stat-value">{_fmt(data["ci95_low"])} / {_fmt(data["ci95_high"])}</div>
            </div>
        </div>

        <p class="notice">
            Experiment: {_escape(data["experiment_id"])} · Mode: {_escape(data["mode"].upper())}
            · Subjects: {data["num_subjects"]} · Files: {data["num_files"]}
            · Total Scores: {data["num_scores"]}
        </p>

        <div class="plot-box">
            <div id="scorePlot"></div>
        </div>

        <div class="plot-box">
            <div id="genderPlot"></div>
        </div>

        <div class="plot-box">
            <div id="orgPlot"></div>
        </div>

        <div class="plot-box">
            <div id="filePlot"></div>
        </div>

        <h2>WAV File Summary</h2>
        <div class="table-wrap">
        <table>
            <thead>
                <tr>
                    <th>Index</th>
                    <th>File</th>
                    <th>N</th>
                    <th>Mean</th>
                    <th>Std</th>
                    <th>Mean ± Std</th>
                    <th>95% CI</th>
                </tr>
            </thead>
            <tbody>
                {file_rows_html}
            </tbody>
        </table>
        </div>

        <h2>Data Export</h2>
        <p class="mobile-stack">
            <a class="btn btn-green" href="/admin/export/ratings">Download Ratings CSV</a>
            <a class="btn btn-purple" href="/admin/export/subjects">Download Subjects CSV</a>
            <a class="btn btn-blue" href="/admin/export/global">Download Global CSV</a>
            <a class="btn" href="/admin/export/file-summary">Download File Summary CSV</a>
        </p>

        <p class="notice">
            Files are saved in: {_escape(state.config.experiment_dir)}
        </p>
    </div>

    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <script>
        const scoreCounts = {score_counts_json};
        const genderCounts = {gender_counts_json};
        const orgCounts = {org_counts_json};
        const fileSummary = {file_summary_json};

        Plotly.newPlot("scorePlot", [{{
            x: Object.keys(scoreCounts),
            y: Object.values(scoreCounts),
            type: "bar",
            marker: {{color: ["#ef4444", "#f97316", "#eab308", "#22c55e", "#0ea5e9"]}}
        }}], {{
            title: "Overall Score Distribution",
            xaxis: {{title: "Score"}},
            yaxis: {{title: "Count"}}
        }}, {{responsive: true}});

        Plotly.newPlot("genderPlot", [{{
            labels: Object.keys(genderCounts),
            values: Object.values(genderCounts),
            type: "pie"
        }}], {{
            title: "Gender Distribution"
        }}, {{responsive: true}});

        Plotly.newPlot("orgPlot", [{{
            x: Object.keys(orgCounts),
            y: Object.values(orgCounts),
            type: "bar",
            marker: {{color: "#ff9933"}}
        }}], {{
            title: "Organization Distribution",
            xaxis: {{title: "Organization"}},
            yaxis: {{title: "Subjects"}}
        }}, {{responsive: true}});

        Plotly.newPlot("filePlot", [{{
            x: fileSummary.map(r => r.file_index),
            y: fileSummary.map(r => r.mean),
            type: "scatter",
            mode: "lines+markers",
            line: {{color: "#138808"}},
            marker: {{size: 8}}
        }}], {{
            title: "Overall File-wise Mean Score",
            xaxis: {{title: "WAV File Index"}},
            yaxis: {{title: "Mean Score", range: [0, 5]}}
        }}, {{responsive: true}});

        setTimeout(() => {{
            window.location.reload();
        }}, 15000);
    </script>
    """

    return _layout(
        title="Admin Dashboard",
        body=body,
        admin=True,
        copyright_text=state.config.copyright_text,
    )


def _admin_settings_page(state: WebStudyState) -> str:
    settings = state.settings()

    shuffle_checked = "checked" if settings.get("shuffle", True) else ""
    retake_checked = "checked" if settings.get("allow_refresh_retake", True) else ""
    nav_checked = "checked" if settings.get("show_subject_file_navigation", True) else ""

    body = f"""
    <div class="card">
        <h1>Admin Settings</h1>

        <p class="notice">
            Shuffle/randomization is controlled only here. Subjects cannot enable or disable shuffle.
            Submission always requires all WAV files to be rated.
        </p>

        <form method="post" action="/admin/settings">
            <p>
                <label>
                    <input type="checkbox" name="shuffle" value="1" {shuffle_checked} style="width:auto;">
                    Shuffle WAV order session-wise to reduce listening bias
                </label>
            </p>

            <p>
                <label>
                    <input type="checkbox" name="allow_refresh_retake" value="1" {retake_checked} style="width:auto;">
                    Allow page refresh and examination continuation
                </label>
            </p>

            <p>
                <label>
                    <input type="checkbox" name="show_subject_file_navigation" value="1" {nav_checked} style="width:auto;">
                    Allow subjects to travel between WAV pages
                </label>
            </p>

            <p>
                <label>
                    <input type="checkbox" name="require_all_files_before_submit" value="1" style="width:auto;" disabled checked>
                    Require all WAV files before final submit. This is always enforced.
                </label>
            </p>

            <button class="btn btn-green" type="submit">Save Settings</button>
        </form>
    </div>
    """

    return _layout(
        title="Admin Settings",
        body=body,
        admin=True,
        copyright_text=state.config.copyright_text,
    )


# ---------------------------------------------------------------------
# Django launcher
# ---------------------------------------------------------------------


def launch_gui(
    audio_dir: str | Path,
    output_csv: str | Path | None = "auto",
    mode: str = "mos",
    ref_dir: str | Path | None = None,
    recursive: bool = False,
    fullscreen: bool = False,
    host: str = "0.0.0.0",
    port: int = 8765,
    open_browser: bool = True,
    experiment_root: str | Path = "results/experiments",
    shuffle: bool = True,
    allow_refresh_retake: bool = True,
    copyright_text: str = (
        "© 2026 SpeechEvalKit. All rights reserved. "
        "For research and evaluation use only."
    ),
) -> None:
    """
    Launch Django-based MOS/SMOS web evaluation website.

    Important behavior
    ------------------
    - Subjects must rate every WAV file.
    - Subjects cannot submit until all WAV files are rated.
    - Shuffle is controlled only by the admin panel.
    - Website is responsive for mobile and desktop.
    """
    _ = fullscreen
    _ = output_csv

    mode = mode.lower().strip()
    if mode not in {"mos", "smos"}:
        raise ValueError("mode must be either 'mos' or 'smos'")

    audio_dir = Path(audio_dir).expanduser().resolve()
    ref_path = Path(ref_dir).expanduser().resolve() if ref_dir is not None else None

    audio_files = _collect_audio(audio_dir, recursive=recursive)

    if not audio_files:
        raise RuntimeError(f"No audio files found in {audio_dir}")

    experiment_dir = _create_fresh_experiment_dir(
        experiment_root=experiment_root,
        mode=mode,
    )

    config = WebStudyConfig(
        audio_dir=audio_dir,
        ref_dir=ref_path,
        mode=mode,
        recursive=recursive,
        experiment_dir=experiment_dir,
        ratings_csv=experiment_dir / f"{mode}_ratings.csv",
        subjects_csv=experiment_dir / "subjects.csv",
        summary_csv=experiment_dir / f"{mode}_summary.csv",
        global_csv=experiment_dir / "global_subject_results.csv",
        settings_json=experiment_dir / "admin_settings.json",
        shuffle_default=shuffle,
        allow_refresh_retake_default=allow_refresh_retake,
        copyright_text=copyright_text,
    )

    state = WebStudyState(config=config, audio_files=audio_files)

    try:
        import django
        from django.conf import settings
        from django.core.management import call_command
        from django.http import FileResponse, Http404, HttpRequest, HttpResponse
        from django.shortcuts import redirect
        from django.urls import path
    except ImportError as exc:
        raise ImportError(
            "Django is required for SpeechEvalKit web GUI.\n"
            "Install with:\n\n"
            "    pip install django\n"
        ) from exc

    if not settings.configured:
        settings.configure(
            DEBUG=True,
            SECRET_KEY="speechevalkit-web-secret-key",
            ROOT_URLCONF=__name__,
            ALLOWED_HOSTS=["*"],
            INSTALLED_APPS=[
                "django.contrib.sessions",
            ],
            MIDDLEWARE=[
                "django.contrib.sessions.middleware.SessionMiddleware",
            ],
            DATABASES={
                "default": {
                    "ENGINE": "django.db.backends.sqlite3",
                    "NAME": str(experiment_dir / "django_session.sqlite3"),
                }
            },
            DEFAULT_AUTO_FIELD="django.db.models.AutoField",
            USE_TZ=True,
        )

        django.setup()
        call_command("migrate", "sessions", verbosity=0, interactive=False)

    def is_admin(request: HttpRequest) -> bool:
        return bool(request.session.get("admin_logged_in", False))

    def require_admin(request: HttpRequest) -> HttpResponse | None:
        if not is_admin(request):
            return redirect("/admin/login")
        return None

    def index_view(request: HttpRequest) -> HttpResponse:
        return HttpResponse(_subject_form(state))

    def start_view(request: HttpRequest) -> HttpResponse:
        if request.method != "POST":
            return redirect("/")

        name = request.POST.get("name", "").strip()
        age = request.POST.get("age", "").strip()
        gender = request.POST.get("gender", "").strip()
        organization = request.POST.get("organization", "").strip()

        if not name or not age or not gender or not organization:
            return redirect("/")

        subject_id = state.create_subject(
            name=name,
            age=age,
            gender=gender,
            organization=organization,
        )

        request.session["subject_id"] = subject_id
        return redirect(f"/test/{subject_id}")

    def test_view(request: HttpRequest, subject_id: str) -> HttpResponse:
        return HttpResponse(_subject_rating_page(state=state, subject_id=subject_id))

    def score_view(request: HttpRequest, subject_id: str) -> HttpResponse:
        if request.method != "POST":
            return redirect(f"/test/{subject_id}")

        raw = request.POST.get("score", "0")

        try:
            score = int(raw)
        except ValueError:
            score = 0

        if 1 <= score <= 5:
            state.save_rating(subject_id, score)

        return redirect(f"/test/{subject_id}")

    def goto_view(request: HttpRequest, subject_id: str, pos: int) -> HttpResponse:
        state.go_to(subject_id, int(pos))
        return redirect(f"/test/{subject_id}")

    def submit_view(request: HttpRequest, subject_id: str) -> HttpResponse:
        if request.method != "POST":
            return redirect(f"/test/{subject_id}")

        if not state.subject_completed_all_files(subject_id):
            return HttpResponse(
                _subject_rating_page(
                    state,
                    subject_id,
                    error_message="Submission blocked. You must rate every WAV file before submitting.",
                )
            )

        remark = request.POST.get("remark", "")
        ok = state.complete_subject(subject_id, remark)

        if not ok:
            return HttpResponse(
                _subject_rating_page(
                    state,
                    subject_id,
                    error_message="Submission blocked. Some WAV files are still unrated.",
                )
            )

        return redirect(f"/done/{subject_id}")

    def done_view(request: HttpRequest, subject_id: str) -> HttpResponse:
        return HttpResponse(_submit_done_page(state, subject_id))

    def admin_login_view(request: HttpRequest) -> HttpResponse:
        if request.method == "GET":
            return HttpResponse(_admin_login_page(state))

        username = request.POST.get("username", "")
        password = request.POST.get("password", "")

        if username == ADMIN_USERNAME and password == ADMIN_PASSWORD:
            request.session["admin_logged_in"] = True
            return redirect("/admin/dashboard")

        return HttpResponse(_admin_login_page(state, "Invalid admin credentials."))

    def admin_logout_view(request: HttpRequest) -> HttpResponse:
        request.session["admin_logged_in"] = False
        return redirect("/admin/login")

    def admin_dashboard_view(request: HttpRequest) -> HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        selected_subject_id = request.GET.get("subject_id") or None

        return HttpResponse(
            _admin_dashboard_page(
                state,
                selected_subject_id=selected_subject_id,
            )
        )

    def admin_settings_view(request: HttpRequest) -> HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        if request.method == "POST":
            new_settings = {
                "shuffle": bool(request.POST.get("shuffle")),
                "allow_refresh_retake": bool(request.POST.get("allow_refresh_retake")),
                "show_subject_file_navigation": bool(
                    request.POST.get("show_subject_file_navigation")
                ),
                "require_all_files_before_submit": True,
            }
            state.update_settings(new_settings)
            return redirect("/admin/settings")

        return HttpResponse(_admin_settings_page(state))

    def export_file(path: Path) -> FileResponse:
        if not path.exists():
            raise Http404("Export file does not exist.")

        return FileResponse(
            path.open("rb"),
            content_type="text/csv",
            as_attachment=True,
            filename=path.name,
        )

    def export_ratings_view(request: HttpRequest) -> FileResponse | HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        state.write_summary()
        return export_file(state.config.ratings_csv)

    def export_subjects_view(request: HttpRequest) -> FileResponse | HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        return export_file(state.config.subjects_csv)

    def export_global_view(request: HttpRequest) -> FileResponse | HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        state.write_summary()
        return export_file(state.config.global_csv)

    def export_file_summary_view(request: HttpRequest) -> FileResponse | HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        state.write_summary()
        return export_file(state.config.experiment_dir / "wav_file_score_summary.csv")

    def export_subject_view(
        request: HttpRequest,
        subject_id: str,
    ) -> FileResponse | HttpResponse:
        denied = require_admin(request)
        if denied:
            return denied

        try:
            subject_csv = state.write_subject_csv(subject_id)
        except Exception as exc:
            raise Http404(str(exc)) from exc

        return export_file(subject_csv)

    def audio_view(request: HttpRequest, kind: str, rel_path: str) -> FileResponse:
        rel_path = rel_path.replace("\\", "/")

        if kind == "generated":
            root = state.config.audio_dir
        elif kind == "reference":
            if state.config.ref_dir is None:
                raise Http404("Reference directory is not configured.")
            root = state.config.ref_dir
        else:
            raise Http404("Unknown audio kind.")

        candidate = (root / rel_path).resolve()

        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise Http404("Invalid audio path.") from exc

        if not candidate.exists() or not candidate.is_file():
            raise Http404("Audio file not found.")

        mime_type, _ = mimetypes.guess_type(str(candidate))
        mime_type = mime_type or "application/octet-stream"

        return FileResponse(
            candidate.open("rb"),
            content_type=mime_type,
            as_attachment=False,
            filename=candidate.name,
        )

    global urlpatterns
    urlpatterns = [
        path("", index_view),
        path("start", start_view),
        path("test/<str:subject_id>", test_view),
        path("score/<str:subject_id>", score_view),
        path("goto/<str:subject_id>/<int:pos>", goto_view),
        path("submit/<str:subject_id>", submit_view),
        path("done/<str:subject_id>", done_view),
        path("audio/<str:kind>/<path:rel_path>", audio_view),
        path("admin/login", admin_login_view),
        path("admin/logout", admin_logout_view),
        path("admin/dashboard", admin_dashboard_view),
        path("admin/settings", admin_settings_view),
        path("admin/export/ratings", export_ratings_view),
        path("admin/export/subjects", export_subjects_view),
        path("admin/export/global", export_global_view),
        path("admin/export/file-summary", export_file_summary_view),
        path("admin/export/subject/<str:subject_id>", export_subject_view),
    ]

    local_ip = _get_local_ip()
    local_url = f"http://127.0.0.1:{port}"
    network_url = f"http://{local_ip}:{port}"

    print("\n" + "=" * 78)
    print("🎧 SpeechEvalKit MOS/SMOS Django Website")
    print("=" * 78)
    print(f"Mode              : {mode.upper()}")
    print(f"Generated audio   : {audio_dir}")
    print(f"Reference audio   : {ref_path if ref_path else 'N/A'}")
    print(f"Experiment dir    : {experiment_dir}")
    print(f"Ratings CSV       : {config.ratings_csv}")
    print(f"Subjects CSV      : {config.subjects_csv}")
    print(f"Global CSV        : {config.global_csv}")
    print(f"Summary CSV       : {config.summary_csv}")
    print("-" * 78)
    print(f"Subject URL local : {local_url}")
    print(f"Subject URL LAN   : {network_url}")
    print(f"Admin URL local   : {local_url}/admin/login")
    print(f"Admin URL LAN     : {network_url}/admin/login")
    print("-" * 78)
    print(f"Admin username    : {ADMIN_USERNAME}")
    print(f"Admin password    : {ADMIN_PASSWORD}")
    print("-" * 78)
    print("Subjects must rate every WAV file before final submission.")
    print("Shuffle option is controlled only from Admin Settings.")
    print("Generated audio uses waveform-only player.")
    print("Click waveform to play or jump to a timestamp.")
    print("Website is responsive for mobile and desktop.")
    print("Open LAN URL from another phone/computer on the same Wi-Fi/network.")
    print("Press CTRL+C to stop the website.")
    print("=" * 78 + "\n")

    if open_browser:
        try:
            webbrowser.open(local_url)
        except Exception:
            pass

    from django.core.management import call_command

    call_command(
        "runserver",
        f"{host}:{port}",
        use_reloader=False,
        verbosity=1,
    )


def launch_mos_website(
    audio_dir: str | Path,
    recursive: bool = False,
    host: str = "0.0.0.0",
    port: int = 8765,
    open_browser: bool = True,
    experiment_root: str | Path = "results/experiments",
) -> None:
    launch_gui(
        audio_dir=audio_dir,
        mode="mos",
        ref_dir=None,
        recursive=recursive,
        host=host,
        port=port,
        open_browser=open_browser,
        experiment_root=experiment_root,
    )


def launch_smos_website(
    audio_dir: str | Path,
    ref_dir: str | Path,
    recursive: bool = False,
    host: str = "0.0.0.0",
    port: int = 8765,
    open_browser: bool = True,
    experiment_root: str | Path = "results/experiments",
) -> None:
    launch_gui(
        audio_dir=audio_dir,
        mode="smos",
        ref_dir=ref_dir,
        recursive=recursive,
        host=host,
        port=port,
        open_browser=open_browser,
        experiment_root=experiment_root,
    )