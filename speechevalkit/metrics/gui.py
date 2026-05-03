from __future__ import annotations

import csv
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

from speechevalkit.utils.io import AUDIO_EXTENSIONS


def _play_audio(path: Path) -> None:
    if sys.platform.startswith("linux"):
        subprocess.Popen(["xdg-open", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    elif sys.platform.startswith("win"):
        subprocess.Popen(["cmd", "/c", "start", "", str(path)], shell=True)
    else:
        raise RuntimeError(f"Unsupported platform: {sys.platform}")


def _collect_audio(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(
        p for p in directory.glob("*") if p.suffix.lower() in AUDIO_EXTENSIONS
    )


class MOSGui:
    """
    Simple MOS/SMOS subjective evaluation GUI.
    """

    def __init__(
        self,
        audio_dir: str | Path,
        output_csv: str | Path = "mos_scores.csv",
        mode: str = "mos",
        ref_dir: str | Path | None = None,
    ) -> None:
        self.audio_files = _collect_audio(audio_dir)
        self.ref_dir = Path(ref_dir) if ref_dir is not None else None
        self.output_csv = Path(output_csv)
        self.mode = mode.lower()
        self.index = 0

        if self.mode not in {"mos", "smos"}:
            raise ValueError("mode must be 'mos' or 'smos'")

        if not self.audio_files:
            raise RuntimeError(f"No audio files found in {audio_dir}")

        self.root = tk.Tk()
        self.root.title("SpeechEvalKit Subjective Evaluation")
        self.root.geometry("620x360")

        self.title = tk.Label(
            self.root,
            text="SpeechEvalKit MOS/SMOS GUI",
            font=("Arial", 18, "bold"),
            fg="#0066cc",
        )
        self.title.pack(pady=12)

        self.file_label = tk.Label(
            self.root,
            text="",
            font=("Arial", 12),
            wraplength=560,
        )
        self.file_label.pack(pady=10)

        self.instruction_label = tk.Label(
            self.root,
            text=self._instruction_text(),
            font=("Arial", 11),
            fg="#444444",
            wraplength=560,
        )
        self.instruction_label.pack(pady=8)

        btn_frame = tk.Frame(self.root)
        btn_frame.pack(pady=10)

        self.play_btn = tk.Button(
            btn_frame,
            text="▶ Play Generated",
            command=self.play_current,
            bg="#2e7d32",
            fg="white",
            width=18,
        )
        self.play_btn.grid(row=0, column=0, padx=6)

        if self.mode == "smos":
            self.play_ref_btn = tk.Button(
                btn_frame,
                text="▶ Play Reference",
                command=self.play_reference,
                bg="#1565c0",
                fg="white",
                width=18,
            )
            self.play_ref_btn.grid(row=0, column=1, padx=6)

        score_frame = tk.Frame(self.root)
        score_frame.pack(pady=16)

        for score in range(1, 6):
            btn = tk.Button(
                score_frame,
                text=str(score),
                command=lambda s=score: self.save_score(s),
                width=7,
                height=2,
                bg="#f5f5f5",
            )
            btn.grid(row=0, column=score - 1, padx=5)

        self.status = tk.Label(
            self.root,
            text="",
            font=("Arial", 10),
            fg="#666666",
        )
        self.status.pack(pady=8)

        self._ensure_csv()
        self._refresh()

    def _instruction_text(self) -> str:
        if self.mode == "mos":
            return "Rate naturalness/quality from 1 (bad) to 5 (excellent)."
        return "Rate speaker similarity from 1 (different) to 5 (same speaker)."

    def _ensure_csv(self) -> None:
        if self.output_csv.exists():
            return

        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.mode == "mos":
                writer.writerow(["file", "mos"])
            else:
                writer.writerow(["file", "reference_file", "smos"])

    def _current_file(self) -> Path:
        return self.audio_files[self.index]

    def _reference_file(self) -> Path | None:
        if self.ref_dir is None:
            return None

        current = self._current_file()
        candidate = self.ref_dir / current.name
        return candidate if candidate.exists() else None

    def _refresh(self) -> None:
        current = self._current_file()
        self.file_label.config(
            text=f"File {self.index + 1}/{len(self.audio_files)}: {current.name}"
        )
        self.status.config(text=f"Saving to: {self.output_csv}")

    def play_current(self) -> None:
        _play_audio(self._current_file())

    def play_reference(self) -> None:
        ref = self._reference_file()
        if ref is None:
            messagebox.showwarning(
                "Missing reference",
                "No matching reference file found.",
            )
            return
        _play_audio(ref)

    def save_score(self, score: int) -> None:
        current = self._current_file()
        ref = self._reference_file()

        with self.output_csv.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if self.mode == "mos":
                writer.writerow([current.name, score])
            else:
                writer.writerow([current.name, ref.name if ref else "", score])

        self.index += 1

        if self.index >= len(self.audio_files):
            messagebox.showinfo(
                "Complete",
                f"Evaluation complete. Scores saved to {self.output_csv}",
            )
            self.root.destroy()
            return

        self._refresh()

    def run(self) -> None:
        self.root.mainloop()


def launch_gui(
    audio_dir: str | Path,
    output_csv: str | Path = "mos_scores.csv",
    mode: str = "mos",
    ref_dir: str | Path | None = None,
) -> None:
    gui = MOSGui(
        audio_dir=audio_dir,
        output_csv=output_csv,
        mode=mode,
        ref_dir=ref_dir,
    )
    gui.run()
