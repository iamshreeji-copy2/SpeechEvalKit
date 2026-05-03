from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from speechevalkit import __version__
from speechevalkit.evaluate import SUPPORTED_METRICS, evaluate

console = Console()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="speechevalkit",
        description="Evaluate generated speech against reference speech.",
    )
    parser.add_argument(
        "--ref",
        required=True,
        type=str,
        help="Directory containing reference audio files.",
    )
    parser.add_argument(
        "--pred",
        required=True,
        type=str,
        help="Directory containing predicted/generated audio files.",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["pesq", "stoi", "mcd", "si_sdr", "cosine"],
        choices=sorted(SUPPORTED_METRICS.keys()),
        help="Metrics to compute.",
    )
    parser.add_argument(
        "--sr",
        "--sample-rate",
        dest="sample_rate",
        type=int,
        default=16000,
        help="Target sample rate. Default: 16000.",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Search directories recursively.",
    )
    parser.add_argument(
        "--save-json",
        type=str,
        default=None,
        help="Optional output JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"SpeechEvalKit {__version__}",
    )
    return parser


def _render_summary_table(results: dict[str, Any]) -> None:
    table = Table(
        title="SpeechEvalKit Results",
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Metric", style="bold")
    table.add_column("Average Score", justify="right", style="green")

    for metric, value in results["summary"].items():
        if value != value:
            rendered = "[yellow]NaN[/yellow]"
        else:
            rendered = f"{value:.6f}"
        table.add_row(metric, rendered)

    console.print(table)


def _render_metadata(results: dict[str, Any]) -> None:
    meta = results["metadata"]

    info = (
        f"[bold]Reference:[/bold] {meta['ref_dir']}\n"
        f"[bold]Prediction:[/bold] {meta['pred_dir']}\n"
        f"[bold]Sample rate:[/bold] {meta['sample_rate']} Hz\n"
        f"[bold]Matched files:[/bold] {meta['num_matched']}\n"
        f"[bold]Evaluated files:[/bold] {meta['num_evaluated']}\n"
        f"[bold]Skipped files:[/bold] {meta['num_skipped']}"
    )

    console.print(
        Panel(
            info,
            title="[bold green]Evaluation Complete[/bold green]",
            border_style="green",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    console.print(
        Panel.fit(
            "[bold cyan]SpeechEvalKit[/bold cyan]\n"
            "[white]Unified speech evaluation toolkit[/white]",
            border_style="cyan",
        )
    )

    try:
        results = evaluate(
            ref_dir=Path(args.ref),
            pred_dir=Path(args.pred),
            metrics=args.metrics,
            sample_rate=args.sample_rate,
            recursive=args.recursive,
            show_progress=True,
            save_json=args.save_json,
            verbose=not args.quiet,
        )
    except Exception as exc:
        console.print(f"[bold red]Error:[/bold red] {exc}")
        return 1

    _render_metadata(results)
    _render_summary_table(results)

    if results["missing_predictions"]:
        console.print(
            f"[yellow]Warning:[/yellow] "
            f"{len(results['missing_predictions'])} reference files had no prediction."
        )

    if results["skipped"]:
        console.print(
            f"[yellow]Warning:[/yellow] "
            f"{len(results['skipped'])} matched files were skipped."
        )

    if args.save_json:
        console.print(f"[green]Saved JSON results to:[/green] {args.save_json}")

    console.print("[bold green]Done.[/bold green]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
