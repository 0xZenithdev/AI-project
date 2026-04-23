"""
Show progress for the latest or a selected training run.

Examples:
python -m src.check_training_progress
python -m src.check_training_progress --run-dir output/training_runs/20260420_182618_line_model
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Show progress for the latest training run.")
    parser.add_argument(
        "--runs-root",
        type=str,
        default="output/training_runs",
        help="Directory that contains timestamped training run folders",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default="",
        help="Optional specific run directory to inspect instead of the latest one",
    )
    return parser.parse_args()


def find_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_dir():
            raise SystemExit(f"Run directory was not found: {run_dir}")
        return run_dir

    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise SystemExit(f"Training runs directory was not found: {runs_root}")

    run_dirs = sorted((path for path in runs_root.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime, reverse=True)
    if not run_dirs:
        raise SystemExit(f"No training runs were found under: {runs_root}")
    return run_dirs[0]


def load_summary(run_dir: Path) -> tuple[dict, Path]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Summary file was not found: {summary_path}")
    return json.loads(summary_path.read_text(encoding="utf-8")), summary_path


def compute_progress(summary: dict) -> tuple[int, int, int, float]:
    requested_epochs = int(summary.get("requested_epochs") or 0)
    completed_epochs = int(summary.get("completed_epochs") or 0)
    remaining_epochs = int(summary.get("remaining_epochs") or max(0, requested_epochs - completed_epochs))
    progress_percent = float(summary.get("progress_percent") or (0.0 if requested_epochs <= 0 else round((completed_epochs / requested_epochs) * 100.0, 2)))
    return requested_epochs, completed_epochs, remaining_epochs, progress_percent


def parse_run_started_at(run_dir: Path) -> datetime | None:
    parts = run_dir.name.split("_", 2)
    if len(parts) < 2:
        return None
    try:
        return datetime.strptime(f"{parts[0]}_{parts[1]}", "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def format_duration(duration: timedelta) -> str:
    total_seconds = max(0, int(duration.total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes}m {seconds}s"
    if minutes:
        return f"{minutes}m {seconds}s"
    return f"{seconds}s"


def maybe_format_eta(run_dir: Path, completed_epochs: int, remaining_epochs: int, status: str) -> str:
    if status != "running" or completed_epochs <= 0 or remaining_epochs <= 0:
        return ""

    started_at = parse_run_started_at(run_dir)
    if started_at is None:
        return ""

    elapsed = datetime.now() - started_at
    if elapsed.total_seconds() <= 0:
        return ""

    average_epoch_time = elapsed / completed_epochs
    remaining_time = average_epoch_time * remaining_epochs
    return format_duration(remaining_time)


def main() -> None:
    args = parse_args()
    run_dir = find_run_dir(args)
    summary, summary_path = load_summary(run_dir)
    requested_epochs, completed_epochs, remaining_epochs, progress_percent = compute_progress(summary)
    last_update = datetime.fromtimestamp(summary_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")
    history_path = run_dir / "history.csv"
    eta_text = maybe_format_eta(
        run_dir=run_dir,
        completed_epochs=completed_epochs,
        remaining_epochs=remaining_epochs,
        status=str(summary.get("status", "unknown")),
    )

    print("Training progress")
    print(f"Run dir: {run_dir}")
    print(f"Status: {summary.get('status', 'unknown')}")
    print(f"Checkpoint: {summary.get('output_checkpoint', 'n/a')}")
    print(f"Epochs: {completed_epochs} / {requested_epochs}")
    print(f"Remaining epochs: {remaining_epochs}")
    print(f"Progress: {progress_percent:.2f}%")
    if eta_text:
        print(f"Rough remaining time if still active: {eta_text}")
    print(f"Latest train loss: {summary.get('latest_train_loss', 'n/a')}")
    print(f"Latest val loss: {summary.get('latest_val_loss', 'n/a')}")
    print(f"Best epoch: {summary.get('best_epoch', 'n/a')}")
    print(f"Best val loss: {summary.get('best_val_loss', 'n/a')}")
    print(f"Last artifact update: {last_update}")
    print(f"Summary file: {summary_path}")
    print(f"History file: {history_path}")


if __name__ == "__main__":
    main()
