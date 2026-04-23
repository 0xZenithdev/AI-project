"""
Training entry-point for line segmentation model.

Expected dataset layout:
- dataset/images/*.jpg|png|...
- dataset/masks/*.jpg|png|...

Each image must have a corresponding mask with the same filename stem.
Example:
- dataset/images/cat_01.jpg
- dataset/masks/cat_01.png
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.ml_line_model import LinePairDataset, ModelConfig, TinyUNet, combined_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a line-segmentation model for robot drawing.")

    parser.add_argument("--images-dir", type=str, default="dataset/images", help="Input images directory")
    parser.add_argument("--masks-dir", type=str, default="dataset/masks", help="Target masks directory")
    parser.add_argument(
        "--train-manifest",
        type=str,
        default="",
        help="Optional JSON manifest for the training split",
    )
    parser.add_argument(
        "--val-manifest",
        type=str,
        default="",
        help="Optional JSON manifest for the validation split",
    )

    parser.add_argument("--width", type=int, default=210, help="Training width")
    parser.add_argument("--height", type=int, default=297, help="Training height")

    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")

    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=str, default="models/line_model.pt", help="Checkpoint output path")
    parser.add_argument(
        "--resume-from",
        type=str,
        default="",
        help="Optional checkpoint used to initialize model weights before training",
    )

    return parser.parse_args()


def evaluate(model: TinyUNet, loader: DataLoader, device: torch.device) -> float:
    """Compute average validation loss."""
    model.eval()
    total_loss = 0.0
    n = 0

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            logits = model(images)
            loss = combined_loss(logits, masks)

            total_loss += float(loss.item())
            n += 1

    return total_loss / max(1, n)


def make_run_dir(output_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("output") / "training_runs" / f"{timestamp}_{output_path.stem}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_history_csv(path: Path, history: list[dict]) -> None:
    fieldnames = ["epoch", "train_loss", "val_loss", "is_best_checkpoint"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row["epoch"],
                    "train_loss": f"{row['train_loss']:.6f}",
                    "val_loss": f"{row['val_loss']:.6f}",
                    "is_best_checkpoint": str(bool(row["is_best_checkpoint"])).lower(),
                }
            )


def compute_progress(requested_epochs: int, completed_epochs: int) -> tuple[int, float]:
    remaining_epochs = max(0, requested_epochs - completed_epochs)
    progress_percent = 0.0 if requested_epochs <= 0 else round((completed_epochs / requested_epochs) * 100.0, 2)
    return remaining_epochs, progress_percent


def write_summary_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Training Run Summary",
        "",
        "## Run",
        f"- Status: {summary['status']}",
        f"- Output checkpoint: `{summary['output_checkpoint']}`",
        f"- Run directory: `{summary['run_dir']}`",
        f"- Device: `{summary['device']}`",
        f"- Resume from: `{summary['resume_from'] or 'none'}`",
        "",
        "## Dataset",
        f"- Train samples: {summary['train_samples']}",
        f"- Validation samples: {summary['val_samples']}",
        f"- Train manifest: `{summary['train_manifest'] or 'none'}`",
        f"- Validation manifest: `{summary['val_manifest'] or 'none'}`",
        "",
        "## Model",
        f"- Working size: {summary['width']} x {summary['height']}",
        f"- Resize mode: `{summary['resize_mode']}`",
        "",
        "## Training",
        f"- Requested epochs: {summary['requested_epochs']}",
        f"- Completed epochs: {summary['completed_epochs']}",
        f"- Remaining epochs: {summary['remaining_epochs']}",
        f"- Progress: {summary['completed_epochs']} / {summary['requested_epochs']} ({summary['progress_percent']}%)",
        f"- Batch size: {summary['batch_size']}",
        f"- Learning rate: {summary['learning_rate']}",
        "",
        "## Best Checkpoint",
        f"- Best epoch: {summary['best_epoch'] if summary['best_epoch'] is not None else 'none'}",
        f"- Best validation loss: {summary['best_val_loss'] if summary['best_val_loss'] is not None else 'n/a'}",
        "",
        "## Latest Epoch",
        f"- Train loss: {summary['latest_train_loss'] if summary['latest_train_loss'] is not None else 'n/a'}",
        f"- Validation loss: {summary['latest_val_loss'] if summary['latest_val_loss'] is not None else 'n/a'}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_run_artifacts(
    run_dir: Path,
    args: argparse.Namespace,
    history: list[dict],
    train_samples: int,
    val_samples: int,
    best_epoch: int | None,
    best_val: float | None,
    status: str,
) -> None:
    latest = history[-1] if history else {}
    completed_epochs = len(history)
    remaining_epochs, progress_percent = compute_progress(args.epochs, completed_epochs)
    summary = {
        "status": status,
        "run_dir": str(run_dir),
        "output_checkpoint": str(Path(args.output)),
        "device": args.device,
        "resume_from": args.resume_from,
        "train_manifest": args.train_manifest,
        "val_manifest": args.val_manifest,
        "train_samples": train_samples,
        "val_samples": val_samples,
        "width": args.width,
        "height": args.height,
        "resize_mode": "aspect_pad",
        "requested_epochs": args.epochs,
        "completed_epochs": completed_epochs,
        "remaining_epochs": remaining_epochs,
        "progress_percent": progress_percent,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "best_epoch": best_epoch,
        "best_val_loss": None if best_val is None else round(best_val, 6),
        "latest_train_loss": None if not latest else round(float(latest["train_loss"]), 6),
        "latest_val_loss": None if not latest else round(float(latest["val_loss"]), 6),
    }
    write_json(
        run_dir / "run_config.json",
        {
            "images_dir": args.images_dir,
            "masks_dir": args.masks_dir,
            "train_manifest": args.train_manifest,
            "val_manifest": args.val_manifest,
            "width": args.width,
            "height": args.height,
            "resize_mode": "aspect_pad",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "val_split": args.val_split,
            "device": args.device,
            "output": args.output,
            "resume_from": args.resume_from,
        },
    )
    write_json(run_dir / "history.json", {"epochs": history})
    write_history_csv(run_dir / "history.csv", history)
    write_json(run_dir / "summary.json", summary)
    write_summary_markdown(run_dir / "summary.md", summary)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output)
    run_dir = make_run_dir(output_path)
    print(f"Training artifacts will be written to: {run_dir}")

    device = torch.device(args.device)
    if args.train_manifest:
        train_set = LinePairDataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            size=(args.width, args.height),
            manifest_path=args.train_manifest,
        )
        if args.val_manifest:
            val_set = LinePairDataset(
                images_dir=args.images_dir,
                masks_dir=args.masks_dir,
                size=(args.width, args.height),
                manifest_path=args.val_manifest,
            )
            val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)
        else:
            val_set = None
            val_loader = None
    else:
        dataset = LinePairDataset(
            images_dir=args.images_dir,
            masks_dir=args.masks_dir,
            size=(args.width, args.height),
        )

        # Split into train and validation subsets.
        val_len = int(len(dataset) * args.val_split)
        train_len = len(dataset) - val_len
        train_set, val_set = random_split(dataset, [train_len, val_len])
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if val_len > 0 else None

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)

    model = TinyUNet().to(device)
    if args.resume_from:
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        print(f"Loaded initial weights from: {args.resume_from}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")
    best_epoch: int | None = None
    history: list[dict] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        steps = 0

        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = combined_loss(logits, masks)
            loss.backward()
            optimizer.step()

            running += float(loss.item())
            steps += 1

        train_loss = running / max(1, steps)
        remaining_epochs, _ = compute_progress(args.epochs, epoch)
        epoch_prefix = f"Epoch {epoch:03d}/{args.epochs:03d} | remaining={remaining_epochs:03d}"

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"{epoch_prefix} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        else:
            val_loss = train_loss
            print(f"{epoch_prefix} | train_loss={train_loss:.4f}")

        improved = val_loss < best_val
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "is_best_checkpoint": improved,
            }
        )

        # Save best checkpoint based on validation loss.
        if improved:
            best_val = val_loss
            best_epoch = epoch
            output_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state": model.state_dict(),
                "width": args.width,
                "height": args.height,
                "best_val_loss": best_val,
                "best_epoch": best_epoch,
                "train_samples": len(train_set),
                "val_samples": 0 if val_loader is None else len(val_set),
                "config": ModelConfig(width=args.width, height=args.height, resize_mode="aspect_pad").__dict__,
            }
            torch.save(checkpoint, output_path)
            print(f"Saved best checkpoint: {output_path} (val_loss={best_val:.4f})")

        write_run_artifacts(
            run_dir=run_dir,
            args=args,
            history=history,
            train_samples=len(train_set),
            val_samples=0 if val_loader is None else len(val_set),
            best_epoch=best_epoch,
            best_val=None if best_epoch is None else best_val,
            status="running" if epoch < args.epochs else "completed",
        )

    print(f"Training summary written to: {run_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
