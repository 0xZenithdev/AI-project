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
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from src.ml_line_model import LinePairDataset, ModelConfig, TinyUNet, combined_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a line-segmentation model for robot drawing.")

    parser.add_argument("--images-dir", type=str, default="dataset/images", help="Input images directory")
    parser.add_argument("--masks-dir", type=str, default="dataset/masks", help="Target masks directory")

    parser.add_argument("--width", type=int, default=210, help="Training width")
    parser.add_argument("--height", type=int, default=297, help="Training height")

    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--val-split", type=float, default=0.2, help="Validation split ratio")

    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=str, default="models/line_model.pt", help="Checkpoint output path")

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


def main() -> None:
    args = parse_args()

    device = torch.device(args.device)
    dataset = LinePairDataset(
        images_dir=args.images_dir,
        masks_dir=args.masks_dir,
        size=(args.width, args.height),
    )

    # Split into train and validation subsets.
    val_len = int(len(dataset) * args.val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False) if val_len > 0 else None

    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val = float("inf")

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

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        else:
            val_loss = train_loss
            print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f}")

        # Save best checkpoint based on validation loss.
        if val_loss < best_val:
            best_val = val_loss
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state": model.state_dict(),
                "width": args.width,
                "height": args.height,
                "best_val_loss": best_val,
                "config": ModelConfig(width=args.width, height=args.height).__dict__,
            }
            torch.save(checkpoint, out_path)
            print(f"Saved best checkpoint: {out_path} (val_loss={best_val:.4f})")


if __name__ == "__main__":
    main()
