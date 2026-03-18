"""
Inference entry-point for the trained line model.

Usage example:
python -m src.infer --image images/Testlogo.jpeg --checkpoint models/line_model.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from src.ml_line_model import predict_line_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained line-segmentation model inference.")

    parser.add_argument("--image", type=str, required=True, help="Input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Trained model checkpoint")
    parser.add_argument("--threshold", type=float, default=0.5, help="Binary threshold (0..1)")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--output", type=str, default="output/ml_pred_mask.png", help="Output mask path")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    mask = predict_line_mask(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        device_str=args.device,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), mask)

    print(f"Saved predicted mask to: {out_path}")


if __name__ == "__main__":
    main()
