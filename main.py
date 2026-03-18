"""
Main pipeline runner for the drawing MVP.

Current scope:
image -> vision paths -> path ordering -> plot commands -> export files

Vision stage now supports:
- classical mode (Canny)
- ml mode (trained segmentation checkpoint)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.search import (
    build_plot_commands,
    commands_to_text,
    order_paths_nearest_neighbor,
    scale_paths_to_mm,
)
from src.vision_v2 import get_drawing_paths_classical, get_drawing_paths_ml


def parse_args() -> argparse.Namespace:
    """CLI arguments so you can test different images without code edits."""
    parser = argparse.ArgumentParser(description="Generate drawing commands from an image.")

    # Input image to process.
    parser.add_argument(
        "--image",
        type=str,
        default="images/Testlogo.jpeg",
        help="Path to input image",
    )

    # Working image size in pixels for classical vision processing.
    parser.add_argument("--width", type=int, default=210, help="Working image width in px")
    parser.add_argument("--height", type=int, default=297, help="Working image height in px")

    # Choose line extraction strategy.
    parser.add_argument(
        "--vision-mode",
        type=str,
        choices=["classical", "ml"],
        default="classical",
        help="classical=Canny, ml=trained model",
    )

    # ML-specific inference options.
    parser.add_argument("--model-checkpoint", type=str, default="", help="Path to trained .pt model")
    parser.add_argument("--ml-threshold", type=float, default=0.5, help="Binary threshold for ML mask")
    parser.add_argument("--ml-device", type=str, default="cpu", help="cpu or cuda")

    # Physical paper configuration in millimeters.
    parser.add_argument("--paper-width-mm", type=float, default=210.0, help="Paper width in mm")
    parser.add_argument("--paper-height-mm", type=float, default=297.0, help="Paper height in mm")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Drawing margin in mm")

    # Speed placeholders (to tune later on real robot).
    parser.add_argument("--travel-speed", type=float, default=60.0, help="Pen-up speed")
    parser.add_argument("--draw-speed", type=float, default=35.0, help="Pen-down speed")

    # Output controls.
    parser.add_argument("--output-dir", type=str, default="output", help="Output folder")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run pipeline without writing output files",
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run full MVP pipeline and return in-memory results."""
    target_size = (args.width, args.height)

    # 1) Vision stage: image -> contour paths in pixel coordinates.
    if args.vision_mode == "classical":
        paths_px, edges = get_drawing_paths_classical(args.image, target_size=target_size)
    else:
        if not args.model_checkpoint:
            raise ValueError("--model-checkpoint is required when --vision-mode ml")
        paths_px, edges = get_drawing_paths_ml(
            image_path=args.image,
            checkpoint_path=args.model_checkpoint,
            threshold=args.ml_threshold,
            device=args.ml_device,
        )

    # 2) Geometry stage: pixel paths -> physical coordinates in mm.
    # In ML mode, model checkpoint stores internal size; we infer from edge map shape.
    image_size_px = (edges.shape[1], edges.shape[0])
    paths_mm = scale_paths_to_mm(
        paths_px=paths_px,
        image_size_px=image_size_px,
        paper_size_mm=(args.paper_width_mm, args.paper_height_mm),
        margin_mm=args.margin_mm,
    )

    # 3) Search/planning stage: choose draw order to reduce travel.
    ordered_paths = order_paths_nearest_neighbor(paths_mm)

    # 4) Command stage: convert paths to pen commands.
    commands = build_plot_commands(
        ordered_paths=ordered_paths,
        travel_speed=args.travel_speed,
        draw_speed=args.draw_speed,
    )

    return {
        "edges": edges,
        "paths_px": paths_px,
        "paths_mm": ordered_paths,
        "commands": commands,
    }


def export_outputs(args: argparse.Namespace, results: dict) -> None:
    """Save debug/inspection artifacts for quick iteration and robot testing."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Edge preview helps you verify what the vision stage extracted.
    edges_path = output_dir / "edges_preview.png"
    cv2.imwrite(str(edges_path), results["edges"])

    # Save path metadata (counts + first few points) for inspection.
    paths_summary = {
        "vision_mode": args.vision_mode,
        "num_paths": len(results["paths_mm"]),
        "num_commands": len(results["commands"]),
        "sample_path_first_points": [
            path[:5] for path in results["paths_mm"][:3]
        ],
    }
    with (output_dir / "paths_summary.json").open("w", encoding="utf-8") as f:
        json.dump(paths_summary, f, indent=2)

    # Save robot command text for the future mBot2 bridge/parser.
    cmd_text = commands_to_text(results["commands"])
    (output_dir / "plot_commands.txt").write_text(cmd_text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    results = run_pipeline(args)

    # Print concise runtime summary for terminal use.
    print(f"Input image: {args.image}")
    print(f"Vision mode: {args.vision_mode}")
    print(f"Detected paths: {len(results['paths_px'])}")
    print(f"Generated commands: {len(results['commands'])}")

    if args.no_export:
        print("No files exported (--no-export).")
        return

    export_outputs(args, results)
    print(f"Exported files to: {args.output_dir}")


if __name__ == "__main__":
    main()
