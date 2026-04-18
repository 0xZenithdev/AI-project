"""
Prepare paired robot-validation runs for comparing drawing quality settings.

This generates two export folders from the same input image:
- default cleanup enabled
- cleanup disabled ("raw")

Examples:
python -m src.prepare_validation_runs --image images/Testlogo.jpeg
python -m src.prepare_validation_runs --image images/Testlogo.jpeg --output-dir output/validation
"""

from __future__ import annotations

import argparse
from argparse import Namespace
import json
from pathlib import Path

import main as pipeline_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare default-vs-raw robot validation runs.")
    parser.add_argument("--image", type=str, required=True, help="Input image to process")
    parser.add_argument("--output-dir", type=str, default="output/validation", help="Parent output folder")
    parser.add_argument("--width", type=int, default=210, help="Working image width in px")
    parser.add_argument("--height", type=int, default=297, help="Working image height in px")
    parser.add_argument(
        "--vision-mode",
        type=str,
        choices=["classical", "ml"],
        default="classical",
        help="Path extraction mode",
    )
    parser.add_argument("--model-checkpoint", type=str, default="", help="Checkpoint for ML mode")
    parser.add_argument("--ml-threshold", type=float, default=0.5, help="Binary threshold for ML mask")
    parser.add_argument("--ml-device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--paper-width-mm", type=float, default=210.0, help="Paper width in mm")
    parser.add_argument("--paper-height-mm", type=float, default=297.0, help="Paper height in mm")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Drawing margin in mm")
    parser.add_argument("--travel-speed", type=float, default=60.0, help="Pen-up speed")
    parser.add_argument("--draw-speed", type=float, default=35.0, help="Pen-down speed")
    parser.add_argument(
        "--path-ordering",
        type=str,
        choices=["nearest_neighbor", "two_opt"],
        default="two_opt",
        help="Path planning strategy to use in both validation runs",
    )
    parser.add_argument("--two-opt-max-passes", type=int, default=8, help="2-opt refinement passes")
    return parser.parse_args()


def make_pipeline_args(args: argparse.Namespace, output_dir: Path, cleanup_overrides: dict[str, float]) -> Namespace:
    """
    Build a Namespace compatible with main.run_pipeline/export_outputs.

    Memory note:
    - keep these fields aligned with main.parse_args when the main pipeline CLI changes
    - this script exists so we can regenerate the default/raw comparison in one step
    """
    return Namespace(
        image=args.image,
        width=args.width,
        height=args.height,
        vision_mode=args.vision_mode,
        model_checkpoint=args.model_checkpoint,
        ml_threshold=args.ml_threshold,
        ml_device=args.ml_device,
        paper_width_mm=args.paper_width_mm,
        paper_height_mm=args.paper_height_mm,
        margin_mm=args.margin_mm,
        travel_speed=args.travel_speed,
        draw_speed=args.draw_speed,
        min_path_length_mm=cleanup_overrides["min_path_length_mm"],
        min_segment_length_mm=cleanup_overrides["min_segment_length_mm"],
        simplify_tolerance_mm=cleanup_overrides["simplify_tolerance_mm"],
        travel_move_threshold_mm=cleanup_overrides["travel_move_threshold_mm"],
        draw_move_threshold_mm=cleanup_overrides["draw_move_threshold_mm"],
        path_ordering=args.path_ordering,
        two_opt_max_passes=args.two_opt_max_passes,
        output_dir=str(output_dir),
        no_export=False,
    )


def run_variant(label: str, args: argparse.Namespace, output_dir: Path, cleanup_overrides: dict[str, float]) -> dict:
    run_args = make_pipeline_args(args=args, output_dir=output_dir, cleanup_overrides=cleanup_overrides)
    results = pipeline_main.run_pipeline(run_args)
    pipeline_main.export_outputs(run_args, results)

    return {
        "label": label,
        "output_dir": str(output_dir),
        "plot_commands": str(output_dir / "plot_commands.txt"),
        "paths_summary": str(output_dir / "paths_summary.json"),
        "num_paths": len(results["paths_mm"]),
        "num_commands": len(results["commands"]),
        "plan_metrics": {
            "path_count": results["plan_metrics"].path_count,
            "draw_distance_mm": round(results["plan_metrics"].draw_distance_mm, 2),
            "pen_up_distance_mm": round(results["plan_metrics"].pen_up_distance_mm, 2),
            "total_distance_mm": round(results["plan_metrics"].total_distance_mm, 2),
        },
        "cleanup": cleanup_overrides,
        "cleanup_stats": {
            "input_paths": results["cleanup_stats"].input_paths,
            "output_paths": results["cleanup_stats"].output_paths,
            "input_points": results["cleanup_stats"].input_points,
            "output_points": results["cleanup_stats"].output_points,
            "dropped_paths": results["cleanup_stats"].dropped_paths,
            "dropped_points": results["cleanup_stats"].dropped_points,
        },
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_cleanup = {
        "min_path_length_mm": 2.0,
        "min_segment_length_mm": 0.75,
        "simplify_tolerance_mm": 0.5,
        "travel_move_threshold_mm": 0.5,
        "draw_move_threshold_mm": 0.5,
    }
    raw_cleanup = {
        "min_path_length_mm": 0.0,
        "min_segment_length_mm": 0.0,
        "simplify_tolerance_mm": 0.0,
        "travel_move_threshold_mm": 0.0,
        "draw_move_threshold_mm": 0.0,
    }

    default_run = run_variant(
        label="default",
        args=args,
        output_dir=output_dir / "default",
        cleanup_overrides=default_cleanup,
    )
    raw_run = run_variant(
        label="raw",
        args=args,
        output_dir=output_dir / "raw",
        cleanup_overrides=raw_cleanup,
    )

    command_delta = raw_run["num_commands"] - default_run["num_commands"]
    if raw_run["num_commands"] > 0:
        command_reduction_pct = (command_delta / raw_run["num_commands"]) * 100.0
    else:
        command_reduction_pct = 0.0

    summary = {
        "image": args.image,
        "vision_mode": args.vision_mode,
        "path_ordering": args.path_ordering,
        "runs": {
            "default": default_run,
            "raw": raw_run,
        },
        "comparison": {
            "command_delta_raw_minus_default": command_delta,
            "command_reduction_pct": round(command_reduction_pct, 2),
            "recommended_first_robot_test": "default",
        },
    }

    summary_path = output_dir / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Prepared validation runs for: {args.image}")
    print(f"Default commands: {default_run['num_commands']}")
    print(f"Raw commands: {raw_run['num_commands']}")
    print(f"Command reduction: {command_delta} ({command_reduction_pct:.2f}%)")
    print(f"Wrote validation summary to: {summary_path}")


if __name__ == "__main__":
    main()
