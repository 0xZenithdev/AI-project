"""Run the image-to-drawing pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from src.search import (
    build_plot_commands,
    cleanup_paths,
    commands_to_text,
    order_paths,
    scale_paths_to_mm,
    stitch_ordered_paths,
    summarize_ordered_paths,
)
from src.vision_v2 import get_drawing_paths_classical, get_drawing_paths_ml


DEFAULT_PEN_ADVANCE_FROM_BOTTOM_MM = 200.0


def resolve_cleanup_settings(args: argparse.Namespace, raw_path_count: int) -> tuple[dict[str, float], str]:
    """Return cleanup settings and whether they are default or manual."""
    defaults = {
        "min_path_length_mm": 2.0,
        "min_segment_length_mm": 0.75,
        "simplify_tolerance_mm": 0.5,
    }
    using_default_cleanup = (
        abs(float(args.min_path_length_mm) - defaults["min_path_length_mm"]) < 1e-9
        and abs(float(args.min_segment_length_mm) - defaults["min_segment_length_mm"]) < 1e-9
        and abs(float(args.simplify_tolerance_mm) - defaults["simplify_tolerance_mm"]) < 1e-9
    )

    settings = {
        "min_path_length_mm": float(args.min_path_length_mm),
        "min_segment_length_mm": float(args.min_segment_length_mm),
        "simplify_tolerance_mm": float(args.simplify_tolerance_mm),
    }
    _ = raw_path_count
    if using_default_cleanup:
        return settings, "default"
    return settings, "manual"


def resolve_start_pose(args: argparse.Namespace) -> tuple[tuple[float, float], float]:
    """
    Resolve the pen-tip start pose used by planning and robot export.

    Coordinates follow the paper reference frame:
    - x increases to the right
    - y increases downward
    - heading 90 degrees faces into the page from the bottom edge
    """
    paper_width_mm = float(getattr(args, "paper_width_mm", 210.0))
    paper_height_mm = float(getattr(args, "paper_height_mm", 297.0))
    start_x_mm = getattr(args, "start_x_mm", None)
    start_y_mm = getattr(args, "start_y_mm", None)
    start_heading_deg = float(getattr(args, "start_heading_deg", 90.0))

    if start_x_mm is None:
        start_x_mm = paper_width_mm / 2.0
    if start_y_mm is None:
        start_y_mm = max(0.0, paper_height_mm - DEFAULT_PEN_ADVANCE_FROM_BOTTOM_MM)

    return (float(start_x_mm), float(start_y_mm)), start_heading_deg


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate drawing commands from an image.")
    parser.add_argument(
        "--image",
        type=str,
        default="images/Testlogo.jpeg",
        help="Path to input image",
    )
    parser.add_argument("--width", type=int, default=210, help="Working image width in px")
    parser.add_argument("--height", type=int, default=297, help="Working image height in px")
    parser.add_argument(
        "--vision-mode",
        type=str,
        choices=["classical", "ml"],
        default="classical",
        help="classical=Canny, ml=trained model",
    )
    parser.add_argument("--model-checkpoint", type=str, default="", help="Path to trained .pt model")
    parser.add_argument("--ml-threshold", type=float, default=0.5, help="Binary threshold for ML mask")
    parser.add_argument("--ml-device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument("--paper-width-mm", type=float, default=210.0, help="Paper width in mm")
    parser.add_argument("--paper-height-mm", type=float, default=297.0, help="Paper height in mm")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Drawing margin in mm")
    parser.add_argument(
        "--start-x-mm",
        type=float,
        default=None,
        help="Initial pen-tip x position. Defaults to paper center x.",
    )
    parser.add_argument(
        "--start-y-mm",
        type=float,
        default=None,
        help="Initial pen-tip y position. Defaults to 200 mm inside from the bottom paper edge.",
    )
    parser.add_argument(
        "--start-heading-deg",
        type=float,
        default=90.0,
        help="Initial robot heading in project coordinates; 90 faces into the paper from the bottom edge.",
    )
    parser.add_argument("--travel-speed", type=float, default=60.0, help="Pen-up speed")
    parser.add_argument("--draw-speed", type=float, default=35.0, help="Pen-down speed")
    parser.add_argument(
        "--min-path-length-mm",
        type=float,
        default=2.0,
        help="Drop tiny paths shorter than this length after cleanup",
    )
    parser.add_argument(
        "--min-segment-length-mm",
        type=float,
        default=0.75,
        help="Collapse consecutive points that are closer than this distance",
    )
    parser.add_argument(
        "--simplify-tolerance-mm",
        type=float,
        default=0.5,
        help="Remove middle points that are nearly collinear within this tolerance",
    )
    parser.add_argument(
        "--travel-move-threshold-mm",
        type=float,
        default=0.5,
        help="Skip travel moves smaller than this threshold",
    )
    parser.add_argument(
        "--draw-move-threshold-mm",
        type=float,
        default=0.5,
        help="Skip draw moves smaller than this threshold",
    )
    parser.add_argument(
        "--path-ordering",
        type=str,
        choices=["nearest_neighbor", "two_opt"],
        default="two_opt",
        help="Path planning strategy: greedy baseline or TSP-style 2-opt refinement",
    )
    parser.add_argument(
        "--two-opt-max-passes",
        type=int,
        default=8,
        help="Maximum refinement passes when --path-ordering two_opt",
    )
    parser.add_argument("--output-dir", type=str, default="output", help="Output folder")
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="Run pipeline without writing output files",
    )

    return parser.parse_args()


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run the full pipeline and return the results."""
    target_size = (args.width, args.height)
    start_point, start_heading_deg = resolve_start_pose(args)
    if args.vision_mode == "classical":
        paths_px, edges, vision_info = get_drawing_paths_classical(
            args.image,
            target_size=target_size,
            return_info=True,
        )
    else:
        if not args.model_checkpoint:
            raise ValueError("--model-checkpoint is required when --vision-mode ml")
        paths_px, edges, vision_info = get_drawing_paths_ml(
            image_path=args.image,
            checkpoint_path=args.model_checkpoint,
            threshold=args.ml_threshold,
            device=args.ml_device,
            return_info=True,
        )
    image_size_px = (edges.shape[1], edges.shape[0])
    paths_mm = scale_paths_to_mm(
        paths_px=paths_px,
        image_size_px=image_size_px,
        paper_size_mm=(args.paper_width_mm, args.paper_height_mm),
        margin_mm=args.margin_mm,
    )
    cleanup_settings, cleanup_profile = resolve_cleanup_settings(args, raw_path_count=len(paths_px))
    cleaned_paths_mm, cleanup_stats = cleanup_paths(
        paths_mm,
        min_path_length_mm=cleanup_settings["min_path_length_mm"],
        min_segment_length_mm=cleanup_settings["min_segment_length_mm"],
        simplify_tolerance_mm=cleanup_settings["simplify_tolerance_mm"],
    )
    ordered_paths = order_paths(
        cleaned_paths_mm,
        method=args.path_ordering,
        start_point=start_point,
        two_opt_max_passes=args.two_opt_max_passes,
    )
    stitched_paths_mm, stitch_stats = stitch_ordered_paths(
        ordered_paths,
        join_tolerance_mm=max(args.travel_move_threshold_mm, args.draw_move_threshold_mm),
    )
    plan_metrics = summarize_ordered_paths(stitched_paths_mm, start_point=start_point)
    commands = build_plot_commands(
        ordered_paths=stitched_paths_mm,
        travel_speed=args.travel_speed,
        draw_speed=args.draw_speed,
        travel_move_threshold_mm=args.travel_move_threshold_mm,
        draw_move_threshold_mm=args.draw_move_threshold_mm,
        start_point=start_point,
    )

    return {
        "edges": edges,
        "vision_info": vision_info,
        "paths_px": paths_px,
        "paths_mm_raw": paths_mm,
        "paths_mm_ordered": ordered_paths,
        "paths_mm": stitched_paths_mm,
        "cleanup_stats": cleanup_stats,
        "stitch_stats": stitch_stats,
        "cleanup_profile": cleanup_profile,
        "cleanup_settings": cleanup_settings,
        "plan_metrics": plan_metrics,
        "start_pose": {
            "tip_x_mm": start_point[0],
            "tip_y_mm": start_point[1],
            "heading_deg": start_heading_deg,
        },
        "commands": commands,
    }


def export_outputs(args: argparse.Namespace, results: dict) -> None:
    """Write output files for the current run."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / "edges_preview.png"
    cv2.imwrite(str(edges_path), results["edges"])
    paths_summary = {
        "vision_mode": args.vision_mode,
        "effective_vision_mode": results["vision_info"]["effective_mode"],
        "vision_info": results["vision_info"],
        "path_ordering": args.path_ordering,
        "start_pose": results["start_pose"],
        "cleanup": {
            "profile": results["cleanup_profile"],
            "min_path_length_mm": results["cleanup_settings"]["min_path_length_mm"],
            "min_segment_length_mm": results["cleanup_settings"]["min_segment_length_mm"],
            "simplify_tolerance_mm": results["cleanup_settings"]["simplify_tolerance_mm"],
            "travel_move_threshold_mm": args.travel_move_threshold_mm,
            "draw_move_threshold_mm": args.draw_move_threshold_mm,
        },
        "plan_metrics": {
            "path_count": results["plan_metrics"].path_count,
            "draw_distance_mm": round(results["plan_metrics"].draw_distance_mm, 2),
            "pen_up_distance_mm": round(results["plan_metrics"].pen_up_distance_mm, 2),
            "total_distance_mm": round(results["plan_metrics"].total_distance_mm, 2),
        },
        "cleanup_stats": {
            "input_paths": results["cleanup_stats"].input_paths,
            "output_paths": results["cleanup_stats"].output_paths,
            "input_points": results["cleanup_stats"].input_points,
            "output_points": results["cleanup_stats"].output_points,
            "dropped_paths": results["cleanup_stats"].dropped_paths,
            "dropped_points": results["cleanup_stats"].dropped_points,
        },
        "stitching": {
            "input_paths": results["stitch_stats"].input_paths,
            "output_paths": results["stitch_stats"].output_paths,
            "merged_paths": results["stitch_stats"].merged_paths,
        },
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
    if results["vision_info"]["effective_mode"] != args.vision_mode:
        print(
            "Effective vision mode: "
            f"{results['vision_info']['effective_mode']} "
            f"(fallback reason: {results['vision_info'].get('fallback_reason', 'n/a')})"
        )
    print(f"Path ordering: {args.path_ordering}")
    print(
        "Start pose: "
        f"pen tip=({results['start_pose']['tip_x_mm']:.2f}, "
        f"{results['start_pose']['tip_y_mm']:.2f}) mm, "
        f"heading={results['start_pose']['heading_deg']:.2f} deg"
    )
    print(f"Detected paths: {len(results['paths_px'])}")
    print(
        "Cleanup stats: "
        f"{results['cleanup_stats'].input_points} pts -> "
        f"{results['cleanup_stats'].output_points} pts, "
        f"{results['cleanup_stats'].input_paths} paths -> "
        f"{results['cleanup_stats'].output_paths} paths"
    )
    print(
        "Plan metrics: "
        f"pen-up={results['plan_metrics'].pen_up_distance_mm:.2f} mm, "
        f"draw={results['plan_metrics'].draw_distance_mm:.2f} mm"
    )
    print(
        "Path stitching: "
        f"{results['stitch_stats'].input_paths} ordered paths -> "
        f"{results['stitch_stats'].output_paths} stitched paths"
    )
    print(f"Generated commands: {len(results['commands'])}")

    if args.no_export:
        print("No files exported (--no-export).")
        return

    export_outputs(args, results)
    print(f"Exported files to: {args.output_dir}")


if __name__ == "__main__":
    main()
