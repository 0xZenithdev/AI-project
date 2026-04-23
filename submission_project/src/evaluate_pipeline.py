"""
Run repeatable experiments across perception and planning variants.

This script is the bridge between "the project works" and "the project has
measurable results":
- compare classical vs ML perception on the same images
- compare greedy vs TSP-style planning on the same extracted paths
- record metrics that fit the course story

Example:
python -m src.evaluate_pipeline --image images/Testlogo.jpeg
python -m src.evaluate_pipeline --image-dir images --path-ordering nearest_neighbor --path-ordering two_opt
python -m src.evaluate_pipeline --image-dir images --vision-mode classical --vision-mode ml --model-checkpoint models/line_model.pt --masks-dir dataset/masks
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

import main as pipeline_main


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate robot-drawing perception and planning variants.")
    parser.add_argument(
        "--image",
        action="append",
        default=None,
        help="Specific image path to evaluate; repeat for multiple images",
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        default="images",
        help="Fallback directory scanned when no --image is provided",
    )
    parser.add_argument(
        "--masks-dir",
        type=str,
        default="",
        help="Optional directory of ground-truth binary masks matched by image stem",
    )
    parser.add_argument(
        "--vision-mode",
        action="append",
        choices=["classical", "ml"],
        default=None,
        help="Perception mode(s) to evaluate; defaults to classical, or classical+ml if checkpoint is provided",
    )
    parser.add_argument("--model-checkpoint", type=str, default="", help="Checkpoint required for ML mode")
    parser.add_argument("--ml-threshold", type=float, default=0.5, help="Binary threshold for ML mask")
    parser.add_argument("--ml-device", type=str, default="cpu", help="cpu or cuda")
    parser.add_argument(
        "--path-ordering",
        action="append",
        choices=["nearest_neighbor", "two_opt"],
        default=None,
        help="Planning method(s) to evaluate; defaults to both baseline and improved search",
    )
    parser.add_argument("--two-opt-max-passes", type=int, default=8, help="2-opt refinement passes")
    parser.add_argument("--width", type=int, default=210, help="Working image width in px")
    parser.add_argument("--height", type=int, default=297, help="Working image height in px")
    parser.add_argument("--paper-width-mm", type=float, default=210.0, help="Paper width in mm")
    parser.add_argument("--paper-height-mm", type=float, default=297.0, help="Paper height in mm")
    parser.add_argument("--margin-mm", type=float, default=10.0, help="Drawing margin in mm")
    parser.add_argument("--travel-speed", type=float, default=60.0, help="Pen-up speed")
    parser.add_argument("--draw-speed", type=float, default=35.0, help="Pen-down speed")
    parser.add_argument("--min-path-length-mm", type=float, default=2.0, help="Drop tiny paths below this length")
    parser.add_argument("--min-segment-length-mm", type=float, default=0.75, help="Collapse short segments below this length")
    parser.add_argument("--simplify-tolerance-mm", type=float, default=0.5, help="Collinearity simplification tolerance")
    parser.add_argument("--travel-move-threshold-mm", type=float, default=0.5, help="Skip travel moves smaller than this")
    parser.add_argument("--draw-move-threshold-mm", type=float, default=0.5, help="Skip draw moves smaller than this")
    parser.add_argument("--output-dir", type=str, default="output/ai_experiments", help="Parent output directory")
    return parser.parse_args()


def discover_images(args: argparse.Namespace) -> list[Path]:
    if args.image:
        images = [Path(path) for path in args.image]
    else:
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
        images = [
            path
            for path in sorted(image_dir.iterdir())
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]

    resolved = [path.resolve() for path in images]
    if not resolved:
        raise ValueError("No input images were found for evaluation.")
    return resolved


def choose_vision_modes(args: argparse.Namespace) -> list[str]:
    if args.vision_mode:
        modes = args.vision_mode
    elif args.model_checkpoint:
        modes = ["classical", "ml"]
    else:
        modes = ["classical"]

    if "ml" in modes and not args.model_checkpoint:
        raise ValueError("--model-checkpoint is required when evaluating ML mode.")

    return modes


def choose_path_orderings(args: argparse.Namespace) -> list[str]:
    return args.path_ordering or ["nearest_neighbor", "two_opt"]


def build_run_args(
    args: argparse.Namespace,
    image_path: Path,
    output_dir: Path,
    vision_mode: str,
    path_ordering: str,
) -> SimpleNamespace:
    return SimpleNamespace(
        image=str(image_path),
        width=args.width,
        height=args.height,
        vision_mode=vision_mode,
        model_checkpoint=args.model_checkpoint,
        ml_threshold=args.ml_threshold,
        ml_device=args.ml_device,
        paper_width_mm=args.paper_width_mm,
        paper_height_mm=args.paper_height_mm,
        margin_mm=args.margin_mm,
        travel_speed=args.travel_speed,
        draw_speed=args.draw_speed,
        min_path_length_mm=args.min_path_length_mm,
        min_segment_length_mm=args.min_segment_length_mm,
        simplify_tolerance_mm=args.simplify_tolerance_mm,
        travel_move_threshold_mm=args.travel_move_threshold_mm,
        draw_move_threshold_mm=args.draw_move_threshold_mm,
        path_ordering=path_ordering,
        two_opt_max_passes=args.two_opt_max_passes,
        output_dir=str(output_dir),
        no_export=False,
    )


def find_matching_mask(image_path: Path, masks_dir: Path | None) -> Path | None:
    if masks_dir is None:
        return None

    for extension in SUPPORTED_IMAGE_EXTENSIONS:
        candidate = masks_dir / f"{image_path.stem}{extension}"
        if candidate.exists():
            return candidate.resolve()

    return None


def load_binary_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.uint8)


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, np.logical_not(gt)).sum())
    fn = int(np.logical_and(np.logical_not(pred), gt).sum())

    pred_sum = tp + fp
    gt_sum = tp + fn
    union = tp + fp + fn

    precision = tp / pred_sum if pred_sum else 0.0
    recall = tp / gt_sum if gt_sum else 0.0
    dice = (2.0 * tp) / (pred_sum + gt_sum) if (pred_sum + gt_sum) else 0.0
    iou = tp / union if union else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
    }


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def write_results_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "vision_mode",
        "path_ordering",
        "mask_path",
        "mask_precision",
        "mask_recall",
        "mask_dice",
        "mask_iou",
        "num_paths",
        "num_commands",
        "cleanup_input_points",
        "cleanup_output_points",
        "draw_distance_mm",
        "pen_up_distance_mm",
        "total_distance_mm",
        "output_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_robot_scorecard(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "vision_mode",
        "path_ordering",
        "num_paths",
        "num_commands",
        "pen_up_distance_mm",
        "mask_dice",
        "robot_fidelity_score_1_to_5",
        "robot_cleanliness_score_1_to_5",
        "robot_completion_score_1_to_5",
        "robot_notes",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "image": row["image"],
                    "vision_mode": row["vision_mode"],
                    "path_ordering": row["path_ordering"],
                    "num_paths": row["num_paths"],
                    "num_commands": row["num_commands"],
                    "pen_up_distance_mm": row["pen_up_distance_mm"],
                    "mask_dice": row["mask_dice"],
                    "robot_fidelity_score_1_to_5": "",
                    "robot_cleanliness_score_1_to_5": "",
                    "robot_completion_score_1_to_5": "",
                    "robot_notes": "",
                }
            )


def write_summary_markdown(path: Path, summary: dict) -> None:
    lines = [
        "# Experiment Summary",
        "",
        "This report is organized around the AI framing used by the project:",
        "- perception: classical CV vs ML line extraction",
        "- planning: greedy nearest-neighbor vs TSP-style 2-opt refinement",
        "- action: robot command generation after planning",
        "",
        f"Total runs: {summary['run_count']}",
        f"Images: {summary['image_count']}",
        "",
        "## Aggregate Results",
        "",
    ]

    for aggregate in summary["aggregates"]:
        lines.extend(
            [
                f"### {aggregate['vision_mode']} + {aggregate['path_ordering']}",
                "",
                f"- Runs: {aggregate['runs']}",
                f"- Avg commands: {aggregate['avg_num_commands']}",
                f"- Avg paths: {aggregate['avg_num_paths']}",
                f"- Avg pen-up travel mm: {aggregate['avg_pen_up_distance_mm']}",
                f"- Avg draw distance mm: {aggregate['avg_draw_distance_mm']}",
                f"- Avg mask Dice: {aggregate['avg_mask_dice']}",
                f"- Avg mask IoU: {aggregate['avg_mask_iou']}",
                "",
            ]
        )

    if summary["planning_comparisons"]:
        lines.extend(["## Planning Comparisons", ""])
        for comparison in summary["planning_comparisons"]:
            extra = f" {comparison['comparison_note']}" if comparison.get("comparison_note") else ""
            lines.extend(
                [
                    f"- {comparison['image']} ({comparison['vision_mode']}): "
                    f"2-opt improved pen-up travel by {comparison['pen_up_improvement_mm']} mm "
                    f"({comparison['pen_up_improvement_pct']}%).{extra}",
                ]
            )
        lines.append("")

    if summary["vision_comparisons"]:
        lines.extend(["## Perception Comparisons", ""])
        for comparison in summary["vision_comparisons"]:
            extra = f" {comparison['comparison_note']}" if comparison.get("comparison_note") else ""
            lines.extend(
                [
                    f"- {comparison['image']} ({comparison['path_ordering']}): "
                    f"ML minus classical mask Dice = {comparison['mask_dice_delta']}, "
                    f"commands delta = {comparison['num_commands_delta']}.{extra}",
                ]
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def build_summary(rows: list[dict]) -> dict:
    group_map: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["vision_mode"], row["path_ordering"])
        group_map.setdefault(key, []).append(row)

    aggregates = []
    for (vision_mode, path_ordering), group_rows in sorted(group_map.items()):
        mask_dice_values = [row["mask_dice"] for row in group_rows if row["mask_dice"] is not None]
        mask_iou_values = [row["mask_iou"] for row in group_rows if row["mask_iou"] is not None]
        aggregates.append(
            {
                "vision_mode": vision_mode,
                "path_ordering": path_ordering,
                "runs": len(group_rows),
                "avg_num_paths": round_or_none(average([float(row["num_paths"]) for row in group_rows])),
                "avg_num_commands": round_or_none(average([float(row["num_commands"]) for row in group_rows])),
                "avg_pen_up_distance_mm": round_or_none(average([row["pen_up_distance_mm"] for row in group_rows])),
                "avg_draw_distance_mm": round_or_none(average([row["draw_distance_mm"] for row in group_rows])),
                "avg_mask_dice": round_or_none(average(mask_dice_values), 4),
                "avg_mask_iou": round_or_none(average(mask_iou_values), 4),
            }
        )

    planning_map: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        planning_map.setdefault((row["image"], row["vision_mode"]), {})[row["path_ordering"]] = row

    planning_comparisons = []
    for (image, vision_mode), variants in sorted(planning_map.items()):
        nearest = variants.get("nearest_neighbor")
        two_opt = variants.get("two_opt")
        if nearest is None or two_opt is None:
            continue

        improvement = nearest["pen_up_distance_mm"] - two_opt["pen_up_distance_mm"]
        improvement_pct = (improvement / nearest["pen_up_distance_mm"] * 100.0) if nearest["pen_up_distance_mm"] > 0 else 0.0
        requested_mode = str(nearest.get("requested_vision_mode", vision_mode))
        nearest_effective = str(nearest.get("effective_vision_mode", nearest["vision_mode"]))
        two_opt_effective = str(two_opt.get("effective_vision_mode", two_opt["vision_mode"]))
        comparison_note = None
        if requested_mode == "ml" and (nearest_effective != "ml" or two_opt_effective != "ml"):
            reasons = {
                str(item.get("fallback_reason", "")).strip()
                for item in (nearest, two_opt)
                if str(item.get("fallback_reason", "")).strip()
            }
            suffix = f" (reason: {', '.join(sorted(reasons))})" if reasons else ""
            comparison_note = (
                "Requested ML fell back during this planning comparison, "
                f"so the runs actually used {nearest_effective} and {two_opt_effective}.{suffix}"
            )
        planning_comparisons.append(
            {
                "image": image,
                "vision_mode": vision_mode,
                "requested_vision_mode": requested_mode,
                "effective_vision_mode": nearest_effective if nearest_effective == two_opt_effective else "mixed",
                "comparison_note": comparison_note,
                "nearest_neighbor": {
                    "path_ordering": nearest["path_ordering"],
                    "requested_vision_mode": str(nearest.get("requested_vision_mode", nearest["vision_mode"])),
                    "effective_vision_mode": nearest_effective,
                    "fallback_reason": nearest.get("fallback_reason"),
                    "num_commands": nearest["num_commands"],
                    "num_paths": nearest["num_paths"],
                    "pen_up_distance_mm": nearest["pen_up_distance_mm"],
                },
                "two_opt": {
                    "path_ordering": two_opt["path_ordering"],
                    "requested_vision_mode": str(two_opt.get("requested_vision_mode", two_opt["vision_mode"])),
                    "effective_vision_mode": two_opt_effective,
                    "fallback_reason": two_opt.get("fallback_reason"),
                    "num_commands": two_opt["num_commands"],
                    "num_paths": two_opt["num_paths"],
                    "pen_up_distance_mm": two_opt["pen_up_distance_mm"],
                },
                "pen_up_improvement_mm": round(improvement, 2),
                "pen_up_improvement_pct": round(improvement_pct, 2),
            }
        )

    vision_map: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        vision_map.setdefault((row["image"], row["path_ordering"]), {})[row["vision_mode"]] = row

    vision_comparisons = []
    for (image, path_ordering), variants in sorted(vision_map.items()):
        classical = variants.get("classical")
        ml = variants.get("ml")
        if classical is None or ml is None:
            continue

        classical_dice = classical["mask_dice"]
        ml_dice = ml["mask_dice"]
        if classical_dice is None or ml_dice is None:
            dice_delta = None
        else:
            dice_delta = round(ml_dice - classical_dice, 4)

        classical_effective = str(classical.get("effective_vision_mode", classical["vision_mode"]))
        ml_effective = str(ml.get("effective_vision_mode", ml["vision_mode"]))
        comparison_note = None
        if classical_effective != "classical" or ml_effective != "ml":
            notes = []
            if classical_effective != "classical":
                notes.append(f"classical run resolved to {classical_effective}")
            if ml_effective != "ml":
                fallback_reason = str(ml.get("fallback_reason", "")).strip()
                if fallback_reason:
                    notes.append(f"ML fell back to {ml_effective} (reason: {fallback_reason})")
                else:
                    notes.append(f"ML fell back to {ml_effective}")
            comparison_note = "This is not a pure classical-vs-ML comparison because " + "; ".join(notes) + "."

        vision_comparisons.append(
            {
                "image": image,
                "path_ordering": path_ordering,
                "comparison_note": comparison_note,
                "classical": {
                    "requested_vision_mode": str(classical.get("requested_vision_mode", classical["vision_mode"])),
                    "effective_vision_mode": classical_effective,
                    "fallback_reason": classical.get("fallback_reason"),
                    "num_commands": classical["num_commands"],
                    "num_paths": classical["num_paths"],
                    "pen_up_distance_mm": classical["pen_up_distance_mm"],
                    "mask_dice": classical["mask_dice"],
                },
                "ml": {
                    "requested_vision_mode": str(ml.get("requested_vision_mode", ml["vision_mode"])),
                    "effective_vision_mode": ml_effective,
                    "fallback_reason": ml.get("fallback_reason"),
                    "num_commands": ml["num_commands"],
                    "num_paths": ml["num_paths"],
                    "pen_up_distance_mm": ml["pen_up_distance_mm"],
                    "mask_dice": ml["mask_dice"],
                },
                "mask_dice_delta": dice_delta,
                "num_commands_delta": ml["num_commands"] - classical["num_commands"],
                "pen_up_distance_delta_mm": round(ml["pen_up_distance_mm"] - classical["pen_up_distance_mm"], 2),
            }
        )

    return {
        "run_count": len(rows),
        "image_count": len(sorted({row["image"] for row in rows})),
        "aggregates": aggregates,
        "planning_comparisons": planning_comparisons,
        "vision_comparisons": vision_comparisons,
    }


def main() -> None:
    args = parse_args()
    images = discover_images(args)
    vision_modes = choose_vision_modes(args)
    path_orderings = choose_path_orderings(args)
    masks_dir = Path(args.masks_dir).resolve() if args.masks_dir else None

    session_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = Path(args.output_dir) / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []

    for image_path in images:
        mask_path = find_matching_mask(image_path=image_path, masks_dir=masks_dir)

        for vision_mode in vision_modes:
            for path_ordering in path_orderings:
                run_label = f"{image_path.stem}__{vision_mode}__{path_ordering}"
                run_output_dir = output_dir / "runs" / run_label
                run_args = build_run_args(
                    args=args,
                    image_path=image_path,
                    output_dir=run_output_dir,
                    vision_mode=vision_mode,
                    path_ordering=path_ordering,
                )

                results = pipeline_main.run_pipeline(run_args)
                pipeline_main.export_outputs(run_args, results)

                mask_metrics: dict[str, float] | None = None
                if mask_path is not None:
                    pred_mask = (results["edges"] > 0).astype(np.uint8)
                    gt_mask = load_binary_mask(
                        mask_path=mask_path,
                        target_size=(results["edges"].shape[1], results["edges"].shape[0]),
                    )
                    mask_metrics = compute_mask_metrics(pred_mask=pred_mask, gt_mask=gt_mask)

                row = {
                    "image": image_path.name,
                    "vision_mode": vision_mode,
                    "path_ordering": path_ordering,
                    "mask_path": str(mask_path) if mask_path is not None else "",
                    "mask_precision": round(mask_metrics["precision"], 4) if mask_metrics is not None else None,
                    "mask_recall": round(mask_metrics["recall"], 4) if mask_metrics is not None else None,
                    "mask_dice": round(mask_metrics["dice"], 4) if mask_metrics is not None else None,
                    "mask_iou": round(mask_metrics["iou"], 4) if mask_metrics is not None else None,
                    "num_paths": len(results["paths_mm"]),
                    "num_commands": len(results["commands"]),
                    "cleanup_input_points": results["cleanup_stats"].input_points,
                    "cleanup_output_points": results["cleanup_stats"].output_points,
                    "draw_distance_mm": round(results["plan_metrics"].draw_distance_mm, 2),
                    "pen_up_distance_mm": round(results["plan_metrics"].pen_up_distance_mm, 2),
                    "total_distance_mm": round(results["plan_metrics"].total_distance_mm, 2),
                    "output_dir": str(run_output_dir),
                }
                rows.append(row)

                print(
                    f"[run] {image_path.name} | {vision_mode} | {path_ordering} | "
                    f"commands={row['num_commands']} | pen_up_mm={row['pen_up_distance_mm']}"
                )

    summary = build_summary(rows)
    summary_payload = {
        "images": [str(path) for path in images],
        "vision_modes": vision_modes,
        "path_orderings": path_orderings,
        "masks_dir": str(masks_dir) if masks_dir is not None else "",
        "model_checkpoint": args.model_checkpoint,
        "results": rows,
        "summary": summary,
    }

    summary_json_path = output_dir / "experiment_results.json"
    summary_json_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    write_results_csv(output_dir / "experiment_results.csv", rows)
    write_robot_scorecard(output_dir / "robot_quality_scorecard.csv", rows)
    write_summary_markdown(output_dir / "experiment_summary.md", summary)

    print(f"Wrote experiment JSON to: {summary_json_path}")
    print(f"Wrote experiment CSV to: {output_dir / 'experiment_results.csv'}")
    print(f"Wrote robot scorecard to: {output_dir / 'robot_quality_scorecard.csv'}")
    print(f"Wrote experiment summary to: {output_dir / 'experiment_summary.md'}")


if __name__ == "__main__":
    main()
