"""Helpers used by the UI to compare multiple runs and summarize the results."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_binary_mask(mask_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    """Load a mask, resize it to the run size, and binarize it."""
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")
    resized = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return (resized > 127).astype(np.uint8)


def compute_mask_metrics(pred_mask: np.ndarray, gt_mask: np.ndarray) -> dict[str, float]:
    """Compute simple overlap scores for a predicted mask."""
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


def _average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _round_or_none(value: float | None, digits: int = 2) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def build_summary(rows: list[dict]) -> dict:
    """Group UI comparison runs into the summary structure shown in the browser."""
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
                "avg_num_paths": _round_or_none(_average([float(row["num_paths"]) for row in group_rows])),
                "avg_num_commands": _round_or_none(_average([float(row["num_commands"]) for row in group_rows])),
                "avg_pen_up_distance_mm": _round_or_none(_average([row["pen_up_distance_mm"] for row in group_rows])),
                "avg_draw_distance_mm": _round_or_none(_average([row["draw_distance_mm"] for row in group_rows])),
                "avg_mask_dice": _round_or_none(_average(mask_dice_values), 4),
                "avg_mask_iou": _round_or_none(_average(mask_iou_values), 4),
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
        improvement_pct = (
            improvement / nearest["pen_up_distance_mm"] * 100.0
            if nearest["pen_up_distance_mm"] > 0
            else 0.0
        )
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
                "The requested ML run fell back during this planning comparison, "
                f"so these results actually used {nearest_effective} and {two_opt_effective}.{suffix}"
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
        dice_delta = None if classical_dice is None or ml_dice is None else round(ml_dice - classical_dice, 4)

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
