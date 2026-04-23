"""
Shared dataset utilities for the robot-drawing ML workflow.

Why this file exists:
- keep image/mask matching logic in one place
- let audit, split, training, and readiness tools agree on the same rules
- make the ML data pipeline easier to use without robot access
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def list_supported_files(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []

    return sorted(
        [
            item
            for item in path.iterdir()
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )


def list_supported_files_recursive(path: Path) -> list[Path]:
    if not path.exists() or not path.is_dir():
        return []

    return sorted(
        [
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )


def match_image_mask_pairs(images_dir: Path, masks_dir: Path) -> tuple[list[tuple[Path, Path]], list[Path], list[Path]]:
    image_files = list_supported_files(images_dir)
    mask_files = list_supported_files(masks_dir)

    image_by_stem = {path.stem: path for path in image_files}
    mask_by_stem = {path.stem: path for path in mask_files}

    matched_stems = sorted(set(image_by_stem) & set(mask_by_stem))
    image_only = sorted(set(image_by_stem) - set(mask_by_stem))
    mask_only = sorted(set(mask_by_stem) - set(image_by_stem))

    pairs = [(image_by_stem[stem], mask_by_stem[stem]) for stem in matched_stems]
    return pairs, [image_by_stem[stem] for stem in image_only], [mask_by_stem[stem] for stem in mask_only]


def load_manifest_pairs(manifest_path: Path) -> list[tuple[Path, Path]]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        records = payload.get("pairs", [])
    elif isinstance(payload, list):
        records = payload
    else:
        raise ValueError(f"Unsupported manifest format: {manifest_path}")

    pairs: list[tuple[Path, Path]] = []
    for record in records:
        image_path = Path(record["image"]).resolve()
        mask_path = Path(record["mask"]).resolve()
        pairs.append((image_path, mask_path))

    return pairs


def save_manifest_pairs(
    pairs: Iterable[tuple[Path, Path]],
    output_path: Path,
    split_name: str,
    seed: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "stem": image_path.stem,
            "image": str(image_path.resolve()),
            "mask": str(mask_path.resolve()),
        }
        for image_path, mask_path in pairs
    ]
    payload = {
        "split": split_name,
        "seed": seed,
        "pair_count": len(records),
        "pairs": records,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def inspect_dataset_pair(image_path: Path, mask_path: Path) -> dict:
    """
    Inspect one image/mask pair for common dataset-quality issues.

    "binary_like" is intentionally tolerant:
    - exact binary means pixels are only 0 or 255
    - binary-like means nearly all pixels are near 0 or 255, so the mask is
      probably usable after thresholding even if compression introduced noise
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    if mask_gray is None:
        raise FileNotFoundError(f"Could not read mask: {mask_path}")

    unique_values = np.unique(mask_gray)
    exact_binary = bool(set(int(value) for value in unique_values.tolist()).issubset({0, 255}))
    non_binary_pixels = int(np.count_nonzero((mask_gray > 5) & (mask_gray < 250)))
    total_pixels = int(mask_gray.size)
    binary_like_ratio = 1.0 - (non_binary_pixels / max(1, total_pixels))
    binary_like = binary_like_ratio >= 0.99
    white_ratio = float(np.count_nonzero(mask_gray > 127) / max(1, total_pixels))

    return {
        "stem": image_path.stem,
        "image_path": str(image_path),
        "mask_path": str(mask_path),
        "image_shape": [int(image_bgr.shape[0]), int(image_bgr.shape[1]), int(image_bgr.shape[2])],
        "mask_shape": [int(mask_gray.shape[0]), int(mask_gray.shape[1])],
        "same_size": bool(image_bgr.shape[0] == mask_gray.shape[0] and image_bgr.shape[1] == mask_gray.shape[1]),
        "exact_binary_mask": exact_binary,
        "binary_like_mask": binary_like,
        "binary_like_ratio": round(binary_like_ratio, 4),
        "white_ratio": round(white_ratio, 4),
        "unique_value_count": int(len(unique_values)),
        "unique_values_preview": [int(value) for value in unique_values[:10].tolist()],
    }


def generate_classical_bootstrap_mask(
    image_path: Path,
    blur_kernel: int = 3,
    canny_low: int = 80,
    canny_high: int = 160,
    close_iters: int = 1,
    dilate_iters: int = 0,
    min_component_area: int = 16,
) -> np.ndarray:
    """
    Generate a draft binary mask from an RGB image using classical CV.

    Implementation note:
    - this is a dataset bootstrap helper, not a gold-standard labeler
    - the output should be treated as a draft mask that can be reviewed or
      corrected before serious training
    """
    image_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if image_gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    kernel_size = max(1, int(blur_kernel))
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image_gray, (kernel_size, kernel_size), 0)
    mask = cv2.Canny(blurred, threshold1=int(canny_low), threshold2=int(canny_high))

    kernel = np.ones((3, 3), dtype=np.uint8)
    if close_iters > 0:
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=int(close_iters))
    if dilate_iters > 0:
        mask = cv2.dilate(mask, kernel, iterations=int(dilate_iters))

    if min_component_area > 0:
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        cleaned = np.zeros_like(mask)
        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area >= int(min_component_area):
                cleaned[labels == label_idx] = 255
        mask = cleaned

    return mask


def convert_binary_mask_to_contour_mask(mask_gray: np.ndarray, contour_thickness: int = 1) -> np.ndarray:
    """
    Convert a filled binary segmentation mask into a thin contour mask.

    This is useful when a public dataset provides filled logo/shape regions,
    but the downstream task in this repo wants drawable line targets.
    """
    if mask_gray.ndim != 2:
        raise ValueError("convert_binary_mask_to_contour_mask expects a single-channel mask.")

    thickness = max(1, int(contour_thickness))
    binary = (mask_gray > 127).astype(np.uint8) * 255
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    contour_mask = np.zeros_like(binary)
    if contours:
        cv2.drawContours(contour_mask, contours, contourIdx=-1, color=255, thickness=thickness)
    return contour_mask


def write_bootstrap_preview(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    """
    Save a side-by-side preview for draft-mask review.

    Layout:
    - original image
    - overlay with draft mask highlighted
    - pure binary mask
    """
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if mask.ndim != 2:
        raise ValueError("Bootstrap preview expects a single-channel binary mask.")

    mask_bool = mask > 127
    overlay = image_bgr.copy()
    highlight = np.zeros_like(image_bgr)
    highlight[:, :] = (48, 105, 219)
    overlay[mask_bool] = cv2.addWeighted(overlay, 0.3, highlight, 0.7, 0.0)[mask_bool]

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    preview = np.concatenate([image_bgr, overlay, mask_bgr], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), preview)


def binarize_line_art(
    image_gray: np.ndarray,
    threshold: int = 245,
    foreground_mode: str = "auto",
) -> np.ndarray:
    if image_gray.ndim != 2:
        raise ValueError("binarize_line_art expects a single-channel grayscale image.")

    threshold = max(0, min(255, int(threshold)))
    if foreground_mode not in {"auto", "dark_on_light", "light_on_dark"}:
        raise ValueError(f"Unsupported foreground_mode: {foreground_mode}")

    mode = foreground_mode
    if mode == "auto":
        mode = "dark_on_light" if float(image_gray.mean()) >= 127.5 else "light_on_dark"

    if mode == "dark_on_light":
        mask = image_gray < threshold
    else:
        mask = image_gray > threshold

    return mask.astype(np.uint8) * 255


def audit_dataset(images_dir: Path, masks_dir: Path) -> dict:
    pairs, image_only, mask_only = match_image_mask_pairs(images_dir=images_dir, masks_dir=masks_dir)
    pair_reports: list[dict] = []
    unreadable_pairs: list[dict] = []

    for image_path, mask_path in pairs:
        try:
            pair_reports.append(inspect_dataset_pair(image_path=image_path, mask_path=mask_path))
        except Exception as exc:
            unreadable_pairs.append(
                {
                    "stem": image_path.stem,
                    "image_path": str(image_path),
                    "mask_path": str(mask_path),
                    "error": str(exc),
                }
            )

    exact_binary_count = sum(1 for report in pair_reports if report["exact_binary_mask"])
    binary_like_count = sum(1 for report in pair_reports if report["binary_like_mask"])
    same_size_count = sum(1 for report in pair_reports if report["same_size"])

    suspicious_pairs = [
        report
        for report in pair_reports
        if (not report["binary_like_mask"]) or (report["white_ratio"] <= 0.001) or (report["white_ratio"] >= 0.999)
    ]

    return {
        "paths": {
            "images_dir": str(images_dir),
            "masks_dir": str(masks_dir),
        },
        "counts": {
            "image_files": len(list_supported_files(images_dir)),
            "mask_files": len(list_supported_files(masks_dir)),
            "matched_pairs": len(pairs),
            "image_only": len(image_only),
            "mask_only": len(mask_only),
            "audited_pairs": len(pair_reports),
            "unreadable_pairs": len(unreadable_pairs),
            "exact_binary_masks": exact_binary_count,
            "binary_like_masks": binary_like_count,
            "same_size_pairs": same_size_count,
            "suspicious_pairs": len(suspicious_pairs),
        },
        "examples": {
            "image_only": [path.name for path in image_only[:10]],
            "mask_only": [path.name for path in mask_only[:10]],
            "suspicious_pairs": [report["stem"] for report in suspicious_pairs[:10]],
            "unreadable_pairs": [item["stem"] for item in unreadable_pairs[:10]],
        },
        "pairs": pair_reports,
        "unreadable_pair_details": unreadable_pairs,
    }
