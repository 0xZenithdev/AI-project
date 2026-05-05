"""Functions used to extract drawable paths from an image for the robot workflow."""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

from src.image_geometry import load_image_bgr, load_image_grayscale, resize_with_aspect_pad
from src.path_tracing import (
    Path,
    clean_binary_line_map,
    extract_axis_aligned_stroke_paths,
    extract_edge_contour_paths,
    extract_filled_region_paths,
    extract_paths_from_line_map,
    render_trace_map,
)


Point = Tuple[int, int]


def suppress_border_components(
    binary_mask: np.ndarray,
    min_area_ratio: float = 0.02,
    min_span_ratio: float = 0.35,
) -> np.ndarray:
    """
    Remove giant border-touching components that are usually page/background.

    This is the main protection against tracing the paper border in photographed
    sketches or shaded backgrounds.
    """
    binary = (binary_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return binary * 255

    height, width = binary.shape[:2]
    total_pixels = float(height * width)
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = binary.copy() * 255

    for index in range(1, component_count):
        x, y, component_width, component_height, area = stats[index]
        touches_border = (
            x == 0
            or y == 0
            or (x + component_width) == width
            or (y + component_height) == height
        )
        if not touches_border:
            continue

        area_ratio = float(area) / total_pixels
        span_ratio = max(
            float(component_width) / float(width),
            float(component_height) / float(height),
        )
        if area_ratio >= min_area_ratio or span_ratio >= min_span_ratio:
            filtered[labels == index] = 0

    return filtered


def filter_outline_components(
    binary_mask: np.ndarray,
    min_area_px: int = 12,
    min_short_side_px: int = 12,
    min_line_length_px: int = 40,
    min_line_aspect_ratio: float = 8.0,
) -> np.ndarray:
    """
    Keep components that look like real drawing structure, not tiny caption text.

    A component survives when it is either:
    - shape-like: both dimensions are meaningfully large, or
    - line-like: very elongated, such as a long single stroke
    """
    binary = (binary_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return binary * 255

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary, dtype=np.uint8)

    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        if int(area) < min_area_px:
            continue

        short_side = float(min(width, height))
        long_side = float(max(width, height))
        if short_side >= float(min_short_side_px):
            filtered[labels == index] = 255
            continue

        if short_side <= 0.0 or long_side < float(min_line_length_px):
            continue

        aspect_ratio = long_side / short_side
        if aspect_ratio >= min_line_aspect_ratio:
            filtered[labels == index] = 255

    return filtered


def filter_blob_components(
    binary_mask: np.ndarray,
    min_area_px: int = 40,
    min_short_side_px: int = 8,
    min_line_length_px: int = 40,
    min_line_aspect_ratio: float = 8.0,
) -> np.ndarray:
    """
    Keep meaningful components in a binary mask.

    This preserves both:
    - blob-like filled shapes
    - long thin strokes such as a single vertical or horizontal line
    """
    binary = (binary_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return binary * 255

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary, dtype=np.uint8)

    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        if int(area) < min_area_px:
            continue
        short_side = float(min(int(width), int(height)))
        long_side = float(max(int(width), int(height)))
        if short_side >= float(min_short_side_px):
            filtered[labels == index] = 255
            continue
        if short_side <= 0.0 or long_side < float(min_line_length_px):
            continue
        aspect_ratio = long_side / short_side
        if aspect_ratio >= min_line_aspect_ratio:
            filtered[labels == index] = 255

    return filtered


def preprocess_foreground_mask(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
    blur_kernel: int = 3,
    illumination_sigma: float = 15.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a normalized dark-foreground mask for the classical path."""
    img_gray = load_image_grayscale(image_path)
    img_resized = resize_with_aspect_pad(img_gray, target_size=target_size, pad_value=255)
    illumination = cv2.GaussianBlur(img_resized, (0, 0), illumination_sigma)
    normalized = cv2.divide(img_resized, illumination, scale=255)
    normalized = cv2.GaussianBlur(normalized, (blur_kernel, blur_kernel), 0)
    _, dark_foreground = cv2.threshold(
        normalized,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )
    kernel = np.ones((3, 3), np.uint8)
    line_map = cv2.morphologyEx(dark_foreground, cv2.MORPH_OPEN, kernel)
    line_map = cv2.morphologyEx(line_map, cv2.MORPH_CLOSE, kernel)
    line_map = suppress_border_components(line_map)
    line_map = filter_blob_components(line_map, min_area_px=30, min_short_side_px=6)
    return normalized, line_map


def preprocess_outline_edges(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
    blur_kernel: int = 3,
    canny_low: int = 100,
    canny_high: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract clean outline edges for line-art style inputs.

    This is close to the older "classic mode" behavior, with one extra cleanup
    pass to drop tiny text/noise fragments.
    """
    gray = load_image_grayscale(image_path)
    gray = resize_with_aspect_pad(gray, target_size=target_size, pad_value=255)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(blur, canny_low, canny_high)
    edges = suppress_border_components(
        edges,
        min_area_ratio=0.002,
        min_span_ratio=0.25,
    )
    edges = filter_outline_components(edges)
    return gray, edges


def preprocess_color_regions(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
    saturation_threshold: int = 18,
    white_distance_threshold: float = 20.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Detect colorful filled regions that should be outlined.

    This keeps colorful icons/logos in the classical path without introducing a
    large decision tree.
    """
    image_bgr = load_image_bgr(image_path)
    resized_bgr = resize_with_aspect_pad(
        image_bgr,
        target_size=target_size,
        pad_value=(255, 255, 255),
        interpolation=cv2.INTER_AREA,
    )
    hsv = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2HSV)
    white_distance = np.linalg.norm(resized_bgr.astype(np.int16) - 255, axis=2)

    region_mask = (
        (hsv[:, :, 1] >= saturation_threshold)
        & (white_distance >= white_distance_threshold)
    ).astype(np.uint8) * 255
    kernel = np.ones((3, 3), np.uint8)
    region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_OPEN, kernel)
    region_mask = suppress_border_components(region_mask)
    region_mask = filter_blob_components(region_mask, min_area_px=60, min_short_side_px=8)
    return resized_bgr, region_mask


def binary_overlap_metrics(
    prediction: np.ndarray,
    reference: np.ndarray,
) -> dict[str, float]:
    """Measure how much one binary mask agrees with another."""
    pred = prediction > 0
    ref = reference > 0
    pred_pixels = int(pred.sum())
    ref_pixels = int(ref.sum())
    intersection = int((pred & ref).sum())

    precision = float(intersection) / float(pred_pixels) if pred_pixels else 0.0
    recall = float(intersection) / float(ref_pixels) if ref_pixels else 0.0
    dice = (
        (2.0 * float(intersection)) / float(pred_pixels + ref_pixels)
        if (pred_pixels + ref_pixels)
        else 0.0
    )
    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
    }


def is_color_region_artwork(
    color_region_mask: np.ndarray,
    min_component_area_px: int = 60,
    min_components: int = 6,
    min_total_area_ratio: float = 0.08,
) -> bool:
    """Detect many filled colored regions such as patterned icons/logos."""
    binary = (color_region_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return False

    total_pixels = float(binary.shape[0] * binary.shape[1])
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    significant_components = 0
    significant_area = 0

    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_component_area_px:
            continue
        significant_components += 1
        significant_area += area

    return (
        significant_components >= min_components
        and (float(significant_area) / total_pixels) >= min_total_area_ratio
    )


def has_large_dense_component(
    binary_line_map: np.ndarray,
    min_component_area_ratio: float = 0.08,
    min_component_extent: float = 0.75,
) -> bool:
    """
    Detect large filled blobs that should be outlined instead of skeletonized.
    """
    binary = (binary_line_map > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return False

    total_pixels = float(binary.shape[0] * binary.shape[1])
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        bbox_area = int(width * height)
        if bbox_area <= 0:
            continue

        area_ratio = float(area) / total_pixels
        extent = float(area) / float(bbox_area)
        if area_ratio >= min_component_area_ratio and extent >= min_component_extent:
            return True

    return False


def has_large_elongated_component(
    binary_line_map: np.ndarray,
    min_component_area_ratio: float = 0.01,
    min_component_extent: float = 0.6,
    min_aspect_ratio: float = 6.0,
) -> bool:
    """
    Detect thick bar-like components that should collapse to one centerline.

    This separates "long stroke" inputs from truly filled logos/shapes.
    """
    binary = (binary_line_map > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return False

    total_pixels = float(binary.shape[0] * binary.shape[1])
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        short_side = float(min(width, height))
        long_side = float(max(width, height))
        bbox_area = int(width * height)
        if short_side <= 0.0 or bbox_area <= 0:
            continue

        area_ratio = float(area) / total_pixels
        extent = float(area) / float(bbox_area)
        aspect_ratio = long_side / short_side
        if (
            area_ratio >= min_component_area_ratio
            and extent >= min_component_extent
            and aspect_ratio >= min_aspect_ratio
        ):
            return True

    return False


def has_large_bbox_component(
    binary_line_map: np.ndarray,
    min_bbox_width_ratio: float = 0.65,
    min_bbox_height_ratio: float = 0.65,
) -> bool:
    """Detect coarse masks that span most of the page with only a few paths."""
    binary = (binary_line_map > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return False

    height, width = binary.shape[:2]
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for index in range(1, component_count):
        _x, _y, component_width, component_height, area = stats[index]
        if int(area) <= 0:
            continue
        width_ratio = float(component_width) / float(width)
        height_ratio = float(component_height) / float(height)
        if width_ratio >= min_bbox_width_ratio and height_ratio >= min_bbox_height_ratio:
            return True

    return False


def should_trace_ml_mask_as_filled_region(
    binary_mask: np.ndarray,
    min_component_area_ratio: float = 0.018,
    min_component_extent: float = 0.24,
) -> bool:
    """
    Detect ML masks that represent filled regions, not line strokes.

    Filled-region prediction is much easier for simple colored symbols than
    predicting a perfectly closed 1-pixel contour. When the model outputs a
    filled blob, we should trace its boundary instead of skeletonizing it.
    """
    binary = (binary_mask > 0).astype(np.uint8)
    if cv2.countNonZero(binary) == 0:
        return False

    if has_large_elongated_component(binary_mask):
        return False

    total_pixels = float(binary.shape[0] * binary.shape[1])
    component_count, _labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    for index in range(1, component_count):
        _x, _y, width, height, area = stats[index]
        bbox_area = int(width * height)
        if bbox_area <= 0:
            continue
        area_ratio = float(area) / total_pixels
        extent = float(area) / float(bbox_area)
        if area_ratio >= min_component_area_ratio and extent >= min_component_extent:
            return True

    return False


def _classical_result(
    *,
    paths: list[Path],
    trace_map: np.ndarray,
    strategy: str,
    return_info: bool,
) -> tuple[list[Path], np.ndarray] | tuple[list[Path], np.ndarray, dict]:
    info = {
        "requested_mode": "classical",
        "effective_mode": "classical",
        "extractor": "classical",
        "strategy": strategy,
        "raw_path_count": len(paths),
    }
    if return_info:
        return paths, trace_map, info
    return paths, trace_map


def get_drawing_paths_classical(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
    return_info: bool = False,
) -> tuple[list[Path], np.ndarray] | tuple[list[Path], np.ndarray, dict]:
    """
    Classical path extraction.

    The stages are applied in this order:
    color regions -> filled boundaries -> outline edges.
    """
    _color_bgr, color_region_mask = preprocess_color_regions(
        image_path=image_path,
        target_size=target_size,
    )
    color_region_paths = extract_filled_region_paths(color_region_mask)
    color_region_ratio = float(cv2.countNonZero(color_region_mask)) / float(color_region_mask.size)
    if color_region_paths and (
        color_region_ratio >= 0.025
        or is_color_region_artwork(color_region_mask)
    ):
        trace_map = render_trace_map(color_region_paths, color_region_mask.shape)
        return _classical_result(
            paths=color_region_paths,
            trace_map=trace_map,
            strategy="color_region_boundaries",
            return_info=return_info,
        )

    _normalized_gray, foreground_mask = preprocess_foreground_mask(
        image_path=image_path,
        target_size=target_size,
    )
    axis_paths, axis_trace_map, axis_coverage = extract_axis_aligned_stroke_paths(foreground_mask)
    if axis_paths and axis_coverage >= 0.72:
        return _classical_result(
            paths=axis_paths,
            trace_map=axis_trace_map,
            strategy="foreground_axis_strokes",
            return_info=return_info,
        )

    if has_large_elongated_component(foreground_mask):
        paths, trace_map = extract_paths_from_line_map(foreground_mask)
        return _classical_result(
            paths=paths,
            trace_map=trace_map,
            strategy="foreground_line_map",
            return_info=return_info,
        )

    if has_large_dense_component(
        foreground_mask,
        min_component_area_ratio=0.01,
        min_component_extent=0.65,
    ):
        paths = extract_filled_region_paths(foreground_mask)
        trace_map = render_trace_map(paths, foreground_mask.shape)
        return _classical_result(
            paths=paths,
            trace_map=trace_map,
            strategy="filled_region_boundaries",
            return_info=return_info,
        )

    _gray, outline_edges = preprocess_outline_edges(
        image_path=image_path,
        target_size=target_size,
    )
    paths = extract_edge_contour_paths(outline_edges)
    if not paths and cv2.countNonZero(outline_edges) > 0:
        paths, trace_map = extract_paths_from_line_map(outline_edges)
        return _classical_result(
            paths=paths,
            trace_map=trace_map,
            strategy="outline_centerlines",
            return_info=return_info,
        )
    if paths:
        trace_map = render_trace_map(paths, outline_edges.shape)
        return _classical_result(
            paths=paths,
            trace_map=trace_map,
            strategy="outline_edge_contours",
            return_info=return_info,
        )

    # If the edge map is empty, try to recover paths from the foreground mask.
    fallback_paths, trace_map = extract_paths_from_line_map(foreground_mask)
    return _classical_result(
        paths=fallback_paths,
        trace_map=trace_map,
        strategy="foreground_line_map",
        return_info=return_info,
    )


def get_ml_fallback_reason(
    clean_mask: np.ndarray,
    path_count: int,
    color_region_artwork: bool,
) -> str | None:
    """
    Decide whether ML mode should fall back to classical extraction.

    The policy stays intentionally narrow:
    only fall back when the ML output is empty or clearly degenerate.
    """
    foreground_pixels = int(cv2.countNonZero(clean_mask))
    if foreground_pixels == 0 or path_count == 0:
        return "empty_ml_mask"

    if color_region_artwork:
        return "color_region_classical_fallback"

    foreground_ratio = foreground_pixels / float(clean_mask.size)
    large_bbox = has_large_bbox_component(clean_mask)

    if path_count <= 4:
        if foreground_ratio < 0.002:
            return "too_few_ml_paths"
        if large_bbox:
            return "large_blob_ml_mask"

    return None


def get_drawing_paths_ml(
    image_path: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
    return_info: bool = False,
) -> tuple[list[Path], np.ndarray] | tuple[list[Path], np.ndarray, dict]:
    """Run the ML path extractor and convert its mask into drawable paths."""
    from src.ml_line_model import predict_line_mask

    mask = predict_line_mask(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device_str=device,
    )

    clean_mask = clean_binary_line_map(mask)
    if should_trace_ml_mask_as_filled_region(clean_mask):
        paths = extract_filled_region_paths(clean_mask)
        trace_map = render_trace_map(paths, clean_mask.shape)
        ml_strategy = "predicted_filled_boundaries"
    else:
        axis_paths, axis_trace_map, axis_coverage = extract_axis_aligned_stroke_paths(clean_mask)
        if axis_paths and axis_coverage >= 0.72:
            paths = axis_paths
            trace_map = axis_trace_map
            ml_strategy = "predicted_axis_strokes"
        else:
            paths, trace_map = extract_paths_from_line_map(clean_mask)
            ml_strategy = "predicted_line_mask_paths"

    _color_bgr, color_region_mask = preprocess_color_regions(
        image_path=image_path,
        target_size=(clean_mask.shape[1], clean_mask.shape[0]),
    )
    color_region_artwork = is_color_region_artwork(color_region_mask)
    fallback_reason = get_ml_fallback_reason(
        clean_mask=clean_mask,
        path_count=len(paths),
        color_region_artwork=color_region_artwork,
    )
    if fallback_reason is not None:
        classical_paths, classical_trace, classical_info = get_drawing_paths_classical(
            image_path=image_path,
            target_size=(clean_mask.shape[1], clean_mask.shape[0]),
            return_info=True,
        )
        info = {
            "requested_mode": "ml",
            "effective_mode": "classical",
            "extractor": "classical_fallback",
            "fallback_reason": fallback_reason,
            "checkpoint_path": checkpoint_path,
        "threshold": threshold,
        "mask_pixels": int(cv2.countNonZero(mask)),
        "clean_mask_pixels": int(cv2.countNonZero(clean_mask)),
        "ml_strategy": ml_strategy,
        "ml_raw_path_count": len(paths),
        "classical_strategy": classical_info["strategy"],
        "raw_path_count": len(classical_paths),
        }
        if return_info:
            return classical_paths, classical_trace, info
        return classical_paths, classical_trace

    _reference_gray, reference_mask = preprocess_foreground_mask(
        image_path=image_path,
        target_size=(clean_mask.shape[1], clean_mask.shape[0]),
    )
    reference_paths, _reference_trace = extract_paths_from_line_map(reference_mask)
    reference_overlap = binary_overlap_metrics(clean_mask, reference_mask)
    info = {
        "requested_mode": "ml",
        "effective_mode": "ml",
        "extractor": "ml",
        "fallback_reason": None,
        "checkpoint_path": checkpoint_path,
        "threshold": threshold,
        "mask_pixels": int(cv2.countNonZero(mask)),
        "clean_mask_pixels": int(cv2.countNonZero(clean_mask)),
        "ml_strategy": ml_strategy,
        "raw_path_count": len(paths),
        "reference_path_count": len(reference_paths),
        "reference_overlap_precision": round(reference_overlap["precision"], 4),
        "reference_overlap_recall": round(reference_overlap["recall"], 4),
        "reference_overlap_dice": round(reference_overlap["dice"], 4),
    }
    if return_info:
        return paths, trace_map, info
    return paths, trace_map


def get_drawing_paths(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
) -> tuple[list[Path], np.ndarray]:
    """Backward-compatible alias to classical mode."""
    return get_drawing_paths_classical(image_path=image_path, target_size=target_size)


__all__ = [
    "Path",
    "Point",
    "binary_overlap_metrics",
    "extract_paths_from_line_map",
    "get_drawing_paths",
    "get_drawing_paths_classical",
    "get_drawing_paths_ml",
    "get_ml_fallback_reason",
    "has_large_bbox_component",
    "has_large_dense_component",
    "has_large_elongated_component",
    "is_color_region_artwork",
    "preprocess_color_regions",
    "preprocess_foreground_mask",
    "preprocess_outline_edges",
    "should_trace_ml_mask_as_filled_region",
    "suppress_border_components",
]
