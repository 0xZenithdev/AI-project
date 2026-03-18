"""
Vision module (v2): turn an input image into drawable vector-like paths.

This file supports 2 modes:
- classical: Canny-based edge extraction (no training required)
- ml: learned line-mask prediction from a trained model checkpoint
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np

# Type aliases keep function signatures readable.
Point = Tuple[int, int]  # (x, y) in image pixel coordinates.
Path = List[Point]       # One continuous stroke/path.


def preprocess_line_art(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
    blur_kernel: int = 3,
    canny_low: int = 80,
    canny_high: int = 160,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load an image and convert it into an edge map with classical CV.

    Returns:
    - resized grayscale image
    - binary edge image (white=edge, black=background)
    """
    img_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_resized = cv2.resize(img_gray, target_size, interpolation=cv2.INTER_AREA)
    img_blur = cv2.GaussianBlur(img_resized, (blur_kernel, blur_kernel), 0)
    edges = cv2.Canny(img_blur, canny_low, canny_high)

    return img_resized, edges


def extract_contours(
    edges: np.ndarray,
    min_points: int = 6,
    approx_epsilon_ratio: float = 0.002,
) -> list[Path]:
    """
    Convert binary line map (edge/mask) into ordered contour paths.
    """
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths: list[Path] = []

    for contour in contours:
        if len(contour) < min_points:
            continue

        epsilon = approx_epsilon_ratio * cv2.arcLength(contour, closed=False)
        simplified = cv2.approxPolyDP(contour, epsilon, closed=False)

        if len(simplified) < min_points:
            continue

        path: Path = [(int(p[0][0]), int(p[0][1])) for p in simplified]
        paths.append(path)

    return paths


def get_drawing_paths_classical(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
) -> tuple[list[Path], np.ndarray]:
    """Classical CV path extraction: image -> Canny edges -> contours."""
    _, edges = preprocess_line_art(image_path=image_path, target_size=target_size)
    paths = extract_contours(edges)
    return paths, edges


def get_drawing_paths_ml(
    image_path: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    device: str = "cpu",
) -> tuple[list[Path], np.ndarray]:
    """
    ML path extraction: image -> model-predicted line mask -> contours.

    Requires a trained checkpoint from src/train.py.
    """
    # Lazy import so classical mode does not require torch installation.
    from src.ml_line_model import predict_line_mask

    mask = predict_line_mask(
        image_path=image_path,
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device_str=device,
    )

    # Optional denoise for cleaner contour extraction.
    kernel = np.ones((3, 3), np.uint8)
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    paths = extract_contours(clean_mask)
    return paths, clean_mask


def get_drawing_paths(
    image_path: str,
    target_size: tuple[int, int] = (210, 297),
) -> tuple[list[Path], np.ndarray]:
    """
    Backward-compatible alias to classical mode.
    Existing code calling get_drawing_paths(...) will keep working.
    """
    return get_drawing_paths_classical(image_path=image_path, target_size=target_size)
