"""Helpers used to load images and resize them without stretching the drawing."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np


def _normalize_background_color(
    channels: int,
    background_bgr: Sequence[int],
) -> np.ndarray:
    if channels == 1:
        return np.array(background_bgr[0], dtype=np.float32)
    if channels == 3:
        return np.array(background_bgr[:3], dtype=np.float32)
    raise ValueError(f"Unsupported channel count: {channels}")


def composite_alpha_onto_background(
    image: np.ndarray,
    background_bgr: Sequence[int] = (255, 255, 255),
) -> np.ndarray:
    """Flatten an image with alpha onto a solid background."""
    if image.ndim == 2:
        return image

    if image.shape[2] == 3:
        return image

    if image.shape[2] != 4:
        raise ValueError(f"Unsupported image shape for alpha compositing: {image.shape}")

    alpha = image[:, :, 3:4].astype(np.float32) / 255.0
    foreground = image[:, :, :3].astype(np.float32)
    background = _normalize_background_color(3, background_bgr).reshape(1, 1, 3)
    blended = (foreground * alpha) + (background * (1.0 - alpha))
    return np.clip(blended, 0, 255).astype(np.uint8)


def load_image_bgr(
    image_path: str | Path,
    background_bgr: Sequence[int] = (255, 255, 255),
) -> np.ndarray:
    """
    Load an image as BGR while flattening transparency onto a background.
    """
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    image = composite_alpha_onto_background(image, background_bgr=background_bgr)
    if image.shape[2] == 3:
        return image

    raise ValueError(f"Unsupported loaded image shape: {image.shape}")


def load_image_rgb(
    image_path: str | Path,
    background_bgr: Sequence[int] = (255, 255, 255),
) -> np.ndarray:
    """
    Load an image as RGB while flattening transparency onto a background.
    """
    image_bgr = load_image_bgr(image_path=image_path, background_bgr=background_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)


def load_image_grayscale(
    image_path: str | Path,
    background_bgr: Sequence[int] = (255, 255, 255),
) -> np.ndarray:
    """
    Load an image as grayscale while flattening transparency onto a background.
    """
    image_bgr = load_image_bgr(image_path=image_path, background_bgr=background_bgr)
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def resize_with_aspect_pad(
    image: np.ndarray,
    target_size: tuple[int, int],
    pad_value: int | Sequence[int] = 255,
    interpolation: int = cv2.INTER_AREA,
) -> np.ndarray:
    """
    Resize an image into a fixed canvas without stretching it.

    The resized content is centered on a padded background so downstream stages
    can keep a stable working size while preserving the original geometry.
    """
    target_w, target_h = target_size
    src_h, src_w = image.shape[:2]
    if src_w <= 0 or src_h <= 0:
        raise ValueError("Input image has invalid dimensions")

    scale = min(target_w / src_w, target_h / src_h)
    resized_w = max(1, int(round(src_w * scale)))
    resized_h = max(1, int(round(src_h * scale)))
    resized = cv2.resize(image, (resized_w, resized_h), interpolation=interpolation)

    if image.ndim == 2:
        canvas = np.full((target_h, target_w), pad_value, dtype=image.dtype)
    else:
        canvas = np.full((target_h, target_w, image.shape[2]), pad_value, dtype=image.dtype)

    x_offset = (target_w - resized_w) // 2
    y_offset = (target_h - resized_h) // 2
    canvas[y_offset:y_offset + resized_h, x_offset:x_offset + resized_w] = resized
    return canvas
