"""
Generate a synthetic geometric-logo dataset for ML fine-tuning.

Why this exists:
- the project logos are dominated by geometric shapes
- public logo datasets often provide detection or filled segmentation masks
- this repo needs paired image -> contour-mask supervision

The generator creates:
- grayscale/BGR logo-like images with simple lighting/noise
- binary contour masks derived from filled geometric emblems
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from src.ml_dataset_utils import convert_binary_mask_to_contour_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a synthetic geometric-logo dataset.")
    parser.add_argument("--images-dir", type=str, default="dataset/geometric_logo_images", help="Output image directory")
    parser.add_argument("--masks-dir", type=str, default="dataset/geometric_logo_masks", help="Output mask directory")
    parser.add_argument(
        "--report-path",
        type=str,
        default="output/geometric_logo_dataset/generation_report.json",
        help="JSON report path",
    )
    parser.add_argument("--count", type=int, default=600, help="Number of samples to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--min-size", type=int, default=180, help="Minimum image side length")
    parser.add_argument("--max-size", type=int, default=320, help="Maximum image side length")
    parser.add_argument(
        "--stem-prefix",
        type=str,
        default="geom_logo",
        help="Filename prefix used for generated stems",
    )
    parser.add_argument(
        "--style",
        choices=["emblem", "outline"],
        default="emblem",
        help="Synthetic logo style: filled emblems or thin outline symbols",
    )
    parser.add_argument("--contour-thickness", type=int, default=1, help="Contour thickness for target masks")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser.parse_args()


def regular_polygon(center: tuple[int, int], radius: int, sides: int, rotation_deg: float) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=sides, endpoint=False)
    rotation_rad = np.deg2rad(rotation_deg)
    points = []
    for angle in angles:
        x = int(round(center[0] + radius * np.cos(angle + rotation_rad)))
        y = int(round(center[1] + radius * np.sin(angle + rotation_rad)))
        points.append([x, y])
    return np.array(points, dtype=np.int32)


def star_polygon(
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int,
    points: int,
    rotation_deg: float,
) -> np.ndarray:
    vertices = []
    rotation_rad = np.deg2rad(rotation_deg)
    for index in range(points * 2):
        angle = rotation_rad + (np.pi * index / points)
        radius = outer_radius if index % 2 == 0 else inner_radius
        x = int(round(center[0] + radius * np.cos(angle)))
        y = int(round(center[1] + radius * np.sin(angle)))
        vertices.append([x, y])
    return np.array(vertices, dtype=np.int32)


def draw_filled_polygon(mask: np.ndarray, points: np.ndarray, color: int = 255) -> None:
    cv2.fillPoly(mask, [points], color)


def draw_rotated_rect(mask: np.ndarray, center: tuple[int, int], size: tuple[int, int], angle_deg: float, color: int = 255) -> None:
    rect = (tuple(float(v) for v in center), tuple(float(v) for v in size), float(angle_deg))
    box = cv2.boxPoints(rect).astype(np.int32)
    cv2.fillPoly(mask, [box], color)


def carve_rotated_rect(mask: np.ndarray, center: tuple[int, int], size: tuple[int, int], angle_deg: float) -> None:
    draw_rotated_rect(mask, center=center, size=size, angle_deg=angle_deg, color=0)


def draw_ring_with_bar(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    outer_axes = (int(70 * scale), int(88 * scale))
    inner_axes = (int(42 * scale), int(58 * scale))
    angle = float(rng.uniform(-20, 20))
    cv2.ellipse(mask, center, outer_axes, angle, 0, 360, 255, -1)
    cv2.ellipse(mask, center, inner_axes, angle, 0, 360, 0, -1)
    draw_rotated_rect(
        mask,
        center=center,
        size=(int(120 * scale), int(30 * scale)),
        angle_deg=angle + float(rng.uniform(-20, 20)),
    )


def draw_hex_badge(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    outer = regular_polygon(center, int(88 * scale), sides=6, rotation_deg=float(rng.uniform(0, 30)))
    inner = regular_polygon(center, int(58 * scale), sides=6, rotation_deg=float(rng.uniform(0, 30)))
    draw_filled_polygon(mask, outer, color=255)
    draw_filled_polygon(mask, inner, color=0)
    draw_rotated_rect(
        mask,
        center=center,
        size=(int(26 * scale), int(120 * scale)),
        angle_deg=float(rng.uniform(-15, 15)),
        color=255,
    )
    draw_rotated_rect(
        mask,
        center=center,
        size=(int(120 * scale), int(24 * scale)),
        angle_deg=float(rng.uniform(-15, 15)),
        color=255,
    )


def draw_diamond_emblem(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    outer = regular_polygon(center, int(88 * scale), sides=4, rotation_deg=45 + float(rng.uniform(-10, 10)))
    inner = regular_polygon(center, int(54 * scale), sides=4, rotation_deg=45 + float(rng.uniform(-10, 10)))
    draw_filled_polygon(mask, outer, color=255)
    draw_filled_polygon(mask, inner, color=0)
    offset = int(24 * scale)
    draw_rotated_rect(mask, (center[0] - offset, center[1]), (int(26 * scale), int(100 * scale)), 45, color=255)
    draw_rotated_rect(mask, (center[0] + offset, center[1]), (int(26 * scale), int(100 * scale)), 45, color=255)


def draw_triangle_orbit(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    triangle = regular_polygon(center, int(92 * scale), sides=3, rotation_deg=-90 + float(rng.uniform(-10, 10)))
    inner_triangle = regular_polygon(center, int(54 * scale), sides=3, rotation_deg=-90 + float(rng.uniform(-10, 10)))
    draw_filled_polygon(mask, triangle, color=255)
    draw_filled_polygon(mask, inner_triangle, color=0)
    orbit_center = (center[0], center[1] + int(6 * scale))
    cv2.ellipse(mask, orbit_center, (int(78 * scale), int(44 * scale)), float(rng.uniform(-20, 20)), 0, 360, 255, int(max(6, round(14 * scale))))


def draw_stripe_circle(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    cv2.circle(mask, center, int(88 * scale), 255, -1)
    cv2.circle(mask, center, int(58 * scale), 0, -1)
    stripe_angle = float(rng.uniform(-30, 30))
    for offset in (-28, 0, 28):
        stripe_center = (center[0] + int(offset * scale), center[1])
        draw_rotated_rect(
            mask,
            center=stripe_center,
            size=(int(22 * scale), int(130 * scale)),
            angle_deg=stripe_angle,
            color=255,
        )


def draw_split_square(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    outer = regular_polygon(center, int(92 * scale), sides=4, rotation_deg=float(rng.uniform(-5, 5)))
    draw_filled_polygon(mask, outer, color=255)
    carve_rotated_rect(
        mask,
        center=(center[0], center[1] - int(24 * scale)),
        size=(int(120 * scale), int(22 * scale)),
        angle_deg=float(rng.uniform(-25, 25)),
    )
    carve_rotated_rect(
        mask,
        center=(center[0], center[1] + int(24 * scale)),
        size=(int(120 * scale), int(22 * scale)),
        angle_deg=float(rng.uniform(-25, 25)),
    )
    draw_rotated_rect(
        mask,
        center=center,
        size=(int(28 * scale), int(132 * scale)),
        angle_deg=float(rng.uniform(-10, 10)),
        color=255,
    )


def draw_star_badge(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    star = star_polygon(
        center=center,
        outer_radius=int(92 * scale),
        inner_radius=int(42 * scale),
        points=5,
        rotation_deg=-90 + float(rng.uniform(-8, 8)),
    )
    draw_filled_polygon(mask, star, color=255)


def draw_arrow_emblem(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    width = int(95 * scale)
    height = int(80 * scale)
    arrow = np.array(
        [
            (center[0] - width, center[1] - int(0.28 * height)),
            (center[0] - int(0.08 * width), center[1] - int(0.28 * height)),
            (center[0] - int(0.08 * width), center[1] - height),
            (center[0] + width, center[1]),
            (center[0] - int(0.08 * width), center[1] + height),
            (center[0] - int(0.08 * width), center[1] + int(0.28 * height)),
            (center[0] - width, center[1] + int(0.28 * height)),
        ],
        dtype=np.int32,
    )
    draw_filled_polygon(mask, arrow, color=255)


def draw_shield_emblem(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    shield = np.array(
        [
            (center[0], center[1] - int(100 * scale)),
            (center[0] + int(82 * scale), center[1] - int(58 * scale)),
            (center[0] + int(66 * scale), center[1] + int(72 * scale)),
            (center[0], center[1] + int(136 * scale)),
            (center[0] - int(66 * scale), center[1] + int(72 * scale)),
            (center[0] - int(82 * scale), center[1] - int(58 * scale)),
        ],
        dtype=np.int32,
    )
    draw_filled_polygon(mask, shield, color=255)


def draw_lightning_emblem(mask: np.ndarray, center: tuple[int, int], scale: float, rng: np.random.Generator) -> None:
    bolt = np.array(
        [
            (center[0] + int(22 * scale), center[1] - int(108 * scale)),
            (center[0] - int(44 * scale), center[1] - int(8 * scale)),
            (center[0], center[1] - int(8 * scale)),
            (center[0] - int(22 * scale), center[1] + int(110 * scale)),
            (center[0] + int(52 * scale), center[1] + int(12 * scale)),
            (center[0] + int(6 * scale), center[1] + int(12 * scale)),
        ],
        dtype=np.int32,
    )
    draw_filled_polygon(mask, bolt, color=255)


MOTIF_BUILDERS = (
    draw_ring_with_bar,
    draw_hex_badge,
    draw_diamond_emblem,
    draw_triangle_orbit,
    draw_stripe_circle,
    draw_split_square,
    draw_star_badge,
    draw_arrow_emblem,
    draw_shield_emblem,
    draw_lightning_emblem,
)


def create_background(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    base = np.full((height, width, 3), int(rng.integers(228, 250)), dtype=np.uint8)
    x_gradient = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y_gradient = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    shading = (x_gradient * rng.uniform(-18, 18) + y_gradient * rng.uniform(-18, 18)).astype(np.float32)
    for channel in range(3):
        base[:, :, channel] = np.clip(base[:, :, channel].astype(np.float32) + shading, 0, 255).astype(np.uint8)
    noise = rng.normal(0, 5, size=base.shape).astype(np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def create_logo_fill(height: int, width: int, rng: np.random.Generator) -> np.ndarray:
    fill = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2 + int(rng.integers(-12, 13)), height // 2 + int(rng.integers(-12, 13)))
    scale = min(height, width) / 220.0
    motif = MOTIF_BUILDERS[int(rng.integers(0, len(MOTIF_BUILDERS)))]
    motif(fill, center=center, scale=scale, rng=rng)

    if int(rng.integers(0, 100)) < 35:
        secondary = MOTIF_BUILDERS[int(rng.integers(0, len(MOTIF_BUILDERS)))]
        secondary_scale = scale * float(rng.uniform(0.45, 0.7))
        secondary_center = (
            center[0] + int(rng.integers(-20, 21)),
            center[1] + int(rng.integers(-20, 21)),
        )
        secondary(fill, center=secondary_center, scale=secondary_scale, rng=rng)

    kernel = np.ones((3, 3), dtype=np.uint8)
    fill = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, kernel, iterations=1)
    return fill


def render_input_image(fill_mask: np.ndarray, contour_mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = fill_mask.shape
    image = create_background(height, width, rng)

    fill_color = int(rng.integers(40, 120))
    edge_color = max(0, fill_color - int(rng.integers(18, 42)))
    highlight_color = min(255, fill_color + int(rng.integers(10, 32)))

    logo_noise = rng.normal(0, 10, size=(height, width)).astype(np.int16)
    fill_region = fill_mask > 127
    for channel in range(3):
        channel_values = np.full((height, width), fill_color, dtype=np.int16)
        if channel == 0:
            channel_values += int(rng.integers(-5, 6))
        if channel == 2:
            channel_values += int(rng.integers(-5, 6))
        channel_values += logo_noise
        image[:, :, channel][fill_region] = np.clip(channel_values[fill_region], 0, 255).astype(np.uint8)

    edge_region = contour_mask > 127
    image[edge_region] = (edge_color, edge_color, edge_color)

    dilated = cv2.dilate(contour_mask, np.ones((3, 3), dtype=np.uint8), iterations=1)
    highlight_region = np.logical_and(dilated > 127, fill_mask > 127)
    highlight_region = np.logical_and(highlight_region, contour_mask == 0)
    image[highlight_region] = (highlight_color, highlight_color, highlight_color)

    blur_kernel = 3 if int(rng.integers(0, 100)) < 70 else 5
    image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigmaX=0)
    image_noise = rng.normal(0, 4, size=image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + image_noise, 0, 255).astype(np.uint8)
    return image


def draw_outline_symbol_pack(mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = mask.shape
    center = (width // 2 + int(rng.integers(-10, 11)), height // 2 - int(rng.integers(15, 35)))
    thickness = int(rng.integers(4, 9))

    triangle = regular_polygon(
        (center[0] - int(width * 0.12), center[1] - int(height * 0.08)),
        int(min(height, width) * 0.16),
        sides=3,
        rotation_deg=-90 + float(rng.uniform(-6, 6)),
    )
    square_size = int(min(height, width) * 0.17)
    square_center = (center[0] + int(width * 0.12), center[1] - int(height * 0.02))
    square_top_left = (square_center[0] - square_size // 2, square_center[1] - square_size // 2)
    square_bottom_right = (square_center[0] + square_size // 2, square_center[1] + square_size // 2)
    circle_center = (center[0], center[1] + int(height * 0.10))
    circle_radius = int(min(height, width) * 0.13)

    cv2.polylines(mask, [triangle], isClosed=True, color=255, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.rectangle(mask, square_top_left, square_bottom_right, 255, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.circle(mask, circle_center, circle_radius, 255, thickness=thickness, lineType=cv2.LINE_AA)

    if int(rng.integers(0, 100)) < 45:
        extra_circle_center = (center[0] - int(width * 0.17), center[1] + int(height * 0.02))
        cv2.circle(mask, extra_circle_center, int(circle_radius * 0.65), 255, thickness=max(3, thickness - 1), lineType=cv2.LINE_AA)

    if int(rng.integers(0, 100)) < 60:
        words = ["LINE", "FORM", "ANGLE", "SHADOW", "SHAPE", "MARK", "GEO", "EDGE"]
        word_count = 2 if int(rng.integers(0, 100)) < 55 else 1
        text = " ".join(words[int(rng.integers(0, len(words)))] for _ in range(word_count))
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = float(rng.uniform(0.75, 1.15))
        text_thickness = max(1, thickness // 2)
        text_size, baseline = cv2.getTextSize(text, font, font_scale, text_thickness)
        text_x = max(10, (width - text_size[0]) // 2 + int(rng.integers(-8, 9)))
        text_y = min(height - 12, center[1] + int(height * 0.33) + baseline)
        cv2.putText(mask, text, (text_x, text_y), font, font_scale, 255, thickness=text_thickness, lineType=cv2.LINE_AA)


def draw_outline_plus(mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = mask.shape
    center = (width // 2 + int(rng.integers(-8, 9)), height // 2 + int(rng.integers(-8, 9)))
    thickness = int(rng.integers(4, 8))
    half_vertical = int(min(height, width) * float(rng.uniform(0.24, 0.34)))
    half_horizontal = int(min(height, width) * float(rng.uniform(0.24, 0.34)))
    cv2.line(mask, (center[0], center[1] - half_vertical), (center[0], center[1] + half_vertical), 255, thickness=thickness, lineType=cv2.LINE_AA)
    cv2.line(mask, (center[0] - half_horizontal, center[1]), (center[0] + half_horizontal, center[1]), 255, thickness=thickness, lineType=cv2.LINE_AA)


def draw_outline_star(mask: np.ndarray, rng: np.random.Generator) -> None:
    height, width = mask.shape
    center = (width // 2 + int(rng.integers(-8, 9)), height // 2 + int(rng.integers(-12, 13)))
    thickness = int(rng.integers(4, 7))
    star = star_polygon(
        center=center,
        outer_radius=int(min(height, width) * 0.28),
        inner_radius=int(min(height, width) * 0.12),
        points=5,
        rotation_deg=-90 + float(rng.uniform(-8, 8)),
    )
    cv2.polylines(mask, [star], isClosed=True, color=255, thickness=thickness, lineType=cv2.LINE_AA)


OUTLINE_BUILDERS = (
    draw_outline_symbol_pack,
    draw_outline_plus,
    draw_outline_star,
)


def draw_outline_shapes(mask: np.ndarray, rng: np.random.Generator) -> None:
    builder = OUTLINE_BUILDERS[int(rng.integers(0, len(OUTLINE_BUILDERS)))]
    builder(mask, rng=rng)


def render_outline_input_image(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    height, width = mask.shape
    image = create_background(height, width, rng)
    line_region = mask > 127

    shadow = np.zeros_like(mask)
    shadow_shift = (int(rng.integers(6, 15)), int(rng.integers(6, 15)))
    transform = np.float32([[1, 0, shadow_shift[0]], [0, 1, shadow_shift[1]]])
    shadow = cv2.warpAffine(mask, transform, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)
    shadow = cv2.GaussianBlur(shadow, (7, 7), sigmaX=0)
    shadow_region = shadow > 20
    shadow_strength = int(rng.integers(155, 210))
    image[shadow_region] = (shadow_strength, shadow_strength, shadow_strength)

    line_color = int(rng.integers(18, 48))
    image[line_region] = (line_color, line_color, line_color)

    blur_kernel = 3 if int(rng.integers(0, 100)) < 75 else 5
    image = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), sigmaX=0)
    image_noise = rng.normal(0, 3, size=image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + image_noise, 0, 255).astype(np.uint8)
    return image


def generate_sample(
    *,
    rng: np.random.Generator,
    contour_thickness: int,
    min_size: int,
    max_size: int,
    style: str,
) -> tuple[np.ndarray, np.ndarray]:
    side = int(rng.integers(min_size, max_size + 1))
    height = side + int(rng.integers(-20, 21))
    width = side + int(rng.integers(-20, 21))
    height = max(min_size, height)
    width = max(min_size, width)

    if style == "outline":
        contour_mask = np.zeros((height, width), dtype=np.uint8)
        draw_outline_shapes(contour_mask, rng=rng)
        contour_mask = (contour_mask > 127).astype(np.uint8) * 255
        image = render_outline_input_image(mask=contour_mask, rng=rng)
    else:
        fill_mask = create_logo_fill(height=height, width=width, rng=rng)
        contour_mask = convert_binary_mask_to_contour_mask(fill_mask, contour_thickness=contour_thickness)
        image = render_input_image(fill_mask=fill_mask, contour_mask=contour_mask, rng=rng)
    return image, contour_mask


def generate_dataset(args: argparse.Namespace) -> dict:
    images_dir = Path(args.images_dir)
    masks_dir = Path(args.masks_dir)
    report_path = Path(args.report_path)
    stem_prefix = str(getattr(args, "stem_prefix", "geom_logo"))

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(int(args.seed))
    generated = 0
    skipped_existing = 0
    rows: list[dict] = []

    for index in range(int(args.count)):
        stem = f"{stem_prefix}_{index:05d}"
        image_path = images_dir / f"{stem}.png"
        mask_path = masks_dir / f"{stem}.png"

        if image_path.exists() and mask_path.exists() and not args.overwrite:
            skipped_existing += 1
            rows.append({"stem": stem, "action": "skipped_existing"})
            continue

        image, contour_mask = generate_sample(
            rng=rng,
            contour_thickness=int(args.contour_thickness),
            min_size=int(args.min_size),
            max_size=int(args.max_size),
            style=args.style,
        )
        cv2.imwrite(str(image_path), image)
        cv2.imwrite(str(mask_path), contour_mask)
        generated += 1
        rows.append({"stem": stem, "action": "generated"})

    report = {
        "images_dir": str(images_dir),
        "masks_dir": str(masks_dir),
        "count_requested": int(args.count),
        "generated": generated,
        "skipped_existing": skipped_existing,
        "stem_prefix": stem_prefix,
        "seed": int(args.seed),
        "min_size": int(args.min_size),
        "max_size": int(args.max_size),
        "style": args.style,
        "contour_thickness": int(args.contour_thickness),
        "rows": rows[:200],
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    report = generate_dataset(args)
    print(f"Generated samples: {report['generated']}")
    print(f"Skipped existing: {report['skipped_existing']}")
    print(f"Wrote report to: {args.report_path}")


if __name__ == "__main__":
    main()
