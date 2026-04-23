"""Generate the curated image set used for the demo and report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


CANVAS_SIZE = 960
BACKGROUND = (255, 255, 255)
INK = (0, 0, 0)
OUTLINE_THICKNESS = 16
LINE_THICKNESS = 18
BLUE = (210, 120, 40)
TEAL = (60, 170, 140)
ORANGE = (40, 140, 235)
RED = (70, 70, 220)
GOLD = (40, 180, 230)
GREEN = (80, 170, 90)


def blank_canvas() -> np.ndarray:
    return np.full((CANVAS_SIZE, CANVAS_SIZE, 3), BACKGROUND, dtype=np.uint8)


def regular_polygon_points(
    center: tuple[int, int],
    radius: int,
    sides: int,
    rotation_deg: float = -90.0,
) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, num=sides, endpoint=False)
    rotation_rad = np.deg2rad(rotation_deg)
    points = []
    for angle in angles:
        x = int(round(center[0] + radius * np.cos(angle + rotation_rad)))
        y = int(round(center[1] + radius * np.sin(angle + rotation_rad)))
        points.append((x, y))
    return np.array(points, dtype=np.int32).reshape(-1, 1, 2)


def star_points(
    center: tuple[int, int],
    outer_radius: int,
    inner_radius: int,
    points: int = 5,
    rotation_deg: float = -90.0,
) -> np.ndarray:
    vertices = []
    rotation_rad = np.deg2rad(rotation_deg)
    for index in range(points * 2):
        angle = rotation_rad + (np.pi * index / points)
        radius = outer_radius if index % 2 == 0 else inner_radius
        x = int(round(center[0] + radius * np.cos(angle)))
        y = int(round(center[1] + radius * np.sin(angle)))
        vertices.append((x, y))
    return np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)


def save_case(output_dir: Path, stem: str, title: str, description: str, image: np.ndarray) -> dict:
    path = output_dir / f"{stem}.png"
    cv2.imwrite(str(path), image)
    return {
        "id": stem,
        "title": title,
        "description": description,
        "path": str(path).replace("\\", "/"),
    }


def draw_vertical_line() -> np.ndarray:
    image = blank_canvas()
    cv2.line(image, (CANVAS_SIZE // 2, 120), (CANVAS_SIZE // 2, 840), INK, LINE_THICKNESS)
    return image


def draw_horizontal_line() -> np.ndarray:
    image = blank_canvas()
    cv2.line(image, (120, CANVAS_SIZE // 2), (840, CANVAS_SIZE // 2), INK, LINE_THICKNESS)
    return image


def draw_cross() -> np.ndarray:
    image = blank_canvas()
    cv2.line(image, (CANVAS_SIZE // 2, 140), (CANVAS_SIZE // 2, 820), INK, LINE_THICKNESS)
    cv2.line(image, (140, CANVAS_SIZE // 2), (820, CANVAS_SIZE // 2), INK, LINE_THICKNESS)
    return image


def draw_filled_square() -> np.ndarray:
    image = blank_canvas()
    cv2.rectangle(image, (220, 220), (740, 740), TEAL, cv2.FILLED)
    return image


def draw_filled_triangle() -> np.ndarray:
    image = blank_canvas()
    points = np.array([(480, 170), (760, 760), (200, 760)], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [points], ORANGE)
    return image


def draw_filled_circle() -> np.ndarray:
    image = blank_canvas()
    cv2.circle(image, (480, 480), 260, BLUE, cv2.FILLED)
    return image


def draw_house_badge() -> np.ndarray:
    image = blank_canvas()
    cv2.rectangle(image, (260, 400), (700, 760), GOLD, cv2.FILLED)
    roof = np.array([(220, 420), (480, 180), (740, 420)], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [roof], RED)
    return image


def draw_filled_diamond() -> np.ndarray:
    image = blank_canvas()
    diamond = np.array([(480, 160), (780, 480), (480, 800), (180, 480)], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [diamond], RED)
    return image


def draw_geometric_logo() -> np.ndarray:
    image = blank_canvas()
    triangle = np.array([(300, 370), (500, 170), (500, 370)], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [triangle], ORANGE)
    cv2.rectangle(image, (500, 250), (680, 430), TEAL, cv2.FILLED)
    cv2.circle(image, (500, 470), 110, BLUE, cv2.FILLED)
    return image


def draw_monogram_blocks() -> np.ndarray:
    image = blank_canvas()
    cv2.rectangle(image, (250, 200), (350, 760), TEAL, cv2.FILLED)
    cv2.circle(image, (560, 330), 140, ORANGE, cv2.FILLED)
    triangle = np.array([(430, 760), (430, 520), (760, 760)], dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [triangle], BLUE)
    return image


def draw_filled_star() -> np.ndarray:
    image = blank_canvas()
    star = star_points(center=(480, 480), outer_radius=310, inner_radius=135)
    cv2.fillPoly(image, [star], GOLD)
    return image


def draw_filled_pentagon() -> np.ndarray:
    image = blank_canvas()
    pentagon = regular_polygon_points(center=(480, 500), radius=290, sides=5)
    cv2.fillPoly(image, [pentagon], ORANGE)
    return image


def draw_filled_hexagon() -> np.ndarray:
    image = blank_canvas()
    hexagon = regular_polygon_points(center=(480, 480), radius=295, sides=6)
    cv2.fillPoly(image, [hexagon], TEAL)
    return image


def draw_arrow_badge() -> np.ndarray:
    image = blank_canvas()
    arrow = np.array(
        [
            (220, 420),
            (520, 420),
            (520, 280),
            (760, 480),
            (520, 680),
            (520, 540),
            (220, 540),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(image, [arrow], BLUE)
    return image


def draw_shield_badge() -> np.ndarray:
    image = blank_canvas()
    shield = np.array(
        [
            (480, 150),
            (715, 250),
            (660, 620),
            (480, 810),
            (300, 620),
            (245, 250),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(image, [shield], RED)
    return image


def draw_lightning_mark() -> np.ndarray:
    image = blank_canvas()
    bolt = np.array(
        [
            (560, 140),
            (380, 450),
            (510, 450),
            (400, 820),
            (600, 520),
            (470, 520),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(image, [bolt], GOLD)
    return image


def draw_ribbon_badge_icon() -> np.ndarray:
    """Draw a simple award-ribbon silhouette with one clean outer contour."""
    image = blank_canvas()
    badge = np.array(
        [
            (300, 180),
            (660, 180),
            (760, 300),
            (760, 520),
            (620, 520),
            (680, 800),
            (520, 660),
            (480, 820),
            (440, 660),
            (280, 800),
            (340, 520),
            (200, 520),
            (200, 300),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(image, [badge], GOLD)
    return image


def draw_crown_icon() -> np.ndarray:
    image = blank_canvas()
    crown = np.array(
        [
            (180, 700),
            (240, 360),
            (360, 520),
            (470, 180),
            (600, 520),
            (720, 330),
            (780, 700),
            (780, 800),
            (180, 800),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(image, [crown], RED)
    return image


def draw_bookmark_icon() -> np.ndarray:
    image = blank_canvas()
    bookmark = np.array(
        [
            (300, 150),
            (660, 150),
            (660, 810),
            (480, 650),
            (300, 810),
        ],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    cv2.fillPoly(image, [bookmark], BLUE)
    return image


def draw_leaf_icon() -> np.ndarray:
    """Draw a simple leaf/petal silhouette without internal detail."""
    image = blank_canvas()
    center = np.array([480.0, 480.0], dtype=np.float32)
    points: list[tuple[int, int]] = []

    for t in np.linspace(-1.0, 1.0, 80):
        y = 300.0 * t
        width = 185.0 * (1.0 - abs(t) ** 1.5)
        x = -width + 30.0 * t
        points.append((int(round(center[0] + x)), int(round(center[1] + y))))

    for t in np.linspace(1.0, -1.0, 80):
        y = 300.0 * t
        width = 185.0 * (1.0 - abs(t) ** 1.5)
        x = width + 30.0 * t
        points.append((int(round(center[0] + x)), int(round(center[1] + y))))

    leaf = np.array(points, dtype=np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(image, [leaf], GREEN)
    return image


def build_cases() -> list[tuple[str, str, str, np.ndarray]]:
    return [
        ("01_vertical_line", "Vertical Line", "Single clean vertical stroke.", draw_vertical_line()),
        ("02_horizontal_line", "Horizontal Line", "Single clean horizontal stroke.", draw_horizontal_line()),
        ("03_cross", "Cross", "Two straight strokes crossing at the center.", draw_cross()),
        ("04_filled_square", "Filled Square", "A filled colored square whose contour should be extracted once.", draw_filled_square()),
        ("05_filled_triangle", "Filled Triangle", "A filled colored triangle for clean contour extraction.", draw_filled_triangle()),
        ("06_filled_circle", "Filled Circle", "A filled colored circle that should become one closed contour.", draw_filled_circle()),
        ("07_house_badge", "House Badge", "A simple filled house-style badge made of colored regions.", draw_house_badge()),
        ("08_filled_diamond", "Filled Diamond", "A filled colored diamond for a single clean contour.", draw_filled_diamond()),
        ("09_geometric_logo", "Geometric Logo", "Triangle, square, and circle arranged as filled colored logo parts.", draw_geometric_logo()),
        ("10_monogram_blocks", "Monogram Blocks", "Three separated filled shapes similar to a simple logo mark.", draw_monogram_blocks()),
        ("11_filled_star", "Filled Star", "A filled five-point star for a more presentation-friendly contour demo.", draw_filled_star()),
        ("12_filled_pentagon", "Filled Pentagon", "A filled pentagon that should become one clean polygon contour.", draw_filled_pentagon()),
        ("13_filled_hexagon", "Filled Hexagon", "A filled hexagon that stays simple for the robot to draw.", draw_filled_hexagon()),
        ("14_arrow_badge", "Arrow Badge", "A filled arrow-shaped badge with one bold contour.", draw_arrow_badge()),
        ("15_shield_badge", "Shield Badge", "A filled shield-style emblem for a single closed outline.", draw_shield_badge()),
        ("16_lightning_mark", "Lightning Mark", "A filled lightning-bolt style symbol with one strong silhouette.", draw_lightning_mark()),
        ("17_ribbon_badge", "Ribbon Badge", "A filled award-ribbon silhouette with one strong outer contour.", draw_ribbon_badge_icon()),
        ("18_crown_icon", "Crown", "A filled crown silhouette that stays clean in both classical and ML extraction.", draw_crown_icon()),
        ("19_bookmark_icon", "Bookmark", "A filled bookmark silhouette with a simple notched base.", draw_bookmark_icon()),
        ("20_leaf_icon", "Leaf", "A clean filled leaf silhouette without inner detail.", draw_leaf_icon()),
    ]


def write_summary_markdown(path: Path, cases: list[dict]) -> None:
    lines = [
        "# Presentation Cases",
        "",
        "This folder contains curated demo images chosen for stable robot-friendly presentation cases.",
        "",
        "Recommended pipeline:",
        "",
        "- `classical` perception",
        "- `two_opt` planning",
        "",
        "## Cases",
        "",
    ]
    for case in cases:
        lines.append(f"- `{case['id']}`: {case['title']} - {case['description']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def cleanup_generated_cases(output_dir: Path) -> None:
    """
    Remove old generated case files before writing the new curated pack.

    This keeps stale outline examples from sitting next to the newer filled
    cases and confusing the presentation folder.
    """
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        if path.name in {"manifest.json", "README.md"}:
            path.unlink()
            continue
        if path.suffix.lower() != ".png":
            continue
        if len(path.stem) >= 3 and path.stem[:2].isdigit() and path.stem[2] == "_":
            path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate presentation-ready demo images.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="images/presentation_cases",
        help="Directory where the generated images will be written",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cleanup_generated_cases(output_dir)

    manifest: list[dict] = []
    for stem, title, description, image in build_cases():
        manifest.append(save_case(output_dir, stem, title, description, image))

    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_summary_markdown(output_dir / "README.md", manifest)
    print(f"Generated {len(manifest)} presentation cases in: {output_dir}")


if __name__ == "__main__":
    main()
