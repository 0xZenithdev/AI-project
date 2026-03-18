"""
Search/planning module:
- Decide drawing order of paths.
- Turn ordered paths into robot-friendly plot commands.

This is a baseline heuristic planner (not yet global optimization).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import List, Tuple

# Basic geometry aliases.
Point = Tuple[float, float]
Path = List[Point]


@dataclass
class PlotCommand:
    """
    One low-level drawing command.

    command values:
    - PEN_UP
    - PEN_DOWN
    - MOVE (requires x, y, speed)
    """

    command: str
    x: float | None = None
    y: float | None = None
    speed: float | None = None


# -----------------------------
# Geometry helper utilities
# -----------------------------
def distance(a: Point, b: Point) -> float:
    """Euclidean distance between 2 points."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def scale_paths_to_mm(
    paths_px: list[list[tuple[int, int]]],
    image_size_px: tuple[int, int],
    paper_size_mm: tuple[float, float],
    margin_mm: float = 10.0,
) -> list[Path]:
    """
    Convert paths from image pixels to real paper millimeters.

    Strategy:
    - Keep aspect ratio.
    - Fit inside paper with a margin.
    - Place drawing centered in the drawable area.

    This gives robot-friendly coordinates independent of image resolution.
    """
    img_w, img_h = image_size_px
    paper_w, paper_h = paper_size_mm

    drawable_w = max(1.0, paper_w - 2 * margin_mm)
    drawable_h = max(1.0, paper_h - 2 * margin_mm)

    scale = min(drawable_w / img_w, drawable_h / img_h)

    used_w = img_w * scale
    used_h = img_h * scale

    # Offset centers the drawing in remaining area.
    x_offset = margin_mm + (drawable_w - used_w) / 2.0
    y_offset = margin_mm + (drawable_h - used_h) / 2.0

    scaled_paths: list[Path] = []

    for path in paths_px:
        scaled: Path = []
        for x_px, y_px in path:
            x_mm = x_offset + x_px * scale
            y_mm = y_offset + y_px * scale
            scaled.append((x_mm, y_mm))
        scaled_paths.append(scaled)

    return scaled_paths


# -----------------------------
# Baseline path ordering
# -----------------------------
def order_paths_nearest_neighbor(
    paths: list[Path],
    start_point: Point = (0.0, 0.0),
) -> list[Path]:
    """
    Greedy path ordering:
    - Start from current pen position.
    - Pick the next path whose start OR end is nearest.
    - Reverse path direction if its end is nearer than its start.

    Why this baseline:
    - Simple and fast.
    - Reduces travel moves vs random order.
    - Easy to improve later (TSP/graph optimization).
    """
    remaining = [p[:] for p in paths if len(p) > 0]
    ordered: list[Path] = []
    current = start_point

    while remaining:
        best_idx = 0
        best_reverse = False
        best_cost = float("inf")

        for i, path in enumerate(remaining):
            d_start = distance(current, path[0])
            d_end = distance(current, path[-1])

            # Choose whichever endpoint is closer.
            if d_start < best_cost:
                best_cost = d_start
                best_idx = i
                best_reverse = False

            if d_end < best_cost:
                best_cost = d_end
                best_idx = i
                best_reverse = True

        chosen = remaining.pop(best_idx)
        if best_reverse:
            chosen.reverse()

        ordered.append(chosen)
        current = chosen[-1]

    return ordered


# -----------------------------
# Command generation
# -----------------------------
def build_plot_commands(
    ordered_paths: list[Path],
    travel_speed: float = 60.0,
    draw_speed: float = 35.0,
) -> list[PlotCommand]:
    """
    Convert ordered paths to low-level pen plot commands.

    Command pattern per path:
    1) PEN_UP
    2) MOVE to path start (travel move)
    3) PEN_DOWN
    4) MOVE through all path points (drawing moves)
    5) PEN_UP at end
    """
    commands: list[PlotCommand] = []

    # Ensure safe initial state.
    commands.append(PlotCommand(command="PEN_UP"))

    for path in ordered_paths:
        if not path:
            continue

        start_x, start_y = path[0]

        commands.append(PlotCommand(command="MOVE", x=start_x, y=start_y, speed=travel_speed))
        commands.append(PlotCommand(command="PEN_DOWN"))

        for x, y in path[1:]:
            commands.append(PlotCommand(command="MOVE", x=x, y=y, speed=draw_speed))

        commands.append(PlotCommand(command="PEN_UP"))

    return commands


def commands_to_text(commands: list[PlotCommand]) -> str:
    """
    Serialize commands to simple line-based text.

    Example lines:
    - PEN_UP
    - PEN_DOWN
    - MOVE 12.34 56.78 35.00

    This text format is intentionally easy to parse in a future mBot2 bridge.
    """
    lines: list[str] = []

    for cmd in commands:
        if cmd.command in {"PEN_UP", "PEN_DOWN"}:
            lines.append(cmd.command)
        elif cmd.command == "MOVE":
            lines.append(f"MOVE {cmd.x:.2f} {cmd.y:.2f} {cmd.speed:.2f}")

    return "\n".join(lines)
