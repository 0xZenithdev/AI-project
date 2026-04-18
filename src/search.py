"""
Search/planning module:
- Decide drawing order of paths.
- Clean noisy geometry into robot-friendlier polylines.
- Turn ordered paths into robot-friendly plot commands.

This started as a baseline heuristic planner and now includes conservative
quality cleanup so the robot draws fewer tiny zig-zags and redundant moves.
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


@dataclass
class PathCleanupStats:
    """Summarize the effect of conservative path cleanup before command generation."""

    input_paths: int
    output_paths: int
    input_points: int
    output_points: int
    dropped_paths: int
    dropped_points: int


@dataclass
class PlanMetrics:
    """Summarize ordered drawing paths from a robot-travel perspective."""

    path_count: int
    draw_distance_mm: float
    pen_up_distance_mm: float
    total_distance_mm: float


# -----------------------------
# Geometry helper utilities
# -----------------------------
def distance(a: Point, b: Point) -> float:
    """Euclidean distance between 2 points."""
    return sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)


def path_length(path: Path) -> float:
    """Total polyline length in millimeters."""
    if len(path) < 2:
        return 0.0
    return sum(distance(path[i - 1], path[i]) for i in range(1, len(path)))


def point_line_distance(point: Point, line_start: Point, line_end: Point) -> float:
    """Perpendicular distance from a point to a line segment."""
    line_dx = line_end[0] - line_start[0]
    line_dy = line_end[1] - line_start[1]
    line_length_sq = (line_dx * line_dx) + (line_dy * line_dy)

    if line_length_sq == 0.0:
        return distance(point, line_start)

    projection = (
        ((point[0] - line_start[0]) * line_dx) + ((point[1] - line_start[1]) * line_dy)
    ) / line_length_sq
    projection = max(0.0, min(1.0, projection))

    nearest = (
        line_start[0] + (projection * line_dx),
        line_start[1] + (projection * line_dy),
    )
    return distance(point, nearest)


def collapse_short_segments(path: Path, min_segment_length_mm: float) -> Path:
    """
    Remove consecutive points that are too close together.

    This reduces tiny jittery moves that often do not matter physically but make
    the robot stop and turn more often than necessary.
    """
    if len(path) <= 1 or min_segment_length_mm <= 0.0:
        return path[:]

    cleaned: Path = [path[0]]

    for point in path[1:]:
        if distance(cleaned[-1], point) >= min_segment_length_mm:
            cleaned.append(point)

    return cleaned


def simplify_collinear_points(path: Path, tolerance_mm: float) -> Path:
    """
    Remove middle points that lie nearly on the line between their neighbors.

    This is intentionally conservative compared to aggressive curve fitting,
    because the robot benefits most from removing near-straight jitter while
    preserving obvious corners and shape structure.
    """
    if len(path) <= 2 or tolerance_mm <= 0.0:
        return path[:]

    current = path[:]

    while True:
        simplified: Path = [current[0]]
        removed_any = False

        for index in range(1, len(current) - 1):
            previous = simplified[-1]
            middle = current[index]
            following = current[index + 1]

            if point_line_distance(middle, previous, following) <= tolerance_mm:
                removed_any = True
                continue

            simplified.append(middle)

        simplified.append(current[-1])

        if not removed_any:
            return simplified

        current = simplified


def clean_path(
    path: Path,
    min_segment_length_mm: float,
    simplify_tolerance_mm: float,
) -> Path:
    """
    Apply conservative cleanup passes to one path.

    Memory note:
    - this is the main path-quality hook before ordering and command generation
    - keep defaults conservative because robot testing should refine them, not
      fight against over-aggressive cleanup
    """
    cleaned = collapse_short_segments(path, min_segment_length_mm=min_segment_length_mm)
    cleaned = simplify_collinear_points(cleaned, tolerance_mm=simplify_tolerance_mm)
    cleaned = collapse_short_segments(cleaned, min_segment_length_mm=min_segment_length_mm)
    return cleaned


def cleanup_paths(
    paths: list[Path],
    min_path_length_mm: float = 2.0,
    min_segment_length_mm: float = 0.75,
    simplify_tolerance_mm: float = 0.5,
) -> tuple[list[Path], PathCleanupStats]:
    """
    Clean paths and drop tiny leftovers that are unlikely to draw well.

    Returns the cleaned paths plus stats so output summaries can show the effect.
    """
    input_points = sum(len(path) for path in paths)
    cleaned_paths: list[Path] = []

    for path in paths:
        cleaned = clean_path(
            path,
            min_segment_length_mm=min_segment_length_mm,
            simplify_tolerance_mm=simplify_tolerance_mm,
        )

        if len(cleaned) < 2:
            continue

        if path_length(cleaned) < min_path_length_mm:
            continue

        cleaned_paths.append(cleaned)

    output_points = sum(len(path) for path in cleaned_paths)
    stats = PathCleanupStats(
        input_paths=len(paths),
        output_paths=len(cleaned_paths),
        input_points=input_points,
        output_points=output_points,
        dropped_paths=len(paths) - len(cleaned_paths),
        dropped_points=input_points - output_points,
    )
    return cleaned_paths, stats


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


def summarize_ordered_paths(
    ordered_paths: list[Path],
    start_point: Point = (0.0, 0.0),
) -> PlanMetrics:
    """
    Compute robot-travel metrics for an ordered set of oriented paths.

    These metrics are useful for both AI evaluation and robot validation:
    - draw distance tells us how much actual ink path exists
    - pen-up distance approximates non-drawing travel we want to minimize
    """
    current = start_point
    valid_paths = 0
    draw_distance_mm = 0.0
    pen_up_distance_mm = 0.0

    for path in ordered_paths:
        if len(path) < 2:
            continue

        valid_paths += 1
        pen_up_distance_mm += distance(current, path[0])
        draw_distance_mm += path_length(path)
        current = path[-1]

    return PlanMetrics(
        path_count=valid_paths,
        draw_distance_mm=draw_distance_mm,
        pen_up_distance_mm=pen_up_distance_mm,
        total_distance_mm=draw_distance_mm + pen_up_distance_mm,
    )


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


def order_paths_two_opt(
    paths: list[Path],
    start_point: Point = (0.0, 0.0),
    max_passes: int = 8,
    min_improvement_mm: float = 0.01,
) -> list[Path]:
    """
    TSP-style local search refinement over the greedy route.

    Strategy:
    - build a fast nearest-neighbor baseline first
    - apply oriented 2-opt moves that reverse a subsequence and flip each path
      direction inside that subsequence

    Why this is useful here:
    - keeps runtime practical for many short drawing paths
    - gives the project a clearer search/optimization story than greedy alone
    - directly optimizes the pen-up travel distance that matters for the robot
    """
    route = order_paths_nearest_neighbor(paths=paths, start_point=start_point)
    if len(route) < 3 or max_passes <= 0:
        return route

    for _ in range(max_passes):
        best_gain = 0.0
        best_move: tuple[int, int] | None = None

        for i in range(len(route) - 1):
            prev_point = start_point if i == 0 else route[i - 1][-1]
            first_start = route[i][0]

            for j in range(i + 1, len(route)):
                last_end = route[j][-1]

                old_cost = distance(prev_point, first_start)
                new_cost = distance(prev_point, last_end)

                if j + 1 < len(route):
                    next_start = route[j + 1][0]
                    old_cost += distance(last_end, next_start)
                    new_cost += distance(first_start, next_start)

                gain = old_cost - new_cost
                if gain > best_gain + min_improvement_mm:
                    best_gain = gain
                    best_move = (i, j)

        if best_move is None:
            break

        i, j = best_move
        reversed_segment = [list(reversed(path)) for path in reversed(route[i : j + 1])]
        route = route[:i] + reversed_segment + route[j + 1 :]

    return route


def order_paths(
    paths: list[Path],
    method: str = "two_opt",
    start_point: Point = (0.0, 0.0),
    two_opt_max_passes: int = 8,
) -> list[Path]:
    """
    Dispatch path ordering so experiments can compare search strategies cleanly.
    """
    if method == "nearest_neighbor":
        return order_paths_nearest_neighbor(paths=paths, start_point=start_point)

    if method == "two_opt":
        return order_paths_two_opt(
            paths=paths,
            start_point=start_point,
            max_passes=two_opt_max_passes,
        )

    raise ValueError(f"Unsupported path ordering method: {method}")


# -----------------------------
# Command generation
# -----------------------------
def build_plot_commands(
    ordered_paths: list[Path],
    travel_speed: float = 60.0,
    draw_speed: float = 35.0,
    travel_move_threshold_mm: float = 0.5,
    draw_move_threshold_mm: float = 0.5,
) -> list[PlotCommand]:
    """
    Convert ordered paths to low-level pen plot commands.

    Command pattern per path:
    1) PEN_UP
    2) MOVE to path start (travel move) if needed
    3) PEN_DOWN
    4) MOVE through meaningful path points (drawing moves)
    5) PEN_UP at end

    Small motion thresholds avoid redundant no-op moves caused by rounding or
    cleanup edge cases.
    """
    commands: list[PlotCommand] = []
    current_position: Point = (0.0, 0.0)

    # Ensure safe initial state.
    commands.append(PlotCommand(command="PEN_UP"))

    for path in ordered_paths:
        if len(path) < 2:
            continue

        start_x, start_y = path[0]
        start_point = (start_x, start_y)

        if distance(current_position, start_point) >= travel_move_threshold_mm:
            commands.append(PlotCommand(command="MOVE", x=start_x, y=start_y, speed=travel_speed))
            current_position = start_point

        commands.append(PlotCommand(command="PEN_DOWN"))

        last_drawn_point = current_position
        emitted_draw_move = False

        for x, y in path[1:]:
            next_point = (x, y)
            if distance(last_drawn_point, next_point) < draw_move_threshold_mm:
                continue

            commands.append(PlotCommand(command="MOVE", x=x, y=y, speed=draw_speed))
            last_drawn_point = next_point
            emitted_draw_move = True

        commands.append(PlotCommand(command="PEN_UP"))
        if emitted_draw_move:
            current_position = last_drawn_point

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
