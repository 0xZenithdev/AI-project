"""
Generic helpers for turning binary raster masks into drawable paths.

These helpers are shared by both the classical vision pipeline and the ML
pipeline. Keeping them separate makes the vision module much easier to read:
`vision_v2.py` decides which mask to use, while this module handles the actual
mask-to-path conversion.
"""

from __future__ import annotations

from typing import List, Tuple

import cv2
import numpy as np


Point = Tuple[int, int]
Path = List[Point]

_NEIGHBOR_OFFSETS = [
    (-1, -1),
    (0, -1),
    (1, -1),
    (-1, 0),
    (1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
]


def clean_binary_line_map(
    line_map: np.ndarray,
) -> np.ndarray:
    """
    Normalize a binary line map before vector tracing.

    Important detail:
    very thin strokes must survive this step. A 3x3 morphological open removes
    exactly the kind of single-line test images we want to support, so we clean
    by removing tiny connected specks instead of eroding the whole mask.
    """
    binary = (line_map > 0).astype(np.uint8) * 255
    if cv2.countNonZero(binary) == 0:
        return binary

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    filtered = np.zeros_like(binary)
    for index in range(1, component_count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < 3:
            continue
        filtered[labels == index] = 255

    kernel = np.ones((3, 3), np.uint8)
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, kernel)
    return filtered


def skeletonize_line_map(
    line_map: np.ndarray,
) -> np.ndarray:
    """
    Thin a binary line map to a 1-pixel-style centerline.

    This keeps thick strokes from turning into duplicated boundary traces.
    """
    binary = clean_binary_line_map(line_map)
    if cv2.countNonZero(binary) == 0:
        return binary

    skeleton = np.zeros_like(binary)
    current = binary.copy()
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(current, cv2.MORPH_OPEN, element)
        residue = cv2.subtract(current, opened)
        eroded = cv2.erode(current, element)
        skeleton = cv2.bitwise_or(skeleton, residue)
        current = eroded

        if cv2.countNonZero(current) == 0:
            break

    return skeleton


def _build_adjacency(
    points: set[Point],
) -> dict[Point, list[Point]]:
    adjacency: dict[Point, list[Point]] = {point: [] for point in points}

    for x, y in points:
        neighbors: list[Point] = []
        for dx, dy in _NEIGHBOR_OFFSETS:
            candidate = (x + dx, y + dy)
            if candidate in points:
                neighbors.append(candidate)
        adjacency[(x, y)] = neighbors

    return adjacency


def _edge_key(a: Point, b: Point) -> tuple[Point, Point]:
    return (a, b) if a <= b else (b, a)


def _walk_path(
    start: Point,
    next_point: Point,
    adjacency: dict[Point, list[Point]],
    nodes: set[Point],
    visited_edges: set[tuple[Point, Point]],
) -> list[Point]:
    path: list[Point] = [start, next_point]
    visited_edges.add(_edge_key(start, next_point))

    previous = start
    current = next_point

    while True:
        if current in nodes and current != start:
            break

        candidates = [
            neighbor
            for neighbor in adjacency[current]
            if neighbor != previous and _edge_key(current, neighbor) not in visited_edges
        ]
        if not candidates:
            break

        candidate = candidates[0]
        visited_edges.add(_edge_key(current, candidate))
        previous, current = current, candidate
        path.append(current)

        if current == start:
            break

    return path


def trace_skeleton_paths(
    skeleton: np.ndarray,
) -> list[list[Point]]:
    """
    Convert a skeletonized line image into traced pixel paths.

    Endpoints and junctions become graph nodes. Walks between them become
    robot-friendly stroke paths.
    """
    ys, xs = np.where(skeleton > 0)
    points = {(int(x), int(y)) for y, x in zip(ys, xs)}
    if not points:
        return []

    adjacency = _build_adjacency(points)
    nodes = {
        point
        for point, neighbors in adjacency.items()
        if len(neighbors) not in {0, 2}
    }

    visited_edges: set[tuple[Point, Point]] = set()
    raw_paths: list[list[Point]] = []

    for node in sorted(nodes, key=lambda point: (point[1], point[0])):
        for neighbor in adjacency[node]:
            edge = _edge_key(node, neighbor)
            if edge in visited_edges:
                continue
            path = _walk_path(node, neighbor, adjacency, nodes, visited_edges)
            if len(path) >= 2:
                raw_paths.append(path)

    for point in sorted(points, key=lambda item: (item[1], item[0])):
        for neighbor in adjacency[point]:
            edge = _edge_key(point, neighbor)
            if edge in visited_edges:
                continue
            path = _walk_path(point, neighbor, adjacency, nodes, visited_edges)
            if len(path) >= 3 and path[0] != path[-1]:
                path.append(path[0])
            if len(path) >= 2:
                raw_paths.append(path)

    return raw_paths


def simplify_traced_path(
    path: list[Point],
    min_points: int = 2,
    approx_epsilon_ratio: float = 0.01,
) -> Path:
    """Simplify a traced path while preserving obvious corners."""
    is_closed = len(path) >= 3 and path[0] == path[-1]
    contour_points = path[:-1] if is_closed else path
    if len(contour_points) < 2:
        return []

    contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
    arc_length = cv2.arcLength(contour, closed=is_closed)
    epsilon = max(1.0, approx_epsilon_ratio * arc_length)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=is_closed)
    simple_path: Path = [(int(point[0][0]), int(point[0][1])) for point in simplified]

    if is_closed:
        if len(simple_path) < max(3, min_points):
            return []
        if simple_path[0] != simple_path[-1]:
            simple_path.append(simple_path[0])
        return simple_path

    if len(simple_path) < min_points:
        return []

    return simple_path


def simplify_closed_contour(
    contour: np.ndarray,
    approx_epsilon_ratio: float = 0.01,
    min_points: int = 3,
) -> Path:
    """Simplify one closed contour and explicitly keep it closed."""
    arc_length = cv2.arcLength(contour, closed=True)
    epsilon = max(1.0, approx_epsilon_ratio * arc_length)
    simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
    path: Path = [(int(point[0][0]), int(point[0][1])) for point in simplified]
    if len(path) < min_points:
        return []
    if path[0] != path[-1]:
        path.append(path[0])
    return path


def align_closed_path(reference: Path, candidate: Path) -> Path:
    """
    Rotate/reverse one closed path so its vertices line up with another path.

    This lets us collapse a thick ring stroke into a centerline-style polygon.
    """
    ref_points = reference[:-1]
    candidate_points = candidate[:-1]
    if len(ref_points) != len(candidate_points):
        return []

    best_cost: float | None = None
    best_points: list[Point] = []

    for reverse in (False, True):
        working = list(reversed(candidate_points)) if reverse else candidate_points[:]
        for shift in range(len(working)):
            rotated = working[shift:] + working[:shift]
            cost = 0.0
            for ref_point, candidate_point in zip(ref_points, rotated):
                dx = ref_point[0] - candidate_point[0]
                dy = ref_point[1] - candidate_point[1]
                cost += float(dx * dx + dy * dy)

            if best_cost is None or cost < best_cost:
                best_cost = cost
                best_points = rotated

    if not best_points:
        return []

    return best_points + [best_points[0]]


def average_closed_paths(
    outer: Path,
    inner: Path,
) -> Path:
    """Average 2 aligned closed polygons into one centerline-style polygon."""
    outer_points = outer[:-1]
    inner_points = inner[:-1]
    if len(outer_points) != len(inner_points):
        return []

    midpoint_path: Path = []
    for outer_point, inner_point in zip(outer_points, inner_points):
        midpoint_path.append(
            (
                int(round((outer_point[0] + inner_point[0]) / 2.0)),
                int(round((outer_point[1] + inner_point[1]) / 2.0)),
            )
        )

    if len(midpoint_path) < 3:
        return []

    midpoint_path.append(midpoint_path[0])
    return midpoint_path


def closed_path_area(path: Path) -> float:
    """Compute polygon area for a closed path."""
    if len(path) < 4:
        return 0.0
    contour_points = np.array(path[:-1], dtype=np.float32).reshape(-1, 1, 2)
    return abs(float(cv2.contourArea(contour_points)))


def closed_path_centroid(path: Path) -> Point:
    """Compute a simple centroid for a closed path."""
    if len(path) < 2:
        return (0, 0)
    points = path[:-1] if len(path) >= 3 and path[0] == path[-1] else path
    if not points:
        return (0, 0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        int(round(sum(xs) / float(len(xs)))),
        int(round(sum(ys) / float(len(ys)))),
    )


def closed_path_bbox(path: Path) -> tuple[int, int]:
    """Return bounding-box width and height for a path."""
    points = path[:-1] if len(path) >= 3 and path[0] == path[-1] else path
    if not points:
        return (0, 0)
    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (max(xs) - min(xs), max(ys) - min(ys))


def resample_closed_path(path: Path, target_points: int) -> Path:
    """
    Resample a closed polygon so another path can be aligned/averaged with it.

    This is useful when 2 duplicate contours represent the same shape but have
    different vertex counts after simplification.
    """
    points = path[:-1] if len(path) >= 3 and path[0] == path[-1] else path[:]
    if len(points) < 3 or target_points < 3:
        return []

    samples = points + [points[0]]
    segment_lengths: list[float] = [0.0]
    total_length = 0.0
    for previous, current in zip(samples, samples[1:]):
        dx = float(current[0] - previous[0])
        dy = float(current[1] - previous[1])
        total_length += float(np.hypot(dx, dy))
        segment_lengths.append(total_length)

    if total_length <= 0.0:
        return []

    resampled: Path = []
    for index in range(target_points):
        target_distance = (total_length * index) / float(target_points)
        segment_index = 0
        while segment_index + 1 < len(segment_lengths) and segment_lengths[segment_index + 1] < target_distance:
            segment_index += 1

        start = samples[segment_index]
        end = samples[segment_index + 1]
        start_distance = segment_lengths[segment_index]
        end_distance = segment_lengths[segment_index + 1]
        span = max(end_distance - start_distance, 1e-6)
        alpha = (target_distance - start_distance) / span
        resampled.append(
            (
                int(round(start[0] + (end[0] - start[0]) * alpha)),
                int(round(start[1] + (end[1] - start[1]) * alpha)),
            )
        )

    if resampled[0] != resampled[-1]:
        resampled.append(resampled[0])
    return resampled


def merge_duplicate_closed_paths(
    paths: list[Path],
    min_area_ratio: float = 0.72,
    max_center_distance_ratio: float = 0.08,
    max_bbox_size_gap_ratio: float = 0.22,
) -> list[Path]:
    """
    Merge near-duplicate closed loops into one centerline-style contour.

    This targets the common Canny failure mode where a thick stroke produces 2
    very similar closed loops around the same intended shape.
    """
    closed_paths = [
        path for path in paths
        if len(path) >= 4 and path[0] == path[-1]
    ]
    if len(closed_paths) < 2:
        return paths[:]

    path_infos: list[dict] = []
    for path in closed_paths:
        area = closed_path_area(path)
        centroid = closed_path_centroid(path)
        bbox_w, bbox_h = closed_path_bbox(path)
        path_infos.append(
            {
                "path": path,
                "area": area,
                "centroid": centroid,
                "bbox_w": bbox_w,
                "bbox_h": bbox_h,
            }
        )

    order = sorted(range(len(path_infos)), key=lambda idx: path_infos[idx]["area"], reverse=True)
    used: set[int] = set()
    merged: list[Path] = []

    for index in order:
        if index in used:
            continue

        info = path_infos[index]
        best_match: int | None = None
        best_score = -1.0

        for candidate_index in order:
            if candidate_index == index or candidate_index in used:
                continue

            candidate = path_infos[candidate_index]
            larger_area = max(info["area"], candidate["area"], 1.0)
            smaller_area = min(info["area"], candidate["area"])
            area_ratio = smaller_area / larger_area
            if area_ratio < min_area_ratio:
                continue

            center_dx = float(info["centroid"][0] - candidate["centroid"][0])
            center_dy = float(info["centroid"][1] - candidate["centroid"][1])
            center_distance = float(np.hypot(center_dx, center_dy))
            max_span = max(info["bbox_w"], info["bbox_h"], candidate["bbox_w"], candidate["bbox_h"], 1)
            if center_distance > (max_span * max_center_distance_ratio):
                continue

            width_gap_ratio = abs(info["bbox_w"] - candidate["bbox_w"]) / float(max(info["bbox_w"], candidate["bbox_w"], 1))
            height_gap_ratio = abs(info["bbox_h"] - candidate["bbox_h"]) / float(max(info["bbox_h"], candidate["bbox_h"], 1))
            if width_gap_ratio > max_bbox_size_gap_ratio or height_gap_ratio > max_bbox_size_gap_ratio:
                continue

            score = area_ratio - (center_distance / float(max_span))
            if score > best_score:
                best_score = score
                best_match = candidate_index

        if best_match is None:
            merged.append(info["path"])
            used.add(index)
            continue

        candidate = path_infos[best_match]
        target_points = max(4, int(round((len(info["path"]) + len(candidate["path"])) / 2.0)) - 1)
        left = resample_closed_path(info["path"], target_points)
        right = resample_closed_path(candidate["path"], target_points)
        aligned = align_closed_path(left, right) if left and right else []
        averaged = average_closed_paths(left, aligned) if aligned else []

        if averaged:
            contour = np.array(averaged[:-1], dtype=np.int32).reshape(-1, 1, 2)
            epsilon = max(1.0, 0.01 * cv2.arcLength(contour, closed=True))
            simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
            merged_path: Path = [(int(point[0][0]), int(point[0][1])) for point in simplified]
            if len(merged_path) >= 3:
                if merged_path[0] != merged_path[-1]:
                    merged_path.append(merged_path[0])
                merged.append(merged_path)
            else:
                merged.append(info["path"])
        else:
            merged.append(info["path"])

        used.add(index)
        used.add(best_match)

    return merged


def extract_ring_paths(
    binary_line_map: np.ndarray,
    approx_epsilon_ratio: float = 0.01,
) -> tuple[list[Path], np.ndarray]:
    """
    Detect outlined closed strokes and collapse their inner/outer borders.

    Example:
    a thick square outline becomes one square path instead of 2 borders.
    """
    contours, hierarchy = cv2.findContours(binary_line_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    processed_mask = np.zeros_like(binary_line_map)
    if hierarchy is None:
        return [], processed_mask

    paths: list[Path] = []

    for index, contour in enumerate(contours):
        parent_index = int(hierarchy[0][index][3])
        first_child_index = int(hierarchy[0][index][2])
        if parent_index != -1 or first_child_index == -1:
            continue

        outer_path = simplify_closed_contour(contour, approx_epsilon_ratio=approx_epsilon_ratio)
        if not outer_path:
            continue

        child_indices: list[int] = []
        current_child = first_child_index
        while current_child != -1:
            if int(hierarchy[0][current_child][3]) == index:
                child_indices.append(current_child)
            current_child = int(hierarchy[0][current_child][0])

        best_centerline: Path = []
        best_hole_area = 0.0

        for child_index in child_indices:
            inner_path = simplify_closed_contour(
                contours[child_index],
                approx_epsilon_ratio=approx_epsilon_ratio,
            )
            if not inner_path or len(inner_path) != len(outer_path):
                continue

            aligned_inner = align_closed_path(outer_path, inner_path)
            if not aligned_inner:
                continue

            centerline = average_closed_paths(outer_path, aligned_inner)
            hole_area = abs(float(cv2.contourArea(contours[child_index])))
            if centerline and hole_area > best_hole_area:
                best_centerline = centerline
                best_hole_area = hole_area

        if not best_centerline:
            continue

        paths.append(best_centerline)
        cv2.drawContours(processed_mask, contours, index, 255, thickness=cv2.FILLED)
        for child_index in child_indices:
            cv2.drawContours(processed_mask, contours, child_index, 0, thickness=cv2.FILLED)

    return paths, processed_mask


def extract_elongated_component_paths(
    binary_line_map: np.ndarray,
    min_area_px: int = 80,
    min_length_px: float = 24.0,
    min_aspect_ratio: float = 6.0,
    min_extent: float = 0.55,
) -> tuple[list[Path], np.ndarray]:
    """
    Collapse thick filled bars into single centerline segments.

    This helps simple vertical/horizontal bars avoid fragmenting into many
    tiny skeleton segments.
    """
    binary = (binary_line_map > 0).astype(np.uint8)
    processed_mask = np.zeros_like(binary_line_map)
    component_count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    paths: list[Path] = []

    height, width = binary.shape[:2]

    for index in range(1, component_count):
        x, y, component_width, component_height, area = stats[index]
        if int(area) < min_area_px:
            continue

        short_side = float(min(component_width, component_height))
        long_side = float(max(component_width, component_height))
        if short_side <= 0.0 or long_side < min_length_px:
            continue

        aspect_ratio = long_side / short_side
        extent = float(area) / float(component_width * component_height)
        if aspect_ratio < min_aspect_ratio or extent < min_extent:
            continue

        ys, xs = np.where(labels == index)
        points = np.column_stack((xs, ys)).astype(np.float32)
        if len(points) < 2:
            continue

        vx, vy, x0, y0 = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        direction = np.array([float(vx[0]), float(vy[0])], dtype=np.float32)
        origin = np.array([float(x0[0]), float(y0[0])], dtype=np.float32)
        projections = (points - origin) @ direction

        start_point = origin + direction * float(projections.min())
        end_point = origin + direction * float(projections.max())

        start = (
            int(np.clip(round(float(start_point[0])), 0, width - 1)),
            int(np.clip(round(float(start_point[1])), 0, height - 1)),
        )
        end = (
            int(np.clip(round(float(end_point[0])), 0, width - 1)),
            int(np.clip(round(float(end_point[1])), 0, height - 1)),
        )
        if start == end:
            continue

        paths.append([start, end])
        processed_mask[labels == index] = 255

    return paths, processed_mask


def _dedupe_open_paths(
    paths: list[Path],
    endpoint_tolerance_px: float = 4.0,
) -> list[Path]:
    """Drop near-identical open line segments produced by multiple passes."""
    deduped: list[Path] = []

    for path in paths:
        if len(path) < 2:
            continue

        start = np.array(path[0], dtype=np.float32)
        end = np.array(path[-1], dtype=np.float32)
        is_duplicate = False

        for existing in deduped:
            if len(existing) < 2:
                continue
            existing_start = np.array(existing[0], dtype=np.float32)
            existing_end = np.array(existing[-1], dtype=np.float32)

            same_direction = (
                np.linalg.norm(start - existing_start) <= endpoint_tolerance_px
                and np.linalg.norm(end - existing_end) <= endpoint_tolerance_px
            )
            reverse_direction = (
                np.linalg.norm(start - existing_end) <= endpoint_tolerance_px
                and np.linalg.norm(end - existing_start) <= endpoint_tolerance_px
            )
            if same_direction or reverse_direction:
                is_duplicate = True
                break

        if not is_duplicate:
            deduped.append(path)

    return deduped


def extract_axis_aligned_stroke_paths(
    binary_line_map: np.ndarray,
    kernel_span_px: int = 15,
    min_coverage_ratio: float = 0.72,
) -> tuple[list[Path], np.ndarray, float]:
    """
    Recover centerline segments for images dominated by straight bars.

    This is the right interpretation for cases like:
    - a plus sign made of 2 thick strokes
    - a single vertical or horizontal bar

    In those cases, tracing the outer silhouette creates a star-like polygon,
    while the intended robot result is the stroke centerline.
    """
    clean_map = clean_binary_line_map(binary_line_map)
    if cv2.countNonZero(clean_map) == 0:
        return [], np.zeros_like(clean_map), 0.0

    kernel_span = max(9, int(kernel_span_px))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, kernel_span))
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_span, 3))

    vertical_mask = cv2.morphologyEx(clean_map, cv2.MORPH_OPEN, vertical_kernel)
    horizontal_mask = cv2.morphologyEx(clean_map, cv2.MORPH_OPEN, horizontal_kernel)

    vertical_paths, vertical_used = extract_elongated_component_paths(
        vertical_mask,
        min_area_px=40,
        min_length_px=32.0,
        min_aspect_ratio=5.0,
        min_extent=0.55,
    )
    horizontal_paths, horizontal_used = extract_elongated_component_paths(
        horizontal_mask,
        min_area_px=40,
        min_length_px=32.0,
        min_aspect_ratio=5.0,
        min_extent=0.55,
    )

    combined_paths = _dedupe_open_paths(vertical_paths + horizontal_paths)
    combined_mask = cv2.bitwise_or(vertical_used, horizontal_used)
    foreground_pixels = max(1, cv2.countNonZero(clean_map))
    coverage_ratio = float(cv2.countNonZero(combined_mask)) / float(foreground_pixels)

    if not combined_paths or coverage_ratio < float(min_coverage_ratio):
        return [], np.zeros_like(clean_map), coverage_ratio

    trace_map = render_trace_map(combined_paths, clean_map.shape)
    return combined_paths, trace_map, coverage_ratio


def render_trace_map(
    paths: list[Path],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Render traced paths back into a debug image."""
    canvas = np.zeros(image_shape, dtype=np.uint8)

    for path in paths:
        if len(path) < 2:
            continue
        is_closed = len(path) >= 3 and path[0] == path[-1]
        contour_points = path[:-1] if is_closed else path
        contour = np.array(contour_points, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [contour], isClosed=is_closed, color=255, thickness=1)

    return canvas


def extract_filled_region_paths(
    binary_line_map: np.ndarray,
    approx_epsilon_ratio: float = 0.01,
) -> list[Path]:
    """
    Trace dense filled regions by their outer/inner boundaries.

    This is the clean fallback for filled shapes that should be outlined rather
    than skeletonized.
    """
    contours, _ = cv2.findContours(binary_line_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    paths: list[Path] = []

    for contour in contours:
        path = simplify_closed_contour(
            contour,
            approx_epsilon_ratio=approx_epsilon_ratio,
        )
        if path:
            paths.append(path)

    return paths


def extract_paths_from_line_map(
    line_map: np.ndarray,
    min_points: int = 2,
    approx_epsilon_ratio: float = 0.01,
) -> tuple[list[Path], np.ndarray]:
    """
    Convert a binary line map into ordered stroke paths plus a traced debug map.
    """
    clean_map = clean_binary_line_map(line_map)
    elongated_paths, elongated_mask = extract_elongated_component_paths(clean_map)
    ring_input = cv2.subtract(clean_map, elongated_mask)
    ring_paths, ring_mask = extract_ring_paths(
        ring_input,
        approx_epsilon_ratio=approx_epsilon_ratio,
    )

    processed_mask = cv2.bitwise_or(elongated_mask, ring_mask)
    leftover_map = cv2.subtract(clean_map, processed_mask)
    skeleton = skeletonize_line_map(leftover_map)
    raw_paths = trace_skeleton_paths(skeleton)

    paths: list[Path] = elongated_paths[:] + ring_paths[:]
    for path in raw_paths:
        simplified = simplify_traced_path(
            path,
            min_points=min_points,
            approx_epsilon_ratio=approx_epsilon_ratio,
        )
        if len(simplified) >= 2:
            paths.append(simplified)

    trace_map = render_trace_map(paths, clean_map.shape)
    return paths, trace_map


def extract_contours(
    edges: np.ndarray,
    min_points: int = 2,
    approx_epsilon_ratio: float = 0.01,
) -> list[Path]:
    """Backward-compatible wrapper around the stroke-tracing pipeline."""
    paths, _ = extract_paths_from_line_map(
        edges,
        min_points=min_points,
        approx_epsilon_ratio=approx_epsilon_ratio,
    )
    return paths


def extract_edge_contour_paths(
    edge_map: np.ndarray,
    min_points: int = 4,
    approx_epsilon_ratio: float = 0.01,
) -> list[Path]:
    """
    Convert a thin edge map into contour paths.

    Keeping child contours when available removes most doubled-outline artifacts
    from Canny-style edge loops.
    """
    contours, hierarchy = cv2.findContours(edge_map, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    if hierarchy is None:
        return []

    paths: list[Path] = []

    preferred_indices = [
        index
        for index, meta in enumerate(hierarchy[0])
        if int(meta[3]) != -1
    ]
    if not preferred_indices:
        preferred_indices = list(range(len(contours)))

    for index in preferred_indices:
        contour = contours[index]
        if len(contour) < min_points:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        if width <= 2 or height <= 2:
            continue

        epsilon = max(1.0, approx_epsilon_ratio * cv2.arcLength(contour, closed=True))
        simplified = cv2.approxPolyDP(contour, epsilon, closed=True)
        if len(simplified) < min_points:
            continue

        path: Path = [(int(point[0][0]), int(point[0][1])) for point in simplified]
        if path[0] != path[-1]:
            path.append(path[0])
        paths.append(path)

    return merge_duplicate_closed_paths(paths)


__all__ = [
    "Path",
    "Point",
    "clean_binary_line_map",
    "extract_contours",
    "extract_axis_aligned_stroke_paths",
    "extract_edge_contour_paths",
    "extract_filled_region_paths",
    "extract_paths_from_line_map",
    "merge_duplicate_closed_paths",
    "render_trace_map",
]
