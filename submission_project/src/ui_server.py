"""Local web UI for the robot-drawing pipeline."""

from __future__ import annotations

import argparse
import base64
from datetime import datetime
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re
from types import SimpleNamespace
from urllib.parse import unquote, urlparse
import webbrowser

import cv2
import numpy as np

import main as pipeline_main
from src import bridge_sender
from src.bridge_protocol import load_plot_commands
from src.evaluate_pipeline import build_summary, compute_mask_metrics, load_binary_mask
from src.mblock_script_generator import (
    DEFAULT_START_HEADING_DEG,
    DEFAULT_START_TIP_X_MM,
    DEFAULT_START_TIP_Y_MM,
    compile_actions,
    render_script,
    write_script_outputs,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
UI_ROOT = REPO_ROOT / "ui"
UI_INDEX = UI_ROOT / "index.html"
UI_OUTPUT_ROOT = REPO_ROOT / "output" / "ui_sessions"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8010
DEFAULT_ROBOT_EXPORT_SETTINGS = {
    "usePen": True,
    "servoPort": "S1",
    "mmPerStraightUnit": 10.0,
    "turnScale": 1.04,
    "minMoveMm": 20.0,
    "minStraightUnits": 2.0,
    "penLiftDelta": 36.0,
    "penDelayS": 0.3,
    "turnDelayS": 0.1,
    "cornerLiftTurnDeg": 30.0,
    "penForwardOffsetMm": 160.0,
    "penLateralOffsetMm": 0.0,
    "startXMm": DEFAULT_START_TIP_X_MM,
    "startYMm": DEFAULT_START_TIP_Y_MM,
    "startHeadingDeg": DEFAULT_START_HEADING_DEG,
}

VERIFIED_CHECKPOINT_NAMES = [
    "line_model_presentation_filledmask_tuned.pt",
    "line_model.pt",
    "line_model_photosketch_w5_aspect_overnight.pt",
    "line_model_photosketch_w5_aspect.pt",
    "line_model_photosketch_w5_aspect_quick.pt",
    "line_model_finetuned.pt",
    "line_model_photosketch_w5.pt",
    "line_model_photosketch.pt",
    "line_model_v2.pt",
]

EXPERIMENTAL_CHECKPOINT_HINTS = (
    "vehicle_logos",
    "geometric",
    "outline_geometric",
)

IMAGE_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local robot-drawing web UI.")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port")
    parser.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the UI in the default browser after the server starts",
    )
    return parser.parse_args()


def json_response(handler: BaseHTTPRequestHandler, payload: dict, status: int = 200) -> None:
    data = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler: BaseHTTPRequestHandler, payload: str, status: int = 200) -> None:
    data = payload.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
    handler.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
    handler.send_header("Pragma", "no-cache")
    handler.send_header("Expires", "0")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def load_request_json(handler: BaseHTTPRequestHandler) -> dict:
    content_length = int(handler.headers.get("Content-Length", "0"))
    raw = handler.rfile.read(content_length)
    if not raw:
        raise ValueError("Request body is empty")
    return json.loads(raw.decode("utf-8"))


def sanitize_filename(filename: str) -> str:
    cleaned = Path(filename or "upload.png").name
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", cleaned).strip("._")
    return cleaned or "upload.png"


def make_session_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def relative_to_repo(path: Path) -> str:
    return str(path.resolve().relative_to(REPO_ROOT.resolve())).replace("\\", "/")


def build_file_url(path: Path) -> str:
    return f"/files/{relative_to_repo(path)}"


def list_model_checkpoints() -> list[dict]:
    models_dir = REPO_ROOT / "models"
    if not models_dir.exists():
        return []

    checkpoints = sorted(models_dir.glob("*.pt"), key=lambda item: item.stat().st_mtime, reverse=True)
    items = []
    for checkpoint in checkpoints:
        if not checkpoint.is_file():
            continue
        name = checkpoint.name
        if name in VERIFIED_CHECKPOINT_NAMES:
            tier = "verified"
        elif any(hint in name for hint in EXPERIMENTAL_CHECKPOINT_HINTS):
            tier = "experimental"
        else:
            tier = "other"
        items.append(
            {
                "name": name,
                "path": relative_to_repo(checkpoint),
                "updatedAt": datetime.fromtimestamp(checkpoint.stat().st_mtime).isoformat(timespec="seconds"),
                "tier": tier,
            }
        )
    return items


def pick_recommended_checkpoint(checkpoints: list[dict]) -> str:
    if not checkpoints:
        return ""

    preferred_names = VERIFIED_CHECKPOINT_NAMES
    path_by_name = {item["name"]: item["path"] for item in checkpoints}
    for name in preferred_names:
        if name in path_by_name:
            return path_by_name[name]
    return str(checkpoints[0]["path"])


def build_ui_config() -> dict:
    checkpoints = list_model_checkpoints()
    return {
        "defaults": {
            "visionMode": "classical",
            "pathOrdering": "two_opt",
            "recommendedCheckpoint": pick_recommended_checkpoint(checkpoints),
            "robotExport": DEFAULT_ROBOT_EXPORT_SETTINGS,
        },
        "checkpoints": checkpoints,
        "notes": {
            "recommendedPipeline": "classical + two_opt",
            "aiSummary": "Perception compares a classical baseline with an ML segmentation model. Planning compares nearest-neighbor with two-opt route optimization.",
            "recommendedCheckpointNote": "Recommended ML checkpoint: line_model_presentation_filledmask_tuned.pt. The live demo path remains classical plus two-opt.",
            "robotSummary": "UI export defaults match the current pen mount: with the mBot2 back at the bottom edge, the pen contact point starts about 20 cm into A4 at (105, 97), facing into the paper. Pen-up uses negative servo delta, pen-down uses positive servo delta, PEN_LIFT_DELTA 36, TURN_SCALE 1.04, 160 mm forward pen offset, and small robot moves are clamped up instead of dropped.",
        },
    }


def resolve_repo_path(path_str: str) -> Path:
    candidate = Path(path_str)
    if not candidate.is_absolute():
        candidate = REPO_ROOT / candidate
    resolved = candidate.resolve()
    repo_root = REPO_ROOT.resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError(f"Path escapes repository: {path_str}") from exc
    return resolved


def decode_data_url(data_url: str) -> tuple[bytes, str]:
    match = IMAGE_DATA_URL_RE.match(data_url.strip())
    if not match:
        raise ValueError("Invalid image data URL")
    raw_bytes = base64.b64decode(match.group("data"))
    return raw_bytes, match.group("mime")


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def render_path_preview(
    paths_mm: list[list[tuple[float, float]]],
    paper_width_mm: float,
    paper_height_mm: float,
    margin_mm: float,
    output_path: Path,
) -> None:
    """
    Render a paper-style preview of the ordered drawing paths.

    Implementation note:
    - the UI preview is for operator confidence, not geometric truth
    - fit the preview canvas to the visible drawing bounds rather than the full
      paper aspect ratio, otherwise square drawings appear cramped inside a tall
      page-shaped image
    """
    all_points = [(x, y) for path in paths_mm for x, y in path]
    if all_points:
        xs = np.array([point[0] for point in all_points], dtype=np.float32)
        ys = np.array([point[1] for point in all_points], dtype=np.float32)
        preview_padding_mm = max(8.0, min(paper_width_mm, paper_height_mm) * 0.04)
        view_min_x = max(0.0, float(xs.min()) - preview_padding_mm)
        view_max_x = min(float(paper_width_mm), float(xs.max()) + preview_padding_mm)
        view_min_y = max(0.0, float(ys.min()) - preview_padding_mm)
        view_max_y = min(float(paper_height_mm), float(ys.max()) + preview_padding_mm)
    else:
        view_min_x = 0.0
        view_max_x = float(paper_width_mm)
        view_min_y = 0.0
        view_max_y = float(paper_height_mm)

    view_width_mm = max(view_max_x - view_min_x, 1.0)
    view_height_mm = max(view_max_y - view_min_y, 1.0)

    inset_px = 24.0
    target_long_edge_px = 920.0
    scale = max(1.0, (target_long_edge_px - inset_px * 2.0) / max(view_width_mm, view_height_mm))
    width_px = max(320, int(round(view_width_mm * scale + inset_px * 2.0)))
    height_px = max(320, int(round(view_height_mm * scale + inset_px * 2.0)))

    canvas = np.full((height_px, width_px, 3), (240, 236, 228), dtype=np.uint8)
    border = (194, 182, 162)
    grid = (226, 219, 206)
    ink = (35, 55, 88)
    accent = (179, 95, 67)

    cv2.rectangle(canvas, (0, 0), (width_px - 1, height_px - 1), border, 2)

    scale = min(
        (width_px - inset_px * 2.0) / view_width_mm,
        (height_px - inset_px * 2.0) / view_height_mm,
    )
    offset_x = (width_px - view_width_mm * scale) / 2.0
    offset_y = (height_px - view_height_mm * scale) / 2.0

    def project_point(x_mm: float, y_mm: float) -> tuple[int, int]:
        return (
            int(round(offset_x + (x_mm - view_min_x) * scale)),
            int(round(offset_y + (y_mm - view_min_y) * scale)),
        )

    grid_step_mm = 10.0
    start_x_mm = float(np.floor(view_min_x / grid_step_mm) * grid_step_mm)
    start_y_mm = float(np.floor(view_min_y / grid_step_mm) * grid_step_mm)

    x_mm = start_x_mm
    while x_mm <= view_max_x + 1e-6:
        x0, y0 = project_point(x_mm, view_min_y)
        x1, y1 = project_point(x_mm, view_max_y)
        cv2.line(canvas, (x0, y0), (x1, y1), grid, 1, cv2.LINE_AA)
        x_mm += grid_step_mm

    y_mm = start_y_mm
    while y_mm <= view_max_y + 1e-6:
        x0, y0 = project_point(view_min_x, y_mm)
        x1, y1 = project_point(view_max_x, y_mm)
        cv2.line(canvas, (x0, y0), (x1, y1), grid, 1, cv2.LINE_AA)
        y_mm += grid_step_mm

    inner_top_left = project_point(view_min_x, view_min_y)
    inner_bottom_right = project_point(view_max_x, view_max_y)
    cv2.rectangle(canvas, inner_top_left, inner_bottom_right, (210, 201, 186), 1)

    drawable_min_x = max(view_min_x, float(margin_mm))
    drawable_max_x = min(view_max_x, float(paper_width_mm - margin_mm))
    drawable_min_y = max(view_min_y, float(margin_mm))
    drawable_max_y = min(view_max_y, float(paper_height_mm - margin_mm))
    if drawable_min_x < drawable_max_x and drawable_min_y < drawable_max_y:
        cv2.rectangle(
            canvas,
            project_point(drawable_min_x, drawable_min_y),
            project_point(drawable_max_x, drawable_max_y),
            (198, 187, 170),
            1,
        )

    for path in paths_mm:
        if len(path) < 2:
            continue
        points = np.array([project_point(x, y) for x, y in path], dtype=np.int32)
        cv2.polylines(canvas, [points], False, ink, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[0]), 3, accent, -1, cv2.LINE_AA)

    ensure_parent(output_path)
    cv2.imwrite(str(output_path), canvas)


def render_detection_preview(mask: np.ndarray, output_path: Path) -> None:
    """
    Render a UI-friendly detection preview.

    The raw edge map uses the fixed working canvas, which is fine for processing
    but looks awkward in the comparison cards. This view crops around the active
    pixels and centers them on a compact preview canvas.
    """
    binary = (mask > 0).astype(np.uint8) * 255
    points = np.column_stack(np.where(binary > 0))
    if points.size == 0:
        cropped = binary
    else:
        y_min, x_min = points.min(axis=0)
        y_max, x_max = points.max(axis=0)
        pad = max(8, int(round(max(binary.shape) * 0.04)))
        x_min = max(0, int(x_min) - pad)
        y_min = max(0, int(y_min) - pad)
        x_max = min(binary.shape[1] - 1, int(x_max) + pad)
        y_max = min(binary.shape[0] - 1, int(y_max) + pad)
        cropped = binary[y_min : y_max + 1, x_min : x_max + 1]

    crop_h, crop_w = cropped.shape[:2]
    inset_px = 22.0
    target_long_edge_px = 920.0
    scale = max(1.0, (target_long_edge_px - inset_px * 2.0) / max(crop_w, crop_h, 1))
    width_px = max(320, int(round(crop_w * scale + inset_px * 2.0)))
    height_px = max(320, int(round(crop_h * scale + inset_px * 2.0)))

    canvas = np.full((height_px, width_px, 3), (12, 12, 12), dtype=np.uint8)
    border = (56, 60, 66)
    line_color = np.array([242, 240, 236], dtype=np.uint8)
    cv2.rectangle(canvas, (0, 0), (width_px - 1, height_px - 1), border, 1)

    scaled_w = max(1, int(round(crop_w * scale)))
    scaled_h = max(1, int(round(crop_h * scale)))
    resized = cv2.resize(cropped, (scaled_w, scaled_h), interpolation=cv2.INTER_NEAREST)
    layer = np.zeros((scaled_h, scaled_w, 3), dtype=np.uint8)
    layer[resized > 0] = line_color

    offset_x = (width_px - scaled_w) // 2
    offset_y = (height_px - scaled_h) // 2
    canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w] = np.maximum(
        canvas[offset_y : offset_y + scaled_h, offset_x : offset_x + scaled_w],
        layer,
    )

    ensure_parent(output_path)
    cv2.imwrite(str(output_path), canvas)


def build_pipeline_args(image_path: Path, output_dir: Path, options: dict) -> SimpleNamespace:
    defaults = {
        "image": str(image_path),
        "width": 210,
        "height": 297,
        "vision_mode": "classical",
        "model_checkpoint": "",
        "ml_threshold": 0.5,
        "ml_device": "cpu",
        "paper_width_mm": 210.0,
        "paper_height_mm": 297.0,
        "margin_mm": 10.0,
        "travel_speed": 60.0,
        "draw_speed": 35.0,
        "min_path_length_mm": 2.0,
        "min_segment_length_mm": 0.75,
        "simplify_tolerance_mm": 0.5,
        "travel_move_threshold_mm": 0.5,
        "draw_move_threshold_mm": 0.5,
        "path_ordering": "two_opt",
        "two_opt_max_passes": 8,
        "output_dir": str(output_dir),
        "no_export": False,
    }
    defaults.update(options)
    defaults["image"] = str(image_path)
    defaults["output_dir"] = str(output_dir)
    return SimpleNamespace(**defaults)


def summarize_results(args: SimpleNamespace, results: dict, session_dir: Path, image_path: Path) -> dict:
    edges_preview_path = session_dir / "edges_preview_ui.png"
    render_detection_preview(results["edges"], edges_preview_path)
    path_preview_path = session_dir / "path_preview.png"
    render_path_preview(
        paths_mm=results["paths_mm"],
        paper_width_mm=args.paper_width_mm,
        paper_height_mm=args.paper_height_mm,
        margin_mm=args.margin_mm,
        output_path=path_preview_path,
    )

    return {
        "sessionId": session_dir.name,
        "imagePath": str(image_path),
        "commandsPath": str(session_dir / "plot_commands.txt"),
        "summaryPath": str(session_dir / "paths_summary.json"),
        "edgesPreviewPath": str(edges_preview_path),
        "pathPreviewPath": str(path_preview_path),
        "numPaths": len(results["paths_mm"]),
        "numCommands": len(results["commands"]),
        "effectiveVisionMode": results["vision_info"]["effective_mode"],
        "visionDiagnostics": results["vision_info"],
        "cleanupStats": {
            "inputPaths": results["cleanup_stats"].input_paths,
            "outputPaths": results["cleanup_stats"].output_paths,
            "inputPoints": results["cleanup_stats"].input_points,
            "outputPoints": results["cleanup_stats"].output_points,
            "droppedPaths": results["cleanup_stats"].dropped_paths,
            "droppedPoints": results["cleanup_stats"].dropped_points,
        },
        "stitchStats": {
            "inputPaths": results["stitch_stats"].input_paths,
            "outputPaths": results["stitch_stats"].output_paths,
            "mergedPaths": results["stitch_stats"].merged_paths,
        },
        "planMetrics": {
            "pathCount": results["plan_metrics"].path_count,
            "drawDistanceMm": round(results["plan_metrics"].draw_distance_mm, 2),
            "penUpDistanceMm": round(results["plan_metrics"].pen_up_distance_mm, 2),
            "totalDistanceMm": round(results["plan_metrics"].total_distance_mm, 2),
        },
        "paper": {
            "widthMm": args.paper_width_mm,
            "heightMm": args.paper_height_mm,
            "marginMm": args.margin_mm,
        },
        "startPose": results["start_pose"],
        "visionMode": args.vision_mode,
        "pathOrdering": args.path_ordering,
        "files": {
            "inputImageUrl": build_file_url(image_path),
            "edgesPreviewUrl": build_file_url(edges_preview_path),
            "pathPreviewUrl": build_file_url(path_preview_path),
            "commandsUrl": build_file_url(session_dir / "plot_commands.txt"),
            "summaryUrl": build_file_url(session_dir / "paths_summary.json"),
        },
    }


def process_image_request(payload: dict) -> dict:
    image_data_url = str(payload.get("imageDataUrl", ""))
    if not image_data_url:
        raise ValueError("No image uploaded")

    filename = sanitize_filename(str(payload.get("filename", "upload.png")))
    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")

    session_id = make_session_id()
    session_dir = UI_OUTPUT_ROOT / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    image_bytes, _mime = decode_data_url(image_data_url)
    image_path = session_dir / filename
    image_path.write_bytes(image_bytes)

    args = build_pipeline_args(image_path=image_path, output_dir=session_dir, options=options)
    results = pipeline_main.run_pipeline(args)
    pipeline_main.export_outputs(args, results)

    summary = summarize_results(args=args, results=results, session_dir=session_dir, image_path=image_path)
    (session_dir / "ui_request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (session_dir / "ui_result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def compare_experiments_request(payload: dict) -> dict:
    image_path = resolve_repo_path(str(payload.get("imagePath", "")))
    if not image_path.exists():
        raise ValueError("The uploaded image path is missing or invalid.")

    options = payload.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("options must be an object")

    vision_modes = [str(item) for item in payload.get("visionModes", [])]
    path_orderings = [str(item) for item in payload.get("pathOrderings", [])]
    if not vision_modes:
        raise ValueError("Select at least one vision mode for the comparison.")
    if not path_orderings:
        raise ValueError("Select at least one planning method for the comparison.")
    if "ml" in vision_modes and not str(options.get("model_checkpoint", "")).strip():
        raise ValueError("ML comparison needs a model checkpoint in Step 2.")

    experiment_dir = image_path.parent / "experiments" / make_session_id()
    experiment_dir.mkdir(parents=True, exist_ok=True)

    mask_data_url = str(payload.get("maskDataUrl", "")).strip()
    gt_mask_path: Path | None = None
    if mask_data_url:
        mask_filename = sanitize_filename(str(payload.get("maskFilename", "ground_truth_mask.png")))
        mask_bytes, _mime = decode_data_url(mask_data_url)
        gt_mask_path = experiment_dir / mask_filename
        gt_mask_path.write_bytes(mask_bytes)

    run_cards: list[dict] = []
    summary_rows: list[dict] = []

    for vision_mode in vision_modes:
        for path_ordering in path_orderings:
            label = f"{vision_mode}__{path_ordering}"
            run_dir = experiment_dir / label
            run_options = dict(options)
            run_options["vision_mode"] = vision_mode
            run_options["path_ordering"] = path_ordering

            args = build_pipeline_args(image_path=image_path, output_dir=run_dir, options=run_options)
            results = pipeline_main.run_pipeline(args)
            pipeline_main.export_outputs(args, results)
            run_summary = summarize_results(args=args, results=results, session_dir=run_dir, image_path=image_path)

            mask_metrics: dict[str, float] | None = None
            if gt_mask_path is not None:
                pred_mask = (results["edges"] > 0).astype(np.uint8)
                gt_mask = load_binary_mask(
                    mask_path=gt_mask_path,
                    target_size=(results["edges"].shape[1], results["edges"].shape[0]),
                )
                mask_metrics = compute_mask_metrics(pred_mask=pred_mask, gt_mask=gt_mask)

            run_cards.append(
                {
                    "label": label,
                    "visionMode": vision_mode,
                    "requestedVisionMode": vision_mode,
                    "effectiveVisionMode": run_summary["effectiveVisionMode"],
                    "fallbackReason": results["vision_info"].get("fallback_reason"),
                    "pathOrdering": path_ordering,
                    "numCommands": run_summary["numCommands"],
                    "numPaths": run_summary["numPaths"],
                    "drawDistanceMm": run_summary["planMetrics"]["drawDistanceMm"],
                    "penUpDistanceMm": run_summary["planMetrics"]["penUpDistanceMm"],
                    "files": {
                        "edgesPreviewUrl": run_summary["files"]["edgesPreviewUrl"],
                        "pathPreviewUrl": run_summary["files"]["pathPreviewUrl"],
                    },
                    "maskMetrics": None
                    if mask_metrics is None
                    else {
                        "precision": round(mask_metrics["precision"], 4),
                        "recall": round(mask_metrics["recall"], 4),
                        "dice": round(mask_metrics["dice"], 4),
                        "iou": round(mask_metrics["iou"], 4),
                    },
                }
            )

            summary_rows.append(
                {
                    "image": image_path.name,
                    "vision_mode": vision_mode,
                    "requested_vision_mode": vision_mode,
                    "effective_vision_mode": run_summary["effectiveVisionMode"],
                    "fallback_reason": results["vision_info"].get("fallback_reason"),
                    "path_ordering": path_ordering,
                    "mask_dice": None if mask_metrics is None else round(mask_metrics["dice"], 4),
                    "mask_iou": None if mask_metrics is None else round(mask_metrics["iou"], 4),
                    "num_commands": run_summary["numCommands"],
                    "num_paths": run_summary["numPaths"],
                    "pen_up_distance_mm": run_summary["planMetrics"]["penUpDistanceMm"],
                    "draw_distance_mm": run_summary["planMetrics"]["drawDistanceMm"],
                }
            )

    summary = build_summary(summary_rows)
    result_payload = {
        "experimentDir": str(experiment_dir),
        "groundTruthMaskPath": "" if gt_mask_path is None else str(gt_mask_path),
        "note": str(payload.get("note", "")).strip(),
        "runs": run_cards,
        "summary": summary,
    }
    (experiment_dir / "ui_compare_request.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (experiment_dir / "ui_compare_result.json").write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
    return result_payload


def send_commands_request(payload: dict) -> dict:
    commands_path = resolve_repo_path(str(payload.get("commandsPath", "")))
    mode = str(payload.get("mode", "socket"))
    prepend_ping = bool(payload.get("prependPing", True))

    commands = load_plot_commands(str(commands_path))
    session = bridge_sender.build_session(commands, prepend_ping=prepend_ping)

    if mode == "socket":
        bridge_sender.send_socket(
            session,
            host=str(payload.get("host", "127.0.0.1")),
            port=int(payload.get("port", 8765)),
            delay_ms=int(payload.get("delayMs", 20)),
            connect_attempts=int(payload.get("connectAttempts", 20)),
            retry_delay_ms=int(payload.get("retryDelayMs", 500)),
        )
        destination = f"{payload.get('host', '127.0.0.1')}:{int(payload.get('port', 8765))}"
    elif mode == "file":
        outbox = resolve_repo_path(str(payload.get("outbox", "output/bridge_commands.jsonl")))
        bridge_sender.send_file(session, str(outbox))
        destination = str(outbox)
    else:
        raise ValueError(f"Unsupported send mode: {mode}")

    return {
        "message": f"Sent {len(session)} bridge commands via {mode}.",
        "destination": destination,
        "commandSummary": bridge_sender.summarize_commands(session),
    }


def export_mcode_request(payload: dict) -> dict:
    commands_path = resolve_repo_path(str(payload.get("commandsPath", "")))
    commands = load_plot_commands(str(commands_path))

    base_name = sanitize_filename(str(payload.get("baseName", commands_path.stem)))
    session_dir = commands_path.parent
    export_dir = session_dir / "exports"
    py_path = export_dir / f"{base_name}.py"
    mcode_path = export_dir / f"{base_name}.mcode"

    actions = compile_actions(
        commands=commands,
        mm_per_straight_unit=float(payload.get("mmPerStraightUnit", DEFAULT_ROBOT_EXPORT_SETTINGS["mmPerStraightUnit"])),
        min_turn_deg=float(payload.get("minTurnDeg", 1.0)),
        min_move_mm=float(payload.get("minMoveMm", DEFAULT_ROBOT_EXPORT_SETTINGS["minMoveMm"])),
        min_straight_units=float(payload.get("minStraightUnits", DEFAULT_ROBOT_EXPORT_SETTINGS["minStraightUnits"])),
        corner_lift_turn_deg=float(payload.get("cornerLiftTurnDeg", DEFAULT_ROBOT_EXPORT_SETTINGS["cornerLiftTurnDeg"])),
        pen_forward_offset_mm=float(payload.get("penForwardOffsetMm", DEFAULT_ROBOT_EXPORT_SETTINGS["penForwardOffsetMm"])),
        pen_lateral_offset_mm=float(payload.get("penLateralOffsetMm", DEFAULT_ROBOT_EXPORT_SETTINGS["penLateralOffsetMm"])),
        start_tip_point=(
            float(payload.get("startXMm", DEFAULT_ROBOT_EXPORT_SETTINGS["startXMm"])),
            float(payload.get("startYMm", DEFAULT_ROBOT_EXPORT_SETTINGS["startYMm"])),
        ),
        start_heading_deg=float(payload.get("startHeadingDeg", DEFAULT_ROBOT_EXPORT_SETTINGS["startHeadingDeg"])),
    )
    script = render_script(
        actions=actions,
        start_button=str(payload.get("startButton", "a")),
        stop_button=str(payload.get("stopButton", "b")),
        use_pen=bool(payload.get("usePen", DEFAULT_ROBOT_EXPORT_SETTINGS["usePen"])),
        servo_port=str(payload.get("servoPort", DEFAULT_ROBOT_EXPORT_SETTINGS["servoPort"])),
        pen_lift_delta=float(payload.get("penLiftDelta", DEFAULT_ROBOT_EXPORT_SETTINGS["penLiftDelta"])),
        pen_delay_s=float(payload.get("penDelayS", DEFAULT_ROBOT_EXPORT_SETTINGS["penDelayS"])),
        turn_delay_s=float(payload.get("turnDelayS", DEFAULT_ROBOT_EXPORT_SETTINGS["turnDelayS"])),
        turn_scale=float(payload.get("turnScale", DEFAULT_ROBOT_EXPORT_SETTINGS["turnScale"])),
        start_tip_point=(
            float(payload.get("startXMm", DEFAULT_ROBOT_EXPORT_SETTINGS["startXMm"])),
            float(payload.get("startYMm", DEFAULT_ROBOT_EXPORT_SETTINGS["startYMm"])),
        ),
        start_heading_deg=float(payload.get("startHeadingDeg", DEFAULT_ROBOT_EXPORT_SETTINGS["startHeadingDeg"])),
    )
    out_py, out_mcode = write_script_outputs(script=script, output_path=py_path, mcode_output_path=mcode_path)

    return {
        "message": f"Exported {len(actions)} robot actions.",
        "pythonPath": str(out_py),
        "mcodePath": str(out_mcode) if out_mcode is not None else "",
        "files": {
            "pythonUrl": build_file_url(out_py),
            "mcodeUrl": build_file_url(out_mcode) if out_mcode is not None else "",
        },
    }


class UIServerHandler(BaseHTTPRequestHandler):
    server_version = "RobotDrawUI/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            if not UI_INDEX.exists():
                text_response(self, "UI file missing: ui/index.html", status=500)
                return
            data = UI_INDEX.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        if parsed.path == "/api/health":
            json_response(
                self,
                {
                    "ok": True,
                    "ui": str(UI_INDEX),
                    "outputRoot": str(UI_OUTPUT_ROOT),
                },
            )
            return

        if parsed.path == "/api/config":
            json_response(self, {"ok": True, "result": build_ui_config()})
            return

        if parsed.path.startswith("/files/"):
            relative_path = unquote(parsed.path[len("/files/") :])
            target = resolve_repo_path(relative_path)
            if not target.exists() or not target.is_file():
                text_response(self, "File not found", status=404)
                return
            data = target.read_bytes()
            mime_type, _encoding = mimetypes.guess_type(str(target))
            self.send_response(200)
            self.send_header("Content-Type", mime_type or "application/octet-stream")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Expires", "0")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        text_response(self, "Not found", status=404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            payload = load_request_json(self)
            if parsed.path == "/api/process":
                json_response(self, {"ok": True, "result": process_image_request(payload)})
                return

            if parsed.path == "/api/compare":
                json_response(self, {"ok": True, "result": compare_experiments_request(payload)})
                return

            if parsed.path == "/api/send":
                json_response(self, {"ok": True, "result": send_commands_request(payload)})
                return

            if parsed.path == "/api/export-mcode":
                json_response(self, {"ok": True, "result": export_mcode_request(payload)})
                return

            json_response(self, {"ok": False, "error": "Not found"}, status=404)
        except Exception as exc:
            json_response(
                self,
                {"ok": False, "error": str(exc)},
                status=HTTPStatus.BAD_REQUEST,
            )

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        print(f"UI {self.address_string()} - {format % args}")


def main() -> None:
    args = parse_args()
    UI_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    server = ThreadingHTTPServer((args.host, args.port), UIServerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"Robot drawing UI listening on: {url}")
    if args.open_browser:
        webbrowser.open(url)
    server.serve_forever()


if __name__ == "__main__":
    main()
