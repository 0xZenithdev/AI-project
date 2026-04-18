"""
Local web UI for the robot-drawing pipeline.

This keeps the image upload and preview experience on the PC side while reusing
the same pipeline, bridge sender, and mBlock export logic as the CLI tools.

Run:
python -m src.ui_server
python -m src.ui_server --port 8010 --open-browser
"""

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
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    handler.wfile.write(data)


def text_response(handler: BaseHTTPRequestHandler, payload: str, status: int = 200) -> None:
    data = payload.encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "text/plain; charset=utf-8")
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

    Memory note:
    - the UI preview is for operator confidence, not geometric truth
    - do not make this more aggressive than the real pipeline output
    """
    px_per_mm = 4
    width_px = max(1, int(round(paper_width_mm * px_per_mm)))
    height_px = max(1, int(round(paper_height_mm * px_per_mm)))

    canvas = np.full((height_px, width_px, 3), (240, 236, 228), dtype=np.uint8)
    border = (194, 182, 162)
    grid = (226, 219, 206)
    ink = (35, 55, 88)
    accent = (179, 95, 67)

    cv2.rectangle(canvas, (0, 0), (width_px - 1, height_px - 1), border, 2)

    step = 10 * px_per_mm
    for x in range(step, width_px, step):
        cv2.line(canvas, (x, 0), (x, height_px - 1), grid, 1, cv2.LINE_AA)
    for y in range(step, height_px, step):
        cv2.line(canvas, (0, y), (width_px - 1, y), grid, 1, cv2.LINE_AA)

    margin_px = int(round(margin_mm * px_per_mm))
    cv2.rectangle(
        canvas,
        (margin_px, margin_px),
        (width_px - margin_px, height_px - margin_px),
        (210, 201, 186),
        1,
    )

    for path in paths_mm:
        if len(path) < 2:
            continue
        points = np.array(
            [[int(round(x * px_per_mm)), int(round(y * px_per_mm))] for x, y in path],
            dtype=np.int32,
        )
        cv2.polylines(canvas, [points], False, ink, 2, cv2.LINE_AA)
        cv2.circle(canvas, tuple(points[0]), 3, accent, -1, cv2.LINE_AA)

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
        "edgesPreviewPath": str(session_dir / "edges_preview.png"),
        "pathPreviewPath": str(path_preview_path),
        "numPaths": len(results["paths_mm"]),
        "numCommands": len(results["commands"]),
        "cleanupStats": {
            "inputPaths": results["cleanup_stats"].input_paths,
            "outputPaths": results["cleanup_stats"].output_paths,
            "inputPoints": results["cleanup_stats"].input_points,
            "outputPoints": results["cleanup_stats"].output_points,
            "droppedPaths": results["cleanup_stats"].dropped_paths,
            "droppedPoints": results["cleanup_stats"].dropped_points,
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
        "visionMode": args.vision_mode,
        "files": {
            "inputImageUrl": build_file_url(image_path),
            "edgesPreviewUrl": build_file_url(session_dir / "edges_preview.png"),
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
        mm_per_straight_unit=float(payload.get("mmPerStraightUnit", 10.0)),
        min_turn_deg=float(payload.get("minTurnDeg", 1.0)),
        min_move_mm=float(payload.get("minMoveMm", 1.0)),
    )
    script = render_script(
        actions=actions,
        start_button=str(payload.get("startButton", "a")),
        stop_button=str(payload.get("stopButton", "b")),
        use_pen=bool(payload.get("usePen", False)),
        servo_port=str(payload.get("servoPort", "S1")),
        pen_lift_delta=float(payload.get("penLiftDelta", 60.0)),
        pen_delay_s=float(payload.get("penDelayS", 0.3)),
        turn_delay_s=float(payload.get("turnDelayS", 0.1)),
        turn_scale=float(payload.get("turnScale", 1.0)),
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
