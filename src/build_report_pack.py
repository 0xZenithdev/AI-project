"""
Build report-ready tables, captions, and narrative text from current project outputs.

Example:
python -m src.build_report_pack `
  --planning-experiment-dir output/ai_experiments/20260419_174251_476268 `
  --comparison-experiment-dir output/ai_experiments/20260419_174449_463196 `
  --ui-session-dir output/ui_sessions/20260420_034558_126123
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a report-ready pack from experiment outputs.")
    parser.add_argument(
        "--planning-experiment-dir",
        required=True,
        help="Experiment directory used for the planning comparison.",
    )
    parser.add_argument(
        "--comparison-experiment-dir",
        required=True,
        help="Experiment directory used for the perception comparison.",
    )
    parser.add_argument(
        "--ui-session-dir",
        default="",
        help="Optional UI session directory used for preview figure paths.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/report_pack_current",
        help="Directory where the report pack will be written.",
    )
    parser.add_argument(
        "--report-path-ordering",
        default="two_opt",
        help="Planning method to keep for the perception table.",
    )
    parser.add_argument(
        "--project-title",
        default="Image-to-Path Robot Drawing with Learned Perception and Path Optimization",
        help="Title used in the generated report text.",
    )
    return parser.parse_args()


def ensure_exists(path_str: str) -> Path:
    path = Path(path_str)
    if not path.is_absolute():
        path = REPO_ROOT / path
    path = path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    return path


def load_payload(experiment_dir: Path) -> dict:
    payload_path = experiment_dir / "experiment_results.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing experiment_results.json in: {experiment_dir}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def to_rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def resolve_run_dir(output_dir_value: str) -> Path:
    path = Path(output_dir_value)
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def fmt(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def select_planning_rows(payload: dict) -> list[dict]:
    return sorted(
        payload["results"],
        key=lambda row: (row["image"].lower(), row["vision_mode"], row["path_ordering"]),
    )


def select_perception_rows(payload: dict, path_ordering: str) -> list[dict]:
    filtered = [row for row in payload["results"] if row["path_ordering"] == path_ordering]
    rows = filtered if filtered else payload["results"]
    return sorted(rows, key=lambda row: (row["image"].lower(), row["vision_mode"]))


def build_planning_table_rows(payload: dict) -> list[dict]:
    rows: list[dict] = []
    improvements = {
        (item["image"], item["vision_mode"]): item
        for item in payload["summary"].get("planning_comparisons", [])
    }
    for row in select_planning_rows(payload):
        improvement = improvements.get((row["image"], row["vision_mode"]))
        observation = "Baseline route"
        if row["path_ordering"] == "two_opt" and improvement:
            observation = (
                f"Reduced pen-up travel by {fmt(improvement['pen_up_improvement_mm'])} mm "
                f"({fmt(improvement['pen_up_improvement_pct'])}%)"
            )
        rows.append(
            {
                "image": row["image"],
                "vision_mode": row["vision_mode"],
                "path_ordering": row["path_ordering"],
                "num_paths": row["num_paths"],
                "num_commands": row["num_commands"],
                "draw_distance_mm": row["draw_distance_mm"],
                "pen_up_distance_mm": row["pen_up_distance_mm"],
                "observation": observation,
            }
        )
    return rows


def build_perception_table_rows(payload: dict, path_ordering: str) -> list[dict]:
    comparisons = {
        (item["image"], item["path_ordering"]): item
        for item in payload["summary"].get("vision_comparisons", [])
    }
    rows: list[dict] = []
    for row in select_perception_rows(payload, path_ordering):
        comparison = comparisons.get((row["image"], row["path_ordering"]))
        if row["vision_mode"] == "classical":
            observation = "Reference baseline for this image"
        elif row["num_commands"] <= 1 and row["num_paths"] == 0:
            observation = "Current ML checkpoint collapsed to an empty or near-empty output"
        elif comparison and comparison["mask_dice_delta"] is not None:
            observation = f"ML Dice delta versus classical: {comparison['mask_dice_delta']}"
        else:
            observation = "ML output available but not yet stronger than the baseline"

        rows.append(
            {
                "image": row["image"],
                "path_ordering": row["path_ordering"],
                "vision_mode": row["vision_mode"],
                "mask_dice": row["mask_dice"],
                "mask_iou": row["mask_iou"],
                "num_paths": row["num_paths"],
                "num_commands": row["num_commands"],
                "pen_up_distance_mm": row["pen_up_distance_mm"],
                "observation": observation,
            }
        )
    return rows


def build_robot_validation_rows() -> list[dict]:
    return [
        {
            "robot_test": "square-motion 100 mm",
            "expected_result": "Closed square with balanced straight segments and right-angle turns",
            "observed_result": "Movement-only square test was reported as perfect during physical validation",
            "conclusion": "Current MM_PER_STRAIGHT_UNIT = 10.0 remains acceptable; pen-mount exports currently use TURN_SCALE = 1.04",
        },
        {
            "robot_test": "compiled mBlock export workflow",
            "expected_result": "Reliable execution path from generated commands to CyberPi/mBlock",
            "observed_result": "Compiled Python/.mcode workflow worked and became the recommended execution path",
            "conclusion": "Keep compiled export as the main robot workflow for the course demo",
        },
        {
            "robot_test": "pen-lift and final paper drawing",
            "expected_result": "Physical validation of pen-up / pen-down drawing quality",
            "observed_result": "Deferred because the pen and lift servo were not available during the current testing phase",
            "conclusion": "Mark as pending hardware-dependent validation",
        },
    ]


def build_figure_captions(
    planning_payload: dict,
    comparison_payload: dict,
    ui_session_dir: Path | None,
    output_dir: Path,
    path_ordering: str,
) -> str:
    planning_best = next(
        (
            item
            for item in planning_payload["summary"].get("planning_comparisons", [])
            if item["vision_mode"] == "classical"
        ),
        None,
    )
    perception_rows = select_perception_rows(comparison_payload, path_ordering)
    classical_row = next((row for row in perception_rows if row["image"] == "test2logo.webp" and row["vision_mode"] == "classical"), None)
    ml_row = next((row for row in perception_rows if row["image"] == "test2logo.webp" and row["vision_mode"] == "ml"), None)

    lines = [
        "# Figure Captions",
        "",
        "## Figure 1",
        "Browser-based interface used to upload an image, compare methods, review previews, and export robot-ready files.",
        "",
        "Suggested placement: Methodology",
        "",
    ]

    if ui_session_dir is not None:
        original = ui_session_dir / "square.png"
        edges = ui_session_dir / "edges_preview.png"
        preview = ui_session_dir / "path_preview.png"
        lines.extend(
            [
                "## Figure 2",
                "Example original input image used in the browser workflow before perception and planning.",
                "",
                f"Artifact path: `{to_rel(original, output_dir)}`",
                "",
                "## Figure 3",
                "Classical line-extraction preview generated by the system for the uploaded square test image.",
                "",
                f"Artifact path: `{to_rel(edges, output_dir)}`",
                "",
                "## Figure 4",
                "Planned drawing preview produced after path ordering. This preview is used to check whether the robot-ready path looks correct before export.",
                "",
                f"Artifact path: `{to_rel(preview, output_dir)}`",
                "",
            ]
        )

    if planning_best is not None:
        lines.extend(
            [
                "## Figure 5",
                (
                    f"Planning comparison on {planning_best['image']} using classical perception. "
                    f"The two_opt strategy reduced pen-up travel by {fmt(planning_best['pen_up_improvement_mm'])} mm "
                    f"({fmt(planning_best['pen_up_improvement_pct'])}%) relative to nearest_neighbor while keeping the same drawing content."
                ),
                "",
                "Suggested placement: Experimentations and Results",
                "",
            ]
        )

    if classical_row is not None and ml_row is not None:
        lines.extend(
            [
                "## Figure 6",
                (
                    f"Perception comparison on {classical_row['image']} under {path_ordering} planning. "
                    f"The classical baseline produced {classical_row['num_commands']} commands with Dice {fmt(classical_row['mask_dice'], 4)}, "
                    f"whereas the current ML checkpoint produced {ml_row['num_commands']} command and Dice {fmt(ml_row['mask_dice'], 4)}."
                ),
                "",
                "Suggested placement: Experimentations and Results",
                "",
            ]
        )

    lines.extend(
        [
            "## Figure 7",
            "mBot2 / CyberPi hardware setup used for physical validation of the generated robot commands.",
            "",
            "Suggested placement: Methodology",
            "Insert your own robot setup photo here.",
            "",
            "## Figure 8",
            "Movement-only square-motion validation used to verify straight movement and turn consistency on the physical robot.",
            "",
            "Suggested placement: Experimentations and Results",
            "Insert your square-motion robot photo here.",
            "",
            "## Figure 9",
            "Final robot drawing result on paper, accompanied by a short supplementary demonstration video if available.",
            "",
            "Suggested placement: Experimentations and Results",
            "Insert your final output photo or a QR code linking to the demo video here.",
            "",
        ]
    )
    return "\n".join(lines)


def build_results_narrative(
    title: str,
    planning_payload: dict,
    comparison_payload: dict,
    path_ordering: str,
) -> str:
    planning_best = next(
        (
            item
            for item in planning_payload["summary"].get("planning_comparisons", [])
            if item["vision_mode"] == "classical"
        ),
        None,
    )
    perception_rows = select_perception_rows(comparison_payload, path_ordering)
    classical_row = next((row for row in perception_rows if row["image"] == "test2logo.webp" and row["vision_mode"] == "classical"), None)
    ml_row = next((row for row in perception_rows if row["image"] == "test2logo.webp" and row["vision_mode"] == "ml"), None)

    lines = [
        f"# Results Narrative for {title}",
        "",
        "## Planning Comparison Paragraph",
        "",
    ]

    if planning_best is not None:
        lines.append(
            (
                f"The planning experiment compared nearest_neighbor and two_opt while keeping the perception mode fixed. "
                f"On {planning_best['image']}, the two_opt method reduced pen-up travel by {fmt(planning_best['pen_up_improvement_mm'])} mm, "
                f"which corresponds to an improvement of {fmt(planning_best['pen_up_improvement_pct'])}%. "
                f"This result shows that the optimization stage improves the execution efficiency of the drawing process "
                f"without changing the extracted visual content."
            )
        )
    else:
        lines.append("Fill this paragraph after the planning comparison is finalized.")

    lines.extend(
        [
            "",
            f"## Perception Comparison Paragraph ({path_ordering})",
            "",
        ]
    )

    if classical_row is not None and ml_row is not None:
        lines.append(
            (
                f"For the perception comparison under {path_ordering} planning, the classical baseline remained stronger than the current ML checkpoint. "
                f"On {classical_row['image']}, the classical method produced {classical_row['num_commands']} commands across {classical_row['num_paths']} paths, "
                f"with a mask Dice score of {fmt(classical_row['mask_dice'], 4)} and IoU of {fmt(classical_row['mask_iou'], 4)}. "
                f"In contrast, the ML checkpoint collapsed to {ml_row['num_commands']} command and {ml_row['num_paths']} paths, with Dice {fmt(ml_row['mask_dice'], 4)} "
                f"and IoU {fmt(ml_row['mask_iou'], 4)}. This indicates that the ML pipeline is structurally complete but still needs additional training data and retraining "
                f"before it can outperform the classical baseline."
            )
        )
    else:
        lines.append("Fill this paragraph after the perception comparison is finalized.")

    lines.extend(
        [
            "",
            "## Robot Validation Paragraph",
            "",
            (
                "The physical validation stage focused first on robot movement consistency. "
                "A movement-only square-motion test was executed and reported as successful, which supported the current calibration assumptions "
                "for straight movement and turning. The compiled Python/.mcode workflow also proved to be the most reliable execution path for the robot. "
                "At the same time, pen-lift-specific drawing validation was deferred because the pen and lift servo were not available during this testing phase."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def build_markdown_tables(
    planning_rows: list[dict],
    perception_rows: list[dict],
    robot_rows: list[dict],
) -> str:
    lines = [
        "# Report Tables",
        "",
        "## Table 1 - Planning Comparison",
        "",
        markdown_table(
            ["Image", "Perception", "Planning", "Paths", "Commands", "Pen-up Travel (mm)", "Observation"],
            [
                [
                    row["image"],
                    row["vision_mode"],
                    row["path_ordering"],
                    str(row["num_paths"]),
                    str(row["num_commands"]),
                    fmt(row["pen_up_distance_mm"]),
                    row["observation"],
                ]
                for row in planning_rows
            ],
        ),
        "",
        "## Table 2 - Perception Comparison",
        "",
        markdown_table(
            ["Image", "Planning", "Perception", "Dice", "IoU", "Paths", "Commands", "Observation"],
            [
                [
                    row["image"],
                    row["path_ordering"],
                    row["vision_mode"],
                    fmt(row["mask_dice"], 4),
                    fmt(row["mask_iou"], 4),
                    str(row["num_paths"]),
                    str(row["num_commands"]),
                    row["observation"],
                ]
                for row in perception_rows
            ],
        ),
        "",
        "## Table 3 - Robot Validation",
        "",
        markdown_table(
            ["Robot Test", "Expected Result", "Observed Result", "Conclusion"],
            [
                [
                    row["robot_test"],
                    row["expected_result"],
                    row["observed_result"],
                    row["conclusion"],
                ]
                for row in robot_rows
            ],
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    planning_dir = ensure_exists(args.planning_experiment_dir)
    comparison_dir = ensure_exists(args.comparison_experiment_dir)
    ui_session_dir = ensure_exists(args.ui_session_dir) if args.ui_session_dir else None
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = REPO_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    planning_payload = load_payload(planning_dir)
    comparison_payload = load_payload(comparison_dir)

    planning_rows = build_planning_table_rows(planning_payload)
    perception_rows = build_perception_table_rows(comparison_payload, args.report_path_ordering)
    robot_rows = build_robot_validation_rows()

    write_csv(
        output_dir / "planning_comparison_table.csv",
        [
            "image",
            "vision_mode",
            "path_ordering",
            "num_paths",
            "num_commands",
            "draw_distance_mm",
            "pen_up_distance_mm",
            "observation",
        ],
        planning_rows,
    )
    write_csv(
        output_dir / "perception_comparison_table.csv",
        [
            "image",
            "path_ordering",
            "vision_mode",
            "mask_dice",
            "mask_iou",
            "num_paths",
            "num_commands",
            "pen_up_distance_mm",
            "observation",
        ],
        perception_rows,
    )
    write_csv(
        output_dir / "robot_validation_table.csv",
        ["robot_test", "expected_result", "observed_result", "conclusion"],
        robot_rows,
    )

    (output_dir / "report_tables.md").write_text(
        build_markdown_tables(planning_rows, perception_rows, robot_rows),
        encoding="utf-8",
    )
    (output_dir / "figure_captions.md").write_text(
        build_figure_captions(
            planning_payload=planning_payload,
            comparison_payload=comparison_payload,
            ui_session_dir=ui_session_dir,
            output_dir=output_dir,
            path_ordering=args.report_path_ordering,
        ),
        encoding="utf-8",
    )
    (output_dir / "results_narrative.md").write_text(
        build_results_narrative(
            title=args.project_title,
            planning_payload=planning_payload,
            comparison_payload=comparison_payload,
            path_ordering=args.report_path_ordering,
        ),
        encoding="utf-8",
    )

    summary = {
        "project_title": args.project_title,
        "planning_experiment_dir": str(planning_dir),
        "comparison_experiment_dir": str(comparison_dir),
        "ui_session_dir": str(ui_session_dir) if ui_session_dir else "",
        "report_path_ordering": args.report_path_ordering,
        "outputs": {
            "planning_table_csv": str(output_dir / "planning_comparison_table.csv"),
            "perception_table_csv": str(output_dir / "perception_comparison_table.csv"),
            "robot_validation_table_csv": str(output_dir / "robot_validation_table.csv"),
            "report_tables_md": str(output_dir / "report_tables.md"),
            "figure_captions_md": str(output_dir / "figure_captions.md"),
            "results_narrative_md": str(output_dir / "results_narrative.md"),
        },
    }
    (output_dir / "report_pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote report pack to: {output_dir}")
    print(f"Tables: {output_dir / 'report_tables.md'}")
    print(f"Captions: {output_dir / 'figure_captions.md'}")
    print(f"Narrative: {output_dir / 'results_narrative.md'}")


if __name__ == "__main__":
    main()
