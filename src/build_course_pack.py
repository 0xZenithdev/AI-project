"""
Build a course-facing presentation/report pack from experiment outputs.

This turns one experiment directory into slide-ready artifacts so the project
can be presented as perception + planning + action with concrete results.

Example:
python -m src.build_course_pack --experiment-dir output/ai_experiments_smoke/20260417_143953_692363
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a course presentation pack from experiment outputs.")
    parser.add_argument(
        "--experiment-dir",
        type=str,
        required=True,
        help="Directory containing experiment_results.json from src.evaluate_pipeline",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Optional output directory; defaults to <experiment-dir>/course_pack",
    )
    parser.add_argument(
        "--project-title",
        type=str,
        default="AI-Powered Robot Drawing System",
        help="Title used in the generated pack",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="",
        help="Optional author/team name included in the generated overview",
    )
    return parser.parse_args()


def ensure_exists(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved}")
    return resolved


def load_experiment_payload(experiment_dir: Path) -> dict:
    payload_path = experiment_dir / "experiment_results.json"
    if not payload_path.exists():
        raise FileNotFoundError(f"Missing experiment_results.json in: {experiment_dir}")
    return json.loads(payload_path.read_text(encoding="utf-8"))


def to_rel(path: Path, base: Path) -> str:
    try:
        return os.path.relpath(path.resolve(), base.resolve()).replace("\\", "/")
    except ValueError:
        return str(path.resolve())


def format_metric(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, int):
        return str(value)
    return f"{value:.{digits}f}"


def find_run_preview_paths(base_dir: Path, row: dict) -> dict[str, str]:
    run_dir = Path(row["output_dir"])
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    edges = run_dir / "edges_preview.png"
    paths = run_dir / "path_preview.png"
    commands = run_dir / "plot_commands.txt"
    return {
        "run_dir": to_rel(run_dir, base_dir),
        "edges_preview": to_rel(edges, base_dir) if edges.exists() else "",
        "path_preview": to_rel(paths, base_dir) if paths.exists() else "",
        "commands": to_rel(commands, base_dir) if commands.exists() else "",
    }


def build_highlights(payload: dict) -> dict:
    rows = payload["results"]
    summary = payload["summary"]

    best_pen_row = min(rows, key=lambda row: row["pen_up_distance_mm"])
    best_dice_row = None
    dice_rows = [row for row in rows if row["mask_dice"] is not None]
    if dice_rows:
        best_dice_row = max(dice_rows, key=lambda row: row["mask_dice"])

    best_planning = summary["planning_comparisons"][0] if summary["planning_comparisons"] else None
    best_vision = None
    if summary["vision_comparisons"]:
        comparable = [
            item
            for item in summary["vision_comparisons"]
            if item["mask_dice_delta"] is not None
        ]
        if comparable:
            best_vision = max(comparable, key=lambda item: item["mask_dice_delta"])

    return {
        "best_pen_row": best_pen_row,
        "best_dice_row": best_dice_row,
        "best_planning": best_planning,
        "best_vision": best_vision,
    }


def write_results_table(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "image",
        "vision_mode",
        "path_ordering",
        "num_paths",
        "num_commands",
        "draw_distance_mm",
        "pen_up_distance_mm",
        "total_distance_mm",
        "mask_dice",
        "mask_iou",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def write_architecture_diagram(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "flowchart LR",
                '    A[Input image] --> B[Perception]',
                '    B --> B1[classical: Canny + contours]',
                '    B --> B2[ml: line segmentation model]',
                '    B1 --> C[Drawable paths]',
                '    B2 --> C[Drawable paths]',
                '    C --> D[Planning]',
                '    D --> D1[nearest_neighbor baseline]',
                '    D --> D2[two_opt refinement]',
                '    D1 --> E[Robot command generation]',
                '    D2 --> E[Robot command generation]',
                '    E --> F[mBlock script or live bridge]',
                '    F --> G[mBot2 draws output]',
                '    H[Evaluation] --> H1[mask Dice / IoU]',
                '    H --> H2[path + command counts]',
                '    H --> H3[pen-up travel distance]',
                '    H --> H4[robot quality scorecard]',
                "    B -. feeds .-> H",
                "    D -. feeds .-> H",
                "    G -. feeds .-> H",
            ]
        ),
        encoding="utf-8",
    )


def write_overview(
    path: Path,
    payload: dict,
    highlights: dict,
    pack_dir: Path,
    title: str,
    author: str,
) -> None:
    summary = payload["summary"]
    best_pen = highlights["best_pen_row"]
    best_planning = highlights["best_planning"]
    best_dice = highlights["best_dice_row"]

    author_line = f"- Team / author: {author}" if author else ""
    lines = [
        f"# {title}",
        "",
        "## One-sentence pitch",
        "",
        "A perception + planning + action pipeline that turns an input image into robot drawing behavior,",
        "compares classical computer vision against learned perception, and measures planning quality through",
        "travel reduction and drawing complexity metrics.",
        "",
        "## Project framing",
        "",
        "- Perception: extract drawable structure from an image",
        "- Planning: order paths to reduce unnecessary pen-up movement",
        "- Action: convert the plan into mBot2 commands through mBlock/CyberPi",
        "",
        "## Experiment snapshot",
        "",
        f"- Images evaluated: {payload['summary']['image_count']}",
        f"- Total runs: {payload['summary']['run_count']}",
        f"- Best current route: {best_pen['vision_mode']} + {best_pen['path_ordering']}",
        f"- Best current pen-up travel: {format_metric(best_pen['pen_up_distance_mm'])} mm",
    ]

    if author_line:
        lines.insert(lines.index(f"- Best current route: {best_pen['vision_mode']} + {best_pen['path_ordering']}"), author_line)

    if best_planning is not None:
        lines.extend(
            [
                f"- Strongest planning win so far: {best_planning['pen_up_improvement_mm']} mm "
                f"({best_planning['pen_up_improvement_pct']}%) improvement from two_opt",
            ]
        )

    if best_dice is not None:
        lines.extend(
            [
                f"- Best current ML mask Dice: {format_metric(best_dice['mask_dice'], 4)} "
                f"on {best_dice['image']} with {best_dice['path_ordering']}",
            ]
        )
    else:
        lines.extend(
            [
                "- ML mask-quality comparison is supported, but this experiment set does not include a ground-truth mask yet.",
            ]
        )

    lines.extend(
        [
            "",
            "## Included pack files",
            "",
            f"- Slide outline: `{to_rel(pack_dir / 'slide_outline.md', pack_dir)}`",
            f"- Speaker notes: `{to_rel(pack_dir / 'speaker_notes.md', pack_dir)}`",
            f"- Results table: `{to_rel(pack_dir / 'presentation_results_table.csv', pack_dir)}`",
            f"- Architecture diagram: `{to_rel(pack_dir / 'architecture_diagram.mmd', pack_dir)}`",
            "",
            "## Key message for the course",
            "",
            "This is not only a robotics demo. The AI lives in the perception choices, the search/planning choices,",
            "and the measurable evaluation of those choices on the same inputs.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_slide_outline(path: Path, payload: dict, highlights: dict, experiment_dir: Path) -> None:
    best_pen = highlights["best_pen_row"]
    preview_paths = find_run_preview_paths(path.parent, best_pen)
    visual_path = preview_paths["path_preview"] or preview_paths["edges_preview"] or preview_paths["run_dir"]

    lines = [
        "# Slide Outline",
        "",
        "## Slide 1 - Problem",
        "- Goal: convert an input image into efficient robot drawing behavior.",
        "- Motivation: make the project clearly AI-oriented instead of only a robot control demo.",
        "",
        "## Slide 2 - System Framing",
        "- Present the pipeline as perception + planning + action.",
        "- Show `architecture_diagram.mmd`.",
        "",
        "## Slide 3 - Perception",
        "- Classical baseline: Canny + contour extraction.",
        "- ML option: line-mask segmentation model.",
        "- Show one edge/mask preview from the experiment outputs.",
        "",
        "## Slide 4 - Planning",
        "- Baseline: nearest-neighbor greedy ordering.",
        "- Improved method: two_opt TSP-style refinement.",
        "- Objective: reduce pen-up travel distance while preserving the drawing.",
        "",
        "## Slide 5 - Evaluation",
        "- Automatic metrics: mask Dice/IoU, path count, command count, draw distance, pen-up distance.",
        "- Manual metrics: robot fidelity, cleanliness, completion.",
        "- Show `presentation_results_table.csv` and mention `robot_quality_scorecard.csv`.",
        "",
        "## Slide 6 - Current Result",
        f"- Best current route: {best_pen['vision_mode']} + {best_pen['path_ordering']}.",
        f"- Pen-up travel: {format_metric(best_pen['pen_up_distance_mm'])} mm.",
    ]

    if highlights["best_planning"] is not None:
        best_planning = highlights["best_planning"]
        lines.append(
            f"- Planning improvement: {best_planning['pen_up_improvement_mm']} mm "
            f"({best_planning['pen_up_improvement_pct']}%) on {best_planning['image']}."
        )

    lines.extend(
        [
            f"- Show preview artifact: `{visual_path}`",
            "",
            "## Slide 7 - UI + Workflow",
            "- Show the local UI wizard and the comparison workspace.",
            "- Explain that the same UI can export manual mBlock scripts or use the live bridge.",
            "",
            "## Slide 8 - Limitations and Next Steps",
            "- Train ML on a real paired dataset and rerun classical vs ML comparisons.",
            "- Add more images and filled robot scorecards.",
            "- Optionally test stronger planners or richer reward functions.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_speaker_notes(path: Path, payload: dict, highlights: dict) -> None:
    best_pen = highlights["best_pen_row"]
    best_planning = highlights["best_planning"]
    best_vision = highlights["best_vision"]

    lines = [
        "# Speaker Notes",
        "",
        "## Slide 1",
        "This project asks a simple question: how can we take a normal image and turn it into robot drawing behavior that is both correct and efficient?",
        "",
        "## Slide 2",
        "The AI framing is perception, planning, and action. That lets us explain exactly where the intelligence is instead of claiming the whole stack is AI.",
        "",
        "## Slide 3",
        "For perception, we support a classical baseline and an ML alternative. That means the project can compare hand-engineered vision against learned vision on the same inputs.",
        "",
        "## Slide 4",
        "For planning, we start with a greedy nearest-neighbor baseline and then improve it with two_opt local search. That makes the search-algorithm part explicit and measurable.",
        "",
        "## Slide 5",
        "Evaluation matters because we want more than a pretty demo. We measure path complexity, command count, travel distance, and optionally mask quality when ground truth exists.",
        "",
        "## Slide 6",
        f"In the current smoke experiment, the best route is {best_pen['vision_mode']} + {best_pen['path_ordering']} "
        f"with {format_metric(best_pen['pen_up_distance_mm'])} mm of pen-up travel.",
    ]

    if best_planning is not None:
        lines.append(
            f"The strongest planning result so far is a {best_planning['pen_up_improvement_pct']}% reduction "
            f"in pen-up travel from two_opt."
        )

    if best_vision is not None:
        lines.append(
            f"On perception, the strongest current ML Dice improvement is {best_vision['mask_dice_delta']} "
            f"under {best_vision['path_ordering']} planning."
        )
    else:
        lines.append(
            "The ML pipeline is ready, but the next important step is training on a real paired dataset so the perception comparison is as strong as the planning comparison."
        )

    lines.extend(
        [
            "",
            "## Slide 7",
            "The UI is useful for the course because it makes the experiment process visible: upload, generate, compare, then either export a manual mBlock script or send through the live bridge.",
            "",
            "## Slide 8",
            "The honest limitation is that the software stack is ahead of the dataset. The next milestone is gathering and curating paired image-mask data, then rerunning the same evaluation pipeline.",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def write_pack_summary(path: Path, payload: dict, highlights: dict) -> None:
    summary = {
        "run_count": payload["summary"]["run_count"],
        "image_count": payload["summary"]["image_count"],
        "best_pen_route": {
            "vision_mode": highlights["best_pen_row"]["vision_mode"],
            "path_ordering": highlights["best_pen_row"]["path_ordering"],
            "pen_up_distance_mm": highlights["best_pen_row"]["pen_up_distance_mm"],
        },
        "best_planning_comparison": highlights["best_planning"],
        "best_vision_comparison": highlights["best_vision"],
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    experiment_dir = ensure_exists(Path(args.experiment_dir))
    payload = load_experiment_payload(experiment_dir)

    output_dir = Path(args.output_dir) if args.output_dir else (experiment_dir / "course_pack")
    output_dir.mkdir(parents=True, exist_ok=True)

    highlights = build_highlights(payload)

    write_architecture_diagram(output_dir / "architecture_diagram.mmd")
    write_results_table(output_dir / "presentation_results_table.csv", payload["results"])
    write_overview(
        output_dir / "overview.md",
        payload=payload,
        highlights=highlights,
        pack_dir=output_dir,
        title=args.project_title,
        author=args.author,
    )
    write_slide_outline(
        output_dir / "slide_outline.md",
        payload=payload,
        highlights=highlights,
        experiment_dir=experiment_dir,
    )
    write_speaker_notes(output_dir / "speaker_notes.md", payload=payload, highlights=highlights)
    write_pack_summary(output_dir / "pack_summary.json", payload=payload, highlights=highlights)

    print(f"Wrote course pack to: {output_dir}")
    print(f"Overview: {output_dir / 'overview.md'}")
    print(f"Slides: {output_dir / 'slide_outline.md'}")
    print(f"Notes: {output_dir / 'speaker_notes.md'}")


if __name__ == "__main__":
    main()
