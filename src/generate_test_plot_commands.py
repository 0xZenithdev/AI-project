"""
Generate reusable calibration plot command files for mBot2 testing.

Supports:
- single tiny shapes for quick checks
- a standard calibration pack with pen, distance, and square tests
- optional .py/.mcode artifacts built from the same commands

Memory note:
- the standard pack order is pen_gaps -> ruler -> square
- that order matches the agreed project plan in PROJECT_MEMORY.md

Examples:
python -m src.generate_test_plot_commands --shape line --size-mm 20
python -m src.generate_test_plot_commands --pack standard --output-dir output/calibration
python -m src.generate_test_plot_commands --pack standard --emit-mcode --use-pen
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import zipfile

from src.bridge_protocol import (
    BridgeCommand,
    save_bridge_commands_as_text,
)
from src.mblock_script_generator import compile_actions, render_script


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate calibration plot command files.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--shape",
        choices=["line", "square", "square-motion", "pen-gaps", "ruler"],
        default="line",
        help="Generate one calibration shape",
    )
    group.add_argument(
        "--pack",
        choices=["standard"],
        help="Generate the full calibration pack",
    )
    parser.add_argument(
        "--size-mm",
        type=float,
        default=20.0,
        help="Line length or square side length in mm for single-shape mode",
    )
    parser.add_argument(
        "--start-x",
        type=float,
        default=20.0,
        help="Start x in mm",
    )
    parser.add_argument(
        "--start-y",
        type=float,
        default=20.0,
        help="Start y in mm",
    )
    parser.add_argument(
        "--travel-speed",
        type=float,
        default=60.0,
        help="Pen-up speed",
    )
    parser.add_argument(
        "--draw-speed",
        type=float,
        default=35.0,
        help="Pen-down speed",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/test_plot_commands.txt",
        help="Output command file for single-shape mode",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output/calibration",
        help="Output folder for pack mode",
    )
    parser.add_argument(
        "--pen-segments",
        type=int,
        default=5,
        help="Number of short pen-lift segments in the pen test",
    )
    parser.add_argument(
        "--pen-segment-mm",
        type=float,
        default=20.0,
        help="Length of each drawn dash in the pen-lift test",
    )
    parser.add_argument(
        "--pen-gap-mm",
        type=float,
        default=15.0,
        help="Gap between drawn dashes in the pen-lift test",
    )
    parser.add_argument(
        "--ruler-lengths-mm",
        type=float,
        nargs="+",
        default=[50.0, 100.0, 150.0],
        help="Line lengths used in the ruler test",
    )
    parser.add_argument(
        "--ruler-row-gap-mm",
        type=float,
        default=20.0,
        help="Vertical gap between ruler-test rows",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=100.0,
        help="Square side length used in the standard pack",
    )
    parser.add_argument(
        "--emit-mcode",
        action="store_true",
        help="Also build .py and .mcode artifacts from the generated commands",
    )
    parser.add_argument(
        "--use-pen",
        action="store_true",
        help="Enable pen control in generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--servo-port",
        type=str,
        default="S1",
        help="Servo port used for generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--pen-lift-delta",
        type=float,
        default=60.0,
        help="Pen-lift delta for generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--mm-per-straight-unit",
        type=float,
        default=10.0,
        help="Current scale assumption for generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=1.0,
        help="Current turn calibration factor for generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--pen-delay-s",
        type=float,
        default=0.3,
        help="Pause after pen movement in generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--turn-delay-s",
        type=float,
        default=0.1,
        help="Pause after turns in generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--start-button",
        type=str,
        default="a",
        choices=["a", "b"],
        help="CyberPi button that starts generated .py/.mcode artifacts",
    )
    parser.add_argument(
        "--stop-button",
        type=str,
        default="b",
        choices=["a", "b"],
        help="CyberPi button that stops generated .py/.mcode artifacts",
    )
    return parser.parse_args()


def move(x: float, y: float, speed: float) -> BridgeCommand:
    return BridgeCommand(cmd="MOVE", x=x, y=y, speed=speed)


def line_commands(
    start_x: float,
    start_y: float,
    size_mm: float,
    travel_speed: float,
    draw_speed: float,
) -> list[BridgeCommand]:
    end_x = start_x + size_mm
    return [
        BridgeCommand(cmd="PEN_UP"),
        move(start_x, start_y, travel_speed),
        BridgeCommand(cmd="PEN_DOWN"),
        move(end_x, start_y, draw_speed),
        BridgeCommand(cmd="PEN_UP"),
    ]


def square_commands(
    start_x: float,
    start_y: float,
    size_mm: float,
    travel_speed: float,
    draw_speed: float,
) -> list[BridgeCommand]:
    x = start_x
    y = start_y
    s = size_mm
    return [
        BridgeCommand(cmd="PEN_UP"),
        move(x, y, travel_speed),
        BridgeCommand(cmd="PEN_DOWN"),
        move(x + s, y, draw_speed),
        move(x + s, y + s, draw_speed),
        move(x, y + s, draw_speed),
        move(x, y, draw_speed),
        BridgeCommand(cmd="PEN_UP"),
    ]


def square_motion_commands(
    size_mm: float,
    draw_speed: float,
) -> list[BridgeCommand]:
    """
    Square path anchored at the robot's current pose.

    This avoids the diagonal travel-to-start move that makes the normal square
    harder to reason about during first hardware tests.
    """
    return [
        BridgeCommand(cmd="PEN_UP"),
        BridgeCommand(cmd="PEN_DOWN"),
        move(size_mm, 0.0, draw_speed),
        move(size_mm, size_mm, draw_speed),
        move(0.0, size_mm, draw_speed),
        move(0.0, 0.0, draw_speed),
        BridgeCommand(cmd="PEN_UP"),
    ]


def square_motion_actions(
    size_mm: float,
    mm_per_straight_unit: float,
    use_pen: bool,
) -> list[tuple[str, float | None]]:
    """
    Direct action version of the motion-square test for clearer robot behavior.

    Memory note:
    - this is intentionally calibration-focused, not the normal drawing path
    - it makes the robot do 4 straight segments and 4 explicit 90-degree turns
    """
    straight_units = round(size_mm / mm_per_straight_unit, 2)
    actions: list[tuple[str, float | None]] = []

    if use_pen:
        actions.append(("PD", None))

    for _ in range(4):
        actions.append(("ST", straight_units))
        actions.append(("TR", -90.0))

    if use_pen:
        actions.append(("PU", None))

    return actions


def pen_gap_commands(
    start_x: float,
    start_y: float,
    segment_mm: float,
    gap_mm: float,
    segments: int,
    travel_speed: float,
    draw_speed: float,
) -> list[BridgeCommand]:
    commands: list[BridgeCommand] = [BridgeCommand(cmd="PEN_UP")]

    for index in range(segments):
        seg_start_x = start_x + index * (segment_mm + gap_mm)
        seg_end_x = seg_start_x + segment_mm
        commands.extend(
            [
                move(seg_start_x, start_y, travel_speed),
                BridgeCommand(cmd="PEN_DOWN"),
                move(seg_end_x, start_y, draw_speed),
                BridgeCommand(cmd="PEN_UP"),
            ]
        )

    return commands


def ruler_commands(
    start_x: float,
    start_y: float,
    lengths_mm: list[float],
    row_gap_mm: float,
    travel_speed: float,
    draw_speed: float,
) -> list[BridgeCommand]:
    commands: list[BridgeCommand] = [BridgeCommand(cmd="PEN_UP")]

    for index, length_mm in enumerate(lengths_mm):
        row_y = start_y + index * row_gap_mm
        commands.extend(
            [
                move(start_x, row_y, travel_speed),
                BridgeCommand(cmd="PEN_DOWN"),
                move(start_x + length_mm, row_y, draw_speed),
                BridgeCommand(cmd="PEN_UP"),
            ]
        )

    return commands


def build_compiled_artifacts(
    commands: list[BridgeCommand],
    output_base: Path,
    args: argparse.Namespace,
    actions_override: list[tuple[str, float | None]] | None = None,
) -> dict[str, str]:
    if actions_override is None:
        actions = compile_actions(
            commands=commands,
            mm_per_straight_unit=args.mm_per_straight_unit,
            min_turn_deg=1.0,
            min_move_mm=1.0,
        )
    else:
        actions = actions_override
    script = render_script(
        actions=actions,
        start_button=args.start_button,
        stop_button=args.stop_button,
        use_pen=args.use_pen,
        servo_port=args.servo_port,
        pen_lift_delta=args.pen_lift_delta,
        pen_delay_s=args.pen_delay_s,
        turn_delay_s=args.turn_delay_s,
        turn_scale=args.turn_scale,
    )

    py_path = output_base.with_suffix(".py")
    mcode_path = output_base.with_suffix(".mcode")
    py_path.write_text(script, encoding="utf-8")
    with zipfile.ZipFile(mcode_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(py_path.name, script)

    return {
        "script_path": str(py_path),
        "mcode_path": str(mcode_path),
        "action_count": str(len(actions)),
    }


def write_single_shape(args: argparse.Namespace) -> None:
    actions_override: list[tuple[str, float | None]] | None = None

    if args.shape == "line":
        commands = line_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            size_mm=args.size_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        )
    elif args.shape == "square":
        commands = square_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            size_mm=args.size_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        )
    elif args.shape == "square-motion":
        commands = square_motion_commands(
            size_mm=args.size_mm,
            draw_speed=args.draw_speed,
        )
        actions_override = square_motion_actions(
            size_mm=args.size_mm,
            mm_per_straight_unit=args.mm_per_straight_unit,
            use_pen=args.use_pen,
        )
    elif args.shape == "pen-gaps":
        commands = pen_gap_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            segment_mm=args.pen_segment_mm,
            gap_mm=args.pen_gap_mm,
            segments=args.pen_segments,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        )
    else:
        commands = ruler_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            lengths_mm=args.ruler_lengths_mm,
            row_gap_mm=args.ruler_row_gap_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_bridge_commands_as_text(commands, str(out_path))

    print(f"Generated {args.shape} test commands at: {out_path}")
    print(f"Command count: {len(commands)}")

    if not args.emit_mcode:
        return

    compiled = build_compiled_artifacts(
        commands,
        out_path.with_suffix(""),
        args,
        actions_override=actions_override,
    )
    print(f"Wrote script to: {compiled['script_path']}")
    print(f"Wrote mcode to: {compiled['mcode_path']}")
    print(f"Compiled actions: {compiled['action_count']}")


def write_standard_pack(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pack: dict[str, list[BridgeCommand]] = {
        "pen_gaps": pen_gap_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            segment_mm=args.pen_segment_mm,
            gap_mm=args.pen_gap_mm,
            segments=args.pen_segments,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        ),
        "ruler": ruler_commands(
            start_x=args.start_x,
            start_y=args.start_y + 30.0,
            lengths_mm=args.ruler_lengths_mm,
            row_gap_mm=args.ruler_row_gap_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        ),
        "square": square_commands(
            start_x=args.start_x,
            start_y=args.start_y + 120.0,
            size_mm=args.square_size_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        ),
    }

    summary: dict[str, object] = {
        "recommended_order": ["pen_gaps", "ruler", "square"],
        "current_assumptions": {
            "pen_lift_delta": args.pen_lift_delta,
            "mm_per_straight_unit": args.mm_per_straight_unit,
            "turn_scale": args.turn_scale,
            "servo_port": args.servo_port,
            "use_pen": args.use_pen,
        },
        "tests": {
            "pen_gaps": {
                "purpose": "Increase PEN_LIFT_DELTA until the dashes are cleanly separated with no connecting marks.",
                "segments": args.pen_segments,
                "segment_mm": args.pen_segment_mm,
                "gap_mm": args.pen_gap_mm,
                "commands_file": str(output_dir / "calibration_pen_gaps.txt"),
            },
            "ruler": {
                "purpose": "Measure each line and feed the results into src.calibration_report to refine MM_PER_STRAIGHT_UNIT.",
                "lengths_mm": args.ruler_lengths_mm,
                "commands_file": str(output_dir / "calibration_ruler.txt"),
            },
            "square": {
                "purpose": "Use the square to judge turn accuracy and square closure, then refine TURN_SCALE.",
                "size_mm": args.square_size_mm,
                "commands_file": str(output_dir / "calibration_square.txt"),
            },
        },
    }

    for name, commands in pack.items():
        txt_path = output_dir / f"calibration_{name}.txt"
        save_bridge_commands_as_text(commands, str(txt_path))
        print(f"Wrote {name} commands to: {txt_path}")

        if args.emit_mcode:
            compiled = build_compiled_artifacts(commands, output_dir / f"calibration_{name}", args)
            summary["tests"][name]["script_path"] = compiled["script_path"]  # type: ignore[index]
            summary["tests"][name]["mcode_path"] = compiled["mcode_path"]  # type: ignore[index]
            summary["tests"][name]["action_count"] = int(compiled["action_count"])  # type: ignore[index]

    summary_path = output_dir / "calibration_pack_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote calibration summary to: {summary_path}")


def main() -> None:
    args = parse_args()
    if args.pack == "standard":
        write_standard_pack(args)
        return
    write_single_shape(args)


if __name__ == "__main__":
    main()
