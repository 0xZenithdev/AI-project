"""
Generate a self-contained CyberPi/mBot2 Python script from plot commands.

This is the offline/downloadable path:
- local PC runs the vision/planning pipeline
- local PC compiles coordinates into primitive robot actions
- generated script can be imported into mBlock and downloaded to CyberPi

The generated script intentionally uses a very small runtime API surface:
- import event, time, cyberpi, mbot2
- @event.start
- @event.is_press("a") / @event.is_press("b")
- mbot2.turn(...)
- mbot2.straight(...)
- mbot2.servo_add(...)
- mbot2.EM_stop("ALL")

Implementation note:
- keep the pen direction convention here aligned with the live receivers
- keep TURN_SCALE support here aligned with live mode so calibration carries over
"""

from __future__ import annotations

import argparse
from math import atan2, cos, degrees, radians, sin, sqrt
import os
from pathlib import Path
import zipfile

from src.bridge_protocol import BridgeCommand, load_plot_commands


DEFAULT_START_TIP_X_MM = 105.0
# With the mBot2 back aligned to the bottom paper edge, the pen tip is already
# about 20 cm into the page when it contacts the paper.
DEFAULT_START_TIP_Y_MM = 97.0
DEFAULT_START_HEADING_DEG = 90.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compile plot commands into a CyberPi/mBot2 Python script.",
    )
    parser.add_argument(
        "--commands-file",
        type=str,
        default="output/plot_commands.txt",
        help="Path to line-based plot commands exported by main.py",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output/cyberpi_draw.py",
        help="Generated CyberPi Python script path",
    )
    parser.add_argument(
        "--mcode-output",
        type=str,
        default="output/cyberpi_draw.mcode",
        help="Generated mBlock Python-project archive path",
    )
    parser.add_argument(
        "--skip-mcode",
        action="store_true",
        help="Only write the raw .py script and skip .mcode packaging",
    )
    parser.add_argument(
        "--open-in-mblock",
        action="store_true",
        help="Open the generated .mcode with the system file association on Windows",
    )
    parser.add_argument(
        "--start-button",
        type=str,
        default="a",
        choices=["a", "b"],
        help="CyberPi button that starts the drawing job",
    )
    parser.add_argument(
        "--stop-button",
        type=str,
        default="b",
        choices=["a", "b"],
        help="CyberPi button that stops the job",
    )
    parser.add_argument(
        "--mm-per-straight-unit",
        type=float,
        default=10.0,
        help="Bridge ratio from drawing mm to mbot2.straight() units",
    )
    parser.add_argument(
        "--min-turn-deg",
        type=float,
        default=1.0,
        help="Ignore tiny heading changes below this threshold",
    )
    parser.add_argument(
        "--min-move-mm",
        type=float,
        default=20.0,
        help="Clamp real robot moves up to at least this distance; only sub-0.5 mm noise is ignored",
    )
    parser.add_argument(
        "--min-straight-units",
        type=float,
        default=2.0,
        help="Clamp mbot2.straight() actions so they are never smaller than this value",
    )
    parser.add_argument(
        "--start-x-mm",
        type=float,
        default=DEFAULT_START_TIP_X_MM,
        help="Initial lifted pen-tip x position in paper coordinates",
    )
    parser.add_argument(
        "--start-y-mm",
        type=float,
        default=DEFAULT_START_TIP_Y_MM,
        help="Initial lifted pen-tip y position in paper coordinates",
    )
    parser.add_argument(
        "--start-heading-deg",
        type=float,
        default=DEFAULT_START_HEADING_DEG,
        help="Initial robot heading; 90 faces into the paper from the bottom edge",
    )
    parser.add_argument(
        "--use-pen",
        dest="use_pen",
        action="store_true",
        default=True,
        help="Emit servo_add-based pen up/down actions",
    )
    parser.add_argument(
        "--no-use-pen",
        dest="use_pen",
        action="store_false",
        help="Disable pen servo actions in the generated script",
    )
    parser.add_argument(
        "--servo-port",
        type=str,
        default="S1",
        help="Servo port used for the pen",
    )
    parser.add_argument(
        "--pen-lift-delta",
        "--pen-down-delta",
        type=float,
        default=36.0,
        help="Magnitude of the relative servo_add movement between pen states",
    )
    parser.add_argument(
        "--pen-delay-s",
        type=float,
        default=0.3,
        help="Pause after pen servo movement",
    )
    parser.add_argument(
        "--turn-delay-s",
        type=float,
        default=0.1,
        help="Pause after each turn for settling",
    )
    parser.add_argument(
        "--turn-scale",
        type=float,
        default=1.04,
        help="Multiplier applied to each commanded turn angle",
    )
    parser.add_argument(
        "--corner-lift-turn-deg",
        type=float,
        default=30.0,
        help="Lift the pen before turns at or above this angle; set 0 to disable",
    )
    parser.add_argument(
        "--pen-forward-offset-mm",
        type=float,
        default=160.0,
        help="Distance from the robot turn center to the pen tip in the forward direction",
    )
    parser.add_argument(
        "--pen-lateral-offset-mm",
        type=float,
        default=0.0,
        help="Distance from the robot turn center to the pen tip toward the robot's left side",
    )
    return parser.parse_args()


def normalize_angle(angle_deg: float) -> float:
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def distance_mm(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return sqrt((dx * dx) + (dy * dy))


def heading_deg_between(start: tuple[float, float], end: tuple[float, float]) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    return degrees(atan2(-dy, dx))


def pen_offset_for_heading(
    heading_deg: float,
    pen_forward_offset_mm: float,
    pen_lateral_offset_mm: float,
) -> tuple[float, float]:
    """
    Convert pen offset in robot-local coordinates to world/screen coordinates.

    Coordinate convention:
    - forward: positive along the robot's current heading
    - lateral: positive toward the robot's left side
    """
    angle_rad = radians(heading_deg)
    world_dx = (cos(angle_rad) * pen_forward_offset_mm) - (sin(angle_rad) * pen_lateral_offset_mm)
    world_dy = -((sin(angle_rad) * pen_forward_offset_mm) + (cos(angle_rad) * pen_lateral_offset_mm))
    return world_dx, world_dy


def center_point_for_tip(
    tip_point: tuple[float, float],
    heading_deg: float,
    pen_forward_offset_mm: float,
    pen_lateral_offset_mm: float,
) -> tuple[float, float]:
    offset_dx, offset_dy = pen_offset_for_heading(
        heading_deg=heading_deg,
        pen_forward_offset_mm=pen_forward_offset_mm,
        pen_lateral_offset_mm=pen_lateral_offset_mm,
    )
    return tip_point[0] - offset_dx, tip_point[1] - offset_dy


def append_turn_action(
    actions: list[tuple[str, float | None]],
    current_heading_deg: float,
    target_heading_deg: float,
    min_turn_deg: float,
) -> float:
    turn_delta = normalize_angle(target_heading_deg - current_heading_deg)
    if abs(turn_delta) >= min_turn_deg:
        actions.append(("TR", round(turn_delta, 2)))
        return target_heading_deg
    return current_heading_deg


def advance_point_for_heading(
    start_point: tuple[float, float],
    heading_deg: float,
    distance_mm: float,
) -> tuple[float, float]:
    angle_rad = radians(heading_deg)
    return (
        start_point[0] + (cos(angle_rad) * distance_mm),
        start_point[1] - (sin(angle_rad) * distance_mm),
    )


def clamp_distance_mm(
    distance_mm: float,
    mm_per_straight_unit: float,
    min_move_mm: float,
    min_straight_units: float,
) -> float:
    minimum_distance_mm = max(float(min_move_mm), float(min_straight_units) * float(mm_per_straight_unit))
    return max(distance_mm, minimum_distance_mm)


def append_move_action(
    actions: list[tuple[str, float | None]],
    current_position: tuple[float, float],
    current_heading_deg: float,
    target_position: tuple[float, float],
    mm_per_straight_unit: float,
    min_turn_deg: float,
    min_move_mm: float,
    min_straight_units: float,
) -> tuple[tuple[float, float], float]:
    segment_distance_mm = distance_mm(current_position, target_position)
    if segment_distance_mm < 0.5:
        return current_position, current_heading_deg

    target_heading_deg = heading_deg_between(current_position, target_position)
    target_heading_deg = append_turn_action(
        actions=actions,
        current_heading_deg=current_heading_deg,
        target_heading_deg=target_heading_deg,
        min_turn_deg=min_turn_deg,
    )

    clamped_distance_mm = clamp_distance_mm(
        distance_mm=segment_distance_mm,
        mm_per_straight_unit=mm_per_straight_unit,
        min_move_mm=min_move_mm,
        min_straight_units=min_straight_units,
    )
    straight_units = round(clamped_distance_mm / mm_per_straight_unit, 2)
    actions.append(("ST", straight_units))
    actual_position = advance_point_for_heading(
        start_point=current_position,
        heading_deg=target_heading_deg,
        distance_mm=clamped_distance_mm,
    )
    return actual_position, target_heading_deg


def extract_strokes(
    commands: list[BridgeCommand],
    initial_tip_point: tuple[float, float] = (0.0, 0.0),
) -> list[list[tuple[float, float]]]:
    strokes: list[list[tuple[float, float]]] = []
    current_tip = initial_tip_point
    last_pen_up_target = current_tip
    current_stroke: list[tuple[float, float]] | None = None
    pen_is_down = False

    for cmd in commands:
        if cmd.cmd == "PEN_UP":
            pen_is_down = False
            current_stroke = None
            continue

        if cmd.cmd == "PEN_DOWN":
            if not pen_is_down:
                current_stroke = [last_pen_up_target]
                strokes.append(current_stroke)
                pen_is_down = True
            continue

        if cmd.cmd != "MOVE":
            raise ValueError(f"Unsupported plot command for code generation: {cmd.cmd}")

        if cmd.x is None or cmd.y is None:
            raise ValueError("MOVE commands require x and y coordinates")

        current_tip = (float(cmd.x), float(cmd.y))
        if pen_is_down:
            if current_stroke is None:
                current_stroke = [current_tip]
                strokes.append(current_stroke)
            current_stroke.append(current_tip)
        else:
            last_pen_up_target = current_tip

    return strokes


def compile_actions(
    commands: list[BridgeCommand],
    mm_per_straight_unit: float,
    min_turn_deg: float,
    min_move_mm: float,
    corner_lift_turn_deg: float = 45.0,
    pen_forward_offset_mm: float = 0.0,
    pen_lateral_offset_mm: float = 0.0,
    start_tip_point: tuple[float, float] = (0.0, 0.0),
    start_heading_deg: float = 0.0,
    min_straight_units: float = 2.0,
) -> list[tuple[str, float | None]]:
    """
    Convert high-level plot commands into primitive robot actions.

    Output action opcodes:
    - "PU"               pen up
    - "PD"               pen down
    - ("TR", degrees)    relative turn in degrees
    - ("ST", distance)   forward distance in mbot2.straight() units
    """
    actions: list[tuple[str, float | None]] = []
    initial_center = center_point_for_tip(
        tip_point=start_tip_point,
        heading_deg=start_heading_deg,
        pen_forward_offset_mm=pen_forward_offset_mm,
        pen_lateral_offset_mm=pen_lateral_offset_mm,
    )
    current_position = initial_center
    heading_deg = start_heading_deg
    pen_is_down = False
    strokes = extract_strokes(commands, initial_tip_point=start_tip_point)
    corner_lift_enabled = corner_lift_turn_deg > 0.0
    reposition_tolerance_mm = max(0.25, min_move_mm)

    for stroke in strokes:
        if len(stroke) < 2:
            continue

        segments: list[dict[str, float | tuple[float, float]]] = []
        for start_tip, end_tip in zip(stroke, stroke[1:]):
            segment_tip_distance = distance_mm(start_tip, end_tip)
            if segment_tip_distance < 0.5:
                continue

            segment_heading_deg = heading_deg_between(start_tip, end_tip)
            start_center = center_point_for_tip(
                tip_point=start_tip,
                heading_deg=segment_heading_deg,
                pen_forward_offset_mm=pen_forward_offset_mm,
                pen_lateral_offset_mm=pen_lateral_offset_mm,
            )
            end_center = center_point_for_tip(
                tip_point=end_tip,
                heading_deg=segment_heading_deg,
                pen_forward_offset_mm=pen_forward_offset_mm,
                pen_lateral_offset_mm=pen_lateral_offset_mm,
            )
            segments.append(
                {
                    "heading_deg": segment_heading_deg,
                    "start_center": start_center,
                    "end_center": end_center,
                    "segment_distance_mm": segment_tip_distance,
                }
            )

        if not segments:
            continue

        first_segment = segments[0]
        if pen_is_down:
            actions.append(("PU", None))
            pen_is_down = False

        current_position, heading_deg = append_move_action(
            actions=actions,
            current_position=current_position,
            current_heading_deg=heading_deg,
            target_position=first_segment["start_center"],
            mm_per_straight_unit=mm_per_straight_unit,
            min_turn_deg=min_turn_deg,
            min_move_mm=min_move_mm,
            min_straight_units=min_straight_units,
        )
        heading_deg = append_turn_action(
            actions=actions,
            current_heading_deg=heading_deg,
            target_heading_deg=float(first_segment["heading_deg"]),
            min_turn_deg=min_turn_deg,
        )
        actions.append(("PD", None))
        pen_is_down = True

        clamped_distance_mm = clamp_distance_mm(
            distance_mm=float(first_segment["segment_distance_mm"]),
            mm_per_straight_unit=mm_per_straight_unit,
            min_move_mm=min_move_mm,
            min_straight_units=min_straight_units,
        )
        straight_units = round(clamped_distance_mm / mm_per_straight_unit, 2)
        actions.append(("ST", straight_units))
        current_position = advance_point_for_heading(
            start_point=current_position,
            heading_deg=float(first_segment["heading_deg"]),
            distance_mm=clamped_distance_mm,
        )
        heading_deg = float(first_segment["heading_deg"])

        for previous_segment, segment in zip(segments, segments[1:]):
            previous_heading_deg = float(previous_segment["heading_deg"])
            next_heading_deg = float(segment["heading_deg"])
            heading_change_deg = abs(normalize_angle(next_heading_deg - previous_heading_deg))
            center_gap_mm = distance_mm(current_position, segment["start_center"])
            requires_pen_lift = (
                center_gap_mm > reposition_tolerance_mm
                or (corner_lift_enabled and heading_change_deg >= corner_lift_turn_deg)
            )

            if requires_pen_lift and pen_is_down:
                actions.append(("PU", None))
                pen_is_down = False

            if not pen_is_down:
                current_position, heading_deg = append_move_action(
                    actions=actions,
                    current_position=current_position,
                    current_heading_deg=heading_deg,
                    target_position=segment["start_center"],
                    mm_per_straight_unit=mm_per_straight_unit,
                    min_turn_deg=min_turn_deg,
                    min_move_mm=min_move_mm,
                    min_straight_units=min_straight_units,
                )
                heading_deg = append_turn_action(
                    actions=actions,
                    current_heading_deg=heading_deg,
                    target_heading_deg=next_heading_deg,
                    min_turn_deg=min_turn_deg,
                )
                actions.append(("PD", None))
                pen_is_down = True
            else:
                heading_deg = append_turn_action(
                    actions=actions,
                    current_heading_deg=heading_deg,
                    target_heading_deg=next_heading_deg,
                    min_turn_deg=min_turn_deg,
                )

            clamped_distance_mm = clamp_distance_mm(
                distance_mm=float(segment["segment_distance_mm"]),
                mm_per_straight_unit=mm_per_straight_unit,
                min_move_mm=min_move_mm,
                min_straight_units=min_straight_units,
            )
            straight_units = round(clamped_distance_mm / mm_per_straight_unit, 2)
            actions.append(("ST", straight_units))
            current_position = advance_point_for_heading(
                start_point=current_position,
                heading_deg=next_heading_deg,
                distance_mm=clamped_distance_mm,
            )
            heading_deg = next_heading_deg

    if pen_is_down:
        actions.append(("PU", None))

    return actions


def format_action(action: tuple[str, float | None]) -> str:
    op, value = action
    if value is None:
        return f'    ("{op}",),'
    return f'    ("{op}", {value:.2f}),'


def render_script(
    actions: list[tuple[str, float | None]],
    start_button: str,
    stop_button: str,
    use_pen: bool,
    servo_port: str,
    pen_lift_delta: float,
    pen_delay_s: float,
    turn_delay_s: float,
    turn_scale: float,
    start_tip_point: tuple[float, float] | None = None,
    start_heading_deg: float | None = None,
) -> str:
    action_lines = "\n".join(format_action(action) for action in actions)
    if start_tip_point is None or start_heading_deg is None:
        start_pose_comment = "# Start pose: (0.00, 0.00) mm, heading 0.00 deg."
    else:
        start_pose_comment = (
            "# Start pose: lifted pen tip at "
            f"({start_tip_point[0]:.2f}, {start_tip_point[1]:.2f}) mm, "
            f"heading {start_heading_deg:.2f} deg."
        )

    return f'''# generated for CyberPi + mBot2

import event, time, cyberpi, mbot2

SERVO_PORT = "{servo_port}"
USE_PEN = {str(use_pen)}
PEN_LIFT_DELTA = {pen_lift_delta:.2f}
PEN_DELAY_S = {pen_delay_s:.2f}
TURN_DELAY_S = {turn_delay_s:.2f}
TURN_SCALE = {turn_scale:.4f}

# The pen is expected to start in the UP position before button "{start_button}" is pressed.
{start_pose_comment}
# Negative servo delta raises the pen. Positive servo delta lowers the pen.
ACTIONS = [
{action_lines}
]

pen_is_down = False


def pen_up():
    global pen_is_down
    if (not USE_PEN) or (not pen_is_down):
        return
    mbot2.servo_add(-PEN_LIFT_DELTA, SERVO_PORT)
    pen_is_down = False
    time.sleep(PEN_DELAY_S)


def pen_down():
    global pen_is_down
    if (not USE_PEN) or pen_is_down:
        return
    mbot2.servo_add(+PEN_LIFT_DELTA, SERVO_PORT)
    pen_is_down = True
    time.sleep(PEN_DELAY_S)


def run_actions():
    global pen_is_down
    pen_is_down = False
    mbot2.EM_stop("ALL")

    for action in ACTIONS:
        op = action[0]

        if op == "TR":
            mbot2.turn(TURN_SCALE * action[1])
            time.sleep(TURN_DELAY_S)
        elif op == "ST":
            mbot2.straight(action[1])
        elif op == "PD":
            pen_down()
        elif op == "PU":
            pen_up()

    mbot2.EM_stop("ALL")
    pen_up()


@event.start
def on_start():
    mbot2.EM_stop("ALL")


@event.is_press("{start_button}")
def start_job():
    run_actions()


@event.is_press("{stop_button}")
def stop_job():
    mbot2.EM_stop("ALL")
    pen_up()
    cyberpi.stop_all()
'''


def write_script_outputs(
    script: str,
    output_path: str | Path,
    mcode_output_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """
    Persist the generated CyberPi script and optional .mcode package.

    Implementation note:
    - keep UI export and CLI export on this shared helper so packaging behavior
      stays aligned in one place
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(script, encoding="utf-8")

    mcode_path: Path | None = None
    if mcode_output_path is not None:
        mcode_path = Path(mcode_output_path)
        mcode_path.parent.mkdir(parents=True, exist_ok=True)
        script_name = out_path.name

        with zipfile.ZipFile(mcode_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(script_name, script)

    return out_path, mcode_path


def main() -> None:
    args = parse_args()
    commands = load_plot_commands(args.commands_file)
    actions = compile_actions(
        commands=commands,
        mm_per_straight_unit=args.mm_per_straight_unit,
        min_turn_deg=args.min_turn_deg,
        min_move_mm=args.min_move_mm,
        corner_lift_turn_deg=args.corner_lift_turn_deg,
        pen_forward_offset_mm=args.pen_forward_offset_mm,
        pen_lateral_offset_mm=args.pen_lateral_offset_mm,
        start_tip_point=(args.start_x_mm, args.start_y_mm),
        start_heading_deg=args.start_heading_deg,
        min_straight_units=args.min_straight_units,
    )

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
        start_tip_point=(args.start_x_mm, args.start_y_mm),
        start_heading_deg=args.start_heading_deg,
    )

    out_path, mcode_path = write_script_outputs(
        script=script,
        output_path=args.output,
        mcode_output_path=None if args.skip_mcode else args.mcode_output,
    )

    if args.open_in_mblock:
        if os.name != "nt":
            raise RuntimeError("--open-in-mblock is only supported on Windows")
        if args.skip_mcode:
            raise RuntimeError("--open-in-mblock requires .mcode packaging; remove --skip-mcode")
        os.startfile(str(mcode_path))

    print(f"Loaded plot commands: {len(commands)}")
    print(f"Generated primitive actions: {len(actions)}")
    print(f"Wrote CyberPi script to: {out_path}")
    if mcode_path is not None:
        print(f"Wrote mBlock project to: {mcode_path}")
    if args.open_in_mblock:
        print(f"Opened in mBlock: {mcode_path}")


if __name__ == "__main__":
    main()
