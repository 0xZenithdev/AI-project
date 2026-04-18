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

Memory note:
- keep the pen direction convention here aligned with the live receivers
- keep TURN_SCALE support here aligned with live mode so calibration carries over
"""

from __future__ import annotations

import argparse
from math import atan2, degrees, sqrt
import os
from pathlib import Path
import zipfile

from src.bridge_protocol import BridgeCommand, load_plot_commands


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
        default=1.0,
        help="Ignore tiny moves below this threshold",
    )
    parser.add_argument(
        "--use-pen",
        action="store_true",
        help="Emit servo_add-based pen up/down actions",
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
        default=60.0,
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
        default=1.0,
        help="Multiplier applied to each commanded turn angle",
    )
    return parser.parse_args()


def normalize_angle(angle_deg: float) -> float:
    while angle_deg <= -180.0:
        angle_deg += 360.0
    while angle_deg > 180.0:
        angle_deg -= 360.0
    return angle_deg


def compile_actions(
    commands: list[BridgeCommand],
    mm_per_straight_unit: float,
    min_turn_deg: float,
    min_move_mm: float,
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

    current_x = 0.0
    current_y = 0.0
    heading_deg = 0.0
    pen_is_down = False

    for cmd in commands:
        if cmd.cmd == "PEN_UP":
            if pen_is_down:
                actions.append(("PU", None))
                pen_is_down = False
            continue

        if cmd.cmd == "PEN_DOWN":
            if not pen_is_down:
                actions.append(("PD", None))
                pen_is_down = True
            continue

        if cmd.cmd != "MOVE":
            raise ValueError(f"Unsupported plot command for code generation: {cmd.cmd}")

        if cmd.x is None or cmd.y is None:
            raise ValueError("MOVE commands require x and y coordinates")

        dx = cmd.x - current_x
        dy = cmd.y - current_y
        distance_mm = sqrt((dx * dx) + (dy * dy))

        if distance_mm < min_move_mm:
            current_x = cmd.x
            current_y = cmd.y
            continue

        target_heading = degrees(atan2(-dy, dx))
        turn_delta = normalize_angle(target_heading - heading_deg)

        if abs(turn_delta) >= min_turn_deg:
            actions.append(("TR", round(turn_delta, 2)))

        straight_units = round(distance_mm / mm_per_straight_unit, 2)
        if straight_units != 0.0:
            actions.append(("ST", straight_units))

        current_x = cmd.x
        current_y = cmd.y
        heading_deg = target_heading

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
) -> str:
    action_lines = "\n".join(format_action(action) for action in actions)

    return f'''# generated by AI-project for CyberPi + mBot2

import event, time, cyberpi, mbot2

SERVO_PORT = "{servo_port}"
USE_PEN = {str(use_pen)}
PEN_LIFT_DELTA = {pen_lift_delta:.2f}
PEN_DELAY_S = {pen_delay_s:.2f}
TURN_DELAY_S = {turn_delay_s:.2f}
TURN_SCALE = {turn_scale:.4f}

# Assumption: the pen starts in the UP position before button "{start_button}" is pressed.
# If it does not, adjust the mechanical setup or the configured delta before downloading.
# Pen-lift convention:
# - positive servo delta raises the pen
# - negative servo delta lowers the pen
ACTIONS = [
{action_lines}
]

pen_is_down = False


def pen_up():
    global pen_is_down
    if (not USE_PEN) or (not pen_is_down):
        return
    mbot2.servo_add(PEN_LIFT_DELTA, SERVO_PORT)
    pen_is_down = False
    time.sleep(PEN_DELAY_S)


def pen_down():
    global pen_is_down
    if (not USE_PEN) or pen_is_down:
        return
    mbot2.servo_add(-PEN_LIFT_DELTA, SERVO_PORT)
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

    Memory note:
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
