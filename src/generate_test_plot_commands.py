"""
Generate very small plot command files for mBot2 calibration.

These are intentionally tiny so you can verify:
- pen up / pen down
- forward motion
- turning direction
- approximate scale

Examples:
python -m src.generate_test_plot_commands --shape line
python -m src.generate_test_plot_commands --shape square --size-mm 20
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate tiny calibration plot commands.")
    parser.add_argument(
        "--shape",
        choices=["line", "square"],
        default="line",
        help="Calibration shape to generate",
    )
    parser.add_argument(
        "--size-mm",
        type=float,
        default=20.0,
        help="Line length or square side length in mm",
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
        help="Output command file",
    )
    return parser.parse_args()


def line_commands(start_x: float, start_y: float, size_mm: float, travel_speed: float, draw_speed: float) -> list[str]:
    end_x = start_x + size_mm
    return [
        "PEN_UP",
        f"MOVE {start_x:.2f} {start_y:.2f} {travel_speed:.2f}",
        "PEN_DOWN",
        f"MOVE {end_x:.2f} {start_y:.2f} {draw_speed:.2f}",
        "PEN_UP",
    ]


def square_commands(start_x: float, start_y: float, size_mm: float, travel_speed: float, draw_speed: float) -> list[str]:
    x = start_x
    y = start_y
    s = size_mm
    return [
        "PEN_UP",
        f"MOVE {x:.2f} {y:.2f} {travel_speed:.2f}",
        "PEN_DOWN",
        f"MOVE {x + s:.2f} {y:.2f} {draw_speed:.2f}",
        f"MOVE {x + s:.2f} {y + s:.2f} {draw_speed:.2f}",
        f"MOVE {x:.2f} {y + s:.2f} {draw_speed:.2f}",
        f"MOVE {x:.2f} {y:.2f} {draw_speed:.2f}",
        "PEN_UP",
    ]


def main() -> None:
    args = parse_args()

    if args.shape == "line":
        commands = line_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            size_mm=args.size_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        )
    else:
        commands = square_commands(
            start_x=args.start_x,
            start_y=args.start_y,
            size_mm=args.size_mm,
            travel_speed=args.travel_speed,
            draw_speed=args.draw_speed,
        )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(commands), encoding="utf-8")

    print(f"Generated {args.shape} test commands at: {out_path}")
    print(f"Command count: {len(commands)}")


if __name__ == "__main__":
    main()
