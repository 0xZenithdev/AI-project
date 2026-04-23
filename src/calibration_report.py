"""
Turn real-world calibration measurements into recommended bridge constants.

Implementation note:
- PEN_LIFT_DELTA is intentionally left as a manual decision from the pen-gaps test
- this tool only computes MM_PER_STRAIGHT_UNIT and TURN_SCALE from measurements

Examples:
python -m src.calibration_report --current-mm-per-straight-unit 10 --distance-observation 50=48 --distance-observation 100=96
python -m src.calibration_report --current-turn-scale 1.0 --turn-observation 90=84
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_ratio(raw: str) -> tuple[float, float]:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Observations must use the format commanded=measured")

    commanded_raw, measured_raw = raw.split("=", 1)
    try:
        commanded = float(commanded_raw)
        measured = float(measured_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid numeric observation: {raw}") from exc

    if commanded <= 0 or measured <= 0:
        raise argparse.ArgumentTypeError("Observation values must be positive")

    return commanded, measured


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute recommended robot calibration constants.")
    parser.add_argument(
        "--current-mm-per-straight-unit",
        type=float,
        default=10.0,
        help="The MM_PER_STRAIGHT_UNIT value used during the test run",
    )
    parser.add_argument(
        "--distance-observation",
        type=parse_ratio,
        action="append",
        default=[],
        help="Target line length in mm and measured line length in mm, as target=measured",
    )
    parser.add_argument(
        "--current-turn-scale",
        type=float,
        default=1.0,
        help="The TURN_SCALE value used during the test run",
    )
    parser.add_argument(
        "--turn-observation",
        type=parse_ratio,
        action="append",
        default=[],
        help="Commanded angle and observed angle in degrees, as commanded=observed",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Optional JSON report output path",
    )
    return parser.parse_args()


def recommend_mm_per_straight_unit(
    current_mm_per_straight_unit: float,
    observations: list[tuple[float, float]],
) -> float | None:
    if not observations:
        return None

    target_total = sum(target for target, _measured in observations)
    measured_total = sum(measured for _target, measured in observations)
    return current_mm_per_straight_unit * (measured_total / target_total)


def recommend_turn_scale(
    current_turn_scale: float,
    observations: list[tuple[float, float]],
) -> float | None:
    if not observations:
        return None

    commanded_total = sum(commanded for commanded, _observed in observations)
    observed_total = sum(observed for _commanded, observed in observations)
    return current_turn_scale * (commanded_total / observed_total)


def main() -> None:
    args = parse_args()

    recommended_mm = recommend_mm_per_straight_unit(
        current_mm_per_straight_unit=args.current_mm_per_straight_unit,
        observations=args.distance_observation,
    )
    recommended_turn = recommend_turn_scale(
        current_turn_scale=args.current_turn_scale,
        observations=args.turn_observation,
    )

    if recommended_mm is None and recommended_turn is None:
        raise SystemExit("No observations supplied. Provide at least one distance or turn observation.")

    report = {
        "inputs": {
            "current_mm_per_straight_unit": args.current_mm_per_straight_unit,
            "distance_observations": args.distance_observation,
            "current_turn_scale": args.current_turn_scale,
            "turn_observations": args.turn_observation,
        },
        "recommended": {
            "MM_PER_STRAIGHT_UNIT": recommended_mm,
            "TURN_SCALE": recommended_turn,
        },
        "notes": [
            "PEN_LIFT_DELTA still needs manual tuning from the pen-gaps test.",
            "Distance recommendation is weighted by total commanded and measured line length.",
            "Turn recommendation is weighted by total commanded and observed angle.",
        ],
    }

    if recommended_mm is not None:
        print(f"Recommended MM_PER_STRAIGHT_UNIT: {recommended_mm:.4f}")
    else:
        print("Recommended MM_PER_STRAIGHT_UNIT: not enough data")

    if recommended_turn is not None:
        print(f"Recommended TURN_SCALE: {recommended_turn:.4f}")
    else:
        print("Recommended TURN_SCALE: not enough data")

    print("PEN_LIFT_DELTA: tune manually with the pen-gaps test")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote calibration report to: {out_path}")


if __name__ == "__main__":
    main()
