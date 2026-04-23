"""
Shared bridge protocol utilities for local Python <-> mBlock Live communication.

We keep this protocol deliberately small:
- PEN_UP
- PEN_DOWN
- MOVE x y speed
- START
- END
- PING

The robot-side bridge only needs to understand these commands.

Implementation note:
- this command vocabulary is shared by the planner, calibration generator,
  bridge sender, and mBlock receivers
- extend it carefully and update all consumers together
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass
class BridgeCommand:
    """Simple transport-friendly drawing command."""

    cmd: str
    x: float | None = None
    y: float | None = None
    speed: float | None = None


def parse_plot_command_line(line: str) -> BridgeCommand:
    """
    Parse one line from output/plot_commands.txt.

    Supported formats:
    - PEN_UP
    - PEN_DOWN
    - MOVE 12.34 56.78 35.00
    """
    stripped = line.strip()
    if not stripped:
        raise ValueError("Cannot parse empty command line")

    parts = stripped.split()
    head = parts[0]

    if head in {"PEN_UP", "PEN_DOWN", "START", "END", "PING"}:
        return BridgeCommand(cmd=head)

    if head == "MOVE":
        if len(parts) != 4:
            raise ValueError(f"Invalid MOVE command: {line}")
        return BridgeCommand(
            cmd="MOVE",
            x=float(parts[1]),
            y=float(parts[2]),
            speed=float(parts[3]),
        )

    raise ValueError(f"Unsupported command: {line}")


def load_plot_commands(path: str) -> list[BridgeCommand]:
    """Load line-based plot commands from disk."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [parse_plot_command_line(line) for line in lines if line.strip()]


def bridge_command_to_line(cmd: BridgeCommand) -> str:
    """Serialize one bridge command to the line-based text transport format."""
    if cmd.cmd in {"PEN_UP", "PEN_DOWN", "START", "END", "PING"}:
        return cmd.cmd

    if cmd.cmd == "MOVE":
        if cmd.x is None or cmd.y is None or cmd.speed is None:
            raise ValueError("MOVE commands require x, y, and speed values")
        return f"MOVE {cmd.x:.2f} {cmd.y:.2f} {cmd.speed:.2f}"

    raise ValueError(f"Unsupported bridge command for text serialization: {cmd.cmd}")


def bridge_commands_to_text(commands: list[BridgeCommand]) -> str:
    """Serialize bridge commands into the project's line-based command file format."""
    return "\n".join(bridge_command_to_line(cmd) for cmd in commands)


def save_bridge_commands_as_text(commands: list[BridgeCommand], path: str) -> None:
    """Write bridge commands to a line-based plot command file."""
    Path(path).write_text(bridge_commands_to_text(commands), encoding="utf-8")


def command_to_json(cmd: BridgeCommand) -> str:
    """Serialize one command to one JSON line."""
    payload = {"cmd": cmd.cmd}
    if cmd.x is not None:
        payload["x"] = cmd.x
    if cmd.y is not None:
        payload["y"] = cmd.y
    if cmd.speed is not None:
        payload["speed"] = cmd.speed
    return json.dumps(payload)


def command_from_json(line: str) -> BridgeCommand:
    """Parse one JSON line from the bridge wire format."""
    payload = json.loads(line)
    return BridgeCommand(
        cmd=str(payload["cmd"]),
        x=payload.get("x"),
        y=payload.get("y"),
        speed=payload.get("speed"),
    )


def save_commands_as_jsonl(commands: list[BridgeCommand], path: str) -> None:
    """Write JSON-lines file for the fallback file bridge."""
    data = "\n".join(command_to_json(cmd) for cmd in commands)
    Path(path).write_text(data, encoding="utf-8")
