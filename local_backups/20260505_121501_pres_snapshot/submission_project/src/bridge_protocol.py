"""Helpers used to read and write the text commands shared by planning and export."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class BridgeCommand:
    """Simple drawing command used across the project."""

    cmd: str
    x: float | None = None
    y: float | None = None
    speed: float | None = None


def parse_plot_command_line(line: str) -> BridgeCommand:
    """Parse one line from a plot command text file."""
    stripped = line.strip()
    if not stripped:
        raise ValueError("Cannot parse an empty command line")

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
    """Serialize one bridge command to the text format used in plot_commands.txt."""
    if cmd.cmd in {"PEN_UP", "PEN_DOWN", "START", "END", "PING"}:
        return cmd.cmd

    if cmd.cmd == "MOVE":
        if cmd.x is None or cmd.y is None or cmd.speed is None:
            raise ValueError("MOVE commands require x, y, and speed values")
        return f"MOVE {cmd.x:.2f} {cmd.y:.2f} {cmd.speed:.2f}"

    raise ValueError(f"Unsupported bridge command for serialization: {cmd.cmd}")


def bridge_commands_to_text(commands: list[BridgeCommand]) -> str:
    """Serialize a list of commands to the line-based text format."""
    return "\n".join(bridge_command_to_line(cmd) for cmd in commands)


def save_bridge_commands_as_text(commands: list[BridgeCommand], path: str) -> None:
    """Write bridge commands to a text file."""
    Path(path).write_text(bridge_commands_to_text(commands), encoding="utf-8")
