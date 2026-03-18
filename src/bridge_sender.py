"""
Bridge sender for moving drawing commands from this repo to mBlock Live mode.

Two supported delivery modes:
- socket: send commands in real time to a local TCP receiver
- file: write JSONL commands to a file that mBlock can poll/read

This keeps OpenCV / NumPy / Torch on the local machine and makes the mBlock side
small enough to survive restricted environments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import socket
import time

from src.bridge_protocol import (
    BridgeCommand,
    command_to_json,
    load_plot_commands,
    save_commands_as_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Send drawing commands to an mBlock bridge.")
    parser.add_argument(
        "--commands-file",
        type=str,
        default="output/plot_commands.txt",
        help="Path to line-based plot commands exported by main.py",
    )
    parser.add_argument(
        "--mode",
        choices=["socket", "file"],
        default="socket",
        help="Bridge transport",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Socket host")
    parser.add_argument("--port", type=int, default=8765, help="Socket port")
    parser.add_argument(
        "--outbox",
        type=str,
        default="output/bridge_commands.jsonl",
        help="JSONL path for file-mode bridge",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=20,
        help="Inter-command delay for socket mode",
    )
    return parser.parse_args()


def build_session(commands: list[BridgeCommand]) -> list[BridgeCommand]:
    """
    Wrap command stream with lifecycle markers.

    START and END make it easier for the robot-side bridge to reset state.
    """
    return [BridgeCommand(cmd="START"), *commands, BridgeCommand(cmd="END")]


def send_socket(commands: list[BridgeCommand], host: str, port: int, delay_ms: int) -> None:
    """Send JSON-line commands over TCP to the mBlock Live receiver."""
    with socket.create_connection((host, port), timeout=10) as sock:
        writer = sock.makefile("w", encoding="utf-8")

        for cmd in commands:
            writer.write(command_to_json(cmd) + "\n")
            writer.flush()
            time.sleep(delay_ms / 1000.0)


def send_file(commands: list[BridgeCommand], outbox: str) -> None:
    """Write commands as JSON lines so a file-polling bridge can consume them."""
    out_path = Path(outbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_commands_as_jsonl(commands, outbox)


def main() -> None:
    args = parse_args()
    commands = load_plot_commands(args.commands_file)
    session = build_session(commands)

    print(f"Loaded commands: {len(commands)}")
    print(f"Bridge mode: {args.mode}")

    if args.mode == "socket":
        send_socket(session, host=args.host, port=args.port, delay_ms=args.delay_ms)
        print(f"Sent session to {args.host}:{args.port}")
        return

    send_file(session, outbox=args.outbox)
    print(f"Wrote JSONL bridge file to: {args.outbox}")


if __name__ == "__main__":
    main()
