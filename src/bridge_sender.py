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
from collections import Counter
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
    parser.add_argument(
        "--connect-attempts",
        type=int,
        default=20,
        help="Total socket connection attempts before giving up",
    )
    parser.add_argument(
        "--retry-delay-ms",
        type=int,
        default=500,
        help="Delay between socket connection attempts",
    )
    parser.add_argument(
        "--prepend-ping",
        action="store_true",
        help="Insert a diagnostic PING before the START marker",
    )
    return parser.parse_args()


def build_session(commands: list[BridgeCommand], prepend_ping: bool = False) -> list[BridgeCommand]:
    """
    Wrap command stream with lifecycle markers.

    START and END make it easier for the robot-side bridge to reset state.
    """
    prefix: list[BridgeCommand] = []
    if prepend_ping:
        prefix.append(BridgeCommand(cmd="PING"))
    return [*prefix, BridgeCommand(cmd="START"), *commands, BridgeCommand(cmd="END")]


def summarize_commands(commands: list[BridgeCommand]) -> str:
    counts = Counter(cmd.cmd for cmd in commands)
    ordered = ["PING", "START", "PEN_UP", "PEN_DOWN", "MOVE", "END"]
    parts = [f"{name}={counts[name]}" for name in ordered if counts[name]]
    return ", ".join(parts)


def connect_with_retries(host: str, port: int, connect_attempts: int, retry_delay_ms: int) -> socket.socket:
    last_error: OSError | None = None
    attempts = max(1, connect_attempts)

    for attempt in range(1, attempts + 1):
        try:
            return socket.create_connection((host, port), timeout=10)
        except OSError as exc:
            last_error = exc
            if attempt == attempts:
                break
            print(f"Connect attempt {attempt}/{attempts} failed: {exc}")
            time.sleep(retry_delay_ms / 1000.0)

    raise RuntimeError(f"Could not connect to {host}:{port} after {attempts} attempts: {last_error}")


def send_socket(
    commands: list[BridgeCommand],
    host: str,
    port: int,
    delay_ms: int,
    connect_attempts: int,
    retry_delay_ms: int,
) -> None:
    """Send JSON-line commands over TCP to the mBlock Live receiver."""
    with connect_with_retries(
        host=host,
        port=port,
        connect_attempts=connect_attempts,
        retry_delay_ms=retry_delay_ms,
    ) as sock:
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        writer = sock.makefile("w", encoding="utf-8")

        for index, cmd in enumerate(commands, start=1):
            try:
                writer.write(command_to_json(cmd) + "\n")
                writer.flush()
            except OSError as exc:
                raise RuntimeError(
                    f"Socket send failed at command {index}/{len(commands)} ({cmd.cmd}): {exc}"
                ) from exc
            time.sleep(delay_ms / 1000.0)


def send_file(commands: list[BridgeCommand], outbox: str) -> None:
    """Write commands as JSON lines so a file-polling bridge can consume them atomically."""
    out_path = Path(outbox)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = out_path.with_name(out_path.name + ".tmp")
    save_commands_as_jsonl(commands, str(temp_path))
    temp_path.replace(out_path)


def main() -> None:
    args = parse_args()
    commands = load_plot_commands(args.commands_file)
    session = build_session(commands, prepend_ping=args.prepend_ping)

    print(f"Loaded commands: {len(commands)}")
    print(f"Session command counts: {summarize_commands(session)}")
    print(f"Bridge mode: {args.mode}")

    if args.mode == "socket":
        send_socket(
            session,
            host=args.host,
            port=args.port,
            delay_ms=args.delay_ms,
            connect_attempts=args.connect_attempts,
            retry_delay_ms=args.retry_delay_ms,
        )
        print(f"Sent session to {args.host}:{args.port}")
        return

    send_file(session, outbox=args.outbox)
    print(f"Wrote JSONL bridge file to: {args.outbox}")


if __name__ == "__main__":
    main()
