# generated bridge template for mBlock5 + CyberPi + mBot2

import cyberpi
import event
import json
import math
import mbot2
import socket
import time


HOST = "127.0.0.1"
PORT = 8765
USE_PEN = False
# Use the actual servo port wired to the pen lift: S1, S2, S3, or S4.
SERVO_PORT = "S1"

# Pen-lift convention confirmed by the user:
# - negative servo delta raises the pen
# - positive servo delta lowers the pen
# Keep this constant block aligned with the file receiver, inline test receiver,
# and PROJECT_NOTES.md.
PEN_LIFT_DELTA = 36
TURN_SIGN = 1.0
TURN_SCALE = 1.04
MM_PER_STRAIGHT_UNIT = 10.0
CORNER_LIFT_TURN_DEG = 30.0
PEN_DELAY_S = 1.0
TURN_DELAY_S = 0.1
SERVER_ACCEPT_TIMEOUT_S = 0.5
CLIENT_READ_TIMEOUT_S = 0.5
MAX_BUFFER_CHARS = 8192

SERVER_RUNNING = False
STOP_REQUESTED = False
ACTIVE_ROBOT = None


def log(event_name, *parts):
    if parts:
        print("BRIDGE", event_name, *parts)
    else:
        print("BRIDGE", event_name)


class RobotAdapter:
    def __init__(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.heading_deg = 0.0
        self.pen_is_down = False

    def reset_job(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.heading_deg = 0.0
        self.pen_is_down = False
        # The receiver assumes the pen is already physically UP
        # before the job starts, so we do not auto-raise it here.
        mbot2.EM_stop("ALL")
        log("START")

    def finish_job(self):
        mbot2.EM_stop("ALL")
        self.pen_up()
        log("END")

    def pen_up(self):
        if not USE_PEN:
            log("PEN_UP_SKIPPED")
            return
        if not self.pen_is_down:
            return
        self._servo_by(-PEN_LIFT_DELTA)
        self.pen_is_down = False
        log("PEN_UP")

    def pen_down(self):
        if not USE_PEN:
            log("PEN_DOWN_SKIPPED")
            return
        if self.pen_is_down:
            return
        self._servo_by(+PEN_LIFT_DELTA)
        self.pen_is_down = True
        log("PEN_DOWN")

    def move_to(self, x, y, speed):
        dx = x - self.current_x
        dy = y - self.current_y
        distance_mm = math.sqrt((dx * dx) + (dy * dy))

        if distance_mm < 1.0:
            log("MOVE_SKIPPED", "tiny move", round(distance_mm, 2))
            return

        target_heading = math.degrees(math.atan2(-dy, dx))
        turn_delta = self._normalize_angle(target_heading - self.heading_deg)

        restore_pen_after_turn = USE_PEN and self.pen_is_down and abs(turn_delta) >= CORNER_LIFT_TURN_DEG
        if restore_pen_after_turn:
            self.pen_up()

        self._turn_by(turn_delta)

        if restore_pen_after_turn:
            self.pen_down()

        mbot2.straight(round(distance_mm / MM_PER_STRAIGHT_UNIT, 2))
        log("MOVE", x, y, speed)

        self.current_x = x
        self.current_y = y
        self.heading_deg = target_heading

    def _servo_by(self, delta_angle):
        mbot2.servo_add(delta_angle, SERVO_PORT)
        time.sleep(PEN_DELAY_S)

    def _turn_by(self, delta_deg):
        if abs(delta_deg) < 1.0:
            return

        mbot2.turn(TURN_SIGN * TURN_SCALE * delta_deg)

        self.heading_deg = self._normalize_angle(self.heading_deg + delta_deg)
        time.sleep(TURN_DELAY_S)

    def _normalize_angle(self, angle_deg):
        while angle_deg <= -180:
            angle_deg += 360
        while angle_deg > 180:
            angle_deg -= 360
        return angle_deg

    def emergency_stop(self):
        mbot2.EM_stop("ALL")
        if USE_PEN and self.pen_is_down:
            self.pen_up()


def request_stop(reason):
    global STOP_REQUESTED
    STOP_REQUESTED = True
    log("STOP_REQUESTED", reason)
    if ACTIVE_ROBOT is not None:
        ACTIVE_ROBOT.emergency_stop()
    else:
        mbot2.EM_stop("ALL")


def execute_command(robot, payload):
    cmd = str(payload["cmd"])

    if cmd == "START":
        robot.reset_job()
        return

    if cmd == "END":
        robot.finish_job()
        return

    if cmd == "PEN_UP":
        robot.pen_up()
        return

    if cmd == "PEN_DOWN":
        robot.pen_down()
        return

    if cmd == "PING":
        log("PONG")
        return

    if cmd == "MOVE":
        robot.move_to(
            x=float(payload["x"]),
            y=float(payload["y"]),
            speed=float(payload.get("speed", 30.0)),
        )
        return

    log("UNKNOWN_CMD", cmd)


def execute_payload_safe(robot, payload, source_label):
    try:
        if not isinstance(payload, dict):
            log("PAYLOAD_ERROR", source_label, "payload is not an object")
            return
        if "cmd" not in payload:
            log("PAYLOAD_ERROR", source_label, "missing cmd")
            return
        execute_command(robot, payload)
    except Exception as exc:
        log("COMMAND_ERROR", source_label, exc)


def process_text_line(robot, line, source_label):
    stripped = line.strip()
    if not stripped:
        return

    try:
        payload = json.loads(stripped)
    except ValueError as exc:
        log("JSON_ERROR", source_label, exc)
        return

    execute_payload_safe(robot, payload, source_label)


def handle_client_connection(robot, conn, addr):
    conn.settimeout(CLIENT_READ_TIMEOUT_S)
    buffer = ""
    line_number = 0

    while SERVER_RUNNING and not STOP_REQUESTED:
        try:
            chunk = conn.recv(4096)
        except socket.timeout:
            continue
        except OSError as exc:
            log("CONNECTION_ERROR", addr, exc)
            return

        if not chunk:
            break

        buffer += chunk.decode("utf-8", errors="replace")
        if len(buffer) > MAX_BUFFER_CHARS:
            log("BUFFER_ERROR", addr, "dropping oversized input buffer")
            buffer = ""
            continue

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            line_number += 1
            process_text_line(robot, line, f"{addr} line {line_number}")

    if buffer.strip():
        line_number += 1
        process_text_line(robot, buffer, f"{addr} line {line_number}")


def run_socket_server():
    global ACTIVE_ROBOT, SERVER_RUNNING, STOP_REQUESTED

    if SERVER_RUNNING:
        log("SERVER_ALREADY_RUNNING", HOST, PORT)
        return

    robot = RobotAdapter()
    ACTIVE_ROBOT = robot
    SERVER_RUNNING = True
    STOP_REQUESTED = False

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(1)
    server.settimeout(SERVER_ACCEPT_TIMEOUT_S)

    log("SERVER_LISTENING", HOST, PORT, "press B to request stop")

    try:
        while SERVER_RUNNING and not STOP_REQUESTED:
            try:
                conn, addr = server.accept()
            except socket.timeout:
                continue
            except OSError as exc:
                if STOP_REQUESTED:
                    break
                log("ACCEPT_ERROR", exc)
                continue

            log("CLIENT_CONNECTED", addr)
            try:
                handle_client_connection(robot, conn, addr)
            finally:
                conn.close()
                log("CLIENT_DISCONNECTED", addr)
    finally:
        server.close()
        robot.emergency_stop()
        ACTIVE_ROBOT = None
        SERVER_RUNNING = False
        STOP_REQUESTED = False
        log("SERVER_STOPPED")


@event.is_press("a")
def start_socket_server():
    run_socket_server()


@event.is_press("b")
def stop_socket_server():
    request_stop("button_b")
