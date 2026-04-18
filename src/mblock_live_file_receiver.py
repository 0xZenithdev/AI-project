# generated bridge template for mBlock5 + CyberPi + mBot2

import cyberpi
import event
import json
import math
import mbot2
import time


COMMAND_FILE = r"E:\Yara 2025\LU 3rd year CS\2nd sem\I3335(intro to ai)\AI-project\output\bridge_commands.jsonl"
USE_PEN = False
# Use the actual servo port wired to the pen lift: S1, S2, S3, or S4.
SERVO_PORT = "S1"

# Pen-lift convention confirmed by the user:
# - positive servo delta raises the pen
# - negative servo delta lowers the pen
# Memory note:
# keep this constant block aligned with the socket receiver, inline test
# receiver, and PROJECT_MEMORY.md so future sessions do not drift.
PEN_LIFT_DELTA = 60

TURN_SIGN = 1.0
TURN_SCALE = 1.0
MM_PER_STRAIGHT_UNIT = 10.0
PEN_DELAY_S = 1.0
TURN_DELAY_S = 0.1
MAX_LINE_CHARS = 8192

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
        # Memory note: the receiver assumes the pen is already physically UP
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
        self._servo_by(PEN_LIFT_DELTA)
        self.pen_is_down = False
        log("PEN_UP")

    def pen_down(self):
        if not USE_PEN:
            log("PEN_DOWN_SKIPPED")
            return
        if self.pen_is_down:
            return
        self._servo_by(-PEN_LIFT_DELTA)
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

        self._turn_by(turn_delta)
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
    elif cmd == "END":
        robot.finish_job()
    elif cmd == "PEN_UP":
        robot.pen_up()
    elif cmd == "PEN_DOWN":
        robot.pen_down()
    elif cmd == "PING":
        log("PONG")
    elif cmd == "MOVE":
        robot.move_to(
            x=float(payload["x"]),
            y=float(payload["y"]),
            speed=float(payload.get("speed", 30.0)),
        )
    else:
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


def run_file_job():
    global ACTIVE_ROBOT, STOP_REQUESTED

    robot = RobotAdapter()
    ACTIVE_ROBOT = robot
    STOP_REQUESTED = False
    processed_lines = 0

    log("FILE_START", COMMAND_FILE)
    try:
        with open(COMMAND_FILE, "r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if STOP_REQUESTED:
                    log("FILE_STOPPED", f"line {line_number}")
                    break
                if not line.strip():
                    continue
                if len(line) > MAX_LINE_CHARS:
                    log("LINE_TOO_LONG", f"line {line_number}")
                    continue
                processed_lines += 1
                try:
                    payload = json.loads(line)
                except ValueError as exc:
                    log("JSON_ERROR", f"line {line_number}", exc)
                    continue
                execute_payload_safe(robot, payload, f"line {line_number}")
    except FileNotFoundError:
        log("FILE_MISSING", COMMAND_FILE)
    except Exception as exc:
        log("FILE_ERROR", exc)
    finally:
        robot.emergency_stop()
        ACTIVE_ROBOT = None
        STOP_REQUESTED = False
        log("FILE_DONE", f"processed_lines={processed_lines}")


@event.is_press("a")
def run_file_bridge():
    run_file_job()


@event.is_press("b")
def stop_file_bridge():
    request_stop("button_b")
