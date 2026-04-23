# generated inline test for mBlock5 + CyberPi + mBot2

import cyberpi
import event
import math
import mbot2
import time


USE_PEN = False
# Use the actual servo port wired to the pen lift: S1, S2, S3, or S4.
SERVO_PORT = "S1"
# Keep this test receiver aligned with the live receivers when
# pen direction or calibration constants change.
PEN_LIFT_DELTA = 36
TURN_SIGN = 1.0
TURN_SCALE = 1.04
MM_PER_STRAIGHT_UNIT = 10.0
CORNER_LIFT_TURN_DEG = 30.0
PEN_DELAY_S = 1.0
TURN_DELAY_S = 0.1

# Coordinate-based square test:
# this uses our bridge-style MOVE commands and should produce a 20 cm square
# once mm are converted into the units expected by mbot2.straight().
COORD_SQUARE_COMMANDS = [
    {"cmd": "START"},
    {"cmd": "MOVE", "x": 20.0, "y": 20.0, "speed": 60.0},
    {"cmd": "MOVE", "x": 220.0, "y": 20.0, "speed": 35.0},
    {"cmd": "MOVE", "x": 220.0, "y": 220.0, "speed": 35.0},
    {"cmd": "MOVE", "x": 20.0, "y": 220.0, "speed": 35.0},
    {"cmd": "MOVE", "x": 20.0, "y": 20.0, "speed": 35.0},
    {"cmd": "END"},
]


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
        mbot2.EM_stop("ALL")
        print("START")

    def finish_job(self):
        mbot2.EM_stop("ALL")
        self.pen_up()
        print("END")

    def pen_up(self):
        if not USE_PEN:
            print("PEN_UP_SKIPPED")
            return
        if not self.pen_is_down:
            return
        self._servo_by(-PEN_LIFT_DELTA)
        self.pen_is_down = False
        print("PEN_UP")

    def pen_down(self):
        if not USE_PEN:
            print("PEN_DOWN_SKIPPED")
            return
        if self.pen_is_down:
            return
        self._servo_by(+PEN_LIFT_DELTA)
        self.pen_is_down = True
        print("PEN_DOWN")

    def move_to(self, x, y, speed):
        dx = x - self.current_x
        dy = y - self.current_y
        distance_mm = math.sqrt((dx * dx) + (dy * dy))

        if distance_mm < 1.0:
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
        print("MOVE", x, y, speed)

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


def execute_command(robot, payload):
    cmd = payload["cmd"]

    if cmd == "START":
        robot.reset_job()
    elif cmd == "END":
        robot.finish_job()
    elif cmd == "PEN_UP":
        robot.pen_up()
    elif cmd == "PEN_DOWN":
        robot.pen_down()
    elif cmd == "PING":
        print("PONG")
    elif cmd == "MOVE":
        robot.move_to(
            x=float(payload["x"]),
            y=float(payload["y"]),
            speed=float(payload.get("speed", 30.0)),
        )


def run_inline_job(commands):
    robot = RobotAdapter()

    for payload in commands:
        execute_command(robot, payload)


@event.is_press('a')
def is_btn_press():
    for count in range(4):
        mbot2.straight(20)
        mbot2.turn(90)


@event.is_press('b')
def is_btn_press_b():
    run_inline_job(COORD_SQUARE_COMMANDS)
