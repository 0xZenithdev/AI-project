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
SERVO_PORT = "S1"

PEN_UP_ANGLE = 0
PEN_DOWN_ANGLE = 90
TURN_SIGN = 1.0
MM_PER_STRAIGHT_UNIT = 10.0


class RobotAdapter:
    def __init__(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.heading_deg = 0.0

    def reset_job(self):
        self.current_x = 0.0
        self.current_y = 0.0
        self.heading_deg = 0.0
        if USE_PEN:
            self.pen_up()
        print("START")

    def finish_job(self):
        if USE_PEN:
            self.pen_up()
        print("END")

    def pen_up(self):
        if not USE_PEN:
            print("PEN_UP_SKIPPED")
            return
        self._servo_to(PEN_UP_ANGLE)
        print("PEN_UP")

    def pen_down(self):
        if not USE_PEN:
            print("PEN_DOWN_SKIPPED")
            return
        self._servo_to(PEN_DOWN_ANGLE)
        print("PEN_DOWN")

    def move_to(self, x, y, speed):
        dx = x - self.current_x
        dy = y - self.current_y
        distance_mm = math.sqrt((dx * dx) + (dy * dy))

        if distance_mm < 1.0:
            return

        target_heading = math.degrees(math.atan2(-dy, dx))
        turn_delta = self._normalize_angle(target_heading - self.heading_deg)

        self._turn_by(turn_delta)
        mbot2.straight(round(distance_mm / MM_PER_STRAIGHT_UNIT, 2))
        print("MOVE", x, y, speed)

        self.current_x = x
        self.current_y = y
        self.heading_deg = target_heading

    def _servo_to(self, target_angle):
        current_angle = mbot2.servo_get(SERVO_PORT)
        delta = target_angle - current_angle
        if delta != 0:
            mbot2.servo_add(delta, SERVO_PORT)
            time.sleep(0.3)

    def _turn_by(self, delta_deg):
        if abs(delta_deg) < 1.0:
            return

        mbot2.turn(TURN_SIGN * delta_deg)

        self.heading_deg = self._normalize_angle(self.heading_deg + delta_deg)
        time.sleep(0.1)

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

    if cmd == "MOVE":
        robot.move_to(
            x=float(payload["x"]),
            y=float(payload["y"]),
            speed=float(payload.get("speed", 30.0)),
        )


def run_socket_server():
    robot = RobotAdapter()
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((HOST, PORT))
    server.listen(1)

    print("Listening on", HOST, PORT)

    while True:
        conn, addr = server.accept()
        print("Client connected:", addr)
        reader = conn.makefile("r", encoding="utf-8")

        try:
            for line in reader:
                if not line.strip():
                    continue
                payload = json.loads(line)
                execute_command(robot, payload)
        finally:
            conn.close()


@event.is_press('a')
def is_btn_press():
    run_socket_server()
