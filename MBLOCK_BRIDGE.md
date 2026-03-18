# mBot2 Bridge Plan

This file explains the first bridge layer between our local Python project and
mBlock Live mode.

## Why this exists

Our local project uses heavy libraries like OpenCV, NumPy, and Torch. The Python
environment inside mBlock can be more limited, so we keep mBlock's job small:

- receive simple movement commands
- control the robot and pen servo

## Two bridge modes

### 1. Socket mode

Best for real-time control in mBlock Live mode.

Flow:
- run `main.py` to generate `output/plot_commands.txt`
- run `python -m src.bridge_sender --mode socket`
- run `src/mblock_live_socket_receiver.py` in mBlock Live mode

### 2. File mode

Fallback if socket support is missing or unreliable in mBlock.

Flow:
- run `main.py` to generate `output/plot_commands.txt`
- run `python -m src.bridge_sender --mode file`
- run `src/mblock_live_file_receiver.py` in mBlock Live mode

## Important limitation

The receiver files are scaffolds. They currently print commands and expose a
`RobotAdapter` class where you will replace placeholder methods with the actual
mBot2 / CyberPi movement and servo calls available in your mBlock environment.

After inspecting generated mBlock Python for CyberPi + mBot2, we now know these
APIs exist and are usable in the runner:

- `mbot2.straight(distance)`
- `mbot2.turn_left(speed, seconds)`
- `mbot2.turn_right(speed, seconds)`
- `mbot2.servo_get("S1")`
- `mbot2.servo_add(delta_angle, "S1")`
- `mbot2.EM_set_speed(...)`
- `mbot2.EM_stop(...)`

The current runner uses `straight()` plus time-based turning. That means two
constants still need real-world calibration:

- `PEN_UP_ANGLE` / `PEN_DOWN_ANGLE`
- `TURN_DEGREES_PER_SECOND_AT_SPEED_50`

## Local commands

Generate draw commands:

```powershell
python main.py --image images/Testlogo.jpeg
```

Send them over socket:

```powershell
python -m src.bridge_sender --mode socket --commands-file output/plot_commands.txt
```

Write fallback JSONL bridge file:

```powershell
python -m src.bridge_sender --mode file --commands-file output/plot_commands.txt
```
