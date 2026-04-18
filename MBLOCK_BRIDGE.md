# mBot2 Bridge Plan

This file explains the first bridge layer between our local Python project and
mBlock Live mode.

For persistent project context across sessions, also read `PROJECT_MEMORY.md`.

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

Current hardened behavior:
- button `A` starts the socket server
- button `B` requests an emergency stop / server shutdown
- malformed JSON or bad commands are logged and skipped instead of crashing the session
- the sender now retries the socket connection by default, which helps when the
  mBlock server is still starting up

### 2. File mode

Fallback if socket support is missing or unreliable in mBlock.

Flow:
- run `main.py` to generate `output/plot_commands.txt`
- run `python -m src.bridge_sender --mode file`
- run `src/mblock_live_file_receiver.py` in mBlock Live mode

Current hardened behavior:
- button `A` runs the current bridge file once
- button `B` requests an emergency stop
- malformed lines are logged and skipped instead of aborting the whole file
- the sender now writes the JSONL file atomically, which reduces partial-read issues

## Important limitation

The receiver files are scaffolds. They currently print commands and expose a
`RobotAdapter` class where you will replace placeholder methods with the actual
mBot2 / CyberPi movement and servo calls available in your mBlock environment.

From the generated mBlock Python examples we now know these APIs exist and are
usable in the live runner:

- `import event, time, cyberpi, mbot2, mbuild`
- `@event.is_press("a")`
- `mbot2.forward(speed)`
- `mbot2.backward(speed)`
- `mbot2.turn(angle_degrees)`
- `mbot2.straight(distance_units)`
- `mbot2.servo_set(angle, target)`
- `mbot2.EM_reset_angle("EM1")`
- `mbot2.EM_get_angle("EM1")`
- `mbot2.EM_stop("ALL")`
- `mbuild.ultrasonic2.get(1)`
- `mbuild.quad_rgb_sensor.get_line_sta("all", 1)`
- `mbuild.quad_rgb_sensor.get_white_sta("all", 1)`

The live bridge now uses the small confirmed subset it needs:

- `mbot2.turn(angle_degrees)`
- `mbot2.straight(distance_units)`
- `mbot2.servo_add(delta_angle, target)`
- `mbot2.EM_stop("ALL")`

The current live templates default to `SERVO_PORT = "S1"`. You can switch that
to `S2`, `S3`, or `S4` if your pen servo is wired elsewhere.

That means the main live-mode constants that still need real-world calibration
are:

- `PEN_LIFT_DELTA`
- `MM_PER_STRAIGHT_UNIT`
- `TURN_SCALE`

Pen convention confirmed by the user:

- positive servo delta raises the pen
- negative servo delta lowers the pen

The live templates assume the pen starts in the UP position before the job
begins, then use relative `servo_add()` moves from there.

After inspecting generated mBlock Python for CyberPi + mBot2, we also knew these
APIs exist and are usable in the runner:

- `mbot2.straight(distance)`
- `mbot2.turn_left(speed, seconds)`
- `mbot2.turn_right(speed, seconds)`
- `mbot2.servo_get("S1")`
- `mbot2.servo_add(delta_angle, "S1")`
- `mbot2.EM_set_speed(...)`
- `mbot2.EM_stop(...)`

## Local commands

Generate draw commands:

```powershell
python main.py --image images/Testlogo.jpeg
```

Send them over socket:

```powershell
python -m src.bridge_sender --mode socket --commands-file output/plot_commands.txt
```

Useful socket-mode retry flags:

```powershell
python -m src.bridge_sender --mode socket --connect-attempts 20 --retry-delay-ms 500
```

Write fallback JSONL bridge file:

```powershell
python -m src.bridge_sender --mode file --commands-file output/plot_commands.txt
```

## Calibration Pack

Generate the standard live-mode calibration pack:

```powershell
python -m src.generate_test_plot_commands --pack standard --output-dir output/calibration
```

This writes:

- `output/calibration/calibration_pen_gaps.txt`
- `output/calibration/calibration_ruler.txt`
- `output/calibration/calibration_square.txt`
- `output/calibration/calibration_pack_summary.json`

Recommended order:

1. Run `calibration_pen_gaps.txt` first and increase `PEN_LIFT_DELTA` until the
   short dashes are clearly separated with no connecting marks.
2. Run `calibration_ruler.txt`, measure the drawn lines, and compute a new
   `MM_PER_STRAIGHT_UNIT`.
3. Run `calibration_square.txt` and adjust `TURN_SCALE` until the square closes
   cleanly and the corners look like true 90-degree turns.

You can compute updated constants from your measurements with:

```powershell
python -m src.calibration_report `
  --current-mm-per-straight-unit 10 `
  --distance-observation 50=48 `
  --distance-observation 100=96 `
  --current-turn-scale 1.0 `
  --turn-observation 90=84 `
  --output-json output/calibration/calibration_report.json
```

If you also want downloadable calibration projects, add `--emit-mcode` and
optionally `--use-pen`:

```powershell
python -m src.generate_test_plot_commands --pack standard --emit-mcode --use-pen
```

## Compiled script mode

Recommended for the workflow where code should be generated automatically,
loaded into mBlock, and then downloaded onto the CyberPi.

Flow:
- run `main.py` to generate `output/plot_commands.txt`
- run `python -m src.mblock_script_generator`
- open the generated `output/cyberpi_draw.mcode` in mBlock
- start the job from CyberPi button `A` by default

The generator now writes both:

- `output/cyberpi_draw.py` for inspection/debugging
- `output/cyberpi_draw.mcode` for direct opening in mBlock

On Windows, mBlock is registered to open `.mcode` projects directly, so you can
also ask the generator to launch the project automatically:

```powershell
python -m src.mblock_script_generator --open-in-mblock
```

This mode precomputes turns and distances on the local machine, so the generated
CyberPi code only needs a small confirmed API surface:

- `import event, time, cyberpi, mbot2`
- `@event.start`
- `@event.is_press("a")`
- `mbot2.turn(...)`
- `mbot2.straight(...)`
- `mbot2.servo_add(..., "S1")`
- `mbot2.EM_stop("ALL")`

Current assumption for pen control:
- the pen starts in the UP position before the script runs
- `servo_add(+delta)` raises the pen
- `servo_add(-delta)` lowers the pen

The generated script also supports a `TURN_SCALE` multiplier via:

```powershell
python -m src.mblock_script_generator --turn-scale 1.0
```
