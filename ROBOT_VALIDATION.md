# Robot Validation

Use this file when comparing real robot behavior before locking in settings.

For broader project context, also read:

- `PROJECT_MEMORY.md`
- `MBLOCK_BRIDGE.md`

## Goal

Compare two versions of the same image on the real robot:

1. `default`
   Conservative cleanup enabled
2. `raw`
   Cleanup disabled

This tells us whether the new planning cleanup improves drawing quality on the
actual robot, not just in code summaries.

## One-Command Preparation

Generate both validation runs:

```powershell
python -m src.prepare_validation_runs --image images/Testlogo.jpeg --output-dir output/validation
```

This writes:

- `output/validation/default/plot_commands.txt`
- `output/validation/default/paths_summary.json`
- `output/validation/raw/plot_commands.txt`
- `output/validation/raw/paths_summary.json`
- `output/validation/validation_summary.json`

## Socket Mode Commands

Start the live socket receiver in mBlock, then send the default run:

```powershell
python -m src.bridge_sender `
  --mode socket `
  --commands-file output/validation/default/plot_commands.txt `
  --connect-attempts 20 `
  --retry-delay-ms 500 `
  --prepend-ping
```

Send the raw run:

```powershell
python -m src.bridge_sender `
  --mode socket `
  --commands-file output/validation/raw/plot_commands.txt `
  --connect-attempts 20 `
  --retry-delay-ms 500 `
  --prepend-ping
```

## File Mode Commands

If socket mode is unreliable, write the default run to the bridge file:

```powershell
python -m src.bridge_sender `
  --mode file `
  --commands-file output/validation/default/plot_commands.txt `
  --outbox output/bridge_commands.jsonl `
  --prepend-ping
```

Write the raw run:

```powershell
python -m src.bridge_sender `
  --mode file `
  --commands-file output/validation/raw/plot_commands.txt `
  --outbox output/bridge_commands.jsonl `
  --prepend-ping
```

## Pre-Run Checklist

- Pen starts physically UP before the job.
- `SERVO_PORT` matches your real servo connection.
- `PEN_LIFT_DELTA`, `MM_PER_STRAIGHT_UNIT`, and `TURN_SCALE` are set to the
  latest known calibration values.
- Paper is fixed in place and robot start position is repeatable.
- You know whether you are testing `default` or `raw`.

## During-Run Checklist

For each run, observe:

- Does the pen lift cleanly with no accidental connecting lines?
- Does the robot hesitate or jitter on small segments?
- Do corners still look sharp and intentional?
- Does the overall shape stay recognizable?
- Does the run finish faster or smoother?

## Results Checklist

Record these for both `default` and `raw`:

- Visual quality: which one looks cleaner?
- Jitter: which one jitters less?
- Detail preservation: did cleanup remove anything important?
- Runtime feel: which one is more stable?
- Corner quality: which one keeps corners better?

## Decision Rule

Keep `default` if:

- it reduces jitter
- it keeps the important shape details
- it does not visibly damage corners or structure

Tune cleanup thresholds only if:

- default removes important detail
- or default still leaves too much jitter

## ML Note

Robot validation and ML are related but separate:

- The bridge/calibration/live-mode stack is now good enough to support ML work.
- The ML pipeline code already exists.
- What still blocks useful ML results is training data and a trained checkpoint,
  not the robot bridge structure itself.
