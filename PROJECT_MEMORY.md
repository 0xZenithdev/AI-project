# Project Memory

This file is the persistent memory for the project so future sessions can
recover context quickly without re-discovering key decisions.

## Current Goal

Build an image-to-mBot2 drawing pipeline where:

1. Local Python does the heavy work:
   - image processing
   - path planning
   - command generation
2. mBlock / CyberPi does the light work:
   - receive movement commands
   - move the robot
   - raise/lower the pen with a servo

## Current Status

- The local image pipeline exists and exports drawing commands.
- Drawing quality cleanup has been added in the planner:
  - conservative point simplification
  - tiny-segment collapse
  - tiny-path filtering
  - redundant move suppression during command generation
- Live mode bridge exists for socket mode and file mode.
- Live mode hardening has been added:
  - socket retries in the sender
  - atomic file writes in file mode
  - safer stop behavior on button `B`
  - malformed payload logging instead of full-session crashes
- Compiled `.mcode` generation exists.
- Calibration pack has been added and should be used before serious drawing.
- Robot validation workflow now exists:
  - `src.prepare_validation_runs`
  - `ROBOT_VALIDATION.md`
- Local browser UI now exists:
  - `src.ui_server.py`
  - `ui/index.html`
  - `UI_WORKFLOW.md`
- Course presentation / report pack now exists:
  - `src.build_course_pack.py`
  - `COURSE_PRESENTATION_PACK.md`
- AI experiment / report support now exists:
  - `src.evaluate_pipeline.py`
  - `AI_EXPERIMENTS.md`
  - `ML_DATASET_SOURCES.md`
- ML data-preparation workflow now exists:
  - `src.ml_dataset_utils.py`
  - `src.manage_ml_dataset.py`
  - `ML_DATA_WORKFLOW.md`

## Confirmed Facts From The User

- The pen servo can be on `S1`, `S2`, `S3`, or `S4`.
- Default repo assumption is `S1` unless changed.
- The servo is responsible for bringing the pen up and down.
- Pen convention is:
  - positive `servo_add` raises the pen
  - negative `servo_add` lowers the pen
- The live bridge should be finished before the rest of the workflow.

## Important Runtime Decisions

### Bridge protocol

The shared line-based command vocabulary is intentionally small:

- `START`
- `END`
- `PEN_UP`
- `PEN_DOWN`
- `MOVE x y speed`
- `PING`

Do not casually expand this unless all of these are updated together:

- `src/bridge_protocol.py`
- `src/bridge_sender.py`
- live-mode receiver templates
- calibration/test generators

### Live mode conventions

- Main live receiver file:
  `src/mblock_live_socket_receiver.py`
- File fallback:
  `src/mblock_live_file_receiver.py`
- Live templates use:
  - `PEN_LIFT_DELTA`
  - `MM_PER_STRAIGHT_UNIT`
  - `TURN_SCALE`
- Live templates assume the pen starts in the UP position before a job.
- Live templates track pen state internally to avoid duplicate servo moves.

### Compiled script conventions

- Main generator:
  `src/mblock_script_generator.py`
- Generated scripts must follow the same pen direction convention as live mode.
- Generated scripts now support `TURN_SCALE` too, so live mode and downloaded
  mode can be calibrated consistently.

### Drawing quality conventions

- Main planning/cleanup file:
  `src/search.py`
- Search/planning now has two explicit variants:
  - `nearest_neighbor` greedy baseline
  - `two_opt` TSP-style local-search refinement
- Main pipeline CLI:
  `main.py`
- Validation runner:
  `src/prepare_validation_runs.py`
- Robot validation checklist:
  `ROBOT_VALIDATION.md`
- Conservative cleanup controls now exist for:
  - `min_path_length_mm`
  - `min_segment_length_mm`
  - `simplify_tolerance_mm`
  - `travel_move_threshold_mm`
  - `draw_move_threshold_mm`
- These defaults are meant to reduce robot jitter without deleting obvious shape
  structure.
- Current default planning method is now `two_opt` because it keeps the same
  drawing content while reducing estimated pen-up travel.
- On `images/Testlogo.jpeg`, the default cleanup reduced command count from
  `864` to `741` and point count from `811` to `688` without dropping any full
  paths.
- On `images/Testlogo.jpeg`, planning with `two_opt` reduced estimated pen-up
  travel from `413.84 mm` to `375.66 mm` compared with `nearest_neighbor`.
- The paired validation run generator was executed successfully:
  - summary file: `output/validation/validation_summary.json`
  - recommended first robot test: `default`
  - current reduction on the sample image: `123` commands (`14.24%`)

### AI evaluation conventions

- Main experiment runner:
  `src/evaluate_pipeline.py`
- Main AI-framing / reporting guide:
  `AI_EXPERIMENTS.md`
- Main dataset-research guide:
  `ML_DATASET_SOURCES.md`
- Experiment outputs now include:
  - JSON summary
  - CSV results table
  - Markdown summary
  - robot quality scorecard CSV
- Evaluation metrics now include:
  - mask precision / recall / Dice / IoU when ground-truth masks exist
  - path count
  - command count
  - draw distance
  - pen-up travel distance
  - manual robot quality placeholders for physical tests

### ML readiness

- The ML code path already exists:
  - `src/ml_line_model.py`
  - `src/train.py`
  - `src/infer.py`
  - `src/vision_v2.py` supports `--vision-mode ml`
- The project is structurally ready for ML experiments.
- The repo now has a small seed dataset and a first repo-level checkpoint.
- Current repo reality:
  - `dataset/images/` and `dataset/masks/` contain `2` matched starter pairs
  - `models/line_model.pt` now exists as the first repo-level checkpoint
  - seed-stage ML audit / split / inference artifacts can be regenerated under
    `output/ml_dataset/` when needed
  - `src/check_ml_readiness.py` can be used to verify this quickly in future
    sessions
- Dataset expectations are documented in `dataset/README.md`.
- Dataset preparation is now more practical:
  - `import` external images/masks into `dataset/images` and `dataset/masks`
  - `prepare-photosketch` converts the Photo-Sketching dataset into binary
    masks plus train/val/test manifests
  - `bootstrap` draft masks from raw images using classical CV
  - `audit` mask quality and pair quality
  - `split` reproducible train/val/test manifests
- `src.train.py` now supports manifest-based training for repeatable
  experiments.
- `prepare-photosketch` was smoke-tested on a local synthetic fixture:
  - auto-detected `image/`, `png/`, and `split/` under `--source-root`
  - generated binary masks
  - wrote train/val manifests
  - trained successfully from those manifests for one smoke epoch
- The default `210 x 297` ML image size exposed an off-by-one U-Net skip
  connection bug; `src.ml_line_model.py` now resizes decoder maps to skip
  shapes so training works with the paper-oriented default size.
- Seed-stage ML workflow now verified in the main repo:
  - `sample_logo` was imported as a clean paired sample
  - `test2logo` was added and given a classical bootstrap draft mask
  - audit and readiness reports now show:
    - `2` matched pairs
    - `2` exact binary masks
    - training ready = `true`
    - inference ready = `true`
  - manifest-based training flow was verified successfully
  - `src.train` produced:
    - `models/line_model.pt`
  - `src.infer` now works against the repo checkpoint
  - `main.py --vision-mode ml` is wired end-to-end, but the current seed
    checkpoint is still low-confidence:
    - on `images/Testlogo.jpeg`, threshold `0.5` produced an empty mask
    - lowering to `--ml-threshold 0.1` produced `4` paths and `37` commands
- The main remaining blocker for useful ML results is no longer "can we train
  anything at all"; it is expanding the dataset beyond the tiny seed set and
  improving model confidence on real target images.

### UI status

- The project now has a local browser UI for image upload and preview.
- Entry point:
  - `python -m src.ui_server`
- Main files:
  - `src/ui_server.py`
  - `ui/index.html`
  - `UI_WORKFLOW.md`
- The UI was later simplified on purpose:
  - one primary action first: generate commands
  - advanced tuning hidden behind collapsible sections
  - send/export treated as optional secondary steps
- The UI was then changed again into a guided step-by-step wizard:
  - image selection unlocks generation
  - generation unlocks review and optional actions
  - visible progress indicator shows where the user is in the flow
- The desktop UI layout was later improved again:
  - sticky left status rail
  - open/close control for the status rail
  - step cards use the horizontal screen space better to reduce scrolling
  - the outer page shell was widened so laptop screens are used more fully and
    the preview/result area has more room
- The UI now also supports manual mBlock testing after export:
  - open generated Python directly
  - preview generated Python in the UI
  - copy generated Python for paste into mBlock
- Step 4 now makes the testing split explicit:
  - manual mBlock paste mode first
  - automatic live bridge mode second
- Step 4 now also includes an `AI Experiment Lab`:
  - compare `classical` vs `ml`
  - compare `nearest_neighbor` vs `two_opt`
  - optionally upload a ground-truth mask for Dice / IoU
  - review experiment summary and per-run preview cards
- The UI reuses the existing pipeline, bridge sender, and `.mcode` generator
  rather than re-implementing them separately.
- Verified locally:
  - image upload and processing
  - preview generation
  - `plot_commands.txt` generation
  - `.mcode` export
  - file-mode send through the UI API
  - AI comparison through `/api/compare`
  - planning comparison summary in the UI

### Course packaging

- A reusable presentation-pack builder now exists:
  - `src.build_course_pack.py`
- It converts one experiment directory into:
  - `overview.md`
  - `slide_outline.md`
  - `speaker_notes.md`
  - `presentation_results_table.csv`
  - `architecture_diagram.mmd`
  - `pack_summary.json`
- Main usage guide:
  - `COURSE_PRESENTATION_PACK.md`
- This keeps the AI-course story reproducible instead of rewriting slides by hand
  each time.

## Calibration Workflow

Calibration is the first major task that is now usable.

Files:

- Generator:
  `src/generate_test_plot_commands.py`
- Measurement-to-constants tool:
  `src/calibration_report.py`
- Generated pack output:
  `output/calibration/`

Recommended calibration order:

1. `pen_gaps`
   Goal: tune `PEN_LIFT_DELTA`
2. `ruler`
   Goal: tune `MM_PER_STRAIGHT_UNIT`
3. `square`
   Goal: tune `TURN_SCALE`

Pack entrypoint:

```powershell
python -m src.generate_test_plot_commands --pack standard --output-dir output/calibration
```

If downloadable calibration projects are needed:

```powershell
python -m src.generate_test_plot_commands --pack standard --emit-mcode --use-pen
```

## Progress Order Agreed With The User

The user chose this order:

1. Calibration pack
2. Live mode hardening
3. Drawing quality upgrade

Current state:

- `1` is started and usable.
- `2` is implemented in code and needs real robot testing.
- `3` is implemented in code and should be validated on real drawings.

## Next Recommended Step

Test the hardened live bridge on the robot, then move to drawing quality
upgrade. The first checks should be:

- confirm button `A` starts the socket/file bridge as expected
- confirm button `B` really stops motion safely in your runtime
- confirm sender retries are enough for your machine
- confirm malformed or partial inputs no longer kill the receiver

After that, validate the drawing-quality defaults on the robot:

- compare the cleaned default output against a no-cleanup run
- check whether corners still look correct after simplification
- tune cleanup thresholds only if the robot still shows visible jitter or detail loss

Once robot validation is acceptable, the ML seed phase does not need to be
started from zero again. The next ML-focused steps are:

- expand `dataset/images` and `dataset/masks` beyond the current `2`-pair seed
  set
- import a serious paired source first, ideally `Photo-Sketching`, then add
  project-specific logo images
- retrain a stronger checkpoint with `src.train`
- re-check ML threshold behavior with `src.infer` and
  `main.py --vision-mode ml --model-checkpoint ...`
- then run `src.evaluate_pipeline` for the classical-vs-ML comparison

## Files To Read First In Future Sessions

1. `PROJECT_MEMORY.md`
2. `MBLOCK_BRIDGE.md`
3. `UI_WORKFLOW.md`
4. `ROBOT_VALIDATION.md`
5. `src/search.py`
6. `src/mblock_live_socket_receiver.py`
7. `src/generate_test_plot_commands.py`
8. `src/calibration_report.py`
9. `src/mblock_script_generator.py`
10. `src/check_ml_readiness.py`

## Things Not To Forget

- The user wants persistent context, not re-discovery every session.
- The pen-direction convention was corrected after explicit user confirmation.
- Socket live mode is the priority path.
- Calibration values should not stay as guesses once robot measurements arrive.
