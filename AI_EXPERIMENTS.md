# AI Experiments

This file explains how to present and evaluate this project as an AI-course
project rather than only as a robotics integration project.

## AI Framing

The cleanest framing for this repo is:

1. `Perception`
   - understand an input image and extract drawable structure
2. `Planning`
   - decide a drawing order that reduces non-drawing robot travel
3. `Action`
   - convert the plan into robot commands and execute them on mBot2/CyberPi

That framing maps directly to the code:

- `Perception`
  - `src/vision_v2.py`
  - `src/ml_line_model.py`
  - `src/train.py`
  - `src/infer.py`
- `Planning`
  - `src/search.py`
- `Action`
  - `main.py`
  - `src/mblock_script_generator.py`
  - live bridge receivers and sender

## Perception Variants

The project now supports two perception modes:

- `classical`
  - Canny + contour extraction
- `ml`
  - trained line-mask model + contour extraction

These are selected through:

```powershell
python main.py --image images/Testlogo.jpeg --vision-mode classical
python main.py --image images/Testlogo.jpeg --vision-mode ml --model-checkpoint models/line_model.pt
```

## Planning Variants

The project now supports two planning/search strategies:

- `nearest_neighbor`
  - greedy baseline
- `two_opt`
  - TSP-style local search refinement over the greedy route

These are selected through:

```powershell
python main.py --image images/Testlogo.jpeg --path-ordering nearest_neighbor
python main.py --image images/Testlogo.jpeg --path-ordering two_opt
```

Why this matters for the AI story:

- `nearest_neighbor` is a heuristic planner baseline
- `two_opt` adds explicit route optimization instead of stopping at greedy
- the optimized objective is physically meaningful for the robot:
  reduce `pen-up travel distance`

## Experiment Runner

Use the experiment runner to compare perception and planning variants on the
same images:

```powershell
python -m src.evaluate_pipeline --image images/Testlogo.jpeg
```

This generates:

- `experiment_results.json`
- `experiment_results.csv`
- `robot_quality_scorecard.csv`
- `experiment_summary.md`

inside a timestamped folder under:

- `output/ai_experiments/`

If you prefer a browser flow, the same idea is now available in the UI:

```powershell
python -m src.ui_server --open-browser
```

Then go to Step 4 `AI Experiment Lab` and run the comparison there.

### Example comparisons

Compare planning methods on one image:

```powershell
python -m src.evaluate_pipeline `
  --image images/Testlogo.jpeg `
  --vision-mode classical `
  --path-ordering nearest_neighbor `
  --path-ordering two_opt
```

Compare classical vs ML on the same images:

```powershell
python -m src.evaluate_pipeline `
  --image-dir images `
  --vision-mode classical `
  --vision-mode ml `
  --model-checkpoint models/line_model.pt `
  --masks-dir dataset/masks
```

## Metrics

The experiment runner records the following metrics.

### Automatic metrics

- `mask_precision`
- `mask_recall`
- `mask_dice`
- `mask_iou`
- `num_paths`
- `num_commands`
- `cleanup_input_points`
- `cleanup_output_points`
- `draw_distance_mm`
- `pen_up_distance_mm`
- `total_distance_mm`

### Manual robot metrics

The generated `robot_quality_scorecard.csv` is meant to be filled after real
robot runs. It includes blank columns for:

- `robot_fidelity_score_1_to_5`
- `robot_cleanliness_score_1_to_5`
- `robot_completion_score_1_to_5`
- `robot_notes`

This lets the project evaluate both:

- algorithm quality on the laptop
- physical drawing quality on the robot

## Current Search Improvement

On the current sample image `images/Testlogo.jpeg`, the stronger planning
method already improved the estimated pen-up travel:

- `nearest_neighbor`: `413.84 mm`
- `two_opt`: `375.66 mm`

That is an improvement of about `38.18 mm` or `9.23%` with the same draw
distance and the same command count.

## Suggested Report Structure

If you need to explain this project in a report or presentation, the strongest
structure is:

1. Problem
   - convert images into efficient robot drawing behavior
2. Perception
   - classical CV baseline
   - learned segmentation alternative
3. Planning
   - greedy baseline
   - TSP-style local search improvement
4. Action
   - convert planned paths to robot commands
5. Evaluation
   - mask quality
   - path/command complexity
   - pen-up travel distance
   - robot drawing quality
6. Results
   - classical vs ML
   - nearest-neighbor vs two-opt
7. Limitations / future work
   - more data
   - stronger models
   - richer planning

## Honest Current Status

The repo is now structurally strong for an AI course:

- AI perception exists
- AI-style planning exists
- evaluation tooling now exists

The remaining gap is ML data and training:

- the ML code path is ready
- the experiment path is ready
- but useful ML comparison still requires a real dataset and a trained
  checkpoint

## Turn Results Into Slides

After you generate one experiment folder, you can build a course-facing report
pack automatically:

```powershell
python -m src.build_course_pack `
  --experiment-dir output/ai_experiments_smoke/20260417_143953_692363
```

That creates:

- `overview.md`
- `slide_outline.md`
- `speaker_notes.md`
- `presentation_results_table.csv`
- `architecture_diagram.mmd`

inside `course_pack/` under the chosen experiment folder.
