# Course Presentation Pack

Use this file when you want to turn the project into something easy to present
for the AI course, not only something that runs.

## Goal

Package the project around the framing:

1. `Perception`
2. `Planning`
3. `Action`
4. `Evaluation`

The point of this pack is to make the AI contribution explicit and measurable.

## What Already Exists

The repo now has three layers that work together:

- experiment runner:
  - `python -m src.evaluate_pipeline`
- browser comparison flow:
  - `python -m src.ui_server`
  - Step 4 `AI Experiment Lab`
- presentation-pack builder:
  - `python -m src.build_course_pack`

## Recommended Workflow

### 1. Run experiments

For a planning comparison:

```powershell
python -m src.evaluate_pipeline `
  --image images/Testlogo.jpeg `
  --vision-mode classical `
  --path-ordering nearest_neighbor `
  --path-ordering two_opt
```

For a perception comparison after training an ML checkpoint:

```powershell
python -m src.evaluate_pipeline `
  --image-dir images `
  --vision-mode classical `
  --vision-mode ml `
  --model-checkpoint models/line_model.pt `
  --masks-dir dataset/masks
```

### 2. Or run the same idea from the UI

```powershell
python -m src.ui_server --open-browser
```

Then:

1. upload an image
2. generate the preview
3. go to Step 4 `AI Experiment Lab`
4. choose:
   - `classical`
   - optionally `ml`
   - `nearest_neighbor`
   - `two_opt`
5. optionally upload a ground-truth mask
6. click `Run AI Comparison`

This writes a timestamped comparison folder under the uploaded image session:

- `output/ui_sessions/<session>/experiments/<timestamp>/`

### 3. Build the presentation pack

After you have one experiment folder with `experiment_results.json`, run:

```powershell
python -m src.build_course_pack `
  --experiment-dir output/ai_experiments_smoke/20260417_143953_692363
```

By default this creates:

- `course_pack/overview.md`
- `course_pack/slide_outline.md`
- `course_pack/speaker_notes.md`
- `course_pack/presentation_results_table.csv`
- `course_pack/architecture_diagram.mmd`
- `course_pack/pack_summary.json`

inside the experiment folder.

## What To Put In The Slides

Use the generated files like this:

- `overview.md`
  - short project summary for the report intro or first slide notes
- `slide_outline.md`
  - exact slide structure
- `speaker_notes.md`
  - short talk track for each slide
- `presentation_results_table.csv`
  - clean table for comparisons
- `architecture_diagram.mmd`
  - diagram source for perception -> planning -> action

## Minimum Strong AI Story

If time is limited, the minimum convincing course story is:

1. show the project as `perception + planning + action`
2. show `nearest_neighbor` vs `two_opt`
3. report pen-up travel improvement
4. explain that ML comparison is supported and becomes strong once the real
   dataset and checkpoint are ready

## Stronger AI Story

The stronger version is:

1. classical vs ML perception on the same images
2. nearest-neighbor vs two-opt planning on the same extracted paths
3. automatic metrics:
   - Dice / IoU
   - path count
   - command count
   - pen-up travel distance
4. manual robot metrics:
   - fidelity
   - cleanliness
   - completion

## Verified Example

The current smoke experiment already supports the presentation-pack flow:

- experiment folder:
  - `output/ai_experiments_smoke/20260417_143953_692363`
- current planning win:
  - `two_opt` reduced pen-up travel from `413.84 mm` to `375.66 mm`
  - improvement: `38.18 mm` or `9.23%`

The UI comparison flow was also smoke-tested here:

- `output/ui_sessions/20260418_012418_239022/experiments/20260418_012418_567999`

## Honest Limit

The report/presentation pack is now ready, but the strongest `classical vs ml`
story still depends on gathering a real paired dataset and training a useful
checkpoint.
