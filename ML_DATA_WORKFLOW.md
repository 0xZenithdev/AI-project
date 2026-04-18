# ML Data Workflow

This file is the practical laptop-only workflow for the ML phase.

Use it when you do not have the robot with you and want to move the AI part of
the project forward anyway.

## Goal

Prepare a usable dataset for:

`input image -> binary line mask`

Then train and evaluate the model in a repeatable way.

## Step 1: Import Data

If you already have external images and masks in another folder, import them
into the repo dataset folders with:

```powershell
python -m src.manage_ml_dataset import `
  --source-images-dir path\to\source_images `
  --source-masks-dir path\to\source_masks
```

Optional:

- add `--prefix logos_` to avoid name collisions
- add `--overwrite` to replace existing files
- add `--dry-run` to preview before copying

This writes an import report to:

- `output/ml_dataset/import_report.json`

If some images do not have masks yet, the tool also writes a missing-mask CSV.

## Step 1a: Prepare Photo-Sketching Directly

If you download the `Photo-Sketching` dataset, you do not need to flatten it by
hand first.

The repo now has a dedicated preparation command that:

- reads the source images
- reads the rendered sketch PNGs
- converts sketches into binary masks with the repo mask convention
- preserves official train / val / test splits when split files are available
- writes manifest files you can train from immediately

Example:

```powershell
python -m src.manage_ml_dataset prepare-photosketch `
  --source-root path\to\photosketch_unzipped `
  --output-dir output/photosketch_prepared
```

This expects the unzipped dataset root to contain folders such as:

- `image/`
- `png/`
- `split/`

You can also point to them explicitly:

```powershell
python -m src.manage_ml_dataset prepare-photosketch `
  --source-images-dir path\to\photosketch\image `
  --source-sketches-dir path\to\photosketch\png `
  --split-file path\to\photosketch\split `
  --output-dir output/photosketch_prepared
```

Useful options:

- `--limit-per-image 1` to keep only one drawing per source image for a smaller first run
- `--sketch-mode dark_on_light` if the rendered sketches are black lines on white background
- `--overwrite` to rebuild prepared masks

Outputs:

- `output/photosketch_prepared/masks/`
- `output/photosketch_prepared/splits/train_manifest.json`
- `output/photosketch_prepared/splits/val_manifest.json` if available
- `output/photosketch_prepared/splits/test_manifest.json` if available
- `output/photosketch_prepared/photosketch_prepare_report.json`

Important:

- this path is manifest-first, so it can reuse the original source images
  instead of duplicating every image five times
- if no split file is present, the command falls back to a grouped random split
  by source image ID

## Step 1b: Bootstrap Draft Masks

If you have images but no masks yet, generate draft masks with the classical
vision pipeline:

```powershell
python -m src.manage_ml_dataset bootstrap `
  --source-images-dir dataset/images `
  --masks-dir dataset/masks
```

This is useful when you want to start from a reasonable automatic guess instead
of drawing every mask from zero.

It also writes side-by-side review previews to:

- `output/ml_dataset/bootstrap/previews/`

And a summary report to:

- `output/ml_dataset/bootstrap/bootstrap_report.json`

Important:

- these are draft labels, not guaranteed gold-standard labels
- review or clean them before serious training if possible
- existing masks are not overwritten unless you add `--overwrite`

## Step 2: Audit The Dataset

Before training, audit the dataset quality:

```powershell
python -m src.manage_ml_dataset audit
```

This writes:

- `output/ml_dataset/dataset_audit.json`

The audit checks:

- matched image/mask pairs
- unreadable files
- whether masks look binary or at least binary-like
- suspicious masks that are nearly all black or nearly all white
- image/mask size agreement

## Step 3: Check Readiness

For a shorter readiness summary, run:

```powershell
python -m src.check_ml_readiness --write-report output/ml_readiness.json
```

This now reports:

- whether matched pairs exist
- whether the dataset looks trainable
- whether a checkpoint already exists
- the next recommended action

## Step 4: Create Train/Val/Test Splits

To make training experiments repeatable, generate manifest files:

```powershell
python -m src.manage_ml_dataset split `
  --val-ratio 0.2 `
  --test-ratio 0.1 `
  --seed 42
```

This writes:

- `output/ml_dataset/splits/train_manifest.json`
- `output/ml_dataset/splits/val_manifest.json`
- `output/ml_dataset/splits/test_manifest.json` if test ratio is not zero
- `output/ml_dataset/splits/split_summary.json`

## Step 5: Train From Manifests

Recommended training path:

```powershell
python -m src.train `
  --train-manifest output/ml_dataset/splits/train_manifest.json `
  --val-manifest output/ml_dataset/splits/val_manifest.json `
  --output models/line_model.pt
```

Why manifests are useful:

- the split is fixed and reproducible
- experiments are easier to compare fairly
- later results in the course report are easier to justify

If you prepared `Photo-Sketching`, the training command becomes:

```powershell
python -m src.train `
  --train-manifest output/photosketch_prepared/splits/train_manifest.json `
  --val-manifest output/photosketch_prepared/splits/val_manifest.json `
  --output models/line_model.pt
```

## Step 6: Run ML Inference

After training:

```powershell
python -m src.infer `
  --image images/Testlogo.jpeg `
  --checkpoint models/line_model.pt `
  --output output/ml_pred_mask.png
```

## Step 7: Compare Classical vs ML

After you have a checkpoint, use the experiment runner:

```powershell
python -m src.evaluate_pipeline `
  --image-dir images `
  --vision-mode classical `
  --vision-mode ml `
  --model-checkpoint models/line_model.pt `
  --masks-dir dataset/masks
```

This gives the AI-course comparison you need:

- perception: classical vs ML
- planning: nearest-neighbor vs two-opt
- metrics: Dice, IoU, command count, pen-up travel distance

## Current Repo Seed Status

The repo is no longer at the "empty dataset" stage.

Current local state:

- `dataset/images` and `dataset/masks` contain `2` matched starter pairs
- `models/line_model.pt` exists as a seed checkpoint trained from a
  manifest-based split
- readiness and verification artifacts can be regenerated under
  `output/ml_dataset/` when needed

Important caveat:

- this is a smoke / seed checkpoint, not a serious final model
- on `images/Testlogo.jpeg`, the default ML threshold `0.5` produced an empty
  mask
- lowering to `--ml-threshold 0.1` produced usable contours in the pipeline

Repo smoke-verification command:

```powershell
python main.py `
  --image images/Testlogo.jpeg `
  --vision-mode ml `
  --model-checkpoint models/line_model.pt `
  --ml-threshold 0.1 `
  --output-dir output/ml_dataset/pipeline_verify_t01
```

Use this as proof that the ML path is wired end to end, not as proof that the
dataset problem is solved.

## Recommended Order

The best order for the ML phase is:

1. import data
2. bootstrap missing masks if needed
3. audit data
4. fix bad masks
5. split data
6. train the first checkpoint
7. run evaluation experiments

## Honest Rule

Do not start training just because files exist.

Run the audit first.

A small clean dataset is better than a bigger messy dataset with:

- mismatched stems
- unreadable files
- gray masks instead of binary masks
- nearly empty masks

The bootstrap command is meant to reduce manual effort, not to replace review.
