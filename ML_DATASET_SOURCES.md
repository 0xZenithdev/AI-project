# ML Dataset Sources

This file lists practical dataset sources for the ML part of the project.

Goal of the ML model in this repo:

`input image -> binary line mask`

So the best datasets are the ones that provide either:

- paired image-to-contour / image-to-sketch supervision
- strong boundary annotations
- domain-relevant images we can annotate into binary masks

## Best Direct Fits

### 1. Photo-Sketching Contour Drawing Dataset

Link:

- https://mtli.github.io/sketch/

Why it fits:

- this is the closest match to our task
- it provides paired images and human contour drawings
- the target is already close to our required binary line-mask output

Reported on the project page:

- `1,000` outdoor images
- each paired with `5` human drawings
- `5,000` drawings total

Best use in this project:

- first serious supervised dataset for training the line-mask model
- especially good for proving the ML pipeline works end-to-end

Important note:

- the page says the dataset license is `CC BY-NC-SA`

### 2. BSDS500

Links:

- https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/resources.html
- https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/

Why it fits:

- classic benchmark for contour / boundary detection
- gives human boundary annotations
- useful for pretraining or benchmarking edge/contour prediction

Best use in this project:

- pretrain the ML model on contour detection
- compare classical CV against a learned contour model
- evaluate with precision/recall style mask metrics

Important note:

- BSDS500 is boundary-focused, not robot-drawing-specific
- it is still a strong academic baseline source

## Good Supporting Sources For Sketch / Line Learning

### 3. Sketchy Database

Links:

- Georgia Tech graphics page: https://sites.cc.gatech.edu/graphics/
- paper PDF: https://faculty.cc.gatech.edu/~hays/tmp/sketchy-database.pdf

Why it fits:

- photo-sketch pairs are useful for learning cross-domain structure
- sketches can be rasterized into binary masks for our model

Reported in the paper abstract:

- `75,471` sketches
- `12,500` objects
- `125` categories

Best use in this project:

- extra pretraining data for photo-to-sketch structure
- augmentation source when the direct contour dataset is too small

Important note:

- the sketches are more abstract than exact contour labels
- good for representation learning, not as perfect as contour-tracing labels

### 4. Quick, Draw!

Link:

- https://github.com/googlecreativelab/quickdraw-dataset

Why it fits:

- enormous sketch dataset
- vector strokes can be rasterized cleanly into masks
- useful for sketch pretraining and data augmentation

Reported in the dataset repo:

- `50 million` drawings
- `345` categories

Best use in this project:

- pretrain a sketch-aware encoder
- generate synthetic line-mask examples
- augment the model’s understanding of stroke structure

Important note:

- not paired with source photographs
- best as supporting data, not the main supervised dataset for our exact task

### 5. TU-Berlin Sketch Dataset

Link:

- https://cybertron.cg.tu-berlin.de/eitz/projects/classifysketch/

Why it fits:

- classic large sketch dataset
- stroke-style diversity is useful for sketch representation learning

Reported on the project page:

- `20,000` unique sketches
- `250` object categories

Best use in this project:

- auxiliary pretraining
- sketch-style augmentation

Important note:

- sketch classification dataset, not photo-to-mask supervision

## Logo-Focused Sources If The Project Stays Logo-Heavy

If your real test images are mostly logos, then generic contour datasets alone
are not enough. These datasets are useful for collecting logo-domain images,
but they are not direct binary line-mask supervision by themselves.

### 6. QMUL-OpenLogo

Link:

- https://hangsu0730.github.io/qmul-openlogo/

Why it fits:

- real logo-domain images
- better for realistic logo variation than tiny classroom collections

Reported on the project page:

- `27,083` images
- `352` logo classes

Best use in this project:

- source of real logo images for manual mask annotation
- logo-domain fine-tuning set after generic contour pretraining

Important note:

- academic research use only according to the project page
- not direct line-mask labels

### 7. LogoDet-3K

Link:

- https://github.com/Wangjing1551/LogoDet-3K-Dataset

Why it fits:

- large-scale logo image source
- good if the project needs broader brand/logo coverage

Reported in the dataset repo:

- `158,652` images
- `3,000` logo categories
- about `200,000` annotated logo objects

Best use in this project:

- collect logo images to convert or annotate into line masks
- build a domain-specific fine-tuning split

Important note:

- the annotations are for logo detection, not binary line masks

### 8. OSLD

Link:

- https://github.com/mubastan/osld

Why it fits:

- very large logo-class coverage
- includes product images and canonical logo images

Reported in the repo:

- `20K` product images
- `12.1K` logo classes
- `20.8K` canonical logo images

Best use in this project:

- domain-specific image collection
- possible weak-supervision source

Important note:

- research-use license (`CC BY-NC 4.0` according to the repo)
- not direct line-mask labels

## Recommended Training Strategy

For this project, the smartest data strategy is:

1. Start with a dataset that directly matches the task:
   - `Photo-Sketching Contour Drawing Dataset`
2. Use a contour benchmark for stronger contour behavior:
   - `BSDS500`
3. Fine-tune on a small custom dataset made from your real project domain:
   - logos
   - simple drawings
   - images you actually want the robot to draw
4. Use sketch datasets as support / augmentation:
   - `Sketchy`
   - `Quick, Draw!`
   - `TU-Berlin`

## Recommended Project-Specific Dataset Plan

If I were optimizing this repo for the course, I would do:

### Phase 1

Train a first model on:

- `Photo-Sketching`
- optional `BSDS500` support

Purpose:

- prove that the ML path works
- produce the first checkpoint

Repo support for this phase now exists:

```powershell
python -m src.manage_ml_dataset prepare-photosketch `
  --source-root path\to\photosketch_unzipped `
  --output-dir output/photosketch_prepared
```

Then train from the generated manifests:

```powershell
python -m src.train `
  --train-manifest output/photosketch_prepared/splits/train_manifest.json `
  --val-manifest output/photosketch_prepared/splits/val_manifest.json `
  --output models/line_model.pt
```

### Phase 2

Build a custom small dataset of your real target images:

- put images in `dataset/images`
- create binary masks in `dataset/masks`

Suggested size:

- minimum: `100` paired samples
- better: `200-500`

Purpose:

- adapt the model to your actual robot-drawing domain

### Phase 3

Evaluate with the experiment runner:

```powershell
python -m src.evaluate_pipeline `
  --image-dir images `
  --vision-mode classical `
  --vision-mode ml `
  --model-checkpoint models/line_model.pt `
  --masks-dir dataset/masks
```

## Current Repo Status

This repo has already completed a small Phase `0` seed bootstrap:

- `dataset/` now contains `2` matched starter pairs
- `models/line_model.pt` exists as a first seed checkpoint
- the checkpoint works for smoke verification, but on `images/Testlogo.jpeg`
  it needed `--ml-threshold 0.1` to produce usable contours

That means the next serious dataset move is not "prove ML can run at all."

It is:

1. bring in a real external paired source, starting with
   `Photo-Sketching Contour Drawing Dataset`
2. add `BSDS500` if you want stronger contour behavior or a benchmark-style
   comparison
3. grow the project-specific fine-tuning set in `dataset/images` and
   `dataset/masks`

## Honest Bottom Line

There is no single perfect public dataset that exactly equals:

`arbitrary image -> robot-friendly drawing mask`

So the realistic best approach is:

- use a strong public contour/sketch dataset for initial training
- then fine-tune on a small custom dataset made for this project

If collecting custom labels is slow, the repo now also supports a practical
middle step:

- generate draft masks with `python -m src.manage_ml_dataset bootstrap`
- review and correct those masks
- use the cleaned result as the first project-specific fine-tuning set
