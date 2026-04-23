# Image-to-Path Robot Drawing for mBot2

This project converts an input image into drawing paths and then into robot
commands for an `mBot2` with a pen attachment.

Pipeline:

`image -> extracted paths -> ordered paths -> plot commands -> mBot2 script`

## AI Components

The project is explained through two AI ideas:

- `Perception`
  - `classical`: rule-based image processing
  - `ml`: a small segmentation model that predicts a binary line mask
- `Planning`
  - `nearest_neighbor`: greedy path ordering baseline
  - `two_opt`: route refinement to reduce pen-up travel

For the live demo, the recommended path is `classical + two_opt`.

## Main Files

- [main.py](main.py): runs the full pipeline
- [src/vision_v2.py](src/vision_v2.py): image-to-path extraction
- [src/path_tracing.py](src/path_tracing.py): contour and path tracing
- [src/search.py](src/search.py): path cleanup and ordering
- [src/ui_server.py](src/ui_server.py): local web interface
- [src/mblock_script_generator.py](src/mblock_script_generator.py): export to CyberPi/mBlock
- [src/train.py](src/train.py): ML training entry point
- [src/infer.py](src/infer.py): ML inference entry point
- [dataset/README.md](dataset/README.md): dataset format for the ML part

## Run

UI:

```powershell
python -m src.ui_server --open-browser
```

CLI:

```powershell
python main.py --image images/test2logo.webp
```

Train:

```powershell
python -m src.train --images-dir dataset/images --masks-dir dataset/masks --output models/line_model.pt
```

Inference:

```powershell
python -m src.infer --image images/test2logo.webp --checkpoint models/line_model_presentation_filledmask_tuned.pt
```

## Demo Images

Recommended inputs are in:

- `images/test2logo.webp`
- `images/Testlogo.jpeg`
- `images/presentation_cases/`

These are simple logo-like shapes that are easier to explain and draw
reliably on the robot.
