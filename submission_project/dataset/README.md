# Dataset Format

The ML task is:

`input image -> binary line mask`

## Required Structure

Use matched pairs with the same filename stem:

- `dataset/images/`
- `dataset/masks/`

Examples:

- `dataset/images/logo_01.jpg`
- `dataset/masks/logo_01.png`

- `dataset/images/icon_a.webp`
- `dataset/masks/icon_a.png`

The file extension may differ, but the stem must match.

## Mask Rules

- white `255` = foreground line
- black `0` = background

Masks should stay binary.

## Training

```powershell
python -m src.train --images-dir dataset/images --masks-dir dataset/masks --output models/line_model.pt
```

## Inference

```powershell
python -m src.infer --image images/test2logo.webp --checkpoint models/line_model_presentation_filledmask_tuned.pt
```

## Report Summary

For the report, the ML part can be described as supervised segmentation:

1. prepare paired image/mask examples
2. train a small U-Net-like model
3. predict a binary line mask for a new image
4. compare the result with the classical baseline
