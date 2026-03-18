# Dataset Format for Line-Learning Model

This project trains a model to learn:

`input image -> binary line mask`

## Folder Structure

Create this structure:

- `dataset/images/`
- `dataset/masks/`

Each file in `dataset/images/` must have a matching mask in `dataset/masks/` with the same filename stem.

Examples:

- `dataset/images/cat_01.jpg`
- `dataset/masks/cat_01.png`

- `dataset/images/logo_A.jpeg`
- `dataset/masks/logo_A.png`

Extensions can differ. The stem must match.

## Mask Labeling Rules (Important)

Masks must be binary (2 classes only):

- **White (255)** = line to draw
- **Black (0)** = background

No grayscale shading in masks.

## Recommended Data Guidelines

- Start with at least 100 paired samples for a basic result.
- More diversity is better (different shapes, thicknesses, lighting, backgrounds).
- Keep labels clean and consistent.
- Avoid tiny disconnected noise in masks.

## Suggested Labeling Tools

- CVAT
- Labelme
- Roboflow Annotate
- Even simple image editors (if done carefully)

## Quick Validation Checklist

Before training, verify:

1. Every image has one mask with the same stem.
2. Masks are single-channel binary-looking images.
3. White means line, black means background.
4. No accidental inverted labels.

## Training Command

From project root:

```powershell
python -m src.train --images-dir dataset/images --masks-dir dataset/masks --epochs 20 --output models/line_model.pt
```

## Inference Command

```powershell
python -m src.infer --image images/Testlogo.jpeg --checkpoint models/line_model.pt --output output/ml_pred_mask.png
```

## Common Mistakes

- Mismatched names: `cat1.jpg` with `cat_1.png` (not matched).
- Using colored masks instead of binary black/white.
- Inverted meaning (white background and black lines).
- Too little data and expecting generalization.
