"""
Minimal learnable model for line extraction.

This module defines:
- A small U-Net-like model (encoder/decoder CNN).
- Dataset loader for paired training data.
- Utility losses and prediction helpers.

Goal:
Given an input image, predict a binary line mask that later gets converted to robot paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from src.image_geometry import load_image_rgb, resize_with_aspect_pad
from src.ml_dataset_utils import (
    load_manifest_pairs,
    match_image_mask_pairs,
)


@dataclass
class ModelConfig:
    """Configuration saved with checkpoints so inference uses matching settings."""

    width: int = 210
    height: int = 297
    resize_mode: str = "aspect_pad"


class ConvBlock(nn.Module):
    """Two conv layers + ReLU used throughout encoder/decoder."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TinyUNet(nn.Module):
    """
    Small U-Net style architecture for binary line segmentation.

    Input:  [B, 3, H, W]
    Output: [B, 1, H, W] logits (apply sigmoid outside for probabilities)
    """

    def __init__(self) -> None:
        super().__init__()

        self.enc1 = ConvBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ConvBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = ConvBlock(64, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)

        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))

        b = self.bottleneck(self.pool2(e2))

        d2 = self.up2(b)
        if d2.shape[-2:] != e2.shape[-2:]:
            # The default training size 210x297 is not divisible by 4.
            # - default training size 210x297 is not divisible by 4
            # - resize decoder maps to the skip-connection shape so the model
            #   works with the repo's paper-oriented default dimensions
            d2 = nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


class LinePairDataset(Dataset):
    """
    Dataset for paired files:
    - input image at dataset/images/<name>.<ext>
    - target line mask at dataset/masks/<same_name>.<ext>

    Mask convention:
    - white/255 means line
    - black/0 means background
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        size: Tuple[int, int] = (210, 297),
        manifest_path: str = "",
    ) -> None:
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.width, self.height = size
        self.manifest_path = manifest_path

        self.samples = self._match_pairs()
        if len(self.samples) == 0:
            raise ValueError(
                "No matching image/mask pairs found. "
                "Expected same filename stems in dataset/images and dataset/masks."
            )

    def _match_pairs(self) -> List[Tuple[Path, Path]]:
        if self.manifest_path:
            pairs = load_manifest_pairs(Path(self.manifest_path))
        else:
            pairs, _, _ = match_image_mask_pairs(images_dir=self.images_dir, masks_dir=self.masks_dir)

        return sorted(pairs, key=lambda x: x[0].name)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]

        # Input image: BGR -> RGB, resize, normalize [0,1], CHW.
        image_rgb = load_image_rgb(img_path)
        image_rgb = resize_with_aspect_pad(
            image_rgb,
            target_size=(self.width, self.height),
            pad_value=(255, 255, 255),
            interpolation=cv2.INTER_AREA,
        )
        image = torch.from_numpy(image_rgb).float().permute(2, 0, 1) / 255.0

        # Target mask: grayscale, resize, binary [0,1], add channel.
        mask_gray = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_gray is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")
        mask_gray = resize_with_aspect_pad(
            mask_gray,
            target_size=(self.width, self.height),
            pad_value=0,
            interpolation=cv2.INTER_NEAREST,
        )
        mask = (mask_gray > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Dice loss helps with class imbalance common in thin-line segmentation."""
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2.0 * intersection + eps) / (union + eps)
    return 1.0 - dice.mean()


def combined_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Blend BCE (pixel-level) and Dice (shape overlap) losses."""
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets)
    dsc = dice_loss(logits, targets)
    return 0.5 * bce + 0.5 * dsc


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TinyUNet, ModelConfig]:
    """Load model weights and stored config from a saved checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = TinyUNet().to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    cfg = ModelConfig(
        width=int(checkpoint.get("width", 210)),
        height=int(checkpoint.get("height", 297)),
        resize_mode=str(checkpoint.get("config", {}).get("resize_mode", "aspect_pad")),
    )

    return model, cfg


def predict_line_mask(
    image_path: str,
    checkpoint_path: str,
    threshold: float = 0.5,
    device_str: str = "cpu",
) -> np.ndarray:
    """
    Run trained model and return a binary line mask (uint8 0/255).

    The returned mask can directly replace Canny output in downstream contour extraction.
    """
    device = torch.device(device_str)
    model, cfg = load_model_from_checkpoint(checkpoint_path, device=device)

    image_rgb = load_image_rgb(image_path)
    image_rgb = resize_with_aspect_pad(
        image_rgb,
        target_size=(cfg.width, cfg.height),
        pad_value=(255, 255, 255),
        interpolation=cv2.INTER_AREA,
    )

    x = torch.from_numpy(image_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    x = x.to(device)

    with torch.no_grad():
        logits = model(x)
        probs = torch.sigmoid(logits)

    mask = (probs.squeeze().cpu().numpy() >= threshold).astype(np.uint8) * 255
    return mask
