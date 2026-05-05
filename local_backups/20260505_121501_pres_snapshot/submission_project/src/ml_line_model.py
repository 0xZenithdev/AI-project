"""Functions and model classes used by the optional ML line-extraction mode."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch import nn

from src.image_geometry import load_image_rgb, resize_with_aspect_pad


@dataclass
class ModelConfig:
    """Image size settings stored with a checkpoint."""

    width: int = 210
    height: int = 297
    resize_mode: str = "aspect_pad"


class ConvBlock(nn.Module):
    """Two convolution layers followed by ReLU activations."""

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
    """Compact U-Net style model for binary line masks."""

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
            # 210x297 is not divisible by 4, so resize decoder maps before concatenation.
            d2 = nn.functional.interpolate(d2, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        if d1.shape[-2:] != e1.shape[-2:]:
            d1 = nn.functional.interpolate(d1, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        return self.head(d1)


def load_model_from_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[TinyUNet, ModelConfig]:
    """Load the saved model and its image-size settings."""
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
    """Run the saved model and return a binary mask."""
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

    return (probs.squeeze().cpu().numpy() >= threshold).astype(np.uint8) * 255
