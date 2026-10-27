"""Save MediSCAM activation maps and image overlays."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image


def save_cam(cam: torch.Tensor, path: str | Path) -> None:
    """Save a [1, 1, H, W] activation map as an 8-bit grayscale PNG."""
    values = cam.detach().cpu().squeeze(0).squeeze(0).numpy()
    pixels = np.rint(np.clip(values, 0, 1) * 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode="L").save(path)


def save_overlay(image: Image.Image, cam: torch.Tensor, path: str | Path,
                 opacity: float = 0.45) -> None:
    """Save a red activation overlay aligned to the model's image crop."""
    image = image.convert("RGB")
    values = cam.detach().cpu().squeeze(0).squeeze(0).numpy()
    if image.size != (values.shape[1], values.shape[0]):
        image = image.resize((values.shape[1], values.shape[0]), Image.Resampling.BILINEAR)
    base = np.asarray(image, dtype=np.float32)
    alpha = np.clip(values, 0, 1)[..., None] * opacity
    red = np.zeros_like(base)
    red[..., 0] = 255
    result = np.rint(base * (1 - alpha) + red * alpha).clip(0, 255).astype(np.uint8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(result, mode="RGB").save(path)
