"""Classifier loading and image preprocessing for MediSCAM examples."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms


LAYERS = {
    "resnet18": ("layer4", "layer3"),
    "vgg19": ("features.35", "features.26"),
}

PREPROCESS = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])


def load_classifier(
    architecture: str,
    num_classes: int,
    checkpoint: str | Path,
    device: str | torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module, torch.nn.Module]:
    """Load a fine-tuned classifier and its target and guidance layers."""
    if architecture == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "vgg19":
        model = models.vgg19(weights=None)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    state = {name.removeprefix("module."): tensor for name, tensor in state.items()}
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    layers = dict(model.named_modules())
    target_name, guide_name = LAYERS[architecture]
    return model, layers[target_name], layers[guide_name]


def load_image(path: str | Path, device: str | torch.device) -> tuple[torch.Tensor, Image.Image]:
    """Apply the paper's evaluation preprocessing and return its display crop."""
    image = Image.open(path).convert("RGB")
    resized = transforms.Resize(256)(image)
    crop = transforms.CenterCrop(224)(resized)
    return PREPROCESS(image).unsqueeze(0).to(device), crop
