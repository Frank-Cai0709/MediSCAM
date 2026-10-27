"""Generate a MediSCAM map from a fine-tuned ResNet-18 or VGG-19 checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from mediscam import MediSCAM
from mediscam.models import load_classifier, load_image
from mediscam.visualization import save_cam, save_overlay
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a MediSCAM activation map.")
    parser.add_argument("--architecture", choices=("resnet18", "vgg19"), required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overlay", type=Path)
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, target_layer, guide_layer = load_classifier(
        args.architecture, args.classes, args.checkpoint, args.device)
    image_tensor, display_crop = load_image(args.image, args.device)
    generator = MediSCAM(model, target_layer, guide_layer)
    try:
        cam = generator(image_tensor, target_class=args.target_class)
    finally:
        generator.close()

    save_cam(cam, args.output)
    print(args.output)
    if args.overlay:
        save_overlay(display_crop, cam, args.overlay)
        print(args.overlay)


if __name__ == "__main__":
    main()
