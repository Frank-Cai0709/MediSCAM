"""Generate MediSCAM maps for every image in a folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mediscam import MediSCAM
from mediscam.models import load_classifier, load_image
from mediscam.visualization import save_cam, save_overlay


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MediSCAM maps for a folder.")
    parser.add_argument("--architecture", choices=("resnet18", "vgg19"), required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-class", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, target_layer, guide_layer = load_classifier(
        args.architecture, args.classes, args.checkpoint, args.device)
    generator = MediSCAM(model, target_layer, guide_layer)
    output_root = args.output_dir.resolve()
    paths = sorted(path for path in args.input_dir.rglob("*")
                   if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
                   and not path.resolve().is_relative_to(output_root))
    try:
        for image_path in paths:
            image_tensor, display_crop = load_image(image_path, args.device)
            cam = generator(image_tensor, target_class=args.target_class)
            relative = image_path.relative_to(args.input_dir)
            filename = relative.name + ".png"
            save_cam(cam, args.output_dir / "cam" / relative.parent / filename)
            save_overlay(display_crop, cam,
                         args.output_dir / "overlay" / relative.parent / filename)
    finally:
        generator.close()
    print(f"Generated {len(paths)} CAMs in {args.output_dir}")


if __name__ == "__main__":
    main()
