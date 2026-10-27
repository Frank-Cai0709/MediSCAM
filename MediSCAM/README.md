# MediSCAM

**MediSCAM: Progressive, Reliability-Constrained Class Activation Mapping for Medical Image Interpretation**

This repository contains the inference-time MediSCAM method. It generates a
class-specific activation map from a trained classifier without changing its
weights. The method follows three stages:

1. **Structure-guided Progressive Upsampling (SPU)** uses an adjacent shallow
   layer to guide deep activation maps, then fuses direct and progressive
   upsampling branches.
2. **Stable Low-Rank Channel Interaction (SLRCI)** scores the upsampled
   channels with masked classifier forwards and combines first-order evidence
   with a reliability-gated, rank-one interaction map.
3. **Contrastive Evidence Refinement (CER)** compares the target map with the
   highest-scoring competing class and adds positive target-dominant evidence.

## Repository layout

```text
MediSCAM/
├── mediscam/
│   ├── core.py              # SPU, SLRCI, and CER
│   ├── models.py            # classifier checkpoints and preprocessing
│   └── visualization.py     # grayscale CAMs and image overlays
├── examples/
│   ├── generate_cam.py      # one image
│   └── generate_folder.py   # a folder of images
├── requirements.txt
└── README.md
```

## Install

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Run with a fine-tuned checkpoint

Run the following commands from the `MediSCAM/` directory.

```bash
python -m examples.generate_cam \
  --architecture resnet18 \
  --classes 2 \
  --checkpoint /path/to/best.pth \
  --image /path/to/image.png \
  --output /path/to/cam.png \
  --overlay /path/to/overlay.png
```

To generate maps for all images in a directory with one model load:

```bash
python -m examples.generate_folder \
  --architecture resnet18 \
  --classes 2 \
  --checkpoint /path/to/best.pth \
  --input-dir /path/to/images \
  --output-dir /path/to/output
```

The example accepts ResNet-18 and VGG-19 checkpoints containing a model state
dictionary, either directly or under a `state_dict` key. Set `--classes` to the
classifier's output count. Pass `--target-class INDEX` to explain a specified
class; otherwise, MediSCAM explains the predicted class. The single-image
command saves a grayscale CAM and, when requested, a red overlay. The folder
command writes both under `cam/` and `overlay/` within the output directory.

You can also call the method from Python:

```python
from mediscam import MediSCAM

generator = MediSCAM(model, model.layer4, model.layer3)
try:
    cam = generator(image_tensor)  # [1, 1, H, W], in [0, 1]
finally:
    generator.close()
```

For VGG-19, use `model.features[35]` as the target layer and
`model.features[26]` as the guidance layer. `image_tensor` is a single
classifier-ready image with shape `[1, 3, H, W]`. The paper's evaluation
preprocessing is `Resize(256)`, `CenterCrop(224)`, and ImageNet normalization.

## Method settings

| Stage | Parameter | Default |
| --- | --- | ---: |
| SPU | Shallow guidance strength `spu_strength` (λₛ) | 0.4 |
| SLRCI | Retained channel count `interaction_topk` (Kᵣ) | 32 |
| SLRCI | Interaction strength `interaction_strength` (λᵢ) | 0.05 |
| SLRCI | Maximum interaction gain `interaction_cap` (ηₘₐₓ) | 0.05 |
| SLRCI | Active-area threshold (τ) | 0.5 |
| SLRCI | Active-area lower bound (aₘᵢₙ) | 0.03 |
| SLRCI | Active-area upper bound (aₘₐₓ) | 0.40 |
| CER | Positive contrast strength `contrast_strength` (γ) | 0.3 |

The paper specifies the default values of λₛ, Kᵣ, and γ. The remaining
numerical settings follow the evaluation implementation used for the paper.

SPU computes a normalized guide from the mean of rectified shallow channels.
Each deep channel receives bounded residual guidance before progressive
upsampling. A Gaussian-filtered direct branch and progressive branch are
combined by an element-wise maximum, followed by `tanh`.

For each upsampled channel, SLRCI obtains class probabilities from the same
masked forward pass for both the target and competing classes. It builds a
rectified first-order map, then takes the top Kᵣ channels for semantic and
edge-weighted projections. Their product forms the interaction map. Spatial
agreement and active area determine its contribution to the final class map.

CER independently normalizes the target and competing maps. It computes the
positive part of their difference and raises target responses where that
contrastive evidence is stronger. The output is a `[0, 1]` activation map at
the input image resolution.
