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
