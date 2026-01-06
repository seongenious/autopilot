# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Development Commands

```bash
# Install package
pip install -e .
pip install -e ".[nuscenes]"  # with nuScenes support

# Training
python scripts/train.py
python scripts/train.py training.max_epochs=24 dataset.batch_size=2

# Testing
pytest tests/ -v                          # all tests
pytest tests/test_backbones.py -v         # single file
pytest tests/test_backbones.py::TestBiFPN -v  # single class

# Code quality
black autopilot/ scripts/ tests/          # format
isort autopilot/ scripts/ tests/          # sort imports
flake8 autopilot/ scripts/ tests/         # lint
mypy autopilot/ --ignore-missing-imports  # type check
```

## Architecture Overview

Camera-based 3D object detection using BEV (Bird's Eye View) representation:

```
6 Camera Images → ResNet18 → BiFPN → BEV Transformer → Detection Head → 3D Boxes
```

### Key Components

**Backbone** (`autopilot/models/backbones/resnet.py`): ResNet18 extracts multi-scale features [64, 128, 256, 512] at strides [4, 8, 16, 32]. Shared across all 6 cameras.

**Neck** (`autopilot/models/backbones/bifpn.py`): BiFPN fuses multi-scale features bidirectionally with learnable weights. Outputs 4 levels at 128 channels each.

**BEV Transformer** (`autopilot/models/necks/bev_transformer.py`): Transforms camera features to BEV using cross-attention. BEV queries (20×80 grid) attend to all camera features. Output: (B, 256, 20, 80).

**Detection Head** (`autopilot/models/heads/detection/center_head.py`): CenterPoint-style anchor-free detection. Predicts heatmap (focal loss) and box regression (L1 loss) for 10 nuScenes classes.

**Main Model** (`autopilot/models/perception.py`): `SimplePerceptionModel` combines all components. Used by `DetectionModule` (PyTorch Lightning) in `scripts/train.py`.

### Registry Pattern

Components are registered via decorators and built from configs:

```python
from autopilot.utils.registry import BACKBONES, NECKS, HEADS, LOSSES

@BACKBONES.register_module()
class MyBackbone(nn.Module): ...

backbone = BACKBONES.build({'type': 'MyBackbone', ...})
```

## Configuration (Hydra)

Configs compose from `configs/config.yaml`:
- `configs/dataset/nuscenes.yaml` - data paths, classes, img_size, batch_size
- `configs/training/default.yaml` - epochs, optimizer, scheduler, gradient clipping

Override via CLI: `python scripts/train.py training.optimizer.lr=1e-4`

## Data Pipeline

1. Generate annotations: `python tools/data_converter/nuscenes_converter.py --root-path data/nuscenes`
2. Dataset (`autopilot/datasets/nuscenes.py`) loads images, applies transforms, creates GT heatmaps
3. Converter validates image existence and skips missing samples

## Memory Constraints

For 16GB GPUs: `batch_size=1` with `accumulate_grad_batches=4` maintains effective batch size of 4.
