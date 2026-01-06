# Autopilot

Camera-based 3D Object Detection Framework for Autonomous Driving.

Built with PyTorch Lightning and Hydra for modular, configurable training pipelines.

## Architecture Overview

This framework implements an end-to-end perception pipeline that transforms multi-camera images into 3D bounding box predictions using Bird's Eye View (BEV) representation.

### Pipeline Flow

```
Multi-Camera Images (6 views)
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  ResNet18 Backbone (per camera)                         │
│  - Extracts multi-scale features [64, 128, 256, 512]    │
│  - Strides: [4, 8, 16, 32]                              │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  BiFPN Neck (per camera)                                │
│  - Bidirectional feature fusion                         │
│  - Outputs: 4 levels x 128 channels each                │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  BEV Transformer                                        │
│  - Cross-attention: camera features -> BEV queries      │
│  - Self-attention on BEV features                       │
│  - Output: (B, 256, 20, 80) BEV feature map             │
└─────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Detection Head (CenterPoint-style)                     │
│  - Heatmap: (B, 10, 20, 80) class probabilities         │
│  - Regression: center, height, size, rotation           │
└─────────────────────────────────────────────────────────┘
         │
         ▼
    3D Bounding Boxes
```

### Model Components

#### 1. ResNet Backbone ([autopilot/models/backbones/resnet.py](autopilot/models/backbones/resnet.py))

- Uses timm pretrained models (ResNet18/34/50)
- Extracts 4 levels of features at different resolutions
- Shared weights across all camera views

```python
# Output channels for ResNet18/34
channels = [64, 128, 256, 512]  # at strides [4, 8, 16, 32]
```

#### 2. BiFPN Neck ([autopilot/models/backbones/bifpn.py](autopilot/models/backbones/bifpn.py))

- Bidirectional Feature Pyramid Network from EfficientDet
- Learnable weighted feature fusion
- Top-down and bottom-up pathways

```python
# Configuration
in_channels = [64, 128, 256, 512]  # from backbone
out_channels = 128                  # unified output
num_layers = 2                      # BiFPN repetitions
```

#### 3. BEV Transformer ([autopilot/models/necks/bev_transformer.py](autopilot/models/necks/bev_transformer.py))

Transforms multi-camera 2D features into unified BEV representation:

1. **Query Generation**: Creates BEV queries from context + positional encoding
2. **Cross-Attention**: Attends to all camera features with camera/level embeddings
3. **Self-Attention**: Refines BEV features spatially
4. **Output Projection**: Final BEV feature map

```python
# BEV grid configuration
bev_h, bev_w = 20, 80   # BEV spatial dimensions
embed_dim = 256         # feature dimension
num_layers = 2          # transformer layers
num_heads = 8           # attention heads
```

#### 4. Detection Head ([autopilot/models/heads/detection/center_head.py](autopilot/models/heads/detection/center_head.py))

CenterPoint-style anchor-free detection:

- **Heatmap**: Focal loss for object center classification
- **Regression**: L1 loss for box parameters (x, y, z, l, w, h, sin, cos)

### Loss Functions ([autopilot/losses/detection.py](autopilot/losses/detection.py))

| Loss | Target | Description |
|------|--------|-------------|
| FocalLoss | Heatmap | Class-balanced focal loss (alpha=2, gamma=4) |
| RegLoss | Box params | L1 loss at positive locations |

## Project Structure

```
autopilot/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main config entry point
│   ├── dataset/               # Dataset configs (nuscenes, waymo)
│   ├── model/                 # Model component configs
│   │   ├── backbone/          # ResNet, RegNet configs
│   │   └── neck/              # BiFPN configs
│   └── training/              # Training hyperparameters
│
├── autopilot/                  # Main Python package
│   ├── datasets/              # Dataset implementations
│   │   ├── nuscenes.py        # NuScenes dataset loader
│   │   └── waymo.py           # Waymo dataset loader
│   ├── models/                # Neural network modules
│   │   ├── backbones/         # Feature extractors
│   │   │   ├── resnet.py      # ResNet backbone
│   │   │   └── bifpn.py       # BiFPN neck
│   │   ├── necks/             # Feature aggregation
│   │   │   └── bev_transformer.py
│   │   ├── heads/             # Task-specific heads
│   │   │   └── detection/     # 3D detection head
│   │   └── perception.py      # End-to-end model
│   ├── losses/                # Loss functions
│   │   └── detection.py       # Focal + L1 losses
│   └── utils/                 # Utilities
│       ├── registry.py        # Component registry
│       ├── visualization.py   # Visualization tools
│       └── callbacks.py       # Training callbacks
│
├── scripts/                   # Entry point scripts
│   ├── train.py              # Training script
│   ├── evaluate.py           # Evaluation with visualization
│   └── inference.py          # Single-sample inference
│
├── tools/                     # Data processing tools
│   └── data_converter/        # Dataset converters
│       └── nuscenes_converter.py
│
└── tests/                     # Unit tests
```

## Installation

### Requirements

- Python >= 3.10
- PyTorch >= 2.0
- CUDA >= 12.0 (for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/your-org/autopilot.git
cd autopilot

# Create conda environment
conda create -n autopilot python=3.10 -y
conda activate autopilot

# Install PyTorch (CUDA 12.x)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install dependencies
pip install -e .

# For nuScenes dataset support
pip install nuscenes-devkit
```

## Data Preparation

### nuScenes Dataset

1. Download nuScenes dataset from [nuscenes.org](https://www.nuscenes.org/)

2. Create symlink to data directory:
```bash
ln -s /path/to/nuscenes data/nuscenes
```

3. Generate annotation files:
```bash
python tools/data_converter/nuscenes_converter.py \
    --root-path data/nuscenes \
    --version v1.0-trainval
```

This creates:
- `data/nuscenes/nuscenes_infos_train.pkl`
- `data/nuscenes/nuscenes_infos_val.pkl`

**Note**: The converter automatically validates image existence and skips samples with missing files.

## Training

### Basic Training

```bash
# Train with default config
python scripts/train.py

# Train with specific number of epochs
python scripts/train.py training.max_epochs=24

# Train with different batch size (adjust for GPU memory)
python scripts/train.py dataset.batch_size=2 training.accumulate_grad_batches=2
```

### Configuration Override Examples

```bash
# Use different backbone
python scripts/train.py backbone.model_name=resnet34

# Adjust learning rate
python scripts/train.py training.optimizer.lr=1e-4

# Multi-GPU training
python scripts/train.py hardware.gpus=2
```

### Memory Optimization

For GPUs with limited memory (e.g., 16GB):

```yaml
# configs/dataset/nuscenes.yaml
batch_size: 1

# configs/training/default.yaml
accumulate_grad_batches: 4  # effective batch_size = 4
```

## Evaluation

```bash
# Evaluate checkpoint
python scripts/evaluate.py checkpoint=checkpoints/epoch=05.ckpt

# With visualization
python scripts/evaluate.py \
    checkpoint=checkpoints/epoch=05.ckpt \
    visualize=true \
    output_dir=results/
```

## Inference

```bash
# Run inference on images
python scripts/inference.py \
    checkpoint=checkpoints/epoch=05.ckpt \
    input=path/to/images/
```

## Configuration System

This project uses [Hydra](https://hydra.cc/) for configuration management.

### Main Config ([configs/config.yaml](configs/config.yaml))

```yaml
defaults:
  - dataset: nuscenes      # Dataset config
  - training: default      # Training config
  - _self_

seed: 42
hardware:
  gpus: 1
  num_workers: 8
  precision: 16            # Mixed precision training
```

### Dataset Config ([configs/dataset/nuscenes.yaml](configs/dataset/nuscenes.yaml))

```yaml
data_root: data/nuscenes
classes: [car, truck, trailer, bus, ...]
img_size: [640, 960]       # [H, W]
point_cloud_range: [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
batch_size: 1
```

### Training Config ([configs/training/default.yaml](configs/training/default.yaml))

```yaml
max_epochs: 24
optimizer:
  type: AdamW
  lr: 2.0e-4
  weight_decay: 0.01
scheduler:
  type: CosineAnnealingLR
gradient_clip_val: 35.0
accumulate_grad_batches: 4
```

## Registry System

Components are registered using a decorator pattern (similar to mmdetection):

```python
from autopilot.utils.registry import BACKBONES

@BACKBONES.register_module()
class MyBackbone(nn.Module):
    def __init__(self, **kwargs):
        ...

# Build from config
backbone = BACKBONES.build({'type': 'MyBackbone', ...})
```

Available registries:
- `BACKBONES`: Feature extractors
- `NECKS`: Feature aggregation modules
- `HEADS`: Task-specific heads
- `LOSSES`: Loss functions

## Supported Classes (nuScenes)

| Index | Class | Description |
|-------|-------|-------------|
| 0 | car | Cars, vans |
| 1 | truck | Trucks |
| 2 | trailer | Trailers |
| 3 | bus | Buses |
| 4 | construction_vehicle | Construction vehicles |
| 5 | bicycle | Bicycles |
| 6 | motorcycle | Motorcycles |
| 7 | pedestrian | Pedestrians |
| 8 | traffic_cone | Traffic cones |
| 9 | barrier | Barriers |

## Troubleshooting

### CUDA Out of Memory

Reduce batch size and use gradient accumulation:

```bash
python scripts/train.py dataset.batch_size=1 training.accumulate_grad_batches=4
```

### Missing Image Warnings

Regenerate annotation files to filter missing images:

```bash
python tools/data_converter/nuscenes_converter.py \
    --root-path data/nuscenes \
    --version v1.0-trainval
```

### Slow Data Loading

Increase number of workers:

```bash
python scripts/train.py hardware.num_workers=16
```

## License

Apache License 2.0

## Acknowledgements

- [nuScenes](https://www.nuscenes.org/) dataset
- [CenterPoint](https://github.com/tianweiy/CenterPoint) detection head design
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer) BEV transformer architecture
- [timm](https://github.com/huggingface/pytorch-image-models) pretrained backbones
- [EfficientDet](https://arxiv.org/abs/1911.09070) BiFPN design
