# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

- E2E autonomous driving network

## Folder structure

```
autopilot/
├── models/
│   ├── backbones/            # Image feature extraction (ResNet, RegNet)
│   ├── necks/                # Feature aggregation (BiFPN, ContextAggregator)
│   ├── detectors/            # End-to-End model wrapper (Panoptic, Depth)
│   ├── encoders/
│   │   ├── bev.py            # PV→BEV transformation
│   │   └── recurrent.py      # Temporal fusion (BEV state)
│   ├── heads/
│   │   ├── perception2d/     # 2D perception (PanopticHead, DepthHead)
│   │   ├── perception3d/     # 3D perception (detection, lane)
│   │   ├── prediction/       # Motion prediction
│   │   └── planning/         # Trajectory planning
│   ├── base.py               # BaseModule with weight init
│   └── builder.py            # build_backbone, build_neck, build_head, etc.
├── utils/
│   └── registry.py           # BACKBONES, NECKS, HEADS, ENCODERS, LOSSES, HOOKS
├── engine/
│   ├── runner.py             # PyTorch Lightning Runner
│   └── hooks/                # Training hooks (VisualizationHook)
└── datasets/                 # CityscapesDataset (panoptic + depth), transforms

configs/
├── pretrain/                 # Pretraining configs
│   ├── panoptic.yaml
│   └── depth.yaml
├── model/
│   ├── backbone/             # resnet.yaml, regnet.yaml
│   ├── neck/                 # bifpn.yaml
│   ├── detector/             # panoptic.yaml, depth.yaml
│   └── encoder/              # bev.yaml, recurrent.yaml
├── dataset/                  # cityscapes.yaml
└── training/                 # default.yaml

tools/
└── pretrain/                 # Training scripts
    ├── train_panoptic.py
    └── train_depth.py
```

## Network architecture

```
Image → Backbone → Neck → Features
                           ├→ ContextAggregator → perception2d heads
                           └→ bev.py → recurrent.py → BEV features
                                                        ├→ perception3d heads
                                                        ├→ prediction heads → tokens ─┐
                                                        └──────────────────────────────┴→ planning head
```

## Implemented modules

### Backbones (autopilot/models/backbones/)

- **ResNet**: ResNet18/34/50/101/152 via timm, pretrained, frozen_stages support
- **RegNet**: RegNetX/Y variants via timm, pretrained, frozen_stages support

### Necks (autopilot/models/necks/)

- **BiFPN**: Bidirectional Feature Pyramid Network, multi-scale feature fusion
- **ContextAggregator**: Multi-scale context aggregation to single scale

### Detectors (autopilot/models/detectors/)

- **Panoptic**: Backbone → Neck → SemContext + InstContext → PanopticHead
- **Depth**: Backbone → Neck → Context → DepthHead

### Heads (autopilot/models/heads/perception2d/)

- **PanopticHead**: Panoptic segmentation head
  - SemanticHead: stuff + things semantic segmentation (19 classes)
  - InstanceHead: class-agnostic center heatmap + offset
- **DepthHead**: Log-bin depth classification
  - Soft targets with Gaussian distribution
  - Cross-entropy loss (KL divergence equivalent)

### Datasets (autopilot/datasets/)

- **CityscapesDataset**: Panoptic + Depth integrated
  - Panoptic: semantic (19 classes), center heatmap, offset
  - Depth: disparity → depth conversion, log-bin soft targets

### Training infrastructure

- **Registry**: mmdet-style module registration (@BACKBONES.register_module())
- **BaseModule**: Weight initialization (Kaiming, Xavier, Pretrained)
- **Builder**: build_backbone(), build_neck(), build_head(), build_loss()
- **Runner**: PyTorch Lightning based training loop
- **Hooks**: VisualizationHook (auto-detect panoptic/depth)

## TODO

[x] hydra-like configuration strategy
[x] mmdet-like training strategy (Registry, Builder, BaseModule, Runner)
[x] create backbone (ResNet, RegNet)
[x] create neck (BiFPN, ContextAggregator)
[x] create panoptic segmentation head (PanopticHead)
[x] create depth estimation head (DepthHead)
[x] create Cityscapes dataset (panoptic + depth)
[] train panoptic segmentation head
[] train depth estimation head
[] wait for next network
