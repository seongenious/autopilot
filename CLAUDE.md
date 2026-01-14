# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

- E2E autonomous driving network

## Folder structure

```
autopilot/
├── models/
│   ├── backbones/            # Image feature extraction (ResNet, RegNet)
│   ├── necks/                # Feature aggregation (BiFPN)
│   ├── encoders/
│   │   ├── bev.py            # PV→BEV transformation
│   │   └── recurrent.py      # Temporal fusion (BEV state)
│   ├── heads/
│   │   ├── perception2d/     # 2D perception (PanopticFPN)
│   │   ├── perception3d/     # 3D perception (detection, lane)
│   │   ├── prediction/       # Motion prediction
│   │   └── planning/         # Trajectory planning
│   ├── detectors/            # End-to-End model wrapper
│   ├── base.py               # BaseModule with weight init
│   └── builder.py            # build_backbone, build_neck, build_head, etc.
├── utils/
│   └── registry.py           # BACKBONES, NECKS, HEADS, ENCODERS, LOSSES
├── engine/
│   └── runner.py             # PyTorch Lightning Runner
└── datasets/                 # Dataset classes (TODO)

configs/
├── config.yaml               # Main config with Hydra defaults
├── model/
│   ├── backbone/             # resnet.yaml, regnet.yaml
│   ├── neck/                 # bifpn.yaml
│   └── head/                 # panoptic.yaml
├── dataset/                  # nuscenes.yaml, waymo.yaml
└── training/                 # default.yaml
```

## Network architecture

```
Image → Backbone → Neck → Features
                           ├→ perception2d heads
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

### Heads (autopilot/models/heads/perception2d/)
- **PanopticFPN**: Panoptic segmentation head (CVPR 2019)
  - SemanticFPNHead: stuff + things semantic segmentation
  - InstanceFPNHead: center heatmap, offset, embedding for instance segmentation
  - Instance mask post-processing with NMS and embedding-based grouping

### Training infrastructure
- **Registry**: mmdet-style module registration (@BACKBONES.register_module())
- **BaseModule**: Weight initialization (Kaiming, Xavier, Pretrained)
- **Builder**: build_backbone(), build_neck(), build_head(), build_loss()
- **Runner**: PyTorch Lightning based training loop

## TODO

[x] hydra-like configuration strategy
[x] mmdet-like training strategy (Registry, Builder, BaseModule, Runner)
[x] create backbone (ResNet, RegNet)
[x] create neck (BiFPN)
[x] create panoptic segmentation head (PanopticFPN)
[] create nuScenes dataset for panoptic training
[] train panoptic segmentation head
[] wait for next network
