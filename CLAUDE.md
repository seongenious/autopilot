# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

- E2E autonomous driving network

## Folder structure

```
autopilot/models/
├── backbones/            # Image feature extraction (ResNet, RegNet)
├── necks/                # Feature aggregation (BiFPN, FPN)
├── encoders/
│   ├── bev.py            # PV→BEV transformation
│   └── recurrent.py      # Temporal fusion (BEV state)
├── heads/
│   ├── perception2d/     # 2D perception (panoptic segmentation)
│   ├── perception3d/     # 3D perception (detection, lane)
│   ├── prediction/       # Motion prediction
│   └── planning/         # Trajectory planning
└── detectors/            # End-to-End model wrapper
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

## TODO

[x] hydra-like configuration strategy
[] mmdet-like training strategy
[] create backbone, panoptic segmentation head
[] train panoptic segmentation head
[] wait for next network
