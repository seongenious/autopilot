# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Autonomous Driving Network Design Prompt  

*(Final – Context-based VT Prior, Backbone Sampling)*

---

## Objective

Design a **camera-only autonomous driving system** that outputs a **trajectory**, optimized for **long-range highway driving (≥150m)** under a **tight computational budget**, using **sparse BEV queries** and **temporal BEV accumulation**.

The system must:

- Avoid direct raw feature → BEV projection
- Use **task-supervised context features as View Transform (VT) priors**
- Perform **efficient deformable View Transform**
- Maintain temporal stability via **explicit ego-motion warping**
- Keep the planner lightweight by enriching perception representations

---

## Inputs

- 6 calibrated camera images  
- IMU / GPS / vehicle odometry  
- (Optional) navigation map  

---

## Core Design Philosophy

1. **View Transform should not rely on raw features alone**  
   → Construct **task-aware context priors** first.

2. **Depth, semantic, and instance heads do not directly provide priors**  
   → They supervise **context representations**, not VT logic.

3. **Evidence vs guidance separation**  
   - Evidence: backbone / BiFPN features  
   - Guidance: context priors (depth / semantic / instance)

4. **BEV is sparse and temporally accumulated**  
   → Geometry and time consistency are handled in BEV space.

5. **Planning is lightweight**  
   → Intelligence is pushed upstream into perception.

---

## Architecture Overview

---

### 1. Backbone & Multi-scale Feature Extraction

- Shared CNN backbone (e.g., ResNet)
- BiFPN for multi-scale fusion
- Outputs perspective-view features:
  - P2, P3, P4, P5 at resolutions 1/4, 1/8, 1/16, 1/32

---

### 2. Task-specific Context Necks (VT Priors)

Instead of using head outputs directly, construct **three separate context features**.

Each context:

- Takes BiFPN multi-scale features as input
- Aligns all scales to **H/8**
- Uses lightweight fusion (1×1 conv + resize + sum or light concat)
- Refined with 3×3 convolutions (1–2 layers)

#### Context branches

- **Depth context**  
  Encodes geometric structure, depth discontinuity, and scale cues.

- **Semantic context**  
  Encodes road / object / sky-like separation for BEV relevance.

- **Instance context**  
  Encodes boundary, center-ness, and feature mixing suppression.

The three context features are combined channel-wise to form a single **VT prior feature** at 1/8 resolution.

---

### 3. Auxiliary Heads (Context Supervision Only)

Auxiliary heads are **not used directly in View Transform**.  
They exist solely to shape the context representations.

- **Depth head**
  - Self-supervised or weakly supervised
  - Encourages geometric consistency in depth context

- **Semantic head**
  - Coarse semantic classes
  - Shapes semantic context toward gating-relevant structure

- **Instance(-ness) head**
  - Center heatmap, offset, boundary
  - Shapes instance context to reduce feature mixing

Noise or failure in these heads should **not directly break VT**.

---

### 4. Sparse BEV Queries

- Fixed query budget: **Nq ≈ 1000**
- Queries are **BEV-centric** and exist independently of priors
- Query sets may be role-specialized:
  - Lane / road structure queries
  - Object-centric queries
  - Free-space / exploratory queries
- Spatial bias toward ego-lane corridor for long-range coverage

---

### 5. Transformer-based View Transform (Critical Design)

#### Deformable Cross-Attention Roles

- **Query (Q)**: sparse BEV queries  
- **Key / Value (K/V)**: backbone or BiFPN features (**visual evidence**)  
- **Prior**: task-supervised context feature (**guidance only**)

#### Prior Usage Rule (Key Point)

> **Sampling uses backbone features.  
> Offset prediction and attention weighting use context priors.**

Concretely:

- Sampling locations are generated via deformable offsets
- Offset prediction MLP takes:
  - BEV query embedding
  - Sampled context prior feature
- Attention logits include a **prior-dependent bias**
- Backbone features remain the sole evidence source

The context prior:

- Guides *where to look*
- Guides *how strongly to attend*
- Never replaces visual evidence

---

### 6. Sparse BEV Feature Output

- Output: sparse BEV tokens at time *t*
- Encodes geometry, semantics, and instance separation implicitly
- No dense BEV grid is constructed

---

### 7. Ego-motion & Temporal BEV Accumulation

- Single ego-centric motion module
- Inputs:
  - IMU / GPS / odometry
  - Optional visual residual
- Outputs residual ego-motion (Δx, Δy, Δyaw or SE(3))
- Warp BEV(t−1) into BEV(t)
- Temporal fusion via GRU or temporal attention

---

### 8. BEV-level Perception (Abstracted)

From recurrent BEV features:

- Lane graph estimation
- Object detection
- Occupancy / flow estimation

All outputs include **uncertainty estimates**.

---

### 9. Planner

- Lightweight planner head
- Consumes perception tokens
- Outputs final trajectory

---

## Training Strategy

1. **Perspective-view pretraining**
   - Learn context features via depth / semantic / instance auxiliary losses

2. **View Transform training**
   - Sparse BEV queries
   - Deformable attention with context-guided offsets

3. **End-to-end fine-tuning**
   - Planner loss implicitly shapes perception representations

---

## Design Constraints

- Sparse BEV only (no dense grids)
- Strict separation of:
  - Evidence (backbone features)
  - Guidance (context priors)
- No direct dependency on auxiliary head outputs
- Robust to imperfect supervision and domain shift

---

## One-line Summary

> A camera-only autonomous driving system where **task-supervised context features guide deformable View Transform**, enabling **efficient sparse BEV construction and stable long-range planning** with a lightweight planner.

# TODO

## Phase 1: Infrastructure & 2D Perception (완료)

- [x] Hydra-like configuration strategy
- [x] mmdet-like training (Registry, Builder, BaseModule, Runner)
- [x] Backbone (ResNet, RegNet) + Neck (BiFPN)
- [x] Context Aggregator (Depth/Semantic/Instance context)
- [x] DepthHead, PanopticHead (SemanticHead + InstanceHead)
- [x] CityscapesDataset (panoptic + depth)
- [x] 2D perception pretraining (panoptic, depth, joint)

## Phase 2: Multi-camera Dataset (완료)

- [x] nuScenes dataset loader (6 cameras)
- [x] Camera calibration (intrinsic, extrinsic)
- [x] Ego-motion data (IMU/GPS/odometry)
- [x] 3D annotations (bbox, velocity)

## Phase 3: Sparse BEV & View Transform

- [ ] Learnable BEV queries (~1000, role-specialized)
- [ ] Multi-scale deformable cross-attention
- [ ] Context-weighted attention (prior-dependent weighting)
- [ ] Multi-camera feature sampling

## Phase 4: Temporal Accumulation

- [ ] Ego-motion estimation
- [ ] BEV warping (SE(3))
- [ ] Temporal fusion (GRU / temporal attention)

## Phase 5: BEV Perception

- [ ] 3D object detection head
- [ ] Lane graph estimation head
- [ ] Occupancy/flow head
- [ ] Uncertainty estimation

## Phase 6: Planning

- [ ] Lightweight planner head
- [ ] Trajectory output
- [ ] End-to-end training
