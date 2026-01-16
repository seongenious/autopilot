"""Joint Panoptic + Depth model for multi-task learning."""

from typing import Any, Dict, Tuple

import torch

from autopilot.models.base import BaseModule
from autopilot.models.builder import build_backbone, build_head, build_neck
from autopilot.utils.registry import MODELS


@MODELS.register_module()
class Joint2d(BaseModule):
    """Joint Panoptic segmentation and Depth estimation model.

    Shares backbone and neck, with separate context aggregators for each task.
    Combines losses from both tasks with configurable weights.

    Args:
        backbone: Backbone config.
        neck: Neck config (BiFPN).
        sem_context: Semantic context aggregator config.
        inst_context: Instance context aggregator config.
        depth_context: Depth context aggregator config.
        panoptic_head: Panoptic head config.
        depth_head: Depth head config.
        loss_weights: Task loss weights dict with 'panoptic' and 'depth' keys.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        sem_context: dict,
        inst_context: dict,
        depth_context: dict,
        panoptic_head: dict,
        depth_head: dict,
        loss_weights: dict = None,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)

        # Shared feature extraction
        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)

        # Task-specific context aggregators
        self.sem_context = build_neck(sem_context)
        self.inst_context = build_neck(inst_context)
        self.depth_context = build_neck(depth_context)

        # Task heads
        self.panoptic_head = build_head(panoptic_head)
        self.depth_head = build_head(depth_head)

        # Loss weights
        self.loss_weights = loss_weights or {"panoptic": 1.0, "depth": 1.0}

    def extract_feat(
        self, images: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features from images.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Tuple of (sem_context, inst_context, depth_context).
        """
        features = self.backbone(images)
        features = self.neck(features)

        sem_context = self.sem_context(features)
        inst_context = self.inst_context(features)
        depth_context = self.depth_context(features)

        return sem_context, inst_context, depth_context

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass (inference).

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Prediction dict with panoptic and depth outputs.
        """
        sem_context, inst_context, depth_context = self.extract_feat(images)

        # Panoptic predictions
        panoptic_out = self.panoptic_head(sem_context, inst_context)

        # Depth predictions
        depth_out = self.depth_head(depth_context)

        # Merge outputs
        return {**panoptic_out, **depth_out}

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            batch: Input batch with image, panoptic targets, and depth targets.

        Returns:
            Dict with loss values including 'loss' key for total loss.
        """
        images = batch["image"]
        sem_context, inst_context, depth_context = self.extract_feat(images)

        # Panoptic forward and loss
        panoptic_pred = self.panoptic_head(sem_context, inst_context)
        panoptic_targets = {
            "semantic_target": batch["semantic_target"],
            "center_target": batch["center_target"],
            "offset_target": batch["offset_target"],
            "offset_weight": batch["offset_weight"],
        }
        panoptic_losses = self.panoptic_head.loss(panoptic_pred, panoptic_targets)

        # Depth forward and loss
        depth_pred = self.depth_head(depth_context)
        depth_targets = {
            "depth_target": batch["depth_target"],
            "depth_weight": batch["depth_weight"],
        }
        depth_losses = self.depth_head.loss(depth_pred, depth_targets)

        # Combine losses
        panoptic_weight = self.loss_weights["panoptic"]
        depth_weight = self.loss_weights["depth"]

        total_loss = (
            panoptic_weight * panoptic_losses["loss"]
            + depth_weight * depth_losses["loss"]
        )

        # Build output dict
        losses = {"loss": total_loss}

        # Add panoptic losses with prefix
        for key, value in panoptic_losses.items():
            if key != "loss":
                losses[f"panoptic_{key}"] = value
            else:
                losses["panoptic_loss"] = value

        # Add depth losses with prefix
        for key, value in depth_losses.items():
            if key != "loss":
                losses[f"depth_{key}"] = value
            else:
                losses["depth_loss"] = value

        return losses

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Validation step.

        Args:
            batch: Input batch.

        Returns:
            Dict with loss values.
        """
        return self.train_step(batch)

    def predict(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Predict panoptic segmentation and depth.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Prediction dict with panoptic and depth outputs.
        """
        sem_context, inst_context, depth_context = self.extract_feat(images)

        panoptic_out = self.panoptic_head.predict(sem_context, inst_context)
        depth_out = self.depth_head.predict(depth_context)

        return {**panoptic_out, **depth_out}
