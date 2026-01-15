"""Depth Estimation model."""

from typing import Any, Dict

import torch

from autopilot.models.base import BaseModule
from autopilot.models.builder import build_backbone, build_head, build_neck
from autopilot.utils.registry import MODELS


@MODELS.register_module()
class Depth(BaseModule):
    """Depth estimation model.

    Combines backbone, neck, context aggregator, and depth head.
    Provides train_step and val_step for Runner compatibility.

    Args:
        backbone: Backbone config.
        neck: Neck config (BiFPN).
        context: Context aggregator config.
        head: Depth head config.
        init_cfg: Initialization config.
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        context: dict,
        head: dict,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.context = build_neck(context)
        self.head = build_head(head)

    def extract_feat(self, images: torch.Tensor) -> torch.Tensor:
        """Extract features from images.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Context feature (B, C, H', W').
        """
        features = self.backbone(images)
        features = self.neck(features)
        context = self.context(features)

        return context

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass (inference).

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Prediction dict with depth_logits.
        """
        context = self.extract_feat(images)
        return self.head(context)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            batch: Input batch with image, depth_target, depth_weight.

        Returns:
            Dict with loss values including 'loss' key for total loss.
        """
        images = batch['image']
        context = self.extract_feat(images)
        predictions = self.head(context)

        targets = {
            'depth_target': batch['depth_target'],
            'depth_weight': batch['depth_weight'],
        }

        losses = self.head.loss(predictions, targets)

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
        """Predict depth.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Prediction dict with depth and depth_probs.
        """
        context = self.extract_feat(images)
        return self.head.predict(context)
