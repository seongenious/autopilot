"""Panoptic Segmentor model."""

from typing import Any, Dict, Tuple

import torch

from autopilot.models.base import BaseModule
from autopilot.models.builder import build_backbone, build_head, build_neck
from autopilot.utils.registry import MODELS


@MODELS.register_module()
class Panoptic(BaseModule):
    """Panoptic segmentation model.

    Combines backbone, neck, context aggregator, and panoptic head for panoptic segmentation.
    Provides train_step and val_step for Runner compatibility.
    """

    def __init__(
        self,
        backbone: dict,
        neck: dict,
        sem_context: dict,
        inst_context: dict,
        head: dict,
        init_cfg: dict = None,
    ):
        super().__init__(init_cfg=init_cfg)

        self.backbone = build_backbone(backbone)
        self.neck = build_neck(neck)
        self.sem_context = build_neck(sem_context)
        self.inst_context = build_neck(inst_context)
        self.head = build_head(head)

    def extract_feat(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Extract features from images.

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Tuple of (sem_context, inst_context).
        """
        features = self.backbone(images)
        features = self.neck(features)
        sem_context = self.sem_context(features)
        inst_context = self.inst_context(features)

        return sem_context, inst_context

    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass (inference).

        Args:
            images: Input images (B, 3, H, W).

        Returns:
            Prediction dict with sem_logits, center_heatmap, offset.
        """
        sem_context, inst_context = self.extract_feat(images)
        return self.head(sem_context, inst_context)

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Training step.

        Args:
            batch: Input batch with image, semantic_target, center_target,
                   offset_target, offset_weight.

        Returns:
            Dict with loss values including 'loss' key for total loss.
        """
        images = batch["image"]
        sem_context, inst_context = self.extract_feat(images)
        predictions = self.head(sem_context, inst_context)

        targets = {
            "semantic_target": batch["semantic_target"],
            "center_target": batch["center_target"],
            "offset_target": batch["offset_target"],
            "offset_weight": batch["offset_weight"],
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

    def predict(
        self,
        images: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Predict panoptic segmentation.

        Args:
            images: Input images (B, 3, H, W).
            img_size: Original image size for upsampling.
            **kwargs: Additional arguments for head.predict().

        Returns:
            Prediction dict with semantic and instance masks.
        """
        sem_context, inst_context = self.extract_feat(images)
        return self.head.predict(sem_context, inst_context)
