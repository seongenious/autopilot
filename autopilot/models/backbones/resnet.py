"""ResNet backbone using timm pretrained models."""

from typing import List, Tuple

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from autopilot.models.base import BaseModule
from autopilot.utils.registry import BACKBONES


@BACKBONES.register_module()
class ResNet(BaseModule):
    """ResNet backbone with timm pretrained weights.

    Output channels:
        - resnet18/34: [64, 128, 256, 512]
        - resnet50/101/152: [256, 512, 1024, 2048]
    """

    DEPTH_TO_MODEL = {
        18: 'resnet18',
        34: 'resnet34',
        50: 'resnet50',
        101: 'resnet101',
        152: 'resnet152',
    }

    OUT_CHANNELS = {
        18: [64, 128, 256, 512],
        34: [64, 128, 256, 512],
        50: [256, 512, 1024, 2048],
        101: [256, 512, 1024, 2048],
        152: [256, 512, 1024, 2048],
    }

    def __init__(
        self,
        depth: int = 50,
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1,
        with_cp: bool = False,
    ):
        """Initialize ResNet backbone.

        Args:
            depth: ResNet depth (18, 34, 50, 101, 152).
            pretrained: Whether to use pretrained weights.
            out_indices: Output feature indices (1=C2, 2=C3, 3=C4, 4=C5). 0=stem.
            frozen_stages: Stages to freeze (-1=none, 0=stem, 1=stem+stage1, ...).
            with_cp: Use gradient checkpointing to save memory (slower but less memory).
        """
        super().__init__()

        if depth not in self.DEPTH_TO_MODEL:
            raise ValueError(f'Invalid depth {depth}. Choose from {list(self.DEPTH_TO_MODEL.keys())}')

        self.depth = depth
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.with_cp = with_cp

        # Create model with timm
        model_name = self.DEPTH_TO_MODEL[depth]
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # Get output channels from timm (more reliable)
        self.out_channels = self.model.feature_info.channels()

        # Freeze stages
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze stages based on frozen_stages setting."""
        if self.frozen_stages < 0:
            return

        # Freeze stem (conv1, bn1)
        if self.frozen_stages >= 0:
            self.model.conv1.eval()
            self.model.bn1.eval()
            for param in self.model.conv1.parameters():
                param.requires_grad = False
            for param in self.model.bn1.parameters():
                param.requires_grad = False

        # Freeze stages
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self.model, f'layer{i}', None)
            if layer is not None:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            List of feature maps at different scales.
        """
        if self.with_cp and self.training:
            return checkpoint(self.model, x, use_reentrant=False)
        return self.model(x)

    def train(self, mode: bool = True) -> 'ResNet':
        """Set training mode, keeping frozen stages in eval."""
        super().train(mode)
        self._freeze_stages()
        return self
