"""Context module for aggregating multi-scale FPN features."""

from typing import List, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.models.base import BaseModule
from autopilot.utils.registry import NECKS


@NECKS.register_module()
class ContextAggregator(BaseModule):
    """Aggregates multi-scale features into a single-scale context.

    Takes FPN features [P2, P3, P4, P5] and produces a single feature map
    by downsampling or upsampling all levels to target resolution and 
    concatenating.

    Args:
        in_channels: Number of input channels (same for all FPN levels).
        out_channels: Number of output channels after projection.
        num_levels: Number of FPN levels.
        target_level: Target level index for output resolution (0=P2, 1=P3, ...).
        fusion: Fusion method ('concat' or 'sum').
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        num_levels: int = 4,
        target_level: int = 1,  # 0: 1/4, 1: 1/8, 2: 1/16, 3: 1/32
        fusion: Literal["concat", "sum"] = "concat",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = num_levels
        self.target_level = target_level
        self.fusion = fusion

        # Project each level before fusion
        self.proj = nn.ModuleList()
        for i in range(num_levels):
            self.proj.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ))

        # Final projection after fusion
        if fusion == "concat":
            fused_channels = out_channels * num_levels
        else:
            fused_channels = out_channels

        self.output_proj = nn.Sequential(
            nn.Conv2d(fused_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Multi-scale FPN features [P2, P3, P4, P5].
                      All have same channel dimension.

        Returns:
            Aggregated context feature at target resolution.
            Shape: (B, out_channels, H_target, W_target)
        """
        assert len(features) == self.num_levels

        # Get target size from target level
        target_size = features[self.target_level].shape[-2:]

        # Project and resize each level to target size
        projected = []
        for i, (feat, proj) in enumerate(zip(features, self.proj)):
            x = proj(feat)
            if i != self.target_level:
                x = F.interpolate(
                    x, size=target_size,
                    mode='bilinear', align_corners=False
                )
            projected.append(x)

        # Fuse
        if self.fusion == "concat":
            fused = torch.cat(projected, dim=1)
        else:
            fused = sum(projected)

        # Output projection
        out = self.output_proj(fused)

        return out