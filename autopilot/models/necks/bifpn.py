"""BiFPN (Bidirectional Feature Pyramid Network) neck."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.models.base import BaseModule
from autopilot.utils.registry import NECKS


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size,
            padding=kernel_size // 2, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.act(x)
        return x


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with top-down and bottom-up paths."""

    def __init__(self, channels: int, num_levels: int = 4):
        super().__init__()
        self.num_levels = num_levels

        # Learnable weights for feature fusion
        self.w1 = nn.Parameter(torch.ones(2, num_levels - 1))  # top-down
        self.w2 = nn.Parameter(torch.ones(3, num_levels - 1))  # bottom-up

        # Top-down convolutions (P6->P5->P4->P3)
        self.td_convs = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels)
            for _ in range(num_levels - 1)
        ])

        # Bottom-up convolutions (P3->P4->P5->P6)
        self.bu_convs = nn.ModuleList([
            DepthwiseSeparableConv(channels, channels)
            for _ in range(num_levels - 1)
        ])

        self.eps = 1e-4

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            features: List of feature maps [P3, P4, P5, P6] (low to high level).

        Returns:
            List of fused feature maps.
        """
        # Normalize weights
        w1 = F.relu(self.w1)
        w1 = w1 / (w1.sum(dim=0, keepdim=True) + self.eps)
        w2 = F.relu(self.w2)
        w2 = w2 / (w2.sum(dim=0, keepdim=True) + self.eps)

        # Top-down path: P6 -> P5 -> P4 -> P3
        td_features = [None] * self.num_levels
        td_features[-1] = features[-1]  # P6 stays

        for i in range(self.num_levels - 2, -1, -1):
            # Upsample higher level and fuse
            up = F.interpolate(td_features[i + 1], size=features[i].shape[-2:], mode='nearest')
            td_features[i] = self.td_convs[i](
                w1[0, i] * features[i] + w1[1, i] * up
            )

        # Bottom-up path: P3 -> P4 -> P5 -> P6
        out_features = [None] * self.num_levels
        out_features[0] = td_features[0]  # P3 stays

        for i in range(1, self.num_levels):
            # Downsample lower level and fuse
            down = F.interpolate(out_features[i - 1], size=td_features[i].shape[-2:], mode='nearest')
            out_features[i] = self.bu_convs[i - 1](
                w2[0, i - 1] * features[i] + w2[1, i - 1] * td_features[i] + w2[2, i - 1] * down
            )

        return out_features


@NECKS.register_module()
class BiFPN(BaseModule):
    """BiFPN neck for multi-scale feature fusion.

    Takes multi-scale features from backbone and outputs unified channel features.
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_layers: int = 3,
    ):
        """Initialize BiFPN.

        Args:
            in_channels: List of input channel sizes from backbone.
            out_channels: Output channel size (unified).
            num_layers: Number of BiFPN layers to stack.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(in_channels)

        # Lateral convolutions to unify channels
        self.lateral_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SiLU(inplace=True),
            )
            for in_ch in in_channels
        ])

        # Stack of BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, self.num_levels)
            for _ in range(num_layers)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            features: List of feature maps from backbone.

        Returns:
            List of fused feature maps with unified channels.
        """
        # Project to same channel size
        features = [
            lateral(feat) for lateral, feat in zip(self.lateral_convs, features)
        ]

        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            features = bifpn_layer(features)

        return features
