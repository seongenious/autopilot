"""BiFPN (Bidirectional Feature Pyramid Network) implementation.

BiFPN introduces learnable weights for multi-scale feature fusion
and bidirectional cross-scale connections.

Reference:
    - Paper: https://arxiv.org/abs/1911.09070 (EfficientDet)
"""

from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.utils.registry import NECKS


class ConvBnAct(nn.Module):
    """Convolution + BatchNorm + Activation block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activation: Optional[nn.Module] = None,
    ) -> None:
        """Initialize ConvBnAct block."""
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = activation or nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        padding: int = 1,
    ) -> None:
        """Initialize depthwise separable convolution."""
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.act(x)


class BiFPNLayer(nn.Module):
    """Single BiFPN layer with top-down and bottom-up pathways.

    Implements weighted feature fusion with learnable weights.
    """

    def __init__(
        self,
        num_channels: int,
        num_levels: int = 5,
        epsilon: float = 1e-4,
    ) -> None:
        """Initialize BiFPN layer."""
        super().__init__()
        self.num_levels = num_levels
        self.epsilon = epsilon

        # Learnable weights for feature fusion
        # Top-down pathway weights (P6 -> P3)
        self.w1 = nn.Parameter(torch.ones(2, num_levels - 1))

        # Bottom-up pathway weights (P3 -> P7)
        self.w2 = nn.Parameter(torch.ones(3, num_levels - 1))

        # Convolutions for feature fusion
        self.conv_td = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels)
            for _ in range(num_levels - 1)
        ])
        self.conv_bu = nn.ModuleList([
            DepthwiseSeparableConv(num_channels, num_channels)
            for _ in range(num_levels - 1)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            features: List of feature maps from P3 to P7 (or similar).

        Returns:
            List of fused feature maps.
        """
        assert len(features) == self.num_levels

        # Normalize weights using fast normalized fusion
        w1 = F.relu(self.w1)
        w1 = w1 / (w1.sum(dim=0, keepdim=True) + self.epsilon)

        w2 = F.relu(self.w2)
        w2 = w2 / (w2.sum(dim=0, keepdim=True) + self.epsilon)

        # Top-down pathway
        td_features = [None] * self.num_levels
        td_features[-1] = features[-1]

        for i in range(self.num_levels - 2, -1, -1):
            # Upsample higher level feature
            upsampled = F.interpolate(
                td_features[i + 1],
                size=features[i].shape[-2:],
                mode='nearest',
            )
            # Weighted fusion
            td_features[i] = self.conv_td[i](
                w1[0, i] * features[i] + w1[1, i] * upsampled
            )

        # Bottom-up pathway
        out_features = [None] * self.num_levels
        out_features[0] = td_features[0]

        for i in range(1, self.num_levels):
            # Downsample lower level feature
            downsampled = F.interpolate(
                out_features[i - 1],
                size=td_features[i].shape[-2:],
                mode='nearest',
            )
            # Weighted fusion with original, top-down, and bottom-up
            out_features[i] = self.conv_bu[i - 1](
                w2[0, i - 1] * features[i]
                + w2[1, i - 1] * td_features[i]
                + w2[2, i - 1] * downsampled
            )

        return out_features


@NECKS.register_module()
class BiFPN(nn.Module):
    """Bidirectional Feature Pyramid Network.

    Args:
        in_channels: List of input channel dimensions from backbone.
        out_channels: Output channel dimension for all levels.
        num_levels: Number of feature pyramid levels.
        num_layers: Number of BiFPN layers to stack.
        epsilon: Small constant for numerical stability.

    Example:
        >>> bifpn = BiFPN(
        ...     in_channels=[64, 128, 256, 512],
        ...     out_channels=256,
        ...     num_layers=3,
        ... )
        >>> features = [torch.randn(1, c, 80//(2**i), 120//(2**i))
        ...             for i, c in enumerate([64, 128, 256, 512])]
        >>> out = bifpn(features)
    """

    def __init__(
        self,
        in_channels: List[int],
        out_channels: int = 256,
        num_levels: Optional[int] = None,
        num_layers: int = 3,
        epsilon: float = 1e-4,
    ) -> None:
        """Initialize BiFPN."""
        super().__init__()

        self.num_levels = num_levels or len(in_channels)
        self._out_channels = out_channels
        self._out_channels_list = [out_channels] * self.num_levels

        # Lateral connections to match channel dimensions
        self.lateral_convs = nn.ModuleList([
            ConvBnAct(in_ch, out_channels, kernel_size=1)
            for in_ch in in_channels
        ])

        # Additional levels if needed (e.g., P6, P7 from P5)
        extra_levels = self.num_levels - len(in_channels)
        if extra_levels > 0:
            self.extra_convs = nn.ModuleList([
                ConvBnAct(
                    in_channels[-1] if i == 0 else out_channels,
                    out_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                )
                for i in range(extra_levels)
            ])
        else:
            self.extra_convs = None

        # Stack of BiFPN layers
        self.bifpn_layers = nn.ModuleList([
            BiFPNLayer(out_channels, self.num_levels, epsilon)
            for _ in range(num_layers)
        ])

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            features: List of feature maps from backbone.

        Returns:
            List of fused multi-scale feature maps.
        """
        # Apply lateral convolutions
        fpn_features = [
            lateral(feat)
            for lateral, feat in zip(self.lateral_convs, features)
        ]

        # Add extra levels if configured
        if self.extra_convs is not None:
            last_feat = features[-1]
            for extra_conv in self.extra_convs:
                last_feat = extra_conv(last_feat)
                fpn_features.append(last_feat)

        # Apply BiFPN layers
        for bifpn_layer in self.bifpn_layers:
            fpn_features = bifpn_layer(fpn_features)

        return fpn_features

    @property
    def out_channels(self) -> int:
        """Return output channel dimension (same for all levels)."""
        return self._out_channels

    @property
    def out_channels_list(self) -> List[int]:
        """Return output channel dimensions for each level."""
        return self._out_channels_list


# TODO: Add support for attention-based feature fusion
# TODO: Implement efficient memory checkpointing for training
# TODO: Add support for different normalization layers (GroupNorm, etc.)
