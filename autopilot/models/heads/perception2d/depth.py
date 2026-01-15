"""Depth estimation head with log-bin classification."""

from typing import Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.models.base import BaseModule
from autopilot.utils.registry import HEADS


@HEADS.register_module()
class DepthHead(BaseModule):
    """Depth estimation head with log-bin classification.

    Simple structure: upsample conv blocks + global context + classifier.
    Uses log-spaced bins for depth classification with soft targets.

    Args:
        num_bins: Number of depth bins.
        min_depth: Minimum depth in meters.
        max_depth: Maximum depth in meters.
        scale_factor: Upsampling factor (e.g., 8 for 1/8 scale input).
        in_channels: Input channels from context aggregator.
        inner_channels: Intermediate channels for conv blocks.
        loss_weight: Weight for depth loss.
    """

    def __init__(
        self,
        num_bins: int = 64,
        min_depth: float = 1.0,
        max_depth: float = 150.0,
        scale_factor: int = 8,
        in_channels: int = 256,
        inner_channels: int = 256,
        loss_weight: float = 1.0,
    ):
        super().__init__()

        self.num_bins = num_bins
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.loss_weight = loss_weight

        # Compute and register bin centers (not learnable)
        bin_edges = self._compute_log_bins(min_depth, max_depth, num_bins)
        log_edges = np.log(bin_edges)
        log_centers = (log_edges[:-1] + log_edges[1:]) / 2
        bin_centers = np.exp(log_centers)
        self.register_buffer('bin_centers', torch.from_numpy(bin_centers).float())

        # Build upsample conv blocks
        # Progressive upsampling: 2x at each step
        num_upsample = 0
        sf = scale_factor
        while sf > 1:
            sf //= 2
            num_upsample += 1

        self.upsample_blocks = nn.ModuleList()
        for _ in range(num_upsample):
            self.upsample_blocks.append(nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(inner_channels),
                nn.ReLU(inplace=True),
            ))
            in_channels = inner_channels

        # Global context branch (global average pooling + 1x1 conv)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_conv = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )

        # Classifier (1x1 conv)
        self.classifier = nn.Conv2d(inner_channels * 2, num_bins, 1)

    def _compute_log_bins(
        self,
        min_depth: float,
        max_depth: float,
        num_bins: int,
    ) -> np.ndarray:
        """Compute log-spaced bin edges."""
        return np.exp(np.linspace(np.log(min_depth), np.log(max_depth), num_bins + 1))

    def forward(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            context: Context feature from ContextAggregator (B, in_channels, H, W).

        Returns:
            Dictionary containing:
                - depth_logits: Depth bin logits (B, num_bins, H_out, W_out)
        """
        x = context
        for block in self.upsample_blocks:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = block(x)

        # Global context
        B, C, H, W = x.shape
        global_feat = self.global_pool(x)
        global_feat = self.global_conv(global_feat)
        global_feat = global_feat.expand(-1, -1, H, W)

        # Concatenate local and global features
        x = torch.cat([x, global_feat], dim=1)

        # Classifier
        depth_logits = self.classifier(x)

        return {
            'depth_logits': depth_logits,
        }

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.

        Uses cross-entropy with soft bin targets (equivalent to KL divergence).

        Args:
            predictions: Dictionary from forward().
            targets: Dictionary containing:
                - depth_target: Soft bin targets (B, num_bins, H, W)
                - depth_weight: Valid mask (B, 1, H, W)

        Returns:
            Dictionary of losses.
        """
        depth_logits = predictions['depth_logits']
        depth_target = targets['depth_target']
        depth_weight = targets['depth_weight']

        # Resize predictions to target size if needed
        target_size = depth_target.shape[-2:]
        if depth_logits.shape[-2:] != target_size:
            depth_logits = F.interpolate(
                depth_logits, size=target_size, mode='bilinear', align_corners=False
            )

        # Log softmax for predicted probabilities
        depth_log_probs = F.log_softmax(depth_logits, dim=1)

        # Cross-entropy with soft targets: -sum(target * log(pred))
        loss_per_pixel = -(depth_target * depth_log_probs).sum(dim=1, keepdim=True)

        # Weighted mean over valid pixels
        num_valid = depth_weight.sum().clamp(min=1)
        loss_depth = (loss_per_pixel * depth_weight).sum() / num_valid

        total_loss = self.loss_weight * loss_depth

        return {
            'loss': total_loss,
            'loss_depth': loss_depth,
        }

    def predict(self, context: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Generate predictions for inference.

        Args:
            context: Context feature from ContextAggregator.

        Returns:
            Dictionary containing:
                - depth: Predicted depth in meters (B, 1, H, W)
                - depth_probs: Bin probabilities (B, num_bins, H, W)
        """
        outputs = self.forward(context)
        depth_logits = outputs['depth_logits']

        # Softmax to get probabilities
        depth_probs = F.softmax(depth_logits, dim=1)

        # Weighted sum of bin centers to get depth
        bin_centers = self.bin_centers.view(1, -1, 1, 1)
        depth = (depth_probs * bin_centers).sum(dim=1, keepdim=True)

        return {
            'depth': depth,
            'depth_probs': depth_probs,
        }
