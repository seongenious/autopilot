"""Panoptic segmentation head with decoder and prediction heads."""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.models.base import BaseModule
from autopilot.utils.registry import HEADS


class ContextDecoder(nn.Module):
    """Decoder that upsamples context features to target resolution.

    Takes single-scale context from ContextAggregator and upsamples
    to original image size through progressive upsampling.

    Args:
        in_channels: Number of input channels from context.
        out_channels: Number of output channels.
        scale_factor: Total upsampling factor (e.g., 8 for 1/8 scale input).
    """

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 256,
        scale_factor: int = 8,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor

        # Progressive upsampling: 2x at each step
        # e.g., scale_factor=8 -> 3 upsample steps (2^3 = 8)
        num_upsample = 0
        sf = scale_factor
        while sf > 1:
            sf //= 2
            num_upsample += 1

        self.upsample_layers = nn.ModuleList()
        current_channels = in_channels

        for i in range(num_upsample):
            # Last layer outputs out_channels, others keep current_channels
            next_channels = out_channels if i == num_upsample - 1 else current_channels
            self.upsample_layers.append(nn.Sequential(
                nn.Conv2d(current_channels, next_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(next_channels),
                nn.ReLU(inplace=True),
            ))
            current_channels = next_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Context feature (B, in_channels, H, W).

        Returns:
            Upsampled feature (B, out_channels, H * scale_factor, W * scale_factor).
        """
        for layer in self.upsample_layers:
            x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            x = layer(x)
        return x


class SemanticHead(nn.Module):
    """Semantic segmentation head.

    Args:
        in_channels: Number of input channels.
        inner_channels: Number of intermediate channels.
        num_classes: Number of semantic classes (things + stuff).
    """

    def __init__(
        self,
        in_channels: int = 256,
        inner_channels: int = 256,
        num_classes: int = 19,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Conv2d(inner_channels, num_classes, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input feature (B, in_channels, H, W).

        Returns:
            Semantic logits (B, num_classes, H, W).
        """
        x = self.conv(x)
        x = self.classifier(x)
        return x


class InstanceHead(nn.Module):
    """Instance segmentation head for center heatmap and offset prediction.

    Args:
        in_channels: Number of input channels.
        inner_channels: Number of intermediate channels.
    """

    def __init__(
        self,
        in_channels: int = 128,
        inner_channels: int = 32,
    ):
        super().__init__()

        # Center heatmap head (class-agnostic, single channel)
        self.center_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.center_classifier = nn.Conv2d(inner_channels, 1, 1)

        # Offset head
        self.offset_conv = nn.Sequential(
            nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(inplace=True),
        )
        self.offset_regressor = nn.Conv2d(inner_channels, 2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input feature (B, in_channels, H, W).

        Returns:
            center: Center heatmap (B, 1, H, W).
            offset: Offset prediction (B, 2, H, W).
        """
        center = self.center_conv(x)
        center = self.center_classifier(center)

        offset = self.offset_conv(x)
        offset = self.offset_regressor(offset)

        return center, offset


@HEADS.register_module()
class PanopticHead(BaseModule):
    """Panoptic segmentation head.

    Combines decoder, semantic head, and instance head for panoptic segmentation.
    Receives two separate contexts: semantic context and instance context.

    Args:
        num_things_classes: Number of thing classes.
        num_stuff_classes: Number of stuff classes.
        scale_factor: Upsampling factor for decoder.
        sem_in_channels: Input channels for semantic branch (from semantic context).
        sem_decoder_channels: Output channels of semantic decoder.
        sem_head_channels: Inner channels for semantic head.
        inst_in_channels: Input channels for instance branch (from instance context).
        inst_decoder_channels: Output channels of instance decoder.
        inst_head_channels: Inner channels for instance head.
        loss_semantic_weight: Weight for semantic segmentation loss.
        loss_center_weight: Weight for center heatmap loss.
        loss_offset_weight: Weight for offset regression loss.
        ignore_index: Ignore index for semantic loss.
    """

    def __init__(
        self,
        num_things_classes: int = 8,
        num_stuff_classes: int = 11,
        scale_factor: int = 8,
        # Semantic branch
        sem_in_channels: int = 256,
        sem_decoder_channels: int = 256,
        sem_head_channels: int = 256,
        # Instance branch
        inst_in_channels: int = 128,
        inst_decoder_channels: int = 128,
        inst_head_channels: int = 32,
        # Loss weights
        loss_semantic_weight: float = 1.0,
        loss_center_weight: float = 200.0,
        loss_offset_weight: float = 0.01,
        ignore_index: int = 255,
    ):
        super().__init__()

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.loss_semantic_weight = loss_semantic_weight
        self.loss_center_weight = loss_center_weight
        self.loss_offset_weight = loss_offset_weight
        self.ignore_index = ignore_index

        # Semantic Decoder
        self.semantic_decoder = ContextDecoder(
            in_channels=sem_in_channels,
            out_channels=sem_decoder_channels,
            scale_factor=scale_factor,
        )

        # Instance Decoder
        self.instance_decoder = ContextDecoder(
            in_channels=inst_in_channels,
            out_channels=inst_decoder_channels,
            scale_factor=scale_factor,
        )

        # Semantic head
        self.semantic_head = SemanticHead(
            in_channels=sem_decoder_channels,
            inner_channels=sem_head_channels,
            num_classes=self.num_classes,
        )

        # Instance head
        self.instance_head = InstanceHead(
            in_channels=inst_decoder_channels,
            inner_channels=inst_head_channels,
        )

    def forward(
        self,
        sem_context: torch.Tensor,
        inst_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            sem_context: Semantic context from ContextAggregator (B, sem_in_channels, H, W).
            inst_context: Instance context from ContextAggregator (B, inst_in_channels, H, W).

        Returns:
            Dictionary containing:
                - logits: Semantic logits (B, num_classes, H_out, W_out)
                - center: Center heatmap (B, 1, H_out, W_out)
                - offset: Offset prediction (B, 2, H_out, W_out)
        """
        # Semantic pipeline
        sem_decoded = self.semantic_decoder(sem_context)
        semantic = self.semantic_head(sem_decoded)

        # Instance pipeline
        inst_decoded = self.instance_decoder(inst_context)
        center, offset = self.instance_head(inst_decoded)

        return {
            "semantic": semantic,
            "center": center,
            "offset": offset,
        }

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.

        Args:
            predictions: Dictionary from forward().
            targets: Dictionary containing:
                - sem_target: Semantic labels (B, H, W)
                - center_target: Center heatmap GT (B, 1, H, W)
                - offset_target: Offset GT (B, 2, H, W)
                - offset_weight: Offset weight mask (B, 1, H, W)

        Returns:
            Dictionary of losses.
        """
        semantic = predictions["semantic"]
        center = predictions["center"]
        offset = predictions["offset"]

        sem_target = targets["semantic_target"]
        center_target = targets["center_target"]
        offset_target = targets["offset_target"]
        offset_weight = targets["offset_weight"]

        # Resize predictions to target size if needed
        target_size = sem_target.shape[-2:]
        if semantic.shape[-2:] != target_size:
            semantic = F.interpolate(
                semantic, size=target_size, mode='bilinear', align_corners=False
            )
            center = F.interpolate(
                center, size=target_size, mode='bilinear', align_corners=False
            )
            offset = F.interpolate(
                offset, size=target_size, mode='bilinear', align_corners=False
            )

        # Semantic loss (cross entropy)
        loss_semantic = F.cross_entropy(
            semantic, sem_target, ignore_index=self.ignore_index
        )

        # Center loss (MSE with sigmoid)
        loss_center = F.mse_loss(
            torch.sigmoid(center), center_target
        )

        # Offset loss (L1, weighted by instance mask)
        loss_offset = F.l1_loss(
            offset * offset_weight, offset_target * offset_weight, reduction='sum'
        )
        num_pos = offset_weight.sum().clamp(min=1)
        loss_offset = loss_offset / num_pos

        # Weighted sum
        total_loss = (
            self.loss_semantic_weight * loss_semantic
            + self.loss_center_weight * loss_center
            + self.loss_offset_weight * loss_offset
        )

        return {
            "loss": total_loss,
            "loss_semantic": loss_semantic,
            "loss_center": loss_center,
            "loss_offset": loss_offset,
        }

    def predict(
        self,
        sem_context: torch.Tensor,
        inst_context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Generate predictions for inference.

        Args:
            sem_context: Semantic context from ContextAggregator.
            inst_context: Instance context from ContextAggregator.

        Returns:
            Dictionary containing:
                - semantic: Semantic predictions (B, H, W)
                - center: Center heatmap (B, 1, H, W)
                - offset: Offset prediction (B, 2, H, W)
        """
        outputs = self.forward(sem_context, inst_context)

        semantic = outputs["semantic"]
        center = outputs["center"]
        offset = outputs["offset"]

        return {
            "semantic": semantic.argmax(dim=1),
            "center": torch.sigmoid(center),
            "offset": offset,
        }
