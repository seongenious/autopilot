"""Panoptic FPN head based on 'Panoptic Feature Pyramid Networks' (CVPR 2019)."""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.models.base import BaseModule
from autopilot.utils.registry import HEADS


class SemanticHead(nn.Module):
    """Semantic segmentation branch of Panoptic FPN.

    Merges multi-scale FPN features and predicts per-pixel class labels.
    """

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        num_classes: int,
        num_levels: int = 4,
    ):
        super().__init__()

        self.num_levels = num_levels

        # Scale-specific convolutions
        self.scale_heads = nn.ModuleList()
        for i in range(num_levels):
            head = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, inner_channels),
                nn.ReLU(inplace=True),
            )
            self.scale_heads.append(head)

        # Final prediction
        self.predictor = nn.Conv2d(inner_channels, num_classes, 1)

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Multi-scale FPN features [P2, P3, P4, P5].

        Returns:
            Semantic logits at 1/4 scale (same as P2).
        """
        target_size = features[0].shape[-2:]

        # Process each scale and upsample to P2 resolution
        out = self.scale_heads[0](features[0])

        for i in range(1, self.num_levels):
            feat = self.scale_heads[i](features[i])
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            out = out + feat

        # Predict
        out = self.predictor(out)
        return out


class InstanceHead(nn.Module):
    """Instance segmentation branch of Panoptic FPN.

    Center-based instance head that predicts:
    - Center heatmap: object center locations
    - Center offset: sub-pixel offset refinement
    - Instance embedding: for grouping pixels to instances
    """

    def __init__(
        self,
        in_channels: int,
        inner_channels: int,
        num_things_classes: int,
        embed_dim: int = 32,
        num_levels: int = 4,
    ):
        super().__init__()

        self.num_things_classes = num_things_classes
        self.embed_dim = embed_dim
        self.num_levels = num_levels

        # Scale-specific convolutions (similar to semantic head)
        self.scale_heads = nn.ModuleList()
        for i in range(num_levels):
            head = nn.Sequential(
                nn.Conv2d(in_channels, inner_channels, 3, padding=1, bias=False),
                nn.GroupNorm(32, inner_channels),
                nn.ReLU(inplace=True),
            )
            self.scale_heads.append(head)

        # Center heatmap prediction (per thing class)
        self.center_head = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, num_things_classes, 1),
        )

        # Center offset prediction (2D offset)
        self.offset_head = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, 2, 1),
        )

        # Instance embedding prediction
        self.embed_head = nn.Sequential(
            nn.Conv2d(inner_channels, inner_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, inner_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(inner_channels, embed_dim, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: Multi-scale FPN features.

        Returns:
            Dict with 'center', 'offset', 'embedding'.
        """
        target_size = features[0].shape[-2:]

        # Merge multi-scale features
        out = self.scale_heads[0](features[0])
        for i in range(1, self.num_levels):
            feat = self.scale_heads[i](features[i])
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            out = out + feat

        # Predictions
        center = self.center_head(out)
        offset = self.offset_head(out)
        embedding = self.embed_head(out)

        return {
            'center': center,      # (B, num_things_classes, H, W)
            'offset': offset,      # (B, 2, H, W)
            'embedding': embedding,  # (B, embed_dim, H, W)
        }


@HEADS.register_module()
class PanopticHead(BaseModule):
    """Panoptic FPN head for panoptic segmentation.

    Combines semantic segmentation (stuff + things) and instance segmentation (things only).
    """

    def __init__(
        self,
        in_channels: int,
        num_things_classes: int,
        num_stuff_classes: int,
        inner_channels: int = 128,
        num_levels: int = 4,
        embed_dim: int = 32,
        loss_sem_weight: float = 1.0,
        loss_center_weight: float = 1.0,
        loss_offset_weight: float = 0.1,
        ignore_index: int = 255,
    ):
        """Initialize Panoptic FPN.

        Args:
            in_channels: Input channel size from FPN.
            num_things_classes: Number of thing classes (instances).
            num_stuff_classes: Number of stuff classes (background).
            inner_channels: Inner channel size.
            num_levels: Number of FPN levels.
            embed_dim: Instance embedding dimension.
            loss_sem_weight: Weight for semantic loss.
            loss_center_weight: Weight for center heatmap loss.
            loss_offset_weight: Weight for offset loss.
            ignore_index: Ignore index for semantic loss.
        """
        super().__init__()

        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = num_things_classes + num_stuff_classes
        self.loss_sem_weight = loss_sem_weight
        self.loss_center_weight = loss_center_weight
        self.loss_offset_weight = loss_offset_weight
        self.ignore_index = ignore_index

        # Semantic segmentation head (stuff + things)
        self.semantic_head = SemanticHead(
            in_channels=in_channels,
            inner_channels=inner_channels,
            num_classes=self.num_classes,
            num_levels=num_levels,
        )

        # Instance segmentation head (things only)
        self.instance_head = InstanceHead(
            in_channels=in_channels,
            inner_channels=inner_channels,
            num_things_classes=num_things_classes,
            embed_dim=embed_dim,
            num_levels=num_levels,
        )

        # Loss function
        self.sem_loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, features: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Forward pass (inference only).

        Args:
            features: Multi-scale FPN features.

        Returns:
            Dict with predictions.
        """
        sem_logits = self.semantic_head(features)
        inst_outputs = self.instance_head(features)

        return {
            'sem_logits': sem_logits,
            'center': inst_outputs['center'],
            'offset': inst_outputs['offset'],
            'embedding': inst_outputs['embedding'],
        }

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        sem_targets: torch.Tensor,
        center_targets: Optional[torch.Tensor] = None,
        offset_targets: Optional[torch.Tensor] = None,
        offset_weights: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute losses.

        Args:
            predictions: Output from forward().
            sem_targets: Semantic labels (B, H, W).
            center_targets: Center heatmap targets (B, num_things, H, W).
            offset_targets: Offset targets (B, 2, H, W).
            offset_weights: Offset loss weights (B, H, W).

        Returns:
            Dict with loss values.
        """
        losses = {}

        # Semantic loss
        sem_logits = predictions['sem_logits']
        target_size = sem_logits.shape[-2:]
        sem_targets_resized = F.interpolate(
            sem_targets.unsqueeze(1).float(),
            size=target_size,
            mode='nearest',
        ).squeeze(1).long()

        loss_sem = self.sem_loss_fn(sem_logits, sem_targets_resized)
        losses['loss_sem'] = loss_sem * self.loss_sem_weight

        # Center heatmap loss
        if center_targets is not None:
            center_pred = predictions['center']
            target_size = center_pred.shape[-2:]
            center_targets_resized = F.interpolate(
                center_targets, size=target_size, mode='bilinear', align_corners=False
            )
            loss_center = self._gaussian_focal_loss(center_pred, center_targets_resized)
            losses['loss_center'] = loss_center * self.loss_center_weight

        # Offset loss
        if offset_targets is not None and offset_weights is not None:
            offset_pred = predictions['offset']
            target_size = offset_pred.shape[-2:]
            offset_targets_resized = F.interpolate(
                offset_targets, size=target_size, mode='bilinear', align_corners=False
            )
            offset_weights_resized = F.interpolate(
                offset_weights.unsqueeze(1), size=target_size, mode='nearest'
            ).squeeze(1)

            loss_offset = F.l1_loss(
                offset_pred * offset_weights_resized.unsqueeze(1),
                offset_targets_resized * offset_weights_resized.unsqueeze(1),
                reduction='sum',
            ) / (offset_weights_resized.sum() + 1e-8)
            losses['loss_offset'] = loss_offset * self.loss_offset_weight

        return losses

    def _gaussian_focal_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 2.0,
        gamma: float = 4.0,
    ) -> torch.Tensor:
        """Gaussian focal loss for center heatmap.

        Args:
            pred: Predicted heatmap (B, C, H, W).
            target: Target heatmap with Gaussian peaks (B, C, H, W).
            alpha: Focal loss alpha.
            gamma: Focal loss gamma.

        Returns:
            Loss value.
        """
        pred = pred.sigmoid()
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = -((1 - pred) ** alpha) * torch.log(pred + 1e-8) * pos_mask
        neg_loss = -((1 - target) ** gamma) * (pred ** alpha) * torch.log(1 - pred + 1e-8) * neg_mask

        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            return neg_loss
        return (pos_loss + neg_loss) / num_pos

    def predict(
        self,
        features: List[torch.Tensor],
        img_size: Tuple[int, int],
        center_threshold: float = 0.1,
        nms_kernel: int = 3,
        top_k: int = 100,
    ) -> Dict[str, torch.Tensor]:
        """Predict panoptic segmentation with instance mask post-processing.

        Args:
            features: Multi-scale FPN features.
            img_size: Original image size (H, W).
            center_threshold: Threshold for center detection.
            nms_kernel: Kernel size for NMS.
            top_k: Maximum number of instances per image.

        Returns:
            Dict with 'semantic', 'instance', 'center', 'offset', 'embedding'.
        """
        outputs = self.forward(features)

        # Upsample to original size
        sem_logits = F.interpolate(outputs['sem_logits'], size=img_size, mode='bilinear', align_corners=False)
        center = F.interpolate(outputs['center'], size=img_size, mode='bilinear', align_corners=False)
        offset = F.interpolate(outputs['offset'], size=img_size, mode='bilinear', align_corners=False)
        embedding = F.interpolate(outputs['embedding'], size=img_size, mode='bilinear', align_corners=False)

        center_heatmap = center.sigmoid()
        semantic = sem_logits.argmax(dim=1)

        # Generate instance masks
        instance = self._generate_instance_masks(
            center_heatmap=center_heatmap,
            offset=offset,
            embedding=embedding,
            semantic=semantic,
            center_threshold=center_threshold,
            nms_kernel=nms_kernel,
            top_k=top_k,
        )

        return {
            'semantic': semantic,
            'instance': instance,
            'center': center_heatmap,
            'offset': offset,
            'embedding': embedding,
        }

    def _nms_heatmap(self, heatmap: torch.Tensor, kernel: int = 3) -> torch.Tensor:
        """Apply non-maximum suppression to heatmap.

        Args:
            heatmap: Center heatmap (B, C, H, W).
            kernel: Max pooling kernel size.

        Returns:
            NMS-applied heatmap with only local maxima.
        """
        pad = (kernel - 1) // 2
        hmax = F.max_pool2d(heatmap, kernel, stride=1, padding=pad)
        keep = (hmax == heatmap).float()
        return heatmap * keep

    def _generate_instance_masks(
        self,
        center_heatmap: torch.Tensor,
        offset: torch.Tensor,
        embedding: torch.Tensor,
        semantic: torch.Tensor,
        center_threshold: float,
        nms_kernel: int,
        top_k: int,
    ) -> torch.Tensor:
        """Generate instance segmentation masks.

        Args:
            center_heatmap: Center heatmap (B, num_things, H, W).
            offset: Center offset (B, 2, H, W).
            embedding: Instance embedding (B, embed_dim, H, W).
            semantic: Semantic segmentation (B, H, W).
            center_threshold: Threshold for center detection.
            nms_kernel: NMS kernel size.
            top_k: Max instances per image.

        Returns:
            Instance mask (B, H, W) where each pixel has instance ID (0=background).
        """
        B, _, H, W = center_heatmap.shape
        device = center_heatmap.device

        # Apply NMS to find center peaks
        center_nms = self._nms_heatmap(center_heatmap, nms_kernel)

        # Create instance masks for each batch
        instance_masks = torch.zeros(B, H, W, dtype=torch.long, device=device)

        for b in range(B):
            # Find top-k centers across all thing classes
            center_flat = center_nms[b].view(-1)  # (num_things * H * W)
            scores, indices = center_flat.topk(min(top_k, center_flat.numel()))

            # Filter by threshold
            valid_mask = scores > center_threshold
            scores = scores[valid_mask]
            indices = indices[valid_mask]

            if len(indices) == 0:
                continue

            # Convert flat indices to (y, x)
            spatial_indices = indices % (H * W)
            y_indices = spatial_indices // W
            x_indices = spatial_indices % W

            # Apply offset correction
            offsets_y = offset[b, 0, y_indices, x_indices]
            offsets_x = offset[b, 1, y_indices, x_indices]
            center_y = (y_indices.float() + offsets_y).clamp(0, H - 1)
            center_x = (x_indices.float() + offsets_x).clamp(0, W - 1)

            # Get center embeddings
            center_y_int = center_y.long()
            center_x_int = center_x.long()
            center_embeds = embedding[b, :, center_y_int, center_x_int]  # (embed_dim, num_centers)

            # Get all pixel embeddings
            pixel_embeds = embedding[b]  # (embed_dim, H, W)
            pixel_embeds_flat = pixel_embeds.view(pixel_embeds.shape[0], -1)  # (embed_dim, H*W)

            # Compute distance from each pixel to each center embedding
            # Using L2 distance in embedding space
            distances = torch.cdist(
                pixel_embeds_flat.T,  # (H*W, embed_dim)
                center_embeds.T,       # (num_centers, embed_dim)
            )  # (H*W, num_centers)

            # Assign each pixel to nearest center (if within thing class)
            _, nearest_center = distances.min(dim=1)  # (H*W,)
            nearest_center = nearest_center.view(H, W)

            # Only assign instance IDs to thing class pixels
            thing_mask = semantic[b] < self.num_things_classes
            instance_masks[b] = (nearest_center + 1) * thing_mask.long()

        return instance_masks
