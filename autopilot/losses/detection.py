"""Loss functions for 3D object detection.

Includes:
    - FocalLoss: Focal loss for heatmap classification
    - L1Loss: Smooth L1 loss for regression
    - DetectionLoss: Combined loss for CenterPoint-style detection
"""

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.utils.registry import LOSSES


@LOSSES.register_module()
class FocalLoss(nn.Module):
    """Focal loss for classification.

    Focal loss addresses class imbalance by down-weighting easy examples.

    Args:
        alpha: Balancing factor for positive/negative examples.
        gamma: Focusing parameter to reduce loss for well-classified examples.
        reduction: Reduction method ('mean', 'sum', 'none').

    Reference:
        - Paper: https://arxiv.org/abs/1708.02002
    """

    def __init__(
        self,
        alpha: float = 2.0,
        gamma: float = 4.0,
        reduction: str = 'mean',
    ) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Compute focal loss.

        Args:
            pred: Predicted logits (B, C, H, W).
            target: Ground truth heatmap (B, C, H, W).

        Returns:
            Focal loss value.
        """
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)

        # Positive and negative positions
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        # Positive loss
        pos_loss = -torch.pow(1 - pred, self.alpha) * torch.log(pred) * pos_mask

        # Negative loss with penalty reduction for near-positive locations
        neg_weight = torch.pow(1 - target, self.gamma)
        neg_loss = -neg_weight * torch.pow(pred, self.alpha) * torch.log(1 - pred) * neg_mask

        # Normalize by number of positive samples
        num_pos = pos_mask.sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / num_pos

        return loss


@LOSSES.register_module()
class RegLoss(nn.Module):
    """Regression loss for bounding box parameters.

    Uses L1 loss at positive (object center) locations only.

    Args:
        reduction: Reduction method ('mean', 'sum').
    """

    def __init__(self, reduction: str = 'mean') -> None:
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        indices: torch.Tensor,
        num_boxes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute regression loss at positive locations.

        Args:
            pred: Predicted values (B, C, H, W).
            target: Ground truth regression targets (B, max_boxes, C).
            indices: Flattened indices of positive locations (B, max_boxes).
            num_boxes: Number of valid boxes per sample (B,).

        Returns:
            L1 regression loss.
        """
        B, C, H, W = pred.shape
        pred_flat = pred.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)

        total_loss = 0.0
        total_count = 0

        for b in range(B):
            n = num_boxes[b].item()
            if n == 0:
                continue

            # Gather predictions at positive locations
            inds = indices[b, :n].long()  # (n,)
            pred_b = pred_flat[b, inds]  # (n, C)
            target_b = target[b, :n]  # (n, C)

            # L1 loss
            loss = F.l1_loss(pred_b, target_b, reduction='sum')
            total_loss += loss
            total_count += n * C

        if total_count == 0:
            return pred.sum() * 0

        return total_loss / total_count


@LOSSES.register_module()
class DetectionLoss(nn.Module):
    """Combined loss for CenterPoint-style 3D object detection.

    Combines focal loss for heatmap and L1 loss for regression targets.

    Args:
        heatmap_weight: Weight for heatmap loss.
        offset_weight: Weight for center offset loss.
        height_weight: Weight for height loss.
        dim_weight: Weight for dimension loss.
        rot_weight: Weight for rotation loss.
        vel_weight: Weight for velocity loss.
    """

    def __init__(
        self,
        heatmap_weight: float = 1.0,
        offset_weight: float = 1.0,
        height_weight: float = 0.25,
        dim_weight: float = 0.2,
        rot_weight: float = 1.0,
        vel_weight: float = 0.2,
    ) -> None:
        super().__init__()

        self.heatmap_weight = heatmap_weight
        self.offset_weight = offset_weight
        self.height_weight = height_weight
        self.dim_weight = dim_weight
        self.rot_weight = rot_weight
        self.vel_weight = vel_weight

        self.focal_loss = FocalLoss()
        self.reg_loss = RegLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_heatmap: torch.Tensor,
        gt_reg: torch.Tensor,
        gt_indices: torch.Tensor,
        num_boxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute detection loss.

        Args:
            pred: Dictionary with predicted tensors:
                - heatmap: (B, num_classes, H, W)
                - center: (B, 2, H, W)
                - height: (B, 1, H, W)
                - dim: (B, 3, H, W)
                - rot: (B, 2, H, W)
                - vel: (B, 2, H, W) optional
            gt_heatmap: Ground truth heatmap (B, num_classes, H, W).
            gt_reg: Ground truth regression targets (B, max_boxes, 8).
            gt_indices: Positive location indices (B, max_boxes).
            num_boxes: Number of valid boxes per sample (B,).

        Returns:
            Dictionary with individual losses and total loss.
        """
        losses = {}

        # Heatmap loss (focal loss)
        loss_heatmap = self.focal_loss(pred['heatmap'], gt_heatmap)
        losses['loss_heatmap'] = loss_heatmap * self.heatmap_weight

        # Split regression targets
        # gt_reg: [offset_x, offset_y, z, log(l), log(w), log(h), sin, cos]
        gt_offset = gt_reg[..., :2]  # (B, max_boxes, 2)
        gt_height = gt_reg[..., 2:3]  # (B, max_boxes, 1)
        gt_dim = gt_reg[..., 3:6]  # (B, max_boxes, 3)
        gt_rot = gt_reg[..., 6:8]  # (B, max_boxes, 2)

        # Center offset loss
        pred_center = pred['center']  # (B, 2, H, W)
        loss_offset = self.reg_loss(pred_center, gt_offset, gt_indices, num_boxes)
        losses['loss_offset'] = loss_offset * self.offset_weight

        # Height loss
        pred_height = pred['height']  # (B, 1, H, W)
        loss_height = self.reg_loss(pred_height, gt_height, gt_indices, num_boxes)
        losses['loss_height'] = loss_height * self.height_weight

        # Dimension loss
        pred_dim = pred['dim']  # (B, 3, H, W)
        loss_dim = self.reg_loss(pred_dim, gt_dim, gt_indices, num_boxes)
        losses['loss_dim'] = loss_dim * self.dim_weight

        # Rotation loss
        pred_rot = pred['rot']  # (B, 2, H, W)
        loss_rot = self.reg_loss(pred_rot, gt_rot, gt_indices, num_boxes)
        losses['loss_rot'] = loss_rot * self.rot_weight

        # Total loss
        total_loss = sum(losses.values())
        losses['loss'] = total_loss

        return losses


@LOSSES.register_module()
class SimplifiedDetectionLoss(nn.Module):
    """Simplified detection loss for DetectionHead.

    Uses focal loss for classification and L1 for regression.
    """

    def __init__(
        self,
        cls_weight: float = 1.0,
        reg_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.cls_weight = cls_weight
        self.reg_weight = reg_weight
        self.focal_loss = FocalLoss()

    def forward(
        self,
        pred: Dict[str, torch.Tensor],
        gt_heatmap: torch.Tensor,
        gt_reg: torch.Tensor,
        gt_indices: torch.Tensor,
        num_boxes: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute simplified detection loss.

        Args:
            pred: Dictionary with 'cls' and 'reg' predictions.
            gt_heatmap: Ground truth heatmap.
            gt_reg: Ground truth regression targets.
            gt_indices: Positive location indices.
            num_boxes: Number of valid boxes per sample.

        Returns:
            Dictionary with losses.
        """
        losses = {}

        # Classification loss
        loss_cls = self.focal_loss(pred['cls'], gt_heatmap)
        losses['loss_cls'] = loss_cls * self.cls_weight

        # Regression loss
        B, C, H, W = pred['reg'].shape
        pred_flat = pred['reg'].view(B, C, -1).permute(0, 2, 1)

        total_reg_loss = 0.0
        total_count = 0

        for b in range(B):
            n = num_boxes[b].item()
            if n == 0:
                continue

            inds = gt_indices[b, :n].long()
            pred_b = pred_flat[b, inds]
            target_b = gt_reg[b, :n]

            loss = F.l1_loss(pred_b, target_b, reduction='sum')
            total_reg_loss += loss
            total_count += n * C

        if total_count > 0:
            loss_reg = total_reg_loss / total_count
        else:
            loss_reg = pred['reg'].sum() * 0

        losses['loss_reg'] = loss_reg * self.reg_weight
        losses['loss'] = losses['loss_cls'] + losses['loss_reg']

        return losses
