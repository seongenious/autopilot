"""CenterPoint-style 3D Detection Head.

Anchor-free 3D object detection head that predicts:
    - Center heatmap for classification
    - Regression targets: x, y, z, l, w, h, yaw (via sin/cos)

Reference:
    - CenterPoint: https://arxiv.org/abs/2006.11275
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from autopilot.utils.registry import HEADS


class ConvBlock(nn.Module):
    """Convolution block with BatchNorm and ReLU."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        padding: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class SeparateHead(nn.Module):
    """Separate prediction head for a specific task."""

    def __init__(
        self,
        in_channels: int,
        head_channels: int,
        num_convs: int,
        out_channels: int,
        final_kernel: int = 1,
        bias_fill: Optional[float] = None,
    ) -> None:
        """Initialize separate head.

        Args:
            in_channels: Input channels.
            head_channels: Hidden channels.
            num_convs: Number of conv layers.
            out_channels: Output channels.
            final_kernel: Final conv kernel size.
            bias_fill: Value to fill final bias (e.g., -2.19 for heatmap).
        """
        super().__init__()

        layers = []
        for i in range(num_convs):
            in_ch = in_channels if i == 0 else head_channels
            layers.append(ConvBlock(in_ch, head_channels))

        self.convs = nn.Sequential(*layers)
        self.final = nn.Conv2d(
            head_channels, out_channels, final_kernel,
            padding=final_kernel // 2
        )

        if bias_fill is not None:
            self.final.bias.data.fill_(bias_fill)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.convs(x)
        return self.final(x)


@HEADS.register_module()
class CenterHead(nn.Module):
    """CenterPoint-style 3D detection head.

    Predicts object centers as heatmap peaks and regresses bounding box
    parameters at those locations.

    Args:
        in_channels: Input feature channels (from BEV features).
        head_channels: Hidden channels in heads.
        num_classes: Number of object classes.
        num_convs: Number of conv layers per head.
        box_code_size: Size of box encoding (default 8: x, y, z, l, w, h, sin, cos).
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        voxel_size: [vx, vy, vz] voxel dimensions.
        out_size_factor: Output size factor relative to input.

    Example:
        >>> head = CenterHead(
        ...     in_channels=256,
        ...     num_classes=10,
        ...     point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        ... )
        >>> bev_features = torch.randn(2, 256, 20, 80)
        >>> outputs = head(bev_features)
        >>> print(outputs['heatmap'].shape)  # (2, 10, 20, 80)
    """

    def __init__(
        self,
        in_channels: int = 256,
        head_channels: int = 64,
        num_classes: int = 10,
        num_convs: int = 2,
        box_code_size: int = 8,
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        voxel_size: List[float] = [0.2, 0.2, 8.0],
        out_size_factor: int = 1,
    ) -> None:
        """Initialize CenterHead."""
        super().__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.box_code_size = box_code_size
        self.point_cloud_range = point_cloud_range
        self.voxel_size = voxel_size
        self.out_size_factor = out_size_factor

        # Shared convolutions
        self.shared_conv = nn.Sequential(
            ConvBlock(in_channels, head_channels),
            ConvBlock(head_channels, head_channels),
        )

        # Heatmap head (classification)
        # Use -2.19 bias initialization for focal loss
        self.heatmap_head = SeparateHead(
            head_channels, head_channels, num_convs, num_classes,
            bias_fill=-2.19
        )

        # Regression heads
        # Center offset (x, y) - offset from grid center
        self.center_head = SeparateHead(
            head_channels, head_channels, num_convs, 2
        )

        # Height (z)
        self.height_head = SeparateHead(
            head_channels, head_channels, num_convs, 1
        )

        # Dimension (log(l), log(w), log(h))
        self.dim_head = SeparateHead(
            head_channels, head_channels, num_convs, 3
        )

        # Rotation (sin(yaw), cos(yaw))
        self.rot_head = SeparateHead(
            head_channels, head_channels, num_convs, 2
        )

        # Optional: velocity for tracking (vx, vy)
        self.vel_head = SeparateHead(
            head_channels, head_channels, num_convs, 2
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: BEV features of shape (B, C, H, W).

        Returns:
            Dictionary containing:
                - heatmap: (B, num_classes, H, W) class heatmap
                - center: (B, 2, H, W) center offset
                - height: (B, 1, H, W) z height
                - dim: (B, 3, H, W) log dimensions
                - rot: (B, 2, H, W) sin/cos rotation
                - vel: (B, 2, H, W) velocity (optional)
        """
        x = self.shared_conv(x)

        outputs = {
            'heatmap': self.heatmap_head(x),
            'center': self.center_head(x),
            'height': self.height_head(x),
            'dim': self.dim_head(x),
            'rot': self.rot_head(x),
            'vel': self.vel_head(x),
        }

        return outputs

    def decode_predictions(
        self,
        outputs: Dict[str, torch.Tensor],
        score_threshold: float = 0.1,
        max_objects: int = 500,
    ) -> List[Dict[str, torch.Tensor]]:
        """Decode network outputs to 3D bounding boxes.

        Args:
            outputs: Network outputs dictionary.
            score_threshold: Minimum score threshold.
            max_objects: Maximum number of objects per sample.

        Returns:
            List of dictionaries per batch sample containing:
                - boxes: (N, 7) [x, y, z, l, w, h, yaw]
                - scores: (N,) confidence scores
                - labels: (N,) class labels
        """
        heatmap = outputs['heatmap']
        center = outputs['center']
        height = outputs['height']
        dim = outputs['dim']
        rot = outputs['rot']

        B, num_classes, H, W = heatmap.shape
        device = heatmap.device

        # Apply sigmoid to heatmap
        heatmap = torch.sigmoid(heatmap)

        batch_results = []
        for b in range(B):
            # Find local maxima (NMS style)
            heatmap_b = heatmap[b]  # (num_classes, H, W)
            heatmap_max = F.max_pool2d(
                heatmap_b.unsqueeze(0), kernel_size=3, stride=1, padding=1
            ).squeeze(0)
            keep = (heatmap_b == heatmap_max) & (heatmap_b >= score_threshold)

            # Get top-k scores
            scores_flat = heatmap_b.view(num_classes, -1)
            keep_flat = keep.view(num_classes, -1)

            all_scores = []
            all_indices = []
            all_labels = []

            for cls in range(num_classes):
                cls_scores = scores_flat[cls][keep_flat[cls]]
                cls_indices = torch.nonzero(keep_flat[cls]).squeeze(-1)
                cls_labels = torch.full_like(cls_scores, cls, dtype=torch.long)

                all_scores.append(cls_scores)
                all_indices.append(cls_indices)
                all_labels.append(cls_labels)

            if len(all_scores) == 0 or all([len(s) == 0 for s in all_scores]):
                batch_results.append({
                    'boxes': torch.zeros((0, 7), device=device),
                    'scores': torch.zeros((0,), device=device),
                    'labels': torch.zeros((0,), dtype=torch.long, device=device),
                })
                continue

            all_scores = torch.cat(all_scores)
            all_indices = torch.cat(all_indices)
            all_labels = torch.cat(all_labels)

            # Keep top-k
            if len(all_scores) > max_objects:
                topk_scores, topk_inds = all_scores.topk(max_objects)
                all_scores = topk_scores
                all_indices = all_indices[topk_inds]
                all_labels = all_labels[topk_inds]

            # Decode box parameters
            ys = (all_indices // W).float()
            xs = (all_indices % W).float()

            # Add center offset
            center_b = center[b]  # (2, H, W)
            xs = xs + center_b[0].view(-1)[all_indices]
            ys = ys + center_b[1].view(-1)[all_indices]

            # Convert to world coordinates
            x_range = self.point_cloud_range[3] - self.point_cloud_range[0]
            y_range = self.point_cloud_range[4] - self.point_cloud_range[1]

            xs = xs / W * x_range + self.point_cloud_range[0]
            ys = ys / H * y_range + self.point_cloud_range[1]

            # Height
            zs = height[b][0].view(-1)[all_indices]

            # Dimensions (exp of log values)
            dims = dim[b]  # (3, H, W)
            ls = torch.exp(dims[0].view(-1)[all_indices])
            ws = torch.exp(dims[1].view(-1)[all_indices])
            hs = torch.exp(dims[2].view(-1)[all_indices])

            # Rotation
            rot_b = rot[b]  # (2, H, W)
            sin_yaw = rot_b[0].view(-1)[all_indices]
            cos_yaw = rot_b[1].view(-1)[all_indices]
            yaws = torch.atan2(sin_yaw, cos_yaw)

            # Stack boxes
            boxes = torch.stack([xs, ys, zs, ls, ws, hs, yaws], dim=-1)

            batch_results.append({
                'boxes': boxes,
                'scores': all_scores,
                'labels': all_labels,
            })

        return batch_results


@HEADS.register_module()
class DetectionHead(nn.Module):
    """Simple detection head wrapper for compatibility.

    This is a simpler version that can be used for initial experiments.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 10,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes

        # Shared feature processing
        self.shared = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(inplace=True),
        )

        # Classification head
        self.cls_head = nn.Conv2d(hidden_channels, num_classes, 1)

        # Regression head: x, y, z, l, w, h, sin(yaw), cos(yaw)
        self.reg_head = nn.Conv2d(hidden_channels, 8, 1)

        # Initialize
        nn.init.constant_(self.cls_head.bias, -2.19)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            x: BEV features (B, C, H, W).

        Returns:
            Dictionary with 'cls' and 'reg' predictions.
        """
        feat = self.shared(x)

        cls_pred = self.cls_head(feat)  # (B, num_classes, H, W)
        reg_pred = self.reg_head(feat)  # (B, 8, H, W)

        return {
            'cls': cls_pred,
            'reg': reg_pred,
        }
