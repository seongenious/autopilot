"""Complete perception model for 3D object detection.

Combines:
    - Multi-camera image encoding (ResNet + BiFPN per camera)
    - BEV transformation (cross-attention transformer)
    - 3D object detection head
"""

from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn

from autopilot.models.backbones import BiFPN, ResNetBackbone
from autopilot.models.necks import BEVTransformer
from autopilot.models.heads.detection import CenterHead, DetectionHead


class PerceptionModel(nn.Module):
    """End-to-end perception model for autonomous driving.

    Architecture:
        1. Multi-camera images → ResNet backbone (per camera)
        2. Multi-scale features → BiFPN (per camera)
        3. Multi-camera features → BEV Transformer → BEV features
        4. BEV features → Detection Head → 3D boxes

    Args:
        backbone_cfg: Backbone configuration.
        neck_cfg: BiFPN neck configuration.
        bev_cfg: BEV transformer configuration.
        head_cfg: Detection head configuration.
        num_cameras: Number of camera views.

    Example:
        >>> model = PerceptionModel(
        ...     backbone_cfg={'type': 'ResNetBackbone', 'model_name': 'resnet18'},
        ...     neck_cfg={'in_channels': [64, 128, 256, 512], 'out_channels': 128},
        ...     bev_cfg={'in_channels': 128, 'embed_dim': 256, 'bev_h': 20, 'bev_w': 80},
        ...     head_cfg={'in_channels': 256, 'num_classes': 10},
        ... )
        >>> images = torch.randn(2, 6, 3, 480, 640)  # B, num_cam, C, H, W
        >>> outputs = model({'images': images})
    """

    def __init__(
        self,
        backbone_cfg: Optional[Dict] = None,
        neck_cfg: Optional[Dict] = None,
        bev_cfg: Optional[Dict] = None,
        head_cfg: Optional[Dict] = None,
        num_cameras: int = 6,
    ) -> None:
        super().__init__()

        self.num_cameras = num_cameras

        # Default configurations
        backbone_cfg = backbone_cfg or {
            'model_name': 'resnet18',
            'pretrained': True,
            'out_indices': (0, 1, 2, 3),
        }
        neck_cfg = neck_cfg or {
            'in_channels': [64, 128, 256, 512],
            'out_channels': 128,
            'num_levels': 4,
            'num_layers': 3,
        }
        bev_cfg = bev_cfg or {
            'in_channels': 128,
            'embed_dim': 256,
            'bev_h': 20,
            'bev_w': 80,
            'num_cameras': num_cameras,
            'num_layers': 3,
        }
        head_cfg = head_cfg or {
            'in_channels': 256,
            'num_classes': 10,
        }

        # Build backbone (shared across cameras)
        self.backbone = ResNetBackbone(**backbone_cfg)

        # Build neck (shared across cameras)
        self.neck = BiFPN(**neck_cfg)

        # Build BEV transformer
        bev_cfg['num_cameras'] = num_cameras
        self.bev_transformer = BEVTransformer(**bev_cfg)

        # Build detection head
        self.detection_head = CenterHead(**head_cfg)

    def extract_camera_features(
        self,
        images: torch.Tensor,
    ) -> List[List[torch.Tensor]]:
        """Extract features from each camera view.

        Args:
            images: Multi-view images (B, num_cam, C, H, W).

        Returns:
            List of [num_cameras] lists of multi-scale features.
            Each camera has [num_levels] feature tensors.
        """
        B, num_cam, C, H, W = images.shape

        # Process all cameras
        all_cam_features = []
        for cam_idx in range(num_cam):
            cam_images = images[:, cam_idx]  # (B, C, H, W)

            # Backbone
            backbone_features = self.backbone(cam_images)

            # Neck
            neck_features = self.neck(backbone_features)

            all_cam_features.append(neck_features)

        return all_cam_features

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Dictionary containing:
                - images: (B, num_cam, C, H, W) multi-view images

        Returns:
            Dictionary containing:
                - heatmap: (B, num_classes, H, W) classification heatmap
                - center: (B, 2, H, W) center offset
                - height: (B, 1, H, W) height
                - dim: (B, 3, H, W) dimensions
                - rot: (B, 2, H, W) rotation
                - bev_features: (B, embed_dim, bev_h, bev_w) BEV features
        """
        images = batch['images']  # (B, num_cam, C, H, W)

        # Extract multi-camera multi-scale features
        cam_features = self.extract_camera_features(images)

        # Transform to BEV
        bev_features = self.bev_transformer(cam_features)

        # Detection head
        det_outputs = self.detection_head(bev_features)

        # Add BEV features for potential downstream use
        det_outputs['bev_features'] = bev_features

        return det_outputs


class SimplePerceptionModel(nn.Module):
    """Simplified perception model for faster iteration.

    Uses simpler detection head for initial experiments.
    """

    def __init__(
        self,
        backbone_cfg: Optional[Dict] = None,
        neck_cfg: Optional[Dict] = None,
        bev_cfg: Optional[Dict] = None,
        num_classes: int = 10,
        num_cameras: int = 6,
    ) -> None:
        super().__init__()

        self.num_cameras = num_cameras
        self.num_classes = num_classes

        # Default configurations
        backbone_cfg = backbone_cfg or {
            'model_name': 'resnet18',
            'pretrained': True,
            'out_indices': (0, 1, 2, 3),
        }
        neck_cfg = neck_cfg or {
            'in_channels': [64, 128, 256, 512],
            'out_channels': 128,
            'num_levels': 4,
            'num_layers': 2,
        }
        bev_cfg = bev_cfg or {
            'in_channels': 128,
            'embed_dim': 256,
            'bev_h': 20,
            'bev_w': 80,
            'num_cameras': num_cameras,
            'num_layers': 2,
        }

        # Build components
        self.backbone = ResNetBackbone(**backbone_cfg)
        self.neck = BiFPN(**neck_cfg)

        bev_cfg['num_cameras'] = num_cameras
        self.bev_transformer = BEVTransformer(**bev_cfg)

        # Simple detection head
        bev_dim = bev_cfg.get('embed_dim', 256)
        self.detection_head = DetectionHead(
            in_channels=bev_dim,
            num_classes=num_classes,
            hidden_channels=128,
        )

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        images = batch['images']  # (B, num_cam, C, H, W)
        B, num_cam, C, H, W = images.shape

        # Extract features from all cameras
        cam_features = []
        for cam_idx in range(num_cam):
            cam_images = images[:, cam_idx]
            backbone_features = self.backbone(cam_images)
            neck_features = self.neck(backbone_features)
            cam_features.append(neck_features)

        # Transform to BEV
        bev_features = self.bev_transformer(cam_features)

        # Detection
        det_outputs = self.detection_head(bev_features)
        det_outputs['bev_features'] = bev_features

        return det_outputs
