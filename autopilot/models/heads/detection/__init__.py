"""3D Object Detection heads.

Supports various detection paradigms:
    - Anchor-based detection
    - Anchor-free detection (e.g., CenterPoint style)
    - Transformer-based detection (e.g., DETR3D style)
"""

from autopilot.models.heads.detection.center_head import CenterHead, DetectionHead

__all__ = ['CenterHead', 'DetectionHead']
