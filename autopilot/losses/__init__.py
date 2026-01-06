"""Loss functions for perception tasks.

Includes:
    - Detection losses (focal loss, L1, IoU)
    - Segmentation losses (cross-entropy, dice)
    - Depth losses (scale-invariant, smoothness)
"""

from autopilot.losses.detection import (
    DetectionLoss,
    FocalLoss,
    RegLoss,
    SimplifiedDetectionLoss,
)

__all__ = ['FocalLoss', 'RegLoss', 'DetectionLoss', 'SimplifiedDetectionLoss']
