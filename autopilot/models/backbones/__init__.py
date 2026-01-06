"""Backbone modules for feature extraction.

Supports pretrained models from timm library:
    - ResNet: Standard residual network (64, 128, 256, 512 channels)
    - RegNet: Efficient and scalable network architecture
    - BiFPN: Bidirectional Feature Pyramid Network
"""

from autopilot.models.backbones.bifpn import BiFPN
from autopilot.models.backbones.regnet import RegNetBackbone
from autopilot.models.backbones.resnet import ResNetBackbone

__all__ = ['ResNetBackbone', 'RegNetBackbone', 'BiFPN']
