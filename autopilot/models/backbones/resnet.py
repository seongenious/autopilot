"""ResNet backbone using timm pretrained models.

ResNet provides standard channel dimensions (64, 128, 256, 512 for ResNet18/34)
which are commonly used in perception tasks.

Reference:
    - Paper: https://arxiv.org/abs/1512.03385
    - timm: https://github.com/huggingface/pytorch-image-models
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError('Please install timm: pip install timm')

from autopilot.utils.registry import BACKBONES


@BACKBONES.register_module()
class ResNetBackbone(nn.Module):
    """ResNet backbone with timm pretrained weights.

    Extracts multi-scale features from input images using ResNet architecture.
    ResNet18/34 outputs channels [64, 128, 256, 512] at strides [4, 8, 16, 32].

    Args:
        model_name: timm model name (e.g., 'resnet18', 'resnet34', 'resnet50').
        pretrained: Whether to load pretrained weights.
        out_indices: Indices of stages to output features from (0-3).
        frozen_stages: Number of stages to freeze (-1 means no freezing).
        norm_eval: Whether to set BN layers to eval mode during training.

    Example:
        >>> backbone = ResNetBackbone(
        ...     model_name='resnet18',
        ...     pretrained=True,
        ...     out_indices=(0, 1, 2, 3),
        ... )
        >>> x = torch.randn(1, 3, 480, 640)
        >>> features = backbone(x)
        >>> for i, f in enumerate(features):
        ...     print(f'Stage {i}: {f.shape}')
        # Stage 0: torch.Size([1, 64, 120, 160])
        # Stage 1: torch.Size([1, 128, 60, 80])
        # Stage 2: torch.Size([1, 256, 30, 40])
        # Stage 3: torch.Size([1, 512, 15, 20])
    """

    AVAILABLE_MODELS = [
        'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
        'resnet18d', 'resnet34d', 'resnet50d',
        'resnext50_32x4d', 'resnext101_32x8d',
    ]

    # Channel dimensions for different ResNet variants
    CHANNEL_CONFIGS = {
        'resnet18': [64, 128, 256, 512],
        'resnet34': [64, 128, 256, 512],
        'resnet18d': [64, 128, 256, 512],
        'resnet34d': [64, 128, 256, 512],
        'resnet50': [256, 512, 1024, 2048],
        'resnet50d': [256, 512, 1024, 2048],
        'resnet101': [256, 512, 1024, 2048],
        'resnet152': [256, 512, 1024, 2048],
        'resnext50_32x4d': [256, 512, 1024, 2048],
        'resnext101_32x8d': [256, 512, 1024, 2048],
    }

    def __init__(
        self,
        model_name: str = 'resnet18',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        norm_eval: bool = False,
    ) -> None:
        """Initialize ResNetBackbone."""
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.norm_eval = norm_eval

        # Create model with feature extraction
        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=[i + 1 for i in out_indices],  # timm uses 1-indexed for resnet
        )

        # Get feature info
        self.feature_info = self.model.feature_info

        # Get output channels for selected indices
        all_channels = self.CHANNEL_CONFIGS.get(
            model_name.split('.')[0],  # Handle variants like resnet18.a1_in1k
            [64, 128, 256, 512]
        )
        self._out_channels = [all_channels[i] for i in out_indices]

        # Freeze stages if specified
        self._freeze_stages()

    @property
    def out_channels(self) -> List[int]:
        """Return output channel dimensions for each stage."""
        return self._out_channels

    def _freeze_stages(self) -> None:
        """Freeze early stages of the network."""
        if self.frozen_stages < 0:
            return

        # Freeze stem (conv1 + bn1)
        if self.frozen_stages >= 0:
            if hasattr(self.model, 'conv1'):
                self.model.conv1.eval()
                for param in self.model.conv1.parameters():
                    param.requires_grad = False
            if hasattr(self.model, 'bn1'):
                self.model.bn1.eval()
                for param in self.model.bn1.parameters():
                    param.requires_grad = False

        # Freeze stages (layer1, layer2, ...)
        for i in range(min(self.frozen_stages, 4)):
            layer = getattr(self.model, f'layer{i + 1}', None)
            if layer is not None:
                layer.eval()
                for param in layer.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> 'ResNetBackbone':
        """Set training mode, optionally keeping BN in eval mode."""
        super().train(mode)

        self._freeze_stages()

        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()

        return self

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor of shape (B, 3, H, W).

        Returns:
            List of feature maps at different scales.
        """
        return self.model(x)

    def get_output_info(self) -> List[Dict]:
        """Get information about output features."""
        strides = [4, 8, 16, 32]
        return [
            {
                'index': idx,
                'channels': self._out_channels[i],
                'stride': strides[idx],
            }
            for i, idx in enumerate(self.out_indices)
        ]
