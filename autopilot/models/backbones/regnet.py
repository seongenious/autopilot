"""RegNet backbone using timm pretrained models.

RegNet is a family of efficient network architectures designed through
neural architecture search, providing good accuracy-efficiency trade-offs.

Reference:
    - Paper: https://arxiv.org/abs/2003.13678
    - timm: https://github.com/huggingface/pytorch-image-models
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError('Please install timm: pip install timm')

from autopilot.utils.registry import BACKBONES


@BACKBONES.register_module()
class RegNetBackbone(nn.Module):
    """RegNet backbone with timm pretrained weights.

    Extracts multi-scale features from input images using RegNet architecture.
    Supports various RegNet variants (regnetx_*, regnety_*).

    Args:
        model_name: timm model name (e.g., 'regnetx_032', 'regnety_040').
        pretrained: Whether to load pretrained weights.
        out_indices: Indices of stages to output features from (0-3).
        frozen_stages: Number of stages to freeze (-1 means no freezing).
        norm_eval: Whether to set BN layers to eval mode during training.

    Example:
        >>> backbone = RegNetBackbone(
        ...     model_name='regnetx_032',
        ...     pretrained=True,
        ...     out_indices=(1, 2, 3),
        ... )
        >>> x = torch.randn(1, 3, 640, 960)
        >>> features = backbone(x)
        >>> for i, f in enumerate(features):
        ...     print(f'Stage {i}: {f.shape}')
    """

    AVAILABLE_MODELS = [
        # RegNetX variants
        'regnetx_002', 'regnetx_004', 'regnetx_006', 'regnetx_008',
        'regnetx_016', 'regnetx_032', 'regnetx_040', 'regnetx_064',
        'regnetx_080', 'regnetx_120', 'regnetx_160', 'regnetx_320',
        # RegNetY variants (with SE attention)
        'regnety_002', 'regnety_004', 'regnety_006', 'regnety_008',
        'regnety_016', 'regnety_032', 'regnety_040', 'regnety_064',
        'regnety_080', 'regnety_120', 'regnety_160', 'regnety_320',
    ]

    def __init__(
        self,
        model_name: str = 'regnetx_032',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (0, 1, 2, 3),
        frozen_stages: int = -1,
        norm_eval: bool = False,
    ) -> None:
        """Initialize RegNetBackbone."""
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
            out_indices=out_indices,
        )

        # Get feature info for downstream modules (only for selected out_indices)
        self.feature_info = self.model.feature_info
        self._out_channels = [self.feature_info[i]['num_chs'] for i in out_indices]

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

        # Freeze stem
        if self.frozen_stages >= 0:
            self.model.stem.eval()
            for param in self.model.stem.parameters():
                param.requires_grad = False

        # Freeze stages
        for i in range(min(self.frozen_stages, 4)):
            stage = getattr(self.model, f's{i + 1}', None)
            if stage is not None:
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def train(self, mode: bool = True) -> 'RegNetBackbone':
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
        return [
            {
                'index': i,
                'channels': self._out_channels[i],
                'stride': self.feature_info[i]['reduction'],
            }
            for i in range(len(self._out_channels))
        ]


def build_regnet(
    model_name: str = 'regnetx_032',
    pretrained: bool = True,
    **kwargs,
) -> RegNetBackbone:
    """Build RegNet backbone.

    Args:
        model_name: timm model name.
        pretrained: Whether to load pretrained weights.
        **kwargs: Additional arguments for RegNetBackbone.

    Returns:
        Configured RegNetBackbone instance.
    """
    return RegNetBackbone(
        model_name=model_name,
        pretrained=pretrained,
        **kwargs,
    )


# TODO: Add support for loading custom pretrained weights
# TODO: Add support for different input normalization schemes
# TODO: Implement feature map interpolation for consistent output sizes
