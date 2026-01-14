"""RegNet backbone using timm pretrained models."""

from typing import List, Tuple

import timm
import torch

from autopilot.models.base import BaseModule
from autopilot.utils.registry import BACKBONES


@BACKBONES.register_module()
class RegNet(BaseModule):
    """RegNet backbone with timm pretrained weights.

    Supported architectures and output channels:
        - regnetx_002: [24, 56, 152, 368]
        - regnetx_004: [32, 64, 160, 384]
        - regnetx_006: [48, 96, 240, 528]
        - regnetx_008: [64, 128, 288, 672]
        - regnetx_016: [72, 168, 408, 912]
        - regnetx_032: [96, 192, 432, 1008]
        - regnetx_040: [80, 240, 560, 1360]
        - regnetx_064: [168, 392, 784, 1624]
        - regnetx_080: [80, 240, 720, 1920]
        - regnety_002: [24, 56, 152, 368]
        - regnety_004: [48, 104, 208, 440]
        - regnety_006: [48, 112, 256, 608]
        - regnety_008: [64, 128, 320, 768]
        - regnety_016: [48, 120, 336, 888]
        - regnety_032: [72, 216, 576, 1512]
        - regnety_040: [128, 192, 512, 1088]
        - regnety_064: [144, 288, 576, 1296]
        - regnety_080: [168, 448, 896, 2016]
    """

    def __init__(
        self,
        arch: str = 'regnetx_032',
        pretrained: bool = True,
        out_indices: Tuple[int, ...] = (1, 2, 3, 4),
        frozen_stages: int = -1,
    ):
        """Initialize RegNet backbone.

        Args:
            arch: RegNet architecture name (e.g., 'regnetx_032', 'regnety_040').
            pretrained: Whether to use pretrained weights.
            out_indices: Output feature indices (1=s1, 2=s2, 3=s3, 4=s4). 0=stem.
            frozen_stages: Stages to freeze (-1=none, 0=stem, 1=stem+s1, ...).
        """
        super().__init__()

        self.arch = arch
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # Create model with timm
        self.model = timm.create_model(
            arch,
            pretrained=pretrained,
            features_only=True,
            out_indices=out_indices,
        )

        # Get output channels from timm
        self.out_channels = self.model.feature_info.channels()

        # Freeze stages
        self._freeze_stages()

    def _freeze_stages(self) -> None:
        """Freeze stages based on frozen_stages setting."""
        if self.frozen_stages < 0:
            return

        # Freeze stem
        if self.frozen_stages >= 0:
            self.model.stem.eval()
            for param in self.model.stem.parameters():
                param.requires_grad = False

        # Freeze stages (s1, s2, s3, s4)
        for i in range(1, self.frozen_stages + 1):
            stage = getattr(self.model, f's{i}', None)
            if stage is not None:
                stage.eval()
                for param in stage.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Forward pass.

        Args:
            x: Input tensor (B, 3, H, W).

        Returns:
            List of feature maps at different scales.
        """
        return self.model(x)

    def train(self, mode: bool = True) -> 'RegNet':
        """Set training mode, keeping frozen stages in eval."""
        super().train(mode)
        self._freeze_stages()
        return self
