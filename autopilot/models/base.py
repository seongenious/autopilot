"""Base module for all model components."""

import logging
from abc import ABCMeta
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class BaseModule(nn.Module, metaclass=ABCMeta):
    """Base class for all modules in autopilot.

    Provides:
        - Unified weight initialization
        - Pretrained model loading
        - Config handling
    """

    def __init__(self, init_cfg: Optional[Union[Dict, List[Dict]]] = None):
        """Initialize BaseModule.

        Args:
            init_cfg: Initialization config. Examples:
                - dict(type='Pretrained', checkpoint='path/to/ckpt')
                - dict(type='Kaiming', layer='Conv2d')
                - dict(type='Xavier', layer='Linear')
        """
        super().__init__()
        self._is_init = False
        self.init_cfg = init_cfg

    @property
    def is_init(self) -> bool:
        """Check if module is initialized."""
        return self._is_init

    def init_weights(self) -> None:
        """Initialize weights based on init_cfg."""
        if self.init_cfg is None:
            return

        init_cfgs = self.init_cfg if isinstance(self.init_cfg, list) else [self.init_cfg]

        for cfg in init_cfgs:
            cfg = cfg.copy()
            init_type = cfg.pop('type')

            if init_type == 'Pretrained':
                self._load_pretrained(cfg.get('checkpoint'), cfg.get('prefix', ''))
            elif init_type == 'Kaiming':
                self._init_kaiming(cfg)
            elif init_type == 'Xavier':
                self._init_xavier(cfg)
            elif init_type == 'Constant':
                self._init_constant(cfg)
            else:
                raise ValueError(f'Unknown init type: {init_type}')

        self._is_init = True

        # Initialize children
        for m in self.children():
            if hasattr(m, 'init_weights') and not getattr(m, '_is_init', False):
                m.init_weights()

    def _load_pretrained(self, checkpoint: Optional[str], prefix: str = '') -> None:
        """Load pretrained weights."""
        if checkpoint is None:
            return

        logger.info(f'Loading pretrained weights from {checkpoint}')
        state_dict = torch.load(checkpoint, map_location='cpu')

        # Handle different checkpoint formats
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        elif 'model' in state_dict:
            state_dict = state_dict['model']

        # Handle prefix
        if prefix:
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}

        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f'Missing keys: {missing}')
        if unexpected:
            logger.warning(f'Unexpected keys: {unexpected}')

    def _init_kaiming(self, cfg: Dict[str, Any]) -> None:
        """Kaiming initialization."""
        layer_types = cfg.get('layer', ['Conv2d'])
        if isinstance(layer_types, str):
            layer_types = [layer_types]

        for m in self.modules():
            if any(m.__class__.__name__ == lt for lt in layer_types):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_xavier(self, cfg: Dict[str, Any]) -> None:
        """Xavier initialization."""
        layer_types = cfg.get('layer', ['Linear'])
        if isinstance(layer_types, str):
            layer_types = [layer_types]

        for m in self.modules():
            if any(m.__class__.__name__ == lt for lt in layer_types):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_constant(self, cfg: Dict[str, Any]) -> None:
        """Constant initialization."""
        layer_types = cfg.get('layer', ['BatchNorm2d'])
        val = cfg.get('val', 1)

        if isinstance(layer_types, str):
            layer_types = [layer_types]

        for m in self.modules():
            if any(m.__class__.__name__ == lt for lt in layer_types):
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.constant_(m.weight, val)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
