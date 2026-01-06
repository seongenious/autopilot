"""Neck modules for feature aggregation.

Necks process backbone features before passing to task heads.
"""

from autopilot.models.necks.bev_transformer import BEVTransformer

__all__ = ['BEVTransformer']
