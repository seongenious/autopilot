"""Neck modules for feature aggregation."""

from .bifpn import BiFPN
from .context import ContextAggregator

__all__ = ['BiFPN', 'ContextAggregator']
