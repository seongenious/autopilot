"""Model components for Autopilot perception.

This module contains:
    - Backbones: Feature extractors (RegNet, BiFPN, etc.)
    - Necks: Feature aggregation modules
    - Heads: Task-specific prediction heads (detection, map, depth)
"""

from autopilot.models import backbones, heads, necks

__all__ = ['backbones', 'necks', 'heads']
