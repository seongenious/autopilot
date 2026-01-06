"""Task-specific prediction heads.

Includes:
    - detection: 3D object detection heads
    - map: Map prediction and lane detection heads
    - depth: Depth estimation heads
"""

from autopilot.models.heads import depth, detection, map, planning

__all__ = ['detection', 'map', 'depth', 'planning']
