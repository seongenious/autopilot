"""Autopilot - Autonomous Driving Perception Framework.

A modular perception framework for autonomous driving that supports:
    - 3D Object Detection
    - Map Prediction (including lane detection)
    - Depth Estimation

Built with PyTorch Lightning and Hydra for configuration management.
"""

__version__ = '0.1.0'
__author__ = 'Autopilot Team'

from autopilot.utils.registry import Registry

__all__ = ['Registry', '__version__']
