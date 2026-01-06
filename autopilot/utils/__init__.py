"""Utility modules for Autopilot."""

from autopilot.utils.registry import Registry

__all__ = ['Registry']

# Lazy imports for optional dependencies
def get_visualization_callback():
    """Get VisualizationCallback (requires pytorch-lightning)."""
    from autopilot.utils.callbacks import VisualizationCallback
    return VisualizationCallback

def get_gradient_monitor_callback():
    """Get GradientMonitorCallback (requires pytorch-lightning)."""
    from autopilot.utils.callbacks import GradientMonitorCallback
    return GradientMonitorCallback
