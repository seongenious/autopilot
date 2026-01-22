"""Dataset modules for autonomous driving."""

from autopilot.datasets.cityscapes import (
    CityscapesDataset,
    collate_fn as cityscapes_collate_fn,
    compute_log_bins,
    depth_to_soft_target,
    NUM_CLASSES,
    NUM_STUFF,
    NUM_THINGS,
)
from autopilot.datasets.nuscenes import (
    NuScenesDataset,
    collate_fn as nuscenes_collate_fn,
    CAMERA_NAMES,
    DETECTION_CLASSES,
    NUM_CLASSES as NUSCENES_NUM_CLASSES,
)

# Keep backward compatibility
collate_fn = cityscapes_collate_fn

__all__ = [
    # Cityscapes
    'CityscapesDataset',
    'cityscapes_collate_fn',
    'collate_fn',
    'compute_log_bins',
    'depth_to_soft_target',
    'NUM_CLASSES',
    'NUM_STUFF',
    'NUM_THINGS',
    # nuScenes
    'NuScenesDataset',
    'nuscenes_collate_fn',
    'CAMERA_NAMES',
    'DETECTION_CLASSES',
    'NUSCENES_NUM_CLASSES',
]
