"""Waymo Open Dataset implementation.

Reference:
    - Website: https://waymo.com/open/
    - Paper: https://arxiv.org/abs/1912.04838
"""

import os
import pickle
from typing import Any, Callable, Dict, List, Optional

import numpy as np

from autopilot.datasets.base import BaseDataset
from autopilot.utils.registry import DATASETS


@DATASETS.register_module()
class WaymoDataset(BaseDataset):
    """Waymo Open Dataset for 3D object detection.

    Args:
        data_root: Root directory of Waymo dataset.
        ann_file: Path to annotation pickle file.
        pipeline: List of transforms.
        load_interval: Interval for loading frames (1 = all frames).
        test_mode: Whether in test mode.

    Example:
        >>> dataset = WaymoDataset(
        ...     data_root='data/waymo',
        ...     ann_file='data/waymo/waymo_infos_train.pkl',
        ... )
    """

    CLASSES = ('Vehicle', 'Pedestrian', 'Cyclist')

    CAMERAS = (
        'FRONT', 'FRONT_LEFT', 'FRONT_RIGHT', 'SIDE_LEFT', 'SIDE_RIGHT'
    )

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        pipeline: Optional[List[Callable]] = None,
        load_interval: int = 1,
        test_mode: bool = False,
    ) -> None:
        """Initialize WaymoDataset."""
        self.load_interval = load_interval

        super().__init__(
            data_root=data_root,
            ann_file=ann_file,
            pipeline=pipeline,
            test_mode=test_mode,
        )

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load Waymo annotations from pickle file.

        Returns:
            List of annotation dictionaries.

        Raises:
            FileNotFoundError: If annotation file does not exist.
        """
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f'Annotation file not found: {self.ann_file}')

        with open(self.ann_file, 'rb') as f:
            data = pickle.load(f)

        data_infos = data.get('infos', data)

        # Apply load interval
        if self.load_interval > 1:
            data_infos = data_infos[::self.load_interval]

        return data_infos

    def get_data_info(self, idx: int) -> Dict[str, Any]:
        """Get data info for a specific sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing sample information.
        """
        info = self.data_infos[idx]

        data = {
            'sample_idx': idx,
            'context_name': info.get('context_name', ''),
            'timestamp': info.get('timestamp_micros', 0),
        }

        # LiDAR info
        if 'lidar_path' in info:
            data['lidar_path'] = os.path.join(self.data_root, info['lidar_path'])

        # Camera info
        if 'images' in info:
            img_paths = []
            lidar2img = []

            for cam_name in self.CAMERAS:
                if cam_name in info['images']:
                    cam_info = info['images'][cam_name]
                    img_paths.append(
                        os.path.join(self.data_root, cam_info['path'])
                    )
                    # Get transformation matrices
                    if 'lidar2img' in cam_info:
                        lidar2img.append(np.array(cam_info['lidar2img']))

            data['img_paths'] = img_paths
            if lidar2img:
                data['lidar2img'] = np.stack(lidar2img, axis=0)

        # Ground truth annotations
        if not self.test_mode and 'annos' in info:
            annos = info['annos']
            data['gt_bboxes_3d'] = annos.get('gt_boxes_lidar', np.array([]))
            data['gt_labels_3d'] = annos.get('name', [])
            data['difficulty'] = annos.get('difficulty', [])

        return data


# TODO: Add support for Waymo's native TFRecord format
# TODO: Add support for 2D detection annotations
# TODO: Implement data converter for Waymo format
