"""nuScenes dataset implementation.

Reference:
    - Website: https://www.nuscenes.org/
    - Paper: https://arxiv.org/abs/1903.11027
"""

import os
import pickle
from typing import Any, Callable, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from autopilot.utils.registry import DATASETS


@DATASETS.register_module()
class NuScenesDataset(Dataset):
    """nuScenes dataset for 3D object detection.

    Args:
        data_root: Root directory of nuScenes dataset.
        ann_file: Path to annotation pickle file.
        img_size: Image size (H, W) after resize.
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max].
        bev_size: BEV feature map size (H, W).
        classes: List of class names.
        min_visibility: Minimum visibility level (0-4, 3=60%+).
        use_valid_flag: Whether to filter by valid flag.
        test_mode: Whether in test mode.
        img_norm_cfg: Image normalization config.

    Example:
        >>> dataset = NuScenesDataset(
        ...     data_root='data/nuscenes',
        ...     ann_file='data/nuscenes/nuscenes_infos_train.pkl',
        ...     min_visibility=3,
        ... )
    """

    CLASSES = (
        'car', 'truck', 'trailer', 'bus', 'construction_vehicle',
        'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone', 'barrier',
    )

    CAMERA_NAMES = (
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    )

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        img_size: Tuple[int, int] = (480, 640),  # (H, W)
        point_cloud_range: List[float] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
        bev_size: Tuple[int, int] = (20, 80),  # (H, W)
        classes: Optional[List[str]] = None,
        min_visibility: int = 3,
        use_valid_flag: bool = True,
        test_mode: bool = False,
        img_norm_cfg: Optional[Dict] = None,
    ) -> None:
        """Initialize NuScenesDataset."""
        self.data_root = data_root
        self.ann_file = ann_file
        self.img_size = img_size
        self.point_cloud_range = point_cloud_range
        self.bev_size = bev_size
        self.classes = classes or list(self.CLASSES)
        self.min_visibility = min_visibility
        self.use_valid_flag = use_valid_flag
        self.test_mode = test_mode

        # Image normalization
        self.img_norm_cfg = img_norm_cfg or {
            'mean': [123.675, 116.28, 103.53],
            'std': [58.395, 57.12, 57.375],
        }

        # Class name to index mapping
        self.class_to_idx = {name: i for i, name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        # Load annotations
        self.data_infos = self.load_annotations()

    def load_annotations(self) -> List[Dict[str, Any]]:
        """Load nuScenes annotations from pickle file.

        Returns:
            List of annotation dictionaries.
        """
        if not os.path.exists(self.ann_file):
            raise FileNotFoundError(f'Annotation file not found: {self.ann_file}')

        with open(self.ann_file, 'rb') as f:
            data = pickle.load(f)

        # Support both list and dict formats
        if isinstance(data, list):
            data_infos = data
        else:
            data_infos = data.get('infos', data)

        # Filter by valid flag if needed
        if self.use_valid_flag and not self.test_mode:
            data_infos = [
                info for info in data_infos
                if info.get('valid_flag', True)
            ]

        return data_infos

    def __len__(self) -> int:
        """Return dataset length."""
        return len(self.data_infos)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary containing:
                - images: (num_cam, 3, H, W) camera images
                - lidar2img: (num_cam, 4, 4) transformation matrices
                - gt_boxes: (N, 7) ground truth boxes [x, y, z, l, w, h, yaw]
                - gt_labels: (N,) ground truth labels
                - gt_heatmap: (num_classes, bev_h, bev_w) heatmap for training
                - gt_reg: (N, 8) regression targets
        """
        info = self.data_infos[idx]

        # Load images
        images, lidar2img = self._load_images(info)

        # Load annotations
        if not self.test_mode:
            gt_boxes, gt_labels, gt_visibility = self._load_annotations(info)

            # Filter by visibility
            if len(gt_boxes) > 0:
                vis_mask = gt_visibility >= self.min_visibility
                gt_boxes = gt_boxes[vis_mask]
                gt_labels = gt_labels[vis_mask]

            # Generate heatmap and regression targets (also filters boxes by range)
            gt_heatmap, gt_reg_targets, gt_indices, valid_mask = self._generate_targets(
                gt_boxes, gt_labels
            )
            # Apply the same range filtering to gt_boxes and gt_labels
            gt_boxes = gt_boxes[valid_mask]
            gt_labels = gt_labels[valid_mask]
        else:
            gt_boxes = np.zeros((0, 7), dtype=np.float32)
            gt_labels = np.zeros((0,), dtype=np.int64)
            gt_heatmap = np.zeros(
                (self.num_classes, self.bev_size[0], self.bev_size[1]),
                dtype=np.float32
            )
            gt_reg_targets = np.zeros((0, 8), dtype=np.float32)
            gt_indices = np.zeros((0,), dtype=np.int64)

        return {
            'images': torch.from_numpy(images),
            'lidar2img': torch.from_numpy(lidar2img),
            'gt_boxes': torch.from_numpy(gt_boxes),
            'gt_labels': torch.from_numpy(gt_labels),
            'gt_heatmap': torch.from_numpy(gt_heatmap),
            'gt_reg': torch.from_numpy(gt_reg_targets),
            'gt_indices': torch.from_numpy(gt_indices),
            'sample_idx': idx,
            'token': info.get('token', ''),
            'scene_name': info.get('scene_name', f'scene_{idx:04d}'),
            'timestamp': info.get('timestamp', 0),
        }

    def _load_images(
        self, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Load and preprocess camera images.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of (images, lidar2img) arrays.
        """
        images = []
        lidar2img_list = []

        cams = info.get('cams', {})

        for cam_name in self.CAMERA_NAMES:
            if cam_name in cams:
                cam_info = cams[cam_name]
                # Support both 'data_path' and 'filename' keys
                img_filename = cam_info.get('data_path', cam_info.get('filename', ''))
                img_path = os.path.join(self.data_root, img_filename)

                # Load image
                img = cv2.imread(img_path)
                if img is None:
                    # Create dummy image if file not found
                    img = np.zeros((self.img_size[0], self.img_size[1], 3), dtype=np.uint8)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize
                orig_h, orig_w = img.shape[:2]
                img = cv2.resize(img, (self.img_size[1], self.img_size[0]))

                # Compute scale factors for lidar2img adjustment
                scale_w = self.img_size[1] / orig_w
                scale_h = self.img_size[0] / orig_h

                # Normalize
                img = img.astype(np.float32)
                mean = np.array(self.img_norm_cfg['mean'])
                std = np.array(self.img_norm_cfg['std'])
                img = (img - mean) / std

                # HWC -> CHW
                img = img.transpose(2, 0, 1)
                images.append(img)

                # Get lidar2img transformation
                # Support both precomputed 'lidar2img' and separate 'lidar2cam' + 'intrinsics'
                if 'lidar2img' in cam_info:
                    lidar2img = np.array(cam_info['lidar2img'])
                    # Adjust for resize
                    lidar2img[0, :] *= scale_w
                    lidar2img[1, :] *= scale_h
                    lidar2img_list.append(lidar2img)
                else:
                    # Compute lidar2img from lidar2cam and intrinsics
                    lidar2cam = np.array(cam_info.get('lidar2cam', cam_info.get('extrinsics', np.eye(4)[:3, :])))
                    if lidar2cam.shape == (3, 4):
                        lidar2cam_4x4 = np.eye(4)
                        lidar2cam_4x4[:3, :] = lidar2cam
                        lidar2cam = lidar2cam_4x4
                    intrinsic = np.array(cam_info.get('cam_intrinsic', cam_info.get('intrinsics', np.eye(3))))

                    # Adjust intrinsic for resize
                    intrinsic_scaled = intrinsic.copy()
                    intrinsic_scaled[0, :] *= scale_w
                    intrinsic_scaled[1, :] *= scale_h

                    viewpad = np.eye(4)
                    viewpad[:3, :3] = intrinsic_scaled
                    lidar2img_list.append(viewpad @ lidar2cam)
            else:
                # Create dummy for missing camera
                img = np.zeros((3, self.img_size[0], self.img_size[1]), dtype=np.float32)
                images.append(img)
                lidar2img_list.append(np.eye(4))

        images = np.stack(images, axis=0).astype(np.float32)
        lidar2img = np.stack(lidar2img_list, axis=0).astype(np.float32)

        return images, lidar2img

    def _load_annotations(
        self, info: Dict[str, Any]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Load ground truth annotations.

        Args:
            info: Sample info dictionary.

        Returns:
            Tuple of (gt_boxes, gt_labels, gt_visibility).
            gt_boxes: (N, 7) [x, y, z, l, w, h, yaw]
            gt_labels: (N,) class indices
            gt_visibility: (N,) visibility levels
        """
        gt_boxes = info.get('gt_boxes', np.array([]))
        gt_names = info.get('gt_names', [])
        gt_visibility = info.get('visibility', info.get('num_lidar_pts', np.array([])))

        if len(gt_boxes) == 0:
            return (
                np.zeros((0, 7), dtype=np.float32),
                np.zeros((0,), dtype=np.int64),
                np.zeros((0,), dtype=np.int32),
            )

        # Convert class names to indices
        gt_labels = []
        valid_mask = []
        for name in gt_names:
            if name in self.class_to_idx:
                gt_labels.append(self.class_to_idx[name])
                valid_mask.append(True)
            else:
                valid_mask.append(False)

        valid_mask = np.array(valid_mask)
        gt_boxes = gt_boxes[valid_mask]
        gt_labels = np.array(gt_labels, dtype=np.int64)

        # Handle visibility
        if isinstance(gt_visibility, np.ndarray) and len(gt_visibility) > 0:
            gt_visibility = gt_visibility[valid_mask]
            # Convert lidar points to visibility level (heuristic)
            if gt_visibility.max() > 4:  # It's num_lidar_pts
                # Map lidar points to visibility: more points = higher visibility
                vis_level = np.clip(gt_visibility // 10, 0, 4).astype(np.int32)
                gt_visibility = vis_level
        else:
            gt_visibility = np.full(len(gt_boxes), 4, dtype=np.int32)

        return gt_boxes.astype(np.float32), gt_labels, gt_visibility

    def _generate_targets(
        self,
        gt_boxes: np.ndarray,
        gt_labels: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Generate training targets (heatmap and regression).

        Args:
            gt_boxes: (N, 7) [x, y, z, l, w, h, yaw]
            gt_labels: (N,) class indices

        Returns:
            Tuple of (heatmap, reg_targets, indices, valid_mask).
        """
        bev_h, bev_w = self.bev_size
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range

        # Initialize heatmap
        heatmap = np.zeros((self.num_classes, bev_h, bev_w), dtype=np.float32)

        # Regression targets per object
        reg_targets = []
        indices = []
        valid_indices = []  # Track which boxes are valid

        for i, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
            x, y, z, l, w, h, yaw = box

            # Check if box is within range
            if not (x_min <= x <= x_max and y_min <= y <= y_max):
                continue

            valid_indices.append(i)

            # Convert to BEV grid coordinates
            grid_x = (x - x_min) / (x_max - x_min) * bev_w
            grid_y = (y - y_min) / (y_max - y_min) * bev_h

            grid_x_int = int(np.clip(grid_x, 0, bev_w - 1))
            grid_y_int = int(np.clip(grid_y, 0, bev_h - 1))

            # Compute gaussian radius based on box size
            # Smaller radius for smaller objects
            radius = max(1, int(min(l, w) / (x_max - x_min) * bev_w / 2))

            # Draw gaussian on heatmap
            self._draw_gaussian(heatmap[label], (grid_x_int, grid_y_int), radius)

            # Regression targets
            # [offset_x, offset_y, z, log(l), log(w), log(h), sin(yaw), cos(yaw)]
            offset_x = grid_x - grid_x_int
            offset_y = grid_y - grid_y_int
            reg = np.array([
                offset_x, offset_y, z,
                np.log(l + 1e-6), np.log(w + 1e-6), np.log(h + 1e-6),
                np.sin(yaw), np.cos(yaw)
            ], dtype=np.float32)
            reg_targets.append(reg)

            # Store index (y * W + x)
            indices.append(grid_y_int * bev_w + grid_x_int)

        # Create valid mask
        valid_mask = np.zeros(len(gt_boxes), dtype=bool)
        if len(valid_indices) > 0:
            valid_mask[valid_indices] = True

        if len(reg_targets) > 0:
            reg_targets = np.stack(reg_targets, axis=0)
            indices = np.array(indices, dtype=np.int64)
        else:
            reg_targets = np.zeros((0, 8), dtype=np.float32)
            indices = np.zeros((0,), dtype=np.int64)

        return heatmap, reg_targets, indices, valid_mask

    def _draw_gaussian(
        self,
        heatmap: np.ndarray,
        center: Tuple[int, int],
        radius: int,
        k: float = 1.0,
    ) -> None:
        """Draw gaussian on heatmap.

        Args:
            heatmap: (H, W) heatmap to draw on.
            center: (x, y) center coordinate.
            radius: Gaussian radius.
            k: Gaussian scale factor.
        """
        diameter = 2 * radius + 1
        gaussian = self._gaussian2d((diameter, diameter), sigma=diameter / 6)

        x, y = center
        h, w = heatmap.shape

        left = min(x, radius)
        right = min(w - x, radius + 1)
        top = min(y, radius)
        bottom = min(h - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]

        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)

    @staticmethod
    def _gaussian2d(shape: Tuple[int, int], sigma: float = 1.0) -> np.ndarray:
        """Generate 2D gaussian kernel.

        Args:
            shape: (height, width) of the kernel.
            sigma: Standard deviation.

        Returns:
            2D gaussian kernel.
        """
        m, n = [(ss - 1.) / 2. for ss in shape]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        return h

    def get_class_names(self) -> List[str]:
        """Return class names."""
        return self.classes


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for NuScenes dataset.

    Handles variable number of ground truth boxes.

    Args:
        batch: List of sample dictionaries.

    Returns:
        Batched dictionary.
    """
    result = {}

    # Stack fixed-size tensors
    for key in ['images', 'lidar2img', 'gt_heatmap']:
        result[key] = torch.stack([sample[key] for sample in batch], dim=0)

    # Pad variable-size tensors
    max_boxes = max(len(sample['gt_boxes']) for sample in batch)

    gt_boxes = torch.zeros(len(batch), max_boxes, 7)
    gt_labels = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    gt_reg = torch.zeros(len(batch), max_boxes, 8)
    gt_indices = torch.zeros(len(batch), max_boxes, dtype=torch.long)
    num_boxes = torch.zeros(len(batch), dtype=torch.long)

    for i, sample in enumerate(batch):
        n = len(sample['gt_boxes'])
        num_boxes[i] = n
        if n > 0:
            gt_boxes[i, :n] = sample['gt_boxes']
            gt_labels[i, :n] = sample['gt_labels']
            gt_reg[i, :n] = sample['gt_reg']
            gt_indices[i, :n] = sample['gt_indices']

    result['gt_boxes'] = gt_boxes
    result['gt_labels'] = gt_labels
    result['gt_reg'] = gt_reg
    result['gt_indices'] = gt_indices
    result['num_boxes'] = num_boxes

    # Metadata
    result['sample_idx'] = [sample['sample_idx'] for sample in batch]
    result['token'] = [sample['token'] for sample in batch]
    result['scene_name'] = [sample.get('scene_name', f'scene_{i:04d}') for i, sample in enumerate(batch)]
    result['timestamp'] = [sample.get('timestamp', 0) for sample in batch]

    return result
