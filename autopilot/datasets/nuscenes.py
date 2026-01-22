"""nuScenes dataset for multi-camera 3D perception.

This module provides a dataset class for loading nuScenes data with:
- 6 surround-view camera images
- Camera calibration (intrinsic, extrinsic)
- Ego-motion data (ego pose, timestamps)
- 3D annotations (bounding boxes, velocities)
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset

# nuScenes camera names in standard order
CAMERA_NAMES = [
    "CAM_FRONT",
    "CAM_FRONT_RIGHT",
    "CAM_BACK_RIGHT",
    "CAM_BACK",
    "CAM_BACK_LEFT",
    "CAM_FRONT_LEFT",
]

# nuScenes detection classes
DETECTION_CLASSES = [
    "car",
    "truck",
    "construction_vehicle",
    "bus",
    "trailer",
    "barrier",
    "motorcycle",
    "bicycle",
    "pedestrian",
    "traffic_cone",
    "emergency_vehicle",
    "misc",  # unclassifiable objects (animal, etc.)
]

NUM_CLASSES = len(DETECTION_CLASSES)


class NuScenesDataset(Dataset):
    """nuScenes dataset for multi-camera 3D perception.

    Args:
        root: Path to nuScenes dataset root (contains maps/, samples/, sweeps/, etc.)
        version: Dataset version ('v1.0-trainval', 'v1.0-mini', 'v1.0-test')
        split: Data split ('train', 'val')
        transforms: Optional transform to apply to images
        use_valid_flag: Whether to filter annotations by valid_flag
        max_sweeps: Maximum number of sweeps to load for temporal context
    """

    def __init__(
        self,
        root: str,
        version: str = "v1.0-trainval",
        split: str = "train",
        transforms: Optional[Callable] = None,
        use_valid_flag: bool = True,
        max_sweeps: int = 0,
    ) -> None:
        super().__init__()
        self.root = root
        self.version = version
        self.split = split
        self.transforms = transforms
        self.use_valid_flag = use_valid_flag
        self.max_sweeps = max_sweeps

        # Lazy import nuScenes devkit
        from nuscenes.nuscenes import NuScenes

        self.nusc = NuScenes(version=version, dataroot=root, verbose=False)

        # Build sample list
        self.samples = self._build_sample_list()

        # Build class name to index mapping
        self.class_to_idx = {name: idx for idx, name in enumerate(DETECTION_CLASSES)}

    def _build_sample_list(self) -> List[Dict[str, Any]]:
        """Build list of samples with metadata."""
        # Get scene splits
        if self.version == "v1.0-mini":
            train_scenes = self._get_scenes_by_split("mini_train")
            val_scenes = self._get_scenes_by_split("mini_val")
        else:
            train_scenes = self._get_scenes_by_split("train")
            val_scenes = self._get_scenes_by_split("val")

        available_scenes = set(
            scene["name"] for scene in self.nusc.scene
        )

        if self.split == "train":
            scene_names = set(train_scenes) & available_scenes
        else:
            scene_names = set(val_scenes) & available_scenes

        samples = []
        for scene in self.nusc.scene:
            if scene["name"] not in scene_names:
                continue

            # Iterate through all samples in scene
            sample_token = scene["first_sample_token"]
            while sample_token:
                sample = self.nusc.get("sample", sample_token)
                samples.append(
                    {
                        "token": sample_token,
                        "scene_token": scene["token"],
                        "timestamp": sample["timestamp"],
                    }
                )
                sample_token = sample["next"]

        return samples

    def _get_scenes_by_split(self, split: str) -> List[str]:
        """Get scene names for a given split."""
        from nuscenes.utils import splits

        if split == "train":
            return splits.train
        elif split == "val":
            return splits.val
        elif split == "mini_train":
            return splits.mini_train
        elif split == "mini_val":
            return splits.mini_val
        else:
            raise ValueError(f"Unknown split: {split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample by index.

        Returns:
            Dictionary containing:
                - images: (N, 3, H, W) tensor of camera images
                - intrinsics: (N, 3, 3) camera intrinsic matrices
                - extrinsics: (N, 4, 4) camera extrinsic matrices (cam to ego)
                - ego_pose: (4, 4) ego vehicle pose (ego to global)
                - timestamp: sample timestamp in microseconds
                - gt_boxes: (M, 9) tensor of [x, y, z, w, l, h, yaw, vx, vy]
                - gt_labels: (M,) tensor of class indices
        """
        sample_info = self.samples[idx]
        sample = self.nusc.get("sample", sample_info["token"])

        # Load multi-camera data
        images, intrinsics, extrinsics, cam_timestamps = self._load_cameras(sample)

        # Load ego pose
        ego_pose = self._get_ego_pose(sample)

        # Load 3D annotations
        gt_boxes, gt_labels = self._load_annotations(sample)

        # Build output dict
        data = {
            "images": images,
            "intrinsics": intrinsics,
            "extrinsics": extrinsics,
            "ego_pose": ego_pose,
            "timestamp": sample["timestamp"],
            "cam_timestamps": cam_timestamps,
            "gt_boxes": gt_boxes,
            "gt_labels": gt_labels,
            "sample_token": sample_info["token"],
        }

        # Apply transforms if provided
        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def _load_cameras(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load images and calibration for all cameras.

        Returns:
            images: (N, 3, H, W) float tensor, normalized to [0, 1]
            intrinsics: (N, 3, 3) float tensor
            extrinsics: (N, 4, 4) float tensor (camera to ego)
            timestamps: (N,) int tensor of camera timestamps
        """
        images = []
        intrinsics = []
        extrinsics = []
        timestamps = []

        for cam_name in CAMERA_NAMES:
            cam_data = self.nusc.get("sample_data", sample["data"][cam_name])

            # Load image
            img_path = f"{self.root}/{cam_data['filename']}"
            img = Image.open(img_path).convert("RGB")
            img = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
            img = img.transpose(2, 0, 1)  # (3, H, W)
            images.append(img)

            # Get calibration
            calib = self.nusc.get("calibrated_sensor", cam_data["calibrated_sensor_token"])

            # Intrinsic matrix
            K = np.array(calib["camera_intrinsic"], dtype=np.float32)
            intrinsics.append(K)

            # Extrinsic matrix (camera to ego)
            rotation = Quaternion(calib["rotation"]).rotation_matrix
            translation = np.array(calib["translation"])
            extrinsic = np.eye(4, dtype=np.float32)
            extrinsic[:3, :3] = rotation
            extrinsic[:3, 3] = translation
            extrinsics.append(extrinsic)

            # Timestamp
            timestamps.append(cam_data["timestamp"])

        images = torch.from_numpy(np.stack(images, axis=0))
        intrinsics = torch.from_numpy(np.stack(intrinsics, axis=0))
        extrinsics = torch.from_numpy(np.stack(extrinsics, axis=0))
        timestamps = torch.tensor(timestamps, dtype=torch.int64)

        return images, intrinsics, extrinsics, timestamps

    def _get_ego_pose(self, sample: Dict[str, Any]) -> torch.Tensor:
        """Get ego vehicle pose at sample timestamp.

        Returns:
            ego_pose: (4, 4) float tensor (ego to global transformation)
        """
        # Use lidar sample_data to get ego pose (closest to sample timestamp)
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose_record = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])

        rotation = Quaternion(ego_pose_record["rotation"]).rotation_matrix
        translation = np.array(ego_pose_record["translation"])

        ego_pose = np.eye(4, dtype=np.float32)
        ego_pose[:3, :3] = rotation
        ego_pose[:3, 3] = translation

        return torch.from_numpy(ego_pose)

    def _load_annotations(
        self, sample: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load 3D bounding box annotations.

        Returns:
            gt_boxes: (M, 9) tensor of [x, y, z, w, l, h, yaw, vx, vy]
                      in ego vehicle coordinates
            gt_labels: (M,) tensor of class indices
        """
        boxes = []
        labels = []

        # Get ego pose for coordinate transformation
        lidar_data = self.nusc.get("sample_data", sample["data"]["LIDAR_TOP"])
        ego_pose_record = self.nusc.get("ego_pose", lidar_data["ego_pose_token"])
        ego_rotation = Quaternion(ego_pose_record["rotation"])
        ego_translation = np.array(ego_pose_record["translation"])

        for ann_token in sample["anns"]:
            ann = self.nusc.get("sample_annotation", ann_token)

            # Filter by valid flag if requested
            if self.use_valid_flag:
                if ann.get("visibility_token", "") == "":
                    continue
                visibility = self.nusc.get("visibility", ann["visibility_token"])
                if int(visibility["level"]) < 1:  # Skip if not visible
                    continue

            # Get category
            category = ann["category_name"]
            # Map to detection class
            class_name = self._map_category_to_detection_class(category)
            if class_name is None:
                continue

            class_idx = self.class_to_idx[class_name]

            # Get box parameters in global frame
            center = np.array(ann["translation"])
            size = np.array(ann["size"])  # [w, l, h] in nuScenes
            rotation = Quaternion(ann["rotation"])

            # Transform center to ego frame
            center_ego = ego_rotation.inverse.rotate(center - ego_translation)

            # Get yaw angle in ego frame
            yaw_global = rotation.yaw_pitch_roll[0]
            ego_yaw = ego_rotation.yaw_pitch_roll[0]
            yaw_ego = yaw_global - ego_yaw

            # Get velocity
            if ann["velocity"] is not None:
                velocity = np.array(ann["velocity"][:2])  # Only x, y
                # Transform velocity to ego frame
                velocity_ego = ego_rotation.inverse.rotate(
                    np.array([velocity[0], velocity[1], 0])
                )[:2]
            else:
                velocity_ego = np.array([0.0, 0.0])

            # Box format: [x, y, z, w, l, h, yaw, vx, vy]
            box = np.array(
                [
                    center_ego[0],
                    center_ego[1],
                    center_ego[2],
                    size[0],  # width
                    size[1],  # length
                    size[2],  # height
                    yaw_ego,
                    velocity_ego[0],
                    velocity_ego[1],
                ],
                dtype=np.float32,
            )

            boxes.append(box)
            labels.append(class_idx)

        if len(boxes) > 0:
            gt_boxes = torch.from_numpy(np.stack(boxes, axis=0))
            gt_labels = torch.tensor(labels, dtype=torch.int64)
        else:
            gt_boxes = torch.zeros((0, 9), dtype=torch.float32)
            gt_labels = torch.zeros((0,), dtype=torch.int64)

        return gt_boxes, gt_labels

    def _map_category_to_detection_class(self, category: str) -> Optional[str]:
        """Map nuScenes category to detection class name."""
        # Category mapping (all nuScenes categories)
        mapping = {
            # Vehicles
            "vehicle.car": "car",
            "vehicle.truck": "truck",
            "vehicle.bus.bendy": "bus",
            "vehicle.bus.rigid": "bus",
            "vehicle.construction": "construction_vehicle",
            "vehicle.emergency.ambulance": "emergency_vehicle",
            "vehicle.emergency.police": "emergency_vehicle",
            "vehicle.motorcycle": "motorcycle",
            "vehicle.bicycle": "bicycle",
            "vehicle.trailer": "trailer",
            # Humans
            "human.pedestrian.adult": "pedestrian",
            "human.pedestrian.child": "pedestrian",
            "human.pedestrian.construction_worker": "pedestrian",
            "human.pedestrian.police_officer": "pedestrian",
            "human.pedestrian.wheelchair": "pedestrian",
            "human.pedestrian.stroller": "pedestrian",
            "human.pedestrian.personal_mobility": "pedestrian",
            # Movable objects
            "movable_object.barrier": "barrier",
            "movable_object.trafficcone": "traffic_cone",
            # "movable_object.pushable_pullable": None,  # not used
            # "movable_object.debris": None,  # not used
            # Static objects
            # "static_object.bicycle_rack": None,  # not used
            # Animals
            "animal": "misc",
        }

        for key, value in mapping.items():
            if category.startswith(key):
                return value

        return None


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function for nuScenes dataset.

    Handles variable number of annotations per sample.
    """
    collated = {}

    # Stack tensors that have same shape across batch
    for key in ["images", "intrinsics", "extrinsics", "ego_pose", "cam_timestamps"]:
        collated[key] = torch.stack([sample[key] for sample in batch], dim=0)

    # Keep timestamps as tensor
    collated["timestamp"] = torch.tensor(
        [sample["timestamp"] for sample in batch], dtype=torch.int64
    )

    # Keep variable-length annotations as lists
    collated["gt_boxes"] = [sample["gt_boxes"] for sample in batch]
    collated["gt_labels"] = [sample["gt_labels"] for sample in batch]

    # Keep tokens as list
    collated["sample_token"] = [sample["sample_token"] for sample in batch]

    return collated
