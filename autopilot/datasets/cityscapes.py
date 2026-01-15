"""Cityscapes dataset for panoptic segmentation and depth estimation."""

from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# =============================================================================
# Cityscapes Label Definitions
# =============================================================================

# Format: (name, id, trainId, category, catId, hasInstances, ignoreInEval, color)
CITYSCAPES_LABELS = [
    ("unlabeled", 0, 255, "void", 0, False, True, (0, 0, 0)),
    ("ego vehicle", 1, 255, "void", 0, False, True, (0, 0, 0)),
    ("rectification border", 2, 255, "void", 0, False, True, (0, 0, 0)),
    ("out of roi", 3, 255, "void", 0, False, True, (0, 0, 0)),
    ("static", 4, 255, "void", 0, False, True, (0, 0, 0)),
    ("dynamic", 5, 255, "void", 0, False, True, (111, 74, 0)),
    ("ground", 6, 255, "void", 0, False, True, (81, 0, 81)),
    ("road", 7, 0, "flat", 1, False, False, (128, 64, 128)),
    ("sidewalk", 8, 1, "flat", 1, False, False, (244, 35, 232)),
    ("parking", 9, 255, "flat", 1, False, True, (250, 170, 160)),
    ("rail track", 10, 255, "flat", 1, False, True, (230, 150, 140)),
    ("building", 11, 2, "construction", 2, False, False, (70, 70, 70)),
    ("wall", 12, 3, "construction", 2, False, False, (102, 102, 156)),
    ("fence", 13, 4, "construction", 2, False, False, (190, 153, 153)),
    ("guard rail", 14, 255, "construction", 2, False, True, (180, 165, 180)),
    ("bridge", 15, 255, "construction", 2, False, True, (150, 100, 100)),
    ("tunnel", 16, 255, "construction", 2, False, True, (150, 120, 90)),
    ("pole", 17, 5, "object", 3, False, False, (153, 153, 153)),
    ("polegroup", 18, 255, "object", 3, False, True, (153, 153, 153)),
    ("traffic light", 19, 6, "object", 3, False, False, (250, 170, 30)),
    ("traffic sign", 20, 7, "object", 3, False, False, (220, 220, 0)),
    ("vegetation", 21, 8, "nature", 4, False, False, (107, 142, 35)),
    ("terrain", 22, 9, "nature", 4, False, False, (152, 251, 152)),
    ("sky", 23, 10, "sky", 5, False, False, (70, 130, 180)),
    ("person", 24, 11, "human", 6, True, False, (220, 20, 60)),
    ("rider", 25, 12, "human", 6, True, False, (255, 0, 0)),
    ("car", 26, 13, "vehicle", 7, True, False, (0, 0, 142)),
    ("truck", 27, 14, "vehicle", 7, True, False, (0, 0, 70)),
    ("bus", 28, 15, "vehicle", 7, True, False, (0, 60, 100)),
    ("caravan", 29, 255, "vehicle", 7, True, True, (0, 0, 90)),
    ("trailer", 30, 255, "vehicle", 7, True, True, (0, 0, 110)),
    ("train", 31, 16, "vehicle", 7, True, False, (0, 80, 100)),
    ("motorcycle", 32, 17, "vehicle", 7, True, False, (0, 0, 230)),
    ("bicycle", 33, 18, "vehicle", 7, True, False, (119, 11, 32)),
]

# Create mapping from original ID to train ID
ID_TO_TRAINID = {label[1]: label[2] for label in CITYSCAPES_LABELS}

# Things classes (hasInstances=True and ignoreInEval=False)
# trainId: 11 (person), 12 (rider), 13 (car), 14 (truck), 15 (bus), 16 (train), 17 (motorcycle), 18 (bicycle)
THING_TRAIN_IDS = [11, 12, 13, 14, 15, 16, 17, 18]
NUM_THINGS = 8

# Stuff classes (hasInstances=False and ignoreInEval=False)
# trainId: 0-10
STUFF_TRAIN_IDS = list(range(11))
NUM_STUFF = 11

# Total classes for training (things first, then stuff)
# We remap: things -> 0-7, stuff -> 8-18
NUM_CLASSES = NUM_THINGS + NUM_STUFF  # 19


def create_trainid_to_panoptic_id():
    """Create mapping from cityscapes trainId to panoptic training ID.

    Panoptic ID layout: things (0-7), stuff (8-18)
    """
    mapping = {}
    # Things: trainId 11-18 -> panoptic 0-7
    for i, tid in enumerate(THING_TRAIN_IDS):
        mapping[tid] = i
    # Stuff: trainId 0-10 -> panoptic 8-18
    for i, tid in enumerate(STUFF_TRAIN_IDS):
        mapping[tid] = NUM_THINGS + i
    # Ignore
    mapping[255] = 255
    return mapping


TRAINID_TO_PANOPTIC = create_trainid_to_panoptic_id()

# =============================================================================
# Depth Estimation Constants and Functions
# =============================================================================

# Cityscapes camera parameters
# https://www.cityscapes-dataset.com/file-handling/
CITYSCAPES_BASELINE = 0.22  # meters
CITYSCAPES_FOCAL_LENGTH = 2262.52  # pixels


def compute_log_bins(
    min_depth: float = 1.0,
    max_depth: float = 150.0,
    num_bins: int = 64,
) -> np.ndarray:
    """Compute log-spaced bin edges for depth estimation.

    Args:
        min_depth: Minimum depth in meters.
        max_depth: Maximum depth in meters.
        num_bins: Number of bins.

    Returns:
        bin_edges: Array of shape (num_bins + 1,) with bin edges.
    """
    return np.exp(np.linspace(np.log(min_depth), np.log(max_depth), num_bins + 1))


def depth_to_soft_target(
    depth: np.ndarray,
    bin_edges: np.ndarray,
    sigma_ratio: float = 0.1,
) -> np.ndarray:
    """Convert depth to soft bin targets using Gaussian distribution.

    Args:
        depth: Depth map (H, W) in meters.
        bin_edges: Array of shape (num_bins + 1,) with bin edges.
        sigma_ratio: Sigma as ratio of bin width (default 0.1).

    Returns:
        soft_target: Soft target probabilities (num_bins, H, W).
    """
    num_bins = len(bin_edges) - 1
    H, W = depth.shape

    # Compute bin centers (in log space, then convert back)
    log_edges = np.log(bin_edges)
    log_centers = (log_edges[:-1] + log_edges[1:]) / 2
    log_widths = log_edges[1:] - log_edges[:-1]

    # Convert depth to log space for Gaussian computation
    log_depth = np.log(np.clip(depth, bin_edges[0], bin_edges[-1]))

    # Compute soft targets
    soft_target = np.zeros((num_bins, H, W), dtype=np.float32)
    for i in range(num_bins):
        # Sigma proportional to bin width in log space
        sigma = log_widths[i] * sigma_ratio
        diff = log_depth - log_centers[i]
        soft_target[i] = np.exp(-0.5 * (diff / sigma) ** 2)

    # Normalize to sum to 1 along bin dimension
    soft_target_sum = soft_target.sum(axis=0, keepdims=True)
    return soft_target / (soft_target_sum + 1e-8)


def disparity_to_depth(
    disparity: np.ndarray,
    baseline: float = CITYSCAPES_BASELINE,
    focal_length: float = CITYSCAPES_FOCAL_LENGTH,
) -> np.ndarray:
    """Convert Cityscapes disparity to depth.

    Cityscapes disparity format:
        - Stored as 16-bit PNG
        - disparity = (pixel_value - 1) / 256
        - pixel_value = 0 means invalid

    Args:
        disparity: Raw disparity values from PNG (uint16).
        baseline: Camera baseline in meters.
        focal_length: Focal length in pixels.

    Returns:
        depth: Depth in meters. Invalid pixels have value 0.
    """
    # Convert from stored format
    valid_mask = disparity > 0
    disparity_float = (disparity.astype(np.float32) - 1) / 256.0

    # Avoid division by zero
    disparity_float = np.maximum(disparity_float, 1e-6)

    # depth = baseline * focal_length / disparity
    depth = np.zeros_like(disparity, dtype=np.float32)
    depth[valid_mask] = (baseline * focal_length) / disparity_float[valid_mask]

    return depth


# =============================================================================
# Dataset
# =============================================================================

class CityscapesDataset(Dataset):
    """Cityscapes dataset for panoptic segmentation and depth estimation.

    Each sample contains both panoptic (semantic, center, offset) and
    depth (soft bin targets) annotations.

    Args:
        root: Path to cityscapes dataset root.
        split: Dataset split ('train', 'val', 'test').
        transforms: Optional transforms to apply.
        gaussian_sigma: Sigma for center heatmap gaussian.
        min_depth: Minimum depth for binning (meters).
        max_depth: Maximum depth for binning (meters).
        num_bins: Number of depth bins.
        sigma_ratio: Sigma ratio for soft targets.
    """

    def __init__(
        self,
        root: str,
        split: str = "train",
        transforms: Optional[Callable] = None,
        gaussian_sigma: float = 8.0,
        min_depth: float = 1.0,
        max_depth: float = 150.0,
        num_bins: int = 64,
        sigma_ratio: float = 0.1,
    ):
        super().__init__()

        self.root = Path(root)
        self.split = split
        self.transforms = transforms
        self.gaussian_sigma = gaussian_sigma
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.num_bins = num_bins
        self.sigma_ratio = sigma_ratio

        # Precompute bin edges
        self.bin_edges = compute_log_bins(min_depth, max_depth, num_bins)

        # Find all samples
        self.samples = self._find_samples()

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in {self.root} for split {split}")

    def _find_samples(self) -> List[Dict[str, Path]]:
        """Find samples with both panoptic and depth annotations."""
        img_dir = self.root / "leftImg8bit" / self.split
        gt_dir = self.root / "gtFine" / self.split
        disp_dir = self.root / "disparity" / self.split

        samples = []
        for city_dir in sorted(img_dir.iterdir()):
            if not city_dir.is_dir():
                continue

            city = city_dir.name
            for img_path in sorted(city_dir.glob("*_leftImg8bit.png")):
                # Extract base name: {city}_{seq}_{frame}
                base = img_path.stem.replace("_leftImg8bit", "")

                # Construct annotation paths
                label_path = gt_dir / city / f"{base}_gtFine_labelIds.png"
                instance_path = gt_dir / city / f"{base}_gtFine_instanceIds.png"
                disp_path = disp_dir / city / f"{base}_disparity.png"

                # Require all annotations
                if label_path.exists() and instance_path.exists() and disp_path.exists():
                    samples.append({
                        "image": img_path,
                        "label": label_path,
                        "instance": instance_path,
                        "disparity": disp_path,
                    })

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a sample with panoptic and depth targets.

        Returns:
            Dict with:
                - image: (3, H, W) float tensor, normalized
                - semantic_target: (H, W) long tensor, panoptic class IDs
                - center_target: (1, H, W) float tensor, class-agnostic center heatmap
                - offset_target: (2, H, W) float tensor, offset to center
                - offset_weight: (1, H, W) float tensor, weight mask for offset
                - depth_target: (num_bins, H, W) float tensor, soft bin targets
                - depth_weight: (1, H, W) float tensor, valid mask for depth
        """
        paths = self.samples[idx]

        # Load image
        image = Image.open(paths["image"]).convert("RGB")
        image = np.array(image, dtype=np.float32) / 255.0

        # =================================================================
        # Panoptic targets
        # =================================================================
        label_ids = np.array(Image.open(paths["label"]), dtype=np.int32)
        instance_ids = np.array(Image.open(paths["instance"]), dtype=np.int32)

        # Convert to panoptic IDs
        semantic_target = self._convert_to_panoptic_ids(label_ids)

        # Generate instance targets
        center_target, offset_target, offset_weight = self._generate_instance_targets(
            instance_ids, label_ids, semantic_target.shape
        )

        # =================================================================
        # Depth targets
        # =================================================================
        disparity = np.array(Image.open(paths["disparity"]), dtype=np.uint16)
        depth_raw = disparity_to_depth(disparity)

        # Create valid mask (valid depth within range)
        valid_mask = (
            (depth_raw > 0) &
            (depth_raw >= self.min_depth) &
            (depth_raw <= self.max_depth)
        )
        depth_weight = valid_mask.astype(np.float32)

        # Generate soft bin targets
        depth_target = depth_to_soft_target(
            np.clip(depth_raw, self.min_depth, self.max_depth),
            self.bin_edges,
            self.sigma_ratio,
        )

        # Zero out invalid pixels in target
        depth_target[:, ~valid_mask] = 0

        # =================================================================
        # Apply transforms
        # =================================================================
        if self.transforms is not None:
            transformed = self.transforms(
                image=image,
                semantic_target=semantic_target,
                center_target=center_target,
                offset_target=offset_target,
                offset_weight=offset_weight,
                depth_target=depth_target,
                depth_weight=depth_weight,
            )
            image = transformed["image"]
            semantic_target = transformed["semantic_target"]
            center_target = transformed["center_target"]
            offset_target = transformed["offset_target"]
            offset_weight = transformed["offset_weight"]
            depth_target = transformed["depth_target"]
            depth_weight = transformed["depth_weight"]

        # Convert to tensors
        return {
            "image": torch.from_numpy(image).permute(2, 0, 1),
            "semantic_target": torch.from_numpy(semantic_target).long(),
            "center_target": torch.from_numpy(center_target).float(),
            "offset_target": torch.from_numpy(offset_target).float(),
            "offset_weight": torch.from_numpy(offset_weight).float(),
            "depth_target": torch.from_numpy(depth_target).float(),
            "depth_weight": torch.from_numpy(depth_weight[None]).float(),
        }

    def _convert_to_panoptic_ids(self, label_ids: np.ndarray) -> np.ndarray:
        """Convert cityscapes label IDs to panoptic training IDs."""
        # First convert to train IDs
        train_ids = np.vectorize(lambda x: ID_TO_TRAINID.get(x, 255))(label_ids)

        # Then convert to panoptic IDs
        panoptic_ids = np.vectorize(lambda x: TRAINID_TO_PANOPTIC.get(x, 255))(train_ids)

        return panoptic_ids.astype(np.int64)

    def _generate_instance_targets(
        self,
        instance_ids: np.ndarray,
        label_ids: np.ndarray,
        shape: Tuple[int, int],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate center heatmap, offset, and weight targets.

        Args:
            instance_ids: Instance ID map from cityscapes.
            label_ids: Label ID map from cityscapes.
            shape: Output shape (H, W).

        Returns:
            center_target: (1, H, W) class-agnostic center heatmap
            offset_target: (2, H, W)
            offset_weight: (1, H, W)
        """
        H, W = shape

        center_target = np.zeros((1, H, W), dtype=np.float32)
        offset_target = np.zeros((2, H, W), dtype=np.float32)
        offset_weight = np.zeros((1, H, W), dtype=np.float32)

        # Create coordinate grids
        y_coord = np.arange(H).reshape(-1, 1).repeat(W, axis=1)
        x_coord = np.arange(W).reshape(1, -1).repeat(H, axis=0)

        # Process each unique instance
        # In cityscapes, instance IDs >= 1000 indicate instance of thing class
        # instance_id = class_id * 1000 + instance_number
        unique_instances = np.unique(instance_ids)

        for inst_id in unique_instances:
            if inst_id < 1000:
                continue  # Not a thing instance

            # Get class ID
            class_id = inst_id // 1000
            train_id = ID_TO_TRAINID.get(class_id, 255)

            if train_id not in THING_TRAIN_IDS:
                continue

            # Get instance mask
            mask = instance_ids == inst_id
            if mask.sum() == 0:
                continue

            # Compute center
            y_indices, x_indices = np.where(mask)
            center_y = y_indices.mean()
            center_x = x_indices.mean()

            # Generate gaussian heatmap for center (class-agnostic)
            center_target[0] = np.maximum(
                center_target[0],
                self._generate_gaussian(H, W, center_y, center_x),
            )

            # Generate offset (offset from pixel to center)
            offset_target[0, mask] = center_y - y_coord[mask]
            offset_target[1, mask] = center_x - x_coord[mask]

            # Set offset weight
            offset_weight[0, mask] = 1.0

        return center_target, offset_target, offset_weight

    def _generate_gaussian(
        self,
        height: int,
        width: int,
        center_y: float,
        center_x: float,
    ) -> np.ndarray:
        """Generate 2D gaussian heatmap."""
        y = np.arange(height).reshape(-1, 1)
        x = np.arange(width).reshape(1, -1)

        gaussian = np.exp(
            -((y - center_y) ** 2 + (x - center_x) ** 2) / (2 * self.gaussian_sigma ** 2)
        )

        return gaussian.astype(np.float32)

    def get_bin_centers(self) -> np.ndarray:
        """Get bin centers for depth decoding."""
        log_edges = np.log(self.bin_edges)
        log_centers = (log_edges[:-1] + log_edges[1:]) / 2
        return np.exp(log_centers)


# =============================================================================
# Collate Function
# =============================================================================

def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Collate function for DataLoader.

    Args:
        batch: List of samples from dataset.

    Returns:
        Batched tensors.
    """
    return {
        "image": torch.stack([s["image"] for s in batch]),
        "semantic_target": torch.stack([s["semantic_target"] for s in batch]),
        "center_target": torch.stack([s["center_target"] for s in batch]),
        "offset_target": torch.stack([s["offset_target"] for s in batch]),
        "offset_weight": torch.stack([s["offset_weight"] for s in batch]),
        "depth_target": torch.stack([s["depth_target"] for s in batch]),
        "depth_weight": torch.stack([s["depth_weight"] for s in batch]),
    }
