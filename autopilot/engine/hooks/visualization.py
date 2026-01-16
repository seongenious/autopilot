"""Visualization hook for panoptic segmentation and depth estimation."""

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from autopilot.engine.hooks.base import Hook
from autopilot.utils.registry import HOOKS

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch.utils.data import Dataset


# Cityscapes color palette (trainId order: things 0-7, stuff 8-18)
PANOPTIC_PALETTE = np.array([
    # Things (0-7)
    [220, 20, 60],    # 0: person
    [255, 0, 0],      # 1: rider
    [0, 0, 142],      # 2: car
    [0, 0, 70],       # 3: truck
    [0, 60, 100],     # 4: bus
    [0, 80, 100],     # 5: train
    [0, 0, 230],      # 6: motorcycle
    [119, 11, 32],    # 7: bicycle
    # Stuff (8-18)
    [128, 64, 128],   # 8: road
    [244, 35, 232],   # 9: sidewalk
    [70, 70, 70],     # 10: building
    [102, 102, 156],  # 11: wall
    [190, 153, 153],  # 12: fence
    [153, 153, 153],  # 13: pole
    [250, 170, 30],   # 14: traffic light
    [220, 220, 0],    # 15: traffic sign
    [107, 142, 35],   # 16: vegetation
    [152, 251, 152],  # 17: terrain
    [70, 130, 180],   # 18: sky
], dtype=np.uint8)


# Turbo colormap for depth visualization (close=red, far=blue)
TURBO_COLORMAP = np.array([
    [48, 18, 59], [50, 21, 67], [51, 24, 74], [52, 27, 81],
    [53, 30, 88], [54, 33, 95], [55, 36, 102], [56, 39, 109],
    [57, 42, 115], [58, 45, 121], [59, 47, 128], [60, 50, 134],
    [61, 53, 139], [62, 56, 145], [63, 59, 151], [63, 62, 156],
    [64, 64, 162], [65, 67, 167], [65, 70, 172], [66, 73, 177],
    [66, 75, 181], [67, 78, 186], [68, 81, 191], [68, 84, 195],
    [68, 86, 199], [69, 89, 203], [69, 92, 207], [69, 94, 211],
    [70, 97, 214], [70, 100, 218], [70, 102, 221], [70, 105, 224],
    [70, 107, 227], [71, 110, 230], [71, 113, 233], [71, 115, 235],
    [71, 118, 238], [71, 120, 240], [71, 123, 242], [70, 125, 244],
    [70, 128, 246], [70, 130, 248], [70, 133, 250], [70, 135, 251],
    [69, 138, 252], [69, 140, 253], [68, 143, 254], [67, 145, 254],
    [66, 148, 255], [65, 150, 255], [64, 153, 255], [62, 155, 254],
    [61, 158, 254], [59, 160, 253], [58, 163, 252], [56, 165, 251],
    [55, 168, 250], [53, 171, 248], [51, 173, 247], [49, 175, 245],
    [47, 178, 244], [46, 180, 242], [44, 183, 240], [42, 185, 238],
    [40, 188, 235], [39, 190, 233], [37, 192, 231], [35, 195, 228],
    [34, 197, 226], [32, 199, 223], [31, 201, 221], [30, 203, 218],
    [28, 205, 216], [27, 208, 213], [26, 210, 210], [26, 212, 208],
    [25, 213, 205], [24, 215, 202], [24, 217, 200], [24, 219, 197],
    [24, 221, 194], [24, 222, 192], [24, 224, 189], [25, 226, 187],
    [25, 227, 185], [26, 228, 182], [28, 230, 180], [29, 231, 178],
    [31, 233, 175], [32, 234, 172], [34, 235, 170], [37, 236, 167],
    [39, 238, 164], [42, 239, 161], [44, 240, 158], [47, 241, 155],
    [50, 242, 152], [53, 243, 148], [56, 244, 145], [60, 245, 142],
    [63, 246, 138], [67, 247, 135], [70, 248, 132], [74, 248, 128],
    [78, 249, 125], [82, 250, 122], [85, 250, 118], [89, 251, 115],
    [93, 252, 111], [97, 252, 108], [101, 253, 105], [105, 253, 102],
    [109, 254, 98], [113, 254, 95], [117, 254, 92], [121, 254, 89],
    [125, 255, 86], [128, 255, 83], [132, 255, 81], [136, 255, 78],
    [139, 255, 75], [143, 255, 73], [146, 255, 71], [150, 254, 68],
    [153, 254, 66], [156, 254, 64], [159, 253, 63], [161, 253, 61],
    [164, 252, 60], [167, 252, 58], [169, 251, 57], [172, 251, 56],
    [175, 250, 55], [177, 249, 54], [180, 248, 54], [183, 247, 53],
    [185, 246, 53], [188, 245, 52], [190, 244, 52], [193, 243, 52],
    [195, 241, 52], [198, 240, 52], [200, 239, 52], [203, 237, 52],
    [205, 236, 52], [208, 234, 52], [210, 233, 53], [212, 231, 53],
    [215, 229, 53], [217, 228, 54], [219, 226, 54], [221, 224, 55],
    [223, 223, 55], [225, 221, 55], [227, 219, 56], [229, 217, 56],
    [231, 215, 57], [233, 213, 57], [235, 211, 57], [236, 209, 58],
    [238, 207, 58], [239, 205, 58], [241, 203, 58], [242, 201, 58],
    [244, 199, 58], [245, 197, 58], [246, 195, 58], [247, 193, 58],
    [248, 190, 57], [249, 188, 57], [250, 186, 57], [251, 184, 56],
    [251, 182, 55], [252, 179, 54], [252, 177, 54], [253, 174, 53],
    [253, 172, 52], [254, 169, 51], [254, 167, 50], [254, 164, 49],
    [254, 161, 48], [254, 158, 47], [254, 155, 45], [254, 153, 44],
    [254, 150, 43], [254, 147, 42], [254, 144, 41], [253, 141, 39],
    [253, 138, 38], [252, 135, 37], [252, 132, 35], [251, 129, 34],
    [251, 126, 33], [250, 123, 31], [249, 120, 30], [249, 117, 29],
    [248, 114, 28], [247, 111, 26], [246, 108, 25], [245, 105, 24],
    [244, 102, 23], [243, 99, 21], [242, 96, 20], [241, 93, 19],
    [240, 91, 18], [239, 88, 17], [237, 85, 16], [236, 83, 15],
    [235, 80, 14], [234, 78, 13], [232, 75, 12], [231, 73, 12],
    [229, 71, 11], [228, 69, 10], [226, 67, 10], [225, 65, 9],
    [223, 63, 8], [221, 61, 8], [220, 59, 7], [218, 57, 7],
    [216, 55, 6], [214, 53, 6], [212, 51, 5], [210, 49, 5],
    [208, 47, 5], [206, 45, 4], [204, 43, 4], [202, 42, 4],
    [200, 40, 3], [197, 38, 3], [195, 37, 3], [193, 35, 2],
    [190, 33, 2], [188, 32, 2], [185, 30, 2], [183, 29, 2],
    [180, 27, 1], [178, 26, 1], [175, 24, 1], [172, 23, 1],
    [169, 22, 1], [167, 20, 1], [164, 19, 1], [161, 18, 1],
    [158, 16, 1], [155, 15, 1], [152, 14, 1], [149, 13, 1],
    [146, 11, 1], [142, 10, 1], [139, 9, 2], [136, 8, 2],
], dtype=np.uint8)


def denormalize_image(
    image: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
) -> np.ndarray:
    """Denormalize image for visualization."""
    if image.shape[0] == 3:
        image = image.transpose(1, 2, 0)

    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)

    image = image * std + mean
    image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def visualize_semantic(semantic: np.ndarray) -> np.ndarray:
    """Visualize semantic segmentation with color palette."""
    H, W = semantic.shape
    vis = np.zeros((H, W, 3), dtype=np.uint8)

    for class_id in range(len(PANOPTIC_PALETTE)):
        mask = semantic == class_id
        vis[mask] = PANOPTIC_PALETTE[class_id]

    return vis


def visualize_center_heatmap(
    center: np.ndarray,
    image: np.ndarray,
    threshold: float = 0.1,
) -> np.ndarray:
    """Visualize center heatmap overlaid on image.

    High probability = red, low probability = white.

    Args:
        center: Center heatmap (1, H, W) or (H, W).
        image: Original image (H, W, 3) uint8.
        threshold: Only show values above this threshold.
    """
    if center.ndim == 3:
        center = center[0]

    H, W = center.shape
    vis = image.copy()

    # Create mask for values above threshold
    mask = center > threshold

    if mask.any():
        # Normalize center values in [threshold, 1] to [0, 1]
        center_normalized = np.clip((center - threshold) / (1 - threshold + 1e-8), 0, 1)

        # White (low prob) to Red (high prob)
        # RGB: (255, 255, 255) -> (255, 0, 0)
        overlay = np.zeros((H, W, 3), dtype=np.float32)
        overlay[..., 0] = 255  # R always 255
        overlay[..., 1] = 255 * (1 - center_normalized)  # G: 255 -> 0
        overlay[..., 2] = 255 * (1 - center_normalized)  # B: 255 -> 0

        # Blend with original image where mask is True
        alpha = 0.7
        vis[mask] = (alpha * overlay[mask] + (1 - alpha) * vis[mask]).astype(np.uint8)

    return vis


def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
    """Convert HSV to RGB."""
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

    i = (h * 6).astype(np.int32) % 6
    f = h * 6 - i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    rgb = np.zeros_like(hsv)

    for idx in range(6):
        mask = i == idx
        if idx == 0:
            rgb[mask] = np.stack([v[mask], t[mask], p[mask]], axis=-1)
        elif idx == 1:
            rgb[mask] = np.stack([q[mask], v[mask], p[mask]], axis=-1)
        elif idx == 2:
            rgb[mask] = np.stack([p[mask], v[mask], t[mask]], axis=-1)
        elif idx == 3:
            rgb[mask] = np.stack([p[mask], q[mask], v[mask]], axis=-1)
        elif idx == 4:
            rgb[mask] = np.stack([t[mask], p[mask], v[mask]], axis=-1)
        elif idx == 5:
            rgb[mask] = np.stack([v[mask], p[mask], q[mask]], axis=-1)

    return rgb


def visualize_offset(
    offset: np.ndarray,
    max_offset: Optional[float] = None,
) -> np.ndarray:
    """Visualize offset as color-coded image using HSV.

    Center (low magnitude) = white/bright
    Far from center (high magnitude) = colored/dark
    Hue = direction (360 degrees)
    """
    dy = offset[0]
    dx = offset[1]

    angle = np.arctan2(dy, dx)
    magnitude = np.sqrt(dx ** 2 + dy ** 2)

    if max_offset is None:
        max_offset = magnitude.max() + 1e-8

    # Normalize magnitude to [0, 1]
    norm_mag = np.clip(magnitude / max_offset, 0, 1)

    # Hue: direction (0-360 degrees mapped to 0-1)
    hue = (angle + np.pi) / (2 * np.pi)

    # Saturation: low at center (white), high at edges (colored)
    saturation = norm_mag

    # Value: always bright to see colors clearly
    value = np.ones_like(hue)

    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)

    return (rgb * 255).astype(np.uint8)


def visualize_depth(
    depth: np.ndarray,
    min_depth: float = 1.0,
    max_depth: float = 150.0,
) -> np.ndarray:
    """Visualize depth with turbo colormap.

    Closer = red/orange, further = blue/purple.

    Args:
        depth: Depth map (1, H, W) or (H, W) in meters.
        min_depth: Minimum depth for normalization.
        max_depth: Maximum depth for normalization.

    Returns:
        Colored depth visualization (H, W, 3) uint8.
    """
    if depth.ndim == 3:
        depth = depth[0]

    # Normalize to log scale for better visualization
    log_min = np.log(min_depth)
    log_max = np.log(max_depth)
    depth_clipped = np.clip(depth, min_depth, max_depth)
    log_depth = np.log(depth_clipped)

    # Map to colormap indices
    norm_depth = (log_depth - log_min) / (log_max - log_min)
    num_colors = len(TURBO_COLORMAP)
    indices = (norm_depth * (num_colors - 1)).astype(np.int32)
    indices = np.clip(indices, 0, num_colors - 1)

    return TURBO_COLORMAP[indices]


def visualize_predictions(
    image: np.ndarray,
    semantic: np.ndarray,
    center: np.ndarray,
    offset: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    center_threshold: float = 0.1,
) -> np.ndarray:
    """Create a 2x2 grid visualization for panoptic.

    Layout:
        - Top-left: Original image
        - Top-right: Semantic segmentation
        - Bottom-left: Center heatmap (overlay on image, threshold=0.1)
        - Bottom-right: Offset (white=center, colored=direction)
    """
    vis_image = denormalize_image(image, mean, std)
    vis_semantic = visualize_semantic(semantic)
    vis_center = visualize_center_heatmap(center, vis_image, threshold=center_threshold)
    vis_offset = visualize_offset(offset)

    H, W = vis_image.shape[:2]
    grid = np.zeros((2 * H, 2 * W, 3), dtype=np.uint8)

    grid[:H, :W] = vis_image
    grid[:H, W:] = vis_semantic
    grid[H:, :W] = vis_center
    grid[H:, W:] = vis_offset

    return grid


def visualize_depth_predictions(
    image: np.ndarray,
    depth_pred: np.ndarray,
    depth_gt: Optional[np.ndarray] = None,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    min_depth: float = 1.0,
    max_depth: float = 150.0,
) -> np.ndarray:
    """Create a visualization for depth estimation.

    Layout (1x2 or 1x3):
        - Left: Original image
        - Middle: Predicted depth
        - Right: GT depth (if available)
    """
    vis_image = denormalize_image(image, mean, std)
    vis_depth_pred = visualize_depth(depth_pred, min_depth, max_depth)

    H, W = vis_image.shape[:2]

    if depth_gt is not None:
        vis_depth_gt = visualize_depth(depth_gt, min_depth, max_depth)
        grid = np.zeros((H, 3 * W, 3), dtype=np.uint8)
        grid[:, :W] = vis_image
        grid[:, W:2*W] = vis_depth_pred
        grid[:, 2*W:] = vis_depth_gt
    else:
        grid = np.zeros((H, 2 * W, 3), dtype=np.uint8)
        grid[:, :W] = vis_image
        grid[:, W:] = vis_depth_pred

    return grid


def visualize_joint2d_predictions(
    image: np.ndarray,
    semantic_pred: np.ndarray,
    semantic_gt: np.ndarray,
    center_pred: np.ndarray,
    center_gt: np.ndarray,
    offset_pred: np.ndarray,
    offset_gt: np.ndarray,
    depth_pred: np.ndarray,
    depth_gt: np.ndarray,
    mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
    std: Tuple[float, ...] = (0.229, 0.224, 0.225),
    center_threshold: float = 0.1,
    min_depth: float = 1.0,
    max_depth: float = 150.0,
) -> np.ndarray:
    """Create a 2x4 grid visualization for joint panoptic + depth.

    Layout:
        Row 1 (GT):   Semantic GT | Center GT | Offset GT | Depth GT
        Row 2 (Pred): Semantic Pred | Center Pred | Offset Pred | Depth Pred

    Args:
        image: Input image (3, H, W) normalized.
        semantic_pred: Predicted semantic segmentation (H, W).
        semantic_gt: Ground truth semantic segmentation (H, W).
        center_pred: Predicted center heatmap (1, H, W) or (H, W).
        center_gt: Ground truth center heatmap (1, H, W) or (H, W).
        offset_pred: Predicted offset (2, H, W).
        offset_gt: Ground truth offset (2, H, W).
        depth_pred: Predicted depth (1, H, W) or (H, W).
        depth_gt: Ground truth depth (1, H, W) or (H, W).
        mean: Image normalization mean.
        std: Image normalization std.
        center_threshold: Threshold for center heatmap visualization.
        min_depth: Minimum depth for visualization.
        max_depth: Maximum depth for visualization.

    Returns:
        Visualization grid (2*H, 4*W, 3) uint8.
    """
    vis_image = denormalize_image(image, mean, std)

    # Semantic visualizations
    vis_semantic_gt = visualize_semantic(semantic_gt)
    vis_semantic_pred = visualize_semantic(semantic_pred)

    # Center visualizations (overlay on image)
    vis_center_gt = visualize_center_heatmap(center_gt, vis_image, threshold=center_threshold)
    vis_center_pred = visualize_center_heatmap(center_pred, vis_image, threshold=center_threshold)

    # Offset visualizations
    vis_offset_gt = visualize_offset(offset_gt)
    vis_offset_pred = visualize_offset(offset_pred)

    # Depth visualizations
    vis_depth_gt = visualize_depth(depth_gt, min_depth, max_depth)
    vis_depth_pred = visualize_depth(depth_pred, min_depth, max_depth)

    H, W = vis_image.shape[:2]
    grid = np.zeros((2 * H, 4 * W, 3), dtype=np.uint8)

    # Row 1: GT
    grid[:H, :W] = vis_semantic_gt
    grid[:H, W:2*W] = vis_center_gt
    grid[:H, 2*W:3*W] = vis_offset_gt
    grid[:H, 3*W:] = vis_depth_gt

    # Row 2: Pred
    grid[H:, :W] = vis_semantic_pred
    grid[H:, W:2*W] = vis_center_pred
    grid[H:, 2*W:3*W] = vis_offset_pred
    grid[H:, 3*W:] = vis_depth_pred

    return grid


@HOOKS.register_module()
class VisualizationHook(Hook):
    """Hook for visualizing predictions during training.

    Automatically detects model type (panoptic or depth) and visualizes accordingly.

    Args:
        output_dir: Directory to save visualizations.
        interval: Visualize every N epochs.
        num_samples: Number of samples to visualize.
        mean: Image normalization mean.
        std: Image normalization std.
        min_depth: Minimum depth for visualization (depth model).
        max_depth: Maximum depth for visualization (depth model).
    """

    priority = 90  # Low priority (run late)

    def __init__(
        self,
        output_dir: str,
        interval: int = 1,
        num_samples: int = 5,
        mean: Tuple[float, ...] = (0.485, 0.456, 0.406),
        std: Tuple[float, ...] = (0.229, 0.224, 0.225),
        min_depth: float = 1.0,
        max_depth: float = 150.0,
    ):
        super().__init__()
        self.output_dir = Path(output_dir) / 'visualizations'
        self.interval = interval
        self.num_samples = num_samples
        self.mean = mean
        self.std = std
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.val_dataset: Optional['Dataset'] = None

    def set_dataset(self, dataset: 'Dataset') -> None:
        """Set the validation dataset for visualization."""
        self.val_dataset = dataset

    def _get_model_type(self, model) -> str:
        """Get model type for visualization."""
        model_name = model.__class__.__name__
        if model_name == 'Depth':
            return 'depth'
        elif model_name == 'Joint2d':
            return 'joint2d'
        else:
            return 'panoptic'

    def after_train_epoch(
        self,
        trainer: 'pl.Trainer',
        pl_module: 'pl.LightningModule',
    ) -> None:
        """Visualize predictions after each training epoch."""
        epoch = trainer.current_epoch

        if (epoch + 1) % self.interval != 0:
            return

        if self.val_dataset is None:
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)

        model = pl_module.model
        model.eval()
        device = next(model.parameters()).device

        epoch_dir = self.output_dir / f'epoch_{epoch:03d}'
        epoch_dir.mkdir(parents=True, exist_ok=True)

        sample_indices = list(range(min(self.num_samples, len(self.val_dataset))))
        model_type = self._get_model_type(model)

        with torch.no_grad():
            for idx in sample_indices:
                sample = self.val_dataset[idx]
                image = sample['image'].unsqueeze(0).to(device)

                preds = model.predict(image)
                image_np = sample['image'].numpy()

                if model_type == 'depth':
                    # Depth model visualization
                    depth_np = preds['depth'][0].cpu().numpy()
                    grid = visualize_depth_predictions(
                        image_np, depth_np,
                        mean=self.mean, std=self.std,
                        min_depth=self.min_depth, max_depth=self.max_depth,
                    )
                elif model_type == 'joint2d':
                    # Joint2d model visualization (GT row + Pred row)
                    semantic_pred = preds['semantic'][0].cpu().numpy()
                    center_pred = preds['center'][0].cpu().numpy()
                    offset_pred = preds['offset'][0].cpu().numpy()
                    depth_pred = preds['depth'][0].cpu().numpy()

                    # Get GT from sample
                    semantic_gt = sample['semantic_target'].numpy()
                    center_gt = sample['center_target'].numpy()
                    offset_gt = sample['offset_target'].numpy()
                    # For depth GT, convert soft targets to depth values
                    depth_target = sample['depth_target'].numpy()  # (num_bins, H, W)
                    depth_gt = self._soft_target_to_depth(depth_target)

                    grid = visualize_joint2d_predictions(
                        image_np,
                        semantic_pred, semantic_gt,
                        center_pred, center_gt,
                        offset_pred, offset_gt,
                        depth_pred, depth_gt,
                        mean=self.mean, std=self.std,
                        min_depth=self.min_depth, max_depth=self.max_depth,
                    )
                else:
                    # Panoptic model visualization
                    semantic_np = preds['semantic'][0].cpu().numpy()
                    center_np = preds['center'][0].cpu().numpy()
                    offset_np = preds['offset'][0].cpu().numpy()

                    grid = visualize_predictions(
                        image_np, semantic_np, center_np, offset_np,
                        self.mean, self.std,
                    )

                save_path = epoch_dir / f'sample_{idx:04d}.png'
                Image.fromarray(grid).save(save_path)

        model.train()

    def _soft_target_to_depth(self, soft_target: np.ndarray) -> np.ndarray:
        """Convert soft target (num_bins, H, W) to depth values.

        Args:
            soft_target: Soft target probabilities (num_bins, H, W).

        Returns:
            Depth values (H, W) in meters.
        """
        num_bins = soft_target.shape[0]
        # Create log-spaced bin centers
        bin_edges = np.linspace(
            np.log(self.min_depth), np.log(self.max_depth), num_bins + 1
        )
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_centers = np.exp(bin_centers)  # Convert back to linear space

        # Weighted sum of bin centers
        # soft_target: (num_bins, H, W), bin_centers: (num_bins,)
        depth = np.sum(
            soft_target * bin_centers[:, None, None], axis=0
        )
        return depth
