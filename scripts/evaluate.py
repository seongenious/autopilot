#!/usr/bin/env python
"""Evaluation script with visualization for perception models.

Generates evaluation figures with:
- Left: 6 camera images with detection overlay
- Right: BEV visualization with GT and predictions

Results are organized by scene:
    result/<scene_name>/frame_<timestamp>.jpg

Usage:
    python scripts/evaluate.py +checkpoint=path/to/checkpoint.ckpt
    python scripts/evaluate.py +checkpoint=path/to/checkpoint.ckpt +max_samples=100
    python scripts/evaluate.py +checkpoint=path/to/checkpoint.ckpt +output_dir=result
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autopilot.utils.visualization import (
    create_evaluation_figure,
    denormalize_image,
    tensor_to_numpy,
)


def load_model(checkpoint_path: str, cfg: DictConfig) -> Any:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        cfg: Hydra configuration object.

    Returns:
        Loaded model in eval mode.
    """
    from scripts.train import DetectionModule

    model = DetectionModule.load_from_checkpoint(
        checkpoint_path,
        cfg=cfg,
        map_location='cuda' if torch.cuda.is_available() else 'cpu',
    )
    model.eval()
    return model


def extract_gt_boxes(batch: Dict[str, Any]) -> Optional[Dict]:
    """Extract ground truth boxes from batch.

    Args:
        batch: Input batch dictionary.

    Returns:
        Dictionary with box parameters or None.
    """
    gt_boxes = batch.get('gt_boxes', None)
    gt_labels = batch.get('gt_labels', None)

    if gt_boxes is None:
        return None

    gt_boxes = tensor_to_numpy(gt_boxes)
    gt_labels = tensor_to_numpy(gt_labels) if gt_labels is not None else None

    # Remove batch dimension if present
    if gt_boxes.ndim == 3:
        gt_boxes = gt_boxes[0]
    if gt_labels is not None and gt_labels.ndim == 2:
        gt_labels = gt_labels[0]

    if len(gt_boxes) == 0:
        return None

    # gt_boxes format: (N, 7) -> [x, y, z, l, w, h, yaw]
    return {
        'centers': gt_boxes[:, :3],
        'sizes': gt_boxes[:, 3:6],
        'rotations': gt_boxes[:, 6],
        'labels': gt_labels if gt_labels is not None else np.zeros(len(gt_boxes)),
    }


def extract_pred_boxes(outputs: Dict[str, Any], score_threshold: float = 0.3) -> Optional[Dict]:
    """Extract predicted boxes from model outputs.

    Args:
        outputs: Model outputs dictionary.
        score_threshold: Minimum confidence score.

    Returns:
        Dictionary with box parameters or None.
    """
    # Check for detection outputs
    if 'det_boxes' in outputs:
        pred = outputs['det_boxes']
        if isinstance(pred, dict):
            return {k: tensor_to_numpy(v) for k, v in pred.items()}

    # Try to decode from heatmap outputs
    if 'cls' in outputs and 'reg' in outputs:
        cls_pred = tensor_to_numpy(outputs['cls'])
        reg_pred = tensor_to_numpy(outputs['reg'])

        # Remove batch dimension
        if cls_pred.ndim == 4:
            cls_pred = cls_pred[0]
        if reg_pred.ndim == 4:
            reg_pred = reg_pred[0]

        # Simple decoding: find peaks in heatmap
        # cls_pred: (num_classes, H, W)
        # reg_pred: (8, H, W) -> [x, y, z, l, w, h, sin, cos]
        num_classes, H, W = cls_pred.shape

        # Apply sigmoid
        scores = 1 / (1 + np.exp(-cls_pred))

        # Find all positions above threshold
        centers_list = []
        sizes_list = []
        rotations_list = []
        labels_list = []
        scores_list = []

        for cls_id in range(num_classes):
            cls_scores = scores[cls_id]
            mask = cls_scores > score_threshold

            if not mask.any():
                continue

            ys, xs = np.where(mask)
            for y, x in zip(ys, xs):
                score = cls_scores[y, x]
                reg = reg_pred[:, y, x]

                # Decode box parameters
                # Assuming reg format: [dx, dy, z, log(l), log(w), log(h), sin, cos]
                # Convert grid position to world coordinates
                # This is a simplified version - actual decoding depends on model
                x_world = (x / W - 0.5) * 102.4  # Assuming range [-51.2, 51.2]
                y_world = (y / H - 0.5) * 102.4
                x_world += reg[0]
                y_world += reg[1]

                centers_list.append([x_world, y_world, reg[2]])
                sizes_list.append([np.exp(reg[3]), np.exp(reg[4]), np.exp(reg[5])])
                rotations_list.append(np.arctan2(reg[6], reg[7]))
                labels_list.append(cls_id)
                scores_list.append(score)

        if len(centers_list) == 0:
            return None

        return {
            'centers': np.array(centers_list),
            'sizes': np.array(sizes_list),
            'rotations': np.array(rotations_list),
            'labels': np.array(labels_list),
            'scores': np.array(scores_list),
        }

    return None


def evaluate_and_visualize(
    model: Any,
    dataloader: Any,
    output_dir: str,
    class_names: List[str],
    camera_names: List[str],
    point_cloud_range: List[float],
    max_samples: Optional[int] = None,
    score_threshold: float = 0.3,
) -> Dict[str, Any]:
    """Run evaluation and save visualizations organized by scene.

    Args:
        model: Trained model.
        dataloader: Validation dataloader.
        output_dir: Root directory to save visualizations.
        class_names: List of class names.
        camera_names: List of camera names.
        point_cloud_range: BEV range [x_min, y_min, z_min, x_max, y_max, z_max].
        max_samples: Maximum number of samples to process.
        score_threshold: Minimum score for predictions.

    Returns:
        Dictionary of evaluation metrics.
    """
    device = next(model.parameters()).device
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics = {
        'total_samples': 0,
        'scenes': set(),
    }

    # Track frame indices per scene
    scene_frame_counts = {}

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_samples and batch_idx >= max_samples:
                break

            # Move batch to device
            batch_gpu = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}

            # Forward pass
            outputs = model(batch_gpu)

            # Get scene and frame info
            scene_name = batch.get('scene_name', [f'scene_{batch_idx:04d}'])
            if isinstance(scene_name, list):
                scene_name = scene_name[0]
            timestamp = batch.get('timestamp', [batch_idx])
            if isinstance(timestamp, (list, torch.Tensor)):
                timestamp = int(timestamp[0] if isinstance(timestamp, list) else timestamp[0].item())

            # Create scene directory
            scene_dir = output_path / scene_name
            scene_dir.mkdir(parents=True, exist_ok=True)

            # Track frame count for this scene
            if scene_name not in scene_frame_counts:
                scene_frame_counts[scene_name] = 0
            frame_idx = scene_frame_counts[scene_name]
            scene_frame_counts[scene_name] += 1

            # Extract images
            images = tensor_to_numpy(batch['images'])
            if images.ndim == 5:
                images = images[0]  # Remove batch dimension

            # Denormalize images
            vis_images = [denormalize_image(img) for img in images]

            # Extract lidar2img
            lidar2img = tensor_to_numpy(batch.get('lidar2img', None))
            if lidar2img is not None and lidar2img.ndim == 4:
                lidar2img = lidar2img[0]

            # Extract boxes
            gt_boxes = extract_gt_boxes(batch)
            pred_boxes = extract_pred_boxes(outputs, score_threshold)

            # Create combined figure
            figure = create_evaluation_figure(
                images=vis_images,
                gt_boxes=gt_boxes,
                pred_boxes=pred_boxes,
                lidar2img=lidar2img,
                point_cloud_range=point_cloud_range,
                class_names=class_names,
                camera_names=camera_names,
                score_threshold=score_threshold,
            )

            # Save figure
            filename = f"frame_{frame_idx:04d}_{timestamp}.jpg"
            figure_path = scene_dir / filename
            cv2.imwrite(str(figure_path), cv2.cvtColor(figure, cv2.COLOR_RGB2BGR))

            metrics['total_samples'] += 1
            metrics['scenes'].add(scene_name)

    metrics['num_scenes'] = len(metrics['scenes'])
    metrics['scenes'] = list(metrics['scenes'])

    print(f"\nVisualization saved to: {output_dir}")
    print(f"Total samples processed: {metrics['total_samples']}")
    print(f"Total scenes: {metrics['num_scenes']}")

    return metrics


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Main evaluation function.

    Args:
        cfg: Hydra configuration object.
    """
    # Check required arguments
    if not hasattr(cfg, 'checkpoint') or cfg.checkpoint is None:
        print("No checkpoint provided.")
        print("Usage: python scripts/evaluate.py +checkpoint=path/to/checkpoint.ckpt")
        return

    # Output directory
    output_dir = cfg.get('output_dir', 'result')

    # Get config values
    class_names = list(cfg.dataset.classes) if hasattr(cfg.dataset, 'classes') else []
    camera_names = list(cfg.dataset.cameras) if hasattr(cfg.dataset, 'cameras') else [
        'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
        'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT',
    ]
    point_cloud_range = list(cfg.dataset.point_cloud_range) if hasattr(cfg.dataset, 'point_cloud_range') else [
        -51.2, -51.2, -5.0, 51.2, 51.2, 3.0
    ]

    # Load model
    print(f"Loading model from: {cfg.checkpoint}")
    model = load_model(cfg.checkpoint, cfg)

    # Create dataloader
    from scripts.train import DetectionDataModule
    datamodule = DetectionDataModule(cfg)
    datamodule.setup('validate')
    dataloader = datamodule.val_dataloader()

    # Run evaluation
    max_samples = cfg.get('max_samples', None)
    score_threshold = cfg.get('score_threshold', 0.3)

    metrics = evaluate_and_visualize(
        model=model,
        dataloader=dataloader,
        output_dir=output_dir,
        class_names=class_names,
        camera_names=camera_names,
        point_cloud_range=point_cloud_range,
        max_samples=max_samples,
        score_threshold=score_threshold,
    )

    print("\nEvaluation complete!")
    print(f"Scenes: {metrics['scenes'][:5]}{'...' if len(metrics['scenes']) > 5 else ''}")


if __name__ == '__main__':
    main()
