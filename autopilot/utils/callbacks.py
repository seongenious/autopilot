"""PyTorch Lightning callbacks for visualization and monitoring."""

from typing import Any, Dict, List, Optional

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback

from autopilot.utils.visualization import (
    create_multi_view_visualization,
    denormalize_image,
    tensor_to_numpy,
    visualize_bev_map,
    visualize_depth,
    visualize_detection,
)


class VisualizationCallback(Callback):
    """Callback for logging visualizations during training/validation.

    Logs visualizations of predictions vs ground truth to TensorBoard.

    Args:
        log_every_n_epochs: Log visualizations every N epochs.
        num_samples: Number of samples to visualize per batch.
        log_train: Whether to log training visualizations.
        log_val: Whether to log validation visualizations.
        class_names: List of class names for detection.
        camera_names: List of camera names for multi-view.
    """

    def __init__(
        self,
        log_every_n_epochs: int = 1,
        num_samples: int = 2,
        log_train: bool = False,
        log_val: bool = True,
        class_names: Optional[List[str]] = None,
        camera_names: Optional[List[str]] = None,
    ) -> None:
        super().__init__()
        self.log_every_n_epochs = log_every_n_epochs
        self.num_samples = num_samples
        self.log_train = log_train
        self.log_val = log_val
        self.class_names = class_names or []
        self.camera_names = camera_names or [
            'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
            'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
        ]

        # Store samples for visualization
        self._val_samples: List[Dict] = []

    def on_validation_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
        dataloader_idx: int = 0,
    ) -> None:
        """Store samples during validation for visualization."""
        if not self.log_val:
            return

        # Only store first few batches
        if len(self._val_samples) >= self.num_samples:
            return

        # Store batch and outputs for later visualization
        sample = {
            'batch': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()},
            'outputs': {k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
                       for k, v in outputs.items()} if isinstance(outputs, dict) else outputs,
        }
        self._val_samples.append(sample)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Log visualizations at the end of validation epoch."""
        if not self.log_val:
            return

        if trainer.current_epoch % self.log_every_n_epochs != 0:
            self._val_samples.clear()
            return

        logger = trainer.logger
        if logger is None:
            self._val_samples.clear()
            return

        # Process stored samples
        for i, sample in enumerate(self._val_samples):
            self._log_sample_visualization(
                logger, sample, f"val/sample_{i}", trainer.current_epoch
            )

        self._val_samples.clear()

    def _log_sample_visualization(
        self,
        logger: Any,
        sample: Dict,
        tag_prefix: str,
        epoch: int,
    ) -> None:
        """Log visualization for a single sample."""
        batch = sample['batch']
        outputs = sample['outputs']

        # Visualize multi-view images
        if 'images' in batch:
            images = tensor_to_numpy(batch['images'])
            if images.ndim == 5:  # (B, N, C, H, W)
                images = images[0]  # Take first batch

            # Denormalize images
            vis_images = []
            for img in images:
                vis_img = denormalize_image(img)
                vis_images.append(vis_img)

            # Create grid
            grid = create_multi_view_visualization(
                vis_images,
                self.camera_names[:len(vis_images)],
            )

            # Log to tensorboard
            if hasattr(logger, 'experiment'):
                logger.experiment.add_image(
                    f"{tag_prefix}/multi_view",
                    grid.transpose(2, 0, 1),  # HWC -> CHW
                    epoch,
                )

        # Visualize detection if available
        if 'det_boxes' in outputs or 'gt_boxes' in batch:
            self._log_detection_visualization(
                logger, batch, outputs, tag_prefix, epoch
            )

        # Visualize BEV map if available
        if 'map_pred' in outputs or 'gt_map' in batch:
            self._log_map_visualization(
                logger, batch, outputs, tag_prefix, epoch
            )

        # Visualize depth if available
        if 'depth_pred' in outputs or 'gt_depth' in batch:
            self._log_depth_visualization(
                logger, batch, outputs, tag_prefix, epoch
            )

    def _log_detection_visualization(
        self,
        logger: Any,
        batch: Dict,
        outputs: Dict,
        tag_prefix: str,
        epoch: int,
    ) -> None:
        """Log detection visualization."""
        images = tensor_to_numpy(batch.get('images', None))
        if images is None:
            return

        if images.ndim == 5:
            images = images[0]  # First batch

        # Get gt_boxes - must be a dict with expected keys
        gt_boxes = batch.get('gt_boxes', None)
        pred_boxes = outputs.get('det_boxes', None)

        # Skip if boxes are not in the expected format (dict with 'centers', 'dims', etc.)
        if gt_boxes is not None and not isinstance(gt_boxes, dict):
            # Skip visualization for heatmap-based detection targets
            return
        if pred_boxes is not None and not isinstance(pred_boxes, dict):
            return

        lidar2img = tensor_to_numpy(batch.get('lidar2img', None))
        if lidar2img is not None and lidar2img.ndim == 3:
            lidar2img = lidar2img[0]  # First batch, first camera

        # Visualize on front camera
        front_img = denormalize_image(images[0])
        front_lidar2img = lidar2img[0] if lidar2img is not None else None

        vis = visualize_detection(
            front_img,
            gt_boxes=gt_boxes,
            pred_boxes=pred_boxes,
            lidar2img=front_lidar2img,
            class_names=self.class_names,
        )

        if hasattr(logger, 'experiment'):
            logger.experiment.add_image(
                f"{tag_prefix}/detection",
                vis.transpose(2, 0, 1),
                epoch,
            )

    def _log_map_visualization(
        self,
        logger: Any,
        batch: Dict,
        outputs: Dict,
        tag_prefix: str,
        epoch: int,
    ) -> None:
        """Log BEV map visualization."""
        gt_map = tensor_to_numpy(batch.get('gt_map', None))
        pred_map = tensor_to_numpy(outputs.get('map_pred', None))

        if gt_map is not None and gt_map.ndim == 4:
            gt_map = gt_map[0]  # First batch
        if pred_map is not None and pred_map.ndim == 4:
            pred_map = pred_map[0]

        vis = visualize_bev_map(gt_map, pred_map)

        if hasattr(logger, 'experiment'):
            logger.experiment.add_image(
                f"{tag_prefix}/bev_map",
                vis.transpose(2, 0, 1),
                epoch,
            )

    def _log_depth_visualization(
        self,
        logger: Any,
        batch: Dict,
        outputs: Dict,
        tag_prefix: str,
        epoch: int,
    ) -> None:
        """Log depth visualization."""
        gt_depth = tensor_to_numpy(batch.get('gt_depth', None))
        pred_depth = tensor_to_numpy(outputs.get('depth_pred', None))

        if gt_depth is not None and gt_depth.ndim == 4:
            gt_depth = gt_depth[0, 0]  # First batch, first camera
        if pred_depth is not None and pred_depth.ndim == 4:
            pred_depth = pred_depth[0, 0]

        vis = visualize_depth(gt_depth, pred_depth)

        if hasattr(logger, 'experiment'):
            logger.experiment.add_image(
                f"{tag_prefix}/depth",
                vis.transpose(2, 0, 1),
                epoch,
            )


class GradientMonitorCallback(Callback):
    """Callback for monitoring gradient statistics during training.

    Logs gradient norms and histograms to help debug training issues.

    Args:
        log_every_n_steps: Log gradient stats every N steps.
    """

    def __init__(self, log_every_n_steps: int = 100) -> None:
        super().__init__()
        self.log_every_n_steps = log_every_n_steps

    def on_before_optimizer_step(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Log gradient statistics before optimizer step."""
        if trainer.global_step % self.log_every_n_steps != 0:
            return

        grad_norms = {}
        for name, param in pl_module.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm(2).item()
                # Simplify name for logging
                simple_name = name.split('.')[-1]
                if simple_name in grad_norms:
                    grad_norms[simple_name] = max(grad_norms[simple_name], grad_norm)
                else:
                    grad_norms[simple_name] = grad_norm

        # Log total gradient norm
        total_norm = sum(n ** 2 for n in grad_norms.values()) ** 0.5
        pl_module.log('train/grad_norm', total_norm)

        # Log per-module norms
        for name, norm in grad_norms.items():
            pl_module.log(f'train/grad/{name}', norm)
