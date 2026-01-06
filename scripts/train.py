#!/usr/bin/env python
"""Training script for 3D object detection with PyTorch Lightning and Hydra.

Architecture:
    - Multi-camera images (6 cameras) → ResNet18 backbone → BiFPN neck
    - Multi-scale features → BEV Transformer → BEV features (20 x 80 x 256)
    - BEV features → Detection Head → 3D bounding boxes

Usage:
    python scripts/train.py
    python scripts/train.py training.max_epochs=24
    python scripts/train.py hardware.gpus=2 dataset.batch_size=8
"""

import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import TensorBoardLogger

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from autopilot.datasets import NuScenesDataset, collate_fn
from autopilot.losses import SimplifiedDetectionLoss
from autopilot.models.perception import SimplePerceptionModel
from autopilot.utils.callbacks import VisualizationCallback

# Import modules to trigger registration
import autopilot.models.backbones  # noqa: F401
import autopilot.models.necks  # noqa: F401
import autopilot.models.heads.detection  # noqa: F401


class DetectionModule(pl.LightningModule):
    """PyTorch Lightning module for 3D object detection.

    Combines:
        - ResNet18 backbone (per camera)
        - BiFPN neck (per camera)
        - BEV Transformer
        - Detection Head
    """

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize DetectionModule.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__()
        self.cfg = cfg
        self.save_hyperparameters()

        # Build model
        self._build_model()

        # Build loss
        self.loss_fn = SimplifiedDetectionLoss(
            cls_weight=1.0,
            reg_weight=1.0,
        )

    def _build_model(self) -> None:
        """Build model from config."""
        # Get configurations
        num_classes = len(self.cfg.dataset.classes)
        num_cameras = len(self.cfg.dataset.cameras)

        # Image size for determining BEV resolution
        img_h, img_w = self.cfg.dataset.img_size

        # BEV configuration
        # For 102.4m range with 0.2m voxel: 512 voxels, but we use 80 for efficiency
        bev_h = 20  # Y direction
        bev_w = 80  # X direction

        backbone_cfg = {
            'model_name': 'resnet18',
            'pretrained': True,
            'out_indices': (0, 1, 2, 3),
        }

        neck_cfg = {
            'in_channels': [64, 128, 256, 512],
            'out_channels': 128,
            'num_levels': 4,
            'num_layers': 2,
        }

        bev_cfg = {
            'in_channels': 128,
            'embed_dim': 256,
            'bev_h': bev_h,
            'bev_w': bev_w,
            'num_cameras': num_cameras,
            'num_layers': 2,
            'num_heads': 8,
            'ffn_dim': 512,
        }

        self.model = SimplePerceptionModel(
            backbone_cfg=backbone_cfg,
            neck_cfg=neck_cfg,
            bev_cfg=bev_cfg,
            num_classes=num_classes,
            num_cameras=num_cameras,
        )

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            batch: Input batch containing images and metadata.

        Returns:
            Dictionary of predictions.
        """
        return self.model(batch)

    def training_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        outputs = self(batch)

        # Compute loss
        losses = self.loss_fn(
            pred=outputs,
            gt_heatmap=batch['gt_heatmap'],
            gt_reg=batch['gt_reg'],
            gt_indices=batch['gt_indices'],
            num_boxes=batch['num_boxes'],
        )

        # Log losses
        self.log('train/loss', losses['loss'], prog_bar=True)
        self.log('train/loss_cls', losses['loss_cls'])
        self.log('train/loss_reg', losses['loss_reg'])

        return losses['loss']

    def validation_step(
        self, batch: Dict[str, Any], batch_idx: int
    ) -> Dict[str, Any]:
        """Validation step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Dictionary with validation outputs.
        """
        outputs = self(batch)

        # Compute loss
        losses = self.loss_fn(
            pred=outputs,
            gt_heatmap=batch['gt_heatmap'],
            gt_reg=batch['gt_reg'],
            gt_indices=batch['gt_indices'],
            num_boxes=batch['num_boxes'],
        )

        # Log losses
        self.log('val/loss', losses['loss'], prog_bar=True, sync_dist=True)
        self.log('val/loss_cls', losses['loss_cls'], sync_dist=True)
        self.log('val/loss_reg', losses['loss_reg'], sync_dist=True)

        return {
            'val_loss': losses['loss'],
            'outputs': outputs,
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler.

        Returns:
            Dictionary with optimizer and lr_scheduler configuration.
        """
        cfg = self.cfg.training

        # Optimizer
        optimizer_cfg = cfg.optimizer
        if optimizer_cfg.type == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=optimizer_cfg.lr,
                weight_decay=optimizer_cfg.weight_decay,
                betas=tuple(optimizer_cfg.betas),
            )
        else:
            raise ValueError(f'Unknown optimizer: {optimizer_cfg.type}')

        # Scheduler
        scheduler_cfg = cfg.scheduler
        if scheduler_cfg.type == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=cfg.max_epochs,
                eta_min=scheduler_cfg.eta_min,
            )
        else:
            raise ValueError(f'Unknown scheduler: {scheduler_cfg.type}')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }


class DetectionDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for detection datasets."""

    def __init__(self, cfg: DictConfig) -> None:
        """Initialize DetectionDataModule.

        Args:
            cfg: Hydra configuration object.
        """
        super().__init__()
        self.cfg = cfg
        self.train_dataset: Optional[NuScenesDataset] = None
        self.val_dataset: Optional[NuScenesDataset] = None

    def setup(self, stage: Optional[str] = None) -> None:
        """Setup datasets.

        Args:
            stage: Current stage ('fit', 'validate', 'test', 'predict').
        """
        dataset_cfg = self.cfg.dataset

        # Common parameters
        common_params = {
            'img_size': tuple(dataset_cfg.img_size),
            'point_cloud_range': list(dataset_cfg.point_cloud_range),
            'bev_size': (20, 80),  # Match model BEV size
            'classes': list(dataset_cfg.classes),
            'min_visibility': 3,  # Only objects with visibility >= 3
            'img_norm_cfg': OmegaConf.to_container(dataset_cfg.img_norm_cfg),
        }

        data_root = dataset_cfg.data_root

        # Check if annotation files exist
        train_ann_file = os.path.join(data_root, 'nuscenes_infos_train.pkl')
        val_ann_file = os.path.join(data_root, 'nuscenes_infos_val.pkl')

        if stage == 'fit' or stage is None:
            if os.path.exists(train_ann_file):
                self.train_dataset = NuScenesDataset(
                    data_root=data_root,
                    ann_file=train_ann_file,
                    test_mode=False,
                    **common_params,
                )
                print(f"Loaded training dataset: {len(self.train_dataset)} samples")
            else:
                print(f"Warning: Training annotation file not found: {train_ann_file}")
                print("Using dummy dataset for training")

        if stage == 'validate' or stage == 'fit' or stage is None:
            if os.path.exists(val_ann_file):
                self.val_dataset = NuScenesDataset(
                    data_root=data_root,
                    ann_file=val_ann_file,
                    test_mode=False,
                    **common_params,
                )
                print(f"Loaded validation dataset: {len(self.val_dataset)} samples")
            else:
                print(f"Warning: Validation annotation file not found: {val_ann_file}")
                print("Using dummy dataset for validation")

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        """Get training dataloader."""
        if self.train_dataset is None:
            return self._dummy_dataloader()

        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=True,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        """Get validation dataloader."""
        if self.val_dataset is None:
            return self._dummy_dataloader()

        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.cfg.dataset.batch_size,
            shuffle=False,
            num_workers=self.cfg.hardware.num_workers,
            pin_memory=True,
            collate_fn=collate_fn,
        )

    def _dummy_dataloader(self) -> torch.utils.data.DataLoader:
        """Create dummy dataloader for testing without real data."""
        num_classes = len(self.cfg.dataset.classes)
        img_h, img_w = self.cfg.dataset.img_size
        bev_h, bev_w = 20, 80

        class DummyDataset(torch.utils.data.Dataset):
            """Dummy dataset with random data matching real dataset format."""

            def __init__(self, size: int = 100) -> None:
                self.size = size

            def __len__(self) -> int:
                return self.size

            def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
                return {
                    'images': torch.randn(6, 3, img_h, img_w),
                    'lidar2img': torch.eye(4).unsqueeze(0).repeat(6, 1, 1),
                    'gt_heatmap': torch.zeros(num_classes, bev_h, bev_w),
                    'gt_boxes': torch.zeros(0, 7),
                    'gt_labels': torch.zeros(0, dtype=torch.long),
                    'gt_reg': torch.zeros(0, 8),
                    'gt_indices': torch.zeros(0, dtype=torch.long),
                    'sample_idx': idx,
                    'token': f'dummy_{idx}',
                }

        def dummy_collate(batch):
            result = {}
            result['images'] = torch.stack([s['images'] for s in batch])
            result['lidar2img'] = torch.stack([s['lidar2img'] for s in batch])
            result['gt_heatmap'] = torch.stack([s['gt_heatmap'] for s in batch])
            result['gt_boxes'] = torch.zeros(len(batch), 1, 7)
            result['gt_labels'] = torch.zeros(len(batch), 1, dtype=torch.long)
            result['gt_reg'] = torch.zeros(len(batch), 1, 8)
            result['gt_indices'] = torch.zeros(len(batch), 1, dtype=torch.long)
            result['num_boxes'] = torch.zeros(len(batch), dtype=torch.long)
            result['sample_idx'] = [s['sample_idx'] for s in batch]
            result['token'] = [s['token'] for s in batch]
            return result

        return torch.utils.data.DataLoader(
            DummyDataset(),
            batch_size=2,
            shuffle=True,
            num_workers=0,
            collate_fn=dummy_collate,
        )


def setup_callbacks(cfg: DictConfig) -> List:
    """Setup training callbacks.

    Args:
        cfg: Hydra configuration object.

    Returns:
        List of callbacks.
    """
    callbacks = []

    # Checkpoint callback
    ckpt_cfg = cfg.training.checkpoint
    callbacks.append(
        ModelCheckpoint(
            dirpath='checkpoints',
            filename='epoch={epoch:02d}-val_loss={val/loss:.4f}',
            save_top_k=ckpt_cfg.save_top_k,
            monitor=ckpt_cfg.monitor,
            mode=ckpt_cfg.mode,
            save_last=ckpt_cfg.save_last,
            auto_insert_metric_name=False,
        )
    )

    # Learning rate monitor
    callbacks.append(LearningRateMonitor(logging_interval='step'))

    # Progress bar
    callbacks.append(RichProgressBar())

    # Early stopping
    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=cfg.training.early_stopping.monitor,
                patience=cfg.training.early_stopping.patience,
                mode=cfg.training.early_stopping.mode,
            )
        )

    # Visualization callback
    callbacks.append(
        VisualizationCallback(
            log_every_n_epochs=1,
            num_samples=2,
            log_val=True,
            class_names=list(cfg.dataset.classes) if hasattr(cfg.dataset, 'classes') else None,
            camera_names=list(cfg.dataset.cameras) if hasattr(cfg.dataset, 'cameras') else None,
        )
    )

    return callbacks


@hydra.main(version_base=None, config_path='../configs', config_name='config')
def main(cfg: DictConfig) -> None:
    """Main training function.

    Args:
        cfg: Hydra configuration object.
    """
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    pl.seed_everything(cfg.seed, workers=True)

    # Setup logger
    logger = TensorBoardLogger(save_dir='.', name='logs')

    # Setup callbacks
    callbacks = setup_callbacks(cfg)

    # Create model and datamodule
    model = DetectionModule(cfg)
    datamodule = DetectionDataModule(cfg)

    # Create trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.max_epochs,
        accelerator='gpu' if cfg.hardware.gpus > 0 else 'cpu',
        devices=cfg.hardware.gpus if cfg.hardware.gpus > 0 else 1,
        precision=cfg.hardware.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm=cfg.training.gradient_clip_algorithm,
        accumulate_grad_batches=cfg.training.accumulate_grad_batches,
        val_check_interval=cfg.training.val_check_interval,
        log_every_n_steps=cfg.logging.log_every_n_steps,
        logger=logger,
        callbacks=callbacks,
        deterministic=False,  # For performance
    )

    # Train
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
