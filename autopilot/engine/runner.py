"""Runner module based on PyTorch Lightning."""

from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR


class Runner(pl.LightningModule):
    """Base runner for training models.

    Wraps a model with training/validation logic.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_cfg: Optional[DictConfig] = None,
        scheduler_cfg: Optional[DictConfig] = None,
    ):
        """Initialize Runner.

        Args:
            model: The model to train.
            optimizer_cfg: Optimizer configuration.
            scheduler_cfg: LR scheduler configuration.
        """
        super().__init__()
        self.model = model
        self.optimizer_cfg = optimizer_cfg or {}
        self.scheduler_cfg = scheduler_cfg or {}

        # Save hyperparameters (excluding model)
        self.save_hyperparameters(ignore=['model'])

    def forward(self, *args, **kwargs) -> Any:
        """Forward pass."""
        return self.model(*args, **kwargs)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        """Training step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.

        Returns:
            Loss tensor.
        """
        losses = self.model.train_step(batch)

        # Log losses
        total_loss = 0
        for name, loss in losses.items():
            self.log(f'train/{name}', loss, prog_bar=(name == 'loss'))
            if name == 'loss':
                total_loss = loss

        return total_loss

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> None:
        """Validation step.

        Args:
            batch: Input batch.
            batch_idx: Batch index.
        """
        losses = self.model.val_step(batch)

        # Log losses
        for name, loss in losses.items():
            self.log(f'val/{name}', loss, prog_bar=(name == 'loss'), sync_dist=True)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizer and scheduler."""
        # Build optimizer
        optimizer_type = self.optimizer_cfg.get('type', 'AdamW')
        lr = self.optimizer_cfg.get('lr', 2e-4)
        weight_decay = self.optimizer_cfg.get('weight_decay', 0.01)

        if optimizer_type == 'AdamW':
            optimizer = AdamW(
                self.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                betas=self.optimizer_cfg.get('betas', (0.9, 0.999)),
            )
        else:
            raise ValueError(f'Unknown optimizer type: {optimizer_type}')

        # Build scheduler
        scheduler_type = self.scheduler_cfg.get('type', 'CosineAnnealingLR')

        if scheduler_type == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.scheduler_cfg.get('T_max', 24),
                eta_min=self.scheduler_cfg.get('eta_min', 1e-7),
            )
        else:
            raise ValueError(f'Unknown scheduler type: {scheduler_type}')

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
            },
        }
