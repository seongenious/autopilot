"""Hooks module for training callbacks."""

from typing import Any, Dict, List

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback

from autopilot.engine.hooks.base import Hook
from autopilot.engine.hooks.visualization import VisualizationHook
from autopilot.utils.registry import HOOKS

__all__ = [
    "Hook",
    "VisualizationHook",
    "HookCallback",
    "build_hooks",
]


def build_hooks(hook_cfgs: List[Dict[str, Any]]) -> List[Hook]:
    """Build hooks from config list.

    Args:
        hook_cfgs: List of hook configurations.

    Returns:
        List of instantiated hooks.
    """
    hooks = []
    for cfg in hook_cfgs:
        hook = HOOKS.build(cfg)
        hooks.append(hook)

    # Sort by priority (lower = higher priority)
    hooks.sort(key=lambda h: h.priority)
    return hooks


class HookCallback(Callback):
    """PyTorch Lightning callback that wraps custom hooks."""

    def __init__(self, hooks: List[Hook]):
        super().__init__()
        self.hooks = hooks

    def on_fit_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when fit begins."""
        for hook in self.hooks:
            hook.before_run(trainer, pl_module)

    def on_fit_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when fit ends."""
        for hook in self.hooks:
            hook.after_run(trainer, pl_module)

    def on_train_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when a training epoch begins."""
        for hook in self.hooks:
            hook.before_train_epoch(trainer, pl_module)

    def on_train_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when a training epoch ends."""
        for hook in self.hooks:
            hook.after_train_epoch(trainer, pl_module)

    def on_validation_epoch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when a validation epoch begins."""
        for hook in self.hooks:
            hook.before_val_epoch(trainer, pl_module)

    def on_validation_epoch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
    ) -> None:
        """Called when a validation epoch ends."""
        for hook in self.hooks:
            hook.after_val_epoch(trainer, pl_module)

    def on_train_batch_start(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch begins."""
        for hook in self.hooks:
            hook.before_train_step(trainer, pl_module, batch, batch_idx)

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        """Called when a training batch ends."""
        for hook in self.hooks:
            hook.after_train_step(trainer, pl_module, outputs, batch, batch_idx)
