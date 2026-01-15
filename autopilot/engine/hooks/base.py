"""Base hook class for training callbacks."""

from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    import pytorch_lightning as pl


class Hook:
    """Base class for all hooks.

    Hooks are called at specific points during training:
    - before_run: Before training starts
    - after_run: After training ends
    - before_train_epoch: Before each training epoch
    - after_train_epoch: After each training epoch
    - before_val_epoch: Before each validation epoch
    - after_val_epoch: After each validation epoch
    - before_train_step: Before each training step
    - after_train_step: After each training step
    """

    priority: int = 50  # Lower = higher priority (0-100)

    def before_run(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Called before training starts."""
        pass

    def after_run(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Called after training ends."""
        pass

    def before_train_epoch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Called before each training epoch."""
        pass

    def after_train_epoch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Called after each training epoch."""
        pass

    def before_val_epoch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Called before each validation epoch."""
        pass

    def after_val_epoch(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
    ) -> None:
        """Called after each validation epoch."""
        pass

    def before_train_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Called before each training step."""
        pass

    def after_train_step(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Any,
        batch: Dict[str, Any],
        batch_idx: int,
    ) -> None:
        """Called after each training step."""
        pass
