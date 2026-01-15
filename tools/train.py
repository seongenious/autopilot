"""Training script with Hydra configuration."""

from pathlib import Path

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from autopilot.datasets import CityscapesDataset, collate_fn
from autopilot.datasets.transforms import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
)
from autopilot.engine.runner import Runner
from autopilot.models.builder import build_model

# Import to register modules
import autopilot.models.backbones  # noqa: F401
import autopilot.models.necks  # noqa: F401
import autopilot.models.heads.perception2d  # noqa: F401
import autopilot.models.detectors  # noqa: F401


def build_transforms(cfg: DictConfig, is_train: bool = True):
    """Build data transforms from config."""
    crop_size = tuple(cfg.dataset.crop_size)
    transforms_list = []

    if is_train:
        transforms_list.append(RandomCrop(crop_size))
        transforms_list.append(RandomHorizontalFlip(0.5))
    else:
        transforms_list.append(RandomCrop(crop_size))

    transforms_list.append(Normalize(
        mean=tuple(cfg.dataset.img_norm_cfg.mean),
        std=tuple(cfg.dataset.img_norm_cfg.std),
    ))

    return Compose(transforms_list)


def build_dataloader(cfg: DictConfig, split: str = "train"):
    """Build dataloader from config."""
    is_train = split == "train"
    transforms = build_transforms(cfg, is_train=is_train)

    dataset = CityscapesDataset(
        root=cfg.dataset.data_root,
        split=split,
        transforms=transforms,
        gaussian_sigma=cfg.dataset.gaussian_sigma,
    )

    loader_cfg = cfg.dataset.dataloader.train if is_train else cfg.dataset.dataloader.val

    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle,
        num_workers=loader_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.get("drop_last", False),
    )


@hydra.main(version_base=None, config_path="../configs", config_name="panoptic")
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model.detector, resolve=True)
    model = build_model(model_cfg)

    # Build dataloaders
    train_loader = build_dataloader(cfg, split="train")
    val_loader = build_dataloader(cfg, split="val")

    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")

    # Build runner
    runner = Runner(
        model=model,
        optimizer_cfg=cfg.training.optimizer,
        scheduler_cfg=cfg.training.scheduler,
    )

    # Callbacks
    output_dir = Path(cfg.output.exp_dir)
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / "checkpoints",
            filename="{epoch}-{val/loss:.4f}",
            monitor="val/loss",
            mode="min",
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="step"),
    ]

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="logs",
    )

    # Trainer
    trainer_cfg = cfg.training.trainer
    trainer = pl.Trainer(
        max_epochs=trainer_cfg.max_epochs,
        accelerator=trainer_cfg.accelerator,
        devices=trainer_cfg.devices,
        precision=trainer_cfg.precision,
        gradient_clip_val=trainer_cfg.gradient_clip_val,
        accumulate_grad_batches=trainer_cfg.accumulate_grad_batches,
        log_every_n_steps=trainer_cfg.log_every_n_steps,
        callbacks=callbacks,
        logger=logger,
        deterministic=cfg.experiment.deterministic,
    )

    # Train
    trainer.fit(runner, train_loader, val_loader)


if __name__ == "__main__":
    main()
