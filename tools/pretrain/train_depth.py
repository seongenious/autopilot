"""Depth estimation training script.

Trains depth estimation with log-bin classification on Cityscapes.
"""

from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from autopilot.datasets import CityscapesDataset, collate_fn
from autopilot.datasets.transforms.image import (
    Compose,
    Normalize,
    RandomCrop,
    RandomHorizontalFlip,
)
from autopilot.engine import HookCallback, Runner, build_hooks
from autopilot.engine.hooks import VisualizationHook  # noqa: F401 - register hook
from autopilot.models.builder import build_model

# Import to register modules
import autopilot.models.backbones  # noqa: F401
import autopilot.models.detectors  # noqa: F401
import autopilot.models.heads.perception2d  # noqa: F401
import autopilot.models.necks  # noqa: F401


def count_parameters(module: nn.Module) -> int:
    """Count trainable parameters in a module."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def print_model_params(model: nn.Module) -> None:
    """Print parameter counts by component."""
    print('\n' + '=' * 60)
    print('Model Parameters')
    print('=' * 60)

    total = 0
    components = [
        ('backbone', model.backbone),
        ('neck', model.neck),
        ('context', model.context),
        ('head', model.head),
    ]

    for name, module in components:
        params = count_parameters(module)
        total += params
        print(f'  {name:20s}: {params:>12,d} ({params / 1e6:.2f}M)')

    print('-' * 60)
    print(f"  {'Total':20s}: {total:>12,d} ({total / 1e6:.2f}M)")
    print('=' * 60 + '\n')


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


def build_dataset(cfg: DictConfig, split: str = 'train'):
    """Build dataset from config."""
    is_train = split == 'train'
    transforms = build_transforms(cfg, is_train=is_train)

    return CityscapesDataset(
        root=cfg.dataset.data_root,
        split=split,
        transforms=transforms,
        gaussian_sigma=cfg.dataset.gaussian_sigma,
        min_depth=cfg.dataset.min_depth,
        max_depth=cfg.dataset.max_depth,
        num_bins=cfg.dataset.num_bins,
        sigma_ratio=cfg.dataset.sigma_ratio,
    )


def build_dataloader(cfg: DictConfig, dataset, split: str = 'train'):
    """Build dataloader from config."""
    is_train = split == 'train'
    loader_cfg = cfg.dataset.dataloader.train if is_train else cfg.dataset.dataloader.val

    return DataLoader(
        dataset,
        batch_size=loader_cfg.batch_size,
        shuffle=loader_cfg.shuffle,
        num_workers=loader_cfg.num_workers,
        collate_fn=collate_fn,
        pin_memory=loader_cfg.pin_memory,
        drop_last=loader_cfg.get('drop_last', False),
    )


@hydra.main(version_base=None, config_path='../../configs', config_name='pretrain/depth')
def main(cfg: DictConfig):
    """Main training function."""
    # Print config
    print(OmegaConf.to_yaml(cfg))

    # Set seed
    pl.seed_everything(cfg.experiment.seed, workers=True)

    # Build model
    model_cfg = OmegaConf.to_container(cfg.model.detector, resolve=True)
    model = build_model(model_cfg)

    # Print parameter counts
    print_model_params(model)

    # Build datasets and dataloaders
    train_dataset = build_dataset(cfg, split='train')
    val_dataset = build_dataset(cfg, split='val')
    train_loader = build_dataloader(cfg, train_dataset, split='train')
    val_loader = build_dataloader(cfg, val_dataset, split='val')

    print(f'Train samples: {len(train_dataset)}')
    print(f'Val samples: {len(val_dataset)}')

    # Build runner
    runner = Runner(
        model=model,
        optimizer_cfg=cfg.training.optimizer,
        scheduler_cfg=cfg.training.scheduler,
    )

    # Build hooks from config
    output_dir = Path(cfg.output.exp_dir)
    hook_cfgs = OmegaConf.to_container(cfg.get('custom_hooks', []), resolve=True)

    # Inject output_dir into hooks that need it
    for hook_cfg in hook_cfgs:
        if 'output_dir' not in hook_cfg:
            hook_cfg['output_dir'] = str(output_dir)

    hooks = build_hooks(hook_cfgs)

    # Set val_dataset for VisualizationHook
    for hook in hooks:
        if hasattr(hook, 'set_dataset'):
            hook.set_dataset(val_dataset)

    # Callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=output_dir / 'checkpoints',
            filename='{epoch}-{val/loss:.4f}',
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            save_last=True,
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Add hook callback if hooks exist
    if hooks:
        callbacks.append(HookCallback(hooks))

    # Logger
    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name='logs',
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


if __name__ == '__main__':
    main()
