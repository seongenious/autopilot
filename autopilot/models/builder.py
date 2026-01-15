"""Builder functions for model components."""

from autopilot.utils.registry import (
    BACKBONES,
    ENCODERS,
    HEADS,
    LOSSES,
    MODELS,
    NECKS,
)


def build_backbone(cfg):
    """Build backbone from config."""
    return BACKBONES.build(cfg)


def build_neck(cfg):
    """Build neck from config."""
    return NECKS.build(cfg)


def build_encoder(cfg):
    """Build encoder from config."""
    return ENCODERS.build(cfg)


def build_head(cfg):
    """Build head from config."""
    return HEADS.build(cfg)


def build_loss(cfg):
    """Build loss from config."""
    return LOSSES.build(cfg)


def build_model(cfg):
    """Build model from config."""
    return MODELS.build(cfg)
