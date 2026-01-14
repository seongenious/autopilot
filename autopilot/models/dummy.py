"""Dummy model for testing training pipeline."""

from typing import Any, Dict

import torch
import torch.nn as nn

from autopilot.models.base import BaseModule


class DummyModel(BaseModule):
    """Simple dummy model for pipeline testing."""

    def __init__(self, in_channels: int = 3, num_classes: int = 10):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.flatten(1)
        x = self.fc(x)
        return x

    def train_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Training step."""
        images = batch['image']
        labels = batch['label']

        logits = self.forward(images)
        loss = self.loss_fn(logits, labels)

        return {'loss': loss}

    def val_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Validation step."""
        return self.train_step(batch)
