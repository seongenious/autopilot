"""Test training pipeline with dummy model."""

import torch
from torch.utils.data import DataLoader, TensorDataset

from autopilot.engine.runner import Runner
from autopilot.models.dummy import DummyModel


def test_dummy_training():
    """Test that training pipeline works with dummy model."""
    # Create dummy data
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)

    def collate_fn(batch):
        imgs, lbls = zip(*batch)
        return {'image': torch.stack(imgs), 'label': torch.stack(lbls)}

    dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    # Create model and runner
    model = DummyModel(in_channels=3, num_classes=10)
    runner = Runner(model)

    # Test forward pass
    batch = next(iter(dataloader))
    loss = runner.training_step(batch, 0)
    assert loss is not None
    assert loss.requires_grad

    print(f"Test passed! Loss: {loss.item():.4f}")


if __name__ == '__main__':
    test_dummy_training()
