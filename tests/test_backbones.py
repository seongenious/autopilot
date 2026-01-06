"""Tests for backbone modules."""

import pytest
import torch


class TestRegNetBackbone:
    """Test cases for RegNet backbone."""

    @pytest.fixture
    def backbone(self):
        """Create RegNet backbone for testing."""
        from autopilot.models.backbones import RegNetBackbone

        return RegNetBackbone(
            model_name='regnetx_002',  # Smallest model for fast testing
            pretrained=False,
            out_indices=(1, 2, 3),
        )

    def test_forward_shape(self, backbone):
        """Test forward pass output shapes."""
        x = torch.randn(2, 3, 256, 384)
        features = backbone(x)

        assert len(features) == 3
        for feat in features:
            assert feat.shape[0] == 2  # Batch size

    def test_out_channels(self, backbone):
        """Test output channels property."""
        assert len(backbone.out_channels) == 3
        assert all(isinstance(c, int) for c in backbone.out_channels)

    def test_frozen_stages(self):
        """Test stage freezing."""
        from autopilot.models.backbones import RegNetBackbone

        backbone = RegNetBackbone(
            model_name='regnetx_002',
            pretrained=False,
            frozen_stages=1,
        )

        # Check stem is frozen
        for param in backbone.model.stem.parameters():
            assert not param.requires_grad


class TestBiFPN:
    """Test cases for BiFPN neck."""

    @pytest.fixture
    def bifpn(self):
        """Create BiFPN for testing."""
        from autopilot.models.backbones import BiFPN

        return BiFPN(
            in_channels=[24, 56, 152],  # Match regnetx_002 outputs
            out_channels=64,
            num_levels=4,
            num_layers=2,
        )

    def test_forward_shape(self, bifpn):
        """Test forward pass output shapes."""
        # Simulate backbone outputs
        features = [
            torch.randn(2, 24, 64, 96),
            torch.randn(2, 56, 32, 48),
            torch.randn(2, 152, 16, 24),
        ]

        outputs = bifpn(features)

        assert len(outputs) == 4  # num_levels
        for out in outputs:
            assert out.shape[1] == 64  # out_channels
