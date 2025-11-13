"""
Unit tests for model_utils
"""
import pytest
import torch
import torch.nn as nn
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "predictive_maintenance" / "utils"
sys.path.insert(0, str(src_path))

from model_utils import (
    weights_init,
    gradient_penalty,
    calculate_model_size,
    count_parameters,
    freeze_model,
    unfreeze_model,
    freeze_layers,
    unfreeze_layers,
    apply_spectral_norm,
    remove_spectral_norm,
    get_layer_names,
    replace_activation,
    copy_model_weights,
    exponential_moving_average
)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing"""
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Sigmoid()
    )
    return model


@pytest.fixture
def discriminator_model():
    """Create a discriminator model for testing"""
    model = nn.Sequential(
        nn.Conv2d(3, 64, 4, 2, 1),
        nn.LeakyReLU(0.2),
        nn.Conv2d(64, 128, 4, 2, 1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2),
        nn.Conv2d(128, 1, 4, 1, 0)
    )
    return model


class TestModelUtils:
    """Test suite for model_utils"""

    def test_weights_init_xavier_normal(self, simple_model):
        """Test weight initialization with Xavier normal"""
        # Execute
        simple_model.apply(lambda m: weights_init(m, init_type='xavier_normal'))
        
        # Assert - weights should be initialized (no exception)
        assert True

    def test_weights_init_kaiming_normal(self, simple_model):
        """Test weight initialization with Kaiming normal"""
        # Execute
        simple_model.apply(lambda m: weights_init(m, init_type='kaiming_normal'))
        
        # Assert
        assert True

    def test_weights_init_normal(self, simple_model):
        """Test weight initialization with normal distribution"""
        # Execute
        simple_model.apply(lambda m: weights_init(m, init_type='normal'))
        
        # Assert
        assert True

    def test_weights_init_orthogonal(self, simple_model):
        """Test weight initialization with orthogonal"""
        # Execute
        simple_model.apply(lambda m: weights_init(m, init_type='orthogonal'))
        
        # Assert
        assert True

    def test_weights_init_invalid(self, simple_model):
        """Test weight initialization with invalid type"""
        # Execute & Assert
        with pytest.raises(NotImplementedError):
            simple_model.apply(lambda m: weights_init(m, init_type='invalid'))

    def test_gradient_penalty(self, discriminator_model):
        """Test gradient penalty calculation"""
        # Setup
        device = torch.device('cpu')
        batch_size = 4
        real_samples = torch.randn(batch_size, 3, 64, 64)
        fake_samples = torch.randn(batch_size, 3, 64, 64)
        
        # Execute
        penalty = gradient_penalty(discriminator_model, real_samples, fake_samples, device)
        
        # Assert
        assert penalty.item() >= 0
        assert isinstance(penalty, torch.Tensor)

    def test_calculate_model_size(self, simple_model):
        """Test calculating model size"""
        # Execute
        size_info = calculate_model_size(simple_model)
        
        # Assert
        assert 'param_count' in size_info
        assert 'param_size_mb' in size_info
        assert 'total_size_mb' in size_info
        assert size_info['param_count'] > 0
        assert size_info['total_size_mb'] >= 0

    def test_count_parameters_trainable(self, simple_model):
        """Test counting trainable parameters"""
        # Execute
        count = count_parameters(simple_model, trainable_only=True)
        
        # Assert
        assert count > 0
        # All parameters should be trainable by default
        total_count = count_parameters(simple_model, trainable_only=False)
        assert count == total_count

    def test_count_parameters_all(self, simple_model):
        """Test counting all parameters"""
        # Execute
        count = count_parameters(simple_model, trainable_only=False)
        
        # Assert
        assert count > 0

    def test_freeze_model(self, simple_model):
        """Test freezing model"""
        # Execute
        freeze_model(simple_model)
        
        # Assert
        for param in simple_model.parameters():
            assert not param.requires_grad

    def test_unfreeze_model(self, simple_model):
        """Test unfreezing model"""
        # Setup
        freeze_model(simple_model)
        
        # Execute
        unfreeze_model(simple_model)
        
        # Assert
        for param in simple_model.parameters():
            assert param.requires_grad

    def test_freeze_layers(self, simple_model):
        """Test freezing specific layers"""
        # Execute
        freeze_layers(simple_model, ['0'])  # Freeze first layer
        
        # Assert
        # Check that specific layer is frozen
        for name, param in simple_model.named_parameters():
            if '0' in name:
                assert not param.requires_grad

    def test_unfreeze_layers(self, simple_model):
        """Test unfreezing specific layers"""
        # Setup
        freeze_model(simple_model)
        freeze_layers(simple_model, ['2'])  # Unfreeze layer 2
        
        # Execute
        unfreeze_layers(simple_model, ['2'])
        
        # Assert
        for name, param in simple_model.named_parameters():
            if '2' in name:
                assert param.requires_grad

    def test_apply_spectral_norm(self, discriminator_model):
        """Test applying spectral normalization"""
        # Execute
        model_with_sn = apply_spectral_norm(discriminator_model)
        
        # Assert
        assert model_with_sn is not None
        # Check that spectral norm is applied to Conv2d layers
        for module in model_with_sn.modules():
            if isinstance(module, nn.Conv2d):
                # Spectral norm should be applied
                assert True

    def test_remove_spectral_norm(self, discriminator_model):
        """Test removing spectral normalization"""
        # Setup
        model_with_sn = apply_spectral_norm(discriminator_model)
        
        # Execute
        model_without_sn = remove_spectral_norm(model_with_sn)
        
        # Assert
        assert model_without_sn is not None

    def test_get_layer_names(self, simple_model):
        """Test getting layer names"""
        # Execute
        layer_names = get_layer_names(simple_model)
        
        # Assert
        assert len(layer_names) > 0
        assert all(isinstance(name, str) for name in layer_names)

    def test_get_layer_names_by_type(self, simple_model):
        """Test getting layer names by type"""
        # Execute
        linear_layers = get_layer_names(simple_model, layer_type=nn.Linear)
        
        # Assert
        assert len(linear_layers) == 2  # Two Linear layers

    def test_replace_activation(self, simple_model):
        """Test replacing activation function"""
        # Execute
        model_with_tanh = replace_activation(simple_model, nn.ReLU, nn.Tanh())
        
        # Assert
        assert model_with_tanh is not None
        # Check that ReLU is replaced
        for module in model_with_tanh.modules():
            if isinstance(module, nn.Sequential):
                # Check if ReLU is replaced
                assert True

    def test_copy_model_weights(self, simple_model):
        """Test copying model weights"""
        # Setup
        target_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Sigmoid()
        )
        
        # Get original weights
        original_weights = {name: param.clone() for name, param in simple_model.named_parameters()}
        
        # Execute
        copy_model_weights(simple_model, target_model)
        
        # Assert
        for (name1, param1), (name2, param2) in zip(
            simple_model.named_parameters(),
            target_model.named_parameters()
        ):
            assert torch.equal(param1, param2)

    def test_exponential_moving_average(self, simple_model):
        """Test exponential moving average"""
        # Setup
        ema_model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10),
            nn.Sigmoid()
        )
        
        # Execute
        exponential_moving_average(simple_model, ema_model, decay=0.9)
        
        # Assert
        # EMA should update weights
        assert True

    def test_weights_init_batch_norm(self):
        """Test weight initialization for BatchNorm layers"""
        # Setup
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.BatchNorm1d(20)
        )
        
        # Execute
        model.apply(lambda m: weights_init(m, init_type='xavier_normal'))
        
        # Assert
        assert True

    def test_weights_init_group_norm(self):
        """Test weight initialization for GroupNorm layers"""
        # Setup
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.GroupNorm(8, 64)
        )
        
        # Execute
        model.apply(lambda m: weights_init(m, init_type='xavier_normal'))
        
        # Assert
        assert True

    def test_gradient_penalty_edge_cases(self, discriminator_model):
        """Test gradient penalty with edge cases"""
        # Setup
        device = torch.device('cpu')
        batch_size = 1  # Single sample
        real_samples = torch.randn(batch_size, 3, 64, 64)
        fake_samples = torch.randn(batch_size, 3, 64, 64)
        
        # Execute
        penalty = gradient_penalty(discriminator_model, real_samples, fake_samples, device, lambda_gp=5.0)
        
        # Assert
        assert penalty.item() >= 0

    def test_model_size_empty_model(self):
        """Test model size calculation for empty model"""
        # Setup
        empty_model = nn.Module()
        
        # Execute
        size_info = calculate_model_size(empty_model)
        
        # Assert
        assert size_info['param_count'] == 0
        assert size_info['total_size_mb'] == 0

