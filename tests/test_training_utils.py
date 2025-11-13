"""
Unit tests for training_utils
"""
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from unittest.mock import Mock, patch

import sys
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
src_path = project_root / "src" / "predictive_maintenance" / "utils"
sys.path.insert(0, str(src_path))

from training_utils import (
    AverageMeter,
    ProgressMeter,
    save_checkpoint,
    load_checkpoint,
    setup_logging,
    get_device,
    set_seed,
    create_optimizer,
    create_scheduler,
    EarlyStopping,
    GradientClipping,
    WarmupScheduler,
    calculate_gradient_norm,
    get_learning_rate,
    adjust_learning_rate,
    Timer,
    format_time,
    estimate_remaining_time
)


@pytest.fixture
def simple_model():
    """Create a simple PyTorch model for testing"""
    return nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10)
    )


@pytest.fixture
def optimizer(simple_model):
    """Create an optimizer for testing"""
    return optim.Adam(simple_model.parameters(), lr=0.001)


class TestTrainingUtils:
    """Test suite for training_utils"""

    def test_average_meter_reset(self):
        """Test AverageMeter reset"""
        # Setup
        meter = AverageMeter('test')
        meter.update(10.0)
        
        # Execute
        meter.reset()
        
        # Assert
        assert meter.val == 0
        assert meter.avg == 0
        assert meter.sum == 0
        assert meter.count == 0

    def test_average_meter_update(self):
        """Test AverageMeter update"""
        # Setup
        meter = AverageMeter('test')
        
        # Execute
        meter.update(10.0)
        meter.update(20.0)
        
        # Assert
        assert meter.val == 20.0
        assert meter.avg == 15.0
        assert meter.sum == 30.0
        assert meter.count == 2

    def test_average_meter_update_multiple(self):
        """Test AverageMeter update with multiple values"""
        # Setup
        meter = AverageMeter('test')
        
        # Execute
        meter.update(10.0, n=5)
        
        # Assert
        assert meter.val == 10.0
        assert meter.avg == 10.0
        assert meter.sum == 50.0
        assert meter.count == 5

    def test_progress_meter_display(self):
        """Test ProgressMeter display"""
        # Setup
        meters = [AverageMeter('loss'), AverageMeter('acc')]
        progress = ProgressMeter(100, meters, prefix='Test')
        
        # Execute
        # Should not raise exception
        progress.display(50)
        
        # Assert
        assert True

    def test_save_checkpoint(self, simple_model, optimizer, tmp_path):
        """Test saving checkpoint"""
        # Setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        
        state = {
            'epoch': 10,
            'model_state_dict': simple_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5
        }
        
        # Execute
        save_checkpoint(state, is_best=True, checkpoint_dir=str(checkpoint_dir))
        
        # Assert
        assert (checkpoint_dir / "checkpoint.pth").exists()
        assert (checkpoint_dir / "best_model.pth").exists()

    def test_load_checkpoint(self, simple_model, optimizer, tmp_path):
        """Test loading checkpoint"""
        # Setup
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint_path = checkpoint_dir / "checkpoint.pth"
        
        state = {
            'epoch': 10,
            'model_state_dict': simple_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': 0.5
        }
        torch.save(state, checkpoint_path)
        
        # Execute
        metadata = load_checkpoint(str(checkpoint_path), simple_model, optimizer, device='cpu')
        
        # Assert
        assert metadata['epoch'] == 10
        assert metadata['loss'] == 0.5

    def test_setup_logging(self, tmp_path):
        """Test setting up logging"""
        # Setup
        log_dir = tmp_path / "logs"
        
        # Execute
        logger = setup_logging(str(log_dir), log_level='INFO', log_file='test.log')
        
        # Assert
        assert logger is not None
        assert logger.name == 'wellbore_gan'
        assert (log_dir / "test.log").exists()

    def test_get_device_cpu(self):
        """Test getting CPU device"""
        # Setup
        with patch('torch.cuda.is_available', return_value=False):
            # Execute
            device = get_device(force_cpu=True)
            
            # Assert
            assert device.type == 'cpu'

    def test_get_device_cuda(self):
        """Test getting CUDA device"""
        # Setup
        with patch('torch.cuda.is_available', return_value=True):
            with patch('torch.cuda.device_count', return_value=1):
                # Execute
                device = get_device(force_cpu=False)
                
                # Assert
                assert device.type == 'cuda'

    def test_set_seed(self):
        """Test setting random seed"""
        # Execute
        set_seed(42)
        
        # Assert
        # Seed should be set (no exception)
        assert True

    def test_create_optimizer_adam(self, simple_model):
        """Test creating Adam optimizer"""
        # Setup
        config = {'type': 'adam', 'lr': 0.001}
        
        # Execute
        opt = create_optimizer(simple_model, config)
        
        # Assert
        assert isinstance(opt, optim.Adam)
        assert opt.param_groups[0]['lr'] == 0.001

    def test_create_optimizer_sgd(self, simple_model):
        """Test creating SGD optimizer"""
        # Setup
        config = {'type': 'sgd', 'lr': 0.01, 'momentum': 0.9}
        
        # Execute
        opt = create_optimizer(simple_model, config)
        
        # Assert
        assert isinstance(opt, optim.SGD)
        assert opt.param_groups[0]['lr'] == 0.01
        assert opt.param_groups[0]['momentum'] == 0.9

    def test_create_optimizer_rmsprop(self, simple_model):
        """Test creating RMSprop optimizer"""
        # Setup
        config = {'type': 'rmsprop', 'lr': 0.001}
        
        # Execute
        opt = create_optimizer(simple_model, config)
        
        # Assert
        assert isinstance(opt, optim.RMSprop)

    def test_create_scheduler_step(self, optimizer):
        """Test creating StepLR scheduler"""
        # Setup
        config = {'enabled': True, 'type': 'step', 'step_size': 30, 'gamma': 0.1}
        
        # Execute
        scheduler = create_scheduler(optimizer, config)
        
        # Assert
        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler.StepLR)

    def test_create_scheduler_cosine(self, optimizer):
        """Test creating CosineAnnealingLR scheduler"""
        # Setup
        config = {'enabled': True, 'type': 'cosine', 'T_max': 100}
        
        # Execute
        scheduler = create_scheduler(optimizer, config)
        
        # Assert
        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR)

    def test_create_scheduler_disabled(self, optimizer):
        """Test creating scheduler when disabled"""
        # Setup
        config = {'enabled': False}
        
        # Execute
        scheduler = create_scheduler(optimizer, config)
        
        # Assert
        assert scheduler is None

    def test_early_stopping_min_mode(self, simple_model):
        """Test EarlyStopping in min mode"""
        # Setup
        early_stop = EarlyStopping(patience=3, min_delta=0.001, mode='min')
        
        # Execute
        should_stop = early_stop(0.5, simple_model)  # Good score
        should_stop2 = early_stop(0.6, simple_model)  # Worse score
        should_stop3 = early_stop(0.7, simple_model)  # Worse score
        should_stop4 = early_stop(0.8, simple_model)  # Worse score
        
        # Assert
        assert not should_stop
        assert not should_stop2
        assert not should_stop3
        assert should_stop4  # Should stop after patience

    def test_early_stopping_max_mode(self, simple_model):
        """Test EarlyStopping in max mode"""
        # Setup
        early_stop = EarlyStopping(patience=2, mode='max')
        
        # Execute
        should_stop1 = early_stop(0.8, simple_model)  # Good score
        should_stop2 = early_stop(0.7, simple_model)  # Worse score
        should_stop3 = early_stop(0.6, simple_model)  # Worse score
        
        # Assert
        assert not should_stop1
        assert not should_stop2
        assert should_stop3  # Should stop after patience

    def test_gradient_clipping(self, simple_model):
        """Test gradient clipping"""
        # Setup
        clip = GradientClipping(max_norm=1.0)
        
        # Create some gradients
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param)
        
        # Execute
        grad_norm = clip(simple_model)
        
        # Assert
        assert grad_norm is not None
        assert grad_norm >= 0

    def test_warmup_scheduler(self, optimizer):
        """Test WarmupScheduler"""
        # Setup
        warmup = WarmupScheduler(optimizer, warmup_steps=10, base_lr=0.001, target_lr=0.01)
        
        # Execute
        for i in range(15):
            warmup.step()
        
        # Assert
        # After warmup, LR should be target_lr
        assert get_learning_rate(optimizer) == 0.01

    def test_calculate_gradient_norm(self, simple_model):
        """Test calculating gradient norm"""
        # Setup
        for param in simple_model.parameters():
            param.grad = torch.randn_like(param)
        
        # Execute
        grad_norm = calculate_gradient_norm(simple_model)
        
        # Assert
        assert grad_norm >= 0

    def test_get_learning_rate(self, optimizer):
        """Test getting learning rate"""
        # Execute
        lr = get_learning_rate(optimizer)
        
        # Assert
        assert lr == 0.001

    def test_adjust_learning_rate(self, optimizer):
        """Test adjusting learning rate"""
        # Execute
        adjust_learning_rate(optimizer, 0.01)
        
        # Assert
        assert get_learning_rate(optimizer) == 0.01

    def test_timer(self):
        """Test Timer utility"""
        # Setup
        timer = Timer()
        
        # Execute
        import time
        time.sleep(0.1)
        elapsed = timer.elapsed()
        elapsed_str = timer.elapsed_str()
        
        # Assert
        assert elapsed >= 0.1
        assert isinstance(elapsed_str, str)

    def test_format_time_seconds(self):
        """Test formatting time in seconds"""
        # Execute
        result = format_time(45)
        
        # Assert
        assert "45s" in result or "45" in result

    def test_format_time_minutes(self):
        """Test formatting time in minutes"""
        # Execute
        result = format_time(125)
        
        # Assert
        assert "m" in result

    def test_format_time_hours(self):
        """Test formatting time in hours"""
        # Execute
        result = format_time(3665)
        
        # Assert
        assert "h" in result

    def test_estimate_remaining_time(self):
        """Test estimating remaining time"""
        # Execute
        result = estimate_remaining_time(current_epoch=50, total_epochs=100, epoch_time=60.0)
        
        # Assert
        assert isinstance(result, str)
        assert "h" in result or "m" in result

    def test_early_stopping_restore_best_weights(self, simple_model):
        """Test EarlyStopping with weight restoration"""
        # Setup
        early_stop = EarlyStopping(patience=2, restore_best_weights=True, mode='min')
        
        # Execute
        early_stop(0.5, simple_model)  # Best score
        early_stop(0.7, simple_model)  # Worse
        early_stop(0.8, simple_model)  # Worse - should stop
        
        # Assert
        # Best weights should be restored
        assert early_stop.best_weights is not None

    def test_create_scheduler_plateau(self, optimizer):
        """Test creating ReduceLROnPlateau scheduler"""
        # Setup
        config = {'enabled': True, 'type': 'plateau', 'mode': 'min', 'factor': 0.1, 'patience': 10}
        
        # Execute
        scheduler = create_scheduler(optimizer, config)
        
        # Assert
        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    def test_create_scheduler_cyclic(self, optimizer):
        """Test creating CyclicLR scheduler"""
        # Setup
        config = {'enabled': True, 'type': 'cyclic', 'base_lr': 1e-5, 'max_lr': 1e-3, 'step_size_up': 2000}
        
        # Execute
        scheduler = create_scheduler(optimizer, config)
        
        # Assert
        assert scheduler is not None
        assert isinstance(scheduler, optim.lr_scheduler.CyclicLR)

    def test_create_optimizer_with_weight_decay(self, simple_model):
        """Test creating optimizer with weight decay"""
        # Setup
        config = {'type': 'adam', 'lr': 0.001, 'weight_decay': 0.0001}
        
        # Execute
        opt = create_optimizer(simple_model, config)
        
        # Assert
        assert opt.param_groups[0]['weight_decay'] == 0.0001

