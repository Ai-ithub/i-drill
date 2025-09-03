#!/usr/bin/env python3
"""Training utilities for wellbore image generation system"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import (
    StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, 
    ReduceLROnPlateau, CyclicLR, OneCycleLR
)
import numpy as np
import random
import logging
import os
import time
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path

class AverageMeter:
    """Computes and stores the average and current value"""
    
    def __init__(self, name: str, fmt: str = ':f'):
        self.name = name
        self.fmt = fmt
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
    
    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter:
    """Progress meter for training"""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int, logger: Optional[logging.Logger] = None):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        message = '\t'.join(entries)
        
        if logger:
            logger.info(message)
        else:
            print(message)
    
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def save_checkpoint(state: Dict[str, Any], is_best: bool, checkpoint_dir: str,
                   filename: str = 'checkpoint.pth', best_filename: str = 'best_model.pth'):
    """Save model checkpoint
    
    Args:
        state: Dictionary containing model state and metadata
        is_best: Whether this is the best model so far
        checkpoint_dir: Directory to save checkpoints
        filename: Checkpoint filename
        best_filename: Best model filename
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, filename)
    torch.save(state, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(checkpoint_dir, best_filename)
        torch.save(state, best_path)

def load_checkpoint(checkpoint_path: str, model: nn.Module, 
                   optimizer: Optional[optim.Optimizer] = None,
                   scheduler: Optional[Any] = None,
                   device: str = 'cuda') -> Dict[str, Any]:
    """Load model checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load state into
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        device: Device to load checkpoint on
        
    Returns:
        Dictionary with checkpoint metadata
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load optimizer state if provided
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # Load scheduler state if provided
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return {
        'epoch': checkpoint.get('epoch', 0),
        'iteration': checkpoint.get('iteration', 0),
        'best_score': checkpoint.get('best_score', float('inf')),
        'loss': checkpoint.get('loss', 0.0)
    }

def setup_logging(log_dir: str, log_level: str = 'INFO', 
                 log_file: str = 'training.log') -> logging.Logger:
    """Setup logging configuration
    
    Args:
        log_dir: Directory to save log files
        log_level: Logging level
        log_file: Log filename
        
    Returns:
        Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('wellbore_gan')
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file))
    file_handler.setLevel(getattr(logging, log_level.upper()))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger

def get_device(device_id: Optional[int] = None, force_cpu: bool = False) -> torch.device:
    """Get appropriate device for training
    
    Args:
        device_id: Specific GPU device ID (None for auto-select)
        force_cpu: Force CPU usage
        
    Returns:
        PyTorch device
    """
    if force_cpu or not torch.cuda.is_available():
        return torch.device('cpu')
    
    if device_id is not None:
        if device_id >= torch.cuda.device_count():
            raise ValueError(f"Device ID {device_id} not available. Only {torch.cuda.device_count()} GPUs found.")
        return torch.device(f'cuda:{device_id}')
    
    return torch.device('cuda')

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make CuDNN deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_optimizer(model: nn.Module, optimizer_config: Dict[str, Any]) -> optim.Optimizer:
    """Create optimizer from configuration
    
    Args:
        model: Model to optimize
        optimizer_config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    optimizer_type = optimizer_config.get('type', 'adam').lower()
    lr = optimizer_config.get('lr', 0.0002)
    weight_decay = optimizer_config.get('weight_decay', 0.0)
    
    if optimizer_type == 'adam':
        betas = optimizer_config.get('betas', (0.5, 0.999))
        return optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    elif optimizer_type == 'adamw':
        betas = optimizer_config.get('betas', (0.9, 0.999))
        return optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
    
    elif optimizer_type == 'sgd':
        momentum = optimizer_config.get('momentum', 0.9)
        return optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    
    elif optimizer_type == 'rmsprop':
        alpha = optimizer_config.get('alpha', 0.99)
        momentum = optimizer_config.get('momentum', 0.0)
        return optim.RMSprop(model.parameters(), lr=lr, alpha=alpha, 
                           momentum=momentum, weight_decay=weight_decay)
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")

def create_scheduler(optimizer: optim.Optimizer, scheduler_config: Dict[str, Any]) -> Optional[Any]:
    """Create learning rate scheduler from configuration
    
    Args:
        optimizer: Optimizer to schedule
        scheduler_config: Scheduler configuration
        
    Returns:
        Configured scheduler or None
    """
    if not scheduler_config.get('enabled', False):
        return None
    
    scheduler_type = scheduler_config.get('type', 'step').lower()
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        return StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = scheduler_config.get('milestones', [30, 60, 90])
        gamma = scheduler_config.get('gamma', 0.1)
        return MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        return ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', 100)
        eta_min = scheduler_config.get('eta_min', 0)
        return CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'plateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 10)
        return ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience)
    
    elif scheduler_type == 'cyclic':
        base_lr = scheduler_config.get('base_lr', 1e-5)
        max_lr = scheduler_config.get('max_lr', 1e-3)
        step_size_up = scheduler_config.get('step_size_up', 2000)
        return CyclicLR(optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up)
    
    elif scheduler_type == 'onecycle':
        max_lr = scheduler_config.get('max_lr', 1e-3)
        total_steps = scheduler_config.get('total_steps', 1000)
        return OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

class EarlyStopping:
    """Early stopping utility"""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0, 
                 mode: str = 'min', restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' for loss, 'max' for accuracy
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.restore_best_weights = restore_best_weights
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
        if mode == 'min':
            self.monitor_op = np.less
            self.min_delta *= -1
        else:
            self.monitor_op = np.greater
    
    def __call__(self, score: float, model: nn.Module) -> bool:
        """Check if training should stop
        
        Args:
            score: Current validation score
            model: Model to potentially save weights from
            
        Returns:
            True if training should stop
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif self.monitor_op(score, self.best_score + self.min_delta):
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            if self.restore_best_weights and self.best_weights is not None:
                model.load_state_dict(self.best_weights)
            return True
        
        return False
    
    def save_checkpoint(self, model: nn.Module):
        """Save model weights"""
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()

class GradientClipping:
    """Gradient clipping utility"""
    
    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type
    
    def __call__(self, model: nn.Module) -> float:
        """Clip gradients and return gradient norm
        
        Args:
            model: Model to clip gradients for
            
        Returns:
            Gradient norm before clipping
        """
        return torch.nn.utils.clip_grad_norm_(model.parameters(), 
                                            self.max_norm, self.norm_type)

class WarmupScheduler:
    """Learning rate warmup scheduler"""
    
    def __init__(self, optimizer: optim.Optimizer, warmup_steps: int, 
                 base_lr: float, target_lr: float):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        self.current_step = 0
    
    def step(self):
        """Update learning rate"""
        if self.current_step < self.warmup_steps:
            lr = self.base_lr + (self.target_lr - self.base_lr) * \
                 (self.current_step / self.warmup_steps)
            
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        
        self.current_step += 1

def calculate_gradient_norm(model: nn.Module, norm_type: float = 2.0) -> float:
    """Calculate gradient norm for a model
    
    Args:
        model: Model to calculate gradient norm for
        norm_type: Type of norm to calculate
        
    Returns:
        Gradient norm
    """
    total_norm = 0.0
    
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
    
    return total_norm ** (1.0 / norm_type)

def get_learning_rate(optimizer: optim.Optimizer) -> float:
    """Get current learning rate from optimizer
    
    Args:
        optimizer: Optimizer to get learning rate from
        
    Returns:
        Current learning rate
    """
    return optimizer.param_groups[0]['lr']

def adjust_learning_rate(optimizer: optim.Optimizer, lr: float):
    """Adjust learning rate of optimizer
    
    Args:
        optimizer: Optimizer to adjust
        lr: New learning rate
    """
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Timer:
    """Simple timer utility"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.start_time = time.time()
    
    def elapsed(self) -> float:
        return time.time() - self.start_time
    
    def elapsed_str(self) -> str:
        elapsed = self.elapsed()
        hours = int(elapsed // 3600)
        minutes = int((elapsed % 3600) // 60)
        seconds = int(elapsed % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"

def estimate_remaining_time(current_epoch: int, total_epochs: int, 
                          epoch_time: float) -> str:
    """Estimate remaining training time
    
    Args:
        current_epoch: Current epoch number
        total_epochs: Total number of epochs
        epoch_time: Average time per epoch
        
    Returns:
        Formatted remaining time string
    """
    remaining_epochs = total_epochs - current_epoch
    remaining_seconds = remaining_epochs * epoch_time
    return format_time(remaining_seconds)