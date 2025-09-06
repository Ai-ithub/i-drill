#!/usr/bin/env python3
"""Utility functions for StyleGAN2 wellbore image generation"""

import os
import time
import logging
import random
from pathlib import Path
from typing import Dict, Any, Optional, List

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def setup_logging(level: int = logging.INFO, log_file: Optional[str] = None) -> None:
    """Setup logging configuration"""
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    
    # Setup file handler if specified
    if log_file:
        ensure_dir(os.path.dirname(log_file))
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def ensure_dir(directory: str) -> None:
    """Ensure directory exists, create if not"""
    Path(directory).mkdir(parents=True, exist_ok=True)

def set_random_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(device_name: str = 'auto') -> torch.device:
    """Get appropriate device for computation"""
    if device_name == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device_name)
    
    if device.type == 'cuda' and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available, falling back to CPU")
        device = torch.device('cpu')
    
    return device

def save_checkpoint(state: Dict[str, Any], filepath: str) -> None:
    """Save model checkpoint"""
    ensure_dir(os.path.dirname(filepath))
    torch.save(state, filepath)
    logging.info(f"Checkpoint saved: {filepath}")

def load_checkpoint(filepath: str) -> Dict[str, Any]:
    """Load model checkpoint"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Checkpoint not found: {filepath}")
    
    checkpoint = torch.load(filepath, map_location='cpu')
    logging.info(f"Checkpoint loaded: {filepath}")
    return checkpoint

def count_parameters(model: torch.nn.Module) -> int:
    """Count total number of parameters in a model"""
    return sum(p.numel() for p in model.parameters())

def count_trainable_parameters(model: torch.nn.Module) -> int:
    """Count number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def format_time(seconds: float) -> str:
    """Format time in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_number(num: int) -> str:
    """Format large numbers with appropriate suffixes"""
    if num < 1000:
        return str(num)
    elif num < 1000000:
        return f"{num/1000:.1f}K"
    elif num < 1000000000:
        return f"{num/1000000:.1f}M"
    else:
        return f"{num/1000000000:.1f}B"

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
    """Display progress during training"""
    
    def __init__(self, num_batches: int, meters: List[AverageMeter], prefix: str = ""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info('\t'.join(entries))
    
    def _get_batch_fmtstr(self, num_batches: int):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

class Timer:
    """Simple timer context manager"""
    
    def __init__(self, name: str = "Timer"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        logging.info(f"{self.name}: {format_time(elapsed)}")
    
    @property
    def elapsed(self) -> float:
        if self.start_time is None:
            return 0.0
        end_time = self.end_time if self.end_time else time.time()
        return end_time - self.start_time

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image"""
    # Ensure tensor is on CPU and detached
    tensor = tensor.detach().cpu()
    
    # Handle batch dimension
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Convert from [-1, 1] to [0, 1] if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp values
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    if tensor.shape[0] == 1:  # Grayscale
        array = tensor.squeeze(0).numpy()
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode='L')
    else:  # RGB
        array = tensor.permute(1, 2, 0).numpy()
        array = (array * 255).astype(np.uint8)
        return Image.fromarray(array, mode='RGB')

def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to tensor"""
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array
    array = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    
    # Normalize to [-1, 1] if requested
    if normalize:
        tensor = tensor * 2 - 1
    
    return tensor

def create_image_grid(images: List[torch.Tensor], nrow: int = 8, 
                     padding: int = 2, normalize: bool = True) -> torch.Tensor:
    """Create image grid from list of tensors"""
    from torchvision.utils import make_grid
    
    # Stack images
    if isinstance(images, list):
        images = torch.stack(images)
    
    # Create grid
    grid = make_grid(images, nrow=nrow, padding=padding, normalize=normalize)
    
    return grid

def save_image_comparison(real_images: torch.Tensor, fake_images: torch.Tensor,
                         filepath: str, nrow: int = 8) -> None:
    """Save comparison between real and fake images"""
    from torchvision.utils import save_image, make_grid
    
    # Ensure same number of images
    min_size = min(real_images.size(0), fake_images.size(0))
    real_images = real_images[:min_size]
    fake_images = fake_images[:min_size]
    
    # Create grids
    real_grid = make_grid(real_images, nrow=nrow, normalize=True, padding=2)
    fake_grid = make_grid(fake_images, nrow=nrow, normalize=True, padding=2)
    
    # Combine grids vertically
    combined = torch.cat([real_grid, fake_grid], dim=1)
    
    # Save
    save_image(combined, filepath)

def plot_training_curves(losses: Dict[str, List[float]], save_path: str) -> None:
    """Plot training loss curves"""
    plt.figure(figsize=(12, 4))
    
    for i, (name, values) in enumerate(losses.items()):
        plt.subplot(1, len(losses), i + 1)
        plt.plot(values)
        plt.title(f'{name} Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_gradient_penalty(discriminator: torch.nn.Module, real_images: torch.Tensor,
                             fake_images: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_images.size(0)
    
    # Random weight term for interpolation
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    # Get random interpolation between real and fake images
    interpolated = alpha * real_images + (1 - alpha) * fake_images
    interpolated.requires_grad_(True)
    
    # Calculate discriminator output for interpolated images
    d_interpolated = discriminator(interpolated)
    
    # Calculate gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Calculate gradient penalty
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage"""
    import psutil
    
    # System memory
    memory = psutil.virtual_memory()
    
    result = {
        'system_total_gb': memory.total / (1024**3),
        'system_used_gb': memory.used / (1024**3),
        'system_available_gb': memory.available / (1024**3),
        'system_percent': memory.percent
    }
    
    # GPU memory if available
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_memory_stats()
        result.update({
            'gpu_allocated_gb': torch.cuda.memory_allocated() / (1024**3),
            'gpu_reserved_gb': torch.cuda.memory_reserved() / (1024**3),
            'gpu_max_allocated_gb': torch.cuda.max_memory_allocated() / (1024**3)
        })
    
    return result

def log_memory_usage(prefix: str = "") -> None:
    """Log current memory usage"""
    memory_info = get_memory_usage()
    
    log_msg = f"{prefix}Memory Usage - "
    log_msg += f"System: {memory_info['system_used_gb']:.1f}GB/{memory_info['system_total_gb']:.1f}GB ({memory_info['system_percent']:.1f}%)"
    
    if 'gpu_allocated_gb' in memory_info:
        log_msg += f", GPU: {memory_info['gpu_allocated_gb']:.1f}GB allocated, {memory_info['gpu_reserved_gb']:.1f}GB reserved"
    
    logging.info(log_msg)

def cleanup_memory() -> None:
    """Clean up memory"""
    import gc
    
    # Python garbage collection
    gc.collect()
    
    # CUDA memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def validate_config(config) -> None:
    """Validate configuration parameters"""
    required_attrs = [
        'IMAGE_SIZE', 'BATCH_SIZE', 'LATENT_DIM', 'NUM_CHANNELS',
        'LEARNING_RATE_G', 'LEARNING_RATE_D', 'NUM_EPOCHS'
    ]
    
    for attr in required_attrs:
        if not hasattr(config, attr):
            raise ValueError(f"Missing required configuration: {attr}")
    
    # Validate image size is power of 2
    if config.IMAGE_SIZE & (config.IMAGE_SIZE - 1) != 0:
        raise ValueError(f"Image size must be power of 2, got {config.IMAGE_SIZE}")
    
    # Validate positive values
    positive_attrs = ['IMAGE_SIZE', 'BATCH_SIZE', 'LATENT_DIM', 'NUM_CHANNELS', 'NUM_EPOCHS']
    for attr in positive_attrs:
        if getattr(config, attr) <= 0:
            raise ValueError(f"{attr} must be positive, got {getattr(config, attr)}")
    
    logging.info("Configuration validation passed")