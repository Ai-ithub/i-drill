#!/usr/bin/env python3
"""Image processing utilities for wellbore image generation"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2
from typing import Tuple, Optional, Union, List
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8, 
                   normalize: bool = True, padding: int = 2,
                   pad_value: float = 0.0) -> None:
    """Save a grid of images to file
    
    Args:
        images: Tensor of shape (N, C, H, W)
        path: Output file path
        nrow: Number of images per row
        normalize: Whether to normalize images to [0, 1]
        padding: Padding between images
        pad_value: Value for padding pixels
    """
    save_image(images, path, nrow=nrow, normalize=normalize, 
              padding=padding, pad_value=pad_value)

def create_noise(batch_size: int, latent_dim: int, device: torch.device,
                distribution: str = 'normal') -> torch.Tensor:
    """Create random noise tensor for generator input
    
    Args:
        batch_size: Number of samples
        latent_dim: Dimension of latent space
        device: Device to create tensor on
        distribution: Type of noise distribution ('normal', 'uniform')
        
    Returns:
        Random noise tensor of shape (batch_size, latent_dim)
    """
    if distribution == 'normal':
        return torch.randn(batch_size, latent_dim, device=device)
    elif distribution == 'uniform':
        return torch.rand(batch_size, latent_dim, device=device) * 2 - 1
    else:
        raise ValueError(f"Unknown distribution: {distribution}")

def denormalize_image(tensor: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5],
                     std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """Denormalize image tensor from [-1, 1] to [0, 1]
    
    Args:
        tensor: Normalized image tensor
        mean: Mean values used for normalization
        std: Standard deviation values used for normalization
        
    Returns:
        Denormalized image tensor
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    return tensor * std + mean

def normalize_image(tensor: torch.Tensor, mean: List[float] = [0.5, 0.5, 0.5],
                   std: List[float] = [0.5, 0.5, 0.5]) -> torch.Tensor:
    """Normalize image tensor from [0, 1] to [-1, 1]
    
    Args:
        tensor: Image tensor in [0, 1] range
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image tensor
    """
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    
    mean = torch.tensor(mean, device=tensor.device).view(1, -1, 1, 1)
    std = torch.tensor(std, device=tensor.device).view(1, -1, 1, 1)
    
    return (tensor - mean) / std

def resize_image(image: torch.Tensor, size: Tuple[int, int], 
                mode: str = 'bilinear') -> torch.Tensor:
    """Resize image tensor
    
    Args:
        image: Input image tensor
        size: Target size (height, width)
        mode: Interpolation mode
        
    Returns:
        Resized image tensor
    """
    return F.interpolate(image, size=size, mode=mode, align_corners=False)

def apply_clahe(image: np.ndarray, clip_limit: float = 2.0, 
               tile_grid_size: Tuple[int, int] = (8, 8)) -> np.ndarray:
    """Apply Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image: Input image as numpy array
        clip_limit: Clipping limit for contrast enhancement
        tile_grid_size: Size of the neighborhood for local enhancement
        
    Returns:
        Enhanced image
    """
    if len(image.shape) == 3:
        # Convert to LAB color space for better results
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

def detect_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, float]:
    """Detect if image is blurry using Laplacian variance
    
    Args:
        image: Input image as numpy array
        threshold: Blur threshold (lower values indicate more blur)
        
    Returns:
        Tuple of (is_blurry, blur_score)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
    
    # Calculate Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    return laplacian_var < threshold, laplacian_var

def enhance_image(image: Union[torch.Tensor, np.ndarray], 
                 enhance_contrast: bool = True,
                 enhance_sharpness: bool = True,
                 enhance_brightness: bool = False,
                 contrast_factor: float = 1.2,
                 sharpness_factor: float = 1.1,
                 brightness_factor: float = 1.0) -> Union[torch.Tensor, np.ndarray]:
    """Enhance image quality using PIL filters
    
    Args:
        image: Input image (tensor or numpy array)
        enhance_contrast: Whether to enhance contrast
        enhance_sharpness: Whether to enhance sharpness
        enhance_brightness: Whether to enhance brightness
        contrast_factor: Contrast enhancement factor
        sharpness_factor: Sharpness enhancement factor
        brightness_factor: Brightness enhancement factor
        
    Returns:
        Enhanced image in same format as input
    """
    is_tensor = isinstance(image, torch.Tensor)
    
    if is_tensor:
        # Convert tensor to PIL Image
        if image.dim() == 4:
            image = image.squeeze(0)
        
        # Denormalize if needed
        if image.min() < 0:
            image = (image + 1) / 2
        
        image_np = image.permute(1, 2, 0).cpu().numpy()
        image_np = (image_np * 255).astype(np.uint8)
        pil_image = Image.fromarray(image_np)
    else:
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        pil_image = Image.fromarray(image)
    
    # Apply enhancements
    if enhance_contrast:
        enhancer = ImageEnhance.Contrast(pil_image)
        pil_image = enhancer.enhance(contrast_factor)
    
    if enhance_sharpness:
        enhancer = ImageEnhance.Sharpness(pil_image)
        pil_image = enhancer.enhance(sharpness_factor)
    
    if enhance_brightness:
        enhancer = ImageEnhance.Brightness(pil_image)
        pil_image = enhancer.enhance(brightness_factor)
    
    # Convert back to original format
    if is_tensor:
        enhanced_np = np.array(pil_image).astype(np.float32) / 255.0
        enhanced_tensor = torch.from_numpy(enhanced_np).permute(2, 0, 1)
        
        # Renormalize if original was in [-1, 1]
        if image.min() < 0:
            enhanced_tensor = enhanced_tensor * 2 - 1
        
        return enhanced_tensor.unsqueeze(0)
    else:
        return np.array(pil_image)

def create_image_montage(images: List[torch.Tensor], titles: Optional[List[str]] = None,
                        nrow: int = 4, figsize: Tuple[int, int] = (12, 8)) -> np.ndarray:
    """Create a montage of images with optional titles
    
    Args:
        images: List of image tensors
        titles: Optional list of titles for each image
        nrow: Number of images per row
        figsize: Figure size for the montage
        
    Returns:
        Montage as numpy array
    """
    import matplotlib.pyplot as plt
    
    num_images = len(images)
    ncol = (num_images + nrow - 1) // nrow
    
    fig, axes = plt.subplots(ncol, nrow, figsize=figsize)
    if ncol == 1:
        axes = axes.reshape(1, -1)
    elif nrow == 1:
        axes = axes.reshape(-1, 1)
    
    for i, image in enumerate(images):
        row = i // nrow
        col = i % nrow
        
        # Convert tensor to numpy
        if image.dim() == 4:
            image = image.squeeze(0)
        
        if image.min() < 0:
            image = (image + 1) / 2
        
        image_np = image.permute(1, 2, 0).cpu().numpy()
        
        axes[row, col].imshow(image_np)
        axes[row, col].axis('off')
        
        if titles and i < len(titles):
            axes[row, col].set_title(titles[i])
    
    # Hide unused subplots
    for i in range(num_images, nrow * ncol):
        row = i // nrow
        col = i % nrow
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # Convert figure to numpy array
    fig.canvas.draw()
    montage = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    montage = montage.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    return montage

def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert tensor to PIL Image
    
    Args:
        tensor: Image tensor of shape (C, H, W) or (1, C, H, W)
        
    Returns:
        PIL Image
    """
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Denormalize if needed
    if tensor.min() < 0:
        tensor = (tensor + 1) / 2
    
    # Clamp values
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    image_np = tensor.permute(1, 2, 0).cpu().numpy()
    image_np = (image_np * 255).astype(np.uint8)
    
    return Image.fromarray(image_np)

def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert PIL Image to tensor
    
    Args:
        image: PIL Image
        normalize: Whether to normalize to [-1, 1] range
        
    Returns:
        Image tensor of shape (1, C, H, W)
    """
    transform_list = [transforms.ToTensor()]
    
    if normalize:
        transform_list.append(transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    
    transform = transforms.Compose(transform_list)
    tensor = transform(image)
    
    return tensor.unsqueeze(0)

def calculate_image_statistics(images: torch.Tensor) -> dict:
    """Calculate statistics for a batch of images
    
    Args:
        images: Batch of images (N, C, H, W)
        
    Returns:
        Dictionary with image statistics
    """
    stats = {
        'mean': images.mean().item(),
        'std': images.std().item(),
        'min': images.min().item(),
        'max': images.max().item(),
        'shape': list(images.shape)
    }
    
    # Per-channel statistics
    if images.dim() == 4 and images.size(1) > 1:
        stats['channel_means'] = images.mean(dim=[0, 2, 3]).tolist()
        stats['channel_stds'] = images.std(dim=[0, 2, 3]).tolist()
    
    return stats

def apply_random_crop(image: torch.Tensor, crop_size: Tuple[int, int]) -> torch.Tensor:
    """Apply random crop to image tensor
    
    Args:
        image: Input image tensor (C, H, W) or (N, C, H, W)
        crop_size: Target crop size (height, width)
        
    Returns:
        Randomly cropped image
    """
    if image.dim() == 3:
        image = image.unsqueeze(0)
        squeeze_output = True
    else:
        squeeze_output = False
    
    _, _, h, w = image.shape
    crop_h, crop_w = crop_size
    
    if h < crop_h or w < crop_w:
        # Pad if image is smaller than crop size
        pad_h = max(0, crop_h - h)
        pad_w = max(0, crop_w - w)
        image = F.pad(image, (0, pad_w, 0, pad_h), mode='reflect')
        _, _, h, w = image.shape
    
    # Random crop
    top = torch.randint(0, h - crop_h + 1, (1,)).item()
    left = torch.randint(0, w - crop_w + 1, (1,)).item()
    
    cropped = image[:, :, top:top + crop_h, left:left + crop_w]
    
    if squeeze_output:
        cropped = cropped.squeeze(0)
    
    return cropped

def create_circular_mask(size: Tuple[int, int], center: Optional[Tuple[int, int]] = None,
                        radius: Optional[int] = None) -> torch.Tensor:
    """Create a circular mask
    
    Args:
        size: Mask size (height, width)
        center: Center of the circle (y, x). If None, uses image center
        radius: Radius of the circle. If None, uses half of minimum dimension
        
    Returns:
        Binary mask tensor
    """
    h, w = size
    
    if center is None:
        center = (h // 2, w // 2)
    
    if radius is None:
        radius = min(h, w) // 2
    
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    
    # Calculate distance from center
    distance = torch.sqrt((x - center[1]) ** 2 + (y - center[0]) ** 2)
    
    # Create mask
    mask = (distance <= radius).float()
    
    return mask