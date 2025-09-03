"""Utility functions for GAN-based wellbore image generation"""

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
from typing import List, Tuple, Optional

def weights_init(m):
    """Initialize network weights using Xavier initialization"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def save_image_grid(images: torch.Tensor, path: str, nrow: int = 8, normalize: bool = True):
    """Save a grid of images to file"""
    from torchvision.utils import save_image
    save_image(images, path, nrow=nrow, normalize=normalize, padding=2)

def create_noise(batch_size: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    """Create random noise tensor for generator input"""
    return torch.randn(batch_size, latent_dim, device=device)

def gradient_penalty(discriminator, real_samples: torch.Tensor, 
                    fake_samples: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Calculate gradient penalty for WGAN-GP"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    d_interpolated = discriminator(interpolated)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    
    return gradient_penalty

def calculate_fid_score(real_features: np.ndarray, fake_features: np.ndarray) -> float:
    """Calculate Frechet Inception Distance (FID) score"""
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu1 - mu2
    covmean = scipy.linalg.sqrtm(sigma1.dot(sigma2))
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def plot_training_curves(g_losses: List[float], d_losses: List[float], 
                        save_path: str = "training_curves.png"):
    """Plot and save training loss curves"""
    plt.figure(figsize=(10, 5))
    plt.plot(g_losses, label='Generator Loss')
    plt.plot(d_losses, label='Discriminator Loss')
    plt.xlabel('Iteration')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Loss Curves')
    plt.savefig(save_path)
    plt.close()

def ensure_dir(path: str):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)

def load_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, 
                   checkpoint_path: str) -> Tuple[int, float]:
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return epoch, loss

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer,
                   epoch: int, loss: float, checkpoint_path: str):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)