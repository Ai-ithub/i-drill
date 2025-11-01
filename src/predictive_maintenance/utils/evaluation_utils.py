#!/usr/bin/env python3
"""Evaluation utilities for wellbore image generation system"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from scipy import linalg
from scipy.stats import entropy
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
import logging
from pathlib import Path
import json
from collections import defaultdict

class InceptionV3(nn.Module):
    """Inception V3 model for FID calculation"""
    
    def __init__(self, normalize_input: bool = False, requires_grad: bool = False):
        super(InceptionV3, self).__init__()
        
        try:
            from torchvision import models
            inception = models.inception_v3(pretrained=True)
        except ImportError:
            raise ImportError("torchvision is required for FID calculation")
        
        self.normalize_input = normalize_input
        self.requires_grad = requires_grad
        
        # Remove the final classification layer
        self.Conv2d_1a_3x3 = inception.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = inception.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = inception.Conv2d_2b_3x3
        self.maxpool1 = inception.maxpool1
        self.Conv2d_3b_1x1 = inception.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = inception.Conv2d_4a_3x3
        self.maxpool2 = inception.maxpool2
        self.Mixed_5b = inception.Mixed_5b
        self.Mixed_5c = inception.Mixed_5c
        self.Mixed_5d = inception.Mixed_5d
        self.Mixed_6a = inception.Mixed_6a
        self.Mixed_6b = inception.Mixed_6b
        self.Mixed_6c = inception.Mixed_6c
        self.Mixed_6d = inception.Mixed_6d
        self.Mixed_6e = inception.Mixed_6e
        self.Mixed_7a = inception.Mixed_7a
        self.Mixed_7b = inception.Mixed_7b
        self.Mixed_7c = inception.Mixed_7c
        self.avgpool = inception.avgpool
        
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    
    def forward(self, x):
        if self.normalize_input:
            x = x.clone()
            x[:, 0] = x[:, 0] * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x[:, 1] = x[:, 1] * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x[:, 2] = x[:, 2] * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
        
        # Forward pass through Inception layers
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)
        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        x = self.avgpool(x)
        
        return x.view(x.size(0), -1)

def calculate_fid_score(real_features: np.ndarray, fake_features: np.ndarray, 
                       eps: float = 1e-6) -> float:
    """Calculate Frechet Inception Distance (FID) score
    
    Args:
        real_features: Features from real images
        fake_features: Features from generated images
        eps: Small value for numerical stability
        
    Returns:
        FID score
    """
    # Calculate mean and covariance
    mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
    
    # Calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2) ** 2.0)
    
    # Calculate sqrt of product between covariances
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    
    # Check for imaginary numbers
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.absolute(covmean.imag))
            raise ValueError(f'Imaginary component {m}')
        covmean = covmean.real
    
    # Calculate FID
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

def extract_inception_features(images: torch.Tensor, model: InceptionV3, 
                             batch_size: int = 50, device: str = 'cuda') -> np.ndarray:
    """Extract features using Inception V3 model
    
    Args:
        images: Input images tensor
        model: Inception V3 model
        batch_size: Batch size for processing
        device: Device to run on
        
    Returns:
        Extracted features
    """
    model.eval()
    features = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            
            # Resize to 299x299 for Inception
            if batch.shape[-1] != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Ensure 3 channels
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            
            feat = model(batch)
            features.append(feat.cpu().numpy())
    
    return np.concatenate(features, axis=0)

def calculate_inception_score(images: torch.Tensor, model: InceptionV3, 
                            batch_size: int = 50, splits: int = 10, 
                            device: str = 'cuda') -> Tuple[float, float]:
    """Calculate Inception Score (IS)
    
    Args:
        images: Generated images
        model: Inception V3 model with classifier
        batch_size: Batch size for processing
        splits: Number of splits for calculation
        device: Device to run on
        
    Returns:
        Tuple of (mean IS, std IS)
    """
    # Load full Inception model with classifier
    try:
        from torchvision import models
        inception = models.inception_v3(pretrained=True).to(device)
        inception.eval()
    except ImportError:
        raise ImportError("torchvision is required for IS calculation")
    
    scores = []
    
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            
            # Resize to 299x299 for Inception
            if batch.shape[-1] != 299:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            
            # Ensure 3 channels
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            
            # Get predictions
            preds = F.softmax(inception(batch), dim=1)
            scores.append(preds.cpu().numpy())
    
    preds = np.concatenate(scores, axis=0)
    
    # Calculate IS for each split
    split_scores = []
    for k in range(splits):
        part = preds[k * (len(preds) // splits): (k + 1) * (len(preds) // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))
    
    return np.mean(split_scores), np.std(split_scores)

def calculate_lpips_score(real_images: torch.Tensor, fake_images: torch.Tensor,
                         net: str = 'alex', device: str = 'cuda') -> float:
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity) score
    
    Args:
        real_images: Real images tensor
        fake_images: Generated images tensor
        net: Network to use ('alex', 'vgg', 'squeeze')
        device: Device to run on
        
    Returns:
        LPIPS score
    """
    try:
        import lpips
    except ImportError:
        raise ImportError("lpips package is required. Install with: pip install lpips")
    
    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net=net).to(device)
    
    # Ensure same number of images
    min_len = min(len(real_images), len(fake_images))
    real_images = real_images[:min_len]
    fake_images = fake_images[:min_len]
    
    # Calculate LPIPS
    with torch.no_grad():
        lpips_scores = []
        for i in range(len(real_images)):
            real_img = real_images[i:i+1].to(device)
            fake_img = fake_images[i:i+1].to(device)
            
            # Ensure 3 channels
            if real_img.shape[1] == 1:
                real_img = real_img.repeat(1, 3, 1, 1)
            if fake_img.shape[1] == 1:
                fake_img = fake_img.repeat(1, 3, 1, 1)
            
            score = loss_fn(real_img, fake_img)
            lpips_scores.append(score.item())
    
    return np.mean(lpips_scores)

def calculate_ssim_score(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """Calculate Structural Similarity Index (SSIM) score
    
    Args:
        real_images: Real images array (N, H, W, C) or (N, H, W)
        fake_images: Generated images array (N, H, W, C) or (N, H, W)
        
    Returns:
        Average SSIM score
    """
    ssim_scores = []
    
    min_len = min(len(real_images), len(fake_images))
    
    for i in range(min_len):
        real_img = real_images[i]
        fake_img = fake_images[i]
        
        # Convert to grayscale if needed
        if len(real_img.shape) == 3 and real_img.shape[-1] == 3:
            real_img = cv2.cvtColor(real_img, cv2.COLOR_RGB2GRAY)
        if len(fake_img.shape) == 3 and fake_img.shape[-1] == 3:
            fake_img = cv2.cvtColor(fake_img, cv2.COLOR_RGB2GRAY)
        
        # Calculate SSIM
        score = ssim(real_img, fake_img, data_range=real_img.max() - real_img.min())
        ssim_scores.append(score)
    
    return np.mean(ssim_scores)

def calculate_psnr_score(real_images: np.ndarray, fake_images: np.ndarray) -> float:
    """Calculate Peak Signal-to-Noise Ratio (PSNR) score
    
    Args:
        real_images: Real images array
        fake_images: Generated images array
        
    Returns:
        Average PSNR score
    """
    psnr_scores = []
    
    min_len = min(len(real_images), len(fake_images))
    
    for i in range(min_len):
        real_img = real_images[i]
        fake_img = fake_images[i]
        
        # Calculate PSNR
        score = psnr(real_img, fake_img, data_range=real_img.max() - real_img.min())
        psnr_scores.append(score)
    
    return np.mean(psnr_scores)

def calculate_diversity_score(images: torch.Tensor, num_pairs: int = 1000) -> float:
    """Calculate diversity score by measuring pairwise distances
    
    Args:
        images: Generated images tensor
        num_pairs: Number of image pairs to sample
        
    Returns:
        Average pairwise distance (diversity score)
    """
    if len(images) < 2:
        return 0.0
    
    # Flatten images
    images_flat = images.view(len(images), -1)
    
    # Sample random pairs
    indices = torch.randperm(len(images))[:num_pairs * 2]
    pairs_a = images_flat[indices[:num_pairs]]
    pairs_b = images_flat[indices[num_pairs:num_pairs * 2]]
    
    # Calculate L2 distances
    distances = torch.norm(pairs_a - pairs_b, dim=1)
    
    return distances.mean().item()

def calculate_mode_collapse_score(generator: nn.Module, noise_dim: int, 
                                num_samples: int = 1000, 
                                threshold: float = 0.1,
                                device: str = 'cuda') -> float:
    """Calculate mode collapse score
    
    Args:
        generator: Generator model
        noise_dim: Noise dimension
        num_samples: Number of samples to generate
        threshold: Similarity threshold for mode collapse detection
        device: Device to run on
        
    Returns:
        Mode collapse score (0 = no collapse, 1 = complete collapse)
    """
    generator.eval()
    
    with torch.no_grad():
        # Generate samples
        noise = torch.randn(num_samples, noise_dim, device=device)
        generated = generator(noise)
        
        # Flatten images
        generated_flat = generated.view(num_samples, -1)
        
        # Calculate pairwise similarities
        similarities = torch.mm(generated_flat, generated_flat.t())
        similarities = similarities / (torch.norm(generated_flat, dim=1, keepdim=True) * 
                                     torch.norm(generated_flat, dim=1).unsqueeze(0))
        
        # Count similar pairs (excluding diagonal)
        mask = torch.eye(num_samples, device=device).bool()
        similarities.masked_fill_(mask, 0)
        
        similar_pairs = (similarities > threshold).sum().item()
        total_pairs = num_samples * (num_samples - 1)
        
        return similar_pairs / total_pairs

class EvaluationMetrics:
    """Comprehensive evaluation metrics calculator"""
    
    def __init__(self, device: str = 'cuda', cache_features: bool = True):
        self.device = device
        self.cache_features = cache_features
        self.inception_model = None
        self.cached_real_features = None
        
    def _get_inception_model(self):
        """Get or create Inception model"""
        if self.inception_model is None:
            self.inception_model = InceptionV3(normalize_input=True).to(self.device)
        return self.inception_model
    
    def calculate_all_metrics(self, real_images: torch.Tensor, 
                            fake_images: torch.Tensor,
                            generator: Optional[nn.Module] = None,
                            noise_dim: Optional[int] = None) -> Dict[str, float]:
        """Calculate all evaluation metrics
        
        Args:
            real_images: Real images tensor
            fake_images: Generated images tensor
            generator: Generator model (for mode collapse)
            noise_dim: Noise dimension (for mode collapse)
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # Get Inception model
        inception_model = self._get_inception_model()
        
        # Extract features for FID
        if self.cached_real_features is None or not self.cache_features:
            real_features = extract_inception_features(real_images, inception_model, device=self.device)
            if self.cache_features:
                self.cached_real_features = real_features
        else:
            real_features = self.cached_real_features
        
        fake_features = extract_inception_features(fake_images, inception_model, device=self.device)
        
        # Calculate FID
        try:
            metrics['fid'] = calculate_fid_score(real_features, fake_features)
        except Exception as e:
            logging.warning(f"Failed to calculate FID: {e}")
            metrics['fid'] = float('inf')
        
        # Calculate IS
        try:
            is_mean, is_std = calculate_inception_score(fake_images, inception_model, device=self.device)
            metrics['is_mean'] = is_mean
            metrics['is_std'] = is_std
        except Exception as e:
            logging.warning(f"Failed to calculate IS: {e}")
            metrics['is_mean'] = 0.0
            metrics['is_std'] = 0.0
        
        # Calculate LPIPS
        try:
            metrics['lpips'] = calculate_lpips_score(real_images, fake_images, device=self.device)
        except Exception as e:
            logging.warning(f"Failed to calculate LPIPS: {e}")
            metrics['lpips'] = float('inf')
        
        # Convert to numpy for SSIM/PSNR
        real_np = real_images.cpu().numpy().transpose(0, 2, 3, 1)
        fake_np = fake_images.cpu().numpy().transpose(0, 2, 3, 1)
        
        # Normalize to [0, 1]
        real_np = (real_np + 1) / 2
        fake_np = (fake_np + 1) / 2
        
        # Calculate SSIM
        try:
            metrics['ssim'] = calculate_ssim_score(real_np, fake_np)
        except Exception as e:
            logging.warning(f"Failed to calculate SSIM: {e}")
            metrics['ssim'] = 0.0
        
        # Calculate PSNR
        try:
            metrics['psnr'] = calculate_psnr_score(real_np, fake_np)
        except Exception as e:
            logging.warning(f"Failed to calculate PSNR: {e}")
            metrics['psnr'] = 0.0
        
        # Calculate diversity
        try:
            metrics['diversity'] = calculate_diversity_score(fake_images)
        except Exception as e:
            logging.warning(f"Failed to calculate diversity: {e}")
            metrics['diversity'] = 0.0
        
        # Calculate mode collapse (if generator provided)
        if generator is not None and noise_dim is not None:
            try:
                metrics['mode_collapse'] = calculate_mode_collapse_score(
                    generator, noise_dim, device=self.device
                )
            except Exception as e:
                logging.warning(f"Failed to calculate mode collapse: {e}")
                metrics['mode_collapse'] = 0.0
        
        return metrics
    
    def save_metrics(self, metrics: Dict[str, float], save_path: str):
        """Save metrics to JSON file
        
        Args:
            metrics: Dictionary of metrics
            save_path: Path to save metrics
        """
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def load_metrics(self, load_path: str) -> Dict[str, float]:
        """Load metrics from JSON file
        
        Args:
            load_path: Path to load metrics from
            
        Returns:
            Dictionary of metrics
        """
        with open(load_path, 'r') as f:
            return json.load(f)

def plot_metrics_comparison(metrics_dict: Dict[str, Dict[str, float]], 
                          save_path: Optional[str] = None):
    """Plot comparison of metrics across different models/epochs
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        save_path: Path to save plot (optional)
    """
    # Prepare data for plotting
    models = list(metrics_dict.keys())
    metric_names = list(next(iter(metrics_dict.values())).keys())
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metric_names[:6]):  # Plot first 6 metrics
        values = [metrics_dict[model].get(metric, 0) for model in models]
        
        axes[i].bar(models, values)
        axes[i].set_title(f'{metric.upper()}')
        axes[i].set_ylabel('Score')
        axes[i].tick_params(axis='x', rotation=45)
    
    # Remove unused subplots
    for i in range(len(metric_names), len(axes)):
        fig.delaxes(axes[i])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def create_evaluation_report(metrics: Dict[str, float], 
                           model_info: Dict[str, Any],
                           save_path: Optional[str] = None) -> str:
    """Create comprehensive evaluation report
    
    Args:
        metrics: Dictionary of evaluation metrics
        model_info: Dictionary with model information
        save_path: Path to save report (optional)
        
    Returns:
        Report string
    """
    report = []
    report.append("=" * 60)
    report.append("WELLBORE IMAGE GENERATION - EVALUATION REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Model information
    report.append("MODEL INFORMATION:")
    report.append("-" * 20)
    for key, value in model_info.items():
        report.append(f"{key}: {value}")
    report.append("")
    
    # Metrics
    report.append("EVALUATION METRICS:")
    report.append("-" * 20)
    
    # Quality metrics
    report.append("Image Quality:")
    if 'fid' in metrics:
        report.append(f"  FID Score: {metrics['fid']:.4f} (lower is better)")
    if 'ssim' in metrics:
        report.append(f"  SSIM Score: {metrics['ssim']:.4f} (higher is better)")
    if 'psnr' in metrics:
        report.append(f"  PSNR Score: {metrics['psnr']:.4f} (higher is better)")
    if 'lpips' in metrics:
        report.append(f"  LPIPS Score: {metrics['lpips']:.4f} (lower is better)")
    report.append("")
    
    # Diversity metrics
    report.append("Diversity and Mode Collapse:")
    if 'is_mean' in metrics:
        report.append(f"  Inception Score: {metrics['is_mean']:.4f} ± {metrics.get('is_std', 0):.4f}")
    if 'diversity' in metrics:
        report.append(f"  Diversity Score: {metrics['diversity']:.4f} (higher is better)")
    if 'mode_collapse' in metrics:
        report.append(f"  Mode Collapse Score: {metrics['mode_collapse']:.4f} (lower is better)")
    report.append("")
    
    # Overall assessment
    report.append("OVERALL ASSESSMENT:")
    report.append("-" * 20)
    
    # Simple scoring system
    score = 0
    total_metrics = 0
    
    if 'fid' in metrics and metrics['fid'] < float('inf'):
        score += max(0, 100 - metrics['fid'])  # Lower FID is better
        total_metrics += 1
    
    if 'ssim' in metrics:
        score += metrics['ssim'] * 100  # Higher SSIM is better
        total_metrics += 1
    
    if 'is_mean' in metrics:
        score += min(metrics['is_mean'] * 10, 100)  # Higher IS is better
        total_metrics += 1
    
    if total_metrics > 0:
        overall_score = score / total_metrics
        report.append(f"Overall Quality Score: {overall_score:.2f}/100")
        
        if overall_score >= 80:
            assessment = "Excellent"
        elif overall_score >= 60:
            assessment = "Good"
        elif overall_score >= 40:
            assessment = "Fair"
        else:
            assessment = "Poor"
        
        report.append(f"Quality Assessment: {assessment}")
    
    report.append("")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    if save_path:
        with open(save_path, 'w') as f:
            f.write(report_text)
    
    return report_text