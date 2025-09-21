#!/usr/bin/env python3
"""Evaluation metrics for generated wellbore images"""

import os
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
from scipy import linalg
from torchvision import transforms
from torchvision.models import inception_v3

from config import GANConfig
from gan.generator import StyleGAN2Generator
from data import WellboreImageDataset
from utils import load_checkpoint, ensure_dir, format_time

class InceptionV3Features(nn.Module):
    """InceptionV3 model for feature extraction"""
    
    def __init__(self, normalize_input: bool = True, require_grad: bool = False):
        super().__init__()
        
        # Load pretrained InceptionV3
        self.inception = inception_v3(pretrained=True, transform_input=False)
        self.inception.fc = nn.Identity()  # Remove final classification layer
        
        # Set to evaluation mode
        self.inception.eval()
        
        # Freeze parameters if not requiring gradients
        if not require_grad:
            for param in self.inception.parameters():
                param.requires_grad = False
        
        self.normalize_input = normalize_input
        
        # Normalization for ImageNet pretrained models
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input images"""
        # Ensure input is in [0, 1] range
        if x.min() < 0:
            x = (x + 1) / 2  # Convert from [-1, 1] to [0, 1]
        
        # Resize to 299x299 for InceptionV3
        if x.shape[-1] != 299:
            x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=False)
        
        # Normalize for ImageNet
        if self.normalize_input:
            x = (x - self.mean) / self.std
        
        # Extract features
        features = self.inception(x)
        
        return features

class FIDCalculator:
    """Fréchet Inception Distance (FID) calculator"""
    
    def __init__(self, device: torch.device):
        self.device = device
        self.inception_model = InceptionV3Features().to(device)
    
    def calculate_activation_statistics(self, images: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of inception features"""
        self.inception_model.eval()
        
        activations = []
        
        with torch.no_grad():
            for i in range(0, images.size(0), 32):  # Process in batches
                batch = images[i:i+32].to(self.device)
                features = self.inception_model(batch)
                activations.append(features.cpu().numpy())
        
        activations = np.concatenate(activations, axis=0)
        
        # Calculate statistics
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        
        return mu, sigma
    
    def calculate_fid(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Calculate FID between real and fake images"""
        logging.info("Calculating FID score...")
        
        # Calculate statistics for real and fake images
        mu_real, sigma_real = self.calculate_activation_statistics(real_images)
        mu_fake, sigma_fake = self.calculate_activation_statistics(fake_images)
        
        # Calculate FID
        fid_score = self._calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return fid_score
    
    def _calculate_frechet_distance(self, mu1: np.ndarray, sigma1: np.ndarray,
                                   mu2: np.ndarray, sigma2: np.ndarray) -> float:
        """Calculate Fréchet distance between two multivariate Gaussians"""
        # Calculate difference in means
        diff = mu1 - mu2
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical errors
        if not np.isfinite(covmean).all():
            logging.warning("FID calculation resulted in non-finite values")
            offset = np.eye(sigma1.shape[0]) * 1e-6
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Calculate FID
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.absolute(covmean.imag))
                raise ValueError(f"Imaginary component {m}")
            covmean = covmean.real
        
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return float(fid)

class ISCalculator:
    """Inception Score (IS) calculator"""
    
    def __init__(self, device: torch.device):
        self.device = device
        # Use full InceptionV3 with classification head for IS
        self.inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception_model.eval()
        
        # Freeze parameters
        for param in self.inception_model.parameters():
            param.requires_grad = False
    
    def calculate_is(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """Calculate Inception Score"""
        logging.info("Calculating Inception Score...")
        
        # Get predictions
        predictions = self._get_predictions(images)
        
        # Calculate IS for each split
        scores = []
        split_size = predictions.shape[0] // splits
        
        for i in range(splits):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size if i < splits - 1 else predictions.shape[0]
            
            split_predictions = predictions[start_idx:end_idx]
            score = self._calculate_is_for_split(split_predictions)
            scores.append(score)
        
        # Return mean and std
        return float(np.mean(scores)), float(np.std(scores))
    
    def _get_predictions(self, images: torch.Tensor) -> np.ndarray:
        """Get InceptionV3 predictions for images"""
        self.inception_model.eval()
        
        predictions = []
        
        with torch.no_grad():
            for i in range(0, images.size(0), 32):  # Process in batches
                batch = images[i:i+32].to(self.device)
                
                # Ensure input is in [0, 1] range
                if batch.min() < 0:
                    batch = (batch + 1) / 2
                
                # Resize to 299x299 for InceptionV3
                if batch.shape[-1] != 299:
                    batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
                
                # Normalize for ImageNet
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
                batch = (batch - mean) / std
                
                # Get predictions
                logits = self.inception_model(batch)
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu().numpy())
        
        return np.concatenate(predictions, axis=0)
    
    def _calculate_is_for_split(self, predictions: np.ndarray) -> float:
        """Calculate IS for a single split"""
        # Calculate marginal distribution
        marginal = np.mean(predictions, axis=0)
        
        # Calculate KL divergence for each sample
        kl_divs = []
        for i in range(predictions.shape[0]):
            p = predictions[i]
            # Avoid log(0) by adding small epsilon
            kl_div = np.sum(p * np.log(p / (marginal + 1e-16) + 1e-16))
            kl_divs.append(kl_div)
        
        # Return exponential of mean KL divergence
        return np.exp(np.mean(kl_divs))

class LPIPSCalculator:
    """Learned Perceptual Image Patch Similarity (LPIPS) calculator"""
    
    def __init__(self, device: torch.device, net: str = 'alex'):
        self.device = device
        
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net=net).to(device)
            self.available = True
        except ImportError:
            logging.warning("LPIPS not available. Install with: pip install lpips")
            self.available = False
    
    def calculate_lpips(self, images1: torch.Tensor, images2: torch.Tensor) -> float:
        """Calculate LPIPS between two sets of images"""
        if not self.available:
            logging.warning("LPIPS not available, returning 0.0")
            return 0.0
        
        logging.info("Calculating LPIPS score...")
        
        distances = []
        
        with torch.no_grad():
            for i in range(min(images1.size(0), images2.size(0))):
                img1 = images1[i:i+1].to(self.device)
                img2 = images2[i:i+1].to(self.device)
                
                # Ensure images are in [-1, 1] range
                if img1.max() > 1:
                    img1 = img1 * 2 - 1
                if img2.max() > 1:
                    img2 = img2 * 2 - 1
                
                distance = self.lpips_model(img1, img2)
                distances.append(distance.item())
        
        return float(np.mean(distances))

class ModelEvaluator:
    """Comprehensive model evaluation"""
    
    def __init__(self, config: GANConfig, device: torch.device):
        self.config = config
        self.device = device
        
        # Initialize calculators
        self.fid_calculator = FIDCalculator(device)
        self.is_calculator = ISCalculator(device)
        self.lpips_calculator = LPIPSCalculator(device)
    
    def evaluate_model(self, generator: nn.Module, real_dataloader: DataLoader,
                      num_samples: int = 5000) -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        logging.info(f"Starting model evaluation with {num_samples} samples...")
        start_time = time.time()
        
        # Generate fake images
        fake_images = self._generate_samples(generator, num_samples)
        
        # Collect real images
        real_images = self._collect_real_samples(real_dataloader, num_samples)
        
        # Calculate metrics
        results = {}
        
        # FID Score
        try:
            fid_score = self.fid_calculator.calculate_fid(real_images, fake_images)
            results['fid'] = fid_score
            logging.info(f"FID Score: {fid_score:.4f}")
        except Exception as e:
            logging.error(f"Error calculating FID: {e}")
            results['fid'] = None
        
        # Inception Score
        try:
            is_mean, is_std = self.is_calculator.calculate_is(fake_images)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
            logging.info(f"Inception Score: {is_mean:.4f} ± {is_std:.4f}")
        except Exception as e:
            logging.error(f"Error calculating IS: {e}")
            results['is_mean'] = None
            results['is_std'] = None
        
        # LPIPS Score
        try:
            # Sample subset for LPIPS (computationally expensive)
            lpips_samples = min(1000, num_samples)
            lpips_score = self.lpips_calculator.calculate_lpips(
                real_images[:lpips_samples], 
                fake_images[:lpips_samples]
            )
            results['lpips'] = lpips_score
            logging.info(f"LPIPS Score: {lpips_score:.4f}")
        except Exception as e:
            logging.error(f"Error calculating LPIPS: {e}")
            results['lpips'] = None
        
        # Additional metrics
        results.update(self._calculate_additional_metrics(real_images, fake_images))
        
        total_time = time.time() - start_time
        results['evaluation_time'] = total_time
        
        logging.info(f"Evaluation completed in {format_time(total_time)}")
        
        return results
    
    def _generate_samples(self, generator: nn.Module, num_samples: int) -> torch.Tensor:
        """Generate samples from the generator"""
        logging.info(f"Generating {num_samples} samples...")
        
        generator.eval()
        samples = []
        
        with torch.no_grad():
            for i in range(0, num_samples, 32):
                batch_size = min(32, num_samples - i)
                
                # Generate random latent vectors
                latent = torch.randn(batch_size, self.config.LATENT_DIM, device=self.device)
                
                # Generate images
                fake_images = generator(latent)
                samples.append(fake_images.cpu())
                
                if (i + batch_size) % 1000 == 0:
                    logging.info(f"Generated {i + batch_size}/{num_samples} samples")
        
        return torch.cat(samples, dim=0)
    
    def _collect_real_samples(self, dataloader: DataLoader, num_samples: int) -> torch.Tensor:
        """Collect real samples from dataloader"""
        logging.info(f"Collecting {num_samples} real samples...")
        
        samples = []
        collected = 0
        
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                batch = batch[0]  # Handle datasets that return (image, label)
            
            samples.append(batch)
            collected += batch.size(0)
            
            if collected >= num_samples:
                break
        
        all_samples = torch.cat(samples, dim=0)[:num_samples]
        logging.info(f"Collected {all_samples.size(0)} real samples")
        
        return all_samples
    
    def _calculate_additional_metrics(self, real_images: torch.Tensor, 
                                    fake_images: torch.Tensor) -> Dict[str, float]:
        """Calculate additional evaluation metrics"""
        metrics = {}
        
        # Convert to numpy for calculations
        real_np = real_images.numpy()
        fake_np = fake_images.numpy()
        
        # Pixel-wise statistics
        metrics['real_mean'] = float(np.mean(real_np))
        metrics['fake_mean'] = float(np.mean(fake_np))
        metrics['real_std'] = float(np.std(real_np))
        metrics['fake_std'] = float(np.std(fake_np))
        
        # Mean Squared Error
        if real_np.shape == fake_np.shape:
            metrics['mse'] = float(np.mean((real_np - fake_np) ** 2))
        
        return metrics

def evaluate_model(config: GANConfig, checkpoint_path: str, output_dir: str,
                  num_samples: int = 5000) -> None:
    """Main evaluation function"""
    logging.info("Starting model evaluation...")
    
    # Setup device
    device = torch.device(config.DEVICE)
    
    # Load generator
    generator = StyleGAN2Generator(
        latent_dim=config.LATENT_DIM,
        image_size=config.IMAGE_SIZE,
        num_channels=config.NUM_CHANNELS,
        num_layers=config.NUM_LAYERS,
        feature_maps=config.FEATURE_MAPS
    ).to(device)
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    generator.load_state_dict(checkpoint['generator'])
    
    # Create real data loader
    real_dataset = WellboreImageDataset(
        data_path=config.DATA_PATH,
        image_size=config.IMAGE_SIZE,
        augment=False  # No augmentation for evaluation
    )
    
    real_dataloader = DataLoader(
        real_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Initialize evaluator
    evaluator = ModelEvaluator(config, device)
    
    # Run evaluation
    results = evaluator.evaluate_model(generator, real_dataloader, num_samples)
    
    # Save results
    ensure_dir(output_dir)
    results_file = os.path.join(output_dir, 'evaluation_results.txt')
    
    with open(results_file, 'w') as f:
        f.write("Model Evaluation Results\n")
        f.write("=" * 50 + "\n\n")
        
        for key, value in results.items():
            if value is not None:
                if isinstance(value, float):
                    f.write(f"{key}: {value:.6f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            else:
                f.write(f"{key}: N/A\n")
    
    logging.info(f"Evaluation results saved to: {results_file}")
    
    # Print summary
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    
    if results.get('fid') is not None:
        print(f"FID Score: {results['fid']:.4f}")
    
    if results.get('is_mean') is not None:
        print(f"Inception Score: {results['is_mean']:.4f} ± {results['is_std']:.4f}")
    
    if results.get('lpips') is not None:
        print(f"LPIPS Score: {results['lpips']:.4f}")
    
    print("="*50)

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate StyleGAN2 model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Output directory')
    parser.add_argument('--num-samples', type=int, default=5000, help='Number of samples for evaluation')
    
    args = parser.parse_args()
    
    # Load config
    config = GANConfig.from_yaml(args.config)
    
    # Run evaluation
    evaluate_model(
        config=config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        num_samples=args.num_samples
    )