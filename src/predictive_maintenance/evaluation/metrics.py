"""Evaluation metrics for GAN-generated wellbore images"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from scipy import linalg
from typing import Tuple, List, Optional, Union
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips
from torchvision import models, transforms
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class InceptionV3FeatureExtractor(nn.Module):
    """InceptionV3 model for feature extraction (FID and IS calculation)"""
    
    def __init__(self, resize_input=True, normalize_input=True):
        super().__init__()
        
        # Load pre-trained InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False)
        inception.eval()
        
        # Remove the final classification layer
        self.inception = nn.Sequential(*list(inception.children())[:-1])
        
        self.resize_input = resize_input
        self.normalize_input = normalize_input
        
        # Preprocessing transforms
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)) if resize_input else transforms.Lambda(lambda x: x),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) if normalize_input else transforms.Lambda(lambda x: x)
        ])
    
    def forward(self, x):
        """Extract features from input images"""
        # Ensure input is in correct format
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        # Convert grayscale to RGB if needed
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # Preprocess
        if self.resize_input or self.normalize_input:
            x = torch.stack([self.preprocess(img) for img in x])
        
        # Extract features
        with torch.no_grad():
            features = self.inception(x)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = features.view(features.size(0), -1)
        
        return features

class FIDCalculator:
    """Fréchet Inception Distance (FID) calculator"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.inception_model = InceptionV3FeatureExtractor().to(device)
        self.inception_model.eval()
    
    def extract_features(self, images: torch.Tensor, batch_size: int = 50) -> np.ndarray:
        """Extract InceptionV3 features from images"""
        features_list = []
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            
            with torch.no_grad():
                batch_features = self.inception_model(batch)
                features_list.append(batch_features.cpu().numpy())
        
        return np.concatenate(features_list, axis=0)
    
    def calculate_statistics(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate mean and covariance of features"""
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma
    
    def calculate_fid(self, mu1: np.ndarray, sigma1: np.ndarray, 
                     mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6) -> float:
        """Calculate FID score between two distributions"""
        # Calculate squared difference of means
        diff = mu1 - mu2
        
        # Calculate sqrt of product of covariances
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        # Handle numerical issues
        if not np.isfinite(covmean).all():
            msg = ('FID calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
        
        # Numerical precision issues can cause tiny imaginary components
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real
        
        # Calculate FID
        tr_covmean = np.trace(covmean)
        fid = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
        
        return float(fid)
    
    def __call__(self, real_images: torch.Tensor, fake_images: torch.Tensor) -> float:
        """Calculate FID between real and fake images"""
        # Extract features
        real_features = self.extract_features(real_images)
        fake_features = self.extract_features(fake_images)
        
        # Calculate statistics
        mu_real, sigma_real = self.calculate_statistics(real_features)
        mu_fake, sigma_fake = self.calculate_statistics(fake_features)
        
        # Calculate FID
        fid_score = self.calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
        
        return fid_score

class InceptionScoreCalculator:
    """Inception Score (IS) calculator"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Load pre-trained InceptionV3 for classification
        self.inception_model = models.inception_v3(pretrained=True, transform_input=False)
        self.inception_model.eval()
        self.inception_model.to(device)
        
        # Preprocessing
        self.preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def get_predictions(self, images: torch.Tensor, batch_size: int = 50) -> np.ndarray:
        """Get class predictions from InceptionV3"""
        predictions_list = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(self.device)
            
            # Convert grayscale to RGB if needed
            if batch.shape[1] == 1:
                batch = batch.repeat(1, 3, 1, 1)
            
            # Preprocess
            batch = torch.stack([self.preprocess(img) for img in batch])
            
            with torch.no_grad():
                logits = self.inception_model(batch)
                probs = F.softmax(logits, dim=1)
                predictions_list.append(probs.cpu().numpy())
        
        return np.concatenate(predictions_list, axis=0)
    
    def calculate_is(self, predictions: np.ndarray, splits: int = 10) -> Tuple[float, float]:
        """Calculate Inception Score"""
        # Split predictions into groups
        split_size = predictions.shape[0] // splits
        scores = []
        
        for i in range(splits):
            start_idx = i * split_size
            end_idx = start_idx + split_size
            
            if i == splits - 1:  # Last split gets remaining samples
                end_idx = predictions.shape[0]
            
            split_preds = predictions[start_idx:end_idx]
            
            # Calculate marginal distribution
            p_y = np.mean(split_preds, axis=0)
            
            # Calculate KL divergence for each sample
            kl_divs = []
            for pred in split_preds:
                kl_div = np.sum(pred * np.log(pred / (p_y + 1e-16) + 1e-16))
                kl_divs.append(kl_div)
            
            # Calculate IS for this split
            is_score = np.exp(np.mean(kl_divs))
            scores.append(is_score)
        
        # Return mean and std of IS scores
        return np.mean(scores), np.std(scores)
    
    def __call__(self, images: torch.Tensor, splits: int = 10) -> Tuple[float, float]:
        """Calculate Inception Score for images"""
        predictions = self.get_predictions(images)
        return self.calculate_is(predictions, splits)

class LPIPSCalculator:
    """Learned Perceptual Image Patch Similarity (LPIPS) calculator"""
    
    def __init__(self, net='alex', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.lpips_model = lpips.LPIPS(net=net).to(device)
        self.lpips_model.eval()
    
    def preprocess_images(self, images: torch.Tensor) -> torch.Tensor:
        """Preprocess images for LPIPS calculation"""
        # Ensure images are in [-1, 1] range
        if images.max() > 1.0:
            images = images / 255.0 * 2.0 - 1.0
        
        # Convert grayscale to RGB if needed
        if images.shape[1] == 1:
            images = images.repeat(1, 3, 1, 1)
        
        return images
    
    def calculate_lpips_pairwise(self, images1: torch.Tensor, images2: torch.Tensor) -> torch.Tensor:
        """Calculate LPIPS between pairs of images"""
        images1 = self.preprocess_images(images1).to(self.device)
        images2 = self.preprocess_images(images2).to(self.device)
        
        with torch.no_grad():
            lpips_scores = self.lpips_model(images1, images2)
        
        return lpips_scores.cpu()
    
    def calculate_lpips_diversity(self, images: torch.Tensor, num_pairs: int = 1000) -> float:
        """Calculate average LPIPS diversity within a set of images"""
        n_images = len(images)
        
        if n_images < 2:
            return 0.0
        
        # Sample random pairs
        num_pairs = min(num_pairs, n_images * (n_images - 1) // 2)
        
        lpips_scores = []
        indices = np.random.choice(n_images, size=(num_pairs, 2), replace=True)
        
        for i, j in indices:
            if i != j:
                img1 = images[i:i+1]
                img2 = images[j:j+1]
                score = self.calculate_lpips_pairwise(img1, img2)
                lpips_scores.append(score.item())
        
        return np.mean(lpips_scores) if lpips_scores else 0.0
    
    def __call__(self, images1: torch.Tensor, images2: torch.Tensor = None) -> Union[torch.Tensor, float]:
        """Calculate LPIPS score(s)"""
        if images2 is not None:
            # Pairwise LPIPS
            return self.calculate_lpips_pairwise(images1, images2)
        else:
            # Diversity LPIPS
            return self.calculate_lpips_diversity(images1)

class WellboreImageEvaluator:
    """Comprehensive evaluator for wellbore image generation quality"""
    
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
        # Initialize metric calculators
        self.fid_calculator = FIDCalculator(device)
        self.is_calculator = InceptionScoreCalculator(device)
        self.lpips_calculator = LPIPSCalculator(device=device)
    
    def calculate_traditional_metrics(self, real_images: torch.Tensor, 
                                    fake_images: torch.Tensor) -> dict:
        """Calculate traditional image quality metrics"""
        metrics = {}
        
        # Convert to numpy for traditional metrics
        real_np = real_images.cpu().numpy()
        fake_np = fake_images.cpu().numpy()
        
        # Ensure images are in [0, 1] range
        if real_np.max() > 1.0:
            real_np = real_np / 255.0
        if fake_np.max() > 1.0:
            fake_np = fake_np / 255.0
        
        # Calculate SSIM and PSNR for paired images
        ssim_scores = []
        psnr_scores = []
        
        min_samples = min(len(real_np), len(fake_np))
        
        for i in range(min_samples):
            real_img = real_np[i]
            fake_img = fake_np[i]
            
            # Handle different image formats
            if real_img.ndim == 3 and real_img.shape[0] in [1, 3]:
                # (C, H, W) format
                real_img = np.transpose(real_img, (1, 2, 0))
                fake_img = np.transpose(fake_img, (1, 2, 0))
            
            # Convert to grayscale if RGB
            if real_img.ndim == 3 and real_img.shape[2] == 3:
                real_img = np.mean(real_img, axis=2)
                fake_img = np.mean(fake_img, axis=2)
            elif real_img.ndim == 3 and real_img.shape[2] == 1:
                real_img = real_img[:, :, 0]
                fake_img = fake_img[:, :, 0]
            
            # Calculate SSIM
            ssim_score = ssim(real_img, fake_img, data_range=1.0)
            ssim_scores.append(ssim_score)
            
            # Calculate PSNR
            psnr_score = psnr(real_img, fake_img, data_range=1.0)
            psnr_scores.append(psnr_score)
        
        metrics['ssim_mean'] = np.mean(ssim_scores)
        metrics['ssim_std'] = np.std(ssim_scores)
        metrics['psnr_mean'] = np.mean(psnr_scores)
        metrics['psnr_std'] = np.std(psnr_scores)
        
        return metrics
    
    def calculate_wellbore_specific_metrics(self, images: torch.Tensor) -> dict:
        """Calculate wellbore-specific quality metrics"""
        metrics = {}
        
        # Convert to numpy
        images_np = images.cpu().numpy()
        
        # Ensure correct format and range
        if images_np.max() > 1.0:
            images_np = images_np / 255.0
        
        edge_densities = []
        contrast_scores = []
        texture_scores = []
        
        for img in images_np:
            # Handle different formats
            if img.ndim == 3 and img.shape[0] in [1, 3]:
                img = np.transpose(img, (1, 2, 0))
            
            # Convert to grayscale
            if img.ndim == 3:
                if img.shape[2] == 3:
                    gray = np.mean(img, axis=2)
                else:
                    gray = img[:, :, 0]
            else:
                gray = img
            
            # Convert to uint8 for OpenCV
            gray_uint8 = (gray * 255).astype(np.uint8)
            
            # Edge density (important for wellbore structure)
            edges = cv2.Canny(gray_uint8, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
            
            # Contrast (important for distinguishing features)
            contrast = np.std(gray)
            contrast_scores.append(contrast)
            
            # Texture analysis using Local Binary Pattern approximation
            # Simple texture measure: variance of local gradients
            grad_x = cv2.Sobel(gray_uint8, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray_uint8, cv2.CV_64F, 0, 1, ksize=3)
            texture = np.var(np.sqrt(grad_x**2 + grad_y**2))
            texture_scores.append(texture)
        
        metrics['edge_density_mean'] = np.mean(edge_densities)
        metrics['edge_density_std'] = np.std(edge_densities)
        metrics['contrast_mean'] = np.mean(contrast_scores)
        metrics['contrast_std'] = np.std(contrast_scores)
        metrics['texture_mean'] = np.mean(texture_scores)
        metrics['texture_std'] = np.std(texture_scores)
        
        return metrics
    
    def evaluate_comprehensive(self, real_images: torch.Tensor, 
                             fake_images: torch.Tensor) -> dict:
        """Comprehensive evaluation of generated images"""
        results = {}
        
        print("Calculating FID score...")
        # FID Score
        try:
            fid_score = self.fid_calculator(real_images, fake_images)
            results['fid'] = fid_score
        except Exception as e:
            print(f"FID calculation failed: {e}")
            results['fid'] = float('inf')
        
        print("Calculating Inception Score...")
        # Inception Score
        try:
            is_mean, is_std = self.is_calculator(fake_images)
            results['is_mean'] = is_mean
            results['is_std'] = is_std
        except Exception as e:
            print(f"IS calculation failed: {e}")
            results['is_mean'] = 0.0
            results['is_std'] = 0.0
        
        print("Calculating LPIPS diversity...")
        # LPIPS Diversity
        try:
            lpips_diversity = self.lpips_calculator(fake_images)
            results['lpips_diversity'] = lpips_diversity
        except Exception as e:
            print(f"LPIPS calculation failed: {e}")
            results['lpips_diversity'] = 0.0
        
        print("Calculating traditional metrics...")
        # Traditional metrics
        try:
            traditional_metrics = self.calculate_traditional_metrics(real_images, fake_images)
            results.update(traditional_metrics)
        except Exception as e:
            print(f"Traditional metrics calculation failed: {e}")
        
        print("Calculating wellbore-specific metrics...")
        # Wellbore-specific metrics
        try:
            wellbore_metrics_real = self.calculate_wellbore_specific_metrics(real_images)
            wellbore_metrics_fake = self.calculate_wellbore_specific_metrics(fake_images)
            
            # Add prefix to distinguish real vs fake
            for key, value in wellbore_metrics_real.items():
                results[f'real_{key}'] = value
            for key, value in wellbore_metrics_fake.items():
                results[f'fake_{key}'] = value
        except Exception as e:
            print(f"Wellbore metrics calculation failed: {e}")
        
        return results

# Convenience functions
def calculate_fid_score(real_images: torch.Tensor, fake_images: torch.Tensor, 
                       device='cuda' if torch.cuda.is_available() else 'cpu') -> float:
    """Calculate FID score between real and fake images"""
    calculator = FIDCalculator(device)
    return calculator(real_images, fake_images)

def calculate_inception_score(images: torch.Tensor, splits: int = 10,
                            device='cuda' if torch.cuda.is_available() else 'cpu') -> Tuple[float, float]:
    """Calculate Inception Score for images"""
    calculator = InceptionScoreCalculator(device)
    return calculator(images, splits)

def calculate_lpips_score(images1: torch.Tensor, images2: torch.Tensor = None,
                         device='cuda' if torch.cuda.is_available() else 'cpu') -> Union[torch.Tensor, float]:
    """Calculate LPIPS score(s)"""
    calculator = LPIPSCalculator(device=device)
    return calculator(images1, images2)