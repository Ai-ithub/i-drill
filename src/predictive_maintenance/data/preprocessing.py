"""Image preprocessing utilities for wellbore images"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, List, Optional, Union
import cv2
from skimage import exposure, filters, morphology
from skimage.restoration import denoise_nl_means

class WellboreImagePreprocessor:
    """Preprocessing pipeline for wellbore images"""
    
    def __init__(self, 
                 target_size: int = 256,
                 normalize: bool = True,
                 enhance_contrast: bool = True,
                 denoise: bool = True,
                 enhance_edges: bool = True):
        """
        Args:
            target_size: Target image size for resizing
            normalize: Whether to normalize pixel values to [-1, 1]
            enhance_contrast: Whether to apply contrast enhancement
            denoise: Whether to apply denoising
            enhance_edges: Whether to enhance edge features
        """
        self.target_size = target_size
        self.normalize = normalize
        self.enhance_contrast = enhance_contrast
        self.denoise = denoise
        self.enhance_edges = enhance_edges
    
    def resize_image(self, image: Union[Image.Image, np.ndarray], 
                    size: Optional[int] = None) -> Union[Image.Image, np.ndarray]:
        """Resize image while maintaining aspect ratio"""
        if size is None:
            size = self.target_size
        
        if isinstance(image, Image.Image):
            # PIL Image
            original_size = image.size
            ratio = min(size / original_size[0], size / original_size[1])
            new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
            
            # Resize and pad to target size
            resized = image.resize(new_size, Image.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', (size, size), (0, 0, 0))
            paste_x = (size - new_size[0]) // 2
            paste_y = (size - new_size[1]) // 2
            new_image.paste(resized, (paste_x, paste_y))
            
            return new_image
        
        else:
            # NumPy array
            h, w = image.shape[:2]
            ratio = min(size / w, size / h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            
            # Resize
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            # Pad to target size
            if len(image.shape) == 3:
                new_image = np.zeros((size, size, image.shape[2]), dtype=image.dtype)
            else:
                new_image = np.zeros((size, size), dtype=image.dtype)
            
            paste_x = (size - new_w) // 2
            paste_y = (size - new_h) // 2
            
            if len(image.shape) == 3:
                new_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
            else:
                new_image[paste_y:paste_y+new_h, paste_x:paste_x+new_w] = resized
            
            return new_image
    
    def enhance_contrast_adaptive(self, image: np.ndarray) -> np.ndarray:
        """Apply adaptive histogram equalization"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            
            # Convert back to RGB
            enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
        
        return enhanced
    
    def denoise_image(self, image: np.ndarray) -> np.ndarray:
        """Apply non-local means denoising"""
        if len(image.shape) == 3:
            # Color image
            denoised = denoise_nl_means(
                image, 
                h=0.1, 
                fast_mode=True, 
                patch_size=5, 
                patch_distance=3,
                multichannel=True
            )
        else:
            # Grayscale image
            denoised = denoise_nl_means(
                image, 
                h=0.1, 
                fast_mode=True, 
                patch_size=5, 
                patch_distance=3
            )
        
        # Convert back to uint8
        denoised = (denoised * 255).astype(np.uint8)
        return denoised
    
    def enhance_edges(self, image: np.ndarray) -> np.ndarray:
        """Enhance edge features in wellbore images"""
        if len(image.shape) == 3:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Detect edges using Canny
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        if len(image.shape) == 3:
            # Combine with original image
            enhanced = image.copy()
            for c in range(3):
                enhanced[:, :, c] = cv2.addWeighted(
                    enhanced[:, :, c], 0.8, edges, 0.2, 0
                )
        else:
            enhanced = cv2.addWeighted(image, 0.8, edges, 0.2, 0)
        
        return enhanced
    
    def correct_illumination(self, image: np.ndarray) -> np.ndarray:
        """Correct uneven illumination in wellbore images"""
        if len(image.shape) == 3:
            # Convert to grayscale for illumination correction
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # Create background model using morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
        background = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        
        # Smooth the background
        background = cv2.GaussianBlur(background, (51, 51), 0)
        
        # Correct illumination
        corrected = cv2.divide(gray, background, scale=255)
        
        if len(image.shape) == 3:
            # Apply correction to all channels
            correction_factor = corrected.astype(np.float32) / gray.astype(np.float32)
            correction_factor = np.clip(correction_factor, 0.5, 2.0)
            
            enhanced = image.copy().astype(np.float32)
            for c in range(3):
                enhanced[:, :, c] *= correction_factor
            
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
        else:
            enhanced = corrected
        
        return enhanced
    
    def remove_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Remove common artifacts in wellbore images"""
        # Remove small noise using morphological operations
        if len(image.shape) == 3:
            processed = image.copy()
            for c in range(3):
                # Remove small bright spots
                kernel = np.ones((3, 3), np.uint8)
                processed[:, :, c] = cv2.morphologyEx(
                    processed[:, :, c], cv2.MORPH_OPEN, kernel
                )
                
                # Fill small dark holes
                processed[:, :, c] = cv2.morphologyEx(
                    processed[:, :, c], cv2.MORPH_CLOSE, kernel
                )
        else:
            kernel = np.ones((3, 3), np.uint8)
            processed = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            processed = cv2.morphologyEx(processed, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [-1, 1] range for GAN training"""
        normalized = image.astype(np.float32) / 127.5 - 1.0
        return normalized
    
    def preprocess(self, image: Union[Image.Image, np.ndarray]) -> torch.Tensor:
        """Complete preprocessing pipeline"""
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image.copy()
        
        # Ensure RGB format
        if len(image_np.shape) == 3 and image_np.shape[2] == 4:
            # Convert RGBA to RGB
            image_np = image_np[:, :, :3]
        
        # Resize image
        image_np = self.resize_image(image_np, self.target_size)
        
        # Correct illumination
        image_np = self.correct_illumination(image_np)
        
        # Enhance contrast
        if self.enhance_contrast:
            image_np = self.enhance_contrast_adaptive(image_np)
        
        # Denoise
        if self.denoise:
            image_np = self.denoise_image(image_np)
        
        # Enhance edges
        if self.enhance_edges:
            image_np = self.enhance_edges(image_np)
        
        # Remove artifacts
        image_np = self.remove_artifacts(image_np)
        
        # Normalize
        if self.normalize:
            image_np = self.normalize_image(image_np)
        
        # Convert to tensor
        if len(image_np.shape) == 3:
            tensor = torch.from_numpy(image_np).permute(2, 0, 1).float()
        else:
            tensor = torch.from_numpy(image_np).unsqueeze(0).float()
        
        return tensor
    
    def preprocess_batch(self, images: List[Union[Image.Image, np.ndarray]]) -> torch.Tensor:
        """Preprocess a batch of images"""
        processed_images = []
        
        for image in images:
            processed = self.preprocess(image)
            processed_images.append(processed)
        
        # Stack into batch tensor
        batch_tensor = torch.stack(processed_images, dim=0)
        return batch_tensor
    
    def postprocess(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert tensor back to displayable image"""
        # Move to CPU and convert to numpy
        if tensor.is_cuda:
            tensor = tensor.cpu()
        
        image_np = tensor.detach().numpy()
        
        # Handle batch dimension
        if len(image_np.shape) == 4:
            image_np = image_np[0]  # Take first image from batch
        
        # Rearrange dimensions (C, H, W) -> (H, W, C)
        if len(image_np.shape) == 3:
            image_np = np.transpose(image_np, (1, 2, 0))
        
        # Denormalize from [-1, 1] to [0, 255]
        if self.normalize:
            image_np = (image_np + 1.0) * 127.5
        
        # Clip and convert to uint8
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        
        # Handle grayscale
        if len(image_np.shape) == 3 and image_np.shape[2] == 1:
            image_np = image_np.squeeze(2)
        
        return image_np

class WellboreImageAnalyzer:
    """Analyze wellbore images for quality and characteristics"""
    
    def __init__(self):
        pass
    
    def calculate_image_quality_metrics(self, image: np.ndarray) -> dict:
        """Calculate various image quality metrics"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        metrics = {}
        
        # Sharpness (Laplacian variance)
        metrics['sharpness'] = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Contrast (standard deviation)
        metrics['contrast'] = gray.std()
        
        # Brightness (mean intensity)
        metrics['brightness'] = gray.mean()
        
        # Signal-to-noise ratio estimate
        # Using the ratio of signal power to noise power
        signal_power = np.mean(gray ** 2)
        noise_estimate = np.mean((gray - cv2.GaussianBlur(gray, (5, 5), 0)) ** 2)
        metrics['snr'] = 10 * np.log10(signal_power / (noise_estimate + 1e-10))
        
        # Edge density
        edges = cv2.Canny(gray, 50, 150)
        metrics['edge_density'] = np.sum(edges > 0) / edges.size
        
        return metrics
    
    def detect_failure_regions(self, image: np.ndarray) -> dict:
        """Detect potential failure regions in wellbore images"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        results = {}
        
        # Detect dark regions (potential washouts)
        dark_threshold = np.percentile(gray, 10)
        dark_regions = gray < dark_threshold
        results['dark_regions'] = {
            'area_ratio': np.sum(dark_regions) / gray.size,
            'num_components': len(np.unique(cv2.connectedComponents(dark_regions.astype(np.uint8))[1])) - 1
        }
        
        # Detect bright regions (potential breakouts)
        bright_threshold = np.percentile(gray, 90)
        bright_regions = gray > bright_threshold
        results['bright_regions'] = {
            'area_ratio': np.sum(bright_regions) / gray.size,
            'num_components': len(np.unique(cv2.connectedComponents(bright_regions.astype(np.uint8))[1])) - 1
        }
        
        # Detect linear features (potential fractures)
        # Use Hough line transform
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)
        results['linear_features'] = {
            'num_lines': len(lines) if lines is not None else 0
        }
        
        return results
    
    def assess_image_suitability(self, image: np.ndarray) -> dict:
        """Assess if image is suitable for GAN training"""
        quality_metrics = self.calculate_image_quality_metrics(image)
        
        assessment = {
            'suitable': True,
            'issues': [],
            'quality_score': 0.0
        }
        
        # Check sharpness
        if quality_metrics['sharpness'] < 100:
            assessment['issues'].append('Low sharpness')
            assessment['suitable'] = False
        
        # Check contrast
        if quality_metrics['contrast'] < 20:
            assessment['issues'].append('Low contrast')
            assessment['suitable'] = False
        
        # Check brightness
        if quality_metrics['brightness'] < 30 or quality_metrics['brightness'] > 225:
            assessment['issues'].append('Poor brightness')
            assessment['suitable'] = False
        
        # Check SNR
        if quality_metrics['snr'] < 10:
            assessment['issues'].append('Low signal-to-noise ratio')
            assessment['suitable'] = False
        
        # Calculate overall quality score (0-1)
        sharpness_score = min(quality_metrics['sharpness'] / 500, 1.0)
        contrast_score = min(quality_metrics['contrast'] / 100, 1.0)
        brightness_score = 1.0 - abs(quality_metrics['brightness'] - 127.5) / 127.5
        snr_score = min(quality_metrics['snr'] / 30, 1.0)
        
        assessment['quality_score'] = (sharpness_score + contrast_score + brightness_score + snr_score) / 4
        
        return assessment