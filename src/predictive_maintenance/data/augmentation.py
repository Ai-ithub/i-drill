"""Data augmentation utilities for wellbore images"""

import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from typing import Tuple, List, Optional, Union, Callable
import cv2
import random
from scipy import ndimage
from skimage import transform, util
import albumentations as A
from albumentations.pytorch import ToTensorV2

class WellboreAugmentation:
    """Augmentation pipeline specifically designed for wellbore images"""
    
    def __init__(self, 
                 rotation_range: float = 15.0,
                 zoom_range: Tuple[float, float] = (0.8, 1.2),
                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                 contrast_range: Tuple[float, float] = (0.8, 1.2),
                 noise_std: float = 0.02,
                 blur_kernel_size: int = 3,
                 elastic_alpha: float = 50.0,
                 elastic_sigma: float = 5.0,
                 probability: float = 0.5):
        """
        Args:
            rotation_range: Maximum rotation angle in degrees
            zoom_range: Range for random zoom (min, max)
            brightness_range: Range for brightness adjustment
            contrast_range: Range for contrast adjustment
            noise_std: Standard deviation for Gaussian noise
            blur_kernel_size: Kernel size for Gaussian blur
            elastic_alpha: Alpha parameter for elastic deformation
            elastic_sigma: Sigma parameter for elastic deformation
            probability: Probability of applying each augmentation
        """
        self.rotation_range = rotation_range
        self.zoom_range = zoom_range
        self.brightness_range = brightness_range
        self.contrast_range = contrast_range
        self.noise_std = noise_std
        self.blur_kernel_size = blur_kernel_size
        self.elastic_alpha = elastic_alpha
        self.elastic_sigma = elastic_sigma
        self.probability = probability
        
        # Create Albumentations pipeline
        self.albumentations_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(limit=rotation_range, p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2, 
                contrast_limit=0.2, 
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.GaussianBlur(blur_limit=(3, 7), p=0.3),
            A.ElasticTransform(
                alpha=elastic_alpha, 
                sigma=elastic_sigma, 
                alpha_affine=10, 
                p=0.3
            ),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.ShiftScaleRotate(
                shift_limit=0.1, 
                scale_limit=0.2, 
                rotate_limit=rotation_range, 
                p=0.5
            ),
        ])
    
    def random_rotation(self, image: np.ndarray) -> np.ndarray:
        """Apply random rotation"""
        if random.random() > self.probability:
            return image
        
        angle = random.uniform(-self.rotation_range, self.rotation_range)
        
        if len(image.shape) == 3:
            h, w, c = image.shape
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR)
        else:
            rotated = ndimage.rotate(image, angle, reshape=False, mode='reflect')
        
        return rotated
    
    def random_zoom(self, image: np.ndarray) -> np.ndarray:
        """Apply random zoom"""
        if random.random() > self.probability:
            return image
        
        zoom_factor = random.uniform(self.zoom_range[0], self.zoom_range[1])
        
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        
        # Calculate new dimensions
        new_h, new_w = int(h * zoom_factor), int(w * zoom_factor)
        
        if zoom_factor > 1.0:
            # Zoom in - crop center
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                zoomed = resized[start_h:start_h+h, start_w:start_w+w]
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                start_h = (new_h - h) // 2
                start_w = (new_w - w) // 2
                zoomed = resized[start_h:start_h+h, start_w:start_w+w]
        else:
            # Zoom out - pad with zeros
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                zoomed = np.zeros_like(image)
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                zoomed[start_h:start_h+new_h, start_w:start_w+new_w] = resized
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                zoomed = np.zeros_like(image)
                start_h = (h - new_h) // 2
                start_w = (w - new_w) // 2
                zoomed[start_h:start_h+new_h, start_w:start_w+new_w] = resized
        
        return zoomed
    
    def random_brightness_contrast(self, image: np.ndarray) -> np.ndarray:
        """Apply random brightness and contrast adjustment"""
        if random.random() > self.probability:
            return image
        
        brightness_factor = random.uniform(self.brightness_range[0], self.brightness_range[1])
        contrast_factor = random.uniform(self.contrast_range[0], self.contrast_range[1])
        
        # Convert to float for processing
        image_float = image.astype(np.float32)
        
        # Apply brightness
        image_float = image_float * brightness_factor
        
        # Apply contrast
        mean = np.mean(image_float)
        image_float = (image_float - mean) * contrast_factor + mean
        
        # Clip values
        image_float = np.clip(image_float, 0, 255)
        
        return image_float.astype(np.uint8)
    
    def add_gaussian_noise(self, image: np.ndarray) -> np.ndarray:
        """Add Gaussian noise"""
        if random.random() > self.probability:
            return image
        
        noise = np.random.normal(0, self.noise_std * 255, image.shape)
        noisy_image = image.astype(np.float32) + noise
        noisy_image = np.clip(noisy_image, 0, 255)
        
        return noisy_image.astype(np.uint8)
    
    def random_blur(self, image: np.ndarray) -> np.ndarray:
        """Apply random Gaussian blur"""
        if random.random() > self.probability:
            return image
        
        kernel_size = random.choice([3, 5, 7])
        sigma = random.uniform(0.5, 2.0)
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def elastic_deformation(self, image: np.ndarray) -> np.ndarray:
        """Apply elastic deformation"""
        if random.random() > self.probability:
            return image
        
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (h, w)) * self.elastic_alpha
        dy = np.random.uniform(-1, 1, (h, w)) * self.elastic_alpha
        
        # Smooth the displacement fields
        dx = ndimage.gaussian_filter(dx, self.elastic_sigma, mode='reflect')
        dy = ndimage.gaussian_filter(dy, self.elastic_sigma, mode='reflect')
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(w), np.arange(h))
        
        # Apply displacement
        x_new = np.clip(x + dx, 0, w - 1)
        y_new = np.clip(y + dy, 0, h - 1)
        
        # Interpolate
        if len(image.shape) == 3:
            deformed = np.zeros_like(image)
            for c in range(image.shape[2]):
                deformed[:, :, c] = ndimage.map_coordinates(
                    image[:, :, c], [y_new, x_new], order=1, mode='reflect'
                )
        else:
            deformed = ndimage.map_coordinates(
                image, [y_new, x_new], order=1, mode='reflect'
            )
        
        return deformed.astype(np.uint8)
    
    def random_crop_and_resize(self, image: np.ndarray, crop_ratio: float = 0.8) -> np.ndarray:
        """Random crop and resize back to original size"""
        if random.random() > self.probability:
            return image
        
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        
        # Calculate crop size
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
        
        # Random crop position
        start_h = random.randint(0, h - crop_h)
        start_w = random.randint(0, w - crop_w)
        
        # Crop
        if len(image.shape) == 3:
            cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        else:
            cropped = image[start_h:start_h+crop_h, start_w:start_w+crop_w]
        
        # Resize back to original size
        resized = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        
        return resized
    
    def simulate_wellbore_artifacts(self, image: np.ndarray) -> np.ndarray:
        """Simulate common wellbore imaging artifacts"""
        if random.random() > self.probability:
            return image
        
        artifact_type = random.choice(['washout', 'breakout', 'mud_cake', 'tool_marks'])
        
        if len(image.shape) == 3:
            h, w, c = image.shape
        else:
            h, w = image.shape
        
        if artifact_type == 'washout':
            # Simulate washout (dark irregular regions)
            num_washouts = random.randint(1, 3)
            for _ in range(num_washouts):
                center_x = random.randint(w//4, 3*w//4)
                center_y = random.randint(h//4, 3*h//4)
                radius = random.randint(10, 30)
                
                # Create irregular shape
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                
                # Add some irregularity
                noise_mask = np.random.random((h, w)) > 0.7
                mask = mask & noise_mask
                
                # Darken the region
                if len(image.shape) == 3:
                    image[mask] = image[mask] * 0.3
                else:
                    image[mask] = image[mask] * 0.3
        
        elif artifact_type == 'breakout':
            # Simulate breakout (bright irregular regions)
            num_breakouts = random.randint(1, 2)
            for _ in range(num_breakouts):
                center_x = random.randint(w//4, 3*w//4)
                center_y = random.randint(h//4, 3*h//4)
                radius = random.randint(15, 40)
                
                y, x = np.ogrid[:h, :w]
                mask = ((x - center_x)**2 + (y - center_y)**2) <= radius**2
                
                # Brighten the region
                if len(image.shape) == 3:
                    image[mask] = np.minimum(image[mask] * 1.5 + 50, 255)
                else:
                    image[mask] = np.minimum(image[mask] * 1.5 + 50, 255)
        
        elif artifact_type == 'mud_cake':
            # Simulate mud cake (textured regions)
            num_regions = random.randint(1, 3)
            for _ in range(num_regions):
                start_x = random.randint(0, w//2)
                start_y = random.randint(0, h//2)
                end_x = random.randint(start_x + 20, w)
                end_y = random.randint(start_y + 20, h)
                
                # Add texture
                texture = np.random.normal(0, 10, (end_y - start_y, end_x - start_x))
                
                if len(image.shape) == 3:
                    for c in range(image.shape[2]):
                        region = image[start_y:end_y, start_x:end_x, c].astype(np.float32)
                        region += texture
                        image[start_y:end_y, start_x:end_x, c] = np.clip(region, 0, 255).astype(np.uint8)
                else:
                    region = image[start_y:end_y, start_x:end_x].astype(np.float32)
                    region += texture
                    image[start_y:end_y, start_x:end_x] = np.clip(region, 0, 255).astype(np.uint8)
        
        elif artifact_type == 'tool_marks':
            # Simulate tool marks (linear features)
            num_marks = random.randint(2, 5)
            for _ in range(num_marks):
                # Random line
                x1 = random.randint(0, w)
                y1 = random.randint(0, h)
                x2 = random.randint(0, w)
                y2 = random.randint(0, h)
                
                thickness = random.randint(1, 3)
                intensity = random.randint(20, 60)
                
                if len(image.shape) == 3:
                    cv2.line(image, (x1, y1), (x2, y2), 
                            (intensity, intensity, intensity), thickness)
                else:
                    cv2.line(image, (x1, y1), (x2, y2), intensity, thickness)
        
        return image
    
    def augment_with_albumentations(self, image: np.ndarray) -> np.ndarray:
        """Apply Albumentations augmentation pipeline"""
        augmented = self.albumentations_transform(image=image)
        return augmented['image']
    
    def __call__(self, image: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Apply full augmentation pipeline"""
        # Convert tensor to numpy if needed
        if isinstance(image, torch.Tensor):
            if image.is_cuda:
                image_np = image.cpu().numpy()
            else:
                image_np = image.numpy()
            
            # Handle tensor format (C, H, W) -> (H, W, C)
            if len(image_np.shape) == 3:
                image_np = np.transpose(image_np, (1, 2, 0))
            
            # Denormalize if needed (from [-1, 1] to [0, 255])
            if image_np.min() < 0:
                image_np = (image_np + 1.0) * 127.5
            
            image_np = np.clip(image_np, 0, 255).astype(np.uint8)
            return_tensor = True
        else:
            image_np = image.copy()
            return_tensor = False
        
        # Apply augmentations
        augmented = image_np
        
        # Basic geometric augmentations
        augmented = self.random_rotation(augmented)
        augmented = self.random_zoom(augmented)
        augmented = self.random_crop_and_resize(augmented)
        
        # Photometric augmentations
        augmented = self.random_brightness_contrast(augmented)
        augmented = self.add_gaussian_noise(augmented)
        augmented = self.random_blur(augmented)
        
        # Advanced deformations
        augmented = self.elastic_deformation(augmented)
        
        # Domain-specific augmentations
        augmented = self.simulate_wellbore_artifacts(augmented)
        
        # Apply Albumentations pipeline
        augmented = self.augment_with_albumentations(augmented)
        
        # Convert back to tensor if needed
        if return_tensor:
            # Normalize to [-1, 1]
            augmented_float = augmented.astype(np.float32) / 127.5 - 1.0
            
            # Convert to tensor and rearrange dimensions
            if len(augmented_float.shape) == 3:
                tensor = torch.from_numpy(augmented_float).permute(2, 0, 1)
            else:
                tensor = torch.from_numpy(augmented_float).unsqueeze(0)
            
            return tensor
        
        return augmented

class WellboreAugmentationPipeline:
    """Complete augmentation pipeline for wellbore image datasets"""
    
    def __init__(self, 
                 train_augmentations: bool = True,
                 validation_augmentations: bool = False,
                 custom_augmentations: Optional[List[Callable]] = None):
        """
        Args:
            train_augmentations: Whether to apply augmentations for training
            validation_augmentations: Whether to apply light augmentations for validation
            custom_augmentations: List of custom augmentation functions
        """
        self.train_augmentations = train_augmentations
        self.validation_augmentations = validation_augmentations
        self.custom_augmentations = custom_augmentations or []
        
        # Training augmentation pipeline
        self.train_pipeline = WellboreAugmentation(
            rotation_range=20.0,
            zoom_range=(0.7, 1.3),
            brightness_range=(0.7, 1.3),
            contrast_range=(0.7, 1.3),
            noise_std=0.03,
            probability=0.7
        )
        
        # Validation augmentation pipeline (lighter)
        self.val_pipeline = WellboreAugmentation(
            rotation_range=5.0,
            zoom_range=(0.9, 1.1),
            brightness_range=(0.9, 1.1),
            contrast_range=(0.9, 1.1),
            noise_std=0.01,
            probability=0.3
        )
    
    def get_train_transform(self) -> Callable:
        """Get training augmentation transform"""
        def transform(image):
            if self.train_augmentations:
                augmented = self.train_pipeline(image)
                
                # Apply custom augmentations
                for custom_aug in self.custom_augmentations:
                    augmented = custom_aug(augmented)
                
                return augmented
            return image
        
        return transform
    
    def get_val_transform(self) -> Callable:
        """Get validation augmentation transform"""
        def transform(image):
            if self.validation_augmentations:
                return self.val_pipeline(image)
            return image
        
        return transform
    
    def get_test_transform(self) -> Callable:
        """Get test transform (no augmentation)"""
        def transform(image):
            return image
        
        return transform