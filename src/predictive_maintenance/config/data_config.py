#!/usr/bin/env python3
"""Data configuration for wellbore image generation system"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
from .base_config import BaseConfig, ConfigValidationError

class DataConfig(BaseConfig):
    """Configuration for data loading and preprocessing"""
    
    def _set_defaults(self):
        """Set default data configuration values"""
        # Dataset paths
        self.dataset_path = './data/wellbore_images'
        self.synthetic_dataset_path = './data/synthetic_wellbore_images'
        self.metadata_path = './data/metadata.json'
        self.cache_dir = './data/cache'
        
        # Image specifications
        self.image_size = 256
        self.image_channels = 3
        self.image_format = 'RGB'  # 'RGB', 'L' (grayscale)
        self.supported_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']
        
        # Data splitting
        self.train_split = 0.8
        self.val_split = 0.1
        self.test_split = 0.1
        self.split_seed = 42
        self.stratify_by = 'failure_type'  # None, 'failure_type', 'severity'
        
        # Failure types and labels
        self.failure_types = [
            'normal',
            'casing_damage',
            'corrosion',
            'scaling',
            'perforation_damage',
            'cement_bond_failure',
            'fluid_invasion',
            'mechanical_damage'
        ]
        
        self.severity_levels = ['low', 'medium', 'high', 'critical']
        
        # Class balancing
        self.balance_classes = True
        self.balancing_method = 'oversample'  # 'oversample', 'undersample', 'weighted'
        self.min_samples_per_class = 100
        self.max_samples_per_class = 10000
        
        # Data loading
        self.batch_size = 16
        self.num_workers = 4
        self.pin_memory = True
        self.persistent_workers = True
        self.prefetch_factor = 2
        self.drop_last = True
        self.shuffle = True
        
        # Preprocessing
        self.preprocessing = {
            'resize_method': 'bilinear',  # 'bilinear', 'bicubic', 'nearest'
            'normalize': True,
            'mean': [0.5, 0.5, 0.5],  # ImageNet: [0.485, 0.456, 0.406]
            'std': [0.5, 0.5, 0.5],   # ImageNet: [0.229, 0.224, 0.225]
            'to_tensor': True,
            'enhance_contrast': True,
            'contrast_factor': 1.2,
            'enhance_sharpness': False,
            'sharpness_factor': 1.1,
            'gamma_correction': False,
            'gamma': 1.0,
            'histogram_equalization': False,
            'clahe': True,  # Contrast Limited Adaptive Histogram Equalization
            'clahe_clip_limit': 2.0,
            'clahe_tile_grid_size': (8, 8)
        }
        
        # Data augmentation
        self.augmentation = {
            'enabled': True,
            'probability': 0.8,
            
            # Geometric transformations
            'rotation': {
                'enabled': True,
                'angle_range': (-15, 15),
                'probability': 0.5
            },
            
            'horizontal_flip': {
                'enabled': True,
                'probability': 0.5
            },
            
            'vertical_flip': {
                'enabled': False,
                'probability': 0.2
            },
            
            'scaling': {
                'enabled': True,
                'scale_range': (0.9, 1.1),
                'probability': 0.3
            },
            
            'translation': {
                'enabled': True,
                'translate_percent': 0.1,
                'probability': 0.3
            },
            
            'shearing': {
                'enabled': True,
                'shear_range': (-5, 5),
                'probability': 0.2
            },
            
            # Color transformations
            'brightness': {
                'enabled': True,
                'brightness_range': (0.8, 1.2),
                'probability': 0.4
            },
            
            'contrast': {
                'enabled': True,
                'contrast_range': (0.8, 1.2),
                'probability': 0.4
            },
            
            'saturation': {
                'enabled': True,
                'saturation_range': (0.8, 1.2),
                'probability': 0.3
            },
            
            'hue': {
                'enabled': True,
                'hue_range': (-0.1, 0.1),
                'probability': 0.2
            },
            
            # Noise and blur
            'gaussian_noise': {
                'enabled': True,
                'noise_std': 0.02,
                'probability': 0.3
            },
            
            'gaussian_blur': {
                'enabled': True,
                'blur_sigma': (0.1, 2.0),
                'probability': 0.2
            },
            
            'motion_blur': {
                'enabled': True,
                'kernel_size': (3, 7),
                'probability': 0.1
            },
            
            # Wellbore-specific augmentations
            'simulate_artifacts': {
                'enabled': True,
                'artifact_types': ['mud_cake', 'tool_marks', 'washouts'],
                'probability': 0.2
            },
            
            'depth_variation': {
                'enabled': True,
                'depth_factor_range': (0.9, 1.1),
                'probability': 0.3
            },
            
            'lighting_variation': {
                'enabled': True,
                'lighting_factor_range': (0.7, 1.3),
                'probability': 0.4
            }
        }
        
        # Synthetic data generation
        self.synthetic_data = {
            'enabled': False,
            'generator_checkpoint': None,
            'num_synthetic_samples': 1000,
            'synthetic_ratio': 0.2,  # Ratio of synthetic to real data
            'quality_threshold': 0.7,  # Minimum quality score for synthetic images
            'diversity_weight': 0.3,  # Weight for diversity in synthetic generation
        }
        
        # Data validation
        self.validation = {
            'check_image_integrity': True,
            'min_image_size': (64, 64),
            'max_image_size': (2048, 2048),
            'check_color_channels': True,
            'remove_corrupted': True,
            'check_duplicates': True,
            'duplicate_threshold': 0.95,  # SSIM threshold for duplicate detection
        }
        
        # Caching
        self.caching = {
            'enabled': True,
            'cache_preprocessed': True,
            'cache_augmented': False,
            'cache_format': 'hdf5',  # 'hdf5', 'pickle', 'numpy'
            'compression': 'gzip',
            'cache_size_limit': '10GB'
        }
        
        # Memory management
        self.memory = {
            'lazy_loading': True,
            'preload_data': False,
            'memory_map': True,
            'max_memory_usage': '8GB'
        }
        
        # Quality control
        self.quality_control = {
            'enabled': True,
            'blur_threshold': 100,  # Laplacian variance threshold
            'brightness_range': (0.1, 0.9),
            'contrast_threshold': 0.1,
            'noise_threshold': 0.3,
            'artifact_detection': True
        }
    
    def validate(self):
        """Validate data configuration"""
        # Validate paths
        if not self.dataset_path:
            raise ConfigValidationError("dataset_path cannot be empty")
        
        # Validate image specifications
        if self.image_size <= 0:
            raise ConfigValidationError("image_size must be positive")
        
        if self.image_channels not in [1, 3]:
            raise ConfigValidationError("image_channels must be 1 or 3")
        
        if self.image_format not in ['RGB', 'L']:
            raise ConfigValidationError("image_format must be 'RGB' or 'L'")
        
        # Validate data splitting
        total_split = self.train_split + self.val_split + self.test_split
        if abs(total_split - 1.0) > 1e-6:
            raise ConfigValidationError("Data splits must sum to 1.0")
        
        if any(split < 0 or split > 1 for split in [self.train_split, self.val_split, self.test_split]):
            raise ConfigValidationError("All splits must be between 0 and 1")
        
        # Validate class balancing
        if self.balancing_method not in ['oversample', 'undersample', 'weighted']:
            raise ConfigValidationError("Invalid balancing_method")
        
        if self.min_samples_per_class <= 0:
            raise ConfigValidationError("min_samples_per_class must be positive")
        
        if self.max_samples_per_class <= self.min_samples_per_class:
            raise ConfigValidationError("max_samples_per_class must be greater than min_samples_per_class")
        
        # Validate data loading parameters
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        
        if self.num_workers < 0:
            raise ConfigValidationError("num_workers must be non-negative")
        
        # Validate preprocessing
        preprocessing = self.preprocessing
        if preprocessing['resize_method'] not in ['bilinear', 'bicubic', 'nearest']:
            raise ConfigValidationError("Invalid resize_method")
        
        if len(preprocessing['mean']) != self.image_channels:
            raise ConfigValidationError("mean must have same length as image_channels")
        
        if len(preprocessing['std']) != self.image_channels:
            raise ConfigValidationError("std must have same length as image_channels")
        
        # Validate augmentation probabilities
        aug_config = self.augmentation
        if not (0.0 <= aug_config['probability'] <= 1.0):
            raise ConfigValidationError("augmentation probability must be between 0 and 1")
        
        for aug_name, aug_params in aug_config.items():
            if isinstance(aug_params, dict) and 'probability' in aug_params:
                if not (0.0 <= aug_params['probability'] <= 1.0):
                    raise ConfigValidationError(f"{aug_name} probability must be between 0 and 1")
        
        # Validate synthetic data config
        synthetic_config = self.synthetic_data
        if synthetic_config['enabled'] and not synthetic_config['generator_checkpoint']:
            raise ConfigValidationError("generator_checkpoint required when synthetic_data is enabled")
        
        if not (0.0 <= synthetic_config['synthetic_ratio'] <= 1.0):
            raise ConfigValidationError("synthetic_ratio must be between 0 and 1")
        
        # Validate quality control thresholds
        qc_config = self.quality_control
        if not (0.0 <= qc_config['brightness_range'][0] <= qc_config['brightness_range'][1] <= 1.0):
            raise ConfigValidationError("Invalid brightness_range")
    
    def get_class_weights(self, class_counts: Dict[str, int]) -> Dict[str, float]:
        """Calculate class weights for balanced training"""
        if not self.balance_classes:
            return {cls: 1.0 for cls in class_counts.keys()}
        
        total_samples = sum(class_counts.values())
        num_classes = len(class_counts)
        
        weights = {}
        for cls, count in class_counts.items():
            if self.balancing_method == 'weighted':
                weights[cls] = total_samples / (num_classes * count)
            else:
                weights[cls] = 1.0
        
        return weights
    
    def get_augmentation_pipeline(self, mode: str = 'train') -> Dict[str, Any]:
        """Get augmentation pipeline configuration for specific mode"""
        if mode != 'train' or not self.augmentation['enabled']:
            return {'enabled': False}
        
        return self.augmentation.copy()
    
    def get_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Get preprocessing pipeline configuration"""
        return self.preprocessing.copy()
    
    def create_data_directories(self):
        """Create necessary data directories"""
        directories = [
            self.dataset_path,
            self.synthetic_dataset_path,
            self.cache_dir,
            os.path.dirname(self.metadata_path) if self.metadata_path else None
        ]
        
        for directory in directories:
            if directory:
                os.makedirs(directory, exist_ok=True)
    
    def get_dataset_stats_path(self) -> str:
        """Get path for dataset statistics cache"""
        return os.path.join(self.cache_dir, 'dataset_stats.json')
    
    def get_split_info_path(self) -> str:
        """Get path for data split information"""
        return os.path.join(self.cache_dir, 'split_info.json')
    
    def adapt_for_resolution(self, target_resolution: int):
        """Adapt configuration for different image resolution"""
        if target_resolution <= 0:
            raise ValueError("target_resolution must be positive")
        
        self.image_size = target_resolution
        
        # Adjust CLAHE tile grid size based on resolution
        if target_resolution <= 128:
            self.preprocessing['clahe_tile_grid_size'] = (4, 4)
        elif target_resolution <= 256:
            self.preprocessing['clahe_tile_grid_size'] = (8, 8)
        else:
            self.preprocessing['clahe_tile_grid_size'] = (16, 16)
        
        # Re-validate
        self.validate()
    
    def enable_synthetic_data(self, generator_checkpoint: str, num_samples: int = 1000):
        """Enable synthetic data generation"""
        if not os.path.exists(generator_checkpoint):
            raise FileNotFoundError(f"Generator checkpoint not found: {generator_checkpoint}")
        
        self.synthetic_data['enabled'] = True
        self.synthetic_data['generator_checkpoint'] = generator_checkpoint
        self.synthetic_data['num_synthetic_samples'] = num_samples
    
    def disable_augmentation(self):
        """Disable all data augmentation"""
        self.augmentation['enabled'] = False
    
    def set_wellbore_specific_augmentation(self):
        """Configure augmentation specifically for wellbore images"""
        # Disable transformations that don't make sense for wellbore images
        self.augmentation['vertical_flip']['enabled'] = False
        self.augmentation['hue']['enabled'] = False  # Wellbore images are often grayscale-like
        
        # Enable wellbore-specific augmentations
        self.augmentation['simulate_artifacts']['enabled'] = True
        self.augmentation['depth_variation']['enabled'] = True
        self.augmentation['lighting_variation']['enabled'] = True
        
        # Adjust rotation range (wellbore images have natural orientation)
        self.augmentation['rotation']['angle_range'] = (-5, 5)
    
    def get_memory_usage_estimate(self, num_samples: int) -> Dict[str, float]:
        """Estimate memory usage for dataset"""
        # Calculate per-image memory usage
        bytes_per_pixel = 4 if self.preprocessing['to_tensor'] else 1  # float32 vs uint8
        bytes_per_image = self.image_size * self.image_size * self.image_channels * bytes_per_pixel
        
        # Dataset memory
        dataset_memory = num_samples * bytes_per_image / (1024**3)  # GB
        
        # Batch memory
        batch_memory = self.batch_size * bytes_per_image / (1024**3)  # GB
        
        # Cache memory (if enabled)
        cache_memory = 0
        if self.caching['enabled'] and self.caching['cache_preprocessed']:
            cache_memory = dataset_memory * 1.2  # 20% overhead for caching
        
        return {
            'per_image_mb': bytes_per_image / (1024**2),
            'dataset_gb': dataset_memory,
            'batch_gb': batch_memory,
            'cache_gb': cache_memory,
            'total_estimated_gb': dataset_memory + batch_memory + cache_memory
        }
    
    def get_data_summary(self) -> str:
        """Get a summary of the data configuration"""
        summary = f"""
Data Configuration Summary:
==========================
Dataset Path: {self.dataset_path}
Image Size: {self.image_size}x{self.image_size}x{self.image_channels}
Image Format: {self.image_format}

Data Splits:
- Train: {self.train_split:.1%}
- Validation: {self.val_split:.1%}
- Test: {self.test_split:.1%}

Failure Types: {len(self.failure_types)}
{', '.join(self.failure_types)}

Class Balancing: {'Enabled' if self.balance_classes else 'Disabled'} ({self.balancing_method})
Min Samples per Class: {self.min_samples_per_class:,}
Max Samples per Class: {self.max_samples_per_class:,}

Data Loading:
- Batch Size: {self.batch_size}
- Num Workers: {self.num_workers}
- Pin Memory: {self.pin_memory}
- Shuffle: {self.shuffle}

Preprocessing:
- Normalize: {self.preprocessing['normalize']}
- Enhance Contrast: {self.preprocessing['enhance_contrast']}
- CLAHE: {self.preprocessing['clahe']}
- Histogram Equalization: {self.preprocessing['histogram_equalization']}

Augmentation: {'Enabled' if self.augmentation['enabled'] else 'Disabled'}
- Probability: {self.augmentation['probability']:.1%}
- Geometric: Rotation, Scaling, Translation, Shearing
- Color: Brightness, Contrast, Saturation
- Noise: Gaussian Noise, Gaussian Blur, Motion Blur
- Wellbore-specific: Artifacts, Depth Variation, Lighting

Synthetic Data: {'Enabled' if self.synthetic_data['enabled'] else 'Disabled'}
Caching: {'Enabled' if self.caching['enabled'] else 'Disabled'} ({self.caching['cache_format']})
Quality Control: {'Enabled' if self.quality_control['enabled'] else 'Disabled'}
"""
        return summary