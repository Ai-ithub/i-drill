#!/usr/bin/env python3
"""Inference configuration for wellbore image generation system"""

import os
from typing import Dict, Any, List, Optional, Union, Tuple
from .base_config import BaseConfig, ConfigValidationError

class InferenceConfig(BaseConfig):
    """Configuration for inference and image generation"""
    
    def _set_defaults(self):
        """Set default inference configuration values"""
        # Model and checkpoint
        self.checkpoint_path = None
        self.model_type = 'stylegan2'  # 'stylegan2', 'stylegan3', 'custom'
        self.device = 'cuda'  # 'cuda', 'cpu', 'auto'
        self.precision = 'fp32'  # 'fp32', 'fp16', 'mixed'
        
        # Generation parameters
        self.generation = {
            'batch_size': 16,
            'num_samples': 64,
            'latent_dim': 512,
            'truncation_psi': 1.0,
            'truncation_cutoff': None,
            'noise_mode': 'random',  # 'random', 'const'
            'seed': None,  # None for random, int for reproducible
            'use_ema': True  # Use exponential moving average weights
        }
        
        # Style mixing
        self.style_mixing = {
            'enabled': False,
            'mixing_probability': 0.9,
            'crossover_point': None,  # None for random, int for fixed
            'num_mixing_layers': 2
        }
        
        # Interpolation
        self.interpolation = {
            'method': 'linear',  # 'linear', 'spherical', 'cubic'
            'num_steps': 10,
            'smooth_interpolation': True,
            'loop_interpolation': False
        }
        
        # Conditional generation (if supported)
        self.conditional = {
            'enabled': False,
            'class_labels': None,  # List of class indices or None
            'class_weights': None,  # Weights for class sampling
            'failure_types': [
                'normal', 'casing_damage', 'corrosion', 'scaling',
                'perforation_damage', 'cement_bond_failure', 
                'fluid_invasion', 'mechanical_damage'
            ]
        }
        
        # Output configuration
        self.output = {
            'directory': './generated_images',
            'format': 'png',  # 'png', 'jpg', 'tiff'
            'quality': 95,  # For JPEG
            'save_individual': True,
            'save_grid': True,
            'grid_size': 8,  # Images per row in grid
            'grid_padding': 2,
            'add_timestamp': True,
            'add_metadata': True
        }
        
        # Post-processing
        self.post_processing = {
            'enabled': True,
            'denormalize': True,
            'clamp_values': True,
            'enhance_quality': False,
            'upscale': False,
            'upscale_factor': 2,
            'upscale_method': 'bicubic',  # 'bicubic', 'bilinear', 'nearest'
            'apply_sharpening': False,
            'sharpening_strength': 0.5
        }
        
        # Quality filtering
        self.quality_filter = {
            'enabled': False,
            'min_quality_score': 0.7,
            'quality_metrics': ['fid', 'lpips', 'blur_detection'],
            'reject_blurry': True,
            'blur_threshold': 100,
            'reject_artifacts': True,
            'artifact_threshold': 0.8
        }
        
        # Batch processing
        self.batch_processing = {
            'enabled': True,
            'max_batch_size': 32,
            'memory_efficient': True,
            'clear_cache': True,
            'progress_bar': True
        }
        
        # Evaluation during inference
        self.evaluation = {
            'enabled': False,
            'reference_dataset_path': None,
            'metrics': ['fid', 'is', 'lpips'],
            'num_reference_samples': 10000,
            'save_metrics': True
        }
        
        # Visualization
        self.visualization = {
            'create_samples_grid': True,
            'create_interpolation_video': False,
            'create_style_mixing_grid': False,
            'create_latent_walk': False,
            'latent_walk_steps': 100,
            'video_fps': 30,
            'video_format': 'mp4'
        }
        
        # Advanced generation techniques
        self.advanced = {
            'progressive_generation': False,
            'multi_scale_generation': False,
            'attention_guided': False,
            'semantic_editing': False,
            'style_transfer': False
        }
        
        # Performance optimization
        self.optimization = {
            'compile_model': False,  # PyTorch 2.0 compilation
            'use_channels_last': False,
            'enable_cudnn_benchmark': True,
            'memory_format': 'contiguous',  # 'contiguous', 'channels_last'
            'gradient_checkpointing': False,
            'low_memory_mode': False
        }
        
        # Logging and monitoring
        self.logging = {
            'log_level': 'INFO',
            'log_generation_time': True,
            'log_memory_usage': True,
            'save_generation_log': True,
            'log_file': 'inference.log'
        }
    
    def validate(self):
        """Validate inference configuration"""
        # Validate checkpoint path
        if self.checkpoint_path and not os.path.exists(self.checkpoint_path):
            raise ConfigValidationError(f"Checkpoint path does not exist: {self.checkpoint_path}")
        
        # Validate model type
        valid_model_types = ['stylegan2', 'stylegan3', 'custom']
        if self.model_type not in valid_model_types:
            raise ConfigValidationError(f"Invalid model_type. Must be one of {valid_model_types}")
        
        # Validate device
        valid_devices = ['cuda', 'cpu', 'auto']
        if self.device not in valid_devices:
            raise ConfigValidationError(f"Invalid device. Must be one of {valid_devices}")
        
        # Validate precision
        valid_precisions = ['fp32', 'fp16', 'mixed']
        if self.precision not in valid_precisions:
            raise ConfigValidationError(f"Invalid precision. Must be one of {valid_precisions}")
        
        # Validate generation parameters
        gen_config = self.generation
        if gen_config['batch_size'] <= 0:
            raise ConfigValidationError("batch_size must be positive")
        
        if gen_config['num_samples'] <= 0:
            raise ConfigValidationError("num_samples must be positive")
        
        if gen_config['latent_dim'] <= 0:
            raise ConfigValidationError("latent_dim must be positive")
        
        if not (0.0 <= gen_config['truncation_psi'] <= 1.0):
            raise ConfigValidationError("truncation_psi must be between 0 and 1")
        
        if gen_config['noise_mode'] not in ['random', 'const']:
            raise ConfigValidationError("noise_mode must be 'random' or 'const'")
        
        # Validate interpolation
        interp_config = self.interpolation
        valid_methods = ['linear', 'spherical', 'cubic']
        if interp_config['method'] not in valid_methods:
            raise ConfigValidationError(f"Invalid interpolation method. Must be one of {valid_methods}")
        
        if interp_config['num_steps'] <= 1:
            raise ConfigValidationError("interpolation num_steps must be greater than 1")
        
        # Validate output configuration
        output_config = self.output
        valid_formats = ['png', 'jpg', 'jpeg', 'tiff']
        if output_config['format'] not in valid_formats:
            raise ConfigValidationError(f"Invalid output format. Must be one of {valid_formats}")
        
        if not (1 <= output_config['quality'] <= 100):
            raise ConfigValidationError("output quality must be between 1 and 100")
        
        if output_config['grid_size'] <= 0:
            raise ConfigValidationError("grid_size must be positive")
        
        # Validate post-processing
        post_config = self.post_processing
        if post_config['upscale'] and post_config['upscale_factor'] <= 1:
            raise ConfigValidationError("upscale_factor must be greater than 1")
        
        valid_upscale_methods = ['bicubic', 'bilinear', 'nearest']
        if post_config['upscale_method'] not in valid_upscale_methods:
            raise ConfigValidationError(f"Invalid upscale_method. Must be one of {valid_upscale_methods}")
        
        # Validate quality filter
        quality_config = self.quality_filter
        if not (0.0 <= quality_config['min_quality_score'] <= 1.0):
            raise ConfigValidationError("min_quality_score must be between 0 and 1")
        
        if quality_config['blur_threshold'] <= 0:
            raise ConfigValidationError("blur_threshold must be positive")
        
        # Validate batch processing
        batch_config = self.batch_processing
        if batch_config['max_batch_size'] <= 0:
            raise ConfigValidationError("max_batch_size must be positive")
        
        # Validate evaluation
        eval_config = self.evaluation
        if eval_config['enabled'] and not eval_config['reference_dataset_path']:
            raise ConfigValidationError("reference_dataset_path required when evaluation is enabled")
        
        if eval_config['num_reference_samples'] <= 0:
            raise ConfigValidationError("num_reference_samples must be positive")
    
    def get_generation_config(self) -> Dict[str, Any]:
        """Get generation-specific configuration"""
        return self.generation.copy()
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output-specific configuration"""
        return self.output.copy()
    
    def get_post_processing_config(self) -> Dict[str, Any]:
        """Get post-processing configuration"""
        return self.post_processing.copy()
    
    def create_output_directories(self):
        """Create necessary output directories"""
        base_dir = self.output['directory']
        
        directories = [
            base_dir,
            os.path.join(base_dir, 'individual'),
            os.path.join(base_dir, 'grids'),
            os.path.join(base_dir, 'interpolations'),
            os.path.join(base_dir, 'style_mixing'),
            os.path.join(base_dir, 'videos'),
            os.path.join(base_dir, 'metadata')
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def adapt_for_batch_size(self, available_memory_gb: float):
        """Adapt batch size based on available memory"""
        # Rough estimation: 1GB can handle ~4 images at 256x256x3
        estimated_batch_size = max(1, int(available_memory_gb * 4))
        
        # Don't exceed configured maximum
        max_batch = self.batch_processing['max_batch_size']
        self.generation['batch_size'] = min(estimated_batch_size, max_batch)
        
        self.logger.info(f"Adapted batch size to {self.generation['batch_size']} based on available memory")
    
    def enable_conditional_generation(self, failure_types: List[str]):
        """Enable conditional generation for specific failure types"""
        self.conditional['enabled'] = True
        self.conditional['failure_types'] = failure_types
        
        # Create class labels mapping
        self.conditional['class_labels'] = list(range(len(failure_types)))
    
    def enable_quality_filtering(self, min_score: float = 0.7):
        """Enable quality filtering for generated images"""
        self.quality_filter['enabled'] = True
        self.quality_filter['min_quality_score'] = min_score
    
    def enable_evaluation(self, reference_dataset_path: str):
        """Enable evaluation during inference"""
        if not os.path.exists(reference_dataset_path):
            raise FileNotFoundError(f"Reference dataset not found: {reference_dataset_path}")
        
        self.evaluation['enabled'] = True
        self.evaluation['reference_dataset_path'] = reference_dataset_path
    
    def set_high_quality_mode(self):
        """Configure for high-quality generation"""
        # Use higher truncation for better quality
        self.generation['truncation_psi'] = 0.7
        
        # Enable post-processing enhancements
        self.post_processing['enhance_quality'] = True
        self.post_processing['apply_sharpening'] = True
        
        # Enable quality filtering
        self.quality_filter['enabled'] = True
        self.quality_filter['min_quality_score'] = 0.8
        
        # Use smaller batch size for better quality
        self.generation['batch_size'] = min(self.generation['batch_size'], 8)
    
    def set_fast_mode(self):
        """Configure for fast generation"""
        # Disable quality enhancements
        self.post_processing['enhance_quality'] = False
        self.post_processing['apply_sharpening'] = False
        
        # Disable quality filtering
        self.quality_filter['enabled'] = False
        
        # Use larger batch size
        self.generation['batch_size'] = self.batch_processing['max_batch_size']
        
        # Enable memory efficient mode
        self.batch_processing['memory_efficient'] = True
        self.optimization['low_memory_mode'] = True
    
    def estimate_generation_time(self, num_samples: int, device_type: str = 'gpu') -> Dict[str, float]:
        """Estimate generation time based on configuration"""
        # Base generation time per image (rough estimates)
        base_times = {
            'gpu': 0.1,  # seconds per image on GPU
            'cpu': 2.0   # seconds per image on CPU
        }
        
        base_time = base_times.get(device_type, base_times['gpu'])
        
        # Adjust for batch size
        batch_size = self.generation['batch_size']
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        # Base generation time
        generation_time = num_batches * base_time * batch_size / batch_size  # Batch efficiency
        
        # Add post-processing time
        post_processing_time = 0
        if self.post_processing['enabled']:
            post_processing_time = num_samples * 0.01  # 0.01s per image
        
        # Add quality filtering time
        quality_filter_time = 0
        if self.quality_filter['enabled']:
            quality_filter_time = num_samples * 0.05  # 0.05s per image
        
        total_time = generation_time + post_processing_time + quality_filter_time
        
        return {
            'generation_time': generation_time,
            'post_processing_time': post_processing_time,
            'quality_filter_time': quality_filter_time,
            'total_time': total_time,
            'time_per_image': total_time / num_samples
        }
    
    def get_inference_summary(self) -> str:
        """Get a summary of the inference configuration"""
        time_estimate = self.estimate_generation_time(self.generation['num_samples'])
        
        summary = f"""
Inference Configuration Summary:
===============================
Model Type: {self.model_type}
Checkpoint: {self.checkpoint_path or 'Not specified'}
Device: {self.device}
Precision: {self.precision}

Generation:
- Batch Size: {self.generation['batch_size']}
- Num Samples: {self.generation['num_samples']}
- Latent Dim: {self.generation['latent_dim']}
- Truncation Psi: {self.generation['truncation_psi']}
- Noise Mode: {self.generation['noise_mode']}
- Use EMA: {self.generation['use_ema']}

Conditional Generation: {'Enabled' if self.conditional['enabled'] else 'Disabled'}
Style Mixing: {'Enabled' if self.style_mixing['enabled'] else 'Disabled'}

Output:
- Directory: {self.output['directory']}
- Format: {self.output['format'].upper()}
- Save Individual: {self.output['save_individual']}
- Save Grid: {self.output['save_grid']} ({self.output['grid_size']}x{self.output['grid_size']})

Post-processing: {'Enabled' if self.post_processing['enabled'] else 'Disabled'}
- Enhance Quality: {self.post_processing['enhance_quality']}
- Upscale: {self.post_processing['upscale']} ({self.post_processing['upscale_factor']}x)
- Sharpening: {self.post_processing['apply_sharpening']}

Quality Filter: {'Enabled' if self.quality_filter['enabled'] else 'Disabled'}
- Min Quality Score: {self.quality_filter['min_quality_score']}
- Reject Blurry: {self.quality_filter['reject_blurry']}

Visualization:
- Samples Grid: {self.visualization['create_samples_grid']}
- Interpolation Video: {self.visualization['create_interpolation_video']}
- Style Mixing Grid: {self.visualization['create_style_mixing_grid']}

Estimated Generation Time: {time_estimate['total_time']:.1f}s ({time_estimate['time_per_image']:.2f}s per image)

Optimization:
- Compile Model: {self.optimization['compile_model']}
- Low Memory Mode: {self.optimization['low_memory_mode']}
- Memory Efficient: {self.batch_processing['memory_efficient']}
"""
        return summary