#!/usr/bin/env python3
"""Model configuration for StyleGAN2 wellbore image generation"""

import torch
from typing import Dict, Any, List, Optional
from .base_config import BaseConfig, ConfigValidationError

class ModelConfig(BaseConfig):
    """Configuration for StyleGAN2 model architecture"""
    
    def _set_defaults(self):
        """Set default model configuration values"""
        # Basic model parameters
        self.latent_dim = 512
        self.image_size = 256
        self.image_channels = 3
        self.num_classes = 0  # 0 for unconditional generation
        
        # Generator configuration
        self.generator = {
            'mapping_layers': 8,
            'mapping_lr_multiplier': 0.01,
            'synthesis_layers': 14,
            'channel_multiplier': 2,
            'blur_kernel': [1, 3, 3, 1],
            'lr_multiplier': 1.0,
            'enable_noise': True,
            'randomize_noise': True,
            'activation': 'leaky_relu',
            'w_avg_beta': 0.995,
            'style_mixing_prob': 0.9,
            'truncation_psi': 1.0,
            'truncation_cutoff': None
        }
        
        # Discriminator configuration
        self.discriminator = {
            'channel_multiplier': 2,
            'blur_kernel': [1, 3, 3, 1],
            'mbstd_group_size': 4,
            'mbstd_num_channels': 1,
            'activation': 'leaky_relu',
            'architecture': 'resnet',  # 'orig', 'skip', 'resnet'
            'lr_multiplier': 1.0
        }
        
        # Progressive growing (if enabled)
        self.progressive = {
            'enabled': False,
            'start_resolution': 4,
            'target_resolution': 256,
            'transition_kimg': 600,
            'stabilization_kimg': 600
        }
        
        # Regularization
        self.regularization = {
            'r1_gamma': 10.0,
            'r2_gamma': 0.0,
            'path_regularize': 2.0,
            'path_batch_shrink': 2,
            'path_decay': 0.01
        }
        
        # Loss configuration
        self.loss = {
            'type': 'stylegan2',  # 'wgan-gp', 'lsgan', 'hinge', 'stylegan2'
            'r1_reg_weight': 10.0,
            'path_reg_weight': 2.0,
            'lazy_regularization': True,
            'lazy_reg_interval': 4
        }
        
        # Device and precision
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.mixed_precision = True
        self.fp16 = False
        
        # Model initialization
        self.init_method = 'xavier_uniform'
        self.init_gain = 1.0
        
        # Exponential moving average
        self.ema = {
            'enabled': True,
            'beta': 0.999,
            'start_iter': 0,
            'update_interval': 1
        }
    
    def validate(self):
        """Validate model configuration"""
        # Validate basic parameters
        if self.latent_dim <= 0:
            raise ConfigValidationError("latent_dim must be positive")
        
        if self.image_size <= 0 or (self.image_size & (self.image_size - 1)) != 0:
            raise ConfigValidationError("image_size must be a positive power of 2")
        
        if self.image_channels not in [1, 3]:
            raise ConfigValidationError("image_channels must be 1 or 3")
        
        if self.num_classes < 0:
            raise ConfigValidationError("num_classes must be non-negative")
        
        # Validate generator config
        gen_config = self.generator
        if gen_config['mapping_layers'] <= 0:
            raise ConfigValidationError("Generator mapping_layers must be positive")
        
        if gen_config['synthesis_layers'] <= 0:
            raise ConfigValidationError("Generator synthesis_layers must be positive")
        
        if gen_config['channel_multiplier'] <= 0:
            raise ConfigValidationError("Generator channel_multiplier must be positive")
        
        if not (0.0 <= gen_config['style_mixing_prob'] <= 1.0):
            raise ConfigValidationError("style_mixing_prob must be between 0 and 1")
        
        if not (0.0 <= gen_config['truncation_psi'] <= 1.0):
            raise ConfigValidationError("truncation_psi must be between 0 and 1")
        
        # Validate discriminator config
        disc_config = self.discriminator
        if disc_config['channel_multiplier'] <= 0:
            raise ConfigValidationError("Discriminator channel_multiplier must be positive")
        
        if disc_config['mbstd_group_size'] <= 0:
            raise ConfigValidationError("mbstd_group_size must be positive")
        
        if disc_config['architecture'] not in ['orig', 'skip', 'resnet']:
            raise ConfigValidationError("Invalid discriminator architecture")
        
        # Validate regularization
        reg_config = self.regularization
        if reg_config['r1_gamma'] < 0:
            raise ConfigValidationError("r1_gamma must be non-negative")
        
        if reg_config['path_regularize'] < 0:
            raise ConfigValidationError("path_regularize must be non-negative")
        
        # Validate loss config
        loss_config = self.loss
        valid_loss_types = ['wgan-gp', 'lsgan', 'hinge', 'stylegan2']
        if loss_config['type'] not in valid_loss_types:
            raise ConfigValidationError(f"Invalid loss type. Must be one of {valid_loss_types}")
        
        # Validate EMA config
        ema_config = self.ema
        if not (0.0 <= ema_config['beta'] <= 1.0):
            raise ConfigValidationError("EMA beta must be between 0 and 1")
        
        # Validate progressive growing
        if self.progressive['enabled']:
            prog_config = self.progressive
            if prog_config['start_resolution'] <= 0:
                raise ConfigValidationError("start_resolution must be positive")
            
            if prog_config['target_resolution'] <= prog_config['start_resolution']:
                raise ConfigValidationError("target_resolution must be greater than start_resolution")
            
            if prog_config['target_resolution'] != self.image_size:
                raise ConfigValidationError("target_resolution must match image_size")
    
    def get_generator_config(self) -> Dict[str, Any]:
        """Get generator-specific configuration"""
        return {
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
            'image_channels': self.image_channels,
            'num_classes': self.num_classes,
            **self.generator
        }
    
    def get_discriminator_config(self) -> Dict[str, Any]:
        """Get discriminator-specific configuration"""
        return {
            'image_size': self.image_size,
            'image_channels': self.image_channels,
            'num_classes': self.num_classes,
            **self.discriminator
        }
    
    def get_loss_config(self) -> Dict[str, Any]:
        """Get loss-specific configuration"""
        return {
            **self.loss,
            **self.regularization
        }
    
    def calculate_model_size(self) -> Dict[str, int]:
        """Calculate approximate model size in parameters"""
        # Rough estimation based on StyleGAN2 architecture
        
        # Generator parameters
        mapping_params = self.latent_dim * 512 * self.generator['mapping_layers']
        
        # Synthesis network parameters (rough estimate)
        synthesis_params = 0
        channels = [512, 512, 512, 512, 256, 128, 64, 32, 16]
        for i in range(len(channels) - 1):
            # Conv layers
            synthesis_params += channels[i] * channels[i+1] * 3 * 3
            # Style modulation
            synthesis_params += 512 * channels[i] * 2
        
        generator_params = mapping_params + synthesis_params
        
        # Discriminator parameters (rough estimate)
        discriminator_params = 0
        channels_disc = [self.image_channels, 16, 32, 64, 128, 256, 512, 512, 512]
        for i in range(len(channels_disc) - 1):
            discriminator_params += channels_disc[i] * channels_disc[i+1] * 3 * 3
        
        # Final classification layer
        discriminator_params += 512 * (1 + self.num_classes)
        
        return {
            'generator': generator_params,
            'discriminator': discriminator_params,
            'total': generator_params + discriminator_params
        }
    
    def get_memory_requirements(self, batch_size: int = 1) -> Dict[str, float]:
        """Estimate memory requirements in GB"""
        # Rough estimation
        image_memory = batch_size * self.image_channels * self.image_size * self.image_size * 4  # float32
        latent_memory = batch_size * self.latent_dim * 4
        
        # Model parameters memory
        model_sizes = self.calculate_model_size()
        model_memory = (model_sizes['total'] * 4) / (1024**3)  # Convert to GB
        
        # Activation memory (rough estimate)
        activation_memory = image_memory * 10 / (1024**3)  # Multiple intermediate activations
        
        # Gradient memory (same as model parameters)
        gradient_memory = model_memory
        
        total_memory = model_memory + activation_memory + gradient_memory
        
        return {
            'model_parameters': model_memory,
            'activations': activation_memory,
            'gradients': gradient_memory,
            'total_estimated': total_memory
        }
    
    def adapt_for_resolution(self, target_resolution: int):
        """Adapt configuration for different image resolution"""
        if target_resolution <= 0 or (target_resolution & (target_resolution - 1)) != 0:
            raise ValueError("target_resolution must be a positive power of 2")
        
        self.image_size = target_resolution
        
        # Adjust synthesis layers based on resolution
        import math
        num_layers = int(math.log2(target_resolution)) * 2 - 2
        self.generator['synthesis_layers'] = max(num_layers, 2)
        
        # Update progressive growing target
        if self.progressive['enabled']:
            self.progressive['target_resolution'] = target_resolution
        
        # Re-validate
        self.validate()
    
    def create_conditional_config(self, num_classes: int):
        """Create configuration for conditional generation"""
        if num_classes <= 0:
            raise ValueError("num_classes must be positive for conditional generation")
        
        self.num_classes = num_classes
        
        # Adjust discriminator for conditional generation
        self.discriminator['conditional'] = True
        
        # Re-validate
        self.validate()
    
    def enable_mixed_precision(self, fp16: bool = False):
        """Enable mixed precision training"""
        self.mixed_precision = True
        self.fp16 = fp16
        
        if fp16 and not torch.cuda.is_available():
            self.logger.warning("FP16 requires CUDA, falling back to FP32")
            self.fp16 = False
    
    def get_architecture_summary(self) -> str:
        """Get a summary of the model architecture"""
        model_sizes = self.calculate_model_size()
        memory_req = self.get_memory_requirements()
        
        summary = f"""
StyleGAN2 Model Configuration Summary:
========================================
Image Size: {self.image_size}x{self.image_size}x{self.image_channels}
Latent Dimension: {self.latent_dim}
Conditional: {'Yes' if self.num_classes > 0 else 'No'} ({self.num_classes} classes)

Generator:
- Mapping Layers: {self.generator['mapping_layers']}
- Synthesis Layers: {self.generator['synthesis_layers']}
- Channel Multiplier: {self.generator['channel_multiplier']}
- Parameters: ~{model_sizes['generator']:,}

Discriminator:
- Architecture: {self.discriminator['architecture']}
- Channel Multiplier: {self.discriminator['channel_multiplier']}
- Parameters: ~{model_sizes['discriminator']:,}

Total Parameters: ~{model_sizes['total']:,}
Estimated Memory: {memory_req['total_estimated']:.2f} GB

Regularization:
- R1 Gamma: {self.regularization['r1_gamma']}
- Path Regularization: {self.regularization['path_regularize']}

Device: {self.device}
Mixed Precision: {self.mixed_precision}
FP16: {self.fp16}
"""
        return summary