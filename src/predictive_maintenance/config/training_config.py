#!/usr/bin/env python3
"""Training configuration for StyleGAN2 wellbore image generation"""

import os
from typing import Dict, Any, List, Optional, Union
from .base_config import BaseConfig, ConfigValidationError

class TrainingConfig(BaseConfig):
    """Configuration for StyleGAN2 training process"""
    
    def _set_defaults(self):
        """Set default training configuration values"""
        # Basic training parameters
        self.batch_size = 16
        self.num_epochs = 100
        self.max_iterations = None  # If set, overrides num_epochs
        self.resume_from_checkpoint = None
        
        # Learning rates
        self.generator_lr = 0.002
        self.discriminator_lr = 0.002
        self.mapping_lr_multiplier = 0.01
        
        # Optimizers
        self.optimizer = {
            'type': 'adam',  # 'adam', 'rmsprop', 'sgd'
            'beta1': 0.0,
            'beta2': 0.99,
            'eps': 1e-8,
            'weight_decay': 0.0
        }
        
        # Learning rate scheduling
        self.lr_scheduler = {
            'enabled': False,
            'type': 'exponential',  # 'exponential', 'step', 'cosine', 'plateau'
            'gamma': 0.99,
            'step_size': 1000,
            'patience': 10,
            'min_lr': 1e-6
        }
        
        # Loss configuration
        self.loss_weights = {
            'adversarial': 1.0,
            'r1_regularization': 10.0,
            'path_length_regularization': 2.0,
            'perceptual': 0.0,  # Optional perceptual loss
            'feature_matching': 0.0  # Optional feature matching loss
        }
        
        # Regularization schedule
        self.regularization_schedule = {
            'r1_interval': 16,  # Apply R1 regularization every N iterations
            'path_interval': 4,  # Apply path length regularization every N iterations
            'lazy_regularization': True
        }
        
        # Progressive growing
        self.progressive_growing = {
            'enabled': False,
            'start_resolution': 4,
            'target_resolution': 256,
            'transition_kimg': 600,
            'stabilization_kimg': 600,
            'batch_size_schedule': {4: 128, 8: 64, 16: 32, 32: 16, 64: 8, 128: 4, 256: 2}
        }
        
        # Data loading
        self.data_loading = {
            'num_workers': 4,
            'pin_memory': True,
            'persistent_workers': True,
            'prefetch_factor': 2,
            'drop_last': True
        }
        
        # Augmentation during training
        self.augmentation = {
            'enabled': True,
            'ada_target': 0.6,  # Adaptive discriminator augmentation target
            'ada_interval': 4,
            'ada_kimg': 500,
            'probability': 0.0,  # Initial augmentation probability
            'types': ['rotation', 'translation', 'scaling', 'flipping']
        }
        
        # Checkpointing
        self.checkpointing = {
            'save_interval': 1000,  # Save every N iterations
            'max_checkpoints': 5,  # Keep only last N checkpoints
            'save_best': True,
            'metric_for_best': 'fid',  # 'fid', 'is', 'loss'
            'save_optimizer': True,
            'save_scheduler': True
        }
        
        # Logging and monitoring
        self.logging = {
            'log_interval': 100,  # Log every N iterations
            'image_log_interval': 500,  # Generate sample images every N iterations
            'num_sample_images': 64,
            'sample_grid_size': 8,
            'log_gradients': False,
            'log_weights': False,
            'use_wandb': False,
            'use_tensorboard': True
        }
        
        # Evaluation
        self.evaluation = {
            'eval_interval': 2000,  # Evaluate every N iterations
            'num_eval_samples': 10000,
            'metrics': ['fid', 'is', 'lpips'],
            'real_stats_path': None,  # Path to pre-computed real image statistics
            'compute_real_stats': True
        }
        
        # Early stopping
        self.early_stopping = {
            'enabled': False,
            'patience': 20,
            'min_delta': 0.001,
            'metric': 'fid',
            'mode': 'min'  # 'min' for FID, 'max' for IS
        }
        
        # Mixed precision training
        self.mixed_precision = {
            'enabled': True,
            'loss_scale': 'dynamic',
            'init_scale': 2**16,
            'growth_factor': 2.0,
            'backoff_factor': 0.5,
            'growth_interval': 2000
        }
        
        # Distributed training
        self.distributed = {
            'enabled': False,
            'backend': 'nccl',
            'init_method': 'env://',
            'world_size': 1,
            'rank': 0,
            'local_rank': 0
        }
        
        # Paths
        self.paths = {
            'output_dir': './outputs',
            'checkpoint_dir': './checkpoints',
            'log_dir': './logs',
            'sample_dir': './samples'
        }
        
        # Reproducibility
        self.seed = 42
        self.deterministic = False
        
        # Performance optimization
        self.performance = {
            'compile_model': False,  # PyTorch 2.0 compilation
            'channels_last': False,
            'benchmark_cudnn': True,
            'gradient_checkpointing': False
        }
    
    def validate(self):
        """Validate training configuration"""
        # Validate basic parameters
        if self.batch_size <= 0:
            raise ConfigValidationError("batch_size must be positive")
        
        if self.num_epochs <= 0 and self.max_iterations is None:
            raise ConfigValidationError("Either num_epochs or max_iterations must be positive")
        
        if self.max_iterations is not None and self.max_iterations <= 0:
            raise ConfigValidationError("max_iterations must be positive")
        
        # Validate learning rates
        if self.generator_lr <= 0:
            raise ConfigValidationError("generator_lr must be positive")
        
        if self.discriminator_lr <= 0:
            raise ConfigValidationError("discriminator_lr must be positive")
        
        if not (0.0 < self.mapping_lr_multiplier <= 1.0):
            raise ConfigValidationError("mapping_lr_multiplier must be between 0 and 1")
        
        # Validate optimizer config
        valid_optimizers = ['adam', 'rmsprop', 'sgd']
        if self.optimizer['type'] not in valid_optimizers:
            raise ConfigValidationError(f"Invalid optimizer type. Must be one of {valid_optimizers}")
        
        if not (0.0 <= self.optimizer['beta1'] <= 1.0):
            raise ConfigValidationError("optimizer beta1 must be between 0 and 1")
        
        if not (0.0 <= self.optimizer['beta2'] <= 1.0):
            raise ConfigValidationError("optimizer beta2 must be between 0 and 1")
        
        # Validate loss weights
        for weight_name, weight_value in self.loss_weights.items():
            if weight_value < 0:
                raise ConfigValidationError(f"{weight_name} weight must be non-negative")
        
        # Validate regularization schedule
        reg_schedule = self.regularization_schedule
        if reg_schedule['r1_interval'] <= 0:
            raise ConfigValidationError("r1_interval must be positive")
        
        if reg_schedule['path_interval'] <= 0:
            raise ConfigValidationError("path_interval must be positive")
        
        # Validate progressive growing
        if self.progressive_growing['enabled']:
            prog_config = self.progressive_growing
            if prog_config['start_resolution'] <= 0:
                raise ConfigValidationError("start_resolution must be positive")
            
            if prog_config['target_resolution'] <= prog_config['start_resolution']:
                raise ConfigValidationError("target_resolution must be greater than start_resolution")
        
        # Validate data loading
        data_config = self.data_loading
        if data_config['num_workers'] < 0:
            raise ConfigValidationError("num_workers must be non-negative")
        
        # Validate augmentation
        aug_config = self.augmentation
        if not (0.0 <= aug_config['ada_target'] <= 1.0):
            raise ConfigValidationError("ada_target must be between 0 and 1")
        
        if not (0.0 <= aug_config['probability'] <= 1.0):
            raise ConfigValidationError("augmentation probability must be between 0 and 1")
        
        # Validate checkpointing
        checkpoint_config = self.checkpointing
        if checkpoint_config['save_interval'] <= 0:
            raise ConfigValidationError("save_interval must be positive")
        
        if checkpoint_config['max_checkpoints'] <= 0:
            raise ConfigValidationError("max_checkpoints must be positive")
        
        valid_metrics = ['fid', 'is', 'loss']
        if checkpoint_config['metric_for_best'] not in valid_metrics:
            raise ConfigValidationError(f"Invalid metric_for_best. Must be one of {valid_metrics}")
        
        # Validate logging
        log_config = self.logging
        if log_config['log_interval'] <= 0:
            raise ConfigValidationError("log_interval must be positive")
        
        if log_config['image_log_interval'] <= 0:
            raise ConfigValidationError("image_log_interval must be positive")
        
        # Validate evaluation
        eval_config = self.evaluation
        if eval_config['eval_interval'] <= 0:
            raise ConfigValidationError("eval_interval must be positive")
        
        if eval_config['num_eval_samples'] <= 0:
            raise ConfigValidationError("num_eval_samples must be positive")
        
        # Validate early stopping
        if self.early_stopping['enabled']:
            es_config = self.early_stopping
            if es_config['patience'] <= 0:
                raise ConfigValidationError("early_stopping patience must be positive")
            
            if es_config['mode'] not in ['min', 'max']:
                raise ConfigValidationError("early_stopping mode must be 'min' or 'max'")
    
    def get_optimizer_config(self, param_type: str = 'generator') -> Dict[str, Any]:
        """Get optimizer configuration for specific parameter type"""
        base_config = self.optimizer.copy()
        
        if param_type == 'generator':
            base_config['lr'] = self.generator_lr
        elif param_type == 'discriminator':
            base_config['lr'] = self.discriminator_lr
        elif param_type == 'mapping':
            base_config['lr'] = self.generator_lr * self.mapping_lr_multiplier
        else:
            raise ValueError(f"Unknown parameter type: {param_type}")
        
        return base_config
    
    def get_scheduler_config(self) -> Optional[Dict[str, Any]]:
        """Get learning rate scheduler configuration"""
        if not self.lr_scheduler['enabled']:
            return None
        
        return self.lr_scheduler.copy()
    
    def calculate_total_iterations(self, dataset_size: int) -> int:
        """Calculate total number of training iterations"""
        if self.max_iterations is not None:
            return self.max_iterations
        
        iterations_per_epoch = dataset_size // self.batch_size
        return self.num_epochs * iterations_per_epoch
    
    def get_batch_size_for_resolution(self, resolution: int) -> int:
        """Get batch size for specific resolution (for progressive growing)"""
        if not self.progressive_growing['enabled']:
            return self.batch_size
        
        batch_schedule = self.progressive_growing['batch_size_schedule']
        return batch_schedule.get(resolution, self.batch_size)
    
    def create_directories(self):
        """Create necessary directories for training"""
        for dir_key, dir_path in self.paths.items():
            os.makedirs(dir_path, exist_ok=True)
    
    def adapt_for_distributed(self, world_size: int, rank: int, local_rank: int):
        """Adapt configuration for distributed training"""
        self.distributed['enabled'] = True
        self.distributed['world_size'] = world_size
        self.distributed['rank'] = rank
        self.distributed['local_rank'] = local_rank
        
        # Adjust batch size for distributed training
        self.batch_size = self.batch_size // world_size
        
        # Adjust logging intervals
        if rank != 0:
            self.logging['use_wandb'] = False
            self.logging['use_tensorboard'] = False
    
    def enable_mixed_precision(self, enabled: bool = True):
        """Enable or disable mixed precision training"""
        self.mixed_precision['enabled'] = enabled
    
    def set_progressive_growing(self, enabled: bool, start_res: int = 4, target_res: int = 256):
        """Configure progressive growing"""
        self.progressive_growing['enabled'] = enabled
        if enabled:
            self.progressive_growing['start_resolution'] = start_res
            self.progressive_growing['target_resolution'] = target_res
    
    def get_training_summary(self) -> str:
        """Get a summary of the training configuration"""
        total_params = f"Batch Size: {self.batch_size}"
        if self.max_iterations:
            total_params += f"\nMax Iterations: {self.max_iterations:,}"
        else:
            total_params += f"\nEpochs: {self.num_epochs}"
        
        summary = f"""
Training Configuration Summary:
==============================
{total_params}
Generator LR: {self.generator_lr}
Discriminator LR: {self.discriminator_lr}
Optimizer: {self.optimizer['type']}

Regularization:
- R1 Weight: {self.loss_weights['r1_regularization']}
- Path Length Weight: {self.loss_weights['path_length_regularization']}
- R1 Interval: {self.regularization_schedule['r1_interval']}
- Path Interval: {self.regularization_schedule['path_interval']}

Augmentation: {'Enabled' if self.augmentation['enabled'] else 'Disabled'}
Progressive Growing: {'Enabled' if self.progressive_growing['enabled'] else 'Disabled'}
Mixed Precision: {'Enabled' if self.mixed_precision['enabled'] else 'Disabled'}
Distributed: {'Enabled' if self.distributed['enabled'] else 'Disabled'}

Checkpointing:
- Save Interval: {self.checkpointing['save_interval']}
- Max Checkpoints: {self.checkpointing['max_checkpoints']}
- Metric for Best: {self.checkpointing['metric_for_best']}

Logging:
- Log Interval: {self.logging['log_interval']}
- Image Log Interval: {self.logging['image_log_interval']}
- TensorBoard: {'Enabled' if self.logging['use_tensorboard'] else 'Disabled'}
- Weights & Biases: {'Enabled' if self.logging['use_wandb'] else 'Disabled'}

Evaluation:
- Eval Interval: {self.evaluation['eval_interval']}
- Metrics: {', '.join(self.evaluation['metrics'])}
- Num Eval Samples: {self.evaluation['num_eval_samples']:,}

Early Stopping: {'Enabled' if self.early_stopping['enabled'] else 'Disabled'}
"""
        return summary