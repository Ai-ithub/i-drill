#!/usr/bin/env python3
"""Configuration manager for wellbore image generation system"""

import os
import json
import yaml
import logging
from typing import Dict, Any, Optional, Union, Type, List
from pathlib import Path

from .base_config import BaseConfig, ConfigValidationError
from .model_config import ModelConfig
from .training_config import TrainingConfig
from .data_config import DataConfig
from .inference_config import InferenceConfig

class ConfigManager:
    """Central configuration manager for the wellbore image generation system"""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir) if config_dir else Path.cwd() / 'configs'
        self.logger = logging.getLogger(__name__)
        
        # Configuration instances
        self._model_config: Optional[ModelConfig] = None
        self._training_config: Optional[TrainingConfig] = None
        self._data_config: Optional[DataConfig] = None
        self._inference_config: Optional[InferenceConfig] = None
        
        # Configuration registry
        self._config_registry = {
            'model': ModelConfig,
            'training': TrainingConfig,
            'data': DataConfig,
            'inference': InferenceConfig
        }
        
        # Default configuration files
        self._default_config_files = {
            'model': 'model_config.yaml',
            'training': 'training_config.yaml',
            'data': 'data_config.yaml',
            'inference': 'inference_config.yaml'
        }
        
        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_config(self) -> ModelConfig:
        """Get model configuration"""
        if self._model_config is None:
            self._model_config = self.load_config('model')
        return self._model_config
    
    @property
    def training_config(self) -> TrainingConfig:
        """Get training configuration"""
        if self._training_config is None:
            self._training_config = self.load_config('training')
        return self._training_config
    
    @property
    def data_config(self) -> DataConfig:
        """Get data configuration"""
        if self._data_config is None:
            self._data_config = self.load_config('data')
        return self._data_config
    
    @property
    def inference_config(self) -> InferenceConfig:
        """Get inference configuration"""
        if self._inference_config is None:
            self._inference_config = self.load_config('inference')
        return self._inference_config
    
    def load_config(self, config_type: str, config_path: Optional[str] = None) -> BaseConfig:
        """Load configuration from file or create default
        
        Args:
            config_type: Type of configuration ('model', 'training', 'data', 'inference')
            config_path: Path to configuration file (optional)
            
        Returns:
            Configuration instance
        """
        if config_type not in self._config_registry:
            raise ValueError(f"Unknown config type: {config_type}")
        
        config_class = self._config_registry[config_type]
        
        # Determine config file path
        if config_path is None:
            config_path = self.config_dir / self._default_config_files[config_type]
        else:
            config_path = Path(config_path)
        
        # Load from file if exists, otherwise create default
        if config_path.exists():
            try:
                config = config_class.from_file(str(config_path))
                self.logger.info(f"Loaded {config_type} configuration from {config_path}")
            except Exception as e:
                self.logger.warning(f"Failed to load {config_type} config from {config_path}: {e}")
                self.logger.info(f"Creating default {config_type} configuration")
                config = config_class()
        else:
            self.logger.info(f"Creating default {config_type} configuration")
            config = config_class()
        
        # Validate configuration
        try:
            config.validate()
        except ConfigValidationError as e:
            self.logger.error(f"Configuration validation failed for {config_type}: {e}")
            raise
        
        return config
    
    def save_config(self, config_type: str, config_path: Optional[str] = None):
        """Save configuration to file
        
        Args:
            config_type: Type of configuration to save
            config_path: Path to save configuration (optional)
        """
        if config_type not in self._config_registry:
            raise ValueError(f"Unknown config type: {config_type}")
        
        # Get configuration instance
        config = getattr(self, f'{config_type}_config')
        
        # Determine save path
        if config_path is None:
            config_path = self.config_dir / self._default_config_files[config_type]
        else:
            config_path = Path(config_path)
        
        # Save configuration
        config.to_file(str(config_path))
        self.logger.info(f"Saved {config_type} configuration to {config_path}")
    
    def save_all_configs(self):
        """Save all loaded configurations to their default files"""
        for config_type in self._config_registry.keys():
            if getattr(self, f'_{config_type}_config') is not None:
                self.save_config(config_type)
    
    def create_experiment_config(self, experiment_name: str, 
                               base_configs: Optional[Dict[str, str]] = None) -> str:
        """Create a complete experiment configuration
        
        Args:
            experiment_name: Name of the experiment
            base_configs: Dictionary mapping config types to file paths
            
        Returns:
            Path to the experiment configuration directory
        """
        experiment_dir = self.config_dir / 'experiments' / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or create configurations
        configs = {}
        for config_type in self._config_registry.keys():
            if base_configs and config_type in base_configs:
                config = self.load_config(config_type, base_configs[config_type])
            else:
                config = self.load_config(config_type)
            
            configs[config_type] = config
            
            # Save to experiment directory
            config_file = experiment_dir / f'{config_type}_config.yaml'
            config.to_file(str(config_file))
        
        # Create experiment metadata
        metadata = {
            'experiment_name': experiment_name,
            'created_at': str(Path().cwd()),
            'config_files': {
                config_type: f'{config_type}_config.yaml'
                for config_type in self._config_registry.keys()
            }
        }
        
        metadata_file = experiment_dir / 'experiment_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Created experiment configuration: {experiment_dir}")
        return str(experiment_dir)
    
    def load_experiment_config(self, experiment_name: str):
        """Load configuration from an experiment directory
        
        Args:
            experiment_name: Name of the experiment
        """
        experiment_dir = self.config_dir / 'experiments' / experiment_name
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Load all configurations from experiment directory
        for config_type in self._config_registry.keys():
            config_file = experiment_dir / f'{config_type}_config.yaml'
            if config_file.exists():
                config = self.load_config(config_type, str(config_file))
                setattr(self, f'_{config_type}_config', config)
        
        self.logger.info(f"Loaded experiment configuration: {experiment_name}")
    
    def validate_all_configs(self) -> Dict[str, bool]:
        """Validate all loaded configurations
        
        Returns:
            Dictionary mapping config types to validation results
        """
        results = {}
        
        for config_type in self._config_registry.keys():
            config = getattr(self, f'_{config_type}_config')
            if config is not None:
                try:
                    config.validate()
                    results[config_type] = True
                    self.logger.info(f"{config_type} configuration is valid")
                except ConfigValidationError as e:
                    results[config_type] = False
                    self.logger.error(f"{config_type} configuration validation failed: {e}")
            else:
                results[config_type] = None
        
        return results
    
    def get_compatibility_report(self) -> Dict[str, Any]:
        """Check compatibility between different configurations
        
        Returns:
            Compatibility report
        """
        report = {
            'compatible': True,
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Check if configurations are loaded
        model_config = getattr(self, '_model_config')
        training_config = getattr(self, '_training_config')
        data_config = getattr(self, '_data_config')
        inference_config = getattr(self, '_inference_config')
        
        if not all([model_config, training_config, data_config]):
            report['issues'].append("Not all required configurations are loaded")
            report['compatible'] = False
            return report
        
        # Check image size compatibility
        model_img_size = model_config.image_size
        data_img_size = data_config.image.get('target_size', [256, 256])
        
        if model_img_size != data_img_size[0] or model_img_size != data_img_size[1]:
            report['issues'].append(
                f"Image size mismatch: model expects {model_img_size}x{model_img_size}, "
                f"data config has {data_img_size[0]}x{data_img_size[1]}"
            )
            report['compatible'] = False
        
        # Check batch size compatibility
        training_batch = training_config.batch_size
        data_batch = data_config.data_loading.get('batch_size', 32)
        
        if training_batch != data_batch:
            report['warnings'].append(
                f"Batch size mismatch: training config has {training_batch}, "
                f"data config has {data_batch}"
            )
        
        # Check device compatibility
        training_device = training_config.device
        if inference_config:
            inference_device = inference_config.device
            if training_device != inference_device and inference_device != 'auto':
                report['warnings'].append(
                    f"Device mismatch: training uses {training_device}, "
                    f"inference uses {inference_device}"
                )
        
        # Check memory requirements
        if training_config.mixed_precision['enabled'] and model_config.precision != 'fp16':
            report['recommendations'].append(
                "Consider setting model precision to 'fp16' when using mixed precision training"
            )
        
        # Check data augmentation compatibility with model
        if data_config.augmentation['enabled']:
            strong_augs = [
                'random_rotation', 'random_perspective', 'elastic_transform'
            ]
            enabled_strong_augs = [
                aug for aug in strong_augs 
                if data_config.augmentation.get(aug, {}).get('enabled', False)
            ]
            
            if enabled_strong_augs and model_config.architecture == 'stylegan2':
                report['warnings'].append(
                    f"Strong augmentations {enabled_strong_augs} may affect StyleGAN2 training quality"
                )
        
        # Check progressive growing compatibility
        if (training_config.progressive_growing['enabled'] and 
            not model_config.progressive_growing['enabled']):
            report['issues'].append(
                "Progressive growing enabled in training but not supported by model"
            )
            report['compatible'] = False
        
        return report
    
    def auto_adjust_configs(self):
        """Automatically adjust configurations for compatibility"""
        compatibility_report = self.get_compatibility_report()
        
        if not compatibility_report['compatible']:
            self.logger.warning("Configurations are not compatible. Attempting auto-adjustment...")
            
            model_config = self.model_config
            training_config = self.training_config
            data_config = self.data_config
            
            # Adjust image sizes
            model_img_size = model_config.image_size
            data_config.image['target_size'] = [model_img_size, model_img_size]
            
            # Adjust batch sizes
            training_batch = training_config.batch_size
            data_config.data_loading['batch_size'] = training_batch
            
            # Adjust precision settings
            if training_config.mixed_precision['enabled']:
                model_config.precision = 'fp16'
            
            self.logger.info("Auto-adjustment completed. Please validate configurations again.")
    
    def export_config_summary(self, output_path: Optional[str] = None) -> str:
        """Export a summary of all configurations
        
        Args:
            output_path: Path to save the summary (optional)
            
        Returns:
            Configuration summary as string
        """
        summary_parts = []
        
        # Header
        summary_parts.append("Wellbore Image Generation System - Configuration Summary")
        summary_parts.append("=" * 60)
        summary_parts.append("")
        
        # Model configuration summary
        if self._model_config:
            summary_parts.append("MODEL CONFIGURATION:")
            summary_parts.append("-" * 20)
            summary_parts.append(self.model_config.get_model_summary())
            summary_parts.append("")
        
        # Training configuration summary
        if self._training_config:
            summary_parts.append("TRAINING CONFIGURATION:")
            summary_parts.append("-" * 23)
            summary_parts.append(self.training_config.get_training_summary())
            summary_parts.append("")
        
        # Data configuration summary
        if self._data_config:
            summary_parts.append("DATA CONFIGURATION:")
            summary_parts.append("-" * 19)
            summary_parts.append(self.data_config.get_data_summary())
            summary_parts.append("")
        
        # Inference configuration summary
        if self._inference_config:
            summary_parts.append("INFERENCE CONFIGURATION:")
            summary_parts.append("-" * 24)
            summary_parts.append(self.inference_config.get_inference_summary())
            summary_parts.append("")
        
        # Compatibility report
        compatibility_report = self.get_compatibility_report()
        summary_parts.append("COMPATIBILITY REPORT:")
        summary_parts.append("-" * 21)
        summary_parts.append(f"Compatible: {compatibility_report['compatible']}")
        
        if compatibility_report['issues']:
            summary_parts.append("Issues:")
            for issue in compatibility_report['issues']:
                summary_parts.append(f"  - {issue}")
        
        if compatibility_report['warnings']:
            summary_parts.append("Warnings:")
            for warning in compatibility_report['warnings']:
                summary_parts.append(f"  - {warning}")
        
        if compatibility_report['recommendations']:
            summary_parts.append("Recommendations:")
            for rec in compatibility_report['recommendations']:
                summary_parts.append(f"  - {rec}")
        
        summary = "\n".join(summary_parts)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(summary)
            self.logger.info(f"Configuration summary saved to {output_path}")
        
        return summary
    
    def reset_config(self, config_type: str):
        """Reset a configuration to default values
        
        Args:
            config_type: Type of configuration to reset
        """
        if config_type not in self._config_registry:
            raise ValueError(f"Unknown config type: {config_type}")
        
        # Create new default configuration
        config_class = self._config_registry[config_type]
        new_config = config_class()
        
        # Replace current configuration
        setattr(self, f'_{config_type}_config', new_config)
        
        self.logger.info(f"Reset {config_type} configuration to defaults")
    
    def reset_all_configs(self):
        """Reset all configurations to default values"""
        for config_type in self._config_registry.keys():
            self.reset_config(config_type)
        
        self.logger.info("Reset all configurations to defaults")
    
    def list_experiments(self) -> List[str]:
        """List available experiment configurations
        
        Returns:
            List of experiment names
        """
        experiments_dir = self.config_dir / 'experiments'
        
        if not experiments_dir.exists():
            return []
        
        experiments = []
        for item in experiments_dir.iterdir():
            if item.is_dir() and (item / 'experiment_metadata.json').exists():
                experiments.append(item.name)
        
        return sorted(experiments)
    
    def delete_experiment(self, experiment_name: str):
        """Delete an experiment configuration
        
        Args:
            experiment_name: Name of the experiment to delete
        """
        experiment_dir = self.config_dir / 'experiments' / experiment_name
        
        if not experiment_dir.exists():
            raise FileNotFoundError(f"Experiment not found: {experiment_name}")
        
        # Remove experiment directory
        import shutil
        shutil.rmtree(experiment_dir)
        
        self.logger.info(f"Deleted experiment: {experiment_name}")
    
    def __str__(self) -> str:
        """String representation of configuration manager"""
        loaded_configs = []
        for config_type in self._config_registry.keys():
            if getattr(self, f'_{config_type}_config') is not None:
                loaded_configs.append(config_type)
        
        return f"ConfigManager(config_dir={self.config_dir}, loaded_configs={loaded_configs})"