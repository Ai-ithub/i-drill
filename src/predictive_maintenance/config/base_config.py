#!/usr/bin/env python3
"""Base configuration class for wellbore image generation system"""

import os
import json
import yaml
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
from pathlib import Path
import logging

class BaseConfig(ABC):
    """Abstract base class for all configuration classes"""
    
    def __init__(self, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize base configuration
        
        Args:
            config_dict: Optional dictionary of configuration values
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set default values
        self._set_defaults()
        
        # Override with provided config
        if config_dict:
            self.update_from_dict(config_dict)
        
        # Validate configuration
        self.validate()
    
    @abstractmethod
    def _set_defaults(self):
        """Set default configuration values"""
        pass
    
    @abstractmethod
    def validate(self):
        """Validate configuration values"""
        pass
    
    def update_from_dict(self, config_dict: Dict[str, Any]):
        """Update configuration from dictionary"""
        for key, value in config_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.logger.warning(f"Unknown configuration key: {key}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_') and key != 'logger':
                config_dict[key] = value
        return config_dict
    
    def save_to_file(self, filepath: Union[str, Path], format: str = 'yaml'):
        """Save configuration to file
        
        Args:
            filepath: Path to save configuration
            format: File format ('yaml' or 'json')
        """
        filepath = Path(filepath)
        config_dict = self.to_dict()
        
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'yaml':
            with open(filepath, 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        elif format.lower() == 'json':
            with open(filepath, 'w') as f:
                json.dump(config_dict, f, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        self.logger.info(f"Configuration saved to {filepath}")
    
    @classmethod
    def load_from_file(cls, filepath: Union[str, Path]):
        """Load configuration from file
        
        Args:
            filepath: Path to configuration file
            
        Returns:
            Configuration instance
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        # Determine format from extension
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r') as f:
                config_dict = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r') as f:
                config_dict = json.load(f)
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return cls(config_dict)
    
    def merge_with(self, other_config: 'BaseConfig'):
        """Merge with another configuration
        
        Args:
            other_config: Another configuration instance
        """
        other_dict = other_config.to_dict()
        self.update_from_dict(other_dict)
        self.validate()
    
    def get_nested_value(self, key_path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path (e.g., 'model.generator.lr')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self
        
        try:
            for key in keys:
                if hasattr(value, key):
                    value = getattr(value, key)
                elif isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            return value
        except (AttributeError, KeyError, TypeError):
            return default
    
    def set_nested_value(self, key_path: str, value: Any):
        """Set nested configuration value using dot notation
        
        Args:
            key_path: Dot-separated key path
            value: Value to set
        """
        keys = key_path.split('.')
        current = self
        
        # Navigate to parent of target key
        for key in keys[:-1]:
            if not hasattr(current, key):
                setattr(current, key, {})
            current = getattr(current, key)
        
        # Set final value
        final_key = keys[-1]
        if isinstance(current, dict):
            current[final_key] = value
        else:
            setattr(current, final_key, value)
    
    def __str__(self) -> str:
        """String representation of configuration"""
        config_dict = self.to_dict()
        return f"{self.__class__.__name__}({config_dict})"
    
    def __repr__(self) -> str:
        """Detailed string representation"""
        return self.__str__()
    
    def copy(self) -> 'BaseConfig':
        """Create a copy of the configuration"""
        config_dict = self.to_dict()
        return self.__class__(config_dict)
    
    def freeze(self):
        """Make configuration read-only"""
        self._frozen = True
    
    def is_frozen(self) -> bool:
        """Check if configuration is frozen"""
        return getattr(self, '_frozen', False)
    
    def __setattr__(self, name: str, value: Any):
        """Override setattr to prevent modification of frozen configs"""
        if hasattr(self, '_frozen') and self._frozen and not name.startswith('_'):
            raise AttributeError(f"Cannot modify frozen configuration: {name}")
        super().__setattr__(name, value)

class ConfigValidationError(Exception):
    """Exception raised when configuration validation fails"""
    pass

class ConfigurationMixin:
    """Mixin class to add configuration capabilities to other classes"""
    
    def __init__(self, config: Optional[BaseConfig] = None, **kwargs):
        self.config = config
        super().__init__(**kwargs)
    
    def update_config(self, **kwargs):
        """Update configuration with keyword arguments"""
        if self.config is not None:
            for key, value in kwargs.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        if self.config is not None:
            return getattr(self.config, key, default)
        return default