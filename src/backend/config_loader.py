"""
Configuration loader for Kafka and database settings
"""
import yaml
import os
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads configuration from YAML files with environment variable substitution"""
    
    def __init__(self, config_path: str = None):
        if config_path is None:
            # Default to config directory in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config" / "kafka_config.yaml"
        
        self.config_path = Path(config_path)
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self._config is not None:
            return self._config
            
        try:
            with open(self.config_path, 'r', encoding='utf-8') as file:
                config_content = file.read()
                
            # Replace environment variables
            config_content = self._substitute_env_vars(config_content)
            
            self._config = yaml.safe_load(config_content)
            logger.info(f"Configuration loaded from {self.config_path}")
            return self._config
            
        except FileNotFoundError:
            logger.error(f"Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML configuration: {e}")
            raise
    
    def _substitute_env_vars(self, content: str) -> str:
        """Replace ${VAR_NAME} with environment variable values"""
        import re
        
        def replace_env_var(match):
            var_name = match.group(1)
            return os.getenv(var_name, match.group(0))  # Return original if not found
        
        return re.sub(r'\$\{([^}]+)\}', replace_env_var, content)
    
    def get_kafka_config(self) -> Dict[str, Any]:
        """Get Kafka configuration"""
        config = self.load_config()
        return config.get('kafka', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        config = self.load_config()
        return config.get('database', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration"""
        config = self.load_config()
        return config.get('logging', {})

# Global instance
config_loader = ConfigLoader()