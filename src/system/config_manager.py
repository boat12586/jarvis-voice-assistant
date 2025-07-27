"""
Configuration Manager for Jarvis Voice Assistant
Handles loading and saving of configuration files
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging


class ConfigManager:
    """Manages configuration loading and saving"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config_dir = Path(__file__).parent.parent.parent / "config"
        self.default_config_path = self.config_dir / "default_config.yaml"
        self.user_config_path = self.config_dir / "user_config.yaml"
        
        # Ensure config directory exists
        self.config_dir.mkdir(exist_ok=True)
        
        self._config = None
    
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from files"""
        try:
            # Load default configuration
            with open(self.default_config_path, 'r', encoding='utf-8') as f:
                default_config = yaml.safe_load(f)
            
            # Load user configuration if it exists
            user_config = {}
            if self.user_config_path.exists():
                try:
                    with open(self.user_config_path, 'r', encoding='utf-8') as f:
                        user_config = yaml.safe_load(f) or {}
                except Exception as e:
                    self.logger.warning(f"Failed to load user config: {e}")
            
            # Merge configurations (user overrides default)
            merged_config = self._merge_configs(default_config, user_config)
            
            self._config = merged_config
            self.logger.info("Configuration loaded successfully")
            return merged_config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Return minimal default config
            return self._get_minimal_config()
    
    def save_user_config(self, config: Dict[str, Any]) -> bool:
        """Save user configuration to file"""
        try:
            with open(self.user_config_path, 'w', encoding='utf-8') as f:
                yaml.safe_dump(config, f, default_flow_style=False, indent=2)
            
            self.logger.info("User configuration saved successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save user configuration: {e}")
            return False
    
    def get_config(self) -> Dict[str, Any]:
        """Get current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key"""
        config = self.get_config()
        return config.get(key, default)
    
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update a configuration value"""
        try:
            if self._config is None:
                self.load_config()
            
            if section not in self._config:
                self._config[section] = {}
            
            self._config[section][key] = value
            
            # Save to user config
            user_config = {}
            if self.user_config_path.exists():
                with open(self.user_config_path, 'r', encoding='utf-8') as f:
                    user_config = yaml.safe_load(f) or {}
            
            if section not in user_config:
                user_config[section] = {}
            user_config[section][key] = value
            
            return self.save_user_config(user_config)
            
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            return False
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """Merge user configuration with default configuration"""
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def _get_minimal_config(self) -> Dict[str, Any]:
        """Return minimal configuration as fallback"""
        return {
            "ui": {
                "theme": "dark",
                "opacity": 0.9,
                "scale": 1.0,
                "window": {
                    "width": 800,
                    "height": 600,
                    "always_on_top": True
                },
                "colors": {
                    "primary": "#00d4ff",
                    "secondary": "#0099cc",
                    "accent": "#ff6b35",
                    "background": "#1a1a1a",
                    "text": "#ffffff"
                }
            },
            "voice": {
                "input_device": "default",
                "output_device": "default",
                "volume": 0.8,
                "sample_rate": 16000
            },
            "ai": {
                "model_name": "mistral-7b-instruct",
                "temperature": 0.7,
                "max_tokens": 512
            },
            "system": {
                "log_level": "INFO",
                "gpu_memory_limit": 4096
            }
        }