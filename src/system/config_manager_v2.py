#!/usr/bin/env python3
"""
🔧 JARVIS Configuration Manager v2.0
Advanced configuration system with dynamic loading, validation, and environment management

Features:
- Multi-environment support (dev/staging/prod)
- Schema validation with Pydantic
- Hot configuration reloading
- Secure credential management
- Performance optimization settings
- Model configuration management

Version: 2.0.0 (2025 Edition)
Author: JARVIS Development Team
"""

import os
import yaml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass
from pydantic import BaseModel, Field, validator
from enum import Enum
import platform
import psutil

class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    """Logging levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

class ModelProvider(str, Enum):
    """AI Model providers"""
    LOCAL = "local"
    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"

class VoiceConfig(BaseModel):
    """Voice system configuration"""
    wake_word: str = Field(default="hey jarvis", description="Wake word for activation")
    language: str = Field(default="th", description="Primary language")
    fallback_language: str = Field(default="en", description="Fallback language")
    voice_speed: float = Field(default=1.0, ge=0.5, le=2.0, description="TTS speed multiplier")
    voice_volume: float = Field(default=0.8, ge=0.0, le=1.0, description="Voice volume")
    
    # Recognition settings
    recognition_timeout: float = Field(default=5.0, description="Speech recognition timeout")
    recognition_threshold: float = Field(default=0.7, description="Recognition confidence threshold")
    
    # TTS settings
    tts_model: str = Field(default="tts_models/multilingual/multi-dataset/xtts_v2")
    voice_character: str = Field(default="jarvis", description="Voice character profile")

class ModelConfig(BaseModel):
    """AI Model configuration"""
    # LLM Settings
    llm_model: str = Field(default="deepseek-ai/deepseek-r1-distill-llama-8b")
    llm_provider: ModelProvider = Field(default=ModelProvider.LOCAL)
    max_tokens: int = Field(default=2048, ge=128, le=8192)
    context_length: int = Field(default=8192)
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    
    # Embedding Settings  
    embedding_model: str = Field(default="mixedbread-ai/mxbai-embed-large-v1")
    embedding_dimensions: int = Field(default=1024)
    
    # Whisper Settings
    whisper_model: str = Field(default="large-v3")
    whisper_language: Optional[str] = Field(default=None, description="Force whisper language")
    
    # Performance Settings
    use_gpu: bool = Field(default=True)
    gpu_memory_fraction: float = Field(default=0.8, ge=0.1, le=1.0)
    batch_size: int = Field(default=1, ge=1, le=32)
    num_threads: Optional[int] = Field(default=None, description="CPU threads for inference")

class PerformanceConfig(BaseModel):
    """Performance optimization configuration"""
    # Memory Management
    max_memory_gb: Optional[float] = Field(default=None, description="Max memory usage in GB")
    memory_cleanup_threshold: float = Field(default=0.85, description="Memory cleanup threshold")
    
    # Processing
    async_processing: bool = Field(default=True)
    max_concurrent_requests: int = Field(default=5)
    request_timeout: float = Field(default=30.0)
    
    # Caching
    enable_caching: bool = Field(default=True)
    cache_size_mb: int = Field(default=512)
    cache_ttl_hours: int = Field(default=24)

class SystemConfig(BaseModel):
    """System configuration"""
    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_file_enabled: bool = Field(default=True)
    log_file_path: str = Field(default="logs/jarvis.log")
    log_rotation_size_mb: int = Field(default=100)
    log_max_files: int = Field(default=5)
    
    # Directories
    models_directory: str = Field(default="models")
    data_directory: str = Field(default="data")
    cache_directory: str = Field(default=".cache")
    temp_directory: str = Field(default="temp")
    
    # Feature Flags
    enable_gui: bool = Field(default=True)
    enable_voice_commands: bool = Field(default=True)
    enable_continuous_listening: bool = Field(default=False)
    enable_telemetry: bool = Field(default=False)

class SecurityConfig(BaseModel):
    """Security configuration"""
    # API Keys (will be loaded from environment)
    openai_api_key: Optional[str] = Field(default=None)
    anthropic_api_key: Optional[str] = Field(default=None)
    
    # Access Control
    require_authentication: bool = Field(default=False)
    session_timeout_hours: int = Field(default=24)
    
    # Data Protection
    encrypt_conversations: bool = Field(default=True)
    data_retention_days: int = Field(default=30)
    anonymous_usage_stats: bool = Field(default=True)

class JarvisConfig(BaseModel):
    """Main JARVIS configuration"""
    # Metadata
    version: str = Field(default="2.0.0")
    environment: Environment = Field(default=Environment.DEVELOPMENT)
    
    # Configuration sections
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    system: SystemConfig = Field(default_factory=SystemConfig)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    
    # Personality Settings
    personality_name: str = Field(default="JARVIS")
    personality_style: str = Field(default="helpful_professional")
    response_style: str = Field(default="concise_informative")
    
    # User Preferences
    user_name: Optional[str] = Field(default=None)
    timezone: str = Field(default="Asia/Bangkok")
    date_format: str = Field(default="%Y-%m-%d")
    time_format: str = Field(default="%H:%M:%S")
    
    @validator('models')
    def validate_models(cls, v):
        """Validate model configuration"""
        # Auto-detect GPU availability
        try:
            import torch
            if not torch.cuda.is_available():
                v.use_gpu = False
                logging.getLogger(__name__).warning("🚨 GPU not available, switching to CPU mode")
        except ImportError:
            v.use_gpu = False
        
        return v
    
    @validator('performance')
    def validate_performance(cls, v):
        """Validate performance settings based on system capabilities"""
        # Auto-detect system resources
        memory_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        
        if v.max_memory_gb is None:
            v.max_memory_gb = memory_gb * 0.7  # Use 70% of available memory
        
        if v.max_memory_gb > memory_gb * 0.9:
            logging.getLogger(__name__).warning(
                f"⚠️ Memory limit ({v.max_memory_gb:.1f}GB) is close to system limit ({memory_gb:.1f}GB)"
            )
        
        return v

class ConfigurationManager:
    """Advanced configuration management system"""
    
    def __init__(self, 
                 config_dir: Union[str, Path] = None,
                 environment: Environment = None):
        
        self.logger = logging.getLogger(__name__)
        
        # Set paths
        self.project_root = Path(__file__).parent.parent.parent
        if config_dir:
            self.config_dir = Path(config_dir) if isinstance(config_dir, str) else config_dir
        else:
            self.config_dir = self.project_root / "config"
        self.config_dir.mkdir(exist_ok=True)
        
        # Environment detection
        self.environment = environment or self._detect_environment()
        
        # Configuration files
        self.default_config_file = self.config_dir / "default_config.yaml"
        self.env_config_file = self.config_dir / f"{self.environment.value}_config.yaml"
        self.user_config_file = self.config_dir / "user_config.yaml"
        
        # Loaded configuration
        self._config: Optional[JarvisConfig] = None
        self._config_watchers = []
        
        self.logger.info(f"🔧 ConfigurationManager v2.0 initialized")
        self.logger.info(f"📁 Config directory: {self.config_dir}")
        self.logger.info(f"🌍 Environment: {self.environment.value}")
    
    def _detect_environment(self) -> Environment:
        """Auto-detect environment based on various factors"""
        env_var = os.getenv("JARVIS_ENVIRONMENT", "").lower()
        
        if env_var in ["prod", "production"]:
            return Environment.PRODUCTION
        elif env_var in ["staging", "stage"]:
            return Environment.STAGING
        elif env_var in ["dev", "development"]:
            return Environment.DEVELOPMENT
        
        # Auto-detection based on system properties
        if os.getenv("DOCKER_CONTAINER"):
            return Environment.PRODUCTION
        elif platform.system() == "Linux" and "/home" not in str(Path.home()):
            return Environment.PRODUCTION
        else:
            return Environment.DEVELOPMENT
    
    def load_config(self) -> JarvisConfig:
        """Load configuration from multiple sources with precedence"""
        self.logger.info("📊 Loading JARVIS configuration...")
        
        config_data = {}
        
        # 1. Load default configuration
        if self.default_config_file.exists():
            with open(self.default_config_file, 'r', encoding='utf-8') as f:
                default_data = yaml.safe_load(f)
                if default_data:
                    config_data.update(default_data)
                    self.logger.info(f"✅ Loaded default config: {len(default_data)} keys")
        
        # 2. Load environment-specific configuration
        if self.env_config_file.exists():
            with open(self.env_config_file, 'r', encoding='utf-8') as f:
                env_data = yaml.safe_load(f)
                if env_data:
                    config_data = self._deep_merge(config_data, env_data)
                    self.logger.info(f"✅ Loaded {self.environment.value} config")
        
        # 3. Load user-specific configuration
        if self.user_config_file.exists():
            with open(self.user_config_file, 'r', encoding='utf-8') as f:
                user_data = yaml.safe_load(f)
                if user_data:
                    config_data = self._deep_merge(config_data, user_data)
                    self.logger.info(f"✅ Loaded user config")
        
        # 4. Apply environment variables
        env_overrides = self._load_environment_variables()
        if env_overrides:
            config_data = self._deep_merge(config_data, env_overrides)
            self.logger.info(f"✅ Applied {len(env_overrides)} environment overrides")
        
        # 5. Create and validate configuration
        try:
            self._config = JarvisConfig(**config_data)
            self.logger.info("✅ Configuration loaded and validated successfully")
            self._log_config_summary()
            return self._config
            
        except Exception as e:
            self.logger.error(f"❌ Configuration validation failed: {e}")
            # Try to load with defaults
            try:
                self._config = JarvisConfig()
                self.logger.warning("⚠️ Using default configuration due to validation errors")
                return self._config
            except Exception as fallback_error:
                self.logger.critical(f"💥 Failed to load default configuration: {fallback_error}")
                raise
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _load_environment_variables(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        env_config = {}
        
        # Map environment variables to config paths
        env_mappings = {
            'JARVIS_LOG_LEVEL': 'system.log_level',
            'JARVIS_USE_GPU': 'models.use_gpu',
            'JARVIS_MAX_TOKENS': 'models.max_tokens',
            'JARVIS_VOICE_SPEED': 'voice.voice_speed',
            'JARVIS_WAKE_WORD': 'voice.wake_word',
            'JARVIS_LANGUAGE': 'voice.language',
            'OPENAI_API_KEY': 'security.openai_api_key',
            'ANTHROPIC_API_KEY': 'security.anthropic_api_key',
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                # Parse value type
                if value.lower() in ['true', 'false']:
                    value = value.lower() == 'true'
                elif value.isdigit():
                    value = int(value)
                elif self._is_float(value):
                    value = float(value)
                
                # Set nested config value
                self._set_nested_value(env_config, config_path, value)
        
        return env_config
    
    def _is_float(self, value: str) -> bool:
        """Check if string is a valid float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _set_nested_value(self, config: Dict[str, Any], path: str, value: Any):
        """Set a nested configuration value"""
        keys = path.split('.')
        current = config
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _log_config_summary(self):
        """Log configuration summary"""
        if not self._config:
            return
        
        self.logger.info("📋 Configuration Summary:")
        self.logger.info(f"  🤖 JARVIS Version: {self._config.version}")
        self.logger.info(f"  🌍 Environment: {self._config.environment.value}")
        self.logger.info(f"  🧠 LLM Model: {self._config.models.llm_model}")
        self.logger.info(f"  📊 Embedding Model: {self._config.models.embedding_model}")
        self.logger.info(f"  🎮 GPU Enabled: {self._config.models.use_gpu}")
        self.logger.info(f"  🎙️ Wake Word: '{self._config.voice.wake_word}'")
        self.logger.info(f"  🗣️ Language: {self._config.voice.language}")
        self.logger.info(f"  📝 Log Level: {self._config.system.log_level}")
        self.logger.info(f"  💾 Max Memory: {self._config.performance.max_memory_gb:.1f}GB")
    
    def save_config(self, config: JarvisConfig = None, target: str = "user") -> bool:
        """Save configuration to file"""
        try:
            config = config or self._config
            if not config:
                raise ValueError("No configuration to save")
            
            # Choose target file
            if target == "default":
                target_file = self.default_config_file
            elif target == "environment":
                target_file = self.env_config_file
            else:  # user
                target_file = self.user_config_file
            
            # Convert to dictionary and save
            config_dict = config.dict()
            
            with open(target_file, 'w', encoding='utf-8') as f:
                yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True, indent=2)
            
            self.logger.info(f"💾 Configuration saved to {target_file}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save configuration: {e}")
            return False
    
    def create_default_config(self) -> bool:
        """Create default configuration file"""
        try:
            default_config = JarvisConfig()
            return self.save_config(default_config, "default")
        except Exception as e:
            self.logger.error(f"❌ Failed to create default config: {e}")
            return False
    
    def get_config(self) -> Optional[JarvisConfig]:
        """Get current configuration"""
        return self._config
    
    def reload_config(self) -> JarvisConfig:
        """Reload configuration from files"""
        self.logger.info("🔄 Reloading configuration...")
        return self.load_config()
    
    def validate_config(self, config_data: Dict[str, Any]) -> tuple[bool, List[str]]:
        """Validate configuration data"""
        errors = []
        
        try:
            JarvisConfig(**config_data)
            return True, []
        except Exception as e:
            errors.append(str(e))
            return False, errors
    
    def export_config_schema(self, file_path: Path = None) -> Dict[str, Any]:
        """Export configuration schema"""
        schema = JarvisConfig.schema()
        
        if file_path:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(schema, f, indent=2, ensure_ascii=False)
            self.logger.info(f"📄 Configuration schema exported to {file_path}")
        
        return schema
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system-specific configuration recommendations"""
        recommendations = {}
        
        # Memory recommendations
        memory_gb = psutil.virtual_memory().total / (1024**3)
        if memory_gb < 8:
            recommendations['memory'] = "Consider upgrading to 16GB+ RAM for optimal performance"
        elif memory_gb >= 32:
            recommendations['memory'] = "Excellent! You have sufficient memory for all features"
        
        # GPU recommendations
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if gpu_memory < 4:
                    recommendations['gpu'] = "GPU has limited memory. Consider using smaller models"
                elif gpu_memory >= 8:
                    recommendations['gpu'] = "Great! GPU has sufficient memory for all models"
            else:
                recommendations['gpu'] = "No GPU detected. Performance may be limited with large models"
        except ImportError:
            recommendations['gpu'] = "PyTorch not available for GPU detection"
        
        # CPU recommendations
        cpu_count = psutil.cpu_count()
        if cpu_count < 4:
            recommendations['cpu'] = "Consider using lighter models or enabling GPU acceleration"
        elif cpu_count >= 8:
            recommendations['cpu'] = "Excellent CPU configuration for parallel processing"
        
        return recommendations

# Global configuration manager instance
config_manager = ConfigurationManager()