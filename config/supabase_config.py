"""
Configuration for Supabase RAG System
Handles environment variables and configuration for cloud database integration
"""

import os
from typing import Dict, Any, Optional
from pathlib import Path
import json


class SupabaseConfig:
    """Configuration manager for Supabase RAG system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/supabase.json"
        self._config = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file and environment variables"""
        # Load from config file if exists
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    self._config = json.load(f)
            except Exception as e:
                print(f"Warning: Failed to load config file {self.config_path}: {e}")
        
        # Override with environment variables
        self._config.update({
            "supabase_url": os.getenv("SUPABASE_URL", self._config.get("supabase_url")),
            "supabase_key": os.getenv("SUPABASE_ANON_KEY", self._config.get("supabase_key")),
            "supabase_service_key": os.getenv("SUPABASE_SERVICE_KEY", self._config.get("supabase_service_key")),
            "database_url": os.getenv("DATABASE_URL", self._config.get("database_url")),
            "embedding_model": self._config.get("embedding_model", "mixedbread-ai/mxbai-embed-large-v1"),
            "chunk_size": self._config.get("chunk_size", 512),
            "chunk_overlap": self._config.get("chunk_overlap", 50),
            "top_k": self._config.get("top_k", 5),
            "similarity_threshold": self._config.get("similarity_threshold", 0.7),
            "max_documents": self._config.get("max_documents", 10000),
            "enable_logging": self._config.get("enable_logging", True),
            "log_queries": self._config.get("log_queries", True),
            "auto_cleanup": self._config.get("auto_cleanup", True),
            "cleanup_interval": self._config.get("cleanup_interval", 3600),  # 1 hour
        })
    
    def get_rag_config(self) -> Dict[str, Any]:
        """Get RAG system configuration"""
        return {
            "supabase_url": self._config.get("supabase_url"),
            "supabase_key": self._config.get("supabase_key"),
            "embedding_model": self._config.get("embedding_model"),
            "chunk_size": self._config.get("chunk_size"),
            "chunk_overlap": self._config.get("chunk_overlap"),
            "top_k": self._config.get("top_k"),
            "similarity_threshold": self._config.get("similarity_threshold"),
            "enable_logging": self._config.get("enable_logging"),
            "log_queries": self._config.get("log_queries"),
        }
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return {
            "database_url": self._config.get("database_url"),
            "supabase_url": self._config.get("supabase_url"),
            "supabase_service_key": self._config.get("supabase_service_key"),
            "max_documents": self._config.get("max_documents"),
            "auto_cleanup": self._config.get("auto_cleanup"),
            "cleanup_interval": self._config.get("cleanup_interval"),
        }
    
    def is_configured(self) -> bool:
        """Check if Supabase is properly configured"""
        return bool(self._config.get("supabase_url") and self._config.get("supabase_key"))
    
    def validate_config(self) -> tuple[bool, str]:
        """Validate configuration"""
        if not self._config.get("supabase_url"):
            return False, "Supabase URL is required"
        
        if not self._config.get("supabase_key"):
            return False, "Supabase anon key is required"
        
        if not self._config.get("supabase_url").startswith("https://"):
            return False, "Supabase URL must start with https://"
        
        if self._config.get("chunk_size", 0) < 100:
            return False, "Chunk size must be at least 100 characters"
        
        if self._config.get("chunk_overlap", 0) < 0:
            return False, "Chunk overlap cannot be negative"
        
        if not 0 < self._config.get("similarity_threshold", 0) <= 1:
            return False, "Similarity threshold must be between 0 and 1"
        
        if self._config.get("top_k", 0) < 1:
            return False, "top_k must be at least 1"
        
        return True, "Configuration is valid"
    
    def create_sample_config(self, config_path: Optional[str] = None) -> str:
        """Create a sample configuration file"""
        sample_config = {
            "supabase_url": "https://your-project.supabase.co",
            "supabase_key": "your-anon-key-here",
            "supabase_service_key": "your-service-key-here",
            "database_url": "postgresql://postgres:[password]@db.[project].supabase.co:5432/postgres",
            "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
            "similarity_threshold": 0.7,
            "max_documents": 10000,
            "enable_logging": True,
            "log_queries": True,
            "auto_cleanup": True,
            "cleanup_interval": 3600
        }
        
        config_file = config_path or self.config_path
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        with open(config_file, 'w') as f:
            json.dump(sample_config, f, indent=2)
        
        return config_file
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return self._config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set configuration value"""
        self._config[key] = value
    
    def save_config(self, config_path: Optional[str] = None):
        """Save current configuration to file"""
        config_file = config_path or self.config_path
        os.makedirs(os.path.dirname(config_file), exist_ok=True)
        
        # Remove sensitive keys for saving
        safe_config = self._config.copy()
        safe_config.pop("supabase_service_key", None)
        
        with open(config_file, 'w') as f:
            json.dump(safe_config, f, indent=2)


# Default configuration instance
default_config = SupabaseConfig()


def get_supabase_rag_config() -> Dict[str, Any]:
    """Get default Supabase RAG configuration"""
    return default_config.get_rag_config()


def is_supabase_configured() -> bool:
    """Check if Supabase is configured"""
    return default_config.is_configured()


def validate_supabase_config() -> tuple[bool, str]:
    """Validate Supabase configuration"""
    return default_config.validate_config()


def create_sample_supabase_config() -> str:
    """Create sample Supabase configuration file"""
    return default_config.create_sample_config()


# Environment setup instructions
SETUP_INSTRUCTIONS = """
Supabase RAG System Setup Instructions:

1. Create a Supabase project at https://supabase.com

2. Set up environment variables:
   export SUPABASE_URL="https://your-project.supabase.co"
   export SUPABASE_ANON_KEY="your-anon-key"
   export SUPABASE_SERVICE_KEY="your-service-key"

3. Run the database schema:
   - Go to your Supabase project's SQL Editor
   - Execute the contents of database/supabase_schema.sql

4. Install required dependencies:
   pip install supabase psycopg2-binary

5. Create configuration file (optional):
   Create config/supabase.json with your settings

6. Test the connection:
   python -c "from config.supabase_config import validate_supabase_config; print(validate_supabase_config())"

For more details, see the README in the database/ directory.
"""