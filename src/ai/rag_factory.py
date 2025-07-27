"""
RAG System Factory
Provides unified interface to create either local FAISS-based or Supabase-based RAG systems
"""

import os
import logging
from typing import Dict, Any, Optional, Union

from config.supabase_config import SupabaseConfig, is_supabase_configured


class RAGFactory:
    """Factory for creating RAG system instances"""
    
    @staticmethod
    def create_rag_system(config: Dict[str, Any], 
                         force_local: bool = False, 
                         force_supabase: bool = False) -> Union['RAGSystem', 'SupabaseRAGSystem']:
        """
        Create appropriate RAG system based on configuration and availability
        
        Args:
            config: RAG system configuration
            force_local: Force use of local FAISS-based system
            force_supabase: Force use of Supabase-based system
            
        Returns:
            RAG system instance (either local or Supabase-based)
            
        Raises:
            ImportError: If required dependencies are not available
            ValueError: If configuration is invalid
        """
        logger = logging.getLogger(__name__)
        
        # Check for forced selection
        if force_local and force_supabase:
            raise ValueError("Cannot force both local and Supabase RAG systems")
        
        # Determine which system to use
        use_supabase = False
        
        if force_supabase:
            use_supabase = True
        elif force_local:
            use_supabase = False
        else:
            # Auto-detect based on configuration and availability
            try:
                # Check if Supabase is configured
                supabase_config = SupabaseConfig()
                if is_supabase_configured():
                    # Check if Supabase dependencies are available
                    try:
                        import supabase
                        import psycopg2
                        use_supabase = True
                        logger.info("Using Supabase RAG system (auto-detected)")
                    except ImportError:
                        logger.warning("Supabase configured but dependencies not available, falling back to local RAG")
                        use_supabase = False
                else:
                    logger.info("Supabase not configured, using local RAG system")
                    use_supabase = False
            except Exception as e:
                logger.warning(f"Error checking Supabase configuration: {e}, falling back to local RAG")
                use_supabase = False
        
        # Create appropriate RAG system
        if use_supabase:
            return RAGFactory._create_supabase_rag(config)
        else:
            return RAGFactory._create_local_rag(config)
    
    @staticmethod
    def _create_local_rag(config: Dict[str, Any]) -> 'RAGSystem':
        """Create local FAISS-based RAG system"""
        try:
            from ai.rag_system import RAGSystem
            logger = logging.getLogger(__name__)
            logger.info("Creating local FAISS-based RAG system")
            return RAGSystem(config)
        except ImportError as e:
            raise ImportError(f"Failed to import local RAG system: {e}")
    
    @staticmethod
    def _create_supabase_rag(config: Dict[str, Any]) -> 'SupabaseRAGSystem':
        """Create Supabase-based RAG system"""
        try:
            from ai.supabase_rag_system import SupabaseRAGSystem
            from config.supabase_config import get_supabase_rag_config
            
            logger = logging.getLogger(__name__)
            logger.info("Creating Supabase-based RAG system")
            
            # Merge configuration with Supabase-specific settings
            supabase_config = get_supabase_rag_config()
            merged_config = {**config, **supabase_config}
            
            return SupabaseRAGSystem(merged_config)
        except ImportError as e:
            raise ImportError(f"Failed to import Supabase RAG system: {e}")
    
    @staticmethod
    def get_available_systems() -> Dict[str, bool]:
        """Get availability status of RAG systems"""
        systems = {
            "local": False,
            "supabase": False
        }
        
        # Check local RAG system
        try:
            from src.ai.rag_system import RAGSystem
            systems["local"] = True
            logging.info("Local RAG system available")
        except ImportError as e:
            logging.warning(f"Local RAG system not available: {e}")
            systems["local"] = False
        
        # Check Supabase RAG system
        try:
            from src.ai.supabase_rag_system import SupabaseRAGSystem
            import supabase
            import psycopg2
            if is_supabase_configured():
                systems["supabase"] = True
                logging.info("Supabase RAG system available")
            else:
                logging.info("Supabase RAG system not configured")
                systems["supabase"] = False
        except ImportError as e:
            logging.warning(f"Supabase RAG system not available: {e}")
            systems["supabase"] = False
        
        return systems
    
    @staticmethod
    def validate_config(config: Dict[str, Any], system_type: str = "auto") -> tuple[bool, str]:
        """
        Validate configuration for specified RAG system type
        
        Args:
            config: Configuration to validate
            system_type: "local", "supabase", or "auto"
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if system_type == "local":
            return RAGFactory._validate_local_config(config)
        elif system_type == "supabase":
            return RAGFactory._validate_supabase_config(config)
        elif system_type == "auto":
            # Validate for the system that would be selected
            available = RAGFactory.get_available_systems()
            if available["supabase"]:
                return RAGFactory._validate_supabase_config(config)
            elif available["local"]:
                return RAGFactory._validate_local_config(config)
            else:
                return False, "No RAG systems available"
        else:
            return False, f"Unknown system type: {system_type}"
    
    @staticmethod
    def _validate_local_config(config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate local RAG configuration"""
        required_fields = ["embedding_model"]
        
        for field in required_fields:
            if field not in config:
                return False, f"Missing required field for local RAG: {field}"
        
        # Validate chunk size
        chunk_size = config.get("chunk_size", 512)
        if chunk_size < 100:
            return False, "Chunk size must be at least 100 characters"
        
        # Validate similarity threshold
        threshold = config.get("similarity_threshold", 0.7)
        if not 0 < threshold <= 1:
            return False, "Similarity threshold must be between 0 and 1"
        
        return True, "Local RAG configuration is valid"
    
    @staticmethod
    def _validate_supabase_config(config: Dict[str, Any]) -> tuple[bool, str]:
        """Validate Supabase RAG configuration"""
        from config.supabase_config import validate_supabase_config
        
        # First validate Supabase-specific configuration
        is_valid, message = validate_supabase_config()
        if not is_valid:
            return False, f"Supabase configuration error: {message}"
        
        # Then validate general RAG configuration
        return RAGFactory._validate_local_config(config)


class RAGSystemManager:
    """Manager for RAG system lifecycle"""
    
    def __init__(self, config: Dict[str, Any], 
                 force_local: bool = False, 
                 force_supabase: bool = False):
        self.config = config
        self.force_local = force_local
        self.force_supabase = force_supabase
        self.rag_system = None
        self.logger = logging.getLogger(__name__)
    
    def initialize(self) -> bool:
        """Initialize RAG system"""
        try:
            self.rag_system = RAGFactory.create_rag_system(
                self.config, 
                self.force_local, 
                self.force_supabase
            )
            
            if hasattr(self.rag_system, 'is_ready') and self.rag_system.is_ready:
                self.logger.info("RAG system initialized successfully")
                return True
            else:
                self.logger.error("RAG system failed to initialize")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def get_system(self):
        """Get the RAG system instance"""
        return self.rag_system
    
    def shutdown(self):
        """Shutdown RAG system"""
        if self.rag_system:
            try:
                if hasattr(self.rag_system, 'shutdown'):
                    self.rag_system.shutdown()
                elif hasattr(self.rag_system, 'cleanup'):
                    self.rag_system.cleanup()
                self.logger.info("RAG system shutdown completed")
            except Exception as e:
                self.logger.error(f"Error during RAG system shutdown: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        if not self.rag_system:
            return {"error": "RAG system not initialized"}
        
        stats = {}
        if hasattr(self.rag_system, 'get_stats'):
            stats = self.rag_system.get_stats()
        
        # Add system type information
        system_type = "unknown"
        if "supabase_url" in stats:
            system_type = "supabase"
        elif "index_size" in stats:
            system_type = "local"
        
        stats["system_type"] = system_type
        return stats
    
    def reload_knowledge_base(self, knowledge_base_path: str = "data/knowledge_base.json") -> bool:
        """Reload knowledge base from file"""
        if not self.rag_system:
            self.logger.error("RAG system not initialized")
            return False
        
        try:
            import json
            
            if not os.path.exists(knowledge_base_path):
                self.logger.error(f"Knowledge base file not found: {knowledge_base_path}")
                return False
            
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            if hasattr(self.rag_system, 'add_knowledge_base'):
                self.rag_system.add_knowledge_base(knowledge_data)
                self.logger.info(f"Successfully reloaded knowledge base from {knowledge_base_path}")
                return True
            else:
                self.logger.error("RAG system does not support knowledge base loading")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to reload knowledge base: {e}")
            return False