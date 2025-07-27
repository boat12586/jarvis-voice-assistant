"""
Memory Management Utilities for JARVIS
Handles model loading, unloading, and optimization
"""

import gc
import psutil
import torch
import logging
from typing import Dict, Any, Optional
from contextlib import contextmanager

class MemoryManager:
    """Manages memory usage for JARVIS components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.loaded_models = {}
        self.memory_threshold = 0.8  # 80% memory usage threshold
        
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        memory = psutil.virtual_memory()
        
        gpu_memory = None
        if torch.cuda.is_available():
            gpu_memory = {
                "allocated": torch.cuda.memory_allocated() / 1024**3,
                "cached": torch.cuda.memory_reserved() / 1024**3,
                "total": torch.cuda.get_device_properties(0).total_memory / 1024**3
            }
        
        return {
            "ram_used_gb": memory.used / 1024**3,
            "ram_total_gb": memory.total / 1024**3,
            "ram_percent": memory.percent,
            "gpu_memory": gpu_memory
        }
    
    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure"""
        usage = self.get_memory_usage()
        return usage["ram_percent"] > self.memory_threshold * 100
    
    @contextmanager
    def memory_context(self, operation_name: str):
        """Context manager for memory-intensive operations"""
        start_memory = self.get_memory_usage()
        self.logger.info(f"Starting {operation_name} - RAM: {start_memory['ram_percent']:.1f}%")
        
        try:
            yield
        finally:
            # Force garbage collection
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            end_memory = self.get_memory_usage()
            self.logger.info(f"Finished {operation_name} - RAM: {end_memory['ram_percent']:.1f}%")
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.logger.info(f"Unloaded model: {model_name}")
    
    def register_model(self, model_name: str, model: Any):
        """Register a loaded model"""
        self.loaded_models[model_name] = model
        self.logger.info(f"Registered model: {model_name}")
    
    def cleanup_memory(self):
        """Force memory cleanup"""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.logger.info("Memory cleanup completed")

# Global memory manager instance
memory_manager = MemoryManager()
