"""
Memory Optimization System for JARVIS Voice Assistant
Manages memory usage across all components with intelligent cleanup
"""

import os
import gc
import sys
import psutil
import torch
import logging
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import numpy as np

# Enhanced logging support
try:
    from src.system.enhanced_logger import ComponentLogger
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False


@dataclass
class MemoryConfig:
    """Memory optimization configuration"""
    max_gpu_memory_gb: float = 6.0
    max_cpu_memory_gb: float = 8.0
    cleanup_threshold: float = 0.8  # Trigger cleanup at 80%
    aggressive_cleanup_threshold: float = 0.9  # Aggressive cleanup at 90%
    monitoring_interval: int = 30  # seconds
    enable_auto_cleanup: bool = True
    enable_model_unloading: bool = True
    cache_cleanup_age: int = 300  # seconds
    log_memory_usage: bool = True


class MemoryTracker:
    """Track memory usage across different components"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Memory tracking
        self.component_memory: Dict[str, float] = {}
        self.peak_memory: Dict[str, float] = {}
        self.memory_history: List[Dict[str, Any]] = []
        
        # Process info
        self.process = psutil.Process()
        
    def update_component_memory(self, component: str, memory_mb: float):
        """Update memory usage for a component"""
        self.component_memory[component] = memory_mb
        
        # Track peak usage
        if component not in self.peak_memory or memory_mb > self.peak_memory[component]:
            self.peak_memory[component] = memory_mb
    
    def get_system_memory(self) -> Dict[str, float]:
        """Get current system memory usage"""
        try:
            # CPU memory
            memory_info = self.process.memory_info()
            cpu_memory_mb = memory_info.rss / (1024 * 1024)
            
            # System memory
            system_memory = psutil.virtual_memory()
            system_available_gb = system_memory.available / (1024 * 1024 * 1024)
            system_used_percent = system_memory.percent
            
            # GPU memory
            gpu_memory_mb = 0
            gpu_available_mb = 0
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                    gpu_total_mb = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
                    gpu_available_mb = gpu_total_mb - gpu_memory_mb
                except:
                    pass
            
            return {
                "cpu_memory_mb": cpu_memory_mb,
                "gpu_memory_mb": gpu_memory_mb,
                "gpu_available_mb": gpu_available_mb,
                "system_available_gb": system_available_gb,
                "system_used_percent": system_used_percent
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system memory: {e}")
            return {}
    
    def is_memory_pressure(self) -> tuple[bool, str]:
        """Check if system is under memory pressure"""
        try:
            memory_info = self.get_system_memory()
            
            # Check CPU memory
            cpu_memory_gb = memory_info.get("cpu_memory_mb", 0) / 1024
            if cpu_memory_gb > self.config.max_cpu_memory_gb * self.config.cleanup_threshold:
                return True, "cpu"
            
            # Check GPU memory
            gpu_memory_gb = memory_info.get("gpu_memory_mb", 0) / 1024
            if gpu_memory_gb > self.config.max_gpu_memory_gb * self.config.cleanup_threshold:
                return True, "gpu"
            
            # Check system memory
            system_used = memory_info.get("system_used_percent", 0)
            if system_used > 85:  # 85% system memory usage
                return True, "system"
            
            return False, ""
            
        except Exception as e:
            self.logger.error(f"Memory pressure check failed: {e}")
            return False, "error"
    
    def log_memory_status(self):
        """Log current memory status"""
        if not self.config.log_memory_usage:
            return
        
        try:
            memory_info = self.get_system_memory()
            
            self.logger.info(f"💾 Memory Status:")
            self.logger.info(f"  CPU: {memory_info.get('cpu_memory_mb', 0):.1f}MB")
            self.logger.info(f"  GPU: {memory_info.get('gpu_memory_mb', 0):.1f}MB")
            self.logger.info(f"  System: {memory_info.get('system_used_percent', 0):.1f}% used")
            
            # Log component breakdown
            if self.component_memory:
                self.logger.info("  Components:")
                for component, memory in self.component_memory.items():
                    self.logger.info(f"    {component}: {memory:.1f}MB")
            
        except Exception as e:
            self.logger.error(f"Memory status logging failed: {e}")


class ModelManager:
    """Manage model loading and unloading for memory optimization"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Model tracking
        self.loaded_models: Dict[str, Any] = {}
        self.model_sizes: Dict[str, float] = {}  # MB
        self.model_last_used: Dict[str, float] = {}
        self.model_priorities: Dict[str, int] = {}  # Lower = higher priority
        
    def register_model(self, name: str, model: Any, size_mb: float, priority: int = 5):
        """Register a model for memory management"""
        self.loaded_models[name] = model
        self.model_sizes[name] = size_mb
        self.model_priorities[name] = priority
        self.model_last_used[name] = time.time()
        
        self.logger.info(f"📝 Registered model '{name}': {size_mb:.1f}MB, priority={priority}")
    
    def unregister_model(self, name: str):
        """Unregister a model"""
        if name in self.loaded_models:
            del self.loaded_models[name]
            del self.model_sizes[name]
            del self.model_priorities[name]
            del self.model_last_used[name]
            self.logger.info(f"📝 Unregistered model '{name}'")
    
    def access_model(self, name: str) -> Optional[Any]:
        """Access a model (updates last used time)"""
        if name in self.loaded_models:
            self.model_last_used[name] = time.time()
            return self.loaded_models[name]
        return None
    
    def unload_model(self, name: str) -> bool:
        """Unload a specific model to free memory"""
        if name not in self.loaded_models:
            return False
        
        try:
            model = self.loaded_models[name]
            size_mb = self.model_sizes[name]
            
            # Move to CPU if GPU model
            if hasattr(model, 'cpu'):
                model.cpu()
            
            # Clear from GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Remove references
            del self.loaded_models[name]
            del model
            
            # Force garbage collection
            gc.collect()
            
            self.logger.info(f"📤 Unloaded model '{name}': freed {size_mb:.1f}MB")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload model '{name}': {e}")
            return False
    
    def unload_lru_models(self, target_memory_mb: float) -> float:
        """Unload least recently used models to free target memory"""
        if not self.config.enable_model_unloading:
            return 0.0
        
        # Sort models by last used time and priority
        model_scores = []
        current_time = time.time()
        
        for name in self.loaded_models.keys():
            age = current_time - self.model_last_used[name]
            priority = self.model_priorities[name]
            size = self.model_sizes[name]
            
            # Score: higher age and lower priority = higher score (more likely to unload)
            score = age * (priority / 10.0)
            model_scores.append((score, name, size))
        
        # Sort by score (highest first)
        model_scores.sort(reverse=True)
        
        freed_memory = 0.0
        for score, name, size in model_scores:
            if freed_memory >= target_memory_mb:
                break
            
            if self.unload_model(name):
                freed_memory += size
        
        return freed_memory
    
    def get_model_memory_usage(self) -> float:
        """Get total memory usage of managed models"""
        return sum(self.model_sizes.values())


class CacheManager:
    """Manage various caches to optimize memory usage"""
    
    def __init__(self, config: MemoryConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Cache directories to monitor
        self.cache_dirs = [
            "data/tts_cache",
            "data/vector_cache", 
            "data/model_cache",
            "logs",
            ".cache"
        ]
    
    def cleanup_old_files(self, directory: str, max_age_seconds: int) -> int:
        """Clean up old files in a directory"""
        if not os.path.exists(directory):
            return 0
        
        cleaned_count = 0
        current_time = time.time()
        
        try:
            for root, dirs, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Check file age
                        file_age = current_time - os.path.getmtime(file_path)
                        
                        if file_age > max_age_seconds:
                            os.remove(file_path)
                            cleaned_count += 1
                            
                    except OSError:
                        continue  # Skip files we can't access
            
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup {directory}: {e}")
            return 0
    
    def cleanup_all_caches(self) -> Dict[str, int]:
        """Clean up all managed caches"""
        results = {}
        
        for cache_dir in self.cache_dirs:
            cleaned = self.cleanup_old_files(cache_dir, self.config.cache_cleanup_age)
            if cleaned > 0:
                results[cache_dir] = cleaned
                self.logger.info(f"🗑️ Cleaned {cleaned} old files from {cache_dir}")
        
        return results
    
    def get_cache_sizes(self) -> Dict[str, float]:
        """Get sizes of all cache directories"""
        sizes = {}
        
        for cache_dir in self.cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    total_size = 0
                    for root, dirs, files in os.walk(cache_dir):
                        for file in files:
                            file_path = os.path.join(root, file)
                            try:
                                total_size += os.path.getsize(file_path)
                            except OSError:
                                continue
                    
                    sizes[cache_dir] = total_size / (1024 * 1024)  # MB
                    
                except Exception as e:
                    self.logger.error(f"Failed to calculate size of {cache_dir}: {e}")
                    sizes[cache_dir] = 0
        
        return sizes


class MemoryOptimizer:
    """Main memory optimization system"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        config_dict = config or {}
        self.config = MemoryConfig(**config_dict)
        
        # Setup logging
        if ENHANCED_LOGGING:
            self.logger = ComponentLogger("memory_optimizer", config_dict)
        else:
            self.logger = logging.getLogger(__name__)
        
        # Components
        self.tracker = MemoryTracker(self.config)
        self.model_manager = ModelManager(self.config)
        self.cache_manager = CacheManager(self.config)
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self._stop_event = threading.Event()
        
        # Cleanup callbacks
        self.cleanup_callbacks: List[Callable[[], None]] = []
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize memory optimizer"""
        try:
            self.logger.info("🧠 Initializing Memory Optimizer...")
            
            # Log initial memory state
            self.tracker.log_memory_status()
            
            # Start monitoring if enabled
            if self.config.enable_auto_cleanup:
                self.start_monitoring()
            
            self.logger.info("✅ Memory Optimizer initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize memory optimizer: {e}", exception=e)
    
    def start_monitoring(self):
        """Start memory monitoring thread"""
        if self.monitoring_active:
            return
        
        try:
            self.logger.info("👁️ Starting memory monitoring...")
            
            self._stop_event.clear()
            self.monitoring_thread = threading.Thread(
                target=self._monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
            self.monitoring_active = True
            self.logger.info("✅ Memory monitoring started")
            
        except Exception as e:
            self.logger.error(f"Failed to start memory monitoring: {e}")
    
    def stop_monitoring(self):
        """Stop memory monitoring"""
        if not self.monitoring_active:
            return
        
        try:
            self.logger.info("⏹️ Stopping memory monitoring...")
            
            self._stop_event.set()
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=5.0)
            
            self.monitoring_active = False
            self.logger.info("✅ Memory monitoring stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping memory monitoring: {e}")
    
    def _monitoring_loop(self):
        """Memory monitoring loop"""
        self.logger.info("Memory monitoring loop started")
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Check memory pressure
                    has_pressure, pressure_type = self.tracker.is_memory_pressure()
                    
                    if has_pressure:
                        self.logger.warning(f"⚠️ Memory pressure detected: {pressure_type}")
                        
                        # Trigger cleanup
                        if pressure_type in ["cpu", "gpu"]:
                            self.optimize_memory()
                        elif pressure_type == "system":
                            self.aggressive_cleanup()
                    
                    # Log memory status periodically
                    self.tracker.log_memory_status()
                    
                    # Wait for next check
                    if self._stop_event.wait(self.config.monitoring_interval):
                        break
                        
                except Exception as e:
                    self.logger.error(f"Error in monitoring loop: {e}")
                    time.sleep(5)  # Wait before retrying
        
        except Exception as e:
            self.logger.error(f"Monitoring loop crashed: {e}")
        
        self.logger.info("Memory monitoring loop ended")
    
    def optimize_memory(self) -> Dict[str, Any]:
        """Perform standard memory optimization"""
        self.logger.info("🧹 Starting memory optimization...")
        
        results = {}
        freed_memory = 0.0
        
        try:
            # 1. Run garbage collection
            collected = gc.collect()
            results["gc_collected"] = collected
            
            # 2. Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                results["gpu_cache_cleared"] = True
            
            # 3. Unload least recently used models
            target_memory = 500  # MB to free
            model_freed = self.model_manager.unload_lru_models(target_memory)
            results["model_memory_freed_mb"] = model_freed
            freed_memory += model_freed
            
            # 4. Clean up old cache files
            cache_results = self.cache_manager.cleanup_all_caches()
            results["cache_cleanup"] = cache_results
            
            # 5. Run cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    callback()
                except Exception as e:
                    self.logger.error(f"Cleanup callback failed: {e}")
            
            self.logger.info(f"✅ Memory optimization completed: freed {freed_memory:.1f}MB")
            
            # Log final memory state
            self.tracker.log_memory_status()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Memory optimization failed: {e}", exception=e)
            return {"error": str(e)}
    
    def aggressive_cleanup(self) -> Dict[str, Any]:
        """Perform aggressive memory cleanup for critical situations"""
        self.logger.warning("🚨 Starting aggressive memory cleanup...")
        
        # First run standard optimization
        results = self.optimize_memory()
        
        try:
            # Unload ALL non-essential models
            essential_models = ["ai_engine", "embedding_model"]  # Keep only essential
            
            models_to_unload = [
                name for name in self.model_manager.loaded_models.keys()
                if name not in essential_models
            ]
            
            for model_name in models_to_unload:
                self.model_manager.unload_model(model_name)
            
            results["aggressive_model_unload"] = len(models_to_unload)
            
            # Force multiple garbage collection cycles
            for i in range(3):
                collected = gc.collect()
                results[f"gc_cycle_{i}"] = collected
            
            # Clear all GPU memory
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            
            self.logger.warning("🚨 Aggressive cleanup completed")
            return results
            
        except Exception as e:
            self.logger.error(f"Aggressive cleanup failed: {e}")
            results["aggressive_error"] = str(e)
            return results
    
    def register_cleanup_callback(self, callback: Callable[[], None]):
        """Register a cleanup callback function"""
        self.cleanup_callbacks.append(callback)
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get comprehensive memory statistics"""
        try:
            system_memory = self.tracker.get_system_memory()
            model_memory = self.model_manager.get_model_memory_usage()
            cache_sizes = self.cache_manager.get_cache_sizes()
            
            stats = {
                "system": system_memory,
                "models": {
                    "total_memory_mb": model_memory,
                    "loaded_count": len(self.model_manager.loaded_models),
                    "models": list(self.model_manager.loaded_models.keys())
                },
                "caches": cache_sizes,
                "optimizer": {
                    "monitoring_active": self.monitoring_active,
                    "auto_cleanup_enabled": self.config.enable_auto_cleanup,
                    "cleanup_threshold": self.config.cleanup_threshold,
                    "max_cpu_memory_gb": self.config.max_cpu_memory_gb,
                    "max_gpu_memory_gb": self.config.max_gpu_memory_gb
                },
                "component_memory": dict(self.tracker.component_memory),
                "peak_memory": dict(self.tracker.peak_memory)
            }
            
            if ENHANCED_LOGGING:
                component_stats = self.logger.get_component_stats()
                stats.update({"performance": component_stats})
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Cleanup memory optimizer"""
        try:
            self.logger.info("🧹 Cleaning up memory optimizer...")
            
            # Stop monitoring
            self.stop_monitoring()
            
            # Final cleanup
            self.optimize_memory()
            
            self.logger.info("✅ Memory optimizer cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Memory optimizer cleanup failed: {e}")


# Global memory optimizer instance
_memory_optimizer: Optional[MemoryOptimizer] = None

def get_memory_optimizer(config: Optional[Dict[str, Any]] = None) -> MemoryOptimizer:
    """Get global memory optimizer instance"""
    global _memory_optimizer
    
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer(config)
    
    return _memory_optimizer

def optimize_memory() -> Dict[str, Any]:
    """Quick memory optimization"""
    optimizer = get_memory_optimizer()
    return optimizer.optimize_memory()

def get_memory_stats() -> Dict[str, Any]:
    """Quick memory stats"""
    optimizer = get_memory_optimizer()
    return optimizer.get_memory_stats()