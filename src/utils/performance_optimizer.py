#!/usr/bin/env python3
"""
⚡ Performance Optimizer for JARVIS
ระบบปรับปรุงประสิทธิภาพและจัดการหน่วยความจำ
"""

import logging
import gc
import psutil
import time
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import os
import sys
from pathlib import Path


@dataclass
class PerformanceMetrics:
    """เมตริกประสิทธิภาพ"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    gpu_memory_mb: float
    response_time_ms: float
    cache_hit_rate: float
    active_threads: int
    timestamp: float


@dataclass
class OptimizationSettings:
    """การตั้งค่าการปรับปรุง"""
    max_memory_usage: float = 80.0      # % of total memory
    max_gpu_memory: float = 6.0         # GB
    gc_threshold: int = 100             # objects before GC
    cache_cleanup_interval: int = 300   # seconds
    model_unload_delay: int = 600       # seconds
    thread_pool_size: int = 4
    enable_auto_cleanup: bool = True


class PerformanceOptimizer:
    """ระบบปรับปรุงประสิทธิภาพ"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Optimization settings
        self.settings = OptimizationSettings(**self.config.get('optimization', {}))
        
        # Performance tracking
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 100
        
        # Model management
        self.loaded_models = {}
        self.model_last_used = {}
        
        # Cache management
        self.cache_storage = {}
        self.cache_access_times = {}
        
        # Monitoring
        self.monitoring_active = False
        self.monitor_thread = None
        
        # Statistics
        self.stats = {
            'optimizations_performed': 0,
            'memory_cleanups': 0,
            'model_unloads': 0,
            'cache_cleanups': 0,
            'gc_collections': 0
        }
        
        # Cache statistics
        self._cache_hits = 0
        self._cache_misses = 0
        
        self.logger.info("⚡ Performance Optimizer initialized")
    
    def start_monitoring(self, interval: float = 5.0):
        """เริ่มการติดตามประสิทธิภาพ"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info("📊 Performance monitoring started")
    
    def stop_monitoring(self):
        """หยุดการติดตาม"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
        
        self.logger.info("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self, interval: float):
        """ลูปการติดตาม"""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = self._collect_metrics()
                self._add_metrics(metrics)
                
                # Check for optimization needs
                if self.settings.enable_auto_cleanup:
                    self._auto_optimize(metrics)
                
                time.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """เก็บเมตริกประสิทธิภาพ"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / 1024 / 1024
            
            # GPU Memory (if available)
            gpu_memory_mb = self._get_gpu_memory()
            
            # Thread count
            active_threads = threading.active_count()
            
            # Cache hit rate
            cache_hit_rate = self._calculate_cache_hit_rate()
            
            return PerformanceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                gpu_memory_mb=gpu_memory_mb,
                response_time_ms=0.0,  # To be measured per request
                cache_hit_rate=cache_hit_rate,
                active_threads=active_threads,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Metrics collection failed: {e}")
            return PerformanceMetrics(
                cpu_percent=0.0, memory_percent=0.0, memory_used_mb=0.0,
                gpu_memory_mb=0.0, response_time_ms=0.0, cache_hit_rate=0.0,
                active_threads=0, timestamp=time.time()
            )
    
    def _get_gpu_memory(self) -> float:
        """ดึงการใช้งานหน่วยความจำ GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024  # MB
        except ImportError:
            pass
        
        return 0.0
    
    def _calculate_cache_hit_rate(self) -> float:
        """คำนวณอัตราการใช้แคช"""
        if not hasattr(self, '_cache_hits'):
            self._cache_hits = 0
            self._cache_misses = 0
        
        total = self._cache_hits + self._cache_misses
        if total == 0:
            return 0.0
        
        return self._cache_hits / total
    
    def _add_metrics(self, metrics: PerformanceMetrics):
        """เพิ่มเมตริกในประวัติ"""
        self.metrics_history.append(metrics)
        
        # Limit history size
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]
    
    def _auto_optimize(self, metrics: PerformanceMetrics):
        """ปรับปรุงอัตโนมัติ"""
        optimized = False
        
        # Memory cleanup if usage is high
        if metrics.memory_percent > self.settings.max_memory_usage:
            self.cleanup_memory()
            optimized = True
        
        # GPU memory cleanup
        if metrics.gpu_memory_mb > self.settings.max_gpu_memory * 1024:
            self.cleanup_gpu_memory()
            optimized = True
        
        # Model unloading
        self._unload_unused_models()
        
        # Cache cleanup
        self._cleanup_cache()
        
        if optimized:
            self.stats['optimizations_performed'] += 1
    
    def cleanup_memory(self):
        """ทำความสะอาดหน่วยความจำ"""
        self.logger.info("🧹 Performing memory cleanup...")
        
        # Force garbage collection
        collected = gc.collect()
        self.stats['gc_collections'] += 1
        self.stats['memory_cleanups'] += 1
        
        self.logger.info(f"♻️ Collected {collected} objects")
    
    def cleanup_gpu_memory(self):
        """ทำความสะอาดหน่วยความจำ GPU"""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU memory cache cleared")
        except ImportError:
            pass
    
    def register_model(self, model_name: str, model_object: Any):
        """ลงทะเบียนโมเดล"""
        self.loaded_models[model_name] = model_object
        self.model_last_used[model_name] = time.time()
        self.logger.debug(f"📝 Registered model: {model_name}")
    
    def access_model(self, model_name: str) -> Optional[Any]:
        """เข้าถึงโมเดล"""
        if model_name in self.loaded_models:
            self.model_last_used[model_name] = time.time()
            return self.loaded_models[model_name]
        return None
    
    def _unload_unused_models(self):
        """ยกเลิกการโหลดโมเดลที่ไม่ใช้"""
        current_time = time.time()
        models_to_unload = []
        
        for model_name, last_used in self.model_last_used.items():
            if current_time - last_used > self.settings.model_unload_delay:
                models_to_unload.append(model_name)
        
        for model_name in models_to_unload:
            self._unload_model(model_name)
    
    def _unload_model(self, model_name: str):
        """ยกเลิกการโหลดโมเดล"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            del self.model_last_used[model_name]
            self.stats['model_unloads'] += 1
            self.logger.info(f"🗑️ Unloaded model: {model_name}")
    
    def cache_get(self, key: str) -> Optional[Any]:
        """ดึงข้อมูลจากแคช"""
        if key in self.cache_storage:
            self.cache_access_times[key] = time.time()
            self._cache_hits += 1
            return self.cache_storage[key]
        
        self._cache_misses += 1
        return None
    
    def cache_set(self, key: str, value: Any, ttl: Optional[int] = None):
        """เก็บข้อมูลในแคช"""
        self.cache_storage[key] = {
            'value': value,
            'created': time.time(),
            'ttl': ttl
        }
        self.cache_access_times[key] = time.time()
    
    def _cleanup_cache(self):
        """ทำความสะอาดแคช"""
        current_time = time.time()
        keys_to_remove = []
        
        for key, data in self.cache_storage.items():
            # Remove expired items
            if data.get('ttl') and current_time - data['created'] > data['ttl']:
                keys_to_remove.append(key)
            # Remove old unused items
            elif key in self.cache_access_times:
                last_access = self.cache_access_times[key]
                if current_time - last_access > self.settings.cache_cleanup_interval:
                    keys_to_remove.append(key)
        
        for key in keys_to_remove:
            self._remove_cache_item(key)
        
        if keys_to_remove:
            self.stats['cache_cleanups'] += 1
            self.logger.info(f"🧹 Cleaned {len(keys_to_remove)} cache items")
    
    def _remove_cache_item(self, key: str):
        """ลบรายการแคช"""
        self.cache_storage.pop(key, None)
        self.cache_access_times.pop(key, None)
    
    def optimize_startup(self):
        """ปรับปรุงการเริ่มต้น"""
        self.logger.info("🚀 Optimizing startup performance...")
        
        # Set process priority
        try:
            if os.name == 'nt':  # Windows
                import ctypes
                ctypes.windll.kernel32.SetPriorityClass(-1, 0x00000080)  # HIGH_PRIORITY_CLASS
            else:  # Unix-like
                os.nice(-5)  # Higher priority
        except (OSError, AttributeError):
            pass
        
        # Optimize garbage collection
        gc.set_threshold(
            self.settings.gc_threshold,
            self.settings.gc_threshold // 2,
            self.settings.gc_threshold // 4
        )
        
        # Pre-allocate some memory to reduce fragmentation
        self._preallocate_memory()
        
        self.logger.info("✅ Startup optimization complete")
    
    def _preallocate_memory(self):
        """จองหน่วยความจำล่วงหน้า"""
        try:
            # Pre-allocate small chunks to reduce fragmentation
            temp_allocations = []
            for _ in range(10):
                temp_allocations.append(bytearray(1024 * 1024))  # 1MB chunks
            
            # Release them
            del temp_allocations
            gc.collect()
            
        except MemoryError:
            self.logger.warning("⚠️ Memory pre-allocation failed")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """ดึงรายงานประสิทธิภาพ"""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        latest = self.metrics_history[-1]
        
        # Calculate averages
        avg_cpu = sum(m.cpu_percent for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_percent for m in self.metrics_history) / len(self.metrics_history)
        
        return {
            "current_metrics": {
                "cpu_percent": latest.cpu_percent,
                "memory_percent": latest.memory_percent,
                "memory_used_mb": latest.memory_used_mb,
                "gpu_memory_mb": latest.gpu_memory_mb,
                "active_threads": latest.active_threads,
                "cache_hit_rate": latest.cache_hit_rate
            },
            "averages": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "models": {
                "loaded_count": len(self.loaded_models),
                "loaded_models": list(self.loaded_models.keys())
            },
            "cache": {
                "items_count": len(self.cache_storage),
                "hit_rate": latest.cache_hit_rate
            },
            "statistics": self.stats,
            "settings": {
                "max_memory_usage": self.settings.max_memory_usage,
                "max_gpu_memory": self.settings.max_gpu_memory,
                "auto_cleanup": self.settings.enable_auto_cleanup
            }
        }
    
    def optimize_for_voice_processing(self):
        """ปรับปรุงสำหรับการประมวลผลเสียง"""
        self.logger.info("🎤 Optimizing for voice processing...")
        
        # Reduce GC frequency during voice processing
        gc.disable()
        
        # Set thread priorities for audio processing
        try:
            threading.current_thread().name = "VoiceProcessor"
        except:
            pass
        
        self.logger.info("✅ Voice processing optimization applied")
    
    def restore_normal_mode(self):
        """กลับสู่โหมดปกติ"""
        gc.enable()
        self.logger.debug("🔄 Restored normal performance mode")


def test_performance_optimizer():
    """ทดสอบระบบปรับปรุงประสิทธิภาพ"""
    print("🧪 Testing Performance Optimizer...")
    
    # Create optimizer
    config = {
        'optimization': {
            'max_memory_usage': 75.0,
            'max_gpu_memory': 4.0,
            'enable_auto_cleanup': True
        }
    }
    
    optimizer = PerformanceOptimizer(config)
    
    # Test model registration
    print("\n📝 Testing model management...")
    optimizer.register_model("test_model", {"type": "test", "size": "small"})
    model = optimizer.access_model("test_model")
    print(f"   ✅ Model retrieved: {model is not None}")
    
    # Test caching
    print("\n💾 Testing cache system...")
    optimizer.cache_set("test_key", "test_value", ttl=60)
    cached_value = optimizer.cache_get("test_key")
    print(f"   ✅ Cache working: {cached_value is not None}")
    
    # Test memory cleanup
    print("\n🧹 Testing memory cleanup...")
    optimizer.cleanup_memory()
    print("   ✅ Memory cleanup completed")
    
    # Test startup optimization
    print("\n🚀 Testing startup optimization...")
    optimizer.optimize_startup()
    print("   ✅ Startup optimization completed")
    
    # Start monitoring briefly
    print("\n📊 Testing performance monitoring...")
    optimizer.start_monitoring(interval=1.0)
    time.sleep(3)  # Monitor for 3 seconds
    optimizer.stop_monitoring()
    
    # Get performance report
    print("\n📋 Performance Report:")
    report = optimizer.get_performance_report()
    
    if "current_metrics" in report:
        metrics = report["current_metrics"]
        print(f"   💻 CPU: {metrics['cpu_percent']:.1f}%")
        print(f"   💾 Memory: {metrics['memory_percent']:.1f}%")
        print(f"   🧵 Threads: {metrics['active_threads']}")
        print(f"   📊 Cache Hit Rate: {metrics['cache_hit_rate']:.2f}")
    
    print(f"\n📈 Statistics:")
    for key, value in report.get("statistics", {}).items():
        print(f"   {key}: {value}")
    
    print("\n✅ Performance Optimizer test completed!")
    return optimizer


if __name__ == "__main__":
    test_performance_optimizer()