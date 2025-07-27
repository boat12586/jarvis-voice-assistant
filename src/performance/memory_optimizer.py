#!/usr/bin/env python3
"""
🧠 Memory Optimizer for JARVIS
ตัวปรับแต่งหน่วยความจำสำหรับ JARVIS
"""

import gc
import sys
import time
import psutil
import threading
import logging
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import weakref
from collections import defaultdict
import tracemalloc


class OptimizationLevel(Enum):
    """ระดับการปรับแต่ง"""
    CONSERVATIVE = "conservative"  # ปรับแต่งน้อย
    BALANCED = "balanced"         # ปรับแต่งปานกลาง
    AGGRESSIVE = "aggressive"     # ปรับแต่งสูง


@dataclass
class MemoryStats:
    """สถิติหน่วยความจำ"""
    total_mb: float
    available_mb: float
    used_mb: float
    usage_percent: float
    python_memory_mb: float
    gc_collections: int
    cached_objects: int


@dataclass
class CacheEntry:
    """รายการ Cache"""
    data: Any
    timestamp: float
    access_count: int
    size_bytes: int
    ttl: float


class MemoryCache:
    """ระบบ Cache หน่วยความจำ"""
    
    def __init__(self, max_size_mb: float = 100, default_ttl: float = 300):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: List[str] = []
        self.current_size = 0
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """ดึงข้อมูลจาก cache"""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            entry = self.cache[key]
            
            # Check TTL
            if time.time() - entry.timestamp > entry.ttl:
                self._remove_entry(key)
                self.misses += 1
                return None
            
            # Update access
            entry.access_count += 1
            self._update_access_order(key)
            self.hits += 1
            
            return entry.data
    
    def put(self, key: str, data: Any, ttl: Optional[float] = None):
        """เก็บข้อมูลใน cache"""
        with self.lock:
            # Calculate size (rough estimation)
            size_bytes = sys.getsizeof(data)
            ttl = ttl or self.default_ttl
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Check if we need to evict
            while (self.current_size + size_bytes > self.max_size_bytes and 
                   len(self.cache) > 0):
                self._evict_lru()
            
            # Add new entry
            entry = CacheEntry(
                data=data,
                timestamp=time.time(),
                access_count=1,
                size_bytes=size_bytes,
                ttl=ttl
            )
            
            self.cache[key] = entry
            self.access_order.append(key)
            self.current_size += size_bytes
    
    def _remove_entry(self, key: str):
        """ลบรายการจาก cache"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.current_size -= entry.size_bytes
            if key in self.access_order:
                self.access_order.remove(key)
    
    def _evict_lru(self):
        """ลบรายการที่ใช้น้อยที่สุด"""
        if self.access_order:
            lru_key = self.access_order[0]
            self._remove_entry(lru_key)
            self.evictions += 1
    
    def _update_access_order(self, key: str):
        """อัปเดตลำดับการเข้าถึง"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self):
        """เคลียร์ cache ทั้งหมด"""
        with self.lock:
            self.cache.clear()
            self.access_order.clear()
            self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """ดึงสถิติ cache"""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            'entries': len(self.cache),
            'size_mb': self.current_size / 1024 / 1024,
            'max_size_mb': self.max_size_bytes / 1024 / 1024,
            'utilization': self.current_size / self.max_size_bytes,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': hit_rate,
            'evictions': self.evictions
        }


class MemoryOptimizer:
    """ตัวปรับแต่งหน่วยความจำสำหรับ JARVIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Configuration
        self.optimization_level = OptimizationLevel(
            self.config.get('optimization_level', 'balanced')
        )
        self.memory_limit_mb = self.config.get('memory_limit_mb', 1024)
        self.gc_threshold = self.config.get('gc_threshold', 0.8)
        self.auto_optimization = self.config.get('auto_optimization', True)
        self.monitoring_interval = self.config.get('monitoring_interval', 30)
        
        # Cache system
        cache_size = self.config.get('cache_size_mb', 100)
        cache_ttl = self.config.get('cache_ttl', 300)
        self.cache = MemoryCache(cache_size, cache_ttl)
        
        # Object tracking
        self.object_pools = defaultdict(list)
        self.weak_references = set()
        
        # Monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.memory_history = []
        self.max_history_length = 100
        
        # Statistics
        self.optimizations_performed = 0
        self.memory_freed_mb = 0.0
        
        # Start memory tracing if available
        if self.config.get('enable_tracing', False):
            self._start_memory_tracing()
        
        self.logger.info(f"🧠 Memory Optimizer initialized (Level: {self.optimization_level.value})")
    
    def _start_memory_tracing(self):
        """เริ่ม memory tracing"""
        try:
            tracemalloc.start()
            self.logger.info("📊 Memory tracing started")
        except Exception as e:
            self.logger.warning(f"⚠️ Memory tracing unavailable: {e}")
    
    def get_memory_stats(self) -> MemoryStats:
        """ดึงสถิติหน่วยความจำ"""
        try:
            # System memory
            memory = psutil.virtual_memory()
            
            # Python process memory
            process = psutil.Process()
            python_memory = process.memory_info().rss / 1024 / 1024
            
            # GC statistics
            gc_stats = gc.get_stats()
            total_collections = sum(stat['collections'] for stat in gc_stats)
            
            return MemoryStats(
                total_mb=memory.total / 1024 / 1024,
                available_mb=memory.available / 1024 / 1024,
                used_mb=memory.used / 1024 / 1024,
                usage_percent=memory.percent,
                python_memory_mb=python_memory,
                gc_collections=total_collections,
                cached_objects=len(self.cache.cache)
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to get memory stats: {e}")
            return MemoryStats(0, 0, 0, 0, 0, 0, 0)
    
    def optimize_memory(self, force: bool = False) -> Dict[str, Any]:
        """ปรับแต่งหน่วยความจำ"""
        start_time = time.time()
        stats_before = self.get_memory_stats()
        
        # Check if optimization is needed
        if not force and not self._should_optimize(stats_before):
            return {'skipped': True, 'reason': 'Not needed'}
        
        optimization_results = {
            'started_at': time.time(),
            'memory_before_mb': stats_before.python_memory_mb,
            'actions_performed': []
        }
        
        try:
            # Level 1: Basic cleanup (always performed)
            if True:
                freed = self._basic_cleanup()
                if freed > 0:
                    optimization_results['actions_performed'].append(f"Basic cleanup: {freed:.1f}MB")
            
            # Level 2: Cache optimization
            if self.optimization_level in [OptimizationLevel.BALANCED, OptimizationLevel.AGGRESSIVE]:
                freed = self._optimize_cache()
                if freed > 0:
                    optimization_results['actions_performed'].append(f"Cache optimization: {freed:.1f}MB")
            
            # Level 3: Aggressive optimization
            if self.optimization_level == OptimizationLevel.AGGRESSIVE:
                freed = self._aggressive_optimization()
                if freed > 0:
                    optimization_results['actions_performed'].append(f"Aggressive optimization: {freed:.1f}MB")
            
            # Final garbage collection
            collected = self._force_garbage_collection()
            if collected > 0:
                optimization_results['actions_performed'].append(f"GC collected: {collected} objects")
            
            # Calculate results
            stats_after = self.get_memory_stats()
            memory_freed = stats_before.python_memory_mb - stats_after.python_memory_mb
            
            optimization_results.update({
                'memory_after_mb': stats_after.python_memory_mb,
                'memory_freed_mb': memory_freed,
                'optimization_time_ms': (time.time() - start_time) * 1000,
                'success': True
            })
            
            self.optimizations_performed += 1
            self.memory_freed_mb += memory_freed
            
            self.logger.info(f"✅ Memory optimization completed: {memory_freed:.1f}MB freed")
            
        except Exception as e:
            self.logger.error(f"❌ Memory optimization failed: {e}")
            optimization_results.update({
                'success': False,
                'error': str(e)
            })
        
        return optimization_results
    
    def _should_optimize(self, stats: MemoryStats) -> bool:
        """ตรวจสอบว่าควรปรับแต่งหรือไม่"""
        # Check memory usage threshold
        if stats.python_memory_mb > self.memory_limit_mb * self.gc_threshold:
            return True
        
        # Check system memory pressure
        if stats.usage_percent > 85:
            return True
        
        # Check cache utilization
        cache_stats = self.cache.get_stats()
        if cache_stats['utilization'] > 0.9:
            return True
        
        return False
    
    def _basic_cleanup(self) -> float:
        """การทำความสะอาดพื้นฐาน"""
        memory_before = self._get_python_memory()
        
        # Clean up expired cache entries
        self._cleanup_expired_cache()
        
        # Remove dead weak references
        self._cleanup_weak_references()
        
        # Clear object pools that are too large
        self._cleanup_object_pools()
        
        memory_after = self._get_python_memory()
        return memory_before - memory_after
    
    def _optimize_cache(self) -> float:
        """ปรับแต่ง cache"""
        memory_before = self._get_python_memory()
        
        cache_stats = self.cache.get_stats()
        
        # If cache is over 80% full, remove least recently used items
        if cache_stats['utilization'] > 0.8:
            # Remove 25% of cache entries
            entries_to_remove = max(1, len(self.cache.cache) // 4)
            
            with self.cache.lock:
                for _ in range(entries_to_remove):
                    if self.cache.access_order:
                        lru_key = self.cache.access_order[0]
                        self.cache._remove_entry(lru_key)
        
        memory_after = self._get_python_memory()
        return memory_before - memory_after
    
    def _aggressive_optimization(self) -> float:
        """การปรับแต่งแบบรุนแรง"""
        memory_before = self._get_python_memory()
        
        # Clear all caches
        self.cache.clear()
        
        # Clear all object pools
        for pool in self.object_pools.values():
            pool.clear()
        
        # Force clear weak references
        self.weak_references.clear()
        
        # Call gc.collect multiple times
        for _ in range(3):
            gc.collect()
        
        memory_after = self._get_python_memory()
        return memory_before - memory_after
    
    def _force_garbage_collection(self) -> int:
        """บังคับ garbage collection"""
        collected = 0
        
        # Collect each generation
        for generation in range(3):
            collected += gc.collect(generation)
        
        return collected
    
    def _cleanup_expired_cache(self):
        """ทำความสะอาด cache ที่หมดอายุ"""
        current_time = time.time()
        expired_keys = []
        
        with self.cache.lock:
            for key, entry in self.cache.cache.items():
                if current_time - entry.timestamp > entry.ttl:
                    expired_keys.append(key)
            
            for key in expired_keys:
                self.cache._remove_entry(key)
    
    def _cleanup_weak_references(self):
        """ทำความสะอาด weak references ที่ตายแล้ว"""
        dead_refs = []
        for ref in self.weak_references:
            if ref() is None:
                dead_refs.append(ref)
        
        for ref in dead_refs:
            self.weak_references.remove(ref)
    
    def _cleanup_object_pools(self):
        """ทำความสะอาด object pools"""
        max_pool_size = 100
        
        for pool_name, pool in self.object_pools.items():
            if len(pool) > max_pool_size:
                # Keep only the most recent items
                self.object_pools[pool_name] = pool[-max_pool_size:]
    
    def _get_python_memory(self) -> float:
        """ดึงการใช้หน่วยความจำของ Python (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except Exception:
            return 0.0
    
    def start_monitoring(self):
        """เริ่มการตรวจสอบหน่วยความจำ"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("👁️ Memory monitoring started")
    
    def stop_monitoring(self):
        """หยุดการตรวจสอบหน่วยความจำ"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        self.logger.info("🛑 Memory monitoring stopped")
    
    def _monitoring_loop(self):
        """ลูปการตรวจสอบหน่วยความจำ"""
        while self.monitoring_active:
            try:
                stats = self.get_memory_stats()
                
                # Add to history
                self.memory_history.append({
                    'timestamp': time.time(),
                    'usage_percent': stats.usage_percent,
                    'python_memory_mb': stats.python_memory_mb
                })
                
                # Limit history length
                if len(self.memory_history) > self.max_history_length:
                    self.memory_history.pop(0)
                
                # Auto-optimize if needed
                if self.auto_optimization and self._should_optimize(stats):
                    self.logger.info("🔧 Auto-optimizing memory...")
                    self.optimize_memory()
                
                # Check for memory leaks
                self._check_memory_trends()
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _check_memory_trends(self):
        """ตรวจสอบแนวโน้มการใช้หน่วยความจำ"""
        if len(self.memory_history) < 5:
            return
        
        # Check for consistent growth
        recent_usage = [entry['python_memory_mb'] for entry in self.memory_history[-5:]]
        
        # Simple trend detection
        increasing_count = 0
        for i in range(1, len(recent_usage)):
            if recent_usage[i] > recent_usage[i-1]:
                increasing_count += 1
        
        # If memory consistently increasing, warn about potential leak
        if increasing_count >= 4:
            self.logger.warning("⚠️ Potential memory leak detected - consistent growth pattern")
    
    def register_weak_reference(self, obj: Any) -> weakref.ref:
        """ลงทะเบียน weak reference"""
        ref = weakref.ref(obj)
        self.weak_references.add(ref)
        return ref
    
    def get_object_from_pool(self, pool_name: str, factory: Callable = None) -> Any:
        """ดึง object จาก pool"""
        pool = self.object_pools[pool_name]
        
        if pool:
            return pool.pop()
        elif factory:
            return factory()
        else:
            return None
    
    def return_object_to_pool(self, pool_name: str, obj: Any, max_pool_size: int = 50):
        """คืน object ไปยัง pool"""
        pool = self.object_pools[pool_name]
        
        if len(pool) < max_pool_size:
            # Reset object if it has a reset method
            if hasattr(obj, 'reset'):
                obj.reset()
            pool.append(obj)
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """รายงานการปรับแต่ง"""
        current_stats = self.get_memory_stats()
        cache_stats = self.cache.get_stats()
        
        report = {
            'optimization_level': self.optimization_level.value,
            'optimizations_performed': self.optimizations_performed,
            'total_memory_freed_mb': self.memory_freed_mb,
            'current_memory_stats': current_stats.__dict__,
            'cache_stats': cache_stats,
            'monitoring_active': self.monitoring_active,
            'memory_history_length': len(self.memory_history),
            'object_pools': {name: len(pool) for name, pool in self.object_pools.items()},
            'weak_references_count': len(self.weak_references)
        }
        
        if tracemalloc.is_tracing():
            try:
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')[:5]
                
                report['top_memory_allocations'] = [
                    {
                        'file': stat.traceback.format()[-1],
                        'size_mb': stat.size / 1024 / 1024,
                        'count': stat.count
                    }
                    for stat in top_stats
                ]
            except Exception:
                pass
        
        return report
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.stop_monitoring()
        self.cache.clear()
        
        for pool in self.object_pools.values():
            pool.clear()
        
        self.weak_references.clear()
        
        if tracemalloc.is_tracing():
            tracemalloc.stop()
        
        self.logger.info("🧹 Memory optimizer cleaned up")


def test_memory_optimizer():
    """ทดสอบ Memory Optimizer"""
    print("🧪 Testing Memory Optimizer")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create optimizer
    config = {
        'optimization_level': 'balanced',
        'memory_limit_mb': 512,
        'cache_size_mb': 50,
        'auto_optimization': False,
        'enable_tracing': True
    }
    
    optimizer = MemoryOptimizer(config)
    
    # Test cache
    print("📦 Testing cache system...")
    for i in range(100):
        optimizer.cache.put(f"key_{i}", f"data_{i}" * 1000)
    
    cache_stats = optimizer.cache.get_stats()
    print(f"   Cache entries: {cache_stats['entries']}")
    print(f"   Cache size: {cache_stats['size_mb']:.1f}MB")
    print(f"   Hit rate: {cache_stats['hit_rate']:.2%}")
    
    # Test object pooling
    print("\n🏊 Testing object pools...")
    for i in range(20):
        obj = [f"item_{j}" for j in range(100)]
        optimizer.return_object_to_pool("test_pool", obj)
    
    pool_obj = optimizer.get_object_from_pool("test_pool")
    print(f"   Retrieved object from pool: {len(pool_obj) if pool_obj else 0} items")
    
    # Test memory stats
    print("\n📊 Memory statistics:")
    stats = optimizer.get_memory_stats()
    print(f"   System memory usage: {stats.usage_percent:.1f}%")
    print(f"   Python memory: {stats.python_memory_mb:.1f}MB")
    print(f"   Available memory: {stats.available_mb:.1f}MB")
    
    # Test optimization
    print("\n🔧 Testing memory optimization...")
    result = optimizer.optimize_memory(force=True)
    
    if result.get('success'):
        print(f"   Memory freed: {result['memory_freed_mb']:.1f}MB")
        print(f"   Optimization time: {result['optimization_time_ms']:.1f}ms")
        print(f"   Actions: {len(result['actions_performed'])}")
    else:
        print(f"   Optimization failed: {result.get('error', 'Unknown error')}")
    
    # Performance report
    print("\n📈 Optimization report:")
    report = optimizer.get_optimization_report()
    print(f"   Optimizations performed: {report['optimizations_performed']}")
    print(f"   Total memory freed: {report['total_memory_freed_mb']:.1f}MB")
    print(f"   Cache hit rate: {report['cache_stats']['hit_rate']:.2%}")
    
    # Cleanup
    optimizer.cleanup()
    
    return optimizer


if __name__ == "__main__":
    test_memory_optimizer()