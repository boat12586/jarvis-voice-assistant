#!/usr/bin/env python3
"""
⚡ Performance Manager for JARVIS
ตัวจัดการประสิทธิภาพหลักสำหรับ JARVIS
"""

import logging
import threading
import time
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

from .gpu_accelerator import GPUAccelerator
from .memory_optimizer import MemoryOptimizer
from .performance_monitor import PerformanceMonitor


class PerformanceMode(Enum):
    """โหมดประสิทธิภาพ"""
    ECO = "eco"                   # ประหยัดพลังงาน
    BALANCED = "balanced"         # สมดุล
    PERFORMANCE = "performance"   # ประสิทธิภาพสูง
    TURBO = "turbo"              # ประสิทธิภาพสูงสุด


@dataclass
class PerformanceProfile:
    """โปรไฟล์ประสิทธิภาพ"""
    mode: PerformanceMode
    gpu_enabled: bool
    memory_optimization_level: str
    monitoring_interval: float
    cache_size_mb: int
    auto_optimization: bool
    description: str


class PerformanceManager:
    """ตัวจัดการประสิทธิภาพหลักสำหรับ JARVIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Current performance mode
        self.current_mode = PerformanceMode(
            self.config.get('performance_mode', 'balanced')
        )
        
        # Performance profiles
        self.profiles = self._create_performance_profiles()
        
        # Components
        self.gpu_accelerator = None
        self.memory_optimizer = None
        self.performance_monitor = None
        
        # State
        self.is_initialized = False
        self.optimization_active = False
        self.optimization_thread = None
        
        # Performance history
        self.performance_history = []
        self.max_history_length = 100
        
        # Callbacks
        self.mode_change_callbacks: List[Callable[[PerformanceMode], None]] = []
        
        self.logger.info(f"⚡ Performance Manager created (Mode: {self.current_mode.value})")
    
    def _create_performance_profiles(self) -> Dict[PerformanceMode, PerformanceProfile]:
        """สร้างโปรไฟล์ประสิทธิภาพ"""
        return {
            PerformanceMode.ECO: PerformanceProfile(
                mode=PerformanceMode.ECO,
                gpu_enabled=False,
                memory_optimization_level='conservative',
                monitoring_interval=60.0,
                cache_size_mb=25,
                auto_optimization=True,
                description="Energy-saving mode with minimal resource usage"
            ),
            
            PerformanceMode.BALANCED: PerformanceProfile(
                mode=PerformanceMode.BALANCED,
                gpu_enabled=True,
                memory_optimization_level='balanced',
                monitoring_interval=30.0,
                cache_size_mb=100,
                auto_optimization=True,
                description="Balanced performance and resource usage"
            ),
            
            PerformanceMode.PERFORMANCE: PerformanceProfile(
                mode=PerformanceMode.PERFORMANCE,
                gpu_enabled=True,
                memory_optimization_level='balanced',
                monitoring_interval=15.0,
                cache_size_mb=200,
                auto_optimization=True,
                description="High performance with optimized resource usage"
            ),
            
            PerformanceMode.TURBO: PerformanceProfile(
                mode=PerformanceMode.TURBO,
                gpu_enabled=True,
                memory_optimization_level='aggressive',
                monitoring_interval=5.0,
                cache_size_mb=500,
                auto_optimization=False,  # Manual control in turbo mode
                description="Maximum performance with aggressive optimization"
            )
        }
    
    def initialize(self) -> bool:
        """เริ่มต้นตัวจัดการประสิทธิภาพ"""
        try:
            self.logger.info("🚀 Initializing Performance Manager...")
            
            # Get current profile
            profile = self.profiles[self.current_mode]
            
            # Initialize GPU Accelerator
            if profile.gpu_enabled:
                gpu_config = {
                    'acceleration_type': 'auto',
                    'memory_limit': 0.8,
                    'mixed_precision': True,
                    'batch_optimization': True
                }
                gpu_config.update(self.config.get('gpu', {}))
                
                self.gpu_accelerator = GPUAccelerator(gpu_config)
                
                if self.gpu_accelerator.is_initialized:
                    self.logger.info("✅ GPU Accelerator initialized")
                else:
                    self.logger.warning("⚠️ GPU Accelerator initialization failed")
            
            # Initialize Memory Optimizer
            memory_config = {
                'optimization_level': profile.memory_optimization_level,
                'cache_size_mb': profile.cache_size_mb,
                'auto_optimization': profile.auto_optimization,
                'monitoring_interval': profile.monitoring_interval
            }
            memory_config.update(self.config.get('memory', {}))
            
            self.memory_optimizer = MemoryOptimizer(memory_config)
            self.logger.info("✅ Memory Optimizer initialized")
            
            # Initialize Performance Monitor
            monitor_config = {
                'monitoring_interval': profile.monitoring_interval,
                'enable_system_monitoring': True,
                'enable_alerting': True,
                'max_samples': 1000
            }
            monitor_config.update(self.config.get('monitoring', {}))
            
            self.performance_monitor = PerformanceMonitor(monitor_config)
            self.logger.info("✅ Performance Monitor initialized")
            
            # Start monitoring
            self.performance_monitor.start_monitoring()
            if self.memory_optimizer and profile.auto_optimization:
                self.memory_optimizer.start_monitoring()
            
            # Start optimization loop if auto-optimization is enabled
            if profile.auto_optimization:
                self._start_optimization_loop()
            
            self.is_initialized = True
            self.logger.info(f"🎯 Performance Manager initialized in {self.current_mode.value} mode")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Performance Manager initialization failed: {e}")
            return False
    
    def set_performance_mode(self, mode: PerformanceMode) -> bool:
        """เปลี่ยนโหมดประสิทธิภาพ"""
        if mode == self.current_mode:
            return True
        
        try:
            self.logger.info(f"🔄 Changing performance mode: {self.current_mode.value} → {mode.value}")
            
            old_mode = self.current_mode
            self.current_mode = mode
            
            # Reconfigure components based on new profile
            profile = self.profiles[mode]
            
            # Update GPU Accelerator
            if profile.gpu_enabled and not self.gpu_accelerator:
                # Enable GPU acceleration
                gpu_config = self.config.get('gpu', {})
                self.gpu_accelerator = GPUAccelerator(gpu_config)
            elif not profile.gpu_enabled and self.gpu_accelerator:
                # Disable GPU acceleration
                self.gpu_accelerator.cleanup()
                self.gpu_accelerator = None
            
            # Update Memory Optimizer
            if self.memory_optimizer:
                # Reconfigure memory optimizer
                self.memory_optimizer.optimization_level = profile.memory_optimization_level
                self.memory_optimizer.cache.max_size_bytes = profile.cache_size_mb * 1024 * 1024
                
                if profile.auto_optimization:
                    self.memory_optimizer.start_monitoring()
                else:
                    self.memory_optimizer.stop_monitoring()
            
            # Update Performance Monitor
            if self.performance_monitor:
                # Stop current monitoring
                self.performance_monitor.stop_monitoring()
                
                # Update interval
                self.performance_monitor.monitoring_interval = profile.monitoring_interval
                
                # Restart monitoring
                self.performance_monitor.start_monitoring()
            
            # Update optimization loop
            if profile.auto_optimization and not self.optimization_active:
                self._start_optimization_loop()
            elif not profile.auto_optimization and self.optimization_active:
                self._stop_optimization_loop()
            
            # Notify callbacks
            for callback in self.mode_change_callbacks:
                try:
                    callback(mode)
                except Exception as e:
                    self.logger.error(f"❌ Mode change callback failed: {e}")
            
            self.logger.info(f"✅ Performance mode changed to {mode.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to change performance mode: {e}")
            self.current_mode = old_mode  # Rollback
            return False
    
    def _start_optimization_loop(self):
        """เริ่มลูปการปรับแต่งอัตโนมัติ"""
        if self.optimization_active:
            return
        
        self.optimization_active = True
        self.optimization_thread = threading.Thread(
            target=self._optimization_loop,
            daemon=True
        )
        self.optimization_thread.start()
        self.logger.info("🔄 Auto-optimization loop started")
    
    def _stop_optimization_loop(self):
        """หยุดลูปการปรับแต่งอัตโนมัติ"""
        self.optimization_active = False
        if self.optimization_thread:
            self.optimization_thread.join(timeout=10)
        self.logger.info("⏹️ Auto-optimization loop stopped")
    
    def _optimization_loop(self):
        """ลูปการปรับแต่งอัตโนมัติ"""
        profile = self.profiles[self.current_mode]
        optimization_interval = profile.monitoring_interval * 2  # Optimize less frequently than monitoring
        
        while self.optimization_active:
            try:
                # Get current performance metrics
                if self.performance_monitor:
                    summary = self.performance_monitor.get_performance_summary()
                    
                    # Check if optimization is needed
                    if self._should_optimize(summary):
                        self.logger.info("🔧 Auto-optimization triggered")
                        self.optimize_performance()
                
                time.sleep(optimization_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Optimization loop error: {e}")
                time.sleep(optimization_interval)
    
    def _should_optimize(self, performance_summary: Dict[str, Any]) -> bool:
        """ตรวจสอบว่าควรปรับแต่งหรือไม่"""
        if not performance_summary.get('system_metrics'):
            return False
        
        system_metrics = performance_summary['system_metrics']
        
        # Check memory usage
        if system_metrics['memory_percent'] > 80:
            return True
        
        # Check if there are active alerts
        if performance_summary['alerts']['active_count'] > 0:
            return True
        
        # Check performance metrics
        jarvis_perf = performance_summary.get('jarvis_performance', {})
        
        # Check response times
        if 'jarvis.ai_response_time' in jarvis_perf:
            avg_response_ms = jarvis_perf['jarvis.ai_response_time']['avg_ms']
            if avg_response_ms > 2000:  # 2 seconds
                return True
        
        return False
    
    # Performance optimization methods
    def optimize_performance(self) -> Dict[str, Any]:
        """ปรับแต่งประสิทธิภาพ"""
        start_time = time.time()
        optimization_results = {
            'started_at': start_time,
            'mode': self.current_mode.value,
            'actions': []
        }
        
        try:
            # Memory optimization
            if self.memory_optimizer:
                memory_result = self.memory_optimizer.optimize_memory()
                if memory_result.get('success'):
                    optimization_results['actions'].append({
                        'type': 'memory_optimization',
                        'memory_freed_mb': memory_result['memory_freed_mb']
                    })
            
            # GPU cache cleanup
            if self.gpu_accelerator:
                self.gpu_accelerator.cleanup()
                optimization_results['actions'].append({
                    'type': 'gpu_cache_cleanup',
                    'status': 'completed'
                })
            
            # Record optimization in performance history
            optimization_time = time.time() - start_time
            optimization_results.update({
                'optimization_time_ms': optimization_time * 1000,
                'success': True
            })
            
            self.performance_history.append(optimization_results)
            if len(self.performance_history) > self.max_history_length:
                self.performance_history.pop(0)
            
            # Track performance optimization
            if self.performance_monitor:
                self.performance_monitor.record_timer('jarvis.optimization_time', optimization_time)
                self.performance_monitor.record_counter('jarvis.optimizations_performed')
            
            self.logger.info(f"✅ Performance optimization completed in {optimization_time*1000:.1f}ms")
            
        except Exception as e:
            self.logger.error(f"❌ Performance optimization failed: {e}")
            optimization_results.update({
                'success': False,
                'error': str(e)
            })
        
        return optimization_results
    
    # Acceleration methods
    def accelerate_audio_processing(self, audio_data):
        """เร่งความเร็วการประมวลผลเสียง"""
        if self.gpu_accelerator:
            return self.gpu_accelerator.accelerate_audio_processing(audio_data)
        return audio_data
    
    def accelerate_ai_inference(self, input_data):
        """เร่งความเร็วการอนุมาน AI"""
        if self.gpu_accelerator:
            return self.gpu_accelerator.accelerate_ai_inference(input_data)
        return input_data
    
    def accelerate_visualization(self, data):
        """เร่งความเร็วการแสดงผล"""
        if self.gpu_accelerator:
            return self.gpu_accelerator.accelerate_visualization(data)
        return data
    
    # Memory management
    def get_cached_data(self, key: str):
        """ดึงข้อมูลจาก cache"""
        if self.memory_optimizer:
            return self.memory_optimizer.cache.get(key)
        return None
    
    def cache_data(self, key: str, data: Any, ttl: Optional[float] = None):
        """เก็บข้อมูลใน cache"""
        if self.memory_optimizer:
            self.memory_optimizer.cache.put(key, data, ttl)
    
    def get_object_from_pool(self, pool_name: str, factory: Callable = None):
        """ดึง object จาก pool"""
        if self.memory_optimizer:
            return self.memory_optimizer.get_object_from_pool(pool_name, factory)
        return factory() if factory else None
    
    def return_object_to_pool(self, pool_name: str, obj: Any):
        """คืน object ไปยัง pool"""
        if self.memory_optimizer:
            self.memory_optimizer.return_object_to_pool(pool_name, obj)
    
    # Performance tracking
    def track_operation_time(self, operation_name: str, duration: float, component: str = None):
        """ติดตามเวลาการทำงาน"""
        if self.performance_monitor:
            labels = {'component': component} if component else {}
            self.performance_monitor.record_timer(f'jarvis.{operation_name}', duration, labels)
    
    def track_error(self, component: str, error_type: str):
        """ติดตามข้อผิดพลาด"""
        if self.performance_monitor:
            self.performance_monitor.track_error(component, error_type)
    
    def increment_counter(self, counter_name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """เพิ่มค่า counter"""
        if self.performance_monitor:
            self.performance_monitor.record_counter(f'jarvis.{counter_name}', value, labels)
    
    # Reporting
    def get_performance_report(self) -> Dict[str, Any]:
        """ดึงรายงานประสิทธิภาพ"""
        report = {
            'current_mode': self.current_mode.value,
            'is_initialized': self.is_initialized,
            'optimization_active': self.optimization_active,
            'profile': self.profiles[self.current_mode].__dict__,
            'components': {
                'gpu_accelerator': self.gpu_accelerator is not None,
                'memory_optimizer': self.memory_optimizer is not None,
                'performance_monitor': self.performance_monitor is not None
            }
        }
        
        # Add component reports
        if self.gpu_accelerator:
            report['gpu_report'] = self.gpu_accelerator.get_performance_report()
        
        if self.memory_optimizer:
            report['memory_report'] = self.memory_optimizer.get_optimization_report()
        
        if self.performance_monitor:
            report['monitoring_report'] = self.performance_monitor.get_performance_summary()
        
        # Add optimization history
        if self.performance_history:
            recent_optimizations = self.performance_history[-5:]  # Last 5 optimizations
            report['recent_optimizations'] = recent_optimizations
        
        return report
    
    def register_mode_change_callback(self, callback: Callable[[PerformanceMode], None]):
        """ลงทะเบียน callback สำหรับการเปลี่ยนโหมด"""
        self.mode_change_callbacks.append(callback)
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            self.logger.info("🧹 Cleaning up Performance Manager...")
            
            # Stop optimization loop
            self._stop_optimization_loop()
            
            # Cleanup components
            if self.gpu_accelerator:
                self.gpu_accelerator.cleanup()
            
            if self.memory_optimizer:
                self.memory_optimizer.cleanup()
            
            if self.performance_monitor:
                self.performance_monitor.cleanup()
            
            # Clear callbacks
            self.mode_change_callbacks.clear()
            
            self.is_initialized = False
            self.logger.info("✅ Performance Manager cleaned up")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")


def test_performance_manager():
    """ทดสอบ Performance Manager"""
    print("🧪 Testing Performance Manager")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create manager
    config = {
        'performance_mode': 'balanced',
        'gpu': {
            'acceleration_type': 'auto',
            'memory_limit': 0.8
        },
        'memory': {
            'cache_size_mb': 50,
            'auto_optimization': True
        },
        'monitoring': {
            'monitoring_interval': 2.0
        }
    }
    
    manager = PerformanceManager(config)
    
    # Test initialization
    print("🚀 Testing initialization...")
    success = manager.initialize()
    print(f"   Initialization: {'✅ Success' if success else '❌ Failed'}")
    
    # Test performance modes
    print("\n🔄 Testing performance modes...")
    modes = [PerformanceMode.ECO, PerformanceMode.PERFORMANCE, PerformanceMode.TURBO, PerformanceMode.BALANCED]
    
    for mode in modes:
        success = manager.set_performance_mode(mode)
        print(f"   {mode.value}: {'✅' if success else '❌'}")
    
    # Test acceleration
    print("\n⚡ Testing acceleration...")
    import numpy as np
    
    # Test audio processing
    audio_data = np.random.randn(1024).astype(np.float32)
    processed = manager.accelerate_audio_processing(audio_data)
    print(f"   Audio processing: {audio_data.shape} -> {processed.shape}")
    
    # Test AI inference
    ai_data = np.random.randn(64, 128).astype(np.float32)
    inference_result = manager.accelerate_ai_inference(ai_data)
    print(f"   AI inference: {ai_data.shape} -> {inference_result.shape}")
    
    # Test caching
    print("\n💾 Testing caching...")
    test_data = {"key": "value", "number": 42}
    manager.cache_data("test_key", test_data)
    
    cached = manager.get_cached_data("test_key")
    print(f"   Cached data: {'✅ Retrieved' if cached == test_data else '❌ Failed'}")
    
    # Test performance tracking
    print("\n📊 Testing performance tracking...")
    manager.track_operation_time('test_operation', 0.1, 'test_component')
    manager.increment_counter('test_counter', 5)
    print("   ✅ Performance metrics tracked")
    
    # Test optimization
    print("\n🔧 Testing optimization...")
    result = manager.optimize_performance()
    if result['success']:
        print(f"   Optimization: ✅ Success ({len(result['actions'])} actions)")
    else:
        print(f"   Optimization: ❌ Failed - {result.get('error', 'Unknown')}")
    
    # Wait a bit for monitoring
    print("\n⏳ Running monitoring for 3 seconds...")
    time.sleep(3)
    
    # Get performance report
    print("\n📈 Performance report:")
    report = manager.get_performance_report()
    print(f"   Current mode: {report['current_mode']}")
    print(f"   Components active: {sum(report['components'].values())}/3")
    
    if 'monitoring_report' in report:
        monitoring = report['monitoring_report']
        if monitoring.get('system_metrics'):
            sys_metrics = monitoring['system_metrics']
            print(f"   System CPU: {sys_metrics['cpu_percent']:.1f}%")
            print(f"   System Memory: {sys_metrics['memory_percent']:.1f}%")
    
    # Cleanup
    manager.cleanup()
    
    return manager


if __name__ == "__main__":
    test_performance_manager()