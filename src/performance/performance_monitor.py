#!/usr/bin/env python3
"""
📊 Performance Monitor for JARVIS
ระบบตรวจสอบประสิทธิภาพสำหรับ JARVIS
"""

import time
import threading
import logging
import psutil
import json
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import deque, defaultdict
import statistics


class MetricType(Enum):
    """ประเภทของ metric"""
    COUNTER = "counter"           # ตัวนับ
    GAUGE = "gauge"              # ค่าปัจจุบัน
    HISTOGRAM = "histogram"       # กระจายของค่า
    TIMER = "timer"              # เวลาที่ใช้


@dataclass
class PerformanceMetric:
    """ข้อมูล metric ประสิทธิภาพ"""
    name: str
    value: float
    timestamp: float
    metric_type: MetricType
    labels: Dict[str, str] = None
    unit: str = ""


@dataclass
class SystemMetrics:
    """ข้อมูลระบบ"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    memory_available_mb: float
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    gpu_usage_percent: float = 0.0
    gpu_memory_used_mb: float = 0.0
    process_count: int = 0
    thread_count: int = 0


@dataclass
class PerformanceAlert:
    """การแจ้งเตือนประสิทธิภาพ"""
    metric_name: str
    severity: str  # info, warning, critical
    message: str
    threshold: float
    current_value: float
    timestamp: float
    is_resolved: bool = False


class PerformanceTimer:
    """ตัวจับเวลาประสิทธิภาพ"""
    
    def __init__(self, monitor: 'PerformanceMonitor', metric_name: str, labels: Dict[str, str] = None):
        self.monitor = monitor
        self.metric_name = metric_name
        self.labels = labels or {}
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.start_time:
            duration = time.time() - self.start_time
            self.monitor.record_timer(self.metric_name, duration, self.labels)


class MetricCollector:
    """ตัวรวบรวม metrics"""
    
    def __init__(self, max_samples: int = 1000):
        self.max_samples = max_samples
        self.counters: Dict[str, float] = defaultdict(float)
        self.gauges: Dict[str, float] = defaultdict(float)
        self.histograms: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.timers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))
        self.lock = threading.RLock()
    
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """บันทึก counter metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.counters[key] += value
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """บันทึก gauge metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.gauges[key] = value
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """บันทึก histogram metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.histograms[key].append(value)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """บันทึก timer metric"""
        with self.lock:
            key = self._make_key(name, labels)
            self.timers[key].append(duration)
    
    def _make_key(self, name: str, labels: Dict[str, str] = None) -> str:
        """สร้าง key สำหรับ metric"""
        if not labels:
            return name
        
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def get_counter_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """ดึงค่า counter"""
        key = self._make_key(name, labels)
        return self.counters.get(key, 0.0)
    
    def get_gauge_value(self, name: str, labels: Dict[str, str] = None) -> float:
        """ดึงค่า gauge"""
        key = self._make_key(name, labels)
        return self.gauges.get(key, 0.0)
    
    def get_histogram_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """ดึงสถิติ histogram"""
        key = self._make_key(name, labels)
        values = list(self.histograms.get(key, []))
        
        if not values:
            return {'count': 0, 'sum': 0, 'avg': 0, 'min': 0, 'max': 0}
        
        return {
            'count': len(values),
            'sum': sum(values),
            'avg': statistics.mean(values),
            'min': min(values),
            'max': max(values),
            'p50': statistics.median(values),
            'p95': self._percentile(values, 0.95),
            'p99': self._percentile(values, 0.99)
        }
    
    def get_timer_stats(self, name: str, labels: Dict[str, str] = None) -> Dict[str, float]:
        """ดึงสถิติ timer"""
        return self.get_histogram_stats(name, labels)
    
    def _percentile(self, values: List[float], p: float) -> float:
        """คำนวณ percentile"""
        if not values:
            return 0.0
        
        sorted_values = sorted(values)
        index = int(len(sorted_values) * p)
        return sorted_values[min(index, len(sorted_values) - 1)]


class PerformanceMonitor:
    """ระบบตรวจสอบประสิทธิภาพสำหรับ JARVIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Configuration
        self.monitoring_interval = self.config.get('monitoring_interval', 5.0)
        self.max_samples = self.config.get('max_samples', 1000)
        self.enable_system_monitoring = self.config.get('enable_system_monitoring', True)
        self.enable_alerting = self.config.get('enable_alerting', True)
        
        # Components
        self.collector = MetricCollector(self.max_samples)
        
        # Monitoring state
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # System metrics history
        self.system_metrics_history = deque(maxlen=self.max_samples)
        
        # Alerting
        self.alert_thresholds = self.config.get('alert_thresholds', {
            'cpu_percent': {'warning': 80, 'critical': 95},
            'memory_percent': {'warning': 85, 'critical': 95},
            'disk_usage_percent': {'warning': 80, 'critical': 90},
            'response_time_ms': {'warning': 1000, 'critical': 5000}
        })
        self.active_alerts: List[PerformanceAlert] = []
        
        # Performance tracking for JARVIS components
        self.component_timers = {}
        
        self.logger.info("📊 Performance Monitor initialized")
    
    def start_monitoring(self):
        """เริ่มการตรวจสอบประสิทธิภาพ"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        self.logger.info("🚀 Performance monitoring started")
    
    def stop_monitoring(self):
        """หยุดการตรวจสอบประสิทธิภาพ"""
        self.monitoring_active = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=10)
        self.logger.info("🛑 Performance monitoring stopped")
    
    def _monitoring_loop(self):
        """ลูปการตรวจสอบประสิทธิภาพ"""
        while self.monitoring_active:
            try:
                # Collect system metrics
                if self.enable_system_monitoring:
                    system_metrics = self._collect_system_metrics()
                    self.system_metrics_history.append(system_metrics)
                    
                    # Record as gauges
                    self.record_gauge('system.cpu_percent', system_metrics.cpu_percent)
                    self.record_gauge('system.memory_percent', system_metrics.memory_percent)
                    self.record_gauge('system.disk_usage_percent', system_metrics.disk_usage_percent)
                    
                    # Check for alerts
                    if self.enable_alerting:
                        self._check_alerts(system_metrics)
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                self.logger.error(f"❌ Monitoring error: {e}")
                time.sleep(self.monitoring_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """รวบรวม metrics ของระบบ"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            
            # Disk usage (root partition)
            disk = psutil.disk_usage('/')
            
            # Network
            network = psutil.net_io_counters()
            
            # Process info
            current_process = psutil.Process()
            
            # GPU (if available)
            gpu_usage = 0.0
            gpu_memory = 0.0
            
            try:
                import GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # Use first GPU
                    gpu_usage = gpu.load * 100
                    gpu_memory = gpu.memoryUsed
            except ImportError:
                pass
            
            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                memory_used_mb=memory.used / 1024 / 1024,
                memory_available_mb=memory.available / 1024 / 1024,
                disk_usage_percent=(disk.used / disk.total) * 100,
                network_bytes_sent=network.bytes_sent,
                network_bytes_recv=network.bytes_recv,
                gpu_usage_percent=gpu_usage,
                gpu_memory_used_mb=gpu_memory,
                process_count=len(psutil.pids()),
                thread_count=current_process.num_threads()
            )
            
        except Exception as e:
            self.logger.error(f"❌ Failed to collect system metrics: {e}")
            return SystemMetrics(0, 0, 0, 0, 0, 0, 0)
    
    def _check_alerts(self, metrics: SystemMetrics):
        """ตรวจสอบการแจ้งเตือน"""
        current_time = time.time()
        
        # Check each metric against thresholds
        checks = [
            ('cpu_percent', metrics.cpu_percent, '%'),
            ('memory_percent', metrics.memory_percent, '%'),
            ('disk_usage_percent', metrics.disk_usage_percent, '%')
        ]
        
        for metric_name, value, unit in checks:
            if metric_name in self.alert_thresholds:
                thresholds = self.alert_thresholds[metric_name]
                
                # Check critical threshold
                if 'critical' in thresholds and value >= thresholds['critical']:
                    self._create_alert(
                        metric_name, 'critical', 
                        f"{metric_name} is critically high: {value:.1f}{unit}",
                        thresholds['critical'], value, current_time
                    )
                
                # Check warning threshold
                elif 'warning' in thresholds and value >= thresholds['warning']:
                    self._create_alert(
                        metric_name, 'warning',
                        f"{metric_name} is high: {value:.1f}{unit}",
                        thresholds['warning'], value, current_time
                    )
    
    def _create_alert(self, metric_name: str, severity: str, message: str, 
                     threshold: float, current_value: float, timestamp: float):
        """สร้างการแจ้งเตือน"""
        # Check if similar alert already exists
        for alert in self.active_alerts:
            if (alert.metric_name == metric_name and 
                alert.severity == severity and 
                not alert.is_resolved):
                return  # Don't create duplicate alert
        
        alert = PerformanceAlert(
            metric_name=metric_name,
            severity=severity,
            message=message,
            threshold=threshold,
            current_value=current_value,
            timestamp=timestamp
        )
        
        self.active_alerts.append(alert)
        
        # Log alert
        if severity == 'critical':
            self.logger.critical(f"🚨 {message}")
        elif severity == 'warning':
            self.logger.warning(f"⚠️ {message}")
        else:
            self.logger.info(f"ℹ️ {message}")
    
    # Metric recording methods
    def record_counter(self, name: str, value: float = 1.0, labels: Dict[str, str] = None):
        """บันทึก counter metric"""
        self.collector.record_counter(name, value, labels)
    
    def record_gauge(self, name: str, value: float, labels: Dict[str, str] = None):
        """บันทึก gauge metric"""
        self.collector.record_gauge(name, value, labels)
    
    def record_histogram(self, name: str, value: float, labels: Dict[str, str] = None):
        """บันทึก histogram metric"""
        self.collector.record_histogram(name, value, labels)
    
    def record_timer(self, name: str, duration: float, labels: Dict[str, str] = None):
        """บันทึก timer metric"""
        self.collector.record_timer(name, duration, labels)
    
    def timer(self, metric_name: str, labels: Dict[str, str] = None) -> PerformanceTimer:
        """สร้าง context manager สำหรับจับเวลา"""
        return PerformanceTimer(self, metric_name, labels)
    
    # JARVIS-specific performance tracking
    def track_voice_processing_time(self, duration: float):
        """ติดตามเวลาการประมวลผลเสียง"""
        self.record_timer('jarvis.voice_processing_time', duration, {'component': 'voice'})
    
    def track_ai_response_time(self, duration: float):
        """ติดตามเวลาการตอบสนองของ AI"""
        self.record_timer('jarvis.ai_response_time', duration, {'component': 'ai'})
    
    def track_command_execution_time(self, command_type: str, duration: float):
        """ติดตามเวลาการทำงานของคำสั่ง"""
        self.record_timer('jarvis.command_execution_time', duration, {'command_type': command_type})
    
    def track_ui_render_time(self, component: str, duration: float):
        """ติดตามเวลาการเรนเดอร์ UI"""
        self.record_timer('jarvis.ui_render_time', duration, {'component': component})
    
    def increment_command_count(self, command_type: str):
        """นับจำนวนคำสั่งที่ทำงาน"""
        self.record_counter('jarvis.commands_processed', 1.0, {'command_type': command_type})
    
    def track_error(self, component: str, error_type: str):
        """ติดตามข้อผิดพลาด"""
        self.record_counter('jarvis.errors', 1.0, {'component': component, 'error_type': error_type})
    
    def track_memory_usage(self, component: str, memory_mb: float):
        """ติดตามการใช้หน่วยความจำ"""
        self.record_gauge('jarvis.memory_usage_mb', memory_mb, {'component': component})
    
    # Reporting methods
    def get_performance_summary(self) -> Dict[str, Any]:
        """ดึงสรุปประสิทธิภาพ"""
        summary = {
            'monitoring_active': self.monitoring_active,
            'system_metrics': None,
            'jarvis_performance': {},
            'alerts': {
                'active_count': len([a for a in self.active_alerts if not a.is_resolved]),
                'total_count': len(self.active_alerts)
            }
        }
        
        # Latest system metrics
        if self.system_metrics_history:
            latest_metrics = self.system_metrics_history[-1]
            summary['system_metrics'] = asdict(latest_metrics)
        
        # JARVIS-specific metrics
        jarvis_metrics = [
            'jarvis.voice_processing_time',
            'jarvis.ai_response_time', 
            'jarvis.command_execution_time',
            'jarvis.ui_render_time'
        ]
        
        for metric in jarvis_metrics:
            stats = self.collector.get_timer_stats(metric)
            if stats['count'] > 0:
                summary['jarvis_performance'][metric] = {
                    'avg_ms': stats['avg'] * 1000,
                    'p95_ms': stats['p95'] * 1000,
                    'count': stats['count']
                }
        
        # Command counts
        commands_processed = self.collector.get_counter_value('jarvis.commands_processed')
        errors_count = self.collector.get_counter_value('jarvis.errors')
        
        summary['jarvis_performance']['commands_processed'] = commands_processed
        summary['jarvis_performance']['error_rate'] = (
            errors_count / max(commands_processed, 1) * 100
        )
        
        return summary
    
    def get_detailed_metrics(self) -> Dict[str, Any]:
        """ดึง metrics แบบละเอียด"""
        detailed = {
            'counters': {},
            'gauges': {},
            'histograms': {},
            'timers': {}
        }
        
        # Get all counters
        for key, value in self.collector.counters.items():
            detailed['counters'][key] = value
        
        # Get all gauges
        for key, value in self.collector.gauges.items():
            detailed['gauges'][key] = value
        
        # Get histogram stats
        for key in self.collector.histograms.keys():
            detailed['histograms'][key] = self.collector.get_histogram_stats(key.split('{')[0])
        
        # Get timer stats
        for key in self.collector.timers.keys():
            detailed['timers'][key] = self.collector.get_timer_stats(key.split('{')[0])
        
        return detailed
    
    def export_metrics(self, format: str = 'json') -> str:
        """ส่งออก metrics"""
        if format == 'json':
            metrics = self.get_detailed_metrics()
            return json.dumps(metrics, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def reset_metrics(self):
        """รีเซ็ต metrics ทั้งหมด"""
        self.collector = MetricCollector(self.max_samples)
        self.system_metrics_history.clear()
        self.active_alerts.clear()
        self.logger.info("🔄 Performance metrics reset")
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.stop_monitoring()
        self.reset_metrics()
        self.logger.info("🧹 Performance monitor cleaned up")


def test_performance_monitor():
    """ทดสอบ Performance Monitor"""
    print("🧪 Testing Performance Monitor")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create monitor
    config = {
        'monitoring_interval': 2.0,
        'enable_system_monitoring': True,
        'enable_alerting': True,
        'alert_thresholds': {
            'cpu_percent': {'warning': 1, 'critical': 2}  # Low thresholds for testing
        }
    }
    
    monitor = PerformanceMonitor(config)
    
    # Test metric recording
    print("📊 Testing metric recording...")
    
    # Record various metrics
    monitor.record_counter('test.requests', 10)
    monitor.record_gauge('test.active_connections', 5)
    monitor.record_histogram('test.response_size', 1024)
    monitor.record_timer('test.processing_time', 0.125)
    
    # Test JARVIS-specific metrics
    monitor.track_voice_processing_time(0.05)
    monitor.track_ai_response_time(0.2)
    monitor.track_command_execution_time('greeting', 0.01)
    monitor.increment_command_count('greeting')
    
    # Test timer context manager
    with monitor.timer('test.operation_time'):
        time.sleep(0.01)  # Simulate work
    
    print("   ✅ Metrics recorded successfully")
    
    # Test monitoring
    print("\n🔍 Testing system monitoring...")
    monitor.start_monitoring()
    
    # Let it run for a few seconds
    time.sleep(3)
    
    # Get performance summary
    summary = monitor.get_performance_summary()
    print(f"   System CPU: {summary['system_metrics']['cpu_percent']:.1f}%")
    print(f"   System Memory: {summary['system_metrics']['memory_percent']:.1f}%")
    print(f"   Active alerts: {summary['alerts']['active_count']}")
    
    # Test detailed metrics
    print("\n📈 Performance statistics:")
    
    # Voice processing
    voice_stats = monitor.collector.get_timer_stats('jarvis.voice_processing_time')
    if voice_stats['count'] > 0:
        print(f"   Voice processing: {voice_stats['avg']*1000:.1f}ms avg")
    
    # AI response
    ai_stats = monitor.collector.get_timer_stats('jarvis.ai_response_time')
    if ai_stats['count'] > 0:
        print(f"   AI response: {ai_stats['avg']*1000:.1f}ms avg")
    
    # Commands processed
    commands = monitor.collector.get_counter_value('jarvis.commands_processed')
    print(f"   Commands processed: {commands}")
    
    # Test export
    print("\n💾 Testing metrics export...")
    exported = monitor.export_metrics('json')
    print(f"   Exported {len(exported)} characters of JSON data")
    
    # Cleanup
    monitor.cleanup()
    
    return monitor


if __name__ == "__main__":
    test_performance_monitor()