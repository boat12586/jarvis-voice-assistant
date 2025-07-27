"""
Performance Monitoring System for JARVIS
Tracks system health and performance metrics
"""

import time
import psutil
import json
import logging
from typing import Dict, Any, List
from pathlib import Path
from datetime import datetime
import threading

class PerformanceMonitor:
    """Monitors JARVIS performance and system health"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.metrics = {}
        self.start_time = time.time()
        self.monitoring_active = False
        self.monitor_thread = None
        
        # System info cache
        self.system_info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """Collect static system information"""
        import platform
        
        memory_gb = psutil.virtual_memory().total / (1024**3)
        
        info = {
            'platform': platform.system(),
            'platform_release': platform.release(),
            'platform_version': platform.version(),
            'architecture': platform.machine(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'cpu_count': psutil.cpu_count(logical=False),
            'cpu_count_logical': psutil.cpu_count(logical=True),
            'memory_total_gb': round(memory_gb, 2)
        }
        
        return info
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        return self.system_info.copy()
        
    def start_monitoring(self, interval: int = 30):
        """Start continuous monitoring"""
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join()
        self.logger.info("Performance monitoring stopped")
    
    def _monitor_loop(self, interval: int):
        """Monitoring loop"""
        while self.monitoring_active:
            try:
                self._collect_metrics()
                time.sleep(interval)
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
    
    def collect_metrics(self):
        """Collect current system metrics"""
        timestamp = datetime.now().isoformat()
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        metrics = {
            "timestamp": timestamp,
            "uptime": time.time() - self.start_time,
            "memory": {
                "used_gb": round(memory.used / (1024**3), 2),
                "total_gb": round(memory.total / (1024**3), 2),
                "percent": memory.percent
            },
            "cpu_percent": cpu_percent
        }
        
        # GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu"] = {
                    "allocated_gb": round(torch.cuda.memory_allocated() / (1024**3), 2),
                    "cached_gb": round(torch.cuda.memory_reserved() / (1024**3), 2)
                }
        except:
            pass
        
        self.metrics[timestamp] = metrics
        
        # Keep only last 200 entries
        if len(self.metrics) > 200:
            oldest_keys = sorted(self.metrics.keys())[:-200]
            for key in oldest_keys:
                del self.metrics[key]
                
        return metrics
    
    def _collect_metrics(self):
        """Internal metrics collection for monitoring loop"""
        return self.collect_metrics()
    
    def track_operation(self, operation_name: str):
        """Decorator to track operation performance"""
        def decorator(func):
            def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = func(*args, **kwargs)
                    duration = time.time() - start_time
                    self._log_operation(operation_name, duration, True)
                    return result
                except Exception as e:
                    duration = time.time() - start_time
                    self._log_operation(operation_name, duration, False, str(e))
                    raise
            return wrapper
        return decorator
    
    def _log_operation(self, name: str, duration: float, success: bool, error: str = None):
        """Log operation performance"""
        if "operations" not in self.metrics:
            self.metrics["operations"] = []
        
        operation_log = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "duration": duration,
            "success": success
        }
        
        if error:
            operation_log["error"] = error
        
        self.metrics["operations"].append(operation_log)
        
        # Keep only last 50 operations
        if len(self.metrics["operations"]) > 50:
            self.metrics["operations"] = self.metrics["operations"][-50:]
    
    def get_health_report(self) -> Dict[str, Any]:
        """Generate health report"""
        if not self.metrics:
            return {"status": "no_data"}
        
        # Get latest metrics
        latest_timestamp = max(self.metrics.keys())
        latest = self.metrics[latest_timestamp]
        
        # Calculate health score
        health_score = 100
        issues = []
        
        # Memory health
        if latest["memory"]["percent"] > 90:
            health_score -= 30
            issues.append("High memory usage")
        elif latest["memory"]["percent"] > 80:
            health_score -= 15
            issues.append("Elevated memory usage")
        
        # CPU health
        if latest["cpu_percent"] > 90:
            health_score -= 20
            issues.append("High CPU usage")
        
        # Operation success rate
        operations = self.metrics.get("operations", [])
        if operations:
            success_rate = sum(1 for op in operations if op["success"]) / len(operations)
            if success_rate < 0.8:
                health_score -= 25
                issues.append("Low operation success rate")
        
        health_status = "excellent" if health_score >= 90 else                         "good" if health_score >= 70 else                         "fair" if health_score >= 50 else "poor"
        
        return {
            "status": health_status,
            "score": health_score,
            "uptime": latest["uptime"],
            "memory_usage": latest["memory"]["percent"],
            "cpu_usage": latest["cpu_percent"],
            "issues": issues,
            "timestamp": latest_timestamp
        }
    
    def save_report(self, file_path: Path):
        """Save performance report to file"""
        report = {
            "health": self.get_health_report(),
            "full_metrics": self.metrics
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2)

# Global performance monitor
performance_monitor = PerformanceMonitor()
