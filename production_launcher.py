#!/usr/bin/env python3
"""
🏭 JARVIS Production Launcher
ระบบเปิดใช้งาน JARVIS สำหรับ Production พร้อม monitoring
"""

import os
import sys
import logging
import signal
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional
import psutil
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Production imports
try:
    from jarvis_final_complete import JarvisFinalComplete
    import structlog
    from prometheus_client import start_http_server, Counter, Histogram, Gauge
except ImportError as e:
    print(f"❌ Production dependencies missing: {e}")
    print("💡 Install with: pip install -r requirements_production.txt")
    sys.exit(1)

class ProductionMonitoring:
    """ระบบ monitoring สำหรับ production"""
    
    def __init__(self):
        # Prometheus metrics
        self.interactions_counter = Counter('jarvis_interactions_total', 
                                          'Total number of interactions', 
                                          ['type', 'language'])
        
        self.response_time_histogram = Histogram('jarvis_response_time_seconds',
                                               'Response time in seconds',
                                               ['operation'])
        
        self.memory_usage_gauge = Gauge('jarvis_memory_usage_bytes',
                                      'Memory usage in bytes')
        
        self.cpu_usage_gauge = Gauge('jarvis_cpu_usage_percent',
                                   'CPU usage percentage')
        
        self.active_connections_gauge = Gauge('jarvis_active_connections',
                                            'Number of active connections')
        
        # Health status
        self.health_status = {"status": "starting", "timestamp": time.time()}
        
    def record_interaction(self, interaction_type: str, language: str):
        """บันทึกการโต้ตอบ"""
        self.interactions_counter.labels(type=interaction_type, language=language).inc()
    
    def record_response_time(self, operation: str, duration: float):
        """บันทึกเวลาตอบสนอง"""
        self.response_time_histogram.labels(operation=operation).observe(duration)
    
    def update_system_metrics(self):
        """อัพเดทเมตริกระบบ"""
        process = psutil.Process()
        
        # Memory usage
        memory_info = process.memory_info()
        self.memory_usage_gauge.set(memory_info.rss)
        
        # CPU usage
        cpu_percent = process.cpu_percent()
        self.cpu_usage_gauge.set(cpu_percent)
    
    def set_health_status(self, status: str, details: Dict[str, Any] = None):
        """ตั้งค่าสถานะสุขภาพ"""
        self.health_status = {
            "status": status,
            "timestamp": time.time(),
            "details": details or {}
        }

class ProductionJarvis:
    """JARVIS Production System"""
    
    def __init__(self, config_path: str = "config/production_config.yaml"):
        self.config_path = config_path
        self.monitoring = ProductionMonitoring()
        self.jarvis_system = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        
        # Setup structured logging
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        self.logger = structlog.get_logger(__name__)
        
        # Signal handlers for graceful shutdown
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """จัดการสัญญาณ shutdown"""
        self.logger.info("Received shutdown signal", signal=signum)
        self.shutdown()
    
    def start_production_server(self):
        """เริ่มเซิร์ฟเวอร์ production"""
        try:
            self.logger.info("🚀 Starting JARVIS Production Server...")
            
            # Start monitoring
            self.monitoring.set_health_status("starting")
            
            # Start Prometheus metrics server
            prometheus_port = 8082
            start_http_server(prometheus_port)
            self.logger.info(f"📊 Prometheus metrics available at ::{prometheus_port}/metrics")
            
            # Initialize JARVIS system
            self.jarvis_system = JarvisFinalComplete(self.config_path)
            
            # Start JARVIS
            if self.jarvis_system.start_system():
                self.is_running = True
                self.monitoring.set_health_status("healthy", {
                    "version": "2.0.1",
                    "features": ["voice", "memory", "wake_word"],
                    "startup_time": time.time()
                })
                
                self.logger.info("✅ JARVIS Production Server started successfully")
                
                # Start monitoring thread
                monitoring_thread = threading.Thread(target=self._monitoring_loop)
                monitoring_thread.daemon = True
                monitoring_thread.start()
                
                # Start main service loop
                self._service_loop()
                
            else:
                self.logger.error("❌ Failed to start JARVIS system")
                self.monitoring.set_health_status("failed", {"error": "startup_failed"})
                return False
                
        except Exception as e:
            self.logger.error("💥 Production server startup failed", error=str(e))
            self.monitoring.set_health_status("error", {"error": str(e)})
            return False
    
    def _service_loop(self):
        """ลูปหลักของบริการ"""
        self.logger.info("🔄 Service loop started")
        
        try:
            while self.is_running and not self.shutdown_event.is_set():
                # Health check
                if self.jarvis_system and self.jarvis_system.is_active:
                    self.monitoring.set_health_status("healthy")
                else:
                    self.monitoring.set_health_status("degraded", {"issue": "jarvis_inactive"})
                
                # Wait for shutdown signal
                if self.shutdown_event.wait(timeout=10):
                    break
                    
        except Exception as e:
            self.logger.error("❌ Service loop error", error=str(e))
            self.monitoring.set_health_status("error", {"error": str(e)})
    
    def _monitoring_loop(self):
        """ลูป monitoring"""
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Update system metrics
                self.monitoring.update_system_metrics()
                
                # Check JARVIS status
                if self.jarvis_system:
                    status = self.jarvis_system.get_status()
                    
                    # Record metrics
                    if status.get('statistics'):
                        stats = status['statistics']
                        total_interactions = stats.get('total_interactions', 0)
                        # Update Prometheus counters based on current stats
                
                time.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                self.logger.warning("⚠️ Monitoring error", error=str(e))
                time.sleep(60)  # Wait longer on error
    
    def shutdown(self):
        """ปิดระบบอย่างสวยงาม"""
        self.logger.info("🛑 Shutting down JARVIS Production Server...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop JARVIS system
        if self.jarvis_system:
            self.jarvis_system.stop_system()
        
        # Update health status
        self.monitoring.set_health_status("shutdown")
        
        self.logger.info("✅ JARVIS Production Server shutdown complete")
    
    def get_health_status(self) -> Dict[str, Any]:
        """สถานะสุขภาพของระบบ"""
        base_status = self.monitoring.health_status.copy()
        
        if self.jarvis_system:
            jarvis_status = self.jarvis_system.get_status()
            base_status["jarvis"] = {
                "active": jarvis_status.get("active", False),
                "components": jarvis_status.get("components", {}),
                "statistics": jarvis_status.get("statistics", {})
            }
        
        return base_status

# Web API for health checks and management
try:
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
    
    app = FastAPI(title="JARVIS Production API", version="2.0.1")
    
    # Add CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Global production instance
    production_jarvis = None
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        if production_jarvis:
            return production_jarvis.get_health_status()
        return {"status": "not_initialized"}
    
    @app.get("/status")
    async def get_status():
        """Full system status"""
        if production_jarvis and production_jarvis.jarvis_system:
            return production_jarvis.jarvis_system.get_status()
        raise HTTPException(status_code=503, detail="JARVIS not available")
    
    @app.post("/interact")
    async def interact(message: str, language: str = "auto"):
        """Text interaction endpoint"""
        if production_jarvis and production_jarvis.jarvis_system:
            start_time = time.time()
            
            # Process message
            production_jarvis.jarvis_system.process_text_message(message, language)
            
            # Record metrics
            duration = time.time() - start_time
            production_jarvis.monitoring.record_response_time("text_interaction", duration)
            production_jarvis.monitoring.record_interaction("text", language)
            
            return {"status": "processed", "response_time": duration}
        
        raise HTTPException(status_code=503, detail="JARVIS not available")

except ImportError:
    app = None
    print("⚠️ FastAPI not available - web API disabled")

def start_web_api(jarvis_instance, port: int = 8080):
    """เริ่ม Web API"""
    if app:
        global production_jarvis
        production_jarvis = jarvis_instance
        
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

def main():
    """ฟังก์ชันหลัก"""
    print("🏭 JARVIS Production Launcher")
    print("=" * 40)
    
    # Check if running in container
    if os.path.exists('/.dockerenv'):
        print("🐳 Running in Docker container")
        config_path = "/app/config/production_config.yaml"
    else:
        print("🖥️ Running on host system")
        config_path = "config/production_config.yaml"
    
    # Create production instance
    production_system = ProductionJarvis(config_path)
    
    # Start web API in separate thread if available
    if app:
        api_thread = threading.Thread(
            target=start_web_api, 
            args=(production_system, 8080)
        )
        api_thread.daemon = True
        api_thread.start()
        print("🌐 Web API starting on port 8080")
    
    # Start production server
    try:
        production_system.start_production_server()
    except KeyboardInterrupt:
        print("\n🛑 Received keyboard interrupt")
    finally:
        production_system.shutdown()

if __name__ == "__main__":
    main()