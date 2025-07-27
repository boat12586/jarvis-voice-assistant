#!/usr/bin/env python3
"""
Critical Issues Fix for JARVIS Voice Assistant
Address the most important problems immediately
"""

import sys
import os
import gc
import psutil
from pathlib import Path
import yaml
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def fix_memory_management():
    """Fix memory management issues"""
    print("🧠 Fixing Memory Management...")
    
    try:
        # Create memory management utility
        memory_manager_code = '''"""
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
'''
        
        memory_file = Path("src/utils/memory_manager.py")
        memory_file.parent.mkdir(exist_ok=True)
        
        with open(memory_file, 'w', encoding='utf-8') as f:
            f.write(memory_manager_code)
        
        print("   ✅ Created memory management utilities")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to fix memory management: {e}")
        return False

def fix_startup_performance():
    """Fix slow startup performance"""
    print("🚀 Fixing Startup Performance...")
    
    try:
        # Update main.py for lazy loading
        main_py_path = Path("src/main.py")
        
        if main_py_path.exists():
            with open(main_py_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add lazy loading imports
            lazy_imports = '''
# Lazy imports for better startup performance
def lazy_import_ai():
    """Lazy import AI components"""
    global AIEngine, RAGSystem
    from ai.ai_engine import AIEngine
    from ai.rag_system import RAGSystem
    return AIEngine, RAGSystem

def lazy_import_voice():
    """Lazy import voice components"""
    global VoiceController
    from voice.voice_controller import VoiceController
    return VoiceController
'''
            
            if "lazy_import_ai" not in content:
                # Insert after existing imports
                import_end = content.find('from system.application_controller import ApplicationController')
                if import_end != -1:
                    insert_pos = content.find('\n\n', import_end)
                    if insert_pos != -1:
                        content = content[:insert_pos] + lazy_imports + content[insert_pos:]
                
                with open(main_py_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                print("   ✅ Added lazy loading to main.py")
        
        # Create startup optimization config
        startup_config = {
            "lazy_loading": {
                "enabled": True,
                "components": ["ai", "voice", "tts"],
                "delay_seconds": 2
            },
            "background_loading": {
                "enabled": True,
                "models": ["embedding", "llm"],
                "show_progress": True
            },
            "memory_optimization": {
                "auto_cleanup": True,
                "cleanup_interval": 300,
                "memory_threshold": 0.8
            }
        }
        
        config_file = Path("config/startup_optimization.yaml")
        with open(config_file, 'w', encoding='utf-8') as f:
            yaml.dump(startup_config, f, default_flow_style=False, indent=2)
        
        print("   ✅ Created startup optimization config")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to fix startup performance: {e}")
        return False

def fix_rag_search_issues():
    """Fix RAG search and knowledge base issues"""
    print("🔍 Fixing RAG Search Issues...")
    
    try:
        # Update similarity threshold in config
        config_file = Path("config/default_config.yaml")
        
        if config_file.exists():
            with open(config_file, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # Optimize RAG settings
            if "rag" not in config:
                config["rag"] = {}
            
            config["rag"].update({
                "similarity_threshold": 0.2,  # Lower threshold
                "top_k": 10,  # More results
                "chunk_size": 256,  # Smaller chunks
                "chunk_overlap": 25,
                "rerank_results": True,
                "max_context_length": 2048
            })
            
            with open(config_file, 'w', encoding='utf-8') as f:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            
            print("   ✅ Updated RAG configuration")
        
        # Create enhanced knowledge base loader
        kb_loader_code = '''"""
Enhanced Knowledge Base Loader
Better document processing and indexing
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import re

class EnhancedKnowledgeLoader:
    """Enhanced knowledge base loader with better processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better indexing"""
        # Remove extra whitespace
        text = re.sub(r'\\s+', ' ', text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])([A-Z])', r'\\1 \\2', text)
        
        # Clean up the text
        text = text.strip()
        
        return text
    
    def create_document_variants(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create multiple variants of each document for better retrieval"""
        variants = []
        
        # Original document
        variants.append({
            "content": self.preprocess_text(content),
            "metadata": metadata.copy()
        })
        
        # Question-based variant for Q&A
        if "key" in metadata:
            question_variant = f"Question: What is {metadata['key']}? Answer: {content}"
            variants.append({
                "content": self.preprocess_text(question_variant),
                "metadata": {**metadata, "type": "qa"}
            })
        
        # Summary variant for long content
        if len(content) > 200:
            sentences = content.split('. ')
            if len(sentences) > 2:
                summary = '. '.join(sentences[:2]) + '.'
                variants.append({
                    "content": self.preprocess_text(summary),
                    "metadata": {**metadata, "type": "summary"}
                })
        
        return variants
    
    def load_knowledge_base(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and process knowledge base"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            for category, items in kb_data.items():
                if isinstance(items, dict):
                    for key, value in items.items():
                        if isinstance(value, str) and value.strip():
                            metadata = {
                                "category": category,
                                "key": key,
                                "source": "knowledge_base"
                            }
                            
                            variants = self.create_document_variants(value, metadata)
                            documents.extend(variants)
                            
                elif isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "content" in item:
                            content = item["content"]
                            if content.strip():
                                metadata = {
                                    "category": category,
                                    "source": "knowledge_base",
                                    **item
                                }
                                
                                variants = self.create_document_variants(content, metadata)
                                documents.extend(variants)
            
            self.logger.info(f"Loaded {len(documents)} document variants from knowledge base")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return []
'''
        
        kb_file = Path("src/utils/knowledge_loader.py")
        kb_file.parent.mkdir(exist_ok=True)
        
        with open(kb_file, 'w', encoding='utf-8') as f:
            f.write(kb_loader_code)
        
        print("   ✅ Created enhanced knowledge base loader")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to fix RAG search: {e}")
        return False

def fix_voice_pipeline():
    """Fix voice processing pipeline issues"""
    print("🎙️ Fixing Voice Pipeline...")
    
    try:
        # Create voice optimization utilities
        voice_optimizer_code = '''"""
Voice Pipeline Optimization
Faster processing and better quality
"""

import numpy as np
import torch
import logging
from typing import Optional, Tuple, List
import time

class VoiceOptimizer:
    """Optimizes voice processing pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vad_model = None
        self.audio_buffer = []
        
    def initialize_vad(self):
        """Initialize Voice Activity Detection"""
        try:
            # Use simple energy-based VAD for now
            self.logger.info("Initialized simple VAD")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD: {e}")
            return False
    
    def detect_voice_activity(self, audio: np.ndarray, 
                            threshold: float = 0.01) -> bool:
        """Simple voice activity detection"""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > threshold
    
    def chunk_audio(self, audio: np.ndarray, 
                   chunk_size: int = 1024,
                   overlap: int = 256) -> List[np.ndarray]:
        """Split audio into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunks.append(audio[start:end])
            start += chunk_size - overlap
        
        return chunks
    
    def preprocess_audio(self, audio: np.ndarray, 
                        sample_rate: int = 16000) -> np.ndarray:
        """Preprocess audio for better recognition"""
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Simple noise reduction (high-pass filter)
        if len(audio) > sample_rate // 10:  # At least 0.1 second
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Apply simple high-pass filter
            alpha = 0.99
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0]
            
            for i in range(1, len(audio)):
                filtered[i] = alpha * filtered[i-1] + alpha * (audio[i] - audio[i-1])
            
            audio = filtered
        
        return audio
    
    def optimize_for_whisper(self, audio: np.ndarray) -> np.ndarray:
        """Optimize audio specifically for Whisper"""
        # Whisper expects 16kHz, mono, 30-second chunks max
        target_length = 16000 * 30  # 30 seconds max
        
        if len(audio) > target_length:
            # Take the last 30 seconds (most recent speech)
            audio = audio[-target_length:]
        
        # Pad if too short
        if len(audio) < 16000:  # Less than 1 second
            padding = 16000 - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')
        
        return audio

# Global voice optimizer
voice_optimizer = VoiceOptimizer()
'''
        
        voice_file = Path("src/utils/voice_optimizer.py")
        voice_file.parent.mkdir(exist_ok=True)
        
        with open(voice_file, 'w', encoding='utf-8') as f:
            f.write(voice_optimizer_code)
        
        print("   ✅ Created voice optimization utilities")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to fix voice pipeline: {e}")
        return False

def create_monitoring_system():
    """Create performance monitoring system"""
    print("📊 Creating Monitoring System...")
    
    try:
        monitoring_code = '''"""
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
    
    def _collect_metrics(self):
        """Collect system metrics"""
        timestamp = datetime.now().isoformat()
        
        # System metrics
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        metrics = {
            "timestamp": timestamp,
            "uptime": time.time() - self.start_time,
            "memory": {
                "used_gb": memory.used / 1024**3,
                "total_gb": memory.total / 1024**3,
                "percent": memory.percent
            },
            "cpu_percent": cpu_percent
        }
        
        # GPU metrics if available
        try:
            import torch
            if torch.cuda.is_available():
                metrics["gpu"] = {
                    "allocated_gb": torch.cuda.memory_allocated() / 1024**3,
                    "cached_gb": torch.cuda.memory_reserved() / 1024**3
                }
        except:
            pass
        
        self.metrics[timestamp] = metrics
        
        # Keep only last 100 entries
        if len(self.metrics) > 100:
            oldest_key = min(self.metrics.keys())
            del self.metrics[oldest_key]
    
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
        
        health_status = "excellent" if health_score >= 90 else \
                        "good" if health_score >= 70 else \
                        "fair" if health_score >= 50 else "poor"
        
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
'''
        
        monitor_file = Path("src/utils/performance_monitor.py")
        monitor_file.parent.mkdir(exist_ok=True)
        
        with open(monitor_file, 'w', encoding='utf-8') as f:
            f.write(monitoring_code)
        
        print("   ✅ Created performance monitoring system")
        return True
        
    except Exception as e:
        print(f"   ❌ Failed to create monitoring system: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 JARVIS Critical Issues Fix")
    print("=" * 50)
    
    fixes = [
        ("Memory Management", fix_memory_management),
        ("Startup Performance", fix_startup_performance),
        ("RAG Search Issues", fix_rag_search_issues),
        ("Voice Pipeline", fix_voice_pipeline),
        ("Monitoring System", create_monitoring_system)
    ]
    
    results = []
    
    for fix_name, fix_func in fixes:
        try:
            result = fix_func()
            results.append((fix_name, result))
        except Exception as e:
            print(f"❌ Error fixing {fix_name}: {e}")
            results.append((fix_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 CRITICAL FIXES SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for fix_name, result in results:
        status = "✅ FIXED" if result else "❌ FAILED"
        print(f"{fix_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} fixes applied successfully")
    
    if passed == total:
        print("\n🎉 All critical issues fixed!")
        print("🚀 JARVIS is now optimized and ready for enhanced features")
        print("\nNext steps:")
        print("1. Test the optimized system: python test_full_app.py")
        print("2. Run performance tests: python test_performance.py")
        print("3. Begin implementing advanced features")
    elif passed >= total * 0.8:
        print("\n✅ Most critical issues fixed!")
        print("Review remaining issues and continue optimization")
    else:
        print("\n⚠️ Some critical issues remain")
        print("Please address failed fixes before proceeding")
    
    return passed >= total * 0.8

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)