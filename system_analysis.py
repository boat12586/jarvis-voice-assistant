#!/usr/bin/env python3
"""
Comprehensive System Analysis for JARVIS Voice Assistant
Analyzes all processes, identifies issues, and suggests improvements
"""

import sys
import os
import psutil
import time
import threading
from pathlib import Path
import json
import logging
from typing import Dict, List, Any
import traceback

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class SystemAnalyzer:
    """Comprehensive system analyzer for JARVIS"""
    
    def __init__(self):
        self.analysis_results = {}
        self.performance_metrics = {}
        self.issues_found = []
        self.improvement_suggestions = []
        
    def analyze_startup_performance(self):
        """Analyze system startup performance"""
        print("🚀 Analyzing Startup Performance...")
        
        startup_times = {}
        issues = []
        
        try:
            # Test component initialization times
            components = [
                "ConfigManager",
                "Logger", 
                "SpeechRecognizer",
                "TextToSpeech",
                "RAGSystem",
                "AIEngine"
            ]
            
            for component in components:
                start_time = time.time()
                
                try:
                    if component == "ConfigManager":
                        from system.config_manager import ConfigManager
                        obj = ConfigManager()
                        
                    elif component == "Logger":
                        from system.logger import setup_logger
                        obj = setup_logger("test")
                        
                    elif component == "SpeechRecognizer":
                        from voice.speech_recognizer import SimpleSpeechRecognizer
                        obj = SimpleSpeechRecognizer()
                        
                    elif component == "TextToSpeech":
                        from voice.text_to_speech import TextToSpeech
                        from system.config_manager import ConfigManager
                        config = ConfigManager()
                        obj = TextToSpeech(config)
                        
                    elif component == "RAGSystem":
                        from ai.rag_system import RAGSystem
                        from system.config_manager import ConfigManager
                        config = ConfigManager()
                        obj = RAGSystem(config)
                        time.sleep(2)  # Wait for initialization
                        
                    elif component == "AIEngine":
                        from ai.ai_engine import AIEngine
                        from system.config_manager import ConfigManager
                        config = ConfigManager()
                        obj = AIEngine(config)
                        
                    end_time = time.time()
                    startup_times[component] = end_time - start_time
                    
                    if startup_times[component] > 10:
                        issues.append(f"⚠️ {component} slow startup: {startup_times[component]:.2f}s")
                    
                except Exception as e:
                    end_time = time.time()
                    startup_times[component] = end_time - start_time
                    issues.append(f"❌ {component} startup failed: {e}")
            
            self.analysis_results["startup"] = {
                "times": startup_times,
                "issues": issues,
                "total_time": sum(startup_times.values())
            }
            
            print(f"   📊 Total startup time: {sum(startup_times.values()):.2f}s")
            for comp, time_taken in startup_times.items():
                status = "✅" if time_taken < 5 else "⚠️" if time_taken < 10 else "❌"
                print(f"      {status} {comp}: {time_taken:.2f}s")
                
            return len(issues) == 0
            
        except Exception as e:
            print(f"   ❌ Startup analysis failed: {e}")
            return False
    
    def analyze_memory_usage(self):
        """Analyze memory usage patterns"""
        print("\n💾 Analyzing Memory Usage...")
        
        try:
            # Get current process memory
            process = psutil.Process()
            memory_info = process.memory_info()
            
            # System memory
            system_memory = psutil.virtual_memory()
            
            memory_analysis = {
                "process_memory_mb": memory_info.rss / 1024 / 1024,
                "system_memory_total_gb": system_memory.total / 1024 / 1024 / 1024,
                "system_memory_used_percent": system_memory.percent,
                "system_memory_available_gb": system_memory.available / 1024 / 1024 / 1024
            }
            
            issues = []
            
            # Check for memory issues
            if memory_analysis["process_memory_mb"] > 2000:
                issues.append(f"⚠️ High process memory usage: {memory_analysis['process_memory_mb']:.1f}MB")
                
            if memory_analysis["system_memory_used_percent"] > 80:
                issues.append(f"⚠️ High system memory usage: {memory_analysis['system_memory_used_percent']:.1f}%")
                
            if memory_analysis["system_memory_available_gb"] < 2:
                issues.append(f"❌ Low available memory: {memory_analysis['system_memory_available_gb']:.1f}GB")
            
            self.analysis_results["memory"] = {
                "metrics": memory_analysis,
                "issues": issues
            }
            
            print(f"   📊 Process Memory: {memory_analysis['process_memory_mb']:.1f}MB")
            print(f"   📊 System Memory: {memory_analysis['system_memory_used_percent']:.1f}% used")
            print(f"   📊 Available Memory: {memory_analysis['system_memory_available_gb']:.1f}GB")
            
            for issue in issues:
                print(f"   {issue}")
                
            return len(issues) == 0
            
        except Exception as e:
            print(f"   ❌ Memory analysis failed: {e}")
            return False
    
    def analyze_voice_pipeline(self):
        """Analyze voice processing pipeline"""
        print("\n🎙️ Analyzing Voice Pipeline...")
        
        try:
            pipeline_issues = []
            performance_data = {}
            
            # Test speech recognition
            start_time = time.time()
            try:
                from voice.speech_recognizer import SimpleSpeechRecognizer
                recognizer = SimpleSpeechRecognizer()
                
                # Test with dummy audio
                import numpy as np
                dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second silence
                
                recognition_start = time.time()
                # Note: This will likely return empty for silence, but tests the pipeline
                result = recognizer.transcribe_audio(dummy_audio)
                recognition_time = time.time() - recognition_start
                
                performance_data["speech_recognition"] = recognition_time
                
                if recognition_time > 5:
                    pipeline_issues.append(f"⚠️ Slow speech recognition: {recognition_time:.2f}s")
                    
            except Exception as e:
                pipeline_issues.append(f"❌ Speech recognition error: {e}")
                performance_data["speech_recognition"] = -1
            
            # Test TTS system
            try:
                from voice.text_to_speech import TextToSpeech
                from system.config_manager import ConfigManager
                
                tts_start = time.time()
                config = ConfigManager()
                tts = TextToSpeech(config)
                tts_init_time = time.time() - tts_start
                
                performance_data["tts_init"] = tts_init_time
                
                if tts_init_time > 10:
                    pipeline_issues.append(f"⚠️ Slow TTS initialization: {tts_init_time:.2f}s")
                    
            except Exception as e:
                pipeline_issues.append(f"❌ TTS system error: {e}")
                performance_data["tts_init"] = -1
            
            # Test audio devices
            try:
                import sounddevice as sd
                devices = sd.query_devices()
                input_devices = [d for d in devices if d['max_input_channels'] > 0]
                output_devices = [d for d in devices if d['max_output_channels'] > 0]
                
                performance_data["input_devices"] = len(input_devices)
                performance_data["output_devices"] = len(output_devices)
                
                if len(input_devices) == 0:
                    pipeline_issues.append("❌ No audio input devices found")
                if len(output_devices) == 0:
                    pipeline_issues.append("❌ No audio output devices found")
                    
            except Exception as e:
                pipeline_issues.append(f"❌ Audio device detection error: {e}")
            
            self.analysis_results["voice_pipeline"] = {
                "performance": performance_data,
                "issues": pipeline_issues
            }
            
            print(f"   📊 Speech Recognition: {performance_data.get('speech_recognition', 'N/A')}")
            print(f"   📊 TTS Initialization: {performance_data.get('tts_init', 'N/A')}")
            print(f"   📊 Input Devices: {performance_data.get('input_devices', 0)}")
            print(f"   📊 Output Devices: {performance_data.get('output_devices', 0)}")
            
            for issue in pipeline_issues:
                print(f"   {issue}")
                
            return len(pipeline_issues) == 0
            
        except Exception as e:
            print(f"   ❌ Voice pipeline analysis failed: {e}")
            return False
    
    def analyze_ai_system(self):
        """Analyze AI system performance"""
        print("\n🧠 Analyzing AI System...")
        
        try:
            ai_issues = []
            ai_performance = {}
            
            # Test RAG system
            rag_start = time.time()
            try:
                from ai.rag_system import RAGSystem
                from system.config_manager import ConfigManager
                
                config = ConfigManager()
                rag = RAGSystem(config)
                time.sleep(2)  # Wait for initialization
                
                rag_init_time = time.time() - rag_start
                ai_performance["rag_init"] = rag_init_time
                
                if rag_init_time > 15:
                    ai_issues.append(f"⚠️ Slow RAG initialization: {rag_init_time:.2f}s")
                
                # Test search
                if rag.is_ready:
                    search_start = time.time()
                    results = rag.search("test query", top_k=3)
                    search_time = time.time() - search_start
                    
                    ai_performance["search_time"] = search_time
                    ai_performance["search_results"] = len(results)
                    
                    if search_time > 2:
                        ai_issues.append(f"⚠️ Slow search performance: {search_time:.2f}s")
                        
                    if len(results) == 0:
                        ai_issues.append("⚠️ No search results found - knowledge base may be empty")
                else:
                    ai_issues.append("❌ RAG system not ready")
                    
            except Exception as e:
                ai_issues.append(f"❌ RAG system error: {e}")
                ai_performance["rag_init"] = -1
            
            # Test LLM engine
            try:
                from ai.llm_engine import MistralEngine
                from system.config_manager import ConfigManager
                
                config = ConfigManager()
                llm_config = config.get("ai", {})
                
                # Just test initialization, not actual inference
                llm_start = time.time()
                # engine = MistralEngine(llm_config)  # This might be heavy
                llm_init_time = time.time() - llm_start
                
                ai_performance["llm_check"] = llm_init_time
                
            except Exception as e:
                ai_issues.append(f"⚠️ LLM engine initialization issue: {e}")
                ai_performance["llm_check"] = -1
            
            # Check model availability
            try:
                import torch
                cuda_available = torch.cuda.is_available()
                ai_performance["cuda_available"] = cuda_available
                
                if cuda_available:
                    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                    ai_performance["gpu_memory_gb"] = gpu_memory
                    
                    if gpu_memory < 4:
                        ai_issues.append(f"⚠️ Limited GPU memory: {gpu_memory:.1f}GB")
                else:
                    ai_issues.append("⚠️ CUDA not available - using CPU inference")
                    
            except Exception as e:
                ai_issues.append(f"⚠️ GPU detection error: {e}")
            
            self.analysis_results["ai_system"] = {
                "performance": ai_performance,
                "issues": ai_issues
            }
            
            print(f"   📊 RAG Initialization: {ai_performance.get('rag_init', 'N/A'):.2f}s")
            print(f"   📊 Search Performance: {ai_performance.get('search_time', 'N/A')}")
            print(f"   📊 Search Results: {ai_performance.get('search_results', 'N/A')}")
            print(f"   📊 CUDA Available: {ai_performance.get('cuda_available', 'N/A')}")
            
            for issue in ai_issues:
                print(f"   {issue}")
                
            return len(ai_issues) <= 2  # Allow some warnings
            
        except Exception as e:
            print(f"   ❌ AI system analysis failed: {e}")
            return False
    
    def analyze_configuration_system(self):
        """Analyze configuration and data management"""
        print("\n⚙️ Analyzing Configuration System...")
        
        try:
            config_issues = []
            config_metrics = {}
            
            # Test configuration loading
            try:
                from system.config_manager import ConfigManager
                
                config_start = time.time()
                config = ConfigManager()
                config_data = config.get_config()
                config_load_time = time.time() - config_start
                
                config_metrics["load_time"] = config_load_time
                config_metrics["sections"] = len(config_data)
                
                # Check required sections
                required_sections = ["ui", "voice", "ai", "rag", "features", "system"]
                missing_sections = [s for s in required_sections if s not in config_data]
                
                if missing_sections:
                    config_issues.append(f"❌ Missing config sections: {missing_sections}")
                
                # Check file paths
                paths_to_check = [
                    Path("data/knowledge_base.json"),
                    Path("config/default_config.yaml"),
                    Path("logs"),
                    Path("data"),
                    Path("models")
                ]
                
                missing_paths = []
                for path in paths_to_check:
                    if not path.exists():
                        missing_paths.append(str(path))
                
                if missing_paths:
                    config_issues.append(f"⚠️ Missing paths: {missing_paths}")
                
                config_metrics["missing_paths"] = len(missing_paths)
                
            except Exception as e:
                config_issues.append(f"❌ Configuration loading error: {e}")
                config_metrics["load_time"] = -1
            
            # Test knowledge base
            try:
                kb_file = Path("data/knowledge_base.json")
                if kb_file.exists():
                    with open(kb_file, 'r', encoding='utf-8') as f:
                        kb_data = json.load(f)
                    
                    config_metrics["kb_categories"] = len(kb_data)
                    
                    total_items = 0
                    for category, items in kb_data.items():
                        if isinstance(items, dict):
                            total_items += len(items)
                        elif isinstance(items, list):
                            total_items += len(items)
                    
                    config_metrics["kb_total_items"] = total_items
                    
                    if total_items < 10:
                        config_issues.append(f"⚠️ Limited knowledge base content: {total_items} items")
                        
                else:
                    config_issues.append("❌ Knowledge base file not found")
                    
            except Exception as e:
                config_issues.append(f"❌ Knowledge base analysis error: {e}")
            
            self.analysis_results["configuration"] = {
                "metrics": config_metrics,
                "issues": config_issues
            }
            
            print(f"   📊 Config Load Time: {config_metrics.get('load_time', 'N/A'):.3f}s")
            print(f"   📊 Config Sections: {config_metrics.get('sections', 'N/A')}")
            print(f"   📊 KB Categories: {config_metrics.get('kb_categories', 'N/A')}")
            print(f"   📊 KB Total Items: {config_metrics.get('kb_total_items', 'N/A')}")
            
            for issue in config_issues:
                print(f"   {issue}")
                
            return len([i for i in config_issues if i.startswith("❌")]) == 0
            
        except Exception as e:
            print(f"   ❌ Configuration analysis failed: {e}")
            return False
    
    def analyze_ui_system(self):
        """Analyze UI system and responsiveness"""
        print("\n🖥️ Analyzing UI System...")
        
        try:
            ui_issues = []
            ui_metrics = {}
            
            # Test PyQt6 availability
            try:
                from PyQt6.QtWidgets import QApplication
                from PyQt6.QtCore import Qt
                
                ui_metrics["pyqt6_available"] = True
                
                # Test basic UI components (without actually creating windows)
                try:
                    from ui.main_window import MainWindow
                    from ui.styles import get_glassmorphic_style
                    
                    ui_metrics["ui_components_importable"] = True
                    
                except Exception as e:
                    ui_issues.append(f"⚠️ UI component import error: {e}")
                    ui_metrics["ui_components_importable"] = False
                
            except Exception as e:
                ui_issues.append(f"❌ PyQt6 not available: {e}")
                ui_metrics["pyqt6_available"] = False
            
            # Check display availability (for WSL/headless systems)
            display_available = os.environ.get('DISPLAY') is not None
            ui_metrics["display_available"] = display_available
            
            if not display_available:
                ui_issues.append("⚠️ No display available - GUI mode will not work")
            
            # Check for style/theme issues
            try:
                from ui.styles import get_glassmorphic_style
                style = get_glassmorphic_style()
                
                if "backdrop-filter" in style:
                    ui_issues.append("⚠️ CSS backdrop-filter may not work in all environments")
                    
            except Exception as e:
                ui_issues.append(f"⚠️ Style system error: {e}")
            
            self.analysis_results["ui_system"] = {
                "metrics": ui_metrics,
                "issues": ui_issues
            }
            
            print(f"   📊 PyQt6 Available: {ui_metrics.get('pyqt6_available', 'N/A')}")
            print(f"   📊 Display Available: {ui_metrics.get('display_available', 'N/A')}")
            print(f"   📊 UI Components: {ui_metrics.get('ui_components_importable', 'N/A')}")
            
            for issue in ui_issues:
                print(f"   {issue}")
                
            return len([i for i in ui_issues if i.startswith("❌")]) == 0
            
        except Exception as e:
            print(f"   ❌ UI system analysis failed: {e}")
            return False
    
    def generate_improvement_suggestions(self):
        """Generate improvement suggestions based on analysis"""
        print("\n💡 Generating Improvement Suggestions...")
        
        suggestions = []
        
        # Startup performance improvements
        startup_data = self.analysis_results.get("startup", {})
        if startup_data.get("total_time", 0) > 20:
            suggestions.append({
                "category": "Performance",
                "priority": "High",
                "issue": "Slow startup time",
                "suggestion": "Implement lazy loading for heavy components (RAG, LLM)",
                "implementation": "Load models in background threads after basic UI is ready"
            })
        
        # Memory optimization
        memory_data = self.analysis_results.get("memory", {})
        if memory_data.get("metrics", {}).get("process_memory_mb", 0) > 1500:
            suggestions.append({
                "category": "Memory",
                "priority": "Medium",
                "issue": "High memory usage",
                "suggestion": "Implement model quantization and memory pooling",
                "implementation": "Use INT8 quantization, shared memory for embeddings"
            })
        
        # Voice pipeline improvements
        voice_data = self.analysis_results.get("voice_pipeline", {})
        voice_perf = voice_data.get("performance", {})
        if voice_perf.get("speech_recognition", 0) > 3:
            suggestions.append({
                "category": "Voice",
                "priority": "High", 
                "issue": "Slow speech recognition",
                "suggestion": "Optimize Whisper model and implement streaming",
                "implementation": "Use smaller Whisper model or implement real-time streaming"
            })
        
        # AI system improvements
        ai_data = self.analysis_results.get("ai_system", {})
        ai_perf = ai_data.get("performance", {})
        if not ai_perf.get("cuda_available", False):
            suggestions.append({
                "category": "AI",
                "priority": "High",
                "issue": "No GPU acceleration",
                "suggestion": "Enable CUDA support or optimize CPU inference",
                "implementation": "Install CUDA-enabled PyTorch or implement CPU optimizations"
            })
        
        if ai_perf.get("search_results", 0) == 0:
            suggestions.append({
                "category": "Knowledge",
                "priority": "High",
                "issue": "Empty search results",
                "suggestion": "Fix knowledge base indexing and search thresholds",
                "implementation": "Rebuild vector index, adjust similarity thresholds, validate embeddings"
            })
        
        # Configuration improvements
        config_data = self.analysis_results.get("configuration", {})
        if config_data.get("metrics", {}).get("missing_paths", 0) > 0:
            suggestions.append({
                "category": "System",
                "priority": "Medium",
                "issue": "Missing directories/files",
                "suggestion": "Implement auto-creation of required directories",
                "implementation": "Add initialization scripts to create missing paths"
            })
        
        # UI system improvements
        ui_data = self.analysis_results.get("ui_system", {})
        if not ui_data.get("metrics", {}).get("display_available", True):
            suggestions.append({
                "category": "UI",
                "priority": "Medium",
                "issue": "No display available",
                "suggestion": "Enhance headless mode capabilities",
                "implementation": "Add web interface, CLI mode, or remote desktop support"
            })
        
        # General enhancement suggestions
        suggestions.extend([
            {
                "category": "Features",
                "priority": "High",
                "issue": "Limited conversation context",
                "suggestion": "Implement conversation memory and context management",
                "implementation": "Add conversation history, context window management, personality consistency"
            },
            {
                "category": "Features", 
                "priority": "Medium",
                "issue": "No voice activation",
                "suggestion": "Add wake word detection ('Hey JARVIS')",
                "implementation": "Integrate Porcupine or similar wake word detection"
            },
            {
                "category": "Intelligence",
                "priority": "High",
                "issue": "Basic AI responses",
                "suggestion": "Enhance reasoning capabilities with DeepSeek-R1",
                "implementation": "Integrate chain-of-thought, multi-step reasoning, task planning"
            },
            {
                "category": "Language",
                "priority": "Medium",
                "issue": "Limited Thai language support",
                "suggestion": "Enhance Thai language processing",
                "implementation": "Add Thai tokenization, cultural context, Thai-specific models"
            },
            {
                "category": "Performance",
                "priority": "Medium",
                "issue": "No performance monitoring",
                "suggestion": "Add real-time performance monitoring",
                "implementation": "Track response times, memory usage, error rates, user satisfaction"
            }
        ])
        
        self.improvement_suggestions = suggestions
        
        # Display suggestions by priority
        for priority in ["High", "Medium", "Low"]:
            priority_suggestions = [s for s in suggestions if s["priority"] == priority]
            if priority_suggestions:
                print(f"\n   🔥 {priority} Priority Improvements:")
                for i, suggestion in enumerate(priority_suggestions, 1):
                    print(f"      {i}. {suggestion['category']}: {suggestion['suggestion']}")
                    print(f"         💡 {suggestion['implementation']}")
        
        return suggestions
    
    def generate_detailed_report(self):
        """Generate detailed analysis report"""
        report = {
            "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
                "cpu_count": psutil.cpu_count(),
                "memory_total_gb": psutil.virtual_memory().total / 1024**3
            },
            "analysis_results": self.analysis_results,
            "improvement_suggestions": self.improvement_suggestions,
            "summary": {
                "total_issues": sum(len(result.get("issues", [])) for result in self.analysis_results.values()),
                "critical_issues": 0,  # Count critical issues
                "suggestions_count": len(self.improvement_suggestions),
                "overall_health": "Unknown"
            }
        }
        
        # Calculate critical issues
        critical_keywords = ["❌", "failed", "error", "not found", "not available"]
        for result in self.analysis_results.values():
            for issue in result.get("issues", []):
                if any(keyword in issue for keyword in critical_keywords):
                    report["summary"]["critical_issues"] += 1
        
        # Determine overall health
        total_issues = report["summary"]["total_issues"]
        critical_issues = report["summary"]["critical_issues"]
        
        if critical_issues == 0 and total_issues <= 3:
            report["summary"]["overall_health"] = "Excellent"
        elif critical_issues <= 1 and total_issues <= 6:
            report["summary"]["overall_health"] = "Good"
        elif critical_issues <= 3 and total_issues <= 10:
            report["summary"]["overall_health"] = "Fair"
        else:
            report["summary"]["overall_health"] = "Needs Attention"
        
        return report

def main():
    """Main analysis function"""
    print("🔍 JARVIS Voice Assistant - Comprehensive System Analysis")
    print("=" * 80)
    
    analyzer = SystemAnalyzer()
    
    # Run all analysis modules
    analysis_modules = [
        ("Startup Performance", analyzer.analyze_startup_performance),
        ("Memory Usage", analyzer.analyze_memory_usage),
        ("Voice Pipeline", analyzer.analyze_voice_pipeline),
        ("AI System", analyzer.analyze_ai_system),
        ("Configuration", analyzer.analyze_configuration_system),
        ("UI System", analyzer.analyze_ui_system)
    ]
    
    results = []
    
    for module_name, analysis_func in analysis_modules:
        try:
            result = analysis_func()
            results.append((module_name, result))
        except Exception as e:
            print(f"❌ Error analyzing {module_name}: {e}")
            traceback.print_exc()
            results.append((module_name, False))
    
    # Generate improvement suggestions
    suggestions = analyzer.generate_improvement_suggestions()
    
    # Generate detailed report
    report = analyzer.generate_detailed_report()
    
    # Save report
    report_file = Path("system_analysis_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Display summary
    print("\n" + "=" * 80)
    print("📋 SYSTEM ANALYSIS SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for module_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{module_name:20} {status}")
    
    print(f"\nAnalysis Results: {passed}/{total} modules passed")
    print(f"Total Issues Found: {report['summary']['total_issues']}")
    print(f"Critical Issues: {report['summary']['critical_issues']}")
    print(f"Improvement Suggestions: {report['summary']['suggestions_count']}")
    print(f"Overall System Health: {report['summary']['overall_health']}")
    
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Recommendations
    health = report['summary']['overall_health']
    if health == "Excellent":
        print("\n🎉 System is in excellent condition! Ready for advanced features.")
    elif health == "Good":
        print("\n👍 System is working well. Consider implementing suggested improvements.")
    elif health == "Fair":
        print("\n⚠️ System needs some attention. Address high-priority issues first.")
    else:
        print("\n🚨 System needs significant improvements. Address critical issues immediately.")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)