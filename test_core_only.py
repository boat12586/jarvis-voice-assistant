#!/usr/bin/env python3
"""
🎯 JARVIS Core-Only Test
ทดสอบเฉพาะระบบหลักของ JARVIS โดยไม่ใช้ Voice processing

Version: 2.0.0 (2025 Edition)
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_core_system():
    """ทดสอบระบบหลักโดยไม่รวม Voice components"""
    print("🔧 ทดสอบระบบหลัก JARVIS (Core-Only)...")
    
    try:
        # Test configuration
        from system.config_manager_v2 import ConfigurationManager, JarvisConfig
        config_manager = ConfigurationManager()
        config = JarvisConfig()
        print("✅ Configuration system: OK")
        
        # Test performance monitor
        from utils.performance_monitor import PerformanceMonitor
        monitor = PerformanceMonitor()
        metrics = monitor.collect_metrics()
        print("✅ Performance monitoring: OK")
        
        # Test basic AI imports (without running)
        import torch
        import transformers
        print("✅ AI libraries available: OK")
        
        return True
        
    except Exception as e:
        print(f"❌ Core system test failed: {e}")
        return False

def test_gui_core():
    """ทดสอบ GUI ระบบหลัก"""
    print("\\n🖥️ ทดสอบ GUI core...")
    
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
        from PyQt6.QtCore import Qt, QTimer
        
        # Create application
        app = QApplication([])
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle("🤖 JARVIS Voice Assistant v2.0")
        window.setGeometry(100, 100, 600, 400)
        
        # Create central widget
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        
        # Create layout
        layout = QVBoxLayout(central_widget)
        
        # Add labels
        title_label = QLabel("🤖 JARVIS Voice Assistant")
        title_label.setStyleSheet("font-size: 24px; font-weight: bold; color: #00ff41; text-align: center;")
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        status_label = QLabel("✅ Core systems initialized successfully!")
        status_label.setStyleSheet("font-size: 16px; color: #00ff41; text-align: center;")
        status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        info_label = QLabel("Ready for voice processing integration")
        info_label.setStyleSheet("font-size: 12px; color: #888; text-align: center;")
        info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        layout.addWidget(title_label)
        layout.addWidget(status_label)
        layout.addWidget(info_label)
        
        print("✅ JARVIS GUI window created successfully")
        print("ℹ️  Window ready (not showing in test mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI core test failed: {e}")
        return False

def test_ai_readiness():
    """ทดสอบความพร้อมของระบบ AI"""
    print("\\n🧠 ทดสอบความพร้อมของระบบ AI...")
    
    try:
        # Test PyTorch
        import torch
        print(f"✅ PyTorch {torch.__version__} ready")
        print(f"   🎮 CUDA available: {torch.cuda.is_available()}")
        
        # Test Transformers
        import transformers
        print(f"✅ Transformers {transformers.__version__} ready")
        
        # Test Faster Whisper (import only)
        import faster_whisper
        print("✅ Faster Whisper available (dependencies pending)")
        
        return True
        
    except Exception as e:
        print(f"❌ AI readiness test failed: {e}")
        return False

def main():
    """รัน core-only test"""
    print("=" * 60)
    print("🎯 JARVIS Voice Assistant v2.0 - Core System Test")
    print("=" * 60)
    
    tests = [
        ("Core System", test_core_system),
        ("GUI Core", test_gui_core),
        ("AI Readiness", test_ai_readiness)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\\n" + "=" * 60)
    print(f"📊 Core Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 JARVIS Core systems are fully operational!")
        print("✨ Ready to integrate voice processing components.")
        print("🚀 Next step: Complete voice dependencies and run full system.")
        return 0
    else:
        print("⚠️  Some core systems failed. Check the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())