#!/usr/bin/env python3
"""
🧪 JARVIS Basic Startup Test
ทดสอบการเริ่มต้นระบบ JARVIS ขั้นพื้นฐาน

Version: 2.0.0 (2025 Edition)
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_basic_imports():
    """ทดสอบการ import modules พื้นฐาน"""
    print("🔍 ทดสอบการ import modules พื้นฐาน...")
    
    try:
        # Test PyQt6
        from PyQt6.QtWidgets import QApplication
        from PyQt6.QtCore import Qt
        print("✅ PyQt6: OK")
        
        # Test Pydantic
        from pydantic import BaseModel
        print("✅ Pydantic: OK")
        
        # Test YAML
        import yaml
        print("✅ PyYAML: OK")
        
        # Test psutil
        import psutil
        print("✅ psutil: OK")
        
        # Test requests
        import requests
        print("✅ requests: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_manager():
    """ทดสอบ Configuration Manager"""
    print("\\n🔧 ทดสอบ Configuration Manager...")
    
    try:
        from system.config_manager_v2 import ConfigurationManager, JarvisConfig
        
        # Create config manager
        config_manager = ConfigurationManager()
        print("✅ ConfigurationManager created")
        
        # Test default config creation
        config = JarvisConfig()
        print("✅ Default JarvisConfig created")
        
        # Test system recommendations
        recommendations = config_manager.get_system_recommendations()
        print(f"✅ System recommendations: {len(recommendations)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Config Manager test failed: {e}")
        return False

def test_performance_monitor():
    """ทดสอบ Performance Monitor"""
    print("\\n📊 ทดสอบ Performance Monitor...")
    
    try:
        from utils.performance_monitor import PerformanceMonitor
        
        # Create monitor
        monitor = PerformanceMonitor()
        print("✅ PerformanceMonitor created")
        
        # Test system info
        system_info = monitor.get_system_info()
        print(f"✅ System info collected: {len(system_info)} items")
        print(f"   💻 Platform: {system_info.get('platform', 'Unknown')}")
        print(f"   🧠 CPU cores: {system_info.get('cpu_count', 'Unknown')}")
        print(f"   💾 RAM: {system_info.get('memory_total_gb', 'Unknown')} GB")
        
        # Test metrics collection
        metrics = monitor.collect_metrics()
        print("✅ Metrics collected successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance Monitor test failed: {e}")
        return False

def test_basic_gui():
    """ทดสอบ GUI พื้นฐาน"""
    print("\\n🖥️ ทดสอบ GUI พื้นฐาน...")
    
    try:
        from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel
        from PyQt6.QtCore import Qt
        
        # Create QApplication
        app = QApplication([])
        print("✅ QApplication created")
        
        # Create basic window
        window = QMainWindow()
        window.setWindowTitle("JARVIS Test Window")
        window.resize(400, 300)
        
        # Add label
        label = QLabel("JARVIS Voice Assistant v2.0\\nBasic GUI Test", window)
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        window.setCentralWidget(label)
        
        print("✅ Basic window created")
        print("ℹ️  GUI components ready (not showing window in test mode)")
        
        return True
        
    except Exception as e:
        print(f"❌ GUI test failed: {e}")
        return False

def test_directory_structure():
    """ทดสอบโครงสร้างไดเรกทอรี"""
    print("\\n📁 ทดสอบโครงสร้างไดเรกทอรี...")
    
    required_dirs = [
        "src",
        "config", 
        "models",
        "data",
        "logs"
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"✅ {dir_name}/")
        else:
            print(f"❌ {dir_name}/ (missing)")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"⚠️  Missing directories: {missing_dirs}")
        return False
    
    return True

def main():
    """รัน basic startup test"""
    print("=" * 60)
    print("🤖 JARVIS Voice Assistant v2.0 - Basic Startup Test")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Directory Structure", test_directory_structure), 
        ("Configuration Manager", test_config_manager),
        ("Performance Monitor", test_performance_monitor),
        ("Basic GUI", test_basic_gui)
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
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! JARVIS basic systems are working.")
        print("✨ Ready for advanced feature testing and development.")
        return 0
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        print("🔧 Fix the failing components before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())