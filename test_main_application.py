#!/usr/bin/env python3
"""
🚀 JARVIS Main Application Test
ทดสอบการรัน main application ของ JARVIS

Version: 2.0.0 (2025 Edition)
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_main_imports():
    """ทดสอบการ import main application modules"""
    print("🔍 ทดสอบการ import main application...")
    
    try:
        # Test main entry point
        from main import main
        print("✅ main.py: OK")
        
        # Test system components
        from system.application_controller import ApplicationController
        print("✅ ApplicationController: OK")
        
        from system.config_manager import ConfigManager
        print("✅ ConfigManager: OK")
        
        from system.logger import setup_logger
        print("✅ Logger setup: OK")
        
        # Test UI components
        from ui.main_window import MainWindow
        print("✅ MainWindow: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_config_loading():
    """ทดสอบการโหลด configuration"""
    print("\\n🔧 ทดสอบการโหลด configuration...")
    
    try:
        from system.config_manager import ConfigManager
        
        # Create config manager
        config_manager = ConfigManager()
        print("✅ ConfigManager created")
        
        # Load configuration
        config = config_manager.load_config()
        print("✅ Configuration loaded")
        
        # Test config structure
        if hasattr(config, 'get'):
            system_config = config.get('system', {})
            print(f"✅ System config: {len(system_config)} settings")
        
        return True
        
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_logger_setup():
    """ทดสอบ logger setup"""
    print("\\n📝 ทดสอบ logger setup...")
    
    try:
        from system.logger import setup_logger
        
        # Setup logger
        setup_logger('INFO')
        print("✅ Logger setup completed")
        
        # Test logging
        logger = logging.getLogger(__name__)
        logger.info("Test log message")
        print("✅ Test log message written")
        
        return True
        
    except Exception as e:
        print(f"❌ Logger setup failed: {e}")
        return False

def test_application_controller():
    """ทดสอบ Application Controller"""
    print("\\n🎮 ทดสอบ Application Controller...")
    
    try:
        from system.application_controller import ApplicationController
        from system.config_manager import ConfigManager
        
        # Load config
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Create controller
        controller = ApplicationController(config)
        print("✅ ApplicationController created")
        
        return True
        
    except Exception as e:
        print(f"❌ ApplicationController test failed: {e}")
        return False

def test_main_window():
    """ทดสอบ Main Window"""
    print("\\n🖥️ ทดสอบ Main Window...")
    
    try:
        from PyQt6.QtWidgets import QApplication
        from ui.main_window import MainWindow
        from system.application_controller import ApplicationController
        from system.config_manager import ConfigManager
        
        # Create QApplication
        app = QApplication([])
        
        # Load config and create controller
        config_manager = ConfigManager()
        config = config_manager.load_config()
        controller = ApplicationController(config)
        
        # Create main window
        main_window = MainWindow(controller)
        print("✅ MainWindow created")
        
        # Test window properties
        if hasattr(main_window, 'setWindowTitle'):
            main_window.setWindowTitle("JARVIS Test")
            print("✅ Window title set")
        
        return True
        
    except Exception as e:
        print(f"❌ MainWindow test failed: {e}")
        return False

def test_startup_sequence():
    """ทดสอบ startup sequence"""
    print("\\n🚀 ทดสอบ startup sequence...")
    
    try:
        # Simulate startup without actually running the app
        from system.config_manager import ConfigManager
        from system.logger import setup_logger
        
        print("1. Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.load_config()
        print("   ✅ Configuration loaded")
        
        print("2. Setting up logger...")
        setup_logger(config.get('system', {}).get('log_level', 'INFO'))
        print("   ✅ Logger configured")
        
        print("3. Testing imports...")
        from system.application_controller import ApplicationController
        from ui.main_window import MainWindow
        print("   ✅ All modules imported")
        
        print("4. Creating components...")
        controller = ApplicationController(config)
        print("   ✅ Controller created")
        
        # Test without actually showing GUI
        print("5. Startup sequence completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Startup sequence failed: {e}")
        return False

def main():
    """รัน main application test"""
    print("=" * 60)
    print("🚀 JARVIS Voice Assistant v2.0 - Main Application Test")
    print("=" * 60)
    
    tests = [
        ("Main Imports", test_main_imports),
        ("Configuration Loading", test_config_loading),
        ("Logger Setup", test_logger_setup),
        ("Application Controller", test_application_controller),
        ("Main Window", test_main_window),
        ("Startup Sequence", test_startup_sequence)
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
        print("🎉 All tests passed! JARVIS main application is ready.")
        print("✨ You can now run: python src/main.py")
        return 0
    else:
        print("⚠️  Some tests failed. Check the issues above.")
        print("🔧 Fix the failing components before running JARVIS.")
        return 1

if __name__ == "__main__":
    sys.exit(main())