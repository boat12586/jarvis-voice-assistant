#!/usr/bin/env python3
"""
🤖 JARVIS Voice Assistant v2.0 - Core Version
รันระบบหลักของ JARVIS โดยไม่รวม voice processing (ขณะที่รอ dependencies)

Version: 2.0.0 (2025 Edition)
Author: JARVIS Development Team
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_core_logging():
    """ตั้งค่า logging สำหรับ core version"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/jarvis_core.log')
        ]
    )

def main():
    """รัน JARVIS Core version"""
    print("=" * 60)
    print("🤖 JARVIS Voice Assistant v2.0 - Core Version")
    print("=" * 60)
    print("⚡ Starting core systems...")
    
    try:
        # Setup logging
        setup_core_logging()
        logger = logging.getLogger(__name__)
        logger.info("🤖 JARVIS Core starting...")
        
        # Import core systems
        from system.config_manager_v2 import ConfigurationManager, JarvisConfig
        from utils.performance_monitor import PerformanceMonitor
        from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton
        from PyQt6.QtCore import Qt, QTimer
        from PyQt6.QtGui import QFont, QPalette, QColor
        
        print("✅ Core imports successful")
        logger.info("✅ All core modules imported successfully")
        
        # Initialize configuration
        print("🔧 Initializing configuration...")
        config_manager = ConfigurationManager()
        config = JarvisConfig()
        logger.info("✅ Configuration initialized")
        
        # Initialize performance monitoring
        print("📊 Starting performance monitoring...")
        monitor = PerformanceMonitor()
        monitor.start_monitoring()
        logger.info("✅ Performance monitoring started")
        
        # Get system recommendations
        recommendations = config_manager.get_system_recommendations()
        for key, rec in recommendations.items():
            print(f"💡 {key.upper()}: {rec}")
        
        # Create GUI application
        print("🖥️ Creating GUI application...")
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("JARVIS Voice Assistant")
        app.setApplicationVersion("2.0.0")
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle("🤖 JARVIS Voice Assistant v2.0 - Core Mode")
        window.setGeometry(200, 200, 800, 600)
        
        # Set dark theme
        window.setStyleSheet("""
            QMainWindow {
                background-color: #1a1a1a;
                color: #00ff41;
            }
            QLabel {
                color: #00ff41;
                font-family: 'Courier New', monospace;
            }
            QPushButton {
                background-color: #333;
                color: #00ff41;
                border: 1px solid #00ff41;
                padding: 10px;
                border-radius: 5px;
                font-family: 'Courier New', monospace;
            }
            QPushButton:hover {
                background-color: #00ff41;
                color: #000;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🤖 JARVIS VOICE ASSISTANT v2.0")
        title.setFont(QFont("Courier New", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Status
        status = QLabel("✅ CORE SYSTEMS OPERATIONAL")
        status.setFont(QFont("Courier New", 18))
        status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(status)
        
        # System info
        system_info = monitor.get_system_info()
        info_text = f"""
📊 SYSTEM STATUS:
💻 Platform: {system_info.get('platform', 'Unknown')}
🧠 CPU Cores: {system_info.get('cpu_count', 'Unknown')}
💾 RAM: {system_info.get('memory_total_gb', 'Unknown')} GB
🐍 Python: {system_info.get('python_version', 'Unknown')}

⚡ AI SYSTEMS:
✅ PyTorch: Ready
✅ Transformers: Ready
🔄 Voice Processing: Pending dependencies

🎯 READY FOR:
• Configuration Management
• Performance Monitoring  
• GUI Operations
• Basic AI Processing

⏳ PENDING:
• Voice Recognition (faster-whisper deps)
• Text-to-Speech (TTS deps)
• Audio Processing (audio deps)
        """
        
        info_label = QLabel(info_text)
        info_label.setFont(QFont("Courier New", 10))
        info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(info_label)
        
        # Buttons
        button_layout = QVBoxLayout()
        
        config_btn = QPushButton("🔧 Show Configuration")
        performance_btn = QPushButton("📊 Show Performance")
        test_ai_btn = QPushButton("🧠 Test AI Systems")
        
        button_layout.addWidget(config_btn)
        button_layout.addWidget(performance_btn)
        button_layout.addWidget(test_ai_btn)
        
        layout.addLayout(button_layout)
        
        # Button handlers
        def show_config():
            print("\\n🔧 JARVIS Configuration:")
            print(f"   Version: {config.version}")
            print(f"   Environment: {config.environment}")
            print(f"   Personality: {config.personality_name}")
            print(f"   Language: {config.voice.language}")
            
        def show_performance():
            current_metrics = monitor.collect_metrics()
            print("\\n📊 Performance Metrics:")
            print(f"   CPU: {current_metrics['cpu_percent']:.1f}%")
            print(f"   Memory: {current_metrics['memory']['percent']:.1f}%")
            print(f"   Uptime: {current_metrics['uptime']/60:.1f} minutes")
            
        def test_ai():
            print("\\n🧠 Testing AI Systems...")
            try:
                import torch
                import transformers
                print("✅ PyTorch version:", torch.__version__)
                print("✅ Transformers version:", transformers.__version__)
                print("✅ AI systems ready for integration")
            except Exception as e:
                print(f"❌ AI test failed: {e}")
        
        config_btn.clicked.connect(show_config)
        performance_btn.clicked.connect(show_performance)
        test_ai_btn.clicked.connect(test_ai)
        
        # Show window
        window.show()
        logger.info("✅ JARVIS GUI launched successfully")
        
        print("\\n🎉 JARVIS Core v2.0 is running!")
        print("🖥️ GUI window opened - Core systems operational")
        print("📝 Check logs/jarvis_core.log for detailed logs")
        print("🔄 Voice processing will be enabled when dependencies are complete")
        print("\\n⚡ JARVIS Core is ready! Use the GUI to explore features.")
        
        # Auto-show system status every 30 seconds
        def periodic_status():
            metrics = monitor.collect_metrics()
            logger.info(f"📊 CPU: {metrics['cpu_percent']:.1f}%, Memory: {metrics['memory']['percent']:.1f}%")
        
        timer = QTimer()
        timer.timeout.connect(periodic_status)
        timer.start(30000)  # 30 seconds
        
        # Run the application
        exit_code = app.exec()
        
        # Cleanup
        monitor.stop_monitoring()
        logger.info("👋 JARVIS Core shutdown complete")
        
        return exit_code
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        logging.error(f"Critical error in JARVIS Core: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())