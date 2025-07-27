#!/usr/bin/env python3
"""
🎙️ JARVIS Voice Assistant v2.0 - Full Voice System
รันระบบ JARVIS พร้อม Voice Recognition และ TTS

Version: 2.0.0 (2025 Edition)
"""

import sys
import os
import logging
import threading
import time
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_voice_logging():
    """ตั้งค่า logging สำหรับ voice version"""
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('logs/jarvis_voice.log')
        ]
    )

def main():
    """รัน JARVIS Voice version"""
    print("=" * 60)
    print("🎙️ JARVIS Voice Assistant v2.0 - Full Voice System")
    print("=" * 60)
    print("⚡ Starting voice-enabled systems...")
    
    try:
        # Setup logging
        setup_voice_logging()
        logger = logging.getLogger(__name__)
        logger.info("🎙️ JARVIS Voice system starting...")
        
        # Import core systems
        from system.config_manager_v2 import ConfigurationManager, JarvisConfig
        from utils.performance_monitor import PerformanceMonitor
        from voice import VoiceController
        from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QTextEdit
        from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
        from PyQt6.QtGui import QFont
        
        print("✅ All imports successful")
        logger.info("✅ All modules imported successfully")
        
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
        
        # Initialize voice controller
        print("🎙️ Initializing voice system...")
        voice_controller = VoiceController(use_fallback_tts=True)
        voice_status = voice_controller.get_status()
        
        for key, value in voice_status.items():
            print(f"   {key}: {value}")
        logger.info("✅ Voice system initialized")
        
        # Create GUI application
        print("🖥️ Creating voice-enabled GUI...")
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("JARVIS Voice Assistant")
        app.setApplicationVersion("2.0.0")
        
        # Create main window
        window = QMainWindow()
        window.setWindowTitle("🎙️ JARVIS Voice Assistant v2.0 - Voice Enabled")
        window.setGeometry(200, 200, 1000, 700)
        
        # Set Matrix-style theme
        window.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
                color: #00ff41;
            }
            QLabel {
                color: #00ff41;
                font-family: 'Courier New', monospace;
            }
            QPushButton {
                background-color: #1a1a1a;
                color: #00ff41;
                border: 2px solid #00ff41;
                padding: 12px;
                border-radius: 8px;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00ff41;
                color: #000;
                box-shadow: 0 0 15px #00ff41;
            }
            QTextEdit {
                background-color: #1a1a1a;
                color: #00ff41;
                border: 1px solid #00ff41;
                font-family: 'Courier New', monospace;
                font-size: 11px;
            }
        """)
        
        # Create central widget and layout
        central_widget = QWidget()
        window.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        
        # Title
        title = QLabel("🎙️ JARVIS VOICE ASSISTANT v2.0")
        title.setFont(QFont("Courier New", 24, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        
        # Status
        status = QLabel("✅ VOICE SYSTEMS OPERATIONAL")
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

🎙️ VOICE SYSTEMS:
✅ Speech Recognition: {voice_status['speech_recognition']}
✅ Text-to-Speech: {voice_status['text_to_speech']}
🗣️  Language: {voice_status['language']}
🎯 Wake Word: "{voice_status['wake_word']}"

🚀 READY FOR:
• Voice Command Processing
• Speech Recognition
• Text-to-Speech Responses
• Full Voice Conversation
• Real-time Audio Processing
        """
        
        info_label = QLabel(info_text)
        info_label.setFont(QFont("Courier New", 10))
        info_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(info_label)
        
        # Voice output display
        voice_output = QTextEdit()
        voice_output.setMaximumHeight(150)
        voice_output.setPlaceholderText("Voice conversation will appear here...")
        layout.addWidget(voice_output)
        
        # Voice control buttons
        button_layout = QVBoxLayout()
        
        listen_btn = QPushButton("🎤 Listen for Command (5s)")
        test_tts_btn = QPushButton("🗣️  Test Text-to-Speech")
        conversation_btn = QPushButton("💬 Start Voice Conversation")
        
        button_layout.addWidget(listen_btn)
        button_layout.addWidget(test_tts_btn)
        button_layout.addWidget(conversation_btn)
        
        layout.addLayout(button_layout)
        
        # Voice handler functions
        def listen_for_command():
            """Listen for voice command"""
            voice_output.append("🎤 Listening for command...")
            voice_output.repaint()
            
            # Run in thread to avoid blocking GUI
            def listen_thread():
                try:
                    command = voice_controller.listen_for_command(duration=5.0)
                    if command:
                        voice_output.append(f"👤 You said: {command}")
                        response = voice_controller.process_voice_command(command)
                        voice_output.append(f"🤖 JARVIS: {response}")
                        voice_controller.speak_response(response)
                    else:
                        voice_output.append("🔇 No speech detected")
                except Exception as e:
                    voice_output.append(f"❌ Error: {e}")
            
            threading.Thread(target=listen_thread, daemon=True).start()
        
        def test_tts():
            """Test text-to-speech"""
            test_message = "สวัสดีครับ! ผมคือ JARVIS ผู้ช่วยเสียงของคุณ"
            voice_output.append(f"🗣️  Testing TTS: {test_message}")
            voice_controller.speak_response(test_message)
        
        def start_conversation():
            """Start voice conversation"""
            voice_output.append("💬 Starting voice conversation...")
            voice_output.append("🗣️  Say 'หยุด' or 'stop' to end conversation")
            
            def conversation_thread():
                try:
                    voice_controller.speak_response("สวัสดีครับ! ผมคือ JARVIS พร้อมรับคำสั่งจากคุณแล้วครับ")
                    
                    conversation_active = True
                    while conversation_active:
                        voice_controller.speak_response("กรุณาพูดคำสั่งครับ")
                        command = voice_controller.listen_for_command(duration=5.0)
                        
                        if not command:
                            voice_controller.speak_response("ไม่ได้ยินครับ ลองใหม่อีกครั้งได้ครับ")
                            continue
                        
                        voice_output.append(f"👤 Command: {command}")
                        
                        if "หยุด" in command or "ปิด" in command or "stop" in command:
                            voice_controller.speak_response("ลาก่อนครับ!")
                            voice_output.append("👋 Conversation ended")
                            break
                        
                        response = voice_controller.process_voice_command(command)
                        voice_output.append(f"🤖 Response: {response}")
                        voice_controller.speak_response(response)
                        
                except Exception as e:
                    voice_output.append(f"❌ Conversation error: {e}")
            
            threading.Thread(target=conversation_thread, daemon=True).start()
        
        # Connect button handlers
        listen_btn.clicked.connect(listen_for_command)
        test_tts_btn.clicked.connect(test_tts)
        conversation_btn.clicked.connect(start_conversation)
        
        # Show window
        window.show()
        logger.info("✅ JARVIS Voice GUI launched successfully")
        
        print("\n🎉 JARVIS Voice v2.0 is running!")
        print("🎙️ Full voice system operational")
        print("📝 Check logs/jarvis_voice.log for detailed logs")
        print("\n⚡ JARVIS Voice Assistant is ready! Use the voice controls.")
        print("🎤 Click 'Listen for Command' to start voice interaction")
        print("🗣️  Click 'Test TTS' to test speech output")
        print("💬 Click 'Start Voice Conversation' for continuous chat")
        
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
        logger.info("👋 JARVIS Voice system shutdown complete")
        
        return exit_code
        
    except Exception as e:
        print(f"💥 Critical error: {e}")
        logging.error(f"Critical error in JARVIS Voice: {e}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())