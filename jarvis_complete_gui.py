#!/usr/bin/env python3
"""
🖥️ JARVIS Complete GUI System
ระบบ GUI ครบครันที่รวมเสียง AI และอินเทอร์เฟซ
"""

import sys
import logging
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout
from PyQt6.QtCore import QTimer, pyqtSignal, QThread
from PyQt6.QtGui import QFont

try:
    from ui.holographic_interface import HolographicInterface, HologramState
    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False
    logging.warning("GUI components not available")

from jarvis_enhanced_voice import JarvisEnhancedVoice

class VoiceWorkerThread(QThread):
    """เธรดสำหรับประมวลผลเสียงแยกจาก GUI"""
    
    # Signals
    voice_recognized = pyqtSignal(str, str, float)  # text, language, confidence
    response_generated = pyqtSignal(str, str)       # text, language
    state_changed = pyqtSignal(str)                 # state
    error_occurred = pyqtSignal(str)                # error message
    
    def __init__(self, jarvis_system):
        super().__init__()
        self.jarvis = jarvis_system
        self.is_running = False
        self.command_queue = []
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """เรียกใช้เธรด"""
        self.is_running = True
        self.logger.info("🔄 Voice worker thread started")
        
        while self.is_running:
            try:
                # ประมวลผลคำสั่งในคิว
                if self.command_queue:
                    command = self.command_queue.pop(0)
                    self._process_command(command)
                
                self.msleep(100)  # หน่วงเวลา 100ms
                
            except Exception as e:
                self.logger.error(f"❌ Voice worker error: {e}")
                self.error_occurred.emit(str(e))
    
    def stop(self):
        """หยุดเธรด"""
        self.is_running = False
        self.wait()
    
    def add_command(self, command: Dict[str, Any]):
        """เพิ่มคำสั่งลงในคิว"""
        self.command_queue.append(command)
    
    def _process_command(self, command: Dict[str, Any]):
        """ประมวลผลคำสั่ง"""
        cmd_type = command.get('type')
        
        if cmd_type == 'text_message':
            text = command.get('text', '')
            language = command.get('language', 'en')
            
            self.state_changed.emit('processing')
            
            # สร้างการตอบสนอง
            response = self.jarvis.ai.get_response(text, language)
            
            # ส่งสัญญาณกลับ
            self.response_generated.emit(response, language)
            
            # พูดการตอบสนอง
            self.jarvis.tts.speak(response, language)
            
            self.state_changed.emit('idle')
            
        elif cmd_type == 'voice_recognition':
            duration = command.get('duration', 5.0)
            
            self.state_changed.emit('listening')
            
            try:
                # เริ่มการรู้จำเสียง
                self.jarvis.start_voice_conversation(duration)
                
            except Exception as e:
                self.error_occurred.emit(f"Voice recognition failed: {e}")
                self.state_changed.emit('error')

class JarvisCompleteGUI(QMainWindow):
    """อินเทอร์เฟซ GUI ครบครันของ JARVIS"""
    
    def __init__(self):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        
        # เริ่มต้นระบบเสียง
        self.jarvis = JarvisEnhancedVoice()
        if not self.jarvis.start_system():
            self.logger.error("❌ Failed to start JARVIS voice system")
            return
        
        # เริ่มต้น GUI
        if GUI_AVAILABLE:
            self.holographic_interface = HolographicInterface()
            self._setup_ui()
            self._setup_voice_integration()
        else:
            self.logger.error("❌ GUI not available")
            return
        
        self.logger.info("✅ JARVIS Complete GUI initialized")
    
    def _setup_ui(self):
        """ตั้งค่า UI"""
        self.setWindowTitle("JARVIS - Complete AI Assistant")
        self.setGeometry(100, 100, 1400, 900)
        
        # ใช้ holographic interface เป็น central widget
        self.setCentralWidget(self.holographic_interface)
        
        # เชื่อมต่อปุ่มกับฟังก์ชัน
        self.holographic_interface.voice_btn.clicked.connect(self._on_voice_button)
        self.holographic_interface.chat_btn.clicked.connect(self._on_chat_button)
        self.holographic_interface.news_btn.clicked.connect(self._on_news_button)
        self.holographic_interface.status_btn.clicked.connect(self._on_status_button)
        self.holographic_interface.settings_btn.clicked.connect(self._on_settings_button)
        self.holographic_interface.help_btn.clicked.connect(self._on_help_button)
        
        # ตั้งค่าข้อความเริ่มต้น
        self.holographic_interface.add_conversation_entry(
            "SYSTEM", "JARVIS Complete AI Assistant initialized successfully"
        )
        self.holographic_interface.add_conversation_entry(
            "JARVIS", "Good day, sir. I am ready to assist you."
        )
    
    def _setup_voice_integration(self):
        """ตั้งค่าการรวมระบบเสียง"""
        # สร้างเธรดสำหรับประมวลผลเสียง
        self.voice_worker = VoiceWorkerThread(self.jarvis)
        
        # เชื่อมต่อสัญญาณ
        self.voice_worker.voice_recognized.connect(self._on_voice_recognized)
        self.voice_worker.response_generated.connect(self._on_response_generated)
        self.voice_worker.state_changed.connect(self._on_state_changed)
        self.voice_worker.error_occurred.connect(self._on_error_occurred)
        
        # เริ่มเธรด
        self.voice_worker.start()
        
        # ตั้งค่าไทเมอร์สำหรับอัพเดทสถานะ
        self.status_timer = QTimer()
        self.status_timer.timeout.connect(self._update_system_status)
        self.status_timer.start(1000)  # อัพเดททุกวินาที
    
    def _on_voice_button(self):
        """เมื่อกดปุ่มเสียง"""
        self.logger.info("🎤 Voice button clicked")
        
        # เพิ่มคำสั่งในคิว
        command = {
            'type': 'voice_recognition',
            'duration': 5.0
        }
        self.voice_worker.add_command(command)
        
        # แสดงข้อความ
        self.holographic_interface.add_conversation_entry(
            "SYSTEM", "Listening for voice input... (5 seconds)"
        )
    
    def _on_chat_button(self):
        """เมื่อกดปุ่มแชท"""
        self.logger.info("💬 Chat button clicked")
        
        # ตัวอย่างข้อความ (ในอนาคตอาจเป็น dialog)
        test_messages = [
            "Hello JARVIS, how are you?",
            "What time is it?",
            "What can you do?",
            "สวัสดีครับ",
            "ช่วยบอกเวลาหน่อย"
        ]
        
        import random
        message = random.choice(test_messages)
        language = "th" if any(ord(c) > 127 for c in message) else "en"
        
        # เพิ่มในบันทึกการสนทนา
        self.holographic_interface.add_conversation_entry("USER", message)
        
        # ส่งคำสั่ง
        command = {
            'type': 'text_message',
            'text': message,
            'language': language
        }
        self.voice_worker.add_command(command)
    
    def _on_news_button(self):
        """เมื่อกดปุ่มข่าว"""
        self.logger.info("📰 News button clicked")
        self.holographic_interface.add_conversation_entry(
            "SYSTEM", "News system is under development"
        )
    
    def _on_status_button(self):
        """เมื่อกดปุ่มสถานะ"""
        self.logger.info("📊 Status button clicked")
        status = self.jarvis.get_status()
        
        status_text = f"System Status:\n"
        status_text += f"Active: {status['active']}\n"
        status_text += f"Interactions: {status['interactions']}\n"
        status_text += f"Voice Recognitions: {status['voice_recognitions']}\n"
        status_text += f"Uptime: {status['uptime_seconds']:.1f}s\n"
        status_text += f"Components: {status['components']}"
        
        self.holographic_interface.add_conversation_entry("SYSTEM", status_text)
    
    def _on_settings_button(self):
        """เมื่อกดปุ่มตั้งค่า"""
        self.logger.info("⚙️ Settings button clicked")
        self.holographic_interface.add_conversation_entry(
            "SYSTEM", "Settings panel is under development"
        )
    
    def _on_help_button(self):
        """เมื่อกดปุ่มช่วยเหลือ"""
        self.logger.info("❓ Help button clicked")
        
        help_text = """JARVIS Complete AI Assistant

Available Features:
🎤 Voice Recognition - Click to start voice input
💬 Chat - Send text messages
📰 News - Get latest news (coming soon)
📊 Status - Check system status
⚙️ Settings - Configure system (coming soon)
❓ Help - Show this help

Voice Commands:
- "Hello JARVIS" - Greeting
- "What time is it?" - Get current time
- "What's the date?" - Get current date
- "What can you do?" - Show capabilities
- "Thank you" - Express gratitude
- "Goodbye" - End conversation

Thai Commands:
- "สวัสดีครับ" - ทักทาย
- "ช่วยบอกเวลาหน่อย" - ถามเวลา
- "วันนี้วันที่เท่าไหร่" - ถามวันที่
- "ขอบคุณครับ" - ขอบคุณ"""
        
        self.holographic_interface.add_conversation_entry("JARVIS", help_text)
    
    def _on_voice_recognized(self, text: str, language: str, confidence: float):
        """เมื่อรู้จำเสียงได้"""
        self.holographic_interface.add_conversation_entry(
            "USER", f"{text} (confidence: {confidence:.2f})"
        )
    
    def _on_response_generated(self, text: str, language: str):
        """เมื่อสร้างการตอบสนองได้"""
        self.holographic_interface.add_conversation_entry("JARVIS", text)
    
    def _on_state_changed(self, state: str):
        """เมื่อสถานะเปลี่ยน"""
        state_map = {
            'idle': HologramState.IDLE,
            'listening': HologramState.LISTENING,
            'processing': HologramState.PROCESSING,
            'responding': HologramState.RESPONDING,
            'error': HologramState.ERROR
        }
        
        hologram_state = state_map.get(state, HologramState.IDLE)
        self.holographic_interface.set_state(hologram_state)
    
    def _on_error_occurred(self, error: str):
        """เมื่อเกิดข้อผิดพลาด"""
        self.holographic_interface.add_conversation_entry("ERROR", error)
        self.holographic_interface.set_state(HologramState.ERROR)
    
    def _update_system_status(self):
        """อัพเดทสถานะระบบ"""
        # อัพเดทการแสดงผลเวลา
        self.holographic_interface.update_status_display()
        
        # อัพเดทสถิติ
        status = self.jarvis.get_status()
        self.holographic_interface.set_progress(
            min(status['interactions'] / 10.0, 1.0)  # Progress based on interactions
        )
    
    def closeEvent(self, event):
        """เมื่อปิดแอพ"""
        self.logger.info("🛑 Closing JARVIS Complete GUI...")
        
        # หยุดเธรดเสียง
        if hasattr(self, 'voice_worker'):
            self.voice_worker.stop()
        
        # หยุดระบบเสียง
        self.jarvis.stop_system()
        
        event.accept()


def main():
    """ฟังก์ชันหลัก"""
    # ตั้งค่า logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    if not GUI_AVAILABLE:
        print("❌ GUI components not available")
        print("💡 Try running: pip install PyQt6")
        return
    
    # สร้างแอพพลิเคชัน
    app = QApplication(sys.argv)
    app.setApplicationName("JARVIS Complete AI Assistant")
    
    # ตั้งค่าฟอนต์
    font = QFont("Courier New", 10)
    app.setFont(font)
    
    # สร้างและแสดง JARVIS GUI
    try:
        jarvis_gui = JarvisCompleteGUI()
        jarvis_gui.show()
        
        print("🚀 JARVIS Complete GUI started successfully!")
        print("🎙️ Click the Voice button to start voice interaction")
        print("💬 Click the Chat button for text interaction")
        print("📊 Click the Status button to check system status")
        
        # เรียกใช้แอพ
        sys.exit(app.exec())
        
    except Exception as e:
        logging.error(f"❌ Failed to start JARVIS GUI: {e}")
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()