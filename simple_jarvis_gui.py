#!/usr/bin/env python3
"""
Simple JARVIS GUI - พร้อมใช้งานได้ทันที
Basic JARVIS Interface without complex dependencies
"""

import sys
import time
from pathlib import Path
from typing import Dict, Any

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QFrame, QScrollArea
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt6.QtGui import QFont

class SimpleJarvisProcessor(QThread):
    """Simple JARVIS processor without complex imports"""
    
    response_ready = pyqtSignal(str, dict)
    status_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.is_ready = False
    
    def initialize(self):
        """Initialize simple JARVIS"""
        self.status_changed.emit("กำลังเริ่มต้น JARVIS...")
        time.sleep(1)  # Simulate initialization
        self.is_ready = True
        self.status_changed.emit("JARVIS พร้อมใช้งานแล้ว! ✅")
    
    def process_command(self, text: str):
        """Process user command with simple logic"""
        if not self.is_ready:
            self.response_ready.emit("ระบบยังไม่พร้อม กรุณารอสักครู่", {})
            return
        
        self.status_changed.emit("กำลังประมวลผล...")
        
        # Simple response logic
        response = self._generate_simple_response(text)
        metadata = self._analyze_text(text)
        
        self.response_ready.emit(response, metadata)
        self.status_changed.emit("ประมวลผลเสร็จสิ้น")
    
    def _generate_simple_response(self, text: str) -> str:
        """Generate simple responses"""
        text_lower = text.lower()
        
        # Detect language
        is_thai = any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text)
        
        # Greetings
        if any(word in text_lower for word in ["สวัสดี", "hello", "hi", "ทักทาย"]):
            if is_thai:
                return """🤖 สวัสดีครับ! ผม JARVIS ผู้ช่วยอัจฉริยะของคุณ

✨ ความสามารถของผม:
• ตอบคำถามเกี่ยวกับเทคโนโลยี
• อธิบายแนวคิมทางวิทยาศาสตร์
• สนทนาภาษาไทยและอังกฤษ
• จำบริบทการสนทนา

มีอะไรให้ช่วยไหมครับ? 😊"""
            else:
                return """🤖 Hello! I'm JARVIS, your intelligent assistant

✨ My capabilities:
• Answer technology questions
• Explain scientific concepts  
• Chat in Thai and English
• Remember conversation context

How can I help you today? 😊"""
        
        # Information requests
        elif any(word in text_lower for word in ["คือ", "อะไร", "what", "explain", "อธิบาย"]):
            if "ai" in text_lower or "ปัญญาประดิษฐ์" in text:
                if is_thai:
                    return """🧠 ปัญญาประดิษฐ์ (Artificial Intelligence)

ปัญญาประดิษฐ์คือเทคโนโลยีที่ทำให้คอมพิวเตอร์สามารถ:
• เรียนรู้จากข้อมูล (Machine Learning)
• เข้าใจภาษามนุษย์ (Natural Language Processing)
• รับรู้รูปแบบ (Pattern Recognition)
• ตัดสินใจได้เหมือนมนุษย์

🚀 การใช้งานปัจจุบัน:
• ผู้ช่วยเสียงอัจฉริยะ (เช่นผมเอง!)
• รถยนต์ขับขี่อัตโนมัติ
• ระบบแนะนำสินค้า
• การแปลภาษา

สนใจเรื่องไหนเป็นพิเศษไหมครับ?"""
                else:
                    return """🧠 Artificial Intelligence (AI)

AI is technology that enables computers to:
• Learn from data (Machine Learning)
• Understand human language (NLP)
• Recognize patterns
• Make human-like decisions

🚀 Current applications:
• Intelligent voice assistants (like me!)
• Autonomous vehicles
• Recommendation systems
• Language translation

What specific aspect interests you?"""
            
            elif "machine learning" in text_lower or "แมชชีนเลิร์นนิง" in text:
                if is_thai:
                    return """🎯 Machine Learning (การเรียนรู้ของเครื่อง)

เป็นสาขาหนึ่งของ AI ที่เครื่องจักรเรียนรู้จากข้อมูลโดยไม่ต้องเขียนโปรแกรมโดยตรง

📚 ประเภทหลัก:
• Supervised Learning: เรียนรู้จากตัวอย่างที่มีคำตอบ
• Unsupervised Learning: หาแพทเทิร์นจากข้อมูลเอง
• Reinforcement Learning: เรียนรู้จากการลองผิดลองถูก

💡 ตัวอย่างการใช้งาน:
• การรู้จำใบหน้า
• ระบบแนะนำหนัง
• การแปลภาษา
• การวินิจฉัยโรค

อยากรู้เรื่องอะไรเพิ่มเติมไหมครับ?"""
                else:
                    return """🎯 Machine Learning

A subset of AI where machines learn from data without explicit programming

📚 Main types:
• Supervised Learning: Learn from labeled examples
• Unsupervised Learning: Find patterns in data
• Reinforcement Learning: Learn through trial and error

💡 Example applications:
• Facial recognition
• Movie recommendations
• Language translation
• Medical diagnosis

What would you like to know more about?"""
            
            else:
                if is_thai:
                    return f"น่าสนใจมาก! เรื่อง '{text}' คือหัวข้อที่มีความลึกซึ้ง\n\nคุณต้องการให้ผมอธิบายในมุมมองไหนเป็นพิเศษครับ? 🤔"
                else:
                    return f"Interesting! '{text}' is a deep topic\n\nWhat specific aspect would you like me to explain? 🤔"
        
        # How-to requests
        elif any(word in text_lower for word in ["ช่วย", "help", "สอน", "teach", "how"]):
            if is_thai:
                return """📚 ผมยินดีช่วยเหลือครับ!

🔧 วิธีใช้งาน JARVIS:
• พิมพ์คำถามหรือคำสั่งที่ต้องการ
• ใช้ภาษาไทยหรืออังกฤษก็ได้
• ผมจะพยายามตอบให้ดีที่สุด

💡 ตัวอย่างคำถาม:
• "AI คืออะไร"
• "Explain machine learning"
• "Python programming คืออะไร"
• "วิทยาศาสตร์ข้อมูลคือ"

ลองถามอะไรก็ได้เลยครับ! 😊"""
            else:
                return """📚 I'm happy to help!

🔧 How to use JARVIS:
• Type your questions or commands
• Use Thai or English
• I'll do my best to answer

💡 Example questions:
• "What is AI"
• "อธิบาย machine learning"
• "Python programming คืออะไร"
• "What is data science"

Feel free to ask anything! 😊"""
        
        # General conversation
        else:
            if is_thai:
                return f"""ผมได้รับข้อความ: "{text}" แล้วครับ

🎯 ที่ผมเข้าใจ:
• ภาษา: ไทย
• ความยาว: {len(text)} ตัวอักษร
• ประเภท: การสนทนาทั่วไป

มีอะไรอื่นให้ช่วยไหมครับ? ผมพร้อมตอบคำถามเรื่องเทคโนโลยี AI หรือวิทยาศาสตร์ ครับ! 🤖"""
            else:
                return f"""I received your message: "{text}"

🎯 What I understand:
• Language: English
• Length: {len(text)} characters
• Type: General conversation

Anything else I can help with? I'm ready to answer questions about technology, AI, or science! 🤖"""
    
    def _analyze_text(self, text: str) -> Dict[str, Any]:
        """Simple text analysis"""
        is_thai = any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text)
        
        # Simple intent detection
        text_lower = text.lower()
        if any(word in text_lower for word in ["สวัสดี", "hello", "hi"]):
            intent = "greeting"
        elif any(word in text_lower for word in ["คือ", "อะไร", "what", "explain"]):
            intent = "information_request"
        elif any(word in text_lower for word in ["ช่วย", "help", "สอน"]):
            intent = "help_request"
        else:
            intent = "conversation"
        
        return {
            "intent": intent,
            "language": "thai" if is_thai else "english",
            "length": len(text),
            "confidence": 0.85
        }

class ChatBubble(QFrame):
    """Simple chat bubble"""
    
    def __init__(self, text: str, is_user: bool = True, metadata: Dict = None):
        super().__init__()
        self.is_user = is_user
        self.metadata = metadata or {}
        
        # Setup layout
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Message text
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Apply styles
        if is_user:
            text_label.setStyleSheet("""
                QLabel {
                    background-color: #007acc;
                    color: white;
                    border-radius: 15px;
                    padding: 10px 15px;
                    font-size: 14px;
                }
            """)
            text_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            text_label.setStyleSheet("""
                QLabel {
                    background-color: #f0f0f0;
                    color: #333333;
                    border-radius: 15px;
                    padding: 10px 15px;
                    font-size: 14px;
                    border: 1px solid #e0e0e0;
                }
            """)
            text_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        layout.addWidget(text_label)
        
        # Add metadata for JARVIS responses
        if not is_user and metadata:
            meta_label = QLabel(f"Intent: {metadata.get('intent', 'N/A')} | Language: {metadata.get('language', 'N/A')}")
            meta_label.setStyleSheet("""
                QLabel {
                    color: #888888;
                    font-size: 11px;
                    font-style: italic;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """)
            layout.addWidget(meta_label)
        
        self.setLayout(layout)

class SimpleJarvisGUI(QMainWindow):
    """Simple JARVIS GUI"""
    
    def __init__(self):
        super().__init__()
        self.processor = SimpleJarvisProcessor()
        self.setup_ui()
        self.setup_connections()
        self.apply_styles()
        
        # Initialize JARVIS
        QTimer.singleShot(500, self.initialize_jarvis)
    
    def setup_ui(self):
        """Setup UI"""
        self.setWindowTitle("JARVIS - ผู้ช่วยอัจฉริยะ (Simple Mode)")
        self.setGeometry(100, 100, 800, 600)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Header
        header = self.create_header()
        main_layout.addWidget(header)
        
        # Chat area
        self.chat_area = self.create_chat_area()
        main_layout.addWidget(self.chat_area)
        
        # Input area
        input_area = self.create_input_area()
        main_layout.addWidget(input_area)
        
        # Status bar
        self.status_label = QLabel("กำลังเริ่มต้น...")
        self.status_label.setStyleSheet("QLabel { padding: 5px; color: #666; }")
        main_layout.addWidget(self.status_label)
    
    def create_header(self):
        """Create header"""
        header = QFrame()
        header.setFixedHeight(70)
        
        layout = QHBoxLayout()
        header.setLayout(layout)
        
        # Logo
        logo_label = QLabel("🤖")
        logo_label.setStyleSheet("font-size: 40px;")
        layout.addWidget(logo_label)
        
        # Title
        title_layout = QVBoxLayout()
        
        title_label = QLabel("JARVIS")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 24px;
                font-weight: bold;
                color: #007acc;
                margin: 0px;
            }
        """)
        
        subtitle_label = QLabel("ผู้ช่วยอัจฉริยะ - Simple Mode")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 12px;
                color: #666666;
                margin: 0px;
            }
        """)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setContentsMargins(10, 0, 0, 0)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Status
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("QLabel { color: #ff6b6b; font-size: 16px; }")
        layout.addWidget(self.status_indicator)
        
        return header
    
    def create_chat_area(self):
        """Create chat area"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()
        
        chat_container.setLayout(self.chat_layout)
        scroll_area.setWidget(chat_container)
        
        # Welcome message
        self.add_welcome_message()
        
        return scroll_area
    
    def add_welcome_message(self):
        """Add welcome message"""
        welcome_text = """🎉 ยินดีต้อนรับสู่ JARVIS Simple Mode!

ผมคือผู้ช่วยอัจฉริยะที่พร้อมช่วยเหลือคุณ:
• ตอบคำถามเกี่ยวกับเทคโนโลยีและ AI  
• อธิบายแนวคิดซับซ้อนให้เข้าใจง่าย
• สนทนาภาษาไทยและภาษาอังกฤษ
• ระบบพื้นฐานที่ทำงานได้เร็วและเสถียร

เริ่มต้นด้วยการทักทายหรือถามคำถามได้เลยครับ! 😊

ตัวอย่าง:
🗣️ "สวัสดี JARVIS"
🗣️ "AI คืออะไร"  
🗣️ "Explain machine learning"
"""
        
        bubble = ChatBubble(welcome_text, is_user=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
    
    def create_input_area(self):
        """Create input area"""
        input_frame = QFrame()
        input_frame.setFixedHeight(80)
        
        layout = QVBoxLayout()
        input_frame.setLayout(layout)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("พิมพ์ข้อความที่นี่... (Type your message here...)")
        self.input_field.setFixedHeight(35)
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.send_button = QPushButton("ส่ง / Send")
        self.send_button.setFixedHeight(30)
        self.send_button.setFixedWidth(100)
        
        clear_button = QPushButton("ล้าง / Clear")
        clear_button.setFixedHeight(30)
        clear_button.setFixedWidth(100)
        clear_button.clicked.connect(self.clear_chat)
        
        button_layout.addStretch()
        button_layout.addWidget(clear_button)
        button_layout.addWidget(self.send_button)
        
        layout.addWidget(self.input_field)
        layout.addLayout(button_layout)
        
        return input_frame
    
    def setup_connections(self):
        """Setup connections"""
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
        
        self.processor.response_ready.connect(self.on_response_ready)
        self.processor.status_changed.connect(self.on_status_changed)
    
    def apply_styles(self):
        """Apply styles"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #ffffff;
            }
            
            QFrame {
                background-color: #ffffff;
                border: none;
            }
            
            QLineEdit {
                border: 2px solid #e0e0e0;
                border-radius: 17px;
                padding: 6px 12px;
                font-size: 14px;
                background-color: #fafafa;
            }
            
            QLineEdit:focus {
                border-color: #007acc;
                background-color: #ffffff;
            }
            
            QPushButton {
                background-color: #007acc;
                color: white;
                border: none;
                border-radius: 15px;
                font-size: 12px;
                font-weight: bold;
                padding: 6px 15px;
            }
            
            QPushButton:hover {
                background-color: #005fa3;
            }
            
            QPushButton:pressed {
                background-color: #004d82;
            }
            
            QScrollArea {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                background-color: #fafafa;
            }
        """)
    
    def initialize_jarvis(self):
        """Initialize JARVIS"""
        self.processor.start()
        self.processor.initialize()
    
    def send_message(self):
        """Send message"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        # Add user message
        user_bubble = ChatBubble(text, is_user=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_bubble)
        
        # Clear input
        self.input_field.clear()
        
        # Process
        self.send_button.setEnabled(False)
        QTimer.singleShot(100, lambda: self.processor.process_command(text))
        
        # Scroll
        QTimer.singleShot(200, self.scroll_to_bottom)
    
    def on_response_ready(self, response: str, metadata: Dict):
        """Handle response"""
        jarvis_bubble = ChatBubble(response, is_user=False, metadata=metadata)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, jarvis_bubble)
        
        self.send_button.setEnabled(True)
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def on_status_changed(self, status: str):
        """Handle status change"""
        self.status_label.setText(status)
        
        if "พร้อมใช้งาน" in status:
            self.status_indicator.setStyleSheet("QLabel { color: #4CAF50; font-size: 16px; }")
    
    def clear_chat(self):
        """Clear chat"""
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ChatBubble):
                item.widget().setParent(None)
        
        self.add_welcome_message()
    
    def scroll_to_bottom(self):
        """Scroll to bottom"""
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

def main():
    """Main function"""
    app = QApplication(sys.argv)
    
    app.setApplicationName("JARVIS Simple GUI")
    app.setApplicationVersion("1.0.0")
    
    window = SimpleJarvisGUI()
    window.show()
    
    # Center window
    screen = app.primaryScreen().availableGeometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )
    
    print("🚀 JARVIS Simple GUI Started!")
    print("🚀 เปิด JARVIS Simple GUI แล้ว!")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()