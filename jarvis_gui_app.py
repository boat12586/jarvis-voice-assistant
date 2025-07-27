#!/usr/bin/env python3
"""
JARVIS Voice Assistant GUI Application
แอปพลิเคชัน GUI สำหรับผู้ช่วยเสียง JARVIS
"""

import sys
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QFrame, QScrollArea,
    QSplitter, QTabWidget, QProgressBar, QStatusBar, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QIcon, QPixmap, QLinearGradient

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

class JarvisProcessor(QThread):
    """Background thread for processing voice commands"""
    
    response_ready = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)
    status_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.command_parser = None
        self.thai_processor = None
        self.conversation_memory = None
        self.session_id = ""
        self.is_initialized = False
        
    def initialize(self):
        """Initialize JARVIS components"""
        try:
            self.status_changed.emit("กำลังเริ่มต้นระบบ JARVIS...")
            
            # Try to import components with fallback
            try:
                from voice.command_parser import VoiceCommandParser, ParsedCommand, CommandType, CommandPriority
                self.status_changed.emit("โหลดระบบวิเคราะห์คำสั่งเสียง...")
            except Exception as e:
                self.status_changed.emit(f"ไม่สามารถโหลดระบบ command parser: {e}")
                # Create simple fallback parser
                self.command_parser = self._create_fallback_parser()
                self.is_initialized = True
                self.status_changed.emit("JARVIS พร้อมใช้งานแล้ว (โหมดพื้นฐาน) ✅")
                return
            
            try:
                # Try to import Thai language processor with fallback
                try:
                    from features.thai_language_enhanced import ThaiLanguageProcessor
                except ImportError:
                    try:
                        import sys
                        import os
                        src_path = os.path.join(os.path.dirname(__file__), 'src')
                        if src_path not in sys.path:
                            sys.path.insert(0, src_path)
                        from features.thai_language_enhanced import ThaiLanguageProcessor
                    except ImportError:
                        ThaiLanguageProcessor = None
                
                if ThaiLanguageProcessor is not None:
                    self.status_changed.emit("โหลดระบบประมวลผลภาษาไทย...")
                else:
                    raise ImportError("ThaiLanguageProcessor not available")
                    
            except Exception as e:
                self.status_changed.emit(f"Thai language support not available: {e}")
                self.thai_processor = None
                ThaiLanguageProcessor = None
            
            try:
                from features.conversation_memory import ConversationMemorySystem
                self.status_changed.emit("โหลดระบบหน่วยความจำการสนทนา...")
            except Exception as e:
                self.status_changed.emit(f"Conversation memory not available: {e}")
                self.conversation_memory = None
            
            config = {
                "command_parser": {"enabled": True, "confidence_threshold": 0.6},
                "thai_language": {"enabled": True},
                "conversation_memory": {
                    "max_turns_per_session": 50,
                    "context_window_size": 10,
                    "memory_dir": "data/gui_conversation_memory"
                }
            }
            
            self.command_parser = VoiceCommandParser(config)
            
            if self.thai_processor is None:
                self.thai_processor = self._create_fallback_thai_processor()
            else:
                self.thai_processor = ThaiLanguageProcessor(config)
            
            if self.conversation_memory is None:
                self.conversation_memory = self._create_fallback_memory()
            else:
                self.conversation_memory = ConversationMemorySystem(config)
                # Start conversation session
                self.session_id = self.conversation_memory.start_session("gui_user", "th")
            
            self.is_initialized = True
            self.status_changed.emit("JARVIS พร้อมใช้งานแล้ว ✅")
            
        except Exception as e:
            self.error_occurred.emit(f"เริ่มต้นระบบไม่สำเร็จ: {e}")
    
    def _create_fallback_parser(self):
        """Create simple fallback parser"""
        class FallbackParser:
            def parse_command(self, text, language="auto"):
                # Simple intent detection
                text_lower = text.lower()
                
                if any(word in text_lower for word in ["สวัสดี", "hello", "hi", "ทักทาย"]):
                    intent = "greeting"
                elif any(word in text_lower for word in ["คือ", "อะไร", "what", "how", "ยังไง", "เป็นไง"]):
                    intent = "information_request"
                elif any(word in text_lower for word in ["ช่วย", "help", "สอน", "teach"]):
                    intent = "how_to_request"
                elif any(word in text_lower for word in ["ทำ", "จัด", "สร้าง", "create", "make", "do"]):
                    intent = "action_request"
                else:
                    intent = "conversation"
                
                # Detect language
                detected_lang = "th" if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else "en"
                
                class SimpleCommand:
                    def __init__(self):
                        self.original_text = text
                        self.cleaned_text = text
                        self.intent = intent
                        self.entities = {}
                        self.confidence = 0.8
                        self.language = detected_lang
                
                return SimpleCommand()
        
        return FallbackParser()
    
    def _create_fallback_thai_processor(self):
        """Create simple Thai processor fallback"""
        class FallbackThaiProcessor:
            def process_thai_text(self, text):
                return {"processed": text, "language": "th"}
            
            def enhance_for_ai_processing(self, text):
                return {"enhanced_text": text, "context": {}}
        
        return FallbackThaiProcessor()
    
    def _create_fallback_memory(self):
        """Create simple memory fallback"""
        class FallbackMemory:
            def __init__(self):
                self.session_id = "fallback_session"
                self.turns = []
            
            def start_session(self, user_id, language):
                return self.session_id
            
            def get_conversation_context(self, text, max_turns=3):
                return self.turns[-max_turns:] if self.turns else []
            
            def add_conversation_turn(self, **kwargs):
                turn_id = f"turn_{len(self.turns)}"
                self.turns.append(kwargs)
                return turn_id
            
            def end_session(self):
                pass
        
        return FallbackMemory()
    
    def process_command(self, text: str):
        """Process voice command"""
        if not self.is_initialized:
            self.error_occurred.emit("ระบบยังไม่พร้อม กรุณารอสักครู่")
            return
            
        try:
            self.status_changed.emit("กำลังประมวลผล...")
            
            # Detect language
            language = "th" if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else "en"
            
            # Process Thai if needed
            thai_context = {}
            if language == "th" and self.thai_processor:
                thai_result = self.thai_processor.process_thai_text(text)
                thai_context = self.thai_processor.enhance_for_ai_processing(text)
            
            # Parse command
            parsed = self.command_parser.parse_command(text, language)
            
            # Get conversation context
            context = self.conversation_memory.get_conversation_context(parsed.cleaned_text, max_turns=3)
            
            # Generate response based on intent
            response = self._generate_response(parsed, thai_context, context)
            
            # Add to conversation memory
            turn_id = self.conversation_memory.add_conversation_turn(
                user_input=text,
                user_language=language,
                processed_input=parsed.cleaned_text,
                intent=parsed.intent,
                entities=parsed.entities,
                assistant_response=response,
                response_language=language,
                confidence=parsed.confidence
            )
            
            # Prepare response data
            response_data = {
                "intent": parsed.intent,
                "confidence": parsed.confidence,
                "entities": parsed.entities,
                "language": language,
                "thai_context": thai_context,
                "turn_id": turn_id,
                "context_turns": len(context)
            }
            
            self.response_ready.emit(response, response_data)
            self.status_changed.emit("ประมวลผลเสร็จสิ้น")
            
        except Exception as e:
            self.error_occurred.emit(f"ประมวลผลล้มเหลว: {e}")
    
    def _generate_response(self, parsed, thai_context, context):
        """Generate appropriate response"""
        intent = parsed.intent
        text = parsed.cleaned_text
        
        if intent == "greeting":
            if parsed.language == "th":
                return "สวัสดีครับ! ผม JARVIS ผู้ช่วยอัจฉริยะของคุณ 🤖\nมีอะไรให้ช่วยไหมครับ?"
            else:
                return "Hello! I'm JARVIS, your intelligent assistant 🤖\nHow can I help you today?"
                
        elif intent == "information_request":
            if "ปัญญาประดิษฐ์" in text or "artificial intelligence" in text.lower():
                if parsed.language == "th":
                    return """ปัญญาประดิษฐ์ (Artificial Intelligence) คือ เทคโนโลยีที่ทำให้เครื่องจักรสามารถเรียนรู้ คิด และตัดสินใจได้เหมือนมนุษย์ครับ 🧠

🔹 ความสามารถหลัก:
  • การเรียนรู้จากข้อมูล (Machine Learning)
  • การประมวลผลภาษาธรรมชาติ (NLP) 
  • การรู้จำรูปแบบ (Pattern Recognition)
  • การตัดสินใจอัตโนมัติ

🔹 การใช้งานในปัจจุบัน:
  • ผู้ช่วยเสียงอัจฉริยะ (เช่น ผมเอง!)
  • รถยนต์ขับขี่อัตโนมัติ
  • ระบบแนะนำสินค้า
  • การแปลภาษา

คุณสนใจด้านไหนเป็นพิเศษไหมครับ? 😊"""
                else:
                    return """Artificial Intelligence (AI) is technology that enables machines to learn, think, and make decisions like humans 🧠

🔹 Key Capabilities:
  • Machine Learning from data
  • Natural Language Processing (NLP)
  • Pattern Recognition
  • Automated Decision Making

🔹 Current Applications:
  • Intelligent voice assistants (like me!)
  • Autonomous vehicles
  • Recommendation systems
  • Language translation

What specific aspect interests you most? 😊"""
            else:
                topic = parsed.entities.get("topics", [])
                if topic:
                    if parsed.language == "th":
                        return f"เรื่อง {topic[0]} นั้นน่าสนใจมากครับ! ให้ผมค้นหาข้อมูลเพิ่มเติมให้ได้ไหม? 🔍"
                    else:
                        return f"That's an interesting topic about {topic[0]}! Would you like me to search for more information? 🔍"
                else:
                    if parsed.language == "th":
                        return "ผมพร้อมตอบคำถามครับ! คุณต้องการทราบเรื่องอะไรเป็นพิเศษ? 🤔"
                    else:
                        return "I'm ready to answer your questions! What would you specifically like to know? 🤔"
                        
        elif intent == "how_to_request":
            if parsed.language == "th":
                return """ผมยินดีช่วยสอนครับ! 📚

🔧 วิธีใช้งาน JARVIS:
  • พิมพ์คำสั่งหรือคำถามที่ต้องการ
  • ใช้ภาษาไทยหรือภาษาอังกฤษก็ได้
  • ผมจะจำบริบทการสนทนาไว้ครับ

💡 ตัวอย่างคำสั่ง:
  • "อธิบายเรื่อง machine learning"
  • "ช่วยหาข้อมูลเกี่ยวกับ..."
  • "วันนี้เป็นอย่างไร"

มีอะไรให้ช่วยอีกไหมครับ? 😊"""
            else:
                return """I'm happy to help teach you! 📚

🔧 How to use JARVIS:
  • Type commands or questions you want
  • Use Thai or English language
  • I'll remember conversation context

💡 Example commands:
  • "Explain machine learning"
  • "Help me find information about..."
  • "How is today?"

Anything else I can help with? 😊"""
                
        elif intent == "action_request":
            actions = parsed.entities.get("actions", [])
            if actions:
                action = actions[0]
                if parsed.language == "th":
                    return f"เข้าใจแล้วครับ! ผมจะ{action}ให้คุณ ✅\n(หมายเหตุ: ในโหมด GUI นี้ ผมสามารถแสดงข้อมูลและตอบคำถามได้ครับ)"
                else:
                    return f"Understood! I'll {action} that for you ✅\n(Note: In GUI mode, I can display information and answer questions)"
            else:
                if parsed.language == "th":
                    return "ผมพร้อมช่วยเหลือครับ! บอกผมได้เลยว่าต้องการให้ทำอะไร 🚀"
                else:
                    return "I'm ready to help! Just tell me what you need me to do 🚀"
                    
        elif intent == "system_control":
            if parsed.language == "th":
                return """เข้าใจคำสั่งระบบแล้วครับ! ⚙️

📊 สถานะระบบปัจจุบัน:
  • ระบบประมวลผลภาษาไทย: ✅ ทำงาน
  • ระบบจำการสนทนา: ✅ ทำงาน  
  • ระบบวิเคราะห์เจตนา: ✅ ทำงาน
  • การเชื่อมต่อ AI: ✅ พร้อม

(หมายเหตุ: การควบคุมฮาร์ดแวร์จริงต้องเชื่อมต่อกับระบบเพิ่มเติมครับ)"""
            else:
                return """System command understood! ⚙️

📊 Current System Status:
  • Thai Language Processing: ✅ Active
  • Conversation Memory: ✅ Active
  • Intent Analysis: ✅ Active
  • AI Connection: ✅ Ready

(Note: Actual hardware control requires additional system integration)"""
                
        elif intent == "conversation":
            if parsed.language == "th":
                return "ผมพร้อมคุยกับคุณเสมอครับ! 😊 มีเรื่องอะไรที่อยากพูดคุยไหม?"
            else:
                return "I'm always ready to chat with you! 😊 Is there anything you'd like to talk about?"
                
        else:
            if parsed.language == "th":
                return f"""ผมได้รับข้อความ: "{text}" แล้วครับ 

🎯 ที่ผมเข้าใจ:
  • เจตนา: {intent}
  • ความมั่นใจ: {parsed.confidence:.0%}
  
มีอะไรให้ช่วยเพิ่มเติมไหมครับ? 🤖"""
            else:
                return f"""I received your message: "{text}"

🎯 What I understood:
  • Intent: {intent}
  • Confidence: {parsed.confidence:.0%}
  
How else can I help you? 🤖"""

class ChatBubble(QFrame):
    """Custom chat bubble widget"""
    
    def __init__(self, text: str, is_user: bool = True, metadata: Dict = None):
        super().__init__()
        self.is_user = is_user
        self.metadata = metadata or {}
        
        self.setFrameStyle(QFrame.Shape.NoFrame)
        self.setContentsMargins(10, 5, 10, 5)
        
        # Create layout
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Message text
        text_label = QLabel(text)
        text_label.setWordWrap(True)
        text_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Style based on sender
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
        
        # Add metadata if available (for JARVIS responses)
        if not is_user and metadata:
            self._add_metadata(layout)
        
        self.setLayout(layout)
    
    def _add_metadata(self, layout):
        """Add metadata information"""
        meta_frame = QFrame()
        meta_layout = QHBoxLayout()
        meta_layout.setContentsMargins(0, 5, 0, 0)
        
        # Create metadata text
        meta_parts = []
        if "intent" in self.metadata:
            meta_parts.append(f"Intent: {self.metadata['intent']}")
        if "confidence" in self.metadata:
            confidence = self.metadata['confidence']
            meta_parts.append(f"Confidence: {confidence:.0%}")
        if "language" in self.metadata:
            meta_parts.append(f"Language: {self.metadata['language']}")
        
        if meta_parts:
            meta_text = " | ".join(meta_parts)
            meta_label = QLabel(meta_text)
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
            meta_layout.addWidget(meta_label)
            meta_layout.addStretch()
        
        meta_frame.setLayout(meta_layout)
        layout.addWidget(meta_frame)

class JarvisGUI(QMainWindow):
    """Main JARVIS GUI Application"""
    
    def __init__(self):
        super().__init__()
        self.processor = JarvisProcessor()
        self.chat_area = None
        self.input_field = None
        self.send_button = None
        self.status_bar = None
        self.progress_bar = None
        
        self.setup_ui()
        self.setup_connections()
        self.apply_styles()
        
        # Initialize JARVIS in background
        QTimer.singleShot(500, self.initialize_jarvis)
    
    def setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("JARVIS - ผู้ช่วยอัจฉริยะ")
        self.setGeometry(100, 100, 1000, 700)
        
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
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Progress bar in status bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)
    
    def create_header(self):
        """Create header widget"""
        header = QFrame()
        header.setFixedHeight(80)
        
        layout = QHBoxLayout()
        header.setLayout(layout)
        
        # Logo/Icon area
        logo_label = QLabel("🤖")
        logo_label.setStyleSheet("font-size: 48px;")
        layout.addWidget(logo_label)
        
        # Title area
        title_layout = QVBoxLayout()
        
        title_label = QLabel("JARVIS")
        title_label.setStyleSheet("""
            QLabel {
                font-size: 28px;
                font-weight: bold;
                color: #007acc;
                margin: 0px;
            }
        """)
        
        subtitle_label = QLabel("ผู้ช่วยอัจฉริยะด้วย AI - Thai Language Supported")
        subtitle_label.setStyleSheet("""
            QLabel {
                font-size: 14px;
                color: #666666;
                margin: 0px;
            }
        """)
        
        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.setContentsMargins(10, 0, 0, 0)
        
        layout.addLayout(title_layout)
        layout.addStretch()
        
        # Status indicator
        status_layout = QVBoxLayout()
        
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet("QLabel { color: #ff6b6b; font-size: 20px; }")
        
        status_text = QLabel("Initializing...")
        status_text.setStyleSheet("QLabel { color: #666666; font-size: 12px; }")
        
        status_layout.addWidget(self.status_indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        status_layout.addWidget(status_text, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addLayout(status_layout)
        
        return header
    
    def create_chat_area(self):
        """Create chat area"""
        # Scroll area for chat
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Chat container
        chat_container = QWidget()
        self.chat_layout = QVBoxLayout()
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()  # Push messages to bottom
        
        chat_container.setLayout(self.chat_layout)
        scroll_area.setWidget(chat_container)
        
        # Welcome message
        self.add_welcome_message()
        
        return scroll_area
    
    def add_welcome_message(self):
        """Add welcome message"""
        welcome_text = """🎉 ยินดีต้อนรับสู่ JARVIS! 

ผมคือผู้ช่วยอัจฉริยะที่พร้อมช่วยเหลือคุณในการ:
• ตอบคำถามเกี่ยวกับเทคโนโลยีและ AI
• อธิบายแนวคิดซับซ้อนให้เข้าใจง่าย  
• สนทนาภาษาไทยและภาษาอังกฤษ
• จำบริบทการสนทนาของเรา

เริ่มต้นด้วยการทักทาย หรือถามอะไรก็ได้เลยครับ! 😊

ตัวอย่างคำสั่ง:
🗣️ "สวัสดี JARVIS"
🗣️ "ปัญญาประดิษฐ์คืออะไร"
🗣️ "อธิบายเรื่อง machine learning"
"""
        
        bubble = ChatBubble(welcome_text, is_user=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
    
    def create_input_area(self):
        """Create input area"""
        input_frame = QFrame()
        input_frame.setFixedHeight(100)
        
        layout = QVBoxLayout()
        input_frame.setLayout(layout)
        
        # Input field
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("พิมพ์ข้อความหรือคำสั่งที่นี่... (Type your message or command here...)")
        self.input_field.setFixedHeight(40)
        
        # Button layout
        button_layout = QHBoxLayout()
        
        self.send_button = QPushButton("ส่ง / Send")
        self.send_button.setFixedHeight(35)
        self.send_button.setFixedWidth(120)
        
        clear_button = QPushButton("ล้าง / Clear")
        clear_button.setFixedHeight(35)
        clear_button.setFixedWidth(120)
        clear_button.clicked.connect(self.clear_chat)
        
        button_layout.addStretch()
        button_layout.addWidget(clear_button)
        button_layout.addWidget(self.send_button)
        
        layout.addWidget(self.input_field)
        layout.addLayout(button_layout)
        
        return input_frame
    
    def setup_connections(self):
        """Setup signal connections"""
        self.send_button.clicked.connect(self.send_message)
        self.input_field.returnPressed.connect(self.send_message)
        
        # Processor connections
        self.processor.response_ready.connect(self.on_response_ready)
        self.processor.error_occurred.connect(self.on_error)
        self.processor.status_changed.connect(self.on_status_changed)
    
    def apply_styles(self):
        """Apply application styles"""
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
                border-radius: 20px;
                padding: 8px 15px;
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
                border-radius: 17px;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 20px;
            }
            
            QPushButton:hover {
                background-color: #005fa3;
            }
            
            QPushButton:pressed {
                background-color: #004d82;
            }
            
            QScrollArea {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                background-color: #fafafa;
            }
            
            QStatusBar {
                border-top: 1px solid #e0e0e0;
                background-color: #f5f5f5;
            }
        """)
    
    def initialize_jarvis(self):
        """Initialize JARVIS in background"""
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)  # Indeterminate progress
        self.processor.start()
        self.processor.initialize()
    
    def send_message(self):
        """Send user message"""
        text = self.input_field.text().strip()
        if not text:
            return
        
        # Add user message to chat
        user_bubble = ChatBubble(text, is_user=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, user_bubble)
        
        # Clear input
        self.input_field.clear()
        
        # Process command
        self.send_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, 0)
        
        # Process in background
        QTimer.singleShot(100, lambda: self.processor.process_command(text))
        
        # Scroll to bottom
        QTimer.singleShot(200, self.scroll_to_bottom)
    
    def on_response_ready(self, response: str, metadata: Dict):
        """Handle JARVIS response"""
        # Add JARVIS response to chat
        jarvis_bubble = ChatBubble(response, is_user=False, metadata=metadata)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, jarvis_bubble)
        
        # Re-enable send button
        self.send_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        # Scroll to bottom
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def on_error(self, error_message: str):
        """Handle error"""
        error_bubble = ChatBubble(f"❌ เกิดข้อผิดพลาด: {error_message}", is_user=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, error_bubble)
        
        self.send_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        
        QTimer.singleShot(100, self.scroll_to_bottom)
    
    def on_status_changed(self, status: str):
        """Handle status change"""
        self.status_bar.showMessage(status)
        
        if "พร้อมใช้งาน" in status or "Ready" in status:
            self.status_indicator.setStyleSheet("QLabel { color: #4CAF50; font-size: 20px; }")
            self.progress_bar.setVisible(False)
    
    def clear_chat(self):
        """Clear chat area"""
        # Remove all chat bubbles except welcome message
        for i in reversed(range(self.chat_layout.count())):
            item = self.chat_layout.itemAt(i)
            if item and item.widget() and isinstance(item.widget(), ChatBubble):
                item.widget().setParent(None)
        
        # Add welcome message back
        self.add_welcome_message()
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        scrollbar = self.chat_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def closeEvent(self, event):
        """Handle application close"""
        if self.processor.is_initialized and self.processor.conversation_memory:
            self.processor.conversation_memory.end_session()
        event.accept()

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("JARVIS Voice Assistant")
    app.setApplicationVersion("2.1.0")
    app.setOrganizationName("JARVIS AI")
    
    # Create and show main window
    window = JarvisGUI()
    window.show()
    
    # Center window on screen
    screen = app.primaryScreen().availableGeometry()
    window.move(
        (screen.width() - window.width()) // 2,
        (screen.height() - window.height()) // 2
    )
    
    print("🚀 JARVIS GUI Application Started!")
    print("🚀 เริ่มต้นแอปพลิเคชัน JARVIS GUI แล้ว!")
    
    sys.exit(app.exec())

if __name__ == "__main__":
    main()