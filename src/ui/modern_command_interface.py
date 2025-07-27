#!/usr/bin/env python3
"""
💻 Modern Command Interface for JARVIS
อินเทอร์เฟซคำสั่งทันสมัยสำหรับ JARVIS
"""

import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QTextEdit, QScrollArea, QFrame, QGraphicsDropShadowEffect
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient,
    QFont, QFontMetrics, QPalette, QPixmap
)


class CommandType(Enum):
    """ประเภทคำสั่ง"""
    USER_INPUT = "user_input"
    JARVIS_RESPONSE = "jarvis_response"
    SYSTEM_MESSAGE = "system_message"
    ERROR_MESSAGE = "error_message"
    SUCCESS_MESSAGE = "success_message"


@dataclass
class CommandMessage:
    """ข้อความคำสั่ง"""
    content: str
    message_type: CommandType
    timestamp: float
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None


class ModernCommandBubble(QFrame):
    """บับเบิลข้อความทันสมัย"""
    
    def __init__(self, message: CommandMessage, parent=None):
        super().__init__(parent)
        self.message = message
        self.colors = {
            CommandType.USER_INPUT: {"bg": "#1a3a5c", "text": "#ffffff", "border": "#4a9eff"},
            CommandType.JARVIS_RESPONSE: {"bg": "#0a2a1a", "text": "#00ff88", "border": "#00ff88"},
            CommandType.SYSTEM_MESSAGE: {"bg": "#2a2a2a", "text": "#cccccc", "border": "#666666"},
            CommandType.ERROR_MESSAGE: {"bg": "#3a1a1a", "text": "#ff6666", "border": "#ff3366"},
            CommandType.SUCCESS_MESSAGE: {"bg": "#1a3a1a", "text": "#66ff66", "border": "#00ff88"}
        }
        
        self.setup_ui()
        self.apply_animations()
    
    def setup_ui(self):
        """ตั้งค่า UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(15, 10, 15, 10)
        
        # Header with timestamp and type
        header_layout = QHBoxLayout()
        
        # Type indicator
        type_label = QLabel(self._get_type_icon())
        type_label.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Timestamp
        timestamp_str = time.strftime("%H:%M:%S", time.localtime(self.message.timestamp))
        timestamp_label = QLabel(timestamp_str)
        timestamp_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        
        header_layout.addWidget(type_label)
        header_layout.addStretch()
        header_layout.addWidget(timestamp_label)
        
        # Message content
        content_label = QLabel(self.message.content)
        content_label.setWordWrap(True)
        content_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        # Confidence indicator (if available)
        if self.message.confidence is not None:
            confidence_layout = QHBoxLayout()
            confidence_label = QLabel(f"Confidence: {self.message.confidence:.1%}")
            confidence_bar = self._create_confidence_bar()
            
            confidence_layout.addWidget(confidence_label)
            confidence_layout.addWidget(confidence_bar)
            layout.addLayout(confidence_layout)
        
        layout.addLayout(header_layout)
        layout.addWidget(content_label)
        
        self.setLayout(layout)
        self.apply_styling()
    
    def _get_type_icon(self) -> str:
        """ได้รับไอคอนตามประเภท"""
        icons = {
            CommandType.USER_INPUT: "👤",
            CommandType.JARVIS_RESPONSE: "🤖",
            CommandType.SYSTEM_MESSAGE: "⚙️",
            CommandType.ERROR_MESSAGE: "❌",
            CommandType.SUCCESS_MESSAGE: "✅"
        }
        return icons.get(self.message.message_type, "💬")
    
    def _create_confidence_bar(self) -> QWidget:
        """สร้างแถบความเชื่อมั่น"""
        bar = QFrame()
        bar.setFixedHeight(4)
        bar.setStyleSheet(f"""
            QFrame {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #ff3366, stop:0.5 #ffaa00, stop:1 #00ff88);
                border-radius: 2px;
            }}
        """)
        return bar
    
    def apply_styling(self):
        """ใช้สไตล์"""
        colors = self.colors[self.message.message_type]
        
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {colors["bg"]};
                border: 2px solid {colors["border"]};
                border-radius: 15px;
                margin: 5px;
            }}
            QLabel {{
                color: {colors["text"]};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 12px;
                background-color: transparent;
                border: none;
            }}
        """)
        
        # Add glow effect
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(10)
        shadow.setColor(QColor(colors["border"]))
        shadow.setOffset(0, 0)
        self.setGraphicsEffect(shadow)
    
    def apply_animations(self):
        """ใช้แอนิเมชัน"""
        # Fade in animation
        self.setWindowOpacity(0.0)
        
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(500)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_animation.start()


class ModernCommandInterface(QWidget):
    """อินเทอร์เฟซคำสั่งทันสมัย"""
    
    # Signals
    command_sent = pyqtSignal(str)
    clear_requested = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.messages: List[CommandMessage] = []
        self.max_messages = 100
        self.auto_scroll = True
        
        self.setup_ui()
        self.apply_styling()
        
        # Auto-clear timer
        self.auto_clear_timer = QTimer()
        self.auto_clear_timer.timeout.connect(self._auto_clear_old_messages)
        self.auto_clear_timer.start(60000)  # Check every minute
    
    def setup_ui(self):
        """ตั้งค่า UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Header
        header = self._create_header()
        layout.addWidget(header)
        
        # Messages area
        self.messages_area = self._create_messages_area()
        layout.addWidget(self.messages_area)
        
        # Command input area
        input_area = self._create_input_area()
        layout.addWidget(input_area)
        
        self.setLayout(layout)
    
    def _create_header(self) -> QWidget:
        """สร้างส่วนหัว"""
        header = QFrame()
        header.setFixedHeight(50)
        
        layout = QHBoxLayout()
        
        # Title
        title = QLabel("💻 JARVIS Command Interface")
        title.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        # Status
        self.status_label = QLabel("🟢 Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Controls
        controls_layout = QHBoxLayout()
        
        self.clear_button = QPushButton("🗑️ Clear")
        self.clear_button.setFixedSize(80, 30)
        self.clear_button.clicked.connect(self.clear_messages)
        
        self.auto_scroll_button = QPushButton("📜 Auto")
        self.auto_scroll_button.setFixedSize(80, 30)
        self.auto_scroll_button.setCheckable(True)
        self.auto_scroll_button.setChecked(True)
        self.auto_scroll_button.clicked.connect(self._toggle_auto_scroll)
        
        controls_layout.addWidget(self.clear_button)
        controls_layout.addWidget(self.auto_scroll_button)
        
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addLayout(controls_layout)
        
        header.setLayout(layout)
        return header
    
    def _create_messages_area(self) -> QScrollArea:
        """สร้างพื้นที่ข้อความ"""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        
        # Messages container
        self.messages_container = QWidget()
        self.messages_layout = QVBoxLayout()
        self.messages_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.messages_container.setLayout(self.messages_layout)
        
        scroll_area.setWidget(self.messages_container)
        
        return scroll_area
    
    def _create_input_area(self) -> QWidget:
        """สร้างพื้นที่ป้อนคำสั่ง"""
        input_frame = QFrame()
        input_frame.setFixedHeight(80)
        
        layout = QVBoxLayout()
        
        # Input field
        self.command_input = QTextEdit()
        self.command_input.setFixedHeight(50)
        self.command_input.setPlaceholderText("Type your command here... (Press Ctrl+Enter to send)")
        
        # Buttons
        button_layout = QHBoxLayout()
        
        self.send_button = QPushButton("📤 Send Command")
        self.send_button.setFixedHeight(25)
        self.send_button.clicked.connect(self._send_command)
        
        self.voice_button = QPushButton("🎤 Voice")
        self.voice_button.setFixedHeight(25)
        self.voice_button.setCheckable(True)
        
        button_layout.addWidget(self.send_button)
        button_layout.addWidget(self.voice_button)
        button_layout.addStretch()
        
        layout.addWidget(self.command_input)
        layout.addLayout(button_layout)
        
        input_frame.setLayout(layout)
        
        # Connect keyboard shortcut
        self.command_input.keyPressEvent = self._handle_key_press
        
        return input_frame
    
    def _handle_key_press(self, event):
        """จัดการการกดแป้นพิมพ์"""
        if event.key() == Qt.Key.Key_Return and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            self._send_command()
        else:
            QTextEdit.keyPressEvent(self.command_input, event)
    
    def _send_command(self):
        """ส่งคำสั่ง"""
        command_text = self.command_input.toPlainText().strip()
        if command_text:
            self.add_message(command_text, CommandType.USER_INPUT)
            self.command_sent.emit(command_text)
            self.command_input.clear()
    
    def _toggle_auto_scroll(self):
        """เปลี่ยนสถานะ auto scroll"""
        self.auto_scroll = self.auto_scroll_button.isChecked()
        if self.auto_scroll:
            self._scroll_to_bottom()
    
    def _scroll_to_bottom(self):
        """เลื่อนไปด้านล่าง"""
        scrollbar = self.messages_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def _auto_clear_old_messages(self):
        """ลบข้อความเก่าอัตโนมัติ"""
        if len(self.messages) > self.max_messages:
            # Remove oldest messages
            messages_to_remove = len(self.messages) - self.max_messages
            self.messages = self.messages[messages_to_remove:]
            
            # Rebuild UI
            self._rebuild_messages_ui()
    
    def _rebuild_messages_ui(self):
        """สร้าง UI ข้อความใหม่"""
        # Clear existing widgets
        for i in reversed(range(self.messages_layout.count())):
            child = self.messages_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        # Add messages back
        for message in self.messages:
            bubble = ModernCommandBubble(message)
            self.messages_layout.addWidget(bubble)
        
        if self.auto_scroll:
            QTimer.singleShot(100, self._scroll_to_bottom)
    
    def add_message(self, content: str, message_type: CommandType, confidence: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None):
        """เพิ่มข้อความ"""
        message = CommandMessage(
            content=content,
            message_type=message_type,
            timestamp=time.time(),
            confidence=confidence,
            metadata=metadata
        )
        
        self.messages.append(message)
        
        # Create and add bubble
        bubble = ModernCommandBubble(message)
        self.messages_layout.addWidget(bubble)
        
        # Auto scroll if enabled
        if self.auto_scroll:
            QTimer.singleShot(100, self._scroll_to_bottom)
        
        # Update status
        self._update_status()
    
    def add_jarvis_response(self, response: str, confidence: Optional[float] = None):
        """เพิ่มการตอบกลับจาก JARVIS"""
        self.add_message(response, CommandType.JARVIS_RESPONSE, confidence)
    
    def add_system_message(self, message: str):
        """เพิ่มข้อความระบบ"""
        self.add_message(message, CommandType.SYSTEM_MESSAGE)
    
    def add_error_message(self, error: str):
        """เพิ่มข้อความข้อผิดพลาด"""
        self.add_message(error, CommandType.ERROR_MESSAGE)
    
    def add_success_message(self, message: str):
        """เพิ่มข้อความสำเร็จ"""
        self.add_message(message, CommandType.SUCCESS_MESSAGE)
    
    def clear_messages(self):
        """ลบข้อความทั้งหมด"""
        self.messages.clear()
        
        # Clear UI
        for i in reversed(range(self.messages_layout.count())):
            child = self.messages_layout.itemAt(i).widget()
            if child:
                child.setParent(None)
        
        self.add_system_message("💭 Chat history cleared")
        self.clear_requested.emit()
    
    def _update_status(self):
        """อัปเดตสถานะ"""
        message_count = len(self.messages)
        if message_count == 0:
            self.status_label.setText("🟢 Ready")
        else:
            self.status_label.setText(f"💬 {message_count} messages")
    
    def set_voice_mode(self, enabled: bool):
        """ตั้งค่าโหมดเสียง"""
        self.voice_button.setChecked(enabled)
        if enabled:
            self.voice_button.setText("🎤 Listening...")
            self.add_system_message("🎤 Voice mode activated")
        else:
            self.voice_button.setText("🎤 Voice")
    
    def apply_styling(self):
        """ใช้สไตล์"""
        self.setStyleSheet("""
            QWidget {
                background-color: #0a0a0a;
                color: #ffffff;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            
            QFrame {
                background-color: #1a1a1a;
                border: 1px solid #333333;
                border-radius: 10px;
            }
            
            QLabel {
                color: #00d4ff;
                font-weight: bold;
                font-size: 14px;
                background-color: transparent;
                border: none;
            }
            
            QPushButton {
                background-color: rgba(0, 212, 255, 0.1);
                border: 2px solid #00d4ff;
                color: #00d4ff;
                padding: 5px 10px;
                border-radius: 5px;
                font-weight: bold;
                font-size: 11px;
            }
            
            QPushButton:hover {
                background-color: rgba(0, 212, 255, 0.2);
                border: 2px solid #4dffff;
            }
            
            QPushButton:pressed {
                background-color: rgba(0, 212, 255, 0.3);
            }
            
            QPushButton:checked {
                background-color: rgba(0, 212, 255, 0.3);
                border: 2px solid #00ff88;
                color: #00ff88;
            }
            
            QTextEdit {
                background-color: #1a1a1a;
                border: 2px solid #333333;
                color: #ffffff;
                padding: 8px;
                border-radius: 5px;
                font-size: 12px;
            }
            
            QTextEdit:focus {
                border: 2px solid #00d4ff;
            }
            
            QScrollArea {
                background-color: transparent;
                border: none;
            }
            
            QScrollBar:vertical {
                background-color: #1a1a1a;
                width: 10px;
                border-radius: 5px;
            }
            
            QScrollBar::handle:vertical {
                background-color: #00d4ff;
                border-radius: 5px;
                min-height: 20px;
            }
            
            QScrollBar::handle:vertical:hover {
                background-color: #4dffff;
            }
        """)


# Test function
def test_modern_command_interface():
    """ทดสอบ Modern Command Interface"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv if 'sys' in globals() else [])
    
    # Create main window
    window = QWidget()
    window.setWindowTitle("💻 Modern Command Interface Test")
    window.setGeometry(100, 100, 800, 600)
    
    # Create interface
    interface = ModernCommandInterface()
    
    # Test with sample messages
    interface.add_message("Hello JARVIS", CommandType.USER_INPUT)
    interface.add_jarvis_response("Hello! I'm JARVIS, your AI assistant. How can I help you today?", 0.95)
    interface.add_message("What time is it?", CommandType.USER_INPUT)
    interface.add_jarvis_response("The current time is 10:30 AM", 0.98)
    interface.add_system_message("System initialized successfully")
    interface.add_success_message("Voice recognition activated")
    interface.add_error_message("Network connection failed")
    
    layout = QVBoxLayout()
    layout.addWidget(interface)
    window.setLayout(layout)
    
    # Connect signals for testing
    def on_command_sent(command):
        print(f"Command sent: {command}")
        interface.add_jarvis_response(f"I heard: '{command}'. Processing...", 0.85)
    
    interface.command_sent.connect(on_command_sent)
    
    window.show()
    
    print("🧪 Modern Command Interface Test")
    print("Type commands and press Ctrl+Enter to send")
    
    return app, window


if __name__ == "__main__":
    app, window = test_modern_command_interface()
    sys.exit(app.exec())