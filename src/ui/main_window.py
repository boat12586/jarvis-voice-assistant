"""
Main Window for Jarvis Voice Assistant
Implements the glassmorphic UI with action buttons and voice visualization
"""

import logging
from typing import Dict, Any, Optional
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QPushButton, QGraphicsDropShadowEffect, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QPalette, QColor, QFont, QPainter, QBrush, QLinearGradient

from .styles import JarvisStyles
from .components.voice_visualizer import VoiceVisualizer
from .components.action_button import ActionButton
from .components.status_bar import StatusBar


class MainWindow(QMainWindow):
    """Main application window with glassmorphic design"""
    
    # Signals
    button_clicked = pyqtSignal(str)
    voice_button_pressed = pyqtSignal()
    voice_button_released = pyqtSignal()
    
    def __init__(self, controller):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        self.config = controller.config.get('ui', {})
        
        # Window properties
        self.setWindowTitle("J.A.R.V.I.S - Voice Assistant")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # Initialize UI
        self._setup_window()
        self._create_widgets()
        self._setup_layout()
        self._apply_styles()
        self._connect_signals()
        
        # Animation timers
        self._setup_animations()
        
        self.logger.info("Main window initialized")
    
    def _setup_window(self):
        """Setup window properties"""
        window_config = self.config.get('window', {})
        width = window_config.get('width', 800)
        height = window_config.get('height', 600)
        
        self.setFixedSize(width, height)
        
        # Center window on screen
        screen = self.screen().availableGeometry()
        x = (screen.width() - width) // 2
        y = (screen.height() - height) // 2
        self.move(x, y)
    
    def _create_widgets(self):
        """Create all UI widgets"""
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main container with glassmorphic background
        self.main_container = QFrame()
        self.main_container.setObjectName("mainContainer")
        
        # Title label
        self.title_label = QLabel("J.A.R.V.I.S")
        self.title_label.setObjectName("titleLabel")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Voice visualizer
        self.voice_visualizer = VoiceVisualizer()
        
        # Action buttons
        self._create_action_buttons()
        
        # Status bar
        self.status_bar = StatusBar()
        
        # Close button
        self.close_button = QPushButton("✕")
        self.close_button.setObjectName("closeButton")
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)
    
    def _create_action_buttons(self):
        """Create the six main action buttons"""
        self.action_buttons = {}
        
        button_configs = [
            ("talk", "🎙️", "Talk to AI", "Start voice conversation"),
            ("news", "📰", "News", "Show latest news"),
            ("translate", "🌐", "Translate", "Translate or explain"),
            ("learn", "📚", "Learn", "Language learning"),
            ("question", "🤔", "Deep Q", "Ask deep questions"),
            ("image", "🎨", "Generate", "Create image with ComfyUI")
        ]
        
        for key, icon, text, tooltip in button_configs:
            button = ActionButton(icon, text, tooltip)
            button.clicked.connect(lambda checked, k=key: self._on_button_clicked(k))
            self.action_buttons[key] = button
    
    def _setup_layout(self):
        """Setup window layout"""
        # Main layout
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Container layout
        container_layout = QVBoxLayout(self.main_container)
        container_layout.setContentsMargins(30, 30, 30, 30)
        container_layout.setSpacing(20)
        
        # Top section (title and close button)
        top_layout = QHBoxLayout()
        top_layout.addWidget(self.title_label)
        top_layout.addStretch()
        top_layout.addWidget(self.close_button)
        
        # Voice visualizer section
        voice_layout = QHBoxLayout()
        voice_layout.addStretch()
        voice_layout.addWidget(self.voice_visualizer)
        voice_layout.addStretch()
        
        # Action buttons grid
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)
        
        # First row: talk, news, translate
        row1 = QHBoxLayout()
        row1.setSpacing(20)
        row1.addWidget(self.action_buttons["talk"])
        row1.addWidget(self.action_buttons["news"])
        row1.addWidget(self.action_buttons["translate"])
        
        # Second row: learn, question, image
        row2 = QHBoxLayout()
        row2.setSpacing(20)
        row2.addWidget(self.action_buttons["learn"])
        row2.addWidget(self.action_buttons["question"])
        row2.addWidget(self.action_buttons["image"])
        
        buttons_layout.addLayout(row1)
        buttons_layout.addLayout(row2)
        
        # Add to container
        container_layout.addLayout(top_layout)
        container_layout.addStretch()
        container_layout.addLayout(voice_layout)
        container_layout.addStretch()
        container_layout.addLayout(buttons_layout)
        container_layout.addStretch()
        container_layout.addWidget(self.status_bar)
        
        # Add container to main layout
        main_layout.addWidget(self.main_container)
    
    def _apply_styles(self):
        """Apply styles to the window"""
        self.setStyleSheet(JarvisStyles.get_main_window_style(self.config))
        
        # Apply shadow effect to main container
        shadow = QGraphicsDropShadowEffect()
        shadow.setBlurRadius(30)
        shadow.setColor(QColor(0, 212, 255, 80))
        shadow.setOffset(0, 0)
        self.main_container.setGraphicsEffect(shadow)
    
    def _connect_signals(self):
        """Connect signals between components"""
        # Connect controller signals
        self.controller.status_changed.connect(self.status_bar.update_status)
        self.controller.voice_activity_changed.connect(self.voice_visualizer.set_active)
        self.controller.response_ready.connect(self._on_response_ready)
        self.controller.error_occurred.connect(self._on_error)
        
        # Connect voice button signals
        self.action_buttons["talk"].pressed.connect(self._on_voice_button_pressed)
        self.action_buttons["talk"].released.connect(self._on_voice_button_released)
    
    def _setup_animations(self):
        """Setup animations for UI elements"""
        # Breathing animation for title
        self.title_animation = QPropertyAnimation(self.title_label, b"styleSheet")
        self.title_animation.setDuration(2000)
        self.title_animation.setLoopCount(-1)
        
        # Pulse animation timer
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse_animation)
        self.pulse_timer.start(50)  # 20 FPS
    
    def _pulse_animation(self):
        """Animate pulsing effects"""
        # This will be implemented with more sophisticated animations
        pass
    
    def _on_button_clicked(self, button_key: str):
        """Handle action button clicks"""
        self.logger.info(f"Button clicked: {button_key}")
        self.button_clicked.emit(button_key)
        
        # Execute corresponding feature
        if button_key == "talk":
            # This is handled by press/release events
            pass
        elif button_key == "news":
            self.display_message("กำลังโหลดข่าวสาร...", "system")
            result = self.controller.execute_feature("news")
            self.display_message(f"ข่าวสาร: {result}", "news")
        elif button_key == "translate":
            self.display_message("เปิดระบบแปลภาษา - กรุณาพูดข้อความที่ต้องการแปล", "system")
            result = self.controller.execute_feature("translate")
            if result:
                self.display_message(f"แปลภาษา: {result}", "translate")
            else:
                self.display_message("ระบบแปลภาษาพร้อมใช้งาน กรุณาพูดข้อความที่ต้องการแปล", "translate")
        elif button_key == "learn":
            self.display_message("เปิดระบบเรียนรู้ภาษา - เลือกบทเรียนที่ต้องการ", "system")
            result = self.controller.execute_feature("learn")
            if result:
                self.display_message(f"บทเรียน: {result}", "learn")
            else:
                self.display_message("ระบบเรียนรู้ภาษาพร้อมใช้งาน มีบทเรียนภาษาไทยและอังกฤษ", "learn")
        elif button_key == "question":
            self.display_message("พร้อมตอบคำถามลึก - กรุณาถามคำถามที่ต้องการคำตอบเชิงลึก", "system")
            result = self.controller.execute_feature("deep_question")
            if result:
                self.display_message(f"คำตอบเชิงลึก: {result}", "question")
            else:
                self.display_message("ระบบตอบคำถามเชิงลึกพร้อมใช้งาน สามารถถามคำถามปรัชญา วิทยาศาสตร์ หรือเทคโนโลยี", "question")
        elif button_key == "image":
            self.display_message("กำลังสร้างภาพ - กรุณาบอกรายละเอียดภาพที่ต้องการ", "system")
            result = self.controller.execute_feature("image_generation")
            if result:
                self.display_message(f"สร้างภาพ: {result}", "image")
            else:
                self.display_message("ระบบสร้างภาพพร้อมใช้งาน กรุณาพูดรายละเอียดภาพที่ต้องการให้สร้าง", "image")
    
    def _on_voice_button_pressed(self):
        """Handle voice button press"""
        self.logger.info("Voice button pressed")
        self.voice_button_pressed.emit()
        self.controller.start_listening()
        
        # Update button state
        self.action_buttons["talk"].set_active(True)
        self.voice_visualizer.set_listening(True)
    
    def _on_voice_button_released(self):
        """Handle voice button release"""
        self.logger.info("Voice button released")
        self.voice_button_released.emit()
        self.controller.stop_listening()
        
        # Update button state
        self.action_buttons["talk"].set_active(False)
        self.voice_visualizer.set_listening(False)
    
    def _on_response_ready(self, response: str, metadata: Dict[str, Any]):
        """Handle AI response ready"""
        self.logger.info("Response ready, updating UI")
        self.logger.info(f"JARVIS RESPONSE: {response}")
        
        # Display response in log and possibly as overlay
        self.display_message(f"JARVIS: {response}", "assistant")
        
        # Trigger TTS
        if hasattr(self.controller, 'tts_system') and self.controller.tts_system:
            self.controller.tts_system.speak(response)
            
    def display_message(self, message: str, sender: str = "system"):
        """Display message in UI"""
        self.logger.info(f"[{sender.upper()}] {message}")
        # Could add visual display here if needed
    
    def _on_error(self, error_msg: str):
        """Handle error display"""
        self.logger.error(f"Error: {error_msg}")
        self.status_bar.show_error(error_msg)
    
    def mousePressEvent(self, event):
        """Handle mouse press for window dragging"""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_start_position = event.globalPosition().toPoint()
    
    def mouseMoveEvent(self, event):
        """Handle mouse move for window dragging"""
        if hasattr(self, 'drag_start_position'):
            delta = event.globalPosition().toPoint() - self.drag_start_position
            self.move(self.pos() + delta)
            self.drag_start_position = event.globalPosition().toPoint()
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.logger.info("Main window closing")
        self.controller.shutdown()
        event.accept()