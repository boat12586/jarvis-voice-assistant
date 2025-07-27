#!/usr/bin/env python3
"""
🚀 Enhanced Main Window for JARVIS
หน้าต่างหลักที่ปรับปรุงใหม่พร้อม UI ทันสมัย
"""

import logging
from typing import Dict, Any, Optional, Callable
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, 
    QFrame, QStackedWidget, QTabWidget, QSplitter, QTextEdit, QSlider
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient,
    QFont, QFontMetrics, QPalette, QPixmap
)

# Import enhanced UI components
from .enhanced_voice_visualizer import EnhancedVoiceVisualizer, VisualizerMode
from .modern_command_interface import ModernCommandInterface, CommandType
from .holographic_interface import HolographicWidget


class EnhancedMainWindow(QMainWindow):
    """หน้าต่างหลักที่ปรับปรุงใหม่สำหรับ JARVIS"""
    
    # Signals
    command_requested = pyqtSignal(str)
    voice_button_pressed = pyqtSignal()
    voice_button_released = pyqtSignal()
    mode_changed = pyqtSignal(str)
    window_closed = pyqtSignal()
    
    def __init__(self, controller=None):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.controller = controller
        
        # Window properties
        self.setWindowTitle("🤖 J.A.R.V.I.S - Enhanced AI Assistant")
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        
        # State management
        self.current_mode = "idle"
        self.voice_active = False
        
        # Setup UI
        self._setup_window()
        self._create_widgets()
        self._setup_layout()
        self._apply_styling()
        self._connect_signals()
        
        # Animations
        self._setup_animations()
        
        self.logger.info("🚀 Enhanced Main Window initialized")
    
    def _setup_window(self):
        """ตั้งค่าหน้าต่าง"""
        # Set window size
        self.setFixedSize(1200, 800)
        
        # Center window on screen
        screen = self.screen().availableGeometry()
        x = (screen.width() - 1200) // 2
        y = (screen.height() - 800) // 2
        self.move(x, y)
    
    def _create_widgets(self):
        """สร้างวิดเจ็ต"""
        # Central widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        # Main container
        self.main_container = HolographicWidget()
        
        # Title section
        self._create_title_section()
        
        # Main content area (tabbed)
        self._create_content_area()
        
        # Control panel
        self._create_control_panel()
        
        # Status bar
        self._create_status_bar()
    
    def _create_title_section(self):
        """สร้างส่วนหัวเรื่อง"""
        self.title_frame = QFrame()
        self.title_frame.setFixedHeight(80)
        
        layout = QHBoxLayout()
        
        # JARVIS logo/title
        self.title_label = QLabel("🤖 J.A.R.V.I.S")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # System status
        self.system_status = QLabel("🟢 ONLINE")
        self.system_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        # Window controls
        controls_layout = QHBoxLayout()
        
        self.minimize_button = QPushButton("➖")
        self.minimize_button.setFixedSize(30, 30)
        self.minimize_button.clicked.connect(self.showMinimized)
        
        self.close_button = QPushButton("✕")
        self.close_button.setFixedSize(30, 30)
        self.close_button.clicked.connect(self.close)
        
        controls_layout.addWidget(self.minimize_button)
        controls_layout.addWidget(self.close_button)
        
        layout.addWidget(self.title_label)
        layout.addStretch()
        layout.addWidget(self.system_status)
        layout.addStretch()
        layout.addLayout(controls_layout)
        
        self.title_frame.setLayout(layout)
    
    def _create_content_area(self):
        """สร้างพื้นที่เนื้อหาหลัก"""
        # Create tabbed interface
        self.content_tabs = QTabWidget()
        self.content_tabs.setTabPosition(QTabWidget.TabPosition.North)
        
        # Voice Interaction Tab
        self.voice_tab = self._create_voice_tab()
        self.content_tabs.addTab(self.voice_tab, "🎙️ Voice")
        
        # Command Interface Tab
        self.command_tab = self._create_command_tab()
        self.content_tabs.addTab(self.command_tab, "💻 Commands")
        
        # System Monitor Tab
        self.monitor_tab = self._create_monitor_tab()
        self.content_tabs.addTab(self.monitor_tab, "📊 Monitor")
        
        # Settings Tab
        self.settings_tab = self._create_settings_tab()
        self.content_tabs.addTab(self.settings_tab, "⚙️ Settings")
    
    def _create_voice_tab(self) -> QWidget:
        """สร้างแท็บ Voice Interaction"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Voice visualizer (main feature)
        self.voice_visualizer = EnhancedVoiceVisualizer()
        self.voice_visualizer.setMinimumHeight(400)
        
        # Voice controls
        voice_controls = self._create_voice_controls()
        
        layout.addWidget(self.voice_visualizer, 2)
        layout.addWidget(voice_controls, 1)
        
        tab.setLayout(layout)
        return tab
    
    def _create_command_tab(self) -> QWidget:
        """สร้างแท็บ Command Interface"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Command interface
        self.command_interface = ModernCommandInterface()
        layout.addWidget(self.command_interface)
        
        tab.setLayout(layout)
        return tab
    
    def _create_monitor_tab(self) -> QWidget:
        """สร้างแท็บ System Monitor"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # System metrics
        metrics_frame = QFrame()
        metrics_layout = QVBoxLayout()
        
        self.cpu_label = QLabel("💻 CPU: 0%")
        self.memory_label = QLabel("🧠 Memory: 0%")
        self.gpu_label = QLabel("🎮 GPU: 0%")
        self.response_time_label = QLabel("⚡ Response: 0ms")
        
        metrics_layout.addWidget(self.cpu_label)
        metrics_layout.addWidget(self.memory_label)
        metrics_layout.addWidget(self.gpu_label)
        metrics_layout.addWidget(self.response_time_label)
        
        metrics_frame.setLayout(metrics_layout)
        
        # Performance graph (placeholder)
        self.performance_display = QTextEdit()
        self.performance_display.setPlaceholderText("📈 Performance metrics will be displayed here...")
        self.performance_display.setReadOnly(True)
        
        layout.addWidget(metrics_frame)
        layout.addWidget(self.performance_display)
        
        tab.setLayout(layout)
        return tab
    
    def _create_settings_tab(self) -> QWidget:
        """สร้างแท็บ Settings"""
        tab = QWidget()
        layout = QVBoxLayout()
        
        # Voice settings
        voice_settings = QFrame()
        voice_layout = QVBoxLayout()
        
        voice_layout.addWidget(QLabel("🎤 Voice Settings"))
        
        # Volume slider
        volume_layout = QHBoxLayout()
        volume_layout.addWidget(QLabel("🔊 Volume:"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(70)
        volume_layout.addWidget(self.volume_slider)
        voice_layout.addLayout(volume_layout)
        
        # Sensitivity slider
        sensitivity_layout = QHBoxLayout()
        sensitivity_layout.addWidget(QLabel("🎯 Sensitivity:"))
        self.sensitivity_slider = QSlider(Qt.Orientation.Horizontal)
        self.sensitivity_slider.setRange(1, 10)
        self.sensitivity_slider.setValue(7)
        sensitivity_layout.addWidget(self.sensitivity_slider)
        voice_layout.addLayout(sensitivity_layout)
        
        voice_settings.setLayout(voice_layout)
        
        # UI settings
        ui_settings = QFrame()
        ui_layout = QVBoxLayout()
        
        ui_layout.addWidget(QLabel("🎨 UI Settings"))
        
        # Theme buttons
        theme_layout = QHBoxLayout()
        self.theme_dark = QPushButton("🌙 Dark")
        self.theme_light = QPushButton("☀️ Light")
        self.theme_auto = QPushButton("🔄 Auto")
        
        self.theme_dark.setCheckable(True)
        self.theme_light.setCheckable(True)
        self.theme_auto.setCheckable(True)
        self.theme_dark.setChecked(True)
        
        theme_layout.addWidget(self.theme_dark)
        theme_layout.addWidget(self.theme_light)
        theme_layout.addWidget(self.theme_auto)
        ui_layout.addLayout(theme_layout)
        
        ui_settings.setLayout(ui_layout)
        
        layout.addWidget(voice_settings)
        layout.addWidget(ui_settings)
        layout.addStretch()
        
        tab.setLayout(layout)
        return tab
    
    def _create_voice_controls(self) -> QWidget:
        """สร้างตัวควบคุมเสียง"""
        controls = QFrame()
        controls.setFixedHeight(120)
        
        layout = QVBoxLayout()
        
        # Main voice button
        button_layout = QHBoxLayout()
        
        self.main_voice_button = QPushButton("🎤 Push to Talk")
        self.main_voice_button.setFixedHeight(50)
        self.main_voice_button.setCheckable(True)
        
        # Voice mode selector
        mode_layout = QHBoxLayout()
        
        self.mode_listening = QPushButton("👂 Listen")
        self.mode_processing = QPushButton("🧠 Process")
        self.mode_responding = QPushButton("🗣️ Speak")
        
        for btn in [self.mode_listening, self.mode_processing, self.mode_responding]:
            btn.setCheckable(True)
            btn.setFixedHeight(30)
        
        self.mode_listening.setChecked(True)
        
        mode_layout.addWidget(self.mode_listening)
        mode_layout.addWidget(self.mode_processing)
        mode_layout.addWidget(self.mode_responding)
        
        button_layout.addWidget(self.main_voice_button)
        
        layout.addLayout(button_layout)
        layout.addLayout(mode_layout)
        
        controls.setLayout(layout)
        
        # Connect voice button
        self.main_voice_button.pressed.connect(self._on_voice_pressed)
        self.main_voice_button.released.connect(self._on_voice_released)
        
        # Connect mode buttons
        self.mode_listening.clicked.connect(lambda: self._set_visualizer_mode(VisualizerMode.LISTENING))
        self.mode_processing.clicked.connect(lambda: self._set_visualizer_mode(VisualizerMode.PROCESSING))
        self.mode_responding.clicked.connect(lambda: self._set_visualizer_mode(VisualizerMode.RESPONDING))
        
        return controls
    
    def _create_control_panel(self):
        """สร้างแผงควบคุม"""
        self.control_panel = QFrame()
        self.control_panel.setFixedHeight(60)
        
        layout = QHBoxLayout()
        
        # Quick action buttons
        self.quick_help = QPushButton("❓ Help")
        self.quick_status = QPushButton("📊 Status")
        self.quick_clear = QPushButton("🗑️ Clear")
        
        for btn in [self.quick_help, self.quick_status, self.quick_clear]:
            btn.setFixedHeight(40)
        
        # Connect quick buttons
        self.quick_help.clicked.connect(lambda: self._send_command("help"))
        self.quick_status.clicked.connect(lambda: self._send_command("status"))
        self.quick_clear.clicked.connect(self.command_interface.clear_messages)
        
        layout.addWidget(self.quick_help)
        layout.addWidget(self.quick_status)
        layout.addWidget(self.quick_clear)
        layout.addStretch()
        
        self.control_panel.setLayout(layout)
    
    def _create_status_bar(self):
        """สร้างแถบสถานะ"""
        self.status_frame = QFrame()
        self.status_frame.setFixedHeight(30)
        
        layout = QHBoxLayout()
        
        self.connection_status = QLabel("🌐 Connected")
        self.voice_status = QLabel("🎤 Ready")
        self.ai_status = QLabel("🧠 Online")
        self.time_label = QLabel("⏰ 00:00")
        
        layout.addWidget(self.connection_status)
        layout.addWidget(self.voice_status)
        layout.addWidget(self.ai_status)
        layout.addStretch()
        layout.addWidget(self.time_label)
        
        self.status_frame.setLayout(layout)
        
        # Update time regularly
        self.time_timer = QTimer()
        self.time_timer.timeout.connect(self._update_time)
        self.time_timer.start(1000)
    
    def _setup_layout(self):
        """ตั้งค่าเลย์เอาต์"""
        main_layout = QVBoxLayout(self.central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(10)
        
        # Add all components
        main_layout.addWidget(self.title_frame)
        main_layout.addWidget(self.content_tabs, 1)  # Main content gets most space
        main_layout.addWidget(self.control_panel)
        main_layout.addWidget(self.status_frame)
    
    def _apply_styling(self):
        """ใช้สไตล์"""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #0a0a0a;
                color: #ffffff;
            }
            
            QWidget {
                background-color: transparent;
                color: #ffffff;
                font-family: 'Consolas', 'Courier New', monospace;
            }
            
            QFrame {
                background-color: rgba(26, 26, 26, 0.9);
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
                padding: 8px 16px;
                border-radius: 8px;
                font-weight: bold;
                font-size: 12px;
            }
            
            QPushButton:hover {
                background-color: rgba(0, 212, 255, 0.2);
                border: 2px solid #4dffff;
            }
            
            QPushButton:pressed, QPushButton:checked {
                background-color: rgba(0, 212, 255, 0.3);
                border: 2px solid #00ff88;
                color: #00ff88;
            }
            
            QTabWidget::pane {
                border: 1px solid #333333;
                background-color: rgba(26, 26, 26, 0.9);
                border-radius: 10px;
            }
            
            QTabBar::tab {
                background-color: rgba(26, 26, 26, 0.7);
                border: 1px solid #333333;
                color: #00d4ff;
                padding: 10px 20px;
                margin: 2px;
                border-radius: 5px;
            }
            
            QTabBar::tab:selected {
                background-color: rgba(0, 212, 255, 0.2);
                border: 2px solid #00d4ff;
                color: #4dffff;
            }
            
            QTabBar::tab:hover {
                background-color: rgba(0, 212, 255, 0.1);
            }
            
            QSlider::groove:horizontal {
                border: 1px solid #333333;
                height: 8px;
                background: #1a1a1a;
                margin: 2px 0;
                border-radius: 4px;
            }
            
            QSlider::handle:horizontal {
                background: #00d4ff;
                border: 1px solid #00d4ff;
                width: 18px;
                margin: -2px 0;
                border-radius: 9px;
            }
            
            QSlider::sub-page:horizontal {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00d4ff, stop:1 #4dffff);
                border-radius: 4px;
            }
        """)
    
    def _connect_signals(self):
        """เชื่อมต่อสัญญาณ"""
        # Connect command interface
        self.command_interface.command_sent.connect(self._send_command)
        
        # Connect voice visualizer
        self.voice_visualizer.mode_changed.connect(self._on_visualizer_mode_changed)
        
        # Connect tab changes
        self.content_tabs.currentChanged.connect(self._on_tab_changed)
    
    def _setup_animations(self):
        """ตั้งค่าแอนิเมชัน"""
        # Window fade in
        self.setWindowOpacity(0.0)
        
        self.fade_animation = QPropertyAnimation(self, b"windowOpacity")
        self.fade_animation.setDuration(1000)
        self.fade_animation.setStartValue(0.0)
        self.fade_animation.setEndValue(1.0)
        self.fade_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.fade_animation.start()
    
    # Event Handlers
    def _on_voice_pressed(self):
        """เมื่อกดปุ่มเสียง"""
        self.voice_active = True
        self.voice_status.setText("🎤 Listening...")
        self.voice_visualizer.set_mode(VisualizerMode.LISTENING)
        self.voice_button_pressed.emit()
    
    def _on_voice_released(self):
        """เมื่อปล่อยปุ่มเสียง"""
        self.voice_active = False
        self.voice_status.setText("🎤 Ready")
        self.voice_visualizer.set_mode(VisualizerMode.PROCESSING)
        self.voice_button_released.emit()
    
    def _set_visualizer_mode(self, mode: VisualizerMode):
        """ตั้งค่าโหมด visualizer"""
        self.voice_visualizer.set_mode(mode)
        
        # Update mode buttons
        for btn in [self.mode_listening, self.mode_processing, self.mode_responding]:
            btn.setChecked(False)
        
        if mode == VisualizerMode.LISTENING:
            self.mode_listening.setChecked(True)
        elif mode == VisualizerMode.PROCESSING:
            self.mode_processing.setChecked(True)
        elif mode == VisualizerMode.RESPONDING:
            self.mode_responding.setChecked(True)
    
    def _send_command(self, command: str):
        """ส่งคำสั่ง"""
        self.command_interface.add_message(command, CommandType.USER_INPUT)
        self.command_requested.emit(command)
    
    def _on_visualizer_mode_changed(self, mode: str):
        """เมื่อโหมด visualizer เปลี่ยน"""
        self.current_mode = mode
        self.mode_changed.emit(mode)
    
    def _on_tab_changed(self, index: int):
        """เมื่อเปลี่ยนแท็บ"""
        tab_names = ["voice", "command", "monitor", "settings"]
        if 0 <= index < len(tab_names):
            self.mode_changed.emit(f"tab_{tab_names[index]}")
    
    def _update_time(self):
        """อัปเดตเวลา"""
        import time
        current_time = time.strftime("⏰ %H:%M:%S")
        self.time_label.setText(current_time)
    
    # Public Methods
    def add_jarvis_response(self, response: str, confidence: Optional[float] = None):
        """เพิ่มการตอบกลับจาก JARVIS"""
        self.command_interface.add_jarvis_response(response, confidence)
        self.voice_visualizer.set_mode(VisualizerMode.RESPONDING)
        
        # Auto switch back to idle after response
        QTimer.singleShot(3000, lambda: self.voice_visualizer.set_mode(VisualizerMode.IDLE))
    
    def add_system_message(self, message: str):
        """เพิ่มข้อความระบบ"""
        self.command_interface.add_system_message(message)
    
    def add_error_message(self, error: str):
        """เพิ่มข้อความข้อผิดพลาด"""
        self.command_interface.add_error_message(error)
        self.ai_status.setText("🧠 Error")
        
        # Reset status after 5 seconds
        QTimer.singleShot(5000, lambda: self.ai_status.setText("🧠 Online"))
    
    def update_system_metrics(self, cpu: float, memory: float, gpu: float, response_time: float):
        """อัปเดตเมตริกระบบ"""
        self.cpu_label.setText(f"💻 CPU: {cpu:.1f}%")
        self.memory_label.setText(f"🧠 Memory: {memory:.1f}%")
        self.gpu_label.setText(f"🎮 GPU: {gpu:.1f}%")
        self.response_time_label.setText(f"⚡ Response: {response_time:.0f}ms")
    
    def set_connection_status(self, connected: bool):
        """ตั้งค่าสถานะการเชื่อมต่อ"""
        if connected:
            self.connection_status.setText("🌐 Connected")
            self.system_status.setText("🟢 ONLINE")
        else:
            self.connection_status.setText("🔴 Disconnected")
            self.system_status.setText("🔴 OFFLINE")
    
    def closeEvent(self, event):
        """เมื่อปิดหน้าต่าง"""
        self.window_closed.emit()
        
        # Cleanup controller if available
        if self.controller and hasattr(self.controller, 'shutdown'):
            self.controller.shutdown()
        
        event.accept()


# Test function
def test_enhanced_main_window():
    """ทดสอบ Enhanced Main Window"""
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv if 'sys' in globals() else [])
    
    # Create main window
    window = EnhancedMainWindow()
    
    # Add some test messages
    window.add_system_message("🚀 JARVIS Enhanced UI initialized")
    window.add_jarvis_response("Hello! I'm JARVIS with enhanced UI. How can I assist you today?", 0.95)
    
    # Simulate system metrics
    def update_metrics():
        import random
        cpu = random.uniform(10, 80)
        memory = random.uniform(20, 90)
        gpu = random.uniform(5, 70)
        response_time = random.uniform(50, 500)
        window.update_system_metrics(cpu, memory, gpu, response_time)
    
    # Update metrics every 2 seconds
    metrics_timer = QTimer()
    metrics_timer.timeout.connect(update_metrics)
    metrics_timer.start(2000)
    
    # Connect signals for testing
    def on_command(command):
        print(f"Command: {command}")
        window.add_jarvis_response(f"I received: '{command}'. Processing with enhanced UI!", 0.88)
    
    window.command_requested.connect(on_command)
    
    window.show()
    
    print("🧪 Enhanced Main Window Test")
    print("Try different tabs and voice controls")
    
    return app, window


if __name__ == "__main__":
    app, window = test_enhanced_main_window()
    sys.exit(app.exec())