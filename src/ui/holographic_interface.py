#!/usr/bin/env python3
"""
🌌 Holographic Sci-Fi Interface for JARVIS
อินเทอร์เฟซโฮโลแกรมสไตล์ไซไฟขั้นสูง
"""

import sys
import logging
import math
import time
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFrame, QGraphicsView, QGraphicsScene,
    QGraphicsEllipseItem, QGraphicsTextItem, QGraphicsLineItem,
    QProgressBar, QTextEdit, QScrollArea
)
from PyQt6.QtCore import (
    Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect,
    QThread, pyqtSignal, QPoint, QPointF, QRectF
)
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient,
    QFont, QFontMetrics, QPainterPath, QPixmap, QPalette,
    QConicalGradient, QPolygonF
)


class HologramState(Enum):
    """สถานะโฮโลแกรม"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    ERROR = "error"


@dataclass
class HologramColors:
    """สีโฮโลแกรม"""
    primary: str = "#00d4ff"      # Cyan สีหลัก
    secondary: str = "#0099cc"    # Blue เข้ม
    accent: str = "#ff6b35"       # Orange จุดเด่น
    background: str = "#0a0a0a"   # Black เข้ม
    glow: str = "#4dffff"         # Cyan แสง
    warning: str = "#ffaa00"      # Orange คำเตือน
    error: str = "#ff3366"        # Red ข้อผิดพลาด
    success: str = "#00ff88"      # Green สำเร็จ


class HolographicWidget(QWidget):
    """วิดเจ็ตโฮโลแกรมพื้นฐาน"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = HologramColors()
        self.glow_intensity = 0.5
        self.flicker_timer = QTimer()
        self.flicker_timer.timeout.connect(self._update_flicker)
        self.flicker_timer.start(100)  # 10 FPS flicker
        
        self.setStyleSheet(self._get_hologram_stylesheet())
    
    def _get_hologram_stylesheet(self) -> str:
        """สไตล์ชีตโฮโลแกรม"""
        return f"""
        QWidget {{
            background-color: transparent;
            color: {self.colors.primary};
            font-family: 'Courier New', 'Monaco', monospace;
            font-weight: bold;
        }}
        
        QLabel {{
            color: {self.colors.primary};
            background-color: transparent;
            border: none;
        }}
        
        QPushButton {{
            background-color: rgba(0, 212, 255, 0.1);
            border: 2px solid {self.colors.primary};
            color: {self.colors.primary};
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
        }}
        
        QPushButton:hover {{
            background-color: rgba(0, 212, 255, 0.2);
            border: 2px solid {self.colors.glow};
            box-shadow: 0 0 10px {self.colors.glow};
        }}
        
        QPushButton:pressed {{
            background-color: rgba(0, 212, 255, 0.3);
        }}
        
        QTextEdit {{
            background-color: rgba(10, 10, 10, 0.8);
            border: 1px solid {self.colors.primary};
            color: {self.colors.primary};
            selection-background-color: rgba(0, 212, 255, 0.3);
        }}
        
        QProgressBar {{
            border: 2px solid {self.colors.primary};
            border-radius: 5px;
            background-color: rgba(10, 10, 10, 0.6);
        }}
        
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {self.colors.primary}, stop:1 {self.colors.glow});
            border-radius: 3px;
        }}
        """
    
    def _update_flicker(self):
        """อัพเดทเอฟเฟกต์กะพริบ"""
        import random
        if random.random() < 0.05:  # 5% chance to flicker
            self.glow_intensity = random.uniform(0.3, 0.8)
            self.update()


class HologramMatrix(QWidget):
    """เมทริกซ์โฮโลแกรมแบบ Matrix"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = HologramColors()
        self.chars = "01アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン"
        self.columns = []
        self.timer = QTimer()
        self.timer.timeout.connect(self.update)
        self.timer.start(50)  # 20 FPS
        
        # Initialize matrix columns
        self._init_matrix()
    
    def _init_matrix(self):
        """เริ่มต้นคอลัมน์เมทริกซ์"""
        font_size = 12
        self.column_width = font_size
        self.num_columns = self.width() // self.column_width if self.width() > 0 else 50
        self.num_rows = self.height() // font_size if self.height() > 0 else 30
        
        self.columns = []
        for i in range(self.num_columns):
            column = {
                'chars': [],
                'speeds': [],
                'positions': [],
                'brightness': []
            }
            
            # Random number of characters per column
            num_chars = min(15, max(5, self.num_rows // 3))
            for j in range(num_chars):
                import random
                column['chars'].append(random.choice(self.chars))
                column['speeds'].append(random.uniform(0.5, 2.0))
                column['positions'].append(random.uniform(-num_chars, 0))
                column['brightness'].append(random.uniform(0.3, 1.0))
            
            self.columns.append(column)
    
    def paintEvent(self, event):
        """วาดเมทริกซ์โฮโลแกรม"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        if not self.columns:
            self._init_matrix()
        
        font = QFont('Courier New', 10, QFont.Weight.Bold)
        painter.setFont(font)
        
        for col_idx, column in enumerate(self.columns):
            x = col_idx * self.column_width
            
            for char_idx in range(len(column['chars'])):
                char = column['chars'][char_idx]
                y = int(column['positions'][char_idx] * 15)  # 15 pixels per row
                brightness = column['brightness'][char_idx]
                
                # Skip if outside visible area
                if y < -20 or y > self.height() + 20:
                    continue
                
                # Calculate color based on position and brightness
                if char_idx == 0:  # Head of the trail (brightest)
                    color = QColor(self.colors.glow)
                    color.setAlphaF(brightness)
                else:
                    # Fade effect for trail
                    fade = max(0, 1.0 - (char_idx * 0.2))
                    color = QColor(self.colors.primary)
                    color.setAlphaF(brightness * fade)
                
                painter.setPen(QPen(color))
                painter.drawText(x, y, char)
                
                # Update position
                column['positions'][char_idx] += column['speeds'][char_idx]
                
                # Reset if character has moved off screen
                if column['positions'][char_idx] > self.num_rows + 5:
                    column['positions'][char_idx] = -5
                    import random
                    column['chars'][char_idx] = random.choice(self.chars)
                    column['brightness'][char_idx] = random.uniform(0.3, 1.0)
    
    def resizeEvent(self, event):
        """เมื่อขนาดหน้าต่างเปลี่ยน"""
        super().resizeEvent(event)
        self._init_matrix()


class CircularHUD(QWidget):
    """HUD วงกลมสไตล์ JARVIS"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.colors = HologramColors()
        self.state = HologramState.IDLE
        self.progress = 0.0
        self.rotation_angle = 0
        self.pulse_phase = 0
        
        # Animation timer
        self.timer = QTimer()
        self.timer.timeout.connect(self._update_animations)
        self.timer.start(16)  # ~60 FPS
        
        self.setMinimumSize(300, 300)
    
    def set_state(self, state: HologramState):
        """เปลี่ยนสถานะ HUD"""
        self.state = state
        self.update()
    
    def set_progress(self, progress: float):
        """กำหนดความคืบหน้า (0.0 - 1.0)"""
        self.progress = max(0.0, min(1.0, progress))
        self.update()
    
    def _update_animations(self):
        """อัพเดทแอนิเมชัน"""
        self.rotation_angle = (self.rotation_angle + 1) % 360
        self.pulse_phase = (self.pulse_phase + 0.1) % (2 * math.pi)
        self.update()
    
    def paintEvent(self, event):
        """วาด HUD วงกลม"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        center = QPointF(self.width() / 2, self.height() / 2)
        radius = min(self.width(), self.height()) / 2 - 20
        
        # Background circle
        self._draw_background_circle(painter, center, radius)
        
        # State-specific elements
        if self.state == HologramState.IDLE:
            self._draw_idle_state(painter, center, radius)
        elif self.state == HologramState.LISTENING:
            self._draw_listening_state(painter, center, radius)
        elif self.state == HologramState.PROCESSING:
            self._draw_processing_state(painter, center, radius)
        elif self.state == HologramState.RESPONDING:
            self._draw_responding_state(painter, center, radius)
        elif self.state == HologramState.ERROR:
            self._draw_error_state(painter, center, radius)
        
        # Progress ring
        if self.progress > 0:
            self._draw_progress_ring(painter, center, radius)
        
        # Center text
        self._draw_center_text(painter, center)
    
    def _draw_background_circle(self, painter: QPainter, center: QPointF, radius: float):
        """วาดวงกลมพื้นหลัง"""
        pen = QPen(QColor(self.colors.primary))
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 0, 0, 100)))
        painter.drawEllipse(center, radius, radius)
    
    def _draw_idle_state(self, painter: QPainter, center: QPointF, radius: float):
        """วาดสถานะรอ"""
        # Slow breathing effect
        pulse = 0.8 + 0.2 * math.sin(self.pulse_phase * 0.5)
        
        pen = QPen(QColor(self.colors.primary))
        pen.setWidth(3)
        painter.setPen(pen)
        
        # Inner circles
        for i in range(3):
            circle_radius = radius * 0.3 * (1 + i * 0.2) * pulse
            painter.drawEllipse(center, circle_radius, circle_radius)
    
    def _draw_listening_state(self, painter: QPainter, center: QPointF, radius: float):
        """วาดสถานะฟัง"""
        # Sound wave effect
        pen = QPen(QColor(self.colors.success))
        pen.setWidth(4)
        painter.setPen(pen)
        
        for i in range(5):
            wave_radius = radius * (0.2 + i * 0.15)
            wave_opacity = 1.0 - (i * 0.2)
            wave_phase = self.pulse_phase + i * 0.5
            
            color = QColor(self.colors.success)
            color.setAlphaF(wave_opacity * (0.5 + 0.5 * math.sin(wave_phase)))
            pen.setColor(color)
            painter.setPen(pen)
            
            painter.drawEllipse(center, wave_radius, wave_radius)
    
    def _draw_processing_state(self, painter: QPainter, center: QPointF, radius: float):
        """วาดสถานะประมวลผล"""
        # Rotating segments
        pen = QPen(QColor(self.colors.accent))
        pen.setWidth(6)
        painter.setPen(pen)
        
        segment_count = 8
        segment_angle = 360 / segment_count
        
        for i in range(segment_count):
            angle = self.rotation_angle + i * segment_angle
            opacity = 0.3 + 0.7 * ((i + 1) / segment_count)
            
            color = QColor(self.colors.accent)
            color.setAlphaF(opacity)
            pen.setColor(color)
            painter.setPen(pen)
            
            start_angle = int(angle * 16)  # QPainter uses 1/16th degree units
            span_angle = int(segment_angle * 0.7 * 16)
            
            rect = QRectF(center.x() - radius * 0.8, center.y() - radius * 0.8,
                         radius * 1.6, radius * 1.6)
            painter.drawArc(rect, start_angle, span_angle)
    
    def _draw_responding_state(self, painter: QPainter, center: QPointF, radius: float):
        """วาดสถานะตอบ"""
        # Pulsing glow effect
        pulse = 0.5 + 0.5 * math.sin(self.pulse_phase * 2)
        
        # Outer glow
        glow_gradient = QRadialGradient(center, radius * 1.2)
        glow_color = QColor(self.colors.glow)
        glow_color.setAlphaF(0.3 * pulse)
        glow_gradient.setColorAt(0, glow_color)
        glow_gradient.setColorAt(1, QColor(0, 0, 0, 0))
        
        painter.setBrush(QBrush(glow_gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, radius * 1.2, radius * 1.2)
        
        # Main ring
        pen = QPen(QColor(self.colors.primary))
        pen.setWidth(4)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(center, radius * 0.9, radius * 0.9)
    
    def _draw_error_state(self, painter: QPainter, center: QPointF, radius: float):
        """วาดสถานะข้อผิดพลาด"""
        # Flashing red effect
        flash = 0.5 + 0.5 * math.sin(self.pulse_phase * 4)
        
        pen = QPen(QColor(self.colors.error))
        pen.setWidth(5)
        painter.setPen(pen)
        
        # Warning ring
        color = QColor(self.colors.error)
        color.setAlphaF(flash)
        pen.setColor(color)
        painter.setPen(pen)
        painter.drawEllipse(center, radius * 0.8, radius * 0.8)
        
        # X mark
        pen.setColor(QColor(self.colors.error))
        painter.setPen(pen)
        cross_size = radius * 0.3
        painter.drawLine(center.x() - cross_size, center.y() - cross_size,
                        center.x() + cross_size, center.y() + cross_size)
        painter.drawLine(center.x() - cross_size, center.y() + cross_size,
                        center.x() + cross_size, center.y() - cross_size)
    
    def _draw_progress_ring(self, painter: QPainter, center: QPointF, radius: float):
        """วาดวงแหวนความคืบหน้า"""
        pen = QPen(QColor(self.colors.accent))
        pen.setWidth(8)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        
        rect = QRectF(center.x() - radius * 1.1, center.y() - radius * 1.1,
                     radius * 2.2, radius * 2.2)
        
        start_angle = 90 * 16  # Start from top (90 degrees)
        span_angle = int(-360 * 16 * self.progress)  # Negative for clockwise
        
        painter.drawArc(rect, start_angle, span_angle)
    
    def _draw_center_text(self, painter: QPainter, center: QPointF):
        """วาดข้อความตรงกลาง"""
        font = QFont('Arial', 12, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(self.colors.primary)))
        
        state_text = {
            HologramState.IDLE: "JARVIS",
            HologramState.LISTENING: "LISTENING",
            HologramState.PROCESSING: "PROCESSING",
            HologramState.RESPONDING: "RESPONDING",
            HologramState.ERROR: "ERROR"
        }
        
        text = state_text.get(self.state, "JARVIS")
        
        # Calculate text position
        fm = QFontMetrics(font)
        text_width = fm.horizontalAdvance(text)
        text_height = fm.height()
        
        text_x = center.x() - text_width / 2
        text_y = center.y() + text_height / 4
        
        painter.drawText(int(text_x), int(text_y), text)


class HolographicInterface(QMainWindow):
    """อินเทอร์เฟซโฮโลแกรมหลัก"""
    
    # Signals
    state_changed = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.colors = HologramColors()
        self.current_state = HologramState.IDLE
        
        self._setup_ui()
        self._setup_styling()
        
        # Demo timer
        self.demo_timer = QTimer()
        self.demo_timer.timeout.connect(self._demo_state_cycle)
        
    def _setup_ui(self):
        """ตั้งค่า UI"""
        self.setWindowTitle("JARVIS - Holographic Interface")
        self.setMinimumSize(1000, 700)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Matrix background
        self.matrix_widget = HologramMatrix()
        self.matrix_widget.setFixedWidth(200)
        main_layout.addWidget(self.matrix_widget)
        
        # Center panel - Main HUD
        center_layout = QVBoxLayout()
        
        # Status label
        self.status_label = QLabel("JARVIS ONLINE")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {self.colors.primary};
            margin: 20px;
        """)
        center_layout.addWidget(self.status_label)
        
        # Circular HUD
        self.hud = CircularHUD()
        center_layout.addWidget(self.hud, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        center_layout.addWidget(self.progress_bar)
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.listen_btn = QPushButton("🎤 LISTEN")
        self.listen_btn.clicked.connect(lambda: self.set_state(HologramState.LISTENING))
        button_layout.addWidget(self.listen_btn)
        
        self.process_btn = QPushButton("⚙️ PROCESS")
        self.process_btn.clicked.connect(lambda: self.set_state(HologramState.PROCESSING))
        button_layout.addWidget(self.process_btn)
        
        self.respond_btn = QPushButton("🗣️ RESPOND")
        self.respond_btn.clicked.connect(lambda: self.set_state(HologramState.RESPONDING))
        button_layout.addWidget(self.respond_btn)
        
        self.demo_btn = QPushButton("🎭 DEMO")
        self.demo_btn.clicked.connect(self._start_demo)
        button_layout.addWidget(self.demo_btn)
        
        center_layout.addLayout(button_layout)
        
        main_layout.addLayout(center_layout)
        
        # Right panel - Status information
        right_panel = QWidget()
        right_panel.setFixedWidth(300)
        right_layout = QVBoxLayout(right_panel)
        
        # System status
        status_title = QLabel("SYSTEM STATUS")
        status_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {self.colors.accent};")
        right_layout.addWidget(status_title)
        
        self.status_text = QTextEdit()
        self.status_text.setMaximumHeight(200)
        self.status_text.setReadOnly(True)
        right_layout.addWidget(self.status_text)
        
        # Conversation log
        conv_title = QLabel("CONVERSATION LOG")
        conv_title.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {self.colors.accent};")
        right_layout.addWidget(conv_title)
        
        self.conversation_log = QTextEdit()
        self.conversation_log.setReadOnly(True)
        right_layout.addWidget(self.conversation_log)
        
        main_layout.addWidget(right_panel)
        
        # Initialize status
        self.update_status_display()
    
    def _setup_styling(self):
        """ตั้งค่าสไตล์"""
        self.setStyleSheet(f"""
        QMainWindow {{
            background-color: {self.colors.background};
            color: {self.colors.primary};
        }}
        
        QWidget {{
            background-color: transparent;
            color: {self.colors.primary};
            font-family: 'Courier New', monospace;
        }}
        
        QPushButton {{
            background-color: rgba(0, 212, 255, 0.1);
            border: 2px solid {self.colors.primary};
            color: {self.colors.primary};
            padding: 12px 20px;
            border-radius: 8px;
            font-weight: bold;
            font-size: 14px;
        }}
        
        QPushButton:hover {{
            background-color: rgba(0, 212, 255, 0.2);
            border: 2px solid {self.colors.glow};
        }}
        
        QPushButton:pressed {{
            background-color: rgba(0, 212, 255, 0.3);
        }}
        
        QTextEdit {{
            background-color: rgba(10, 10, 10, 0.8);
            border: 1px solid {self.colors.primary};
            color: {self.colors.primary};
            font-family: 'Courier New', monospace;
            font-size: 12px;
            padding: 8px;
        }}
        
        QProgressBar {{
            border: 2px solid {self.colors.primary};
            border-radius: 8px;
            background-color: rgba(10, 10, 10, 0.6);
            height: 20px;
        }}
        
        QProgressBar::chunk {{
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 {self.colors.primary}, stop:1 {self.colors.glow});
            border-radius: 6px;
        }}
        """)
    
    def set_state(self, state: HologramState):
        """เปลี่ยนสถานะระบบ"""
        self.current_state = state
        self.hud.set_state(state)
        
        # Update status label
        state_texts = {
            HologramState.IDLE: "JARVIS ONLINE",
            HologramState.LISTENING: "LISTENING FOR COMMANDS",
            HologramState.PROCESSING: "PROCESSING REQUEST",
            HologramState.RESPONDING: "GENERATING RESPONSE",
            HologramState.ERROR: "SYSTEM ERROR"
        }
        
        self.status_label.setText(state_texts.get(state, "JARVIS ONLINE"))
        
        # Update colors based on state
        state_colors = {
            HologramState.IDLE: self.colors.primary,
            HologramState.LISTENING: self.colors.success,
            HologramState.PROCESSING: self.colors.accent,
            HologramState.RESPONDING: self.colors.glow,
            HologramState.ERROR: self.colors.error
        }
        
        color = state_colors.get(state, self.colors.primary)
        self.status_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: bold;
            color: {color};
            margin: 20px;
        """)
        
        self.update_status_display()
        self.state_changed.emit(state.value)
    
    def set_progress(self, progress: float):
        """กำหนดความคืบหน้า"""
        self.hud.set_progress(progress)
        self.progress_bar.setValue(int(progress * 100))
        
        if progress > 0:
            self.progress_bar.setVisible(True)
        else:
            self.progress_bar.setVisible(False)
    
    def add_conversation_entry(self, speaker: str, text: str):
        """เพิ่มรายการในบันทึกการสนทนา"""
        timestamp = time.strftime("%H:%M:%S")
        color = self.colors.accent if speaker == "USER" else self.colors.primary
        
        entry = f'<span style="color: {color};">[{timestamp}] {speaker}:</span> {text}<br>'
        self.conversation_log.append(entry)
    
    def update_status_display(self):
        """อัพเดทการแสดงสถานะ"""
        status_info = f"""
STATE: {self.current_state.value.upper()}
TIME: {time.strftime("%Y-%m-%d %H:%M:%S")}
MODE: HOLOGRAPHIC INTERFACE
VERSION: 2.0.1
        """.strip()
        
        self.status_text.setText(status_info)
    
    def _start_demo(self):
        """เริ่มโหมดสาธิต"""
        if self.demo_timer.isActive():
            self.demo_timer.stop()
            self.demo_btn.setText("🎭 DEMO")
            self.set_state(HologramState.IDLE)
        else:
            self.demo_timer.start(2000)  # Change state every 2 seconds
            self.demo_btn.setText("⏹️ STOP")
    
    def _demo_state_cycle(self):
        """วนรอบสถานะในโหมดสาธิต"""
        states = [
            HologramState.IDLE,
            HologramState.LISTENING,
            HologramState.PROCESSING,
            HologramState.RESPONDING
        ]
        
        current_index = states.index(self.current_state)
        next_index = (current_index + 1) % len(states)
        self.set_state(states[next_index])


def test_holographic_interface():
    """ทดสอบอินเทอร์เฟซโฮโลแกรม"""
    app = QApplication(sys.argv)
    
    # Create and show interface
    interface = HolographicInterface()
    interface.show()
    
    # Add some sample conversation entries
    interface.add_conversation_entry("USER", "Hello JARVIS")
    interface.add_conversation_entry("JARVIS", "Good morning, sir. How may I assist you today?")
    interface.add_conversation_entry("USER", "What's the weather like?")
    interface.add_conversation_entry("JARVIS", "Checking weather data for your location...")
    
    return app, interface


if __name__ == "__main__":
    app, interface = test_holographic_interface()
    sys.exit(app.exec())