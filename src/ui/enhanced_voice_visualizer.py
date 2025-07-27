#!/usr/bin/env python3
"""
🎵 Enhanced Voice Visualizer for JARVIS
ระบบแสดงผลเสียงแบบ real-time สวยงามและทันสมัย
"""

import math
import time
import random
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, QRect, pyqtSignal, QPointF
from PyQt6.QtGui import (
    QPainter, QPen, QBrush, QColor, QLinearGradient, QRadialGradient,
    QFont, QPainterPath, QConicalGradient, QPolygonF
)
import numpy as np


class VisualizerMode(Enum):
    """โหมดการแสดงผล"""
    IDLE = "idle"                    # พักเงียบ
    LISTENING = "listening"          # กำลังฟัง
    PROCESSING = "processing"        # กำลังประมวลผล
    RESPONDING = "responding"        # กำลังตอบ
    WAVEFORM = "waveform"           # คลื่นเสียง
    SPECTRUM = "spectrum"           # สเปกตรัม
    CIRCULAR = "circular"           # วงกลม
    MATRIX = "matrix"               # เมทริกซ์


@dataclass
class VisualizerColors:
    """สีสำหรับ Visualizer"""
    primary: str = "#00d4ff"        # Cyan หลัก
    secondary: str = "#0099cc"      # Blue เข้ม
    accent: str = "#ff6b35"         # Orange จุดเด่น
    glow: str = "#4dffff"          # Cyan แสง
    success: str = "#00ff88"        # Green สำเร็จ
    warning: str = "#ffaa00"        # Orange คำเตือน
    error: str = "#ff3366"          # Red ข้อผิดพลาด
    background: str = "#0a0a0a"     # Black เข้ม


class EnhancedVoiceVisualizer(QWidget):
    """Voice Visualizer ขั้นสูงสำหรับ JARVIS"""
    
    # Signals
    mode_changed = pyqtSignal(str)
    audio_level_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Colors and styling
        self.colors = VisualizerColors()
        self.current_mode = VisualizerMode.IDLE
        
        # Audio data
        self.audio_data = np.zeros(128)
        self.audio_level = 0.0
        self.noise_floor = 0.02
        
        # Animation properties
        self.animation_phase = 0.0
        self.glow_intensity = 0.5
        self.pulse_intensity = 0.0
        
        # Waveform data
        self.waveform_history = []
        self.max_history = 100
        
        # Spectrum data
        self.spectrum_bands = 32
        self.spectrum_data = np.zeros(self.spectrum_bands)
        
        # Circular visualizer
        self.circle_radius = 80
        self.circle_segments = 64
        
        # Matrix effect
        self.matrix_drops = []
        self.matrix_chars = "JARVIS01"
        
        # Setup UI
        self._setup_ui()
        
        # Animation timer
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self._update_animation)
        self.animation_timer.start(16)  # ~60 FPS
        
        # Audio simulation timer (for testing)
        self.audio_sim_timer = QTimer()
        self.audio_sim_timer.timeout.connect(self._simulate_audio)
        
        self.setMinimumSize(400, 300)
        
    def _setup_ui(self):
        """ตั้งค่า UI"""
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Status label
        self.status_label = QLabel("🎤 JARVIS Voice Visualizer")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet(f"""
            QLabel {{
                color: {self.colors.primary};
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 14px;
                font-weight: bold;
                padding: 5px;
                background-color: rgba(0, 212, 255, 0.1);
                border: 1px solid {self.colors.primary};
                border-radius: 5px;
            }}
        """)
        
        layout.addWidget(self.status_label)
        self.setLayout(layout)
        
        # Set transparent background
        self.setStyleSheet("background-color: transparent;")
        
    def set_mode(self, mode: VisualizerMode):
        """เปลี่ยนโหมดการแสดงผล"""
        if self.current_mode != mode:
            self.current_mode = mode
            self.mode_changed.emit(mode.value)
            self._update_status_label()
            
            # Start/stop audio simulation based on mode
            if mode in [VisualizerMode.LISTENING, VisualizerMode.PROCESSING, VisualizerMode.RESPONDING]:
                self.start_audio_simulation()
            else:
                self.stop_audio_simulation()
    
    def _update_status_label(self):
        """อัปเดตป้ายสถานะ"""
        status_icons = {
            VisualizerMode.IDLE: "😴",
            VisualizerMode.LISTENING: "🎤", 
            VisualizerMode.PROCESSING: "🧠",
            VisualizerMode.RESPONDING: "🗣️",
            VisualizerMode.WAVEFORM: "📊",
            VisualizerMode.SPECTRUM: "🎵",
            VisualizerMode.CIRCULAR: "⭕",
            VisualizerMode.MATRIX: "🔢"
        }
        
        status_texts = {
            VisualizerMode.IDLE: "Standby",
            VisualizerMode.LISTENING: "Listening...",
            VisualizerMode.PROCESSING: "Processing...",
            VisualizerMode.RESPONDING: "Speaking...",
            VisualizerMode.WAVEFORM: "Waveform Mode",
            VisualizerMode.SPECTRUM: "Spectrum Mode", 
            VisualizerMode.CIRCULAR: "Circular Mode",
            VisualizerMode.MATRIX: "Matrix Mode"
        }
        
        icon = status_icons.get(self.current_mode, "🎤")
        text = status_texts.get(self.current_mode, "JARVIS")
        
        self.status_label.setText(f"{icon} {text}")
    
    def start_audio_simulation(self):
        """เริ่มจำลองข้อมูลเสียง"""
        self.audio_sim_timer.start(50)  # 20 FPS
    
    def stop_audio_simulation(self):
        """หยุดจำลองข้อมูลเสียง"""
        self.audio_sim_timer.stop()
        self.audio_level = 0.0
        self.audio_data = np.zeros(128)
        self.spectrum_data = np.zeros(self.spectrum_bands)
    
    def _simulate_audio(self):
        """จำลองข้อมูลเสียงสำหรับทดสอบ"""
        t = time.time()
        
        if self.current_mode == VisualizerMode.LISTENING:
            # Simulate gentle listening pattern
            self.audio_level = 0.3 + 0.2 * math.sin(t * 2) + random.uniform(-0.1, 0.1)
            
        elif self.current_mode == VisualizerMode.PROCESSING:
            # Simulate processing pattern
            self.audio_level = 0.5 + 0.3 * math.sin(t * 4) + 0.1 * math.sin(t * 8)
            
        elif self.current_mode == VisualizerMode.RESPONDING:
            # Simulate speaking pattern
            self.audio_level = 0.7 + 0.4 * math.sin(t * 6) + random.uniform(-0.2, 0.2)
        
        # Generate audio spectrum data
        for i in range(self.spectrum_bands):
            freq = (i + 1) * 50  # Frequency bins
            amplitude = self.audio_level * (1.0 - i / self.spectrum_bands) * (1 + 0.5 * math.sin(t * freq / 100))
            self.spectrum_data[i] = max(0, amplitude + random.uniform(-0.1, 0.1))
        
        # Generate waveform data
        for i in range(len(self.audio_data)):
            phase = t * 10 + i * 0.1
            self.audio_data[i] = self.audio_level * math.sin(phase) * (1 + 0.3 * math.sin(phase * 0.7))
        
        # Add to waveform history
        self.waveform_history.append(self.audio_level)
        if len(self.waveform_history) > self.max_history:
            self.waveform_history.pop(0)
        
        self.audio_level_changed.emit(self.audio_level)
    
    def set_audio_data(self, audio_array: np.ndarray):
        """ตั้งค่าข้อมูลเสียงจริง"""
        if len(audio_array) > 0:
            self.audio_data = audio_array[:128]  # Take first 128 samples
            self.audio_level = np.sqrt(np.mean(audio_array ** 2))
            
            # Calculate spectrum (simplified FFT)
            if len(audio_array) >= self.spectrum_bands * 2:
                fft_data = np.abs(np.fft.fft(audio_array[:self.spectrum_bands * 2]))
                self.spectrum_data = fft_data[:self.spectrum_bands] / np.max(fft_data + 1e-10)
    
    def _update_animation(self):
        """อัปเดตแอนิเมชัน"""
        self.animation_phase += 0.1
        
        # Update glow intensity
        self.glow_intensity = 0.5 + 0.3 * math.sin(self.animation_phase * 0.5)
        
        # Update pulse based on audio level
        self.pulse_intensity = self.audio_level
        
        # Update matrix drops
        self._update_matrix_drops()
        
        self.update()  # Trigger repaint
    
    def _update_matrix_drops(self):
        """อัปเดตการตกของตัวอักษรในโหมด Matrix"""
        # Add new drops randomly
        if random.random() < 0.05 and len(self.matrix_drops) < 20:
            self.matrix_drops.append({
                'x': random.randint(0, self.width() // 10),
                'y': 0,
                'speed': random.uniform(2, 8),
                'char': random.choice(self.matrix_chars)
            })
        
        # Update existing drops
        for drop in self.matrix_drops[:]:
            drop['y'] += drop['speed']
            if drop['y'] > self.height():
                self.matrix_drops.remove(drop)
    
    def paintEvent(self, event):
        """วาดการแสดงผล"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(self.colors.background))
        
        # Draw based on current mode
        if self.current_mode == VisualizerMode.IDLE:
            self._draw_idle_state(painter)
        elif self.current_mode == VisualizerMode.LISTENING:
            self._draw_listening_state(painter)
        elif self.current_mode == VisualizerMode.PROCESSING:
            self._draw_processing_state(painter)
        elif self.current_mode == VisualizerMode.RESPONDING:
            self._draw_responding_state(painter)
        elif self.current_mode == VisualizerMode.WAVEFORM:
            self._draw_waveform(painter)
        elif self.current_mode == VisualizerMode.SPECTRUM:
            self._draw_spectrum(painter)
        elif self.current_mode == VisualizerMode.CIRCULAR:
            self._draw_circular(painter)
        elif self.current_mode == VisualizerMode.MATRIX:
            self._draw_matrix(painter)
    
    def _draw_idle_state(self, painter: QPainter):
        """วาดสถานะ Idle"""
        center = self.rect().center()
        
        # Draw breathing circle
        radius = 30 + 10 * math.sin(self.animation_phase * 0.3)
        
        # Glow effect
        glow_pen = QPen(QColor(self.colors.glow))
        glow_pen.setWidth(3)
        painter.setPen(glow_pen)
        painter.drawEllipse(center, int(radius + 5), int(radius + 5))
        
        # Main circle
        main_pen = QPen(QColor(self.colors.primary))
        main_pen.setWidth(2)
        painter.setPen(main_pen)
        painter.drawEllipse(center, int(radius), int(radius))
        
        # Center dot
        painter.setBrush(QBrush(QColor(self.colors.primary)))
        painter.drawEllipse(center, 3, 3)
    
    def _draw_listening_state(self, painter: QPainter):
        """วาดสถานะ Listening"""
        center = self.rect().center()
        width = self.width()
        height = self.height()
        
        # Draw concentric circles representing sound waves
        for i in range(5):
            radius = 40 + i * 20 + self.pulse_intensity * 30
            alpha = int(255 * (1.0 - i * 0.2) * self.glow_intensity)
            
            pen = QPen(QColor(self.colors.primary))
            pen.setWidth(2)
            color = QColor(self.colors.primary)
            color.setAlpha(alpha)
            pen.setColor(color)
            painter.setPen(pen)
            
            painter.drawEllipse(center, int(radius), int(radius))
        
        # Draw microphone icon in center
        self._draw_microphone_icon(painter, center)
    
    def _draw_processing_state(self, painter: QPainter):
        """วาดสถานะ Processing"""
        center = self.rect().center()
        
        # Draw rotating brain-like pattern
        num_segments = 8
        for i in range(num_segments):
            angle = (self.animation_phase + i * 360 / num_segments) * math.pi / 180
            
            start_radius = 20
            end_radius = 60 + 20 * math.sin(self.animation_phase * 0.5 + i)
            
            start_x = center.x() + start_radius * math.cos(angle)
            start_y = center.y() + start_radius * math.sin(angle)
            end_x = center.x() + end_radius * math.cos(angle)
            end_y = center.y() + end_radius * math.sin(angle)
            
            # Create gradient
            gradient = QLinearGradient(start_x, start_y, end_x, end_y)
            gradient.setColorAt(0, QColor(self.colors.primary))
            gradient.setColorAt(1, QColor(self.colors.accent))
            
            pen = QPen(QBrush(gradient), 3)
            painter.setPen(pen)
            painter.drawLine(int(start_x), int(start_y), int(end_x), int(end_y))
    
    def _draw_responding_state(self, painter: QPainter):
        """วาดสถานะ Responding"""
        center = self.rect().center()
        center_f = QPointF(center.x(), center.y())  # Convert to QPointF
        
        # Draw sound waves emanating from center
        for i in range(3):
            radius = 50 + i * 25 + self.pulse_intensity * 40
            
            # Create conical gradient for wave effect
            gradient = QConicalGradient(center_f, self.animation_phase * 10)
            gradient.setColorAt(0, QColor(self.colors.success))
            gradient.setColorAt(0.5, QColor(self.colors.primary))
            gradient.setColorAt(1, QColor(self.colors.success))
            
            pen = QPen(QBrush(gradient), 3)
            painter.setPen(pen)
            painter.drawEllipse(center, int(radius), int(radius))
        
        # Draw speaker icon in center
        self._draw_speaker_icon(painter, center_f)
    
    def _draw_waveform(self, painter: QPainter):
        """วาดคลื่นเสียง"""
        if len(self.waveform_history) < 2:
            return
        
        width = self.width()
        height = self.height()
        center_y = height // 2
        
        # Draw waveform
        pen = QPen(QColor(self.colors.primary), 2)
        painter.setPen(pen)
        
        for i in range(len(self.waveform_history) - 1):
            x1 = int(i * width / len(self.waveform_history))
            y1 = int(center_y - self.waveform_history[i] * center_y * 0.8)
            x2 = int((i + 1) * width / len(self.waveform_history))
            y2 = int(center_y - self.waveform_history[i + 1] * center_y * 0.8)
            
            painter.drawLine(x1, y1, x2, y2)
    
    def _draw_spectrum(self, painter: QPainter):
        """วาดสเปกตรัม"""
        width = self.width()
        height = self.height()
        
        bar_width = width // self.spectrum_bands
        
        for i in range(self.spectrum_bands):
            x = i * bar_width
            bar_height = int(self.spectrum_data[i] * height * 0.8)
            y = height - bar_height
            
            # Create gradient for each bar
            gradient = QLinearGradient(0, height, 0, y)
            gradient.setColorAt(0, QColor(self.colors.primary))
            gradient.setColorAt(1, QColor(self.colors.glow))
            
            painter.fillRect(x, y, bar_width - 2, bar_height, QBrush(gradient))
    
    def _draw_circular(self, painter: QPainter):
        """วาดแบบวงกลม"""
        center = self.rect().center()
        
        # Draw spectrum in circular form
        for i in range(self.spectrum_bands):
            angle = (i * 360 / self.spectrum_bands) * math.pi / 180
            
            inner_radius = self.circle_radius
            outer_radius = inner_radius + self.spectrum_data[i] * 50
            
            inner_x = center.x() + inner_radius * math.cos(angle)
            inner_y = center.y() + inner_radius * math.sin(angle)
            outer_x = center.x() + outer_radius * math.cos(angle)
            outer_y = center.y() + outer_radius * math.sin(angle)
            
            pen = QPen(QColor(self.colors.primary), 3)
            painter.setPen(pen)
            painter.drawLine(int(inner_x), int(inner_y), int(outer_x), int(outer_y))
    
    def _draw_matrix(self, painter: QPainter):
        """วาดเอฟเฟกต์ Matrix"""
        font = QFont("Consolas", 12)
        painter.setFont(font)
        
        for drop in self.matrix_drops:
            x = drop['x'] * 10
            y = int(drop['y'])
            char = drop['char']
            
            # Fade color based on position
            alpha = max(0, 255 - int(drop['y'] / self.height() * 255))
            color = QColor(self.colors.success)
            color.setAlpha(alpha)
            
            painter.setPen(QPen(color))
            painter.drawText(x, y, char)
    
    def _draw_microphone_icon(self, painter: QPainter, center: QPointF):
        """วาดไอคอนไมโครโฟน"""
        painter.setBrush(QBrush(QColor(self.colors.primary)))
        painter.setPen(QPen(QColor(self.colors.primary), 2))
        
        # Simple microphone shape
        mic_rect = QRect(int(center.x() - 8), int(center.y() - 15), 16, 20)
        painter.drawRoundedRect(mic_rect, 8, 8)
        
        # Microphone stand
        painter.drawLine(int(center.x()), int(center.y() + 5), int(center.x()), int(center.y() + 15))
        painter.drawLine(int(center.x() - 8), int(center.y() + 15), int(center.x() + 8), int(center.y() + 15))
    
    def _draw_speaker_icon(self, painter: QPainter, center: QPointF):
        """วาดไอคอนลำโพง"""
        painter.setBrush(QBrush(QColor(self.colors.success)))
        painter.setPen(QPen(QColor(self.colors.success), 2))
        
        # Simple speaker shape
        speaker_rect = QRect(int(center.x() - 6), int(center.y() - 8), 12, 16)
        painter.drawRect(speaker_rect)
        
        # Sound waves
        for i in range(3):
            radius = 15 + i * 8
            painter.drawArc(int(center.x()), int(center.y() - radius//2), radius, radius, 30 * 16, 120 * 16)
    
    def mousePressEvent(self, event):
        """จัดการการคลิกเมาส์"""
        if event.button() == Qt.MouseButton.LeftButton:
            # Cycle through visualization modes
            modes = list(VisualizerMode)
            current_index = modes.index(self.current_mode)
            next_index = (current_index + 1) % len(modes)
            self.set_mode(modes[next_index])


# Test function
def test_enhanced_voice_visualizer():
    """ทดสอบ Enhanced Voice Visualizer"""
    import sys
    
    app = QApplication(sys.argv if 'sys' in globals() else [])
    
    # Create main window
    window = QWidget()
    window.setWindowTitle("🎵 Enhanced Voice Visualizer Test")
    window.setGeometry(100, 100, 600, 400)
    window.setStyleSheet("background-color: #0a0a0a;")
    
    # Create visualizer
    visualizer = EnhancedVoiceVisualizer()
    
    layout = QVBoxLayout()
    layout.addWidget(visualizer)
    window.setLayout(layout)
    
    # Test mode cycling
    modes = [
        VisualizerMode.IDLE,
        VisualizerMode.LISTENING,
        VisualizerMode.PROCESSING,
        VisualizerMode.RESPONDING,
        VisualizerMode.WAVEFORM,
        VisualizerMode.SPECTRUM,
        VisualizerMode.CIRCULAR,
        VisualizerMode.MATRIX
    ]
    
    current_mode_index = 0
    
    def cycle_mode():
        nonlocal current_mode_index
        visualizer.set_mode(modes[current_mode_index])
        current_mode_index = (current_mode_index + 1) % len(modes)
    
    # Timer to cycle through modes automatically
    mode_timer = QTimer()
    mode_timer.timeout.connect(cycle_mode)
    mode_timer.start(3000)  # Change mode every 3 seconds
    
    # Show window
    window.show()
    
    print("🧪 Enhanced Voice Visualizer Test")
    print("Click on the visualizer to cycle through modes manually")
    print("Modes will also change automatically every 3 seconds")
    
    return app, window


if __name__ == "__main__":
    app, window = test_enhanced_voice_visualizer()
    sys.exit(app.exec())