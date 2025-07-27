"""
Voice Visualizer Component for Jarvis Voice Assistant
Displays audio waveform and voice activity visualization
"""

import math
import random
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QLinearGradient


class VoiceVisualizer(QWidget):
    """Voice activity visualizer with animated waveform"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(200, 100)
        
        # State
        self.is_active = False
        self.is_listening = False
        self.amplitude = 0.0
        self.frequency = 1.0
        
        # Animation
        self.animation_timer = QTimer()
        self.animation_timer.timeout.connect(self.update_animation)
        self.animation_timer.start(50)  # 20 FPS
        
        # Wave data
        self.wave_points = []
        self.wave_offset = 0
        self.wave_amplitude = 10
        
        # Initialize wave points
        self._init_wave_points()
    
    def _init_wave_points(self):
        """Initialize wave points for visualization"""
        self.wave_points = []
        width = self.width()
        for i in range(width):
            self.wave_points.append(0)
    
    def set_active(self, active: bool):
        """Set voice activity state"""
        self.is_active = active
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
    
    def set_listening(self, listening: bool):
        """Set listening state"""
        self.is_listening = listening
        self.setProperty("listening", listening)
        self.style().unpolish(self)
        self.style().polish(self)
    
    def set_amplitude(self, amplitude: float):
        """Set current audio amplitude (0.0 to 1.0)"""
        self.amplitude = max(0.0, min(1.0, amplitude))
    
    def update_animation(self):
        """Update animation frame"""
        if self.is_active or self.is_listening:
            # Update wave offset for animation
            self.wave_offset += 0.2
            
            # Update wave amplitude based on activity
            if self.is_active:
                self.wave_amplitude = 20 + (self.amplitude * 30)
            elif self.is_listening:
                self.wave_amplitude = 10 + (random.random() * 5)
            else:
                self.wave_amplitude = max(2, self.wave_amplitude * 0.95)
            
            # Update wave points
            self._update_wave_points()
        else:
            # Fade out animation
            self.wave_amplitude = max(0, self.wave_amplitude * 0.9)
            if self.wave_amplitude > 0.1:
                self._update_wave_points()
        
        self.update()
    
    def _update_wave_points(self):
        """Update wave points for current frame"""
        width = self.width()
        center_y = self.height() // 2
        
        for i in range(width):
            # Create wave pattern
            x_ratio = i / width
            wave1 = math.sin((x_ratio * 4 * math.pi) + self.wave_offset) * self.wave_amplitude
            wave2 = math.sin((x_ratio * 6 * math.pi) + (self.wave_offset * 1.5)) * (self.wave_amplitude * 0.3)
            wave3 = math.sin((x_ratio * 8 * math.pi) + (self.wave_offset * 0.8)) * (self.wave_amplitude * 0.2)
            
            # Combine waves
            combined_wave = wave1 + wave2 + wave3
            
            # Add some randomness if active
            if self.is_active:
                combined_wave += random.uniform(-2, 2)
            
            self.wave_points[i] = center_y + combined_wave
    
    def paintEvent(self, event):
        """Custom paint event"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Clear background
        painter.fillRect(self.rect(), QColor(0, 0, 0, 0))
        
        # Draw wave
        self._draw_wave(painter)
        
        # Draw center circle
        self._draw_center_circle(painter)
    
    def _draw_wave(self, painter):
        """Draw the waveform"""
        if not self.wave_points:
            return
        
        # Set up gradient
        gradient = QLinearGradient(0, 0, self.width(), 0)
        
        if self.is_listening:
            # Orange gradient for listening
            gradient.setColorAt(0, QColor(255, 107, 53, 100))
            gradient.setColorAt(0.5, QColor(255, 107, 53, 200))
            gradient.setColorAt(1, QColor(255, 107, 53, 100))
        elif self.is_active:
            # Blue gradient for active
            gradient.setColorAt(0, QColor(0, 212, 255, 100))
            gradient.setColorAt(0.5, QColor(0, 212, 255, 200))
            gradient.setColorAt(1, QColor(0, 212, 255, 100))
        else:
            # Gray gradient for idle
            gradient.setColorAt(0, QColor(100, 100, 100, 50))
            gradient.setColorAt(0.5, QColor(100, 100, 100, 100))
            gradient.setColorAt(1, QColor(100, 100, 100, 50))
        
        # Draw wave lines
        pen = QPen(QBrush(gradient), 2)
        painter.setPen(pen)
        
        # Draw multiple wave lines for thickness
        for offset in [-1, 0, 1]:
            for i in range(1, len(self.wave_points)):
                x1, y1 = i - 1, int(self.wave_points[i - 1] + offset)
                x2, y2 = i, int(self.wave_points[i] + offset)
                painter.drawLine(x1, y1, x2, y2)
    
    def _draw_center_circle(self, painter):
        """Draw center circle indicator"""
        center_x = self.width() // 2
        center_y = self.height() // 2
        
        # Circle size based on activity
        if self.is_listening:
            radius = 8 + (self.wave_amplitude * 0.2)
            color = QColor(255, 107, 53, 150)
        elif self.is_active:
            radius = 6 + (self.wave_amplitude * 0.3)
            color = QColor(0, 212, 255, 150)
        else:
            radius = 4
            color = QColor(100, 100, 100, 100)
        
        # Draw circle
        painter.setBrush(QBrush(color))
        painter.setPen(QPen(color.darker(), 1))
        painter.drawEllipse(int(center_x - radius), int(center_y - radius), 
                          int(radius * 2), int(radius * 2))
        
        # Draw inner glow
        inner_color = QColor(color)
        inner_color.setAlpha(50)
        painter.setBrush(QBrush(inner_color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(int(center_x - radius - 3), int(center_y - radius - 3),
                          int((radius + 3) * 2), int((radius + 3) * 2))
    
    def resizeEvent(self, event):
        """Handle resize event"""
        super().resizeEvent(event)
        self._init_wave_points()