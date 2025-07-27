"""
Action Button Component for Jarvis Voice Assistant
Custom button with icon, text, and hover effects
"""

from PyQt6.QtWidgets import QPushButton, QVBoxLayout, QHBoxLayout, QLabel, QWidget
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QRect
from PyQt6.QtGui import QFont, QPixmap, QPainter, QPen, QBrush


class ActionButton(QPushButton):
    """Custom action button with icon and text"""
    
    def __init__(self, icon: str, text: str, tooltip: str = ""):
        super().__init__()
        self.icon_text = icon
        self.button_text = text
        self.is_active = False
        
        # Setup button
        self.setFixedSize(140, 80)
        self.setToolTip(tooltip)
        self.setText(f"{icon}\n{text}")
        
        # Setup font
        font = QFont("Segoe UI", 12)
        font.setBold(True)
        self.setFont(font)
        
        # Setup properties
        self.setProperty("active", False)
        
        # Animation
        self.animation = QPropertyAnimation(self, b"geometry")
        self.animation.setDuration(100)
    
    def set_active(self, active: bool):
        """Set button active state"""
        self.is_active = active
        self.setProperty("active", active)
        self.style().unpolish(self)
        self.style().polish(self)
    
    def enterEvent(self, event):
        """Handle mouse enter event"""
        super().enterEvent(event)
        # Scale up animation
        current_rect = self.geometry()
        new_rect = QRect(
            current_rect.x() - 2,
            current_rect.y() - 2,
            current_rect.width() + 4,
            current_rect.height() + 4
        )
        self.animation.setStartValue(current_rect)
        self.animation.setEndValue(new_rect)
        self.animation.start()
    
    def leaveEvent(self, event):
        """Handle mouse leave event"""
        super().leaveEvent(event)
        # Scale down animation
        current_rect = self.geometry()
        new_rect = QRect(
            current_rect.x() + 2,
            current_rect.y() + 2,
            current_rect.width() - 4,
            current_rect.height() - 4
        )
        self.animation.setStartValue(current_rect)
        self.animation.setEndValue(new_rect)
        self.animation.start()
    
    def paintEvent(self, event):
        """Custom paint event for additional effects"""
        super().paintEvent(event)
        
        # Add glow effect when active
        if self.is_active:
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            
            # Draw glow
            pen = QPen()
            pen.setColor(Qt.GlobalColor.transparent)
            painter.setPen(pen)
            
            # This would add a custom glow effect
            # Implementation would depend on specific visual requirements