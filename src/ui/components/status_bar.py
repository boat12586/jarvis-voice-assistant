"""
Status Bar Component for Jarvis Voice Assistant
Displays current application status and messages
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QFont


class StatusBar(QWidget):
    """Status bar widget for displaying application status"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        
        # Create layout
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        
        # Status label
        self.status_label = QLabel("Initializing...")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        
        # Activity indicator
        self.activity_label = QLabel("●")
        self.activity_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.activity_label.setFixedWidth(20)
        
        # Add to layout
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addWidget(self.activity_label)
        
        # Setup fonts
        font = QFont("Segoe UI", 10)
        self.status_label.setFont(font)
        
        activity_font = QFont("Segoe UI", 12)
        self.activity_label.setFont(activity_font)
        
        # Animation timer for activity indicator
        self.activity_timer = QTimer()
        self.activity_timer.timeout.connect(self._animate_activity)
        self.activity_timer.start(500)  # Blink every 500ms
        
        # Current status
        self.current_status = "idle"
        self.blink_state = False
    
    def update_status(self, status: str):
        """Update status text"""
        self.current_status = status.lower()
        self.status_label.setText(status)
        
        # Update status property for styling
        self.status_label.setProperty("status", self._get_status_type(status))
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        
        # Update activity indicator
        self._update_activity_indicator()
    
    def show_error(self, error_msg: str):
        """Show error message"""
        self.current_status = "error"
        self.status_label.setText(f"Error: {error_msg}")
        self.status_label.setProperty("status", "error")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        
        # Red activity indicator
        self.activity_label.setStyleSheet("color: #ff4444;")
        
        # Auto-clear error after 5 seconds
        QTimer.singleShot(5000, lambda: self.update_status("Ready"))
    
    def show_warning(self, warning_msg: str):
        """Show warning message"""
        self.current_status = "warning"
        self.status_label.setText(f"Warning: {warning_msg}")
        self.status_label.setProperty("status", "warning")
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
        
        # Yellow activity indicator
        self.activity_label.setStyleSheet("color: #ffaa00;")
        
        # Auto-clear warning after 3 seconds
        QTimer.singleShot(3000, lambda: self.update_status("Ready"))
    
    def _get_status_type(self, status: str) -> str:
        """Get status type for styling"""
        status_lower = status.lower()
        
        if "error" in status_lower or "failed" in status_lower:
            return "error"
        elif "warning" in status_lower or "warn" in status_lower:
            return "warning"
        elif "ready" in status_lower or "complete" in status_lower:
            return "success"
        else:
            return "normal"
    
    def _update_activity_indicator(self):
        """Update activity indicator based on current status"""
        status_lower = self.current_status.lower()
        
        if "listening" in status_lower:
            self.activity_label.setStyleSheet("color: #ff6b35;")  # Orange
        elif "processing" in status_lower:
            self.activity_label.setStyleSheet("color: #00d4ff;")  # Blue
        elif "ready" in status_lower:
            self.activity_label.setStyleSheet("color: #00ff88;")  # Green
        elif "error" in status_lower:
            self.activity_label.setStyleSheet("color: #ff4444;")  # Red
        else:
            self.activity_label.setStyleSheet("color: #ffffff;")  # White
    
    def _animate_activity(self):
        """Animate activity indicator"""
        if self.current_status in ["listening", "processing"]:
            # Blink animation for active states
            self.blink_state = not self.blink_state
            if self.blink_state:
                self.activity_label.setText("●")
            else:
                self.activity_label.setText("○")
        else:
            # Solid indicator for idle states
            self.activity_label.setText("●")
            self.blink_state = False