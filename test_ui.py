#!/usr/bin/env python3
"""
Simple UI Test for Jarvis Voice Assistant
Tests basic window and UI components without AI dependencies
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QLabel, QPushButton
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont

class SimpleTestWindow(QMainWindow):
    """Simple test window"""
    
    def __init__(self):
        super().__init__()
        self.init_ui()
    
    def init_ui(self):
        """Initialize UI"""
        self.setWindowTitle("Jarvis Voice Assistant - UI Test")
        self.setFixedSize(800, 600)
        
        # Set window to center
        screen = self.screen().availableGeometry()
        x = (screen.width() - 800) // 2
        y = (screen.height() - 600) // 2
        self.move(x, y)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(20)
        layout.setContentsMargins(50, 50, 50, 50)
        
        # Title
        title = QLabel("J.A.R.V.I.S")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setFont(QFont("Arial", 32, QFont.Weight.Bold))
        title.setStyleSheet("""
            color: #00d4ff;
            background: transparent;
            margin: 20px;
        """)
        
        # Status
        self.status_label = QLabel("UI Test Mode - Ready")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setFont(QFont("Arial", 14))
        self.status_label.setStyleSheet("color: #ffffff; background: transparent;")
        
        # Buttons layout
        buttons_layout = QVBoxLayout()
        buttons_layout.setSpacing(15)
        
        # Button style
        button_style = """
            QPushButton {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 212, 255, 0.3),
                    stop:1 rgba(0, 153, 204, 0.3));
                border: 2px solid rgba(0, 212, 255, 0.5);
                border-radius: 15px;
                color: white;
                font-size: 16px;
                font-weight: bold;
                padding: 15px 30px;
                min-height: 50px;
            }
            QPushButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 212, 255, 0.5),
                    stop:1 rgba(0, 153, 204, 0.5));
                border: 2px solid rgba(0, 212, 255, 0.7);
            }
            QPushButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(0, 212, 255, 0.7),
                    stop:1 rgba(0, 153, 204, 0.7));
            }
        """
        
        # Create test buttons
        buttons = [
            ("🎙️ Talk to AI", self.test_voice),
            ("📰 News", self.test_news),
            ("🌐 Translate", self.test_translate),
            ("📚 Learn", self.test_learn),
            ("🤔 Deep Question", self.test_deep),
            ("🎨 Generate Image", self.test_image)
        ]
        
        # First row
        row1 = QHBoxLayout()
        for i in range(3):
            btn = QPushButton(buttons[i][0])
            btn.setStyleSheet(button_style)
            btn.clicked.connect(buttons[i][1])
            row1.addWidget(btn)
        
        # Second row
        row2 = QHBoxLayout()
        for i in range(3, 6):
            btn = QPushButton(buttons[i][0])
            btn.setStyleSheet(button_style)
            btn.clicked.connect(buttons[i][1])
            row2.addWidget(btn)
        
        buttons_layout.addLayout(row1)
        buttons_layout.addLayout(row2)
        
        # Add to main layout
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.status_label)
        layout.addStretch()
        layout.addLayout(buttons_layout)
        layout.addStretch()
        
        # Set main window style
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 rgba(26, 26, 26, 0.9),
                    stop:1 rgba(0, 212, 255, 0.1));
            }
        """)
        
        # Auto-update status
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_status)
        self.timer.start(2000)  # Update every 2 seconds
        
        self.test_count = 0
    
    def update_status(self):
        """Update status periodically"""
        statuses = [
            "UI Test Mode - Ready",
            "All UI Components Loaded",
            "Buttons Responsive",
            "Window Rendering OK"
        ]
        self.status_label.setText(statuses[self.test_count % len(statuses)])
        self.test_count += 1
    
    def test_voice(self):
        """Test voice button"""
        self.status_label.setText("🎙️ Voice Test - Button Clicked!")
        print("Voice button test: OK")
    
    def test_news(self):
        """Test news button"""
        self.status_label.setText("📰 News Test - Button Clicked!")
        print("News button test: OK")
    
    def test_translate(self):
        """Test translate button"""
        self.status_label.setText("🌐 Translate Test - Button Clicked!")
        print("Translate button test: OK")
    
    def test_learn(self):
        """Test learn button"""
        self.status_label.setText("📚 Learn Test - Button Clicked!")
        print("Learn button test: OK")
    
    def test_deep(self):
        """Test deep question button"""
        self.status_label.setText("🤔 Deep Question Test - Button Clicked!")
        print("Deep question button test: OK")
    
    def test_image(self):
        """Test image generation button"""
        self.status_label.setText("🎨 Image Generation Test - Button Clicked!")
        print("Image generation button test: OK")


def main():
    """Main function"""
    app = QApplication(sys.argv)
    app.setApplicationName("Jarvis UI Test")
    
    # Set application style
    app.setStyleSheet("""
        QApplication {
            background-color: #1a1a1a;
        }
    """)
    
    window = SimpleTestWindow()
    window.show()
    
    print("Jarvis UI Test Started")
    print("Testing basic UI components...")
    print("Click buttons to test functionality")
    print("Close window to exit")
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()