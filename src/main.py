#!/usr/bin/env python3
"""
🤖 JARVIS Voice Assistant v2.0 - Main Entry Point
A J.A.R.V.I.S-inspired local voice assistant with advanced AI capabilities

Version: 2.0.0 (2025 Edition)
Features: DeepSeek-R1, mxbai-embed-large, Multimodal AI, Thai Language
Author: JARVIS Development Team
"""

import sys
import os
import logging
import traceback
import signal
from pathlib import Path
from typing import Optional
import platform

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Version and build info
__version__ = "2.0.0"
__build__ = "2025.07.22"
__author__ = "JARVIS Development Team"

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from system.config_manager import ConfigManager
from system.logger import setup_logger
from ui.main_window import MainWindow
from system.application_controller import ApplicationController
# Lazy imports for better startup performance
def lazy_import_ai():
    """Lazy import AI components"""
    global AIEngine, RAGSystem
    from ai.ai_engine import AIEngine
    from ai.rag_system import RAGSystem
    return AIEngine, RAGSystem

def lazy_import_voice():
    """Lazy import voice components"""
    global VoiceController
    from voice.voice_controller import VoiceController
    return VoiceController



def setup_application():
    """Set up the Qt application with proper configuration"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("Jarvis Voice Assistant")
    app.setApplicationVersion("0.1.0")
    app.setOrganizationName("Jarvis AI")
    app.setOrganizationDomain("jarvis.ai")
    
    # Enable high DPI support (PyQt6 compatible)
    try:
        app.setAttribute(Qt.ApplicationAttribute.AA_EnableHighDpiScaling, True)
        app.setAttribute(Qt.ApplicationAttribute.AA_UseHighDpiPixmaps, True)
    except AttributeError:
        # PyQt6 handles high DPI automatically
        pass
    
    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    return app


def main():
    """Main entry point for the application"""
    try:
        # Initialize configuration
        config_manager = ConfigManager()
        config = config_manager.load_config()
        
        # Setup logging
        setup_logger(config.get('system', {}).get('log_level', 'INFO'))
        logger = logging.getLogger(__name__)
        
        logger.info("Starting Jarvis Voice Assistant...")
        logger.info(f"Python version: {sys.version}")
        logger.info(f"Working directory: {os.getcwd()}")
        
        # Create Qt application
        app = setup_application()
        
        # Create application controller
        controller = ApplicationController(config)
        
        # Create main window
        main_window = MainWindow(controller)
        
        # Connect controller to main window
        controller.set_main_window(main_window)
        
        # Show main window
        main_window.show()
        
        # Initialize controller
        controller.initialize()
        
        logger.info("Application started successfully")
        
        # Run the application
        sys.exit(app.exec())
        
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Failed to start application: {e}", exc_info=True)
        else:
            print(f"Failed to start application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()