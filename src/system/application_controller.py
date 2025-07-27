"""
Application Controller for Jarvis Voice Assistant
Coordinates between different components and manages application state
"""

import logging
from typing import Dict, Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from voice.voice_controller import VoiceController
from ai.ai_engine import AIEngine
from features.feature_manager import FeatureManager


class ApplicationController(QObject):
    """Main application controller that coordinates all components"""
    
    # Signals
    status_changed = pyqtSignal(str)
    voice_activity_changed = pyqtSignal(bool)
    response_ready = pyqtSignal(str, dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Components
        self.voice_controller: Optional[VoiceController] = None
        self.ai_engine: Optional[AIEngine] = None
        self.feature_manager: Optional[FeatureManager] = None
        self.main_window = None
        
        # State
        self.is_initialized = False
        self.is_listening = False
        self.current_mode = "idle"
        
        # Timer for periodic tasks
        self.timer = QTimer()
        self.timer.timeout.connect(self._periodic_update)
        self.timer.start(1000)  # Update every second
    
    def set_main_window(self, main_window):
        """Set reference to main window"""
        self.main_window = main_window
    
    def initialize(self):
        """Initialize all components"""
        try:
            self.logger.info("Initializing application controller...")
            
            # Initialize components
            self._initialize_voice_controller()
            self._initialize_ai_engine()
            self._initialize_feature_manager()
            
            # Connect signals
            self._connect_signals()
            
            self.is_initialized = True
            self.status_changed.emit("Ready")
            self.logger.info("Application controller initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize application controller: {e}")
            self.error_occurred.emit(f"Initialization failed: {e}")
    
    def _initialize_voice_controller(self):
        """Initialize voice processing controller"""
        try:
            # Ensure voice config exists
            voice_config = self.config.copy()
            voice_config.update(self.config.get('voice', {}))
            
            self.voice_controller = VoiceController(voice_config)
            self.logger.info("Voice controller initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize voice controller: {e}")
            raise
    
    def _initialize_ai_engine(self):
        """Initialize AI engine"""
        try:
            # Ensure AI config exists
            ai_config = self.config.copy()
            ai_config.update(self.config.get('ai', {}))
            
            self.ai_engine = AIEngine(ai_config)
            self.logger.info("AI engine initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI engine: {e}")
            raise
    
    def _initialize_feature_manager(self):
        """Initialize feature manager"""
        try:
            # Ensure features config exists
            features_config = self.config.copy()
            features_config.update(self.config.get('features', {}))
            
            self.feature_manager = FeatureManager(features_config)
            self.logger.info("Feature manager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize feature manager: {e}")
            raise
    
    def _connect_signals(self):
        """Connect signals between components"""
        if self.voice_controller:
            self.voice_controller.voice_detected.connect(self._on_voice_detected)
            self.voice_controller.speech_recognized.connect(self._on_speech_recognized)
            self.voice_controller.listening_changed.connect(self._on_listening_changed)
            self.voice_controller.error_occurred.connect(self._on_voice_error)
        
        if self.ai_engine:
            self.ai_engine.response_ready.connect(self._on_ai_response)
            self.ai_engine.error_occurred.connect(self._on_ai_error)
        
        if self.feature_manager:
            self.feature_manager.feature_executed.connect(self._on_feature_executed)
            self.feature_manager.error_occurred.connect(self._on_feature_error)
    
    def start_listening(self):
        """Start voice listening"""
        if not self.is_initialized:
            self.logger.warning("Cannot start listening: not initialized")
            return
        
        if self.voice_controller:
            self.voice_controller.start_listening()
            self.current_mode = "listening"
            self.logger.info("Started listening")
    
    def stop_listening(self):
        """Stop voice listening"""
        if self.voice_controller:
            self.voice_controller.stop_listening()
            self.current_mode = "idle"
            self.logger.info("Stopped listening")
    
    def process_text_input(self, text: str, language: str = "auto"):
        """Process text input directly"""
        if not self.is_initialized:
            self.logger.warning("Cannot process text: not initialized")
            return
        
        self.logger.info(f"Processing text input: {text[:50]}...")
        self.current_mode = "processing"
        self.status_changed.emit("Processing...")
        
        if self.ai_engine:
            self.ai_engine.process_query(text, language)
    
    def execute_feature(self, feature_name: str, parameters: Dict[str, Any] = None):
        """Execute a specific feature"""
        if not self.is_initialized:
            self.logger.warning("Cannot execute feature: not initialized")
            return None
        
        if self.feature_manager:
            self.feature_manager.execute_feature(feature_name, parameters or {})
            return f"Executing {feature_name}..."
        
        return None
    
    def _on_voice_detected(self):
        """Handle voice detection"""
        self.voice_activity_changed.emit(True)
    
    def _on_speech_recognized(self, text: str, language: str):
        """Handle speech recognition result"""
        self.logger.info(f"Speech recognized: {text} (language: {language})")
        self.voice_activity_changed.emit(False)
        self.process_text_input(text, language)
    
    def _on_listening_changed(self, is_listening: bool):
        """Handle listening state change"""
        self.is_listening = is_listening
        if is_listening:
            self.status_changed.emit("Listening...")
        else:
            self.status_changed.emit("Ready")
    
    def _on_voice_error(self, error_msg: str):
        """Handle voice processing error"""
        self.logger.error(f"Voice error: {error_msg}")
        self.error_occurred.emit(f"Voice error: {error_msg}")
        self.current_mode = "idle"
        self.status_changed.emit("Error")
    
    def _on_ai_response(self, response: str, metadata: Dict[str, Any]):
        """Handle AI response"""
        self.logger.info(f"AI response ready: {response[:50]}...")
        self.response_ready.emit(response, metadata)
        self.current_mode = "idle"
        self.status_changed.emit("Ready")
        
        # Speak the response
        if self.voice_controller:
            self.voice_controller.speak(response, metadata.get('language', 'en'))
    
    def _on_ai_error(self, error_msg: str):
        """Handle AI processing error"""
        self.logger.error(f"AI error: {error_msg}")
        self.error_occurred.emit(f"AI error: {error_msg}")
        self.current_mode = "idle"
        self.status_changed.emit("Error")
    
    def _on_feature_executed(self, feature_name: str, result: Dict[str, Any]):
        """Handle feature execution result"""
        self.logger.info(f"Feature '{feature_name}' executed successfully")
        
        # Format result for display
        if feature_name == "news":
            response = result.get("news_text", "News feature executed")
        elif feature_name == "translate":
            if "translated_text" in result:
                response = f"Translation: {result['translated_text']}"
            else:
                response = result.get("message", "Translation feature executed")
        elif feature_name == "learn":
            if "title" in result:
                response = f"Lesson: {result['title']} - {result.get('description', '')}"
            else:
                response = result.get("message", "Learning feature executed")
        elif feature_name == "deep_question":
            if "answer" in result:
                response = result["answer"]
            else:
                response = result.get("message", "Deep question feature executed")
        elif feature_name == "image_generation":
            response = result.get("message", "Image generation feature executed")
        else:
            response = f"Feature '{feature_name}' executed"
        
        # Emit the response
        metadata = {
            "feature": feature_name,
            "language": result.get("language", "en"),
            "result": result
        }
        
        self.response_ready.emit(response, metadata)
        
        # Speak the response if it's reasonable length
        if len(response) < 500 and self.voice_controller:
            self.voice_controller.speak(response, metadata.get('language', 'en'))
    
    def _on_feature_error(self, error_msg: str):
        """Handle feature execution error"""
        self.logger.error(f"Feature error: {error_msg}")
        self.error_occurred.emit(f"Feature error: {error_msg}")
        self.current_mode = "idle"
        self.status_changed.emit("Error")
    
    def _periodic_update(self):
        """Periodic update for maintenance tasks"""
        # Update status, check resources, etc.
        pass
    
    def shutdown(self):
        """Shutdown all components"""
        self.logger.info("Shutting down application controller...")
        
        if self.voice_controller:
            self.voice_controller.shutdown()
        
        if self.ai_engine:
            self.ai_engine.shutdown()
        
        if self.feature_manager:
            self.feature_manager.shutdown()
        
        self.timer.stop()
        self.logger.info("Application controller shutdown complete")