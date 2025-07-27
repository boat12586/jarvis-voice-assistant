"""
Continuous Listening Mode for JARVIS Voice Assistant
Advanced continuous voice interaction with intelligent state management
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from dataclasses import dataclass
from enum import Enum
import queue
import collections

from .voice_controller import VoiceController
from .wake_word_detector import WakeWordDetector
from .speech_recognizer import SimpleSpeechRecognizer


class ListeningState(Enum):
    """Continuous listening states"""
    IDLE = "idle"
    WAKE_WORD_LISTENING = "wake_word_listening"
    COMMAND_LISTENING = "command_listening"
    PROCESSING = "processing"
    RESPONDING = "responding"
    COOLDOWN = "cooldown"
    ERROR = "error"


class ActivityLevel(Enum):
    """Voice activity levels"""
    SILENT = "silent"
    BACKGROUND = "background"
    SPEECH = "speech"
    LOUD = "loud"


@dataclass
class ListeningSession:
    """Continuous listening session data"""
    session_id: str
    start_time: float
    state: ListeningState
    total_wake_words: int = 0
    total_commands: int = 0
    total_errors: int = 0
    last_activity_time: float = 0
    current_conversation_turns: int = 0


class VoiceActivityMonitor:
    """Monitors voice activity for intelligent listening management"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + ".VoiceActivityMonitor")
        self.config = config.get("voice_activity_monitor", {})
        
        # Activity thresholds
        self.silence_threshold = self.config.get("silence_threshold", 0.01)
        self.speech_threshold = self.config.get("speech_threshold", 0.05)
        self.loud_threshold = self.config.get("loud_threshold", 0.15)
        
        # Timing parameters
        self.activity_window = self.config.get("activity_window", 2.0)  # seconds
        self.silence_timeout = self.config.get("silence_timeout", 5.0)  # seconds
        
        # Activity history
        self.activity_history = collections.deque(maxlen=100)
        self.current_level = ActivityLevel.SILENT
        self.last_speech_time = 0
        
    def update_activity(self, audio_level: float) -> ActivityLevel:
        """Update voice activity level"""
        try:
            current_time = time.time()
            
            # Determine activity level
            if audio_level > self.loud_threshold:
                level = ActivityLevel.LOUD
            elif audio_level > self.speech_threshold:
                level = ActivityLevel.SPEECH
                self.last_speech_time = current_time
            elif audio_level > self.silence_threshold:
                level = ActivityLevel.BACKGROUND
            else:
                level = ActivityLevel.SILENT
            
            # Add to history
            self.activity_history.append({
                "timestamp": current_time,
                "level": level,
                "audio_level": audio_level
            })
            
            self.current_level = level
            return level
            
        except Exception as e:
            self.logger.error(f"Activity update failed: {e}")
            return ActivityLevel.SILENT
    
    def get_recent_activity(self, window_seconds: float = None) -> List[Dict[str, Any]]:
        """Get recent activity within time window"""
        try:
            if window_seconds is None:
                window_seconds = self.activity_window
                
            current_time = time.time()
            cutoff_time = current_time - window_seconds
            
            recent_activity = [
                activity for activity in self.activity_history
                if activity["timestamp"] > cutoff_time
            ]
            
            return recent_activity
            
        except Exception as e:
            self.logger.error(f"Activity retrieval failed: {e}")
            return []
    
    def is_speech_active(self) -> bool:
        """Check if speech is currently active"""
        return (
            self.current_level in [ActivityLevel.SPEECH, ActivityLevel.LOUD] or
            (time.time() - self.last_speech_time) < 1.0
        )
    
    def has_been_silent(self, duration: float) -> bool:
        """Check if there has been silence for specified duration"""
        return (time.time() - self.last_speech_time) > duration


class ConversationStateManager:
    """Manages conversation state and context"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + ".ConversationStateManager")
        self.config = config.get("conversation_state", {})
        
        # Conversation parameters
        self.max_turn_gap = self.config.get("max_turn_gap", 30.0)  # seconds
        self.max_conversation_duration = self.config.get("max_conversation_duration", 300.0)  # 5 minutes
        self.followup_timeout = self.config.get("followup_timeout", 10.0)  # seconds
        
        # State tracking
        self.conversation_start_time = 0
        self.last_turn_time = 0
        self.conversation_active = False
        self.expecting_followup = False
        self.turn_count = 0
        
    def start_conversation(self):
        """Start a new conversation"""
        current_time = time.time()
        self.conversation_start_time = current_time
        self.last_turn_time = current_time
        self.conversation_active = True
        self.expecting_followup = False
        self.turn_count = 0
        self.logger.info("Conversation started")
    
    def add_turn(self, requires_followup: bool = False):
        """Add a conversation turn"""
        self.last_turn_time = time.time()
        self.turn_count += 1
        self.expecting_followup = requires_followup
        self.logger.debug(f"Conversation turn {self.turn_count} added")
    
    def end_conversation(self):
        """End the current conversation"""
        if self.conversation_active:
            duration = time.time() - self.conversation_start_time
            self.logger.info(f"Conversation ended after {duration:.1f}s with {self.turn_count} turns")
        
        self.conversation_active = False
        self.expecting_followup = False
        self.turn_count = 0
    
    def should_continue_conversation(self) -> bool:
        """Check if conversation should continue"""
        if not self.conversation_active:
            return False
        
        current_time = time.time()
        
        # Check turn gap timeout
        if (current_time - self.last_turn_time) > self.max_turn_gap:
            return False
        
        # Check overall conversation duration
        if (current_time - self.conversation_start_time) > self.max_conversation_duration:
            return False
        
        return True
    
    def is_expecting_followup(self) -> bool:
        """Check if expecting a followup response"""
        if not self.expecting_followup:
            return False
        
        # Check followup timeout
        return (time.time() - self.last_turn_time) < self.followup_timeout


class ContinuousListeningController(QObject):
    """Advanced continuous listening with intelligent state management"""
    
    # Signals
    state_changed = pyqtSignal(str)  # New listening state
    activity_detected = pyqtSignal(str, float)  # Activity level, audio level
    session_started = pyqtSignal(str)  # Session ID
    session_ended = pyqtSignal(str, dict)  # Session ID, stats
    conversation_started = pyqtSignal()
    conversation_ended = pyqtSignal(int)  # Turn count
    listening_timeout = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any], voice_controller: VoiceController):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("continuous_listening", {})
        
        # Core components
        self.voice_controller = voice_controller
        self.activity_monitor = VoiceActivityMonitor(config)
        self.conversation_manager = ConversationStateManager(config)
        
        # State management
        self.current_state = ListeningState.IDLE
        self.current_session: Optional[ListeningSession] = None
        
        # Configuration
        self.auto_timeout = self.config.get("auto_timeout", 30.0)  # seconds
        self.response_timeout = self.config.get("response_timeout", 10.0)  # seconds
        self.wake_word_cooldown = self.config.get("wake_word_cooldown", 2.0)  # seconds
        self.max_continuous_duration = self.config.get("max_continuous_duration", 600.0)  # 10 minutes
        
        # Timers
        self.timeout_timer = QTimer()
        self.timeout_timer.timeout.connect(self._handle_timeout)
        
        self.activity_timer = QTimer()
        self.activity_timer.timeout.connect(self._check_activity)
        
        # Connect voice controller signals
        self._connect_voice_signals()
        
        # Statistics
        self.stats = {
            "sessions_started": 0,
            "total_wake_words": 0,
            "total_commands": 0,
            "total_conversations": 0,
            "average_session_duration": 0.0,
            "error_count": 0
        }
    
    def _connect_voice_signals(self):
        """Connect to voice controller signals"""
        self.voice_controller.wake_word_detected.connect(self._on_wake_word_detected)
        self.voice_controller.speech_recognized.connect(self._on_speech_recognized)
        self.voice_controller.command_parsed.connect(self._on_command_parsed)
        self.voice_controller.speaking_changed.connect(self._on_speaking_changed)
        self.voice_controller.volume_changed.connect(self._on_audio_level_changed)
        self.voice_controller.error_occurred.connect(self._on_voice_error)
    
    def start_continuous_listening(self) -> str:
        """Start continuous listening mode"""
        try:
            if self.current_session:
                self.logger.warning("Continuous listening already active")
                return self.current_session.session_id
            
            # Create new session
            session_id = f"session_{int(time.time())}_{id(self)}"
            self.current_session = ListeningSession(
                session_id=session_id,
                start_time=time.time(),
                state=ListeningState.WAKE_WORD_LISTENING,
                last_activity_time=time.time()
            )
            
            self.logger.info(f"Starting continuous listening session: {session_id}")
            
            # Start voice pipeline
            self.voice_controller.start_full_voice_pipeline()
            
            # Start activity monitoring
            self.activity_timer.start(100)  # Check activity every 100ms
            
            # Set initial state
            self._set_state(ListeningState.WAKE_WORD_LISTENING)
            
            # Start timeout timer
            self._start_timeout_timer(self.max_continuous_duration)
            
            # Update stats
            self.stats["sessions_started"] += 1
            
            # Emit signal
            self.session_started.emit(session_id)
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start continuous listening: {e}")
            self.error_occurred.emit(f"Start failed: {e}")
            return ""
    
    def stop_continuous_listening(self):
        """Stop continuous listening mode"""
        try:
            if not self.current_session:
                return
            
            session_id = self.current_session.session_id
            self.logger.info(f"Stopping continuous listening session: {session_id}")
            
            # Stop timers
            self.timeout_timer.stop()
            self.activity_timer.stop()
            
            # Stop voice pipeline
            self.voice_controller.stop_full_voice_pipeline()
            
            # End conversation if active
            if self.conversation_manager.conversation_active:
                self.conversation_manager.end_conversation()
                self.conversation_ended.emit(self.conversation_manager.turn_count)
            
            # Calculate session stats
            session_duration = time.time() - self.current_session.start_time
            session_stats = {
                "duration": session_duration,
                "wake_words": self.current_session.total_wake_words,
                "commands": self.current_session.total_commands,
                "errors": self.current_session.total_errors,
                "conversation_turns": self.current_session.current_conversation_turns
            }
            
            # Update overall stats
            total_sessions = self.stats["sessions_started"]
            total_duration = self.stats["average_session_duration"] * (total_sessions - 1)
            self.stats["average_session_duration"] = (total_duration + session_duration) / total_sessions
            
            # Set final state
            self._set_state(ListeningState.IDLE)
            
            # Emit signal
            self.session_ended.emit(session_id, session_stats)
            
            # Clear session
            self.current_session = None
            
        except Exception as e:
            self.logger.error(f"Failed to stop continuous listening: {e}")
            self.error_occurred.emit(f"Stop failed: {e}")
    
    def _set_state(self, new_state: ListeningState):
        """Set listening state"""
        if self.current_state != new_state:
            self.logger.debug(f"State change: {self.current_state.value} → {new_state.value}")
            self.current_state = new_state
            
            if self.current_session:
                self.current_session.state = new_state
            
            self.state_changed.emit(new_state.value)
    
    def _start_timeout_timer(self, timeout_seconds: float):
        """Start timeout timer"""
        self.timeout_timer.stop()
        self.timeout_timer.start(int(timeout_seconds * 1000))
    
    def _check_activity(self):
        """Check voice activity and update state accordingly"""
        try:
            if not self.current_session:
                return
            
            # Get current audio level from voice controller
            # Note: This would be updated by the volume_changed signal
            
            # Check conversation state
            if self.conversation_manager.conversation_active:
                if not self.conversation_manager.should_continue_conversation():
                    self.logger.info("Conversation timeout - ending conversation")
                    self.conversation_manager.end_conversation()
                    self.conversation_ended.emit(self.conversation_manager.turn_count)
                    self._set_state(ListeningState.WAKE_WORD_LISTENING)
            
            # Update activity time
            if self.activity_monitor.is_speech_active():
                self.current_session.last_activity_time = time.time()
                
        except Exception as e:
            self.logger.error(f"Activity check failed: {e}")
    
    def _handle_timeout(self):
        """Handle timeout events"""
        try:
            self.logger.info("Continuous listening timeout")
            self.listening_timeout.emit()
            self.stop_continuous_listening()
            
        except Exception as e:
            self.logger.error(f"Timeout handling failed: {e}")
    
    def _on_wake_word_detected(self, phrase: str, confidence: float):
        """Handle wake word detection"""
        try:
            if not self.current_session:
                return
            
            self.logger.info(f"Wake word in continuous mode: {phrase} ({confidence:.3f})")
            
            # Update session stats
            self.current_session.total_wake_words += 1
            self.stats["total_wake_words"] += 1
            
            # Start conversation if not active
            if not self.conversation_manager.conversation_active:
                self.conversation_manager.start_conversation()
                self.conversation_started.emit()
                self.stats["total_conversations"] += 1
            
            # Set state to command listening
            self._set_state(ListeningState.COMMAND_LISTENING)
            
            # Start response timeout
            self._start_timeout_timer(self.response_timeout)
            
        except Exception as e:
            self.logger.error(f"Wake word handling failed: {e}")
            self._on_error(f"Wake word handling: {e}")
    
    def _on_speech_recognized(self, text: str, language: str):
        """Handle speech recognition"""
        try:
            if not self.current_session:
                return
            
            if self.current_state == ListeningState.COMMAND_LISTENING:
                self.logger.info(f"Speech recognized in continuous mode: '{text}'")
                self._set_state(ListeningState.PROCESSING)
                
                # Update session stats
                self.current_session.total_commands += 1
                self.stats["total_commands"] += 1
                
        except Exception as e:
            self.logger.error(f"Speech recognition handling failed: {e}")
            self._on_error(f"Speech recognition: {e}")
    
    def _on_command_parsed(self, command_data: dict):
        """Handle parsed command"""
        try:
            if not self.current_session:
                return
            
            # Add conversation turn
            requires_followup = command_data.get("requires_response", False)
            self.conversation_manager.add_turn(requires_followup)
            
            self.current_session.current_conversation_turns += 1
            
            # Determine next state based on command type
            if requires_followup or self.conversation_manager.is_expecting_followup():
                # Stay in conversation mode, return to command listening
                self._set_state(ListeningState.COMMAND_LISTENING)
                self._start_timeout_timer(self.auto_timeout)
            else:
                # Return to wake word listening
                self._set_state(ListeningState.WAKE_WORD_LISTENING)
                self._start_timeout_timer(self.max_continuous_duration)
            
        except Exception as e:
            self.logger.error(f"Command parsing handling failed: {e}")
            self._on_error(f"Command parsing: {e}")
    
    def _on_speaking_changed(self, is_speaking: bool):
        """Handle speaking state change"""
        try:
            if not self.current_session:
                return
            
            if is_speaking:
                self._set_state(ListeningState.RESPONDING)
            else:
                # Return to appropriate listening state after speaking
                if self.conversation_manager.conversation_active:
                    if self.conversation_manager.is_expecting_followup():
                        self._set_state(ListeningState.COMMAND_LISTENING)
                        self._start_timeout_timer(self.auto_timeout)
                    else:
                        self._set_state(ListeningState.WAKE_WORD_LISTENING)
                        self._start_timeout_timer(self.max_continuous_duration)
                else:
                    self._set_state(ListeningState.WAKE_WORD_LISTENING)
                    self._start_timeout_timer(self.max_continuous_duration)
            
        except Exception as e:
            self.logger.error(f"Speaking state handling failed: {e}")
    
    def _on_audio_level_changed(self, level: float):
        """Handle audio level change"""
        try:
            # Update activity monitor
            activity_level = self.activity_monitor.update_activity(level)
            
            # Emit activity signal
            self.activity_detected.emit(activity_level.value, level)
            
        except Exception as e:
            self.logger.error(f"Audio level handling failed: {e}")
    
    def _on_voice_error(self, error_msg: str):
        """Handle voice controller error"""
        self._on_error(f"Voice controller: {error_msg}")
    
    def _on_error(self, error_msg: str):
        """Handle errors in continuous listening"""
        try:
            self.logger.error(f"Continuous listening error: {error_msg}")
            
            if self.current_session:
                self.current_session.total_errors += 1
                self.stats["error_count"] += 1
            
            # Set error state temporarily
            self._set_state(ListeningState.ERROR)
            
            # Emit error signal
            self.error_occurred.emit(error_msg)
            
            # Recovery: return to wake word listening after brief delay
            QTimer.singleShot(2000, lambda: self._set_state(ListeningState.WAKE_WORD_LISTENING))
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
    
    def get_session_status(self) -> Dict[str, Any]:
        """Get current session status"""
        if not self.current_session:
            return {"active": False}
        
        current_time = time.time()
        return {
            "active": True,
            "session_id": self.current_session.session_id,
            "state": self.current_state.value,
            "duration": current_time - self.current_session.start_time,
            "wake_words": self.current_session.total_wake_words,
            "commands": self.current_session.total_commands,
            "errors": self.current_session.total_errors,
            "conversation_active": self.conversation_manager.conversation_active,
            "conversation_turns": self.current_session.current_conversation_turns,
            "last_activity": current_time - self.current_session.last_activity_time,
            "activity_level": self.activity_monitor.current_level.value
        }
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get overall statistics"""
        return {
            "session_stats": self.stats.copy(),
            "current_session": self.get_session_status(),
            "configuration": {
                "auto_timeout": self.auto_timeout,
                "response_timeout": self.response_timeout,
                "max_duration": self.max_continuous_duration,
                "wake_word_cooldown": self.wake_word_cooldown
            }
        }
    
    def set_timeout_configuration(self, auto_timeout: float = None, response_timeout: float = None, max_duration: float = None):
        """Update timeout configuration"""
        try:
            if auto_timeout is not None:
                self.auto_timeout = auto_timeout
                self.logger.info(f"Auto timeout set to {auto_timeout}s")
            
            if response_timeout is not None:
                self.response_timeout = response_timeout
                self.logger.info(f"Response timeout set to {response_timeout}s")
            
            if max_duration is not None:
                self.max_continuous_duration = max_duration
                self.logger.info(f"Max duration set to {max_duration}s")
                
        except Exception as e:
            self.logger.error(f"Configuration update failed: {e}")
    
    def force_state_change(self, new_state: str):
        """Force state change (for debugging/testing)"""
        try:
            state_enum = ListeningState(new_state)
            self._set_state(state_enum)
            self.logger.info(f"Forced state change to: {new_state}")
            
        except ValueError:
            self.logger.error(f"Invalid state: {new_state}")
        except Exception as e:
            self.logger.error(f"Force state change failed: {e}")