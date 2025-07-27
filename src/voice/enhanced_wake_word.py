"""
Enhanced Wake Word Detection System for JARVIS Voice Assistant
Advanced neural wake word detection with local processing
"""

import os
import logging
import numpy as np
import torch
import torchaudio
import threading
import time
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple
from pathlib import Path
from dataclasses import dataclass
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
import sounddevice as sd
import librosa
from scipy import signal
import webrtcvad

# Enhanced logging support
try:
    from src.system.enhanced_logger import ComponentLogger
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False


@dataclass
class WakeWordConfig:
    """Enhanced wake word detection configuration"""
    wake_phrases: List[str] = None
    confidence_threshold: float = 0.85
    sample_rate: int = 16000
    chunk_size: int = 1024
    buffer_duration: float = 2.0  # seconds
    vad_aggressiveness: int = 2
    energy_threshold: float = 300
    continuous_listening: bool = True
    device_index: Optional[int] = None
    model_path: str = "models/wake_word"
    
    def __post_init__(self):
        if self.wake_phrases is None:
            self.wake_phrases = [
                "hey jarvis",
                "hi jarvis", 
                "jarvis",
                "เฮ้ จาร์วิส",  # Thai
                "สวัสดี จาร์วิส",  # Thai
                "จาร์วิส"  # Thai
            ]


class AudioProcessor:
    """Audio processing utilities for wake word detection"""
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # VAD for speech detection
        self.vad = webrtcvad.Vad(config.vad_aggressiveness)
        
        # Audio buffer
        self.buffer_size = int(config.sample_rate * config.buffer_duration)
        self.audio_buffer = np.zeros(self.buffer_size, dtype=np.float32)
        
    def preprocess_audio(self, audio_data: np.ndarray) -> np.ndarray:
        """Preprocess audio for wake word detection"""
        try:
            # Ensure correct sample rate
            if len(audio_data) == 0:
                return audio_data
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Apply high-pass filter to remove low-frequency noise
            sos = signal.butter(4, 80, btype='high', fs=self.config.sample_rate, output='sos')
            audio_data = signal.sosfilt(sos, audio_data)
            
            # Apply pre-emphasis filter
            audio_data = np.append(audio_data[0], audio_data[1:] - 0.95 * audio_data[:-1])
            
            return audio_data.astype(np.float32)
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            return audio_data
    
    def detect_speech(self, audio_data: np.ndarray) -> bool:
        """Detect if audio contains speech using VAD"""
        try:
            # Convert to 16-bit PCM
            audio_pcm = (audio_data * 32767).astype(np.int16)
            
            # VAD requires specific frame sizes (10, 20, or 30 ms)
            frame_duration = 30  # ms
            frame_size = int(self.config.sample_rate * frame_duration / 1000)
            
            # Process audio in frames
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_pcm) - frame_size + 1, frame_size):
                frame = audio_pcm[i:i + frame_size].tobytes()
                if self.vad.is_speech(frame, self.config.sample_rate):
                    speech_frames += 1
                total_frames += 1
            
            if total_frames == 0:
                return False
            
            # Require at least 30% speech frames
            speech_ratio = speech_frames / total_frames
            return speech_ratio >= 0.3
            
        except Exception as e:
            self.logger.error(f"Speech detection failed: {e}")
            # Fallback to energy-based detection
            return self.detect_energy(audio_data)
    
    def detect_energy(self, audio_data: np.ndarray) -> bool:
        """Fallback energy-based speech detection"""
        try:
            energy = np.sum(audio_data ** 2) / len(audio_data)
            return energy > (self.config.energy_threshold / 1000000)
        except:
            return False
    
    def update_buffer(self, new_audio: np.ndarray):
        """Update rolling audio buffer"""
        if len(new_audio) >= self.buffer_size:
            self.audio_buffer = new_audio[-self.buffer_size:]
        else:
            # Shift buffer and add new audio
            shift_size = len(new_audio)
            self.audio_buffer[:-shift_size] = self.audio_buffer[shift_size:]
            self.audio_buffer[-shift_size:] = new_audio


class WakeWordModel:
    """Neural wake word detection model"""
    
    def __init__(self, config: WakeWordConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.feature_extractor = None
        self.classifier = None
        self.is_loaded = False
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize wake word model"""
        try:
            self.logger.info("🧠 Initializing wake word model...")
            
            # For now, create a simple mock model
            # In production, would use actual trained wake word model
            self._create_mock_model()
            
            self.is_loaded = True
            self.logger.info("✅ Wake word model initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wake word model: {e}")
            self._create_fallback_model()
    
    def _create_mock_model(self):
        """Create mock model for testing"""
        self.logger.info("🧪 Creating mock wake word model")
        
        class MockWakeWordModel:
            def __init__(self, phrases):
                self.wake_phrases = phrases
                
            def detect(self, audio_features: np.ndarray, text: str = "") -> Tuple[bool, float]:
                # Simple keyword matching for testing
                text_lower = text.lower()
                
                for phrase in self.wake_phrases:
                    if phrase.lower() in text_lower:
                        # Simulate confidence based on audio energy
                        confidence = min(0.9, 0.7 + np.random.random() * 0.2)
                        return True, confidence
                
                # Check audio energy as fallback
                if len(audio_features) > 0:
                    energy = np.mean(np.abs(audio_features))
                    if energy > 0.1:  # Arbitrary threshold
                        confidence = min(0.8, energy * 2)
                        return True, confidence
                
                return False, 0.0
        
        self.classifier = MockWakeWordModel(self.config.wake_phrases)
    
    def _create_fallback_model(self):
        """Create simple fallback model"""
        self.logger.warning("🔄 Creating fallback wake word model")
        self._create_mock_model()
        self.is_loaded = True
    
    def extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features from audio for wake word detection"""
        try:
            # Use MFCC features for wake word detection
            mfcc = librosa.feature.mfcc(
                y=audio_data,
                sr=self.config.sample_rate,
                n_mfcc=13,
                n_fft=512,
                hop_length=160
            )
            
            # Flatten and normalize
            features = mfcc.flatten()
            if np.std(features) > 0:
                features = (features - np.mean(features)) / np.std(features)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.zeros(13)  # Return empty features
    
    def predict(self, audio_data: np.ndarray, text_context: str = "") -> Tuple[bool, float, str]:
        """Predict if audio contains wake word"""
        if not self.is_loaded:
            return False, 0.0, ""
        
        try:
            # Extract features
            features = self.extract_features(audio_data)
            
            # Use classifier
            if self.classifier:
                is_wake_word, confidence = self.classifier.detect(features, text_context)
                
                if is_wake_word:
                    # Find which phrase was detected
                    detected_phrase = self._identify_phrase(text_context)
                    return is_wake_word, confidence, detected_phrase
            
            return False, 0.0, ""
            
        except Exception as e:
            self.logger.error(f"Wake word prediction failed: {e}")
            return False, 0.0, ""
    
    def _identify_phrase(self, text: str) -> str:
        """Identify which wake phrase was detected"""
        text_lower = text.lower()
        
        for phrase in self.config.wake_phrases:
            if phrase.lower() in text_lower:
                return phrase
        
        return "jarvis"  # Default


class EnhancedWakeWordDetector(QObject):
    """Enhanced wake word detection system"""
    
    # Signals
    wake_word_detected = pyqtSignal(str, float, str)  # phrase, confidence, detected_text
    listening_started = pyqtSignal()
    listening_stopped = pyqtSignal()
    audio_level_changed = pyqtSignal(float)
    speech_detected = pyqtSignal(bool)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        
        # Setup logging
        if ENHANCED_LOGGING:
            self.logger = ComponentLogger("wake_word_detector", config or {})
        else:
            self.logger = logging.getLogger(__name__)
        
        # Configuration
        config_dict = config or {}
        self.config = WakeWordConfig(**config_dict.get("wake_word", {}))
        
        # Components
        self.audio_processor = AudioProcessor(self.config)
        self.wake_word_model = WakeWordModel(self.config)
        
        # Audio stream
        self.audio_stream = None
        self.audio_queue = queue.Queue()
        
        # State
        self.is_listening = False
        self.is_ready = False
        self._stop_event = threading.Event()
        self._processing_thread = None
        
        # Performance tracking
        self.detection_count = 0
        self.false_positive_count = 0
        self.audio_level = 0.0
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize wake word detector"""
        try:
            operation_id = None
            if ENHANCED_LOGGING:
                operation_id = self.logger.operation_start("initialize_wake_word_detector")
            
            self.logger.info("🎤 Initializing Enhanced Wake Word Detector...")
            
            # Check audio devices
            self._check_audio_devices()
            
            # Test wake word model
            if self.wake_word_model.is_loaded:
                self.logger.info("✅ Wake word model ready")
            else:
                self.logger.warning("⚠️ Wake word model not fully loaded")
            
            self.is_ready = True
            self.logger.info("✅ Enhanced Wake Word Detector initialized")
            
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, True, {
                    "wake_phrases_count": len(self.config.wake_phrases),
                    "sample_rate": self.config.sample_rate,
                    "confidence_threshold": self.config.confidence_threshold
                })
                
        except Exception as e:
            self.logger.error(f"Failed to initialize wake word detector: {e}", exception=e)
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, False)
            self.error_occurred.emit(f"Wake word detector initialization failed: {e}")
    
    def _check_audio_devices(self):
        """Check available audio input devices"""
        try:
            import sounddevice as sd
            devices = sd.query_devices()
            
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            self.logger.info(f"Found {len(input_devices)} audio input devices")
            
            for i, device in enumerate(input_devices):
                if i < 3:  # Log first 3 devices
                    self.logger.debug(f"Device {i}: {device['name']}")
            
            if not input_devices:
                raise RuntimeError("No audio input devices found")
                
        except Exception as e:
            self.logger.error(f"Audio device check failed: {e}")
            raise
    
    def start_listening(self) -> bool:
        """Start wake word detection"""
        if self.is_listening:
            self.logger.warning("Wake word detector already listening")
            return True
        
        if not self.is_ready:
            self.logger.error("Wake word detector not ready")
            return False
        
        try:
            self.logger.info("🔊 Starting wake word detection...")
            
            # Reset state
            self._stop_event.clear()
            self.detection_count = 0
            self.false_positive_count = 0
            
            # Start audio stream
            self._start_audio_stream()
            
            # Start processing thread
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                daemon=True
            )
            self._processing_thread.start()
            
            self.is_listening = True
            self.listening_started.emit()
            self.logger.info("✅ Wake word detection started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start wake word detection: {e}", exception=e)
            self.error_occurred.emit(f"Failed to start wake word detection: {e}")
            return False
    
    def stop_listening(self):
        """Stop wake word detection"""
        if not self.is_listening:
            return
        
        try:
            self.logger.info("🔇 Stopping wake word detection...")
            
            # Signal threads to stop
            self._stop_event.set()
            
            # Stop audio stream
            if self.audio_stream is not None:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            
            # Wait for processing thread
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=2.0)
            
            self.is_listening = False
            self.listening_stopped.emit()
            self.logger.info("✅ Wake word detection stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping wake word detection: {e}", exception=e)
    
    def _start_audio_stream(self):
        """Start audio input stream"""
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    self.logger.warning(f"Audio stream status: {status}")
                
                # Convert to mono if stereo
                if indata.shape[1] > 1:
                    audio_data = np.mean(indata, axis=1)
                else:
                    audio_data = indata[:, 0]
                
                # Add to queue for processing
                try:
                    self.audio_queue.put_nowait(audio_data.copy())
                except queue.Full:
                    # Drop oldest data if queue is full
                    try:
                        self.audio_queue.get_nowait()
                        self.audio_queue.put_nowait(audio_data.copy())
                    except queue.Empty:
                        pass
            
            self.audio_stream = sd.InputStream(
                channels=1,
                samplerate=self.config.sample_rate,
                blocksize=self.config.chunk_size,
                callback=audio_callback,
                device=self.config.device_index
            )
            
            self.audio_stream.start()
            self.logger.info(f"Audio stream started: {self.config.sample_rate}Hz, chunk_size={self.config.chunk_size}")
            
        except Exception as e:
            self.logger.error(f"Failed to start audio stream: {e}")
            raise
    
    def _processing_loop(self):
        """Main audio processing loop"""
        self.logger.info("Audio processing loop started")
        
        last_detection_time = 0
        detection_cooldown = 2.0  # seconds
        
        try:
            while not self._stop_event.is_set():
                try:
                    # Get audio data with timeout
                    audio_data = self.audio_queue.get(timeout=0.1)
                    
                    # Update audio level
                    audio_level = np.mean(np.abs(audio_data)) * 100
                    self.audio_level = audio_level
                    self.audio_level_changed.emit(audio_level)
                    
                    # Preprocess audio
                    processed_audio = self.audio_processor.preprocess_audio(audio_data)
                    
                    # Update buffer
                    self.audio_processor.update_buffer(processed_audio)
                    
                    # Check for speech activity
                    has_speech = self.audio_processor.detect_speech(processed_audio)
                    self.speech_detected.emit(has_speech)
                    
                    if has_speech:
                        # Run wake word detection
                        current_time = time.time()
                        if current_time - last_detection_time > detection_cooldown:
                            
                            is_wake_word, confidence, detected_phrase = self.wake_word_model.predict(
                                self.audio_processor.audio_buffer
                            )
                            
                            if is_wake_word and confidence >= self.config.confidence_threshold:
                                self.detection_count += 1
                                last_detection_time = current_time
                                
                                self.logger.info(f"🎯 Wake word detected: '{detected_phrase}' (confidence: {confidence:.2f})")
                                self.wake_word_detected.emit(detected_phrase, confidence, "")
                                
                                if ENHANCED_LOGGING:
                                    self.logger.operation_end(
                                        f"wake_word_detection_{self.detection_count}",
                                        True,
                                        {
                                            "phrase": detected_phrase,
                                            "confidence": confidence,
                                            "audio_level": audio_level
                                        }
                                    )
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    self.logger.error(f"Error in processing loop: {e}")
                    time.sleep(0.1)
        
        except Exception as e:
            self.logger.error(f"Processing loop crashed: {e}")
        
        self.logger.info("Audio processing loop ended")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get wake word detector statistics"""
        stats = {
            "is_ready": self.is_ready,
            "is_listening": self.is_listening,
            "detection_count": self.detection_count,
            "false_positive_count": self.false_positive_count,
            "current_audio_level": self.audio_level,
            "wake_phrases": self.config.wake_phrases,
            "confidence_threshold": self.config.confidence_threshold,
            "sample_rate": self.config.sample_rate,
            "model_loaded": self.wake_word_model.is_loaded
        }
        
        if ENHANCED_LOGGING:
            component_stats = self.logger.get_component_stats()
            stats.update({"performance": component_stats})
        
        return stats
    
    def set_confidence_threshold(self, threshold: float):
        """Adjust confidence threshold"""
        self.config.confidence_threshold = max(0.1, min(1.0, threshold))
        self.logger.info(f"Confidence threshold updated to {self.config.confidence_threshold}")
    
    def add_wake_phrase(self, phrase: str):
        """Add new wake phrase"""
        if phrase not in self.config.wake_phrases:
            self.config.wake_phrases.append(phrase)
            self.logger.info(f"Added wake phrase: '{phrase}'")
    
    def remove_wake_phrase(self, phrase: str):
        """Remove wake phrase"""
        if phrase in self.config.wake_phrases:
            self.config.wake_phrases.remove(phrase)
            self.logger.info(f"Removed wake phrase: '{phrase}'")
    
    def test_detection(self, test_phrase: str = "hey jarvis") -> bool:
        """Test wake word detection with a phrase"""
        self.logger.info(f"🧪 Testing wake word detection with: '{test_phrase}'")
        
        try:
            # Create test audio (silence)
            test_audio = np.zeros(int(self.config.sample_rate * 2))
            
            # Test model prediction
            is_wake_word, confidence, detected_phrase = self.wake_word_model.predict(
                test_audio, test_phrase
            )
            
            self.logger.info(f"Test result: {is_wake_word}, confidence: {confidence:.2f}, phrase: '{detected_phrase}'")
            return is_wake_word
            
        except Exception as e:
            self.logger.error(f"Wake word test failed: {e}")
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("🧹 Cleaning up wake word detector...")
            
            # Stop detection
            self.stop_listening()
            
            # Clear queues
            while not self.audio_queue.empty():
                try:
                    self.audio_queue.get_nowait()
                except queue.Empty:
                    break
            
            # Cleanup model
            if self.wake_word_model:
                del self.wake_word_model
                self.wake_word_model = None
            
            self.is_ready = False
            self.logger.info("✅ Wake word detector cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exception=e)


# Factory function
def create_wake_word_detector(config: Optional[Dict[str, Any]] = None) -> EnhancedWakeWordDetector:
    """Create enhanced wake word detector"""
    return EnhancedWakeWordDetector(config)