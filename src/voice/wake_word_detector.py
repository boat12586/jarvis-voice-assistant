"""
Wake Word Detection System for JARVIS Voice Assistant
Detects "Hey JARVIS" wake phrase with both English and Thai variants
"""

import logging
import numpy as np
import threading
import time
from typing import Dict, Any, List, Optional, Callable
from PyQt6.QtCore import QObject, pyqtSignal, QTimer, QThread
import torch
import torchaudio
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import pyaudio
import webrtcvad
from scipy import signal
import queue

class WakeWordDetector(QObject):
    """Wake word detection system for JARVIS"""
    
    # Signals
    wake_word_detected = pyqtSignal(str, float)  # wake_phrase, confidence
    listening_started = pyqtSignal()
    listening_stopped = pyqtSignal()
    audio_level_changed = pyqtSignal(float)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("wake_word", {})
        
        # Wake word configuration
        self.wake_phrases = [
            "hey jarvis",
            "hi jarvis", 
            "jarvis",
            "เฮ้ จาร์วิส",  # Thai: Hey JARVIS
            "สวัสดี จาร์วิส",  # Thai: Hello JARVIS
            "จาร์วิส"  # Thai: JARVIS
        ]
        
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.sample_rate = self.config.get("sample_rate", 16000)
        self.chunk_size = self.config.get("chunk_size", 1024)
        self.audio_timeout = self.config.get("audio_timeout", 5.0)
        
        # Audio processing
        self.audio_queue = queue.Queue()
        self.is_listening = False
        self.audio_thread = None
        self.processing_thread = None
        
        # Voice Activity Detection
        self.vad = webrtcvad.Vad(2)  # Aggressiveness level 2
        self.vad_buffer = []
        self.speech_frames = 0
        self.silence_frames = 0
        
        # Audio components
        self.pyaudio_instance = None
        self.stream = None
        
        # Wake word models (lightweight approach)
        self.wake_word_templates = {}
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize wake word detection system"""
        try:
            self.logger.info("Initializing wake word detection...")
            
            # Initialize audio
            self._initialize_audio()
            
            # Create wake word templates
            self._create_wake_word_templates()
            
            self.logger.info("Wake word detection initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize wake word detection: {e}")
            self.error_occurred.emit(f"Wake word init failed: {e}")
    
    def _initialize_audio(self):
        """Initialize audio input"""
        try:
            self.pyaudio_instance = pyaudio.PyAudio()
            
            # Find default input device
            device_info = self.pyaudio_instance.get_default_input_device_info()
            self.logger.info(f"Using audio device: {device_info['name']}")
            
        except Exception as e:
            self.logger.error(f"Audio initialization failed: {e}")
            raise
    
    def _create_wake_word_templates(self):
        """Create wake word templates for pattern matching"""
        try:
            # Simple phonetic patterns for wake words
            self.wake_word_templates = {
                "hey jarvis": {
                    "pattern": ["hey", "jarvis"],
                    "phonetic": ["heɪ", "ˈdʒɑːrvɪs"],
                    "keywords": ["hey", "hi", "jarvis"],
                    "threshold": 0.7
                },
                "jarvis": {
                    "pattern": ["jarvis"],
                    "phonetic": ["ˈdʒɑːrvɪs"],
                    "keywords": ["jarvis"],
                    "threshold": 0.8
                },
                "thai_jarvis": {
                    "pattern": ["จาร์วิส"],
                    "phonetic": ["tʃaːwis"],
                    "keywords": ["จาร์วิส", "jarvis"],
                    "threshold": 0.7
                }
            }
            
            self.logger.info(f"Created {len(self.wake_word_templates)} wake word templates")
            
        except Exception as e:
            self.logger.error(f"Template creation failed: {e}")
    
    def start_listening(self):
        """Start wake word detection"""
        if self.is_listening:
            return
            
        try:
            self.logger.info("Starting wake word detection...")
            
            # Start audio stream
            self.stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=self.sample_rate,
                input=True,
                frames_per_buffer=self.chunk_size,
                stream_callback=self._audio_callback
            )
            
            self.stream.start_stream()
            self.is_listening = True
            
            # Start processing thread
            self.processing_thread = threading.Thread(target=self._process_audio_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            
            self.listening_started.emit()
            self.logger.info("Wake word detection started")
            
        except Exception as e:
            self.logger.error(f"Failed to start listening: {e}")
            self.error_occurred.emit(f"Start listening failed: {e}")
    
    def stop_listening(self):
        """Stop wake word detection"""
        if not self.is_listening:
            return
            
        try:
            self.is_listening = False
            
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
            
            self.listening_stopped.emit()
            self.logger.info("Wake word detection stopped")
            
        except Exception as e:
            self.logger.error(f"Failed to stop listening: {e}")
    
    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Audio stream callback"""
        try:
            if self.is_listening:
                # Convert to numpy array
                audio_data = np.frombuffer(in_data, dtype=np.int16)
                
                # Add to queue for processing
                self.audio_queue.put(audio_data)
                
                # Calculate audio level
                audio_level = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))
                self.audio_level_changed.emit(float(audio_level))
            
            return (in_data, pyaudio.paContinue)
            
        except Exception as e:
            self.logger.error(f"Audio callback error: {e}")
            return (in_data, pyaudio.paAbort)
    
    def _process_audio_loop(self):
        """Main audio processing loop"""
        self.logger.info("Audio processing loop started")
        
        audio_buffer = []
        last_detection_time = 0
        
        while self.is_listening:
            try:
                # Get audio data from queue
                try:
                    audio_chunk = self.audio_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                
                # Add to buffer
                audio_buffer.extend(audio_chunk)
                
                # Process when buffer is large enough
                if len(audio_buffer) >= self.sample_rate * 2:  # 2 seconds of audio
                    audio_array = np.array(audio_buffer, dtype=np.float32)
                    
                    # Voice Activity Detection
                    if self._detect_voice_activity(audio_array):
                        # Check for wake word
                        wake_result = self._detect_wake_word(audio_array)
                        
                        if wake_result and wake_result['confidence'] > self.confidence_threshold:
                            current_time = time.time()
                            
                            # Prevent duplicate detections
                            if current_time - last_detection_time > 2.0:
                                self.logger.info(f"Wake word detected: {wake_result['phrase']} "
                                               f"(confidence: {wake_result['confidence']:.2f})")
                                
                                self.wake_word_detected.emit(
                                    wake_result['phrase'], 
                                    wake_result['confidence']
                                )
                                
                                last_detection_time = current_time
                    
                    # Keep only last 1 second in buffer
                    audio_buffer = audio_buffer[-self.sample_rate:]
                
            except Exception as e:
                self.logger.error(f"Audio processing error: {e}")
                time.sleep(0.1)
        
        self.logger.info("Audio processing loop stopped")
    
    def _detect_voice_activity(self, audio_data: np.ndarray) -> bool:
        """Detect if audio contains speech"""
        try:
            # Convert to the format expected by WebRTC VAD
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # WebRTC VAD expects specific frame sizes
            frame_duration = 30  # ms
            frame_size = int(self.sample_rate * frame_duration / 1000)
            
            speech_frames = 0
            total_frames = 0
            
            for i in range(0, len(audio_int16) - frame_size, frame_size):
                frame = audio_int16[i:i + frame_size]
                
                try:
                    is_speech = self.vad.is_speech(frame.tobytes(), self.sample_rate)
                    if is_speech:
                        speech_frames += 1
                    total_frames += 1
                except:
                    continue
            
            if total_frames == 0:
                return False
                
            speech_ratio = speech_frames / total_frames
            return speech_ratio > 0.3  # At least 30% speech
            
        except Exception as e:
            self.logger.error(f"Voice activity detection error: {e}")
            return False
    
    def _detect_wake_word(self, audio_data: np.ndarray) -> Optional[Dict[str, Any]]:
        """Detect wake word in audio data"""
        try:
            # Simple approach: use speech recognition for wake word detection
            from .speech_recognizer import SpeechRecognizer
            
            # Create temporary speech recognizer
            recognizer = SpeechRecognizer(self.config)
            
            # Convert audio to format expected by recognizer
            audio_bytes = (audio_data * 32767).astype(np.int16).tobytes()
            
            # Recognize speech
            result = recognizer._recognize_audio_data(audio_bytes, self.sample_rate)
            
            if result and 'text' in result:
                text = result['text'].lower().strip()
                
                # Check against wake phrases
                for phrase in self.wake_phrases:
                    if self._match_wake_phrase(text, phrase):
                        confidence = self._calculate_confidence(text, phrase)
                        
                        return {
                            'phrase': phrase,
                            'recognized_text': text,
                            'confidence': confidence,
                            'language': result.get('language', 'en')
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Wake word detection error: {e}")
            return None
    
    def _match_wake_phrase(self, text: str, wake_phrase: str) -> bool:
        """Check if text matches wake phrase"""
        try:
            # Direct match
            if wake_phrase in text:
                return True
            
            # Fuzzy matching
            words = text.split()
            wake_words = wake_phrase.split()
            
            # Check if all wake words are present
            matches = 0
            for wake_word in wake_words:
                for word in words:
                    if self._words_similar(word, wake_word):
                        matches += 1
                        break
            
            return matches >= len(wake_words)
            
        except Exception:
            return False
    
    def _words_similar(self, word1: str, word2: str) -> bool:
        """Check if two words are similar"""
        try:
            # Simple similarity check
            if word1 == word2:
                return True
            
            # Check if one contains the other
            if word1 in word2 or word2 in word1:
                return True
            
            # Levenshtein distance check
            distance = self._levenshtein_distance(word1, word2)
            max_len = max(len(word1), len(word2))
            
            if max_len == 0:
                return True
            
            similarity = 1 - (distance / max_len)
            return similarity > 0.7
            
        except Exception:
            return False
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)

        if len(s2) == 0:
            return len(s1)

        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _calculate_confidence(self, text: str, wake_phrase: str) -> float:
        """Calculate confidence score for wake word detection"""
        try:
            # Base confidence
            confidence = 0.5
            
            # Exact match bonus
            if wake_phrase in text:
                confidence += 0.3
            
            # Word coverage bonus
            words = text.split()
            wake_words = wake_phrase.split()
            
            word_matches = 0
            for wake_word in wake_words:
                for word in words:
                    if self._words_similar(word, wake_word):
                        word_matches += 1
                        break
            
            word_coverage = word_matches / len(wake_words)
            confidence += word_coverage * 0.2
            
            # Length penalty (too much extra text reduces confidence)
            if len(words) > len(wake_words) * 2:
                confidence -= 0.1
            
            return min(1.0, max(0.0, confidence))
            
        except Exception:
            return 0.5
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {
            "is_listening": self.is_listening,
            "wake_phrases": self.wake_phrases,
            "confidence_threshold": self.confidence_threshold,
            "sample_rate": self.sample_rate,
            "audio_queue_size": self.audio_queue.qsize() if hasattr(self, 'audio_queue') else 0,
            "templates_loaded": len(self.wake_word_templates)
        }
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.stop_listening()
            
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()
            
            self.logger.info("Wake word detector cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")


class WakeWordConfig:
    """Configuration helper for wake word detection"""
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default wake word configuration"""
        return {
            "wake_word": {
                "confidence_threshold": 0.7,
                "sample_rate": 16000,
                "chunk_size": 1024,
                "audio_timeout": 5.0,
                "enabled": True,
                "wake_phrases": [
                    "hey jarvis",
                    "hi jarvis", 
                    "jarvis",
                    "เฮ้ จาร์วิส",
                    "สวัสดี จาร์วิส",
                    "จาร์วิส"
                ]
            }
        }
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate wake word configuration"""
        try:
            wake_config = config.get("wake_word", {})
            
            required_fields = ["confidence_threshold", "sample_rate", "chunk_size"]
            for field in required_fields:
                if field not in wake_config:
                    return False
            
            # Validate ranges
            if not (0.1 <= wake_config["confidence_threshold"] <= 1.0):
                return False
            
            if wake_config["sample_rate"] not in [8000, 16000, 22050, 44100]:
                return False
            
            return True
            
        except Exception:
            return False