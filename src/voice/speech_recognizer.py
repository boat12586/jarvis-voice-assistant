#!/usr/bin/env python3
"""
🎤 Simple Speech Recognizer for JARVIS
ใช้ faster-whisper สำหรับการรู้จำเสียง
"""

import logging
import numpy as np
from typing import Optional, Tuple
import threading
import queue
import time

try:
    import sounddevice as sd
    from faster_whisper import WhisperModel
    AUDIO_AVAILABLE = True
except ImportError as e:
    AUDIO_AVAILABLE = False
    logging.warning(f"Audio libraries not available: {e}")

class SimpleSpeechRecognizer:
    """Simple speech recognizer using faster-whisper"""
    
    def __init__(self, model_size: str = "tiny", device: str = "cpu"):
        self.logger = logging.getLogger(__name__)
        self.model_size = model_size
        self.device = device
        self.model = None
        self.is_listening = False
        self.audio_queue = queue.Queue()
        
        if AUDIO_AVAILABLE:
            self._initialize_model()
        else:
            self.logger.warning("🔇 Audio not available - Speech recognition disabled")
    
    def _initialize_model(self):
        """Initialize Whisper model"""
        try:
            self.logger.info(f"🧠 Loading Whisper {self.model_size} model...")
            self.model = WhisperModel(
                self.model_size, 
                device=self.device,
                compute_type="int8"
            )
            self.logger.info("✅ Speech recognition ready")
        except Exception as e:
            self.logger.error(f"❌ Failed to load Whisper model: {e}")
            self.model = None
    
    def recognize_from_mic(self, duration: float = 5.0) -> Optional[str]:
        """Record from microphone and recognize speech"""
        if not self.model or not AUDIO_AVAILABLE:
            return "Speech recognition not available"
        
        try:
            self.logger.info(f"🎤 Recording for {duration} seconds...")
            
            # Record audio
            sample_rate = 16000
            recording = sd.rec(
                int(duration * sample_rate),
                samplerate=sample_rate,
                channels=1,
                dtype=np.float32
            )
            sd.wait()  # Wait for recording to complete
            
            # Convert to the format expected by Whisper
            audio_data = recording.flatten()
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio_data,
                beam_size=5,
                language="th"  # Thai language
            )
            
            # Extract text
            text = " ".join([segment.text for segment in segments]).strip()
            
            if text:
                self.logger.info(f"🎯 Recognized: '{text}'")
                return text
            else:
                self.logger.info("🔇 No speech detected")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Speech recognition failed: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if speech recognition is available"""
        return self.model is not None and AUDIO_AVAILABLE
    
    def get_audio_devices(self):
        """Get available audio devices"""
        if not AUDIO_AVAILABLE:
            return []
        
        try:
            devices = sd.query_devices()
            return devices
        except Exception as e:
            self.logger.error(f"Failed to query audio devices: {e}")
            return []
