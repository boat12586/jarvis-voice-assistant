#!/usr/bin/env python3
"""
🎙️ Simple Wake Word Detection for JARVIS
ระบบตรวจจับคำปลุกแบบง่ายที่ใช้ Faster-Whisper
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable
import re

try:
    import sounddevice as sd
    import numpy as np
    from faster_whisper import WhisperModel
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio libraries not available")


class SimpleWakeWordDetector:
    """ตัวตรวจจับคำปลุกแบบง่าย"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Wake word patterns
        self.wake_patterns = {
            'english': [
                r'\bhey\s+jarvis\b',
                r'\bhi\s+jarvis\b', 
                r'\bhello\s+jarvis\b',
                r'\bjarvis\b'
            ],
            'thai': [
                r'เฮ้\s*จาร์วิส',
                r'สวัสดี\s*จาร์วิส',
                r'หวัดดี\s*จาร์วิส',
                r'จาร์วิส'
            ]
        }
        
        # Audio settings
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.listen_duration = self.config.get('listen_duration', 3.0)  # seconds
        self.silence_threshold = self.config.get('silence_threshold', 0.01)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        
        # Wake word detection
        self.whisper_model = None
        self.is_listening = False
        self.audio_buffer = []
        self.last_detection_time = 0
        self.detection_cooldown = 2.0  # seconds
        
        # Threading
        self.listen_thread = None
        self.audio_queue = queue.Queue()
        
        # Callbacks
        self.on_wake_word: Optional[Callable[[str, float], None]] = None
        self.on_listening_started: Optional[Callable] = None
        self.on_listening_stopped: Optional[Callable] = None
        
        if AUDIO_AVAILABLE:
            self._initialize_whisper()
        else:
            self.logger.warning("🔇 Audio not available - Wake word detection disabled")
    
    def _initialize_whisper(self):
        """เริ่มต้น Whisper model"""
        try:
            model_size = self.config.get('whisper_model', 'tiny')
            self.logger.info(f"🧠 Loading Whisper {model_size} for wake word detection...")
            
            self.whisper_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8"
            )
            
            self.logger.info("✅ Wake word detection ready")
            
        except Exception as e:
            self.logger.error(f"❌ Whisper initialization failed: {e}")
            self.whisper_model = None
    
    def start_listening(self) -> bool:
        """เริ่มฟังคำปลุก"""
        if not AUDIO_AVAILABLE or not self.whisper_model:
            self.logger.error("❌ Cannot start wake word detection - audio not available")
            return False
        
        if self.is_listening:
            self.logger.warning("⚠️ Wake word detection already running")
            return True
        
        self.logger.info("🎙️ Starting wake word detection...")
        self.is_listening = True
        
        # Start listening thread
        self.listen_thread = threading.Thread(target=self._listen_loop, daemon=True)
        self.listen_thread.start()
        
        if self.on_listening_started:
            self.on_listening_started()
        
        self.logger.info("✅ Wake word detection started!")
        return True
    
    def stop_listening(self):
        """หยุดฟังคำปลุก"""
        if not self.is_listening:
            return
        
        self.logger.info("🛑 Stopping wake word detection...")
        self.is_listening = False
        
        # Wait for thread to finish
        if self.listen_thread and self.listen_thread.is_alive():
            self.listen_thread.join(timeout=2)
        
        if self.on_listening_stopped:
            self.on_listening_stopped()
        
        self.logger.info("✅ Wake word detection stopped")
    
    def _listen_loop(self):
        """การฟังแบบต่อเนื่อง"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while self.is_listening:
                    time.sleep(0.1)
                    
                    # Process accumulated audio
                    if len(self.audio_buffer) >= self.sample_rate * self.listen_duration:
                        self._process_audio_buffer()
                        
        except Exception as e:
            self.logger.error(f"❌ Listen loop error: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Callback สำหรับข้อมูลเสียง"""
        if not self.is_listening:
            return
        
        # Calculate audio level
        audio_level = np.sqrt(np.mean(indata ** 2))
        
        # Only process if there's significant audio
        if audio_level > self.silence_threshold:
            self.audio_buffer.extend(indata.flatten())
            
            # Limit buffer size
            max_buffer_size = int(self.sample_rate * self.listen_duration * 2)
            if len(self.audio_buffer) > max_buffer_size:
                # Keep only the last portion
                self.audio_buffer = self.audio_buffer[-max_buffer_size:]
    
    def _process_audio_buffer(self):
        """ประมวลผลบัฟเฟอร์เสียง"""
        if not self.audio_buffer:
            return
        
        # Check cooldown
        current_time = time.time()
        if current_time - self.last_detection_time < self.detection_cooldown:
            self.audio_buffer = []
            return
        
        try:
            # Convert to numpy array
            audio_data = np.array(self.audio_buffer, dtype=np.float32)
            
            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language=None,  # Auto-detect
                task="transcribe"
            )
            
            # Extract text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            full_text = " ".join(text_parts).strip().lower()
            
            if full_text:
                self.logger.debug(f"🎯 Detected speech: {full_text}")
                
                # Check for wake words
                wake_detected, confidence = self._check_wake_patterns(full_text, info.language)
                
                if wake_detected:
                    self.last_detection_time = current_time
                    self.logger.info(f"🚨 Wake word detected! ({info.language}): {full_text}")
                    
                    if self.on_wake_word:
                        self.on_wake_word(full_text, confidence)
            
        except Exception as e:
            self.logger.error(f"❌ Audio processing failed: {e}")
        
        finally:
            # Clear buffer
            self.audio_buffer = []
    
    def _check_wake_patterns(self, text: str, language: str) -> tuple[bool, float]:
        """ตรวจสอบรูปแบบคำปลุก"""
        
        # Determine language patterns to use
        if language == 'th':
            patterns = self.wake_patterns['thai']
        elif language == 'en':
            patterns = self.wake_patterns['english']
        else:
            # Try both languages
            patterns = self.wake_patterns['english'] + self.wake_patterns['thai']
        
        # Check each pattern
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                # Calculate simple confidence based on pattern match quality
                confidence = self._calculate_confidence(text, pattern)
                
                if confidence >= self.confidence_threshold:
                    return True, confidence
        
        return False, 0.0
    
    def _calculate_confidence(self, text: str, pattern: str) -> float:
        """คำนวณความมั่นใจของการตรวจจับ"""
        # Simple confidence calculation
        base_confidence = 0.8
        
        # Bonus for exact matches
        if 'jarvis' in text.lower() or 'จาร์วิส' in text:
            base_confidence += 0.1
        
        # Bonus for wake words
        if any(word in text.lower() for word in ['hey', 'hi', 'hello', 'เฮ้', 'สวัสดี', 'หวัดดี']):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def test_wake_word_detection(self, test_text: str) -> tuple[bool, float]:
        """ทดสอบการตรวจจับคำปลุกด้วยข้อความ"""
        # Auto-detect language for testing
        has_thai = any(ord(char) >= 0x0E00 and ord(char) <= 0x0E7F for char in test_text)
        language = 'th' if has_thai else 'en'
        return self._check_wake_patterns(test_text.lower(), language)
    
    def get_status(self) -> Dict[str, Any]:
        """สถานะของระบบ"""
        return {
            "is_listening": self.is_listening,
            "whisper_ready": self.whisper_model is not None,
            "audio_available": AUDIO_AVAILABLE,
            "wake_patterns": len(self.wake_patterns['english']) + len(self.wake_patterns['thai']),
            "last_detection": self.last_detection_time,
            "buffer_size": len(self.audio_buffer)
        }


def test_wake_word_detector():
    """ทดสอบระบบตรวจจับคำปลุก"""
    print("🧪 Testing Simple Wake Word Detector...")
    
    detector = SimpleWakeWordDetector({
        'whisper_model': 'tiny',
        'confidence_threshold': 0.7
    })
    
    # Set up callbacks
    def on_wake_detected(text: str, confidence: float):
        print(f"🚨 WAKE WORD DETECTED: '{text}' (confidence: {confidence:.2f})")
    
    def on_listening_started():
        print("🎙️ Started listening for wake words...")
    
    def on_listening_stopped():
        print("🛑 Stopped listening for wake words")
    
    detector.on_wake_word = on_wake_detected
    detector.on_listening_started = on_listening_started
    detector.on_listening_stopped = on_listening_stopped
    
    # Test text-based detection
    print("\n📝 Testing text-based wake word detection:")
    test_cases = [
        "hey jarvis what time is it",
        "hi jarvis how are you",
        "jarvis please help me",
        "เฮ้ จาร์วิส ช่วยหน่อย",
        "สวัสดี จาร์วิส",
        "hello there friend",  # Should not trigger
        "just a normal sentence"  # Should not trigger
    ]
    
    for test_text in test_cases:
        detected, confidence = detector.test_wake_word_detection(test_text)
        status = "✅ DETECTED" if detected else "❌ NOT DETECTED"
        print(f"   {status}: '{test_text}' (confidence: {confidence:.2f})")
    
    # Test system status
    print(f"\n📊 System status: {detector.get_status()}")
    
    print("\n✅ Wake word detector test completed!")
    return detector


if __name__ == "__main__":
    test_wake_word_detector()