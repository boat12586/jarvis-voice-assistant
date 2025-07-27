#!/usr/bin/env python3
"""
🎙️ Real-time Voice Conversation System for JARVIS
ระบบสนทนาเสียงแบบเรียลไทม์ที่ตอบสนองได้ทันที
"""

import logging
import time
import threading
import queue
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum
import uuid

try:
    import sounddevice as sd
    import numpy as np
    from faster_whisper import WhisperModel
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logging.warning("Audio libraries not available")


class ConversationState(Enum):
    """สถานะการสนทนา"""
    IDLE = "idle"
    LISTENING = "listening"  
    PROCESSING = "processing"
    RESPONDING = "responding"
    PAUSED = "paused"


@dataclass
class VoiceMessage:
    """ข้อความเสียงในการสนทนา"""
    message_id: str
    text: str
    language: str
    timestamp: float
    speaker: str  # 'user' หรือ 'jarvis'
    confidence: float = 0.0
    audio_duration: float = 0.0


class RealTimeVoiceConversation:
    """ระบบสนทนาเสียงแบบเรียลไทม์"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Conversation state
        self.state = ConversationState.IDLE
        self.conversation_history: List[VoiceMessage] = []
        self.session_id = str(uuid.uuid4())
        
        # Audio settings
        self.sample_rate = self.config.get('sample_rate', 16000)
        self.chunk_size = self.config.get('chunk_size', 1024)
        self.silence_threshold = self.config.get('silence_threshold', 0.01)
        self.silence_duration = self.config.get('silence_duration', 2.0)
        
        # Audio buffers
        self.audio_queue = queue.Queue()
        self.recording_buffer = []
        self.is_recording = False
        
        # Speech recognition
        self.whisper_model = None
        self.last_speech_time = 0
        
        # Threading
        self.audio_thread = None
        self.processing_thread = None
        self.is_running = False
        
        # Callbacks
        self.on_speech_detected: Optional[Callable] = None
        self.on_text_recognized: Optional[Callable] = None
        self.on_response_generated: Optional[Callable] = None
        
        if AUDIO_AVAILABLE:
            self._initialize_audio()
        else:
            self.logger.warning("🔇 Audio not available - Voice conversation disabled")
    
    def _initialize_audio(self):
        """เริ่มต้นระบบเสียง"""
        try:
            # Initialize Whisper
            model_size = self.config.get('whisper_model', 'tiny')
            self.logger.info(f"🧠 Loading Whisper {model_size} model...")
            self.whisper_model = WhisperModel(
                model_size,
                device="cpu",
                compute_type="int8"
            )
            self.logger.info("✅ Speech recognition ready")
            
            # Test audio devices
            devices = sd.query_devices()
            self.logger.info(f"🎤 Found {len(devices)} audio devices")
            
        except Exception as e:
            self.logger.error(f"❌ Audio initialization failed: {e}")
            self.whisper_model = None
    
    def start_conversation(self) -> bool:
        """เริ่มการสนทนา"""
        if not AUDIO_AVAILABLE or not self.whisper_model:
            self.logger.error("❌ Cannot start conversation - audio not available")
            return False
        
        if self.is_running:
            self.logger.warning("⚠️ Conversation already running")
            return True
        
        self.logger.info("🚀 Starting real-time voice conversation...")
        self.is_running = True
        self.state = ConversationState.LISTENING
        
        # Start audio processing thread
        self.audio_thread = threading.Thread(target=self._audio_loop, daemon=True)
        self.audio_thread.start()
        
        # Start speech processing thread
        self.processing_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self.processing_thread.start()
        
        self.logger.info("✅ Voice conversation started!")
        return True
    
    def stop_conversation(self):
        """หยุดการสนทนา"""
        self.logger.info("🛑 Stopping voice conversation...")
        self.is_running = False
        self.state = ConversationState.IDLE
        
        # Wait for threads to finish
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=2)
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=2)
        
        self.logger.info("✅ Voice conversation stopped")
    
    def _audio_loop(self):
        """Audio capture loop"""
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                blocksize=self.chunk_size,
                callback=self._audio_callback
            ):
                while self.is_running:
                    time.sleep(0.1)
        except Exception as e:
            self.logger.error(f"❌ Audio loop error: {e}")
    
    def _audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if not self.is_running:
            return
        
        # Calculate audio level
        audio_level = np.sqrt(np.mean(indata ** 2))
        
        # Detect speech
        if audio_level > self.silence_threshold:
            self.last_speech_time = time.time()
            if not self.is_recording:
                self._start_recording()
            self.recording_buffer.extend(indata.flatten())
        else:
            # Check for end of speech
            if self.is_recording and (time.time() - self.last_speech_time) > self.silence_duration:
                self._stop_recording()
    
    def _start_recording(self):
        """เริ่มบันทึกเสียง"""
        if self.state != ConversationState.LISTENING:
            return
        
        self.is_recording = True
        self.recording_buffer = []
        self.logger.debug("🎤 Started recording...")
        
        if self.on_speech_detected:
            self.on_speech_detected()
    
    def _stop_recording(self):
        """หยุดบันทึกเสียงและส่งไปประมวลผล"""
        if not self.is_recording:
            return
        
        self.is_recording = False
        audio_data = np.array(self.recording_buffer, dtype=np.float32)
        
        if len(audio_data) > 0:
            self.logger.debug("🎤 Stopped recording, queuing for processing...")
            self.audio_queue.put(audio_data)
        
        self.recording_buffer = []
    
    def _processing_loop(self):
        """Speech processing loop"""
        while self.is_running:
            try:
                # Wait for audio data
                audio_data = self.audio_queue.get(timeout=1.0)
                self._process_speech(audio_data)
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"❌ Processing error: {e}")
    
    def _process_speech(self, audio_data: np.ndarray):
        """ประมวลผลเสียงเป็นข้อความ"""
        if not self.whisper_model:
            return
        
        self.state = ConversationState.PROCESSING
        start_time = time.time()
        
        try:
            # Convert to format expected by Whisper
            if len(audio_data) < self.sample_rate * 0.1:  # Less than 100ms
                self.logger.debug("🔇 Audio too short, skipping...")
                self.state = ConversationState.LISTENING
                return
            
            # Transcribe with Whisper
            segments, info = self.whisper_model.transcribe(
                audio_data,
                language=self.config.get('language', 'auto'),
                task="transcribe"
            )
            
            # Extract text
            text_parts = []
            for segment in segments:
                text_parts.append(segment.text.strip())
            
            full_text = " ".join(text_parts).strip()
            
            if full_text and len(full_text) > 1:
                processing_time = time.time() - start_time
                
                # Create voice message
                message = VoiceMessage(
                    message_id=str(uuid.uuid4()),
                    text=full_text,
                    language=info.language,
                    timestamp=time.time(),
                    speaker="user",
                    confidence=getattr(info, 'language_probability', 0.0),
                    audio_duration=len(audio_data) / self.sample_rate
                )
                
                self.conversation_history.append(message)
                
                self.logger.info(f"🎯 Recognized ({info.language}): {full_text}")
                self.logger.debug(f"⏱️ Processing time: {processing_time:.2f}s")
                
                # Trigger callback
                if self.on_text_recognized:
                    self.on_text_recognized(message)
            
        except Exception as e:
            self.logger.error(f"❌ Speech processing failed: {e}")
        
        finally:
            self.state = ConversationState.LISTENING
    
    def add_jarvis_response(self, text: str, language: str = 'en'):
        """เพิ่มการตอบสนองของ JARVIS"""
        message = VoiceMessage(
            message_id=str(uuid.uuid4()),
            text=text,
            language=language,
            timestamp=time.time(),
            speaker="jarvis"
        )
        
        self.conversation_history.append(message)
        
        if self.on_response_generated:
            self.on_response_generated(message)
    
    def get_conversation_context(self, turns: int = 5) -> List[Dict[str, str]]:
        """ดึงบริบทการสนทนาล่าสุด"""
        recent_messages = self.conversation_history[-turns*2:] if turns > 0 else self.conversation_history
        
        context = []
        for msg in recent_messages:
            context.append({
                "role": "user" if msg.speaker == "user" else "assistant",
                "content": msg.text,
                "language": msg.language
            })
        
        return context
    
    def clear_conversation(self):
        """ล้างประวัติการสนทนา"""
        self.conversation_history.clear()
        self.session_id = str(uuid.uuid4())
        self.logger.info("🗑️ Conversation history cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """สถิติการสนทนา"""
        user_messages = [msg for msg in self.conversation_history if msg.speaker == "user"]
        jarvis_messages = [msg for msg in self.conversation_history if msg.speaker == "jarvis"]
        
        return {
            "session_id": self.session_id,
            "state": self.state.value,
            "total_messages": len(self.conversation_history),
            "user_messages": len(user_messages),
            "jarvis_messages": len(jarvis_messages),
            "languages_detected": list(set(msg.language for msg in user_messages)),
            "average_confidence": np.mean([msg.confidence for msg in user_messages]) if user_messages else 0.0,
            "conversation_duration": time.time() - (self.conversation_history[0].timestamp if self.conversation_history else time.time())
        }


def test_realtime_conversation():
    """ทดสอบระบบสนทนาเรียลไทม์"""
    print("🧪 Testing Real-time Voice Conversation System...")
    
    # Create conversation system
    config = {
        'whisper_model': 'tiny',
        'sample_rate': 16000,
        'silence_threshold': 0.01,
        'silence_duration': 2.0
    }
    
    conversation = RealTimeVoiceConversation(config)
    
    # Set up callbacks
    def on_speech_detected():
        print("🎤 Speech detected...")
    
    def on_text_recognized(message: VoiceMessage):
        print(f"📝 Recognized: {message.text} ({message.language})")
    
    def on_response_generated(message: VoiceMessage):
        print(f"🤖 JARVIS: {message.text}")
    
    conversation.on_speech_detected = on_speech_detected
    conversation.on_text_recognized = on_text_recognized
    conversation.on_response_generated = on_response_generated
    
    # Test conversation
    print("✅ Real-time conversation system ready!")
    print("📊 Statistics:", conversation.get_statistics())
    
    return conversation


if __name__ == "__main__":
    test_realtime_conversation()