#!/usr/bin/env python3
"""
🎙️ JARVIS Enhanced Voice System
ระบบเสียงที่พัฒนาแล้วด้วย Whisper และ TTS จริง
"""

import logging
import time
import threading
import queue
import wave
import io
from pathlib import Path
from typing import Dict, Any, Optional
import sounddevice as sd
import numpy as np
import yaml

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False
    logging.warning("Faster-Whisper not available")

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    logging.warning("TTS not available")

# Enhanced AI with more responses
class EnhancedAI:
    """AI ที่ตอบได้หลากหลายขึ้น"""
    
    def __init__(self):
        self.responses = {
            'en': {
                'greetings': [
                    "Hello! I'm JARVIS, your voice assistant. How can I help you?",
                    "Good day! JARVIS at your service. What can I do for you?",
                    "Hi there! I'm ready to assist you."
                ],
                'time': [
                    f"The current time is {time.strftime('%H:%M:%S')}",
                    f"It's {time.strftime('%H:%M')} right now",
                    f"The time is {time.strftime('%H:%M:%S')}"
                ],
                'date': [
                    f"Today is {time.strftime('%Y-%m-%d')}",
                    f"The date is {time.strftime('%B %d, %Y')}",
                    f"Today's date is {time.strftime('%A, %B %d, %Y')}"
                ],
                'weather': [
                    "I don't have weather data access yet, but I'm working on it!",
                    "Weather integration is coming soon!",
                    "I can't check the weather right now, but it's a great feature for the future!"
                ],
                'capabilities': [
                    "I can help you with time, date, simple conversations, and voice commands!",
                    "I'm capable of voice recognition, text responses, and basic assistance.",
                    "My current abilities include voice interaction, time/date queries, and conversation."
                ],
                'thanks': [
                    "You're welcome! Happy to help!",
                    "My pleasure! Is there anything else?",
                    "Glad I could assist you!"
                ],
                'goodbye': [
                    "Goodbye! Have a great day!",
                    "See you later! Take care!",
                    "Farewell! Come back anytime!"
                ],
                'default': [
                    "I understand. How can I assist you further?",
                    "Interesting! What else can I help you with?",
                    "I'm here to help. What would you like to know?"
                ]
            },
            'th': {
                'greetings': [
                    "สวัสดีครับ ผม JARVIS ผู้ช่วยเสียงของคุณ ต้องการให้ช่วยอะไรไหมครับ",
                    "วันดีครับ ผม JARVIS พร้อมให้บริการครับ",
                    "สวัสดีครับ ผมพร้อมช่วยเหลือคุณครับ"
                ],
                'time': [
                    f"เวลาตอนนี้คือ {time.strftime('%H:%M:%S')} นาฬิกาครับ",
                    f"ตอนนี้เวลา {time.strftime('%H:%M')} ครับ",
                    f"เวลาปัจจุบันคือ {time.strftime('%H:%M:%S')} ครับ"
                ],
                'date': [
                    f"วันนี้เป็นวันที่ {time.strftime('%d-%m-%Y')} ครับ",
                    f"วันที่วันนี้คือ {time.strftime('%d %B %Y')} ครับ",
                    f"วันนี้เป็นวันที่ {time.strftime('%A %d %B %Y')} ครับ"
                ],
                'weather': [
                    "ตอนนี้ผมยังเข้าถึงข้อมูลสภาพอากาศไม่ได้ครับ แต่กำลังพัฒนาอยู่ครับ",
                    "ระบบสภาพอากาศกำลังจะมาเร็วๆ นี้ครับ",
                    "ผมยังตรวจสอบสภาพอากาศไม่ได้ครับ แต่เป็นฟีเจอร์ที่น่าสนใจสำหรับอนาคตครับ"
                ],
                'capabilities': [
                    "ผมช่วยเรื่องเวลา วันที่ การสนทนาง่ายๆ และคำสั่งเสียงได้ครับ",
                    "ผมสามารถรู้จำเสียง ตอบข้อความ และช่วยเหลือพื้นฐานได้ครับ",
                    "ความสามารถปัจจุบันของผมมีการโต้ตอบด้วยเสียง การสอบถามเวลา/วันที่ และการสนทนาครับ"
                ],
                'thanks': [
                    "ยินดีครับ! ดีใจที่ได้ช่วยครับ",
                    "เป็นความยินดีครับ มีอะไรอีกไหมครับ",
                    "ดีใจที่ช่วยได้ครับ!"
                ],
                'goodbye': [
                    "ลาก่อนครับ ขอให้มีวันที่ดีครับ",
                    "แล้วพบกันใหม่ครับ ดูแลตัวด้วยนะครับ",
                    "ลาก่อนครับ กลับมาคุยใหม่นะครับ"
                ],
                'default': [
                    "เข้าใจแล้วครับ มีอะไรให้ช่วยอีกไหมครับ",
                    "น่าสนใจครับ มีอะไรอื่นที่ต้องการความช่วยเหลือไหมครับ",
                    "ผมอยู่ที่นี่เพื่อช่วยครับ อยากทราบอะไรเพิ่มเติมไหมครับ"
                ]
            }
        }
        
        # Conversation memory
        self.conversation_history = []
        self.user_preferences = {}
    
    def get_response(self, text: str, language: str = 'en') -> str:
        """สร้างการตอบสนองที่ฉลาดขึ้น"""
        import random
        
        text_lower = text.lower()
        lang_responses = self.responses.get(language, self.responses['en'])
        
        # Add to conversation history
        self.conversation_history.append({
            'user': text,
            'timestamp': time.time(),
            'language': language
        })
        
        # Keep only last 10 conversations
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
        
        # Enhanced keyword matching with random responses
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'สวัสดี', 'หวัดดี']):
            return random.choice(lang_responses['greetings'])
        elif any(word in text_lower for word in ['time', 'เวลา', 'กี่โมง']):
            return random.choice(lang_responses['time'])
        elif any(word in text_lower for word in ['date', 'วันที่', 'วันนี้']):
            return random.choice(lang_responses['date'])
        elif any(word in text_lower for word in ['weather', 'อากาศ', 'สภาพอากาศ']):
            return random.choice(lang_responses['weather'])
        elif any(word in text_lower for word in ['what can you do', 'capabilities', 'ทำอะไรได้บ้าง', 'ความสามารถ']):
            return random.choice(lang_responses['capabilities'])
        elif any(word in text_lower for word in ['thank', 'thanks', 'ขอบคุณ', 'ขอบใจ']):
            return random.choice(lang_responses['thanks'])
        elif any(word in text_lower for word in ['bye', 'goodbye', 'ลาก่อน', 'บาย']):
            return random.choice(lang_responses['goodbye'])
        else:
            return random.choice(lang_responses['default'])

# Enhanced Speech Recognition
class EnhancedSpeechRecognition:
    """ระบบรู้จำเสียงที่พัฒนาแล้ว"""
    
    def __init__(self, model_size="base"):
        self.logger = logging.getLogger(__name__)
        
        if WHISPER_AVAILABLE:
            try:
                self.logger.info(f"🧠 Loading Whisper model: {model_size}")
                self.model = WhisperModel(model_size, device="cpu")
                self.available = True
                self.logger.info("✅ Whisper model loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load Whisper: {e}")
                self.available = False
        else:
            self.logger.warning("⚠️ Whisper not available - using mock recognition")
            self.available = False
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> tuple[str, str, float]:
        """แปลงเสียงเป็นข้อความ"""
        if not self.available:
            # Mock transcription for testing
            mock_texts = ["Hello JARVIS", "What time is it", "สวัสดีครับ", "ช่วยบอกเวลาหน่อย"]
            import random
            text = random.choice(mock_texts)
            language = "th" if any(ord(c) > 127 for c in text) else "en"
            return text, language, 0.9
        
        try:
            # Convert numpy array to audio file in memory
            audio_buffer = io.BytesIO()
            
            # Normalize audio
            audio_data = audio_data.astype(np.float32)
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data))
            
            # Convert to int16
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            # Create WAV file in memory
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(audio_int16.tobytes())
            
            audio_buffer.seek(0)
            
            # Transcribe
            segments, info = self.model.transcribe(audio_buffer)
            
            # Get the first segment
            for segment in segments:
                text = segment.text.strip()
                if text:
                    return text, info.language, segment.avg_logprob
            
            return "", "en", 0.0
            
        except Exception as e:
            self.logger.error(f"❌ Transcription error: {e}")
            return "", "en", 0.0

# Enhanced TTS
class EnhancedTTS:
    """ระบบแปลงข้อความเป็นเสียงที่พัฒนาแล้ว"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if TTS_AVAILABLE:
            try:
                # Initialize TTS models
                self.tts_en = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
                self.tts_available = True
                self.logger.info("✅ TTS models loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load TTS: {e}")
                self.tts_available = False
        else:
            self.logger.warning("⚠️ TTS not available - using text output")
            self.tts_available = False
    
    def speak(self, text: str, language: str = 'en'):
        """พูดข้อความ"""
        self.logger.info(f"🗣️ JARVIS says ({language}): {text}")
        
        if self.tts_available:
            try:
                # Generate audio file
                output_path = f"temp_tts_{int(time.time())}.wav"
                
                if language.startswith('en'):
                    self.tts_en.tts_to_file(text=text, file_path=output_path)
                else:
                    # For Thai, use English TTS for now
                    self.tts_en.tts_to_file(text=text, file_path=output_path)
                
                # Play audio (platform-specific)
                self._play_audio_file(output_path)
                
                # Clean up
                Path(output_path).unlink(missing_ok=True)
                
            except Exception as e:
                self.logger.error(f"❌ TTS error: {e}")
                print(f"🤖 JARVIS: {text}")
        else:
            print(f"🤖 JARVIS: {text}")
    
    def _play_audio_file(self, file_path: str):
        """เล่นไฟล์เสียง"""
        try:
            import pygame
            pygame.mixer.init()
            pygame.mixer.music.load(file_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
                
        except ImportError:
            # Fallback to system command
            import subprocess
            import platform
            
            system = platform.system()
            if system == "Windows":
                subprocess.run(["powershell", "-c", f"(New-Object Media.SoundPlayer '{file_path}').PlaySync()"])
            elif system == "Darwin":  # macOS
                subprocess.run(["afplay", file_path])
            elif system == "Linux":
                subprocess.run(["aplay", file_path])

# Enhanced Voice Activity Detection
class EnhancedVoiceDetector:
    """ตรวจจับเสียงพูดขั้นสูง"""
    
    def __init__(self, sample_rate=16000, frame_duration=30):
        self.sample_rate = sample_rate
        self.frame_duration = frame_duration
        self.frame_size = int(sample_rate * frame_duration / 1000)
        
        # Voice activity detection parameters
        self.energy_threshold = 0.01
        self.zero_crossing_threshold = 0.3
        self.speech_frames = 0
        self.min_speech_frames = 3
        
    def detect_speech(self, audio_data):
        """ตรวจสอบการพูดแบบขั้นสูง"""
        # Calculate energy
        energy = np.sum(audio_data ** 2) / len(audio_data)
        
        # Calculate zero crossing rate
        zero_crossings = np.sum(np.abs(np.diff(np.sign(audio_data)))) / (2 * len(audio_data))
        
        # Check if speech is detected
        is_speech = (energy > self.energy_threshold and 
                    zero_crossings > self.zero_crossing_threshold)
        
        if is_speech:
            self.speech_frames += 1
        else:
            self.speech_frames = max(0, self.speech_frames - 1)
        
        return self.speech_frames >= self.min_speech_frames

# Main Enhanced JARVIS System
class JarvisEnhancedVoice:
    """ระบบ JARVIS เสียงขั้นสูง"""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize enhanced components
        self.ai = EnhancedAI()
        self.speech_recognition = EnhancedSpeechRecognition(
            model_size=self.config.get('voice', {}).get('whisper', {}).get('model_size', 'base')
        )
        self.tts = EnhancedTTS()
        self.voice_detector = EnhancedVoiceDetector()
        
        # System state
        self.is_active = False
        self.is_listening = False
        self.conversation_active = False
        
        # Audio settings
        self.sample_rate = self.config.get('voice', {}).get('sample_rate', 16000)
        self.chunk_size = self.config.get('voice', {}).get('chunk_size', 1024)
        self.audio_queue = queue.Queue()
        self.audio_buffer = []
        
        # Statistics
        self.stats = {
            "interactions": 0,
            "voice_recognitions": 0,
            "start_time": time.time()
        }
        
        self.logger.info("✅ JARVIS Enhanced Voice System initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """โหลดการตั้งค่า"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f) or {}
                self.logger.info(f"✅ Config loaded from {config_path}")
                return config
            else:
                self.logger.warning(f"⚠️ Config file not found: {config_path}")
                return {}
        except Exception as e:
            self.logger.error(f"❌ Config loading failed: {e}")
            return {}
    
    def start_system(self) -> bool:
        """เริ่มระบบ JARVIS"""
        if self.is_active:
            self.logger.warning("⚠️ System already active")
            return True
        
        self.logger.info("🚀 Starting JARVIS Enhanced Voice System...")
        
        try:
            # Check audio devices
            devices = sd.query_devices()
            self.logger.info(f"📱 Found {len(devices)} audio devices")
            
            self.is_active = True
            self.stats["start_time"] = time.time()
            
            self.logger.info("✅ JARVIS Enhanced Voice is now active!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System startup failed: {e}")
            return False
    
    def stop_system(self):
        """หยุดระบบ JARVIS"""
        if not self.is_active:
            return
        
        self.logger.info("🛑 Stopping JARVIS Enhanced Voice System...")
        
        self.is_active = False
        self.is_listening = False
        self.conversation_active = False
        
        self.logger.info("✅ JARVIS stopped")
    
    def start_voice_conversation(self, duration: float = 5.0):
        """เริ่มการสนทนาด้วยเสียง"""
        if self.is_listening:
            self.logger.warning("⚠️ Already listening")
            return
        
        self.logger.info(f"🎤 Starting voice conversation for {duration} seconds...")
        self.is_listening = True
        self.audio_buffer = []
        
        try:
            # Record audio
            audio_data = sd.rec(int(duration * self.sample_rate), 
                              samplerate=self.sample_rate, 
                              channels=1, 
                              dtype=np.float32)
            sd.wait()  # Wait until recording is finished
            
            self.logger.info("🔍 Processing recorded audio...")
            
            # Process the recorded audio
            if len(audio_data) > 0:
                self._process_voice_input(audio_data.flatten())
            
        except Exception as e:
            self.logger.error(f"❌ Voice conversation error: {e}")
        finally:
            self.is_listening = False
    
    def _process_voice_input(self, audio_data: np.ndarray):
        """ประมวลผลเสียงที่ได้รับ"""
        # Check if there's actual speech
        if not self.voice_detector.detect_speech(audio_data):
            self.logger.info("🔇 No speech detected")
            return
        
        self.logger.info("🎤 Speech detected, transcribing...")
        
        # Transcribe speech
        text, language, confidence = self.speech_recognition.transcribe_audio(
            audio_data, self.sample_rate
        )
        
        if text and confidence > 0.3:
            self.logger.info(f"📝 Transcribed ({language}, {confidence:.2f}): '{text}'")
            
            # Generate and speak response
            self._handle_recognized_speech(text, language)
            self.stats["voice_recognitions"] += 1
        else:
            self.logger.info("❌ Could not transcribe speech clearly")
    
    def _handle_recognized_speech(self, text: str, language: str):
        """จัดการข้อความที่รู้จำได้"""
        # Generate response
        response = self.ai.get_response(text, language)
        
        # Speak response
        self.tts.speak(response, language)
        
        # Update statistics
        self.stats["interactions"] += 1
    
    def process_text_message(self, text: str, language: str = "en"):
        """ประมวลผลข้อความ"""
        self.logger.info(f"💬 Processing text: '{text}'")
        
        # Generate response
        response = self.ai.get_response(text, language)
        
        # Speak response
        self.tts.speak(response, language)
        
        # Update statistics
        self.stats["interactions"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """สถานะระบบ"""
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "active": self.is_active,
            "listening": self.is_listening,
            "conversation_active": self.conversation_active,
            "interactions": self.stats["interactions"],
            "voice_recognitions": self.stats["voice_recognitions"],
            "uptime_seconds": uptime,
            "components": {
                "speech_recognition": self.speech_recognition.available,
                "tts": self.tts.tts_available,
                "ai": True
            }
        }


def test_enhanced_voice():
    """ทดสอบระบบเสียงขั้นสูง"""
    # ตั้งค่า logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🎙️ Testing JARVIS Enhanced Voice System")
    print("======================================")
    
    # สร้าง JARVIS
    jarvis = JarvisEnhancedVoice()
    
    # เริ่มระบบ
    print("\n1. Starting enhanced system...")
    if jarvis.start_system():
        print("✅ Enhanced system started successfully!")
    else:
        print("❌ Failed to start enhanced system")
        return
    
    # ทดสอบการตอบสนองข้อความ
    print("\n2. Testing enhanced text responses...")
    test_messages = [
        ("Hello JARVIS, how are you today?", "en"),
        ("What time is it now?", "en"),
        ("What's today's date?", "en"),
        ("What can you do?", "en"),
        ("สวัสดีครับ JARVIS", "th"),
        ("ช่วยบอกเวลาหน่อยครับ", "th"),
        ("Thank you JARVIS", "en"),
        ("Goodbye", "en")
    ]
    
    for message, language in test_messages:
        print(f"\nTesting: '{message}' ({language})")
        jarvis.process_text_message(message, language)
        time.sleep(1)  # Brief pause between responses
    
    # แสดงสถานะ
    print("\n3. Enhanced system status:")
    status = jarvis.get_status()
    for key, value in status.items():
        print(f"   {key}: {value}")
    
    # ทดสอบเสียง (ถ้าผู้ใช้ต้องการ)
    print("\n4. Voice recognition test (optional)")
    print("Would you like to test voice recognition? (y/n)")
    
    # หยุดระบบ
    print("\n5. Stopping enhanced system...")
    jarvis.stop_system()
    print("✅ Enhanced system stopped")
    
    print("\n🎉 Enhanced voice system test completed!")


if __name__ == "__main__":
    test_enhanced_voice()