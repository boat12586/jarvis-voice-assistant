#!/usr/bin/env python3
"""
🤖 JARVIS Final Complete System
ระบบ JARVIS ที่สมบูรณ์พร้อมทุกฟีเจอร์
"""

import logging
import time
import threading
import queue
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
import sounddevice as sd
import numpy as np
import yaml

try:
    from faster_whisper import WhisperModel
    WHISPER_AVAILABLE = True
except ImportError:
    WHISPER_AVAILABLE = False

try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Wake Word Detection
class WakeWordDetector:
    """ตรวจจับคำปลุก"""
    
    def __init__(self, wake_words: List[str] = None):
        self.logger = logging.getLogger(__name__)
        self.wake_words = wake_words or ["hey jarvis", "jarvis", "ok jarvis"]
        self.is_listening = False
        self.confidence_threshold = 0.7
        
        # Initialize speech recognition for wake word
        if WHISPER_AVAILABLE:
            try:
                self.whisper = WhisperModel("tiny", device="cpu")
                self.available = True
                self.logger.info("✅ Wake word detector ready")
            except Exception as e:
                self.logger.error(f"❌ Wake word detector failed: {e}")
                self.available = False
        else:
            self.available = False
    
    def start_detection(self):
        """เริ่มตรวจจับคำปลุก"""
        if not self.available:
            self.logger.warning("⚠️ Wake word detection not available")
            return False
        
        self.is_listening = True
        self.detection_thread = threading.Thread(target=self._detection_loop)
        self.detection_thread.daemon = True
        self.detection_thread.start()
        
        self.logger.info("👂 Wake word detection started")
        return True
    
    def stop_detection(self):
        """หยุดตรวจจับคำปลุก"""
        self.is_listening = False
        if hasattr(self, 'detection_thread'):
            self.detection_thread.join(timeout=1)
        self.logger.info("🔇 Wake word detection stopped")
    
    def _detection_loop(self):
        """ลูปตรวจจับคำปลุก"""
        try:
            while self.is_listening:
                # Record a short audio clip
                duration = 2.0  # 2 seconds
                audio_data = sd.rec(int(duration * 16000), 
                                  samplerate=16000, 
                                  channels=1, 
                                  dtype=np.float32)
                sd.wait()
                
                # Check for wake words
                if self._check_wake_word(audio_data.flatten()):
                    self.logger.info("🚨 Wake word detected!")
                    if hasattr(self, 'on_wake_word'):
                        self.on_wake_word()
                    
                time.sleep(0.5)  # Brief pause between checks
                
        except Exception as e:
            self.logger.error(f"❌ Wake word detection error: {e}")
    
    def _check_wake_word(self, audio_data: np.ndarray) -> bool:
        """ตรวจสอบคำปลุก"""
        try:
            # Quick energy check first
            energy = np.sum(audio_data ** 2) / len(audio_data)
            if energy < 0.001:  # Too quiet
                return False
            
            # Transcribe audio
            import io
            import wave
            
            # Convert to WAV format
            audio_buffer = io.BytesIO()
            audio_int16 = (audio_data * 32767).astype(np.int16)
            
            with wave.open(audio_buffer, 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(audio_int16.tobytes())
            
            audio_buffer.seek(0)
            
            # Transcribe
            segments, _ = self.whisper.transcribe(audio_buffer)
            
            for segment in segments:
                text = segment.text.lower().strip()
                
                # Check if any wake word is in the text
                for wake_word in self.wake_words:
                    if wake_word in text:
                        return True
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Wake word check error: {e}")
            return False

# Conversation Memory
class ConversationMemory:
    """หน่วยความจำการสนทนา"""
    
    def __init__(self, data_dir: str = "data/conversation_memory"):
        self.logger = logging.getLogger(__name__)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.conversations = []
        self.user_profile = {}
        self.preferences = {}
        
        self._load_memory()
        self.logger.info("✅ Conversation memory initialized")
    
    def add_conversation(self, user_input: str, ai_response: str, 
                        language: str = "en", context: Dict[str, Any] = None):
        """เพิ่มการสนทนา"""
        conversation_entry = {
            "timestamp": time.time(),
            "user_input": user_input,
            "ai_response": ai_response,
            "language": language,
            "context": context or {}
        }
        
        self.conversations.append(conversation_entry)
        
        # Keep only last 100 conversations
        if len(self.conversations) > 100:
            self.conversations = self.conversations[-100:]
        
        # Auto-save every 10 conversations
        if len(self.conversations) % 10 == 0:
            self._save_memory()
    
    def get_recent_context(self, num_turns: int = 3) -> List[Dict[str, Any]]:
        """ดึงบริบทการสนทนาล่าสุด"""
        return self.conversations[-num_turns:] if self.conversations else []
    
    def update_user_profile(self, key: str, value: Any):
        """อัพเดทโปรไฟล์ผู้ใช้"""
        self.user_profile[key] = value
        self._save_memory()
    
    def update_preferences(self, key: str, value: Any):
        """อัพเดทค่าที่ต้องการ"""
        self.preferences[key] = value
        self._save_memory()
    
    def _load_memory(self):
        """โหลดหน่วยความจำ"""
        try:
            # Load conversations
            conv_file = self.data_dir / "conversations.json"
            if conv_file.exists():
                with open(conv_file, 'r', encoding='utf-8') as f:
                    self.conversations = json.load(f)
            
            # Load user profile
            profile_file = self.data_dir / "user_profile.json"
            if profile_file.exists():
                with open(profile_file, 'r', encoding='utf-8') as f:
                    self.user_profile = json.load(f)
            
            # Load preferences
            pref_file = self.data_dir / "preferences.json"
            if pref_file.exists():
                with open(pref_file, 'r', encoding='utf-8') as f:
                    self.preferences = json.load(f)
            
            self.logger.info(f"📂 Loaded {len(self.conversations)} conversations")
            
        except Exception as e:
            self.logger.error(f"❌ Memory loading error: {e}")
    
    def _save_memory(self):
        """บันทึกหน่วยความจำ"""
        try:
            # Save conversations
            with open(self.data_dir / "conversations.json", 'w', encoding='utf-8') as f:
                json.dump(self.conversations, f, ensure_ascii=False, indent=2)
            
            # Save user profile
            with open(self.data_dir / "user_profile.json", 'w', encoding='utf-8') as f:
                json.dump(self.user_profile, f, ensure_ascii=False, indent=2)
            
            # Save preferences
            with open(self.data_dir / "preferences.json", 'w', encoding='utf-8') as f:
                json.dump(self.preferences, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"❌ Memory saving error: {e}")

# Enhanced AI with Memory
class MemoryEnhancedAI:
    """AI ที่มีหน่วยความจำ"""
    
    def __init__(self, memory: ConversationMemory):
        self.memory = memory
        self.responses = {
            'en': {
                'greetings': [
                    "Hello! I'm JARVIS. How can I help you today?",
                    "Good day! Ready to assist you.",
                    "Hi there! What can I do for you?"
                ],
                'time': lambda: f"The current time is {time.strftime('%H:%M:%S')}",
                'date': lambda: f"Today is {time.strftime('%A, %B %d, %Y')}",
                'weather': [
                    "I don't have weather access yet, but I'm working on it!",
                    "Weather integration is coming soon!"
                ],
                'memory': [
                    "I remember our conversations and learn from them.",
                    "My memory helps me provide better assistance over time.",
                    "I keep track of our interactions to serve you better."
                ],
                'thanks': [
                    "You're very welcome!",
                    "Happy to help!",
                    "My pleasure!"
                ],
                'goodbye': [
                    "Goodbye! Have a wonderful day!",
                    "See you later! Take care!",
                    "Farewell! I'll be here when you need me."
                ],
                'default': [
                    "I understand. How can I help you further?",
                    "Interesting! What else can I assist with?",
                    "I'm here to help. What would you like to know?"
                ]
            },
            'th': {
                'greetings': [
                    "สวัสดีครับ ผม JARVIS พร้อมช่วยเหลือคุณครับ",
                    "วันดีครับ ผมพร้อมให้บริการครับ",
                    "สวัสดีครับ มีอะไรให้ช่วยไหมครับ"
                ],
                'time': lambda: f"เวลาตอนนี้คือ {time.strftime('%H:%M:%S')} ครับ",
                'date': lambda: f"วันนี้เป็นวันที่ {time.strftime('%d %B %Y')} ครับ",
                'weather': [
                    "ผมยังเข้าถึงข้อมูลสภาพอากาศไม่ได้ครับ กำลังพัฒนาอยู่ครับ",
                    "ระบบสภาพอากาศจะมาเร็วๆ นี้ครับ"
                ],
                'memory': [
                    "ผมจำการสนทนาของเราได้และเรียนรู้จากมันครับ",
                    "หน่วยความจำของผมช่วยให้ผมช่วยเหลือคุณได้ดีขึ้นครับ",
                    "ผมเก็บประวัติการสนทนาเพื่อให้บริการที่ดีขึ้นครับ"
                ],
                'thanks': [
                    "ยินดีครับ!",
                    "ดีใจที่ได้ช่วยครับ",
                    "เป็นความยินดีของผมครับ"
                ],
                'goodbye': [
                    "ลาก่อนครับ ขอให้มีวันที่ดีครับ",
                    "แล้วพบกันใหม่ครับ ดูแลตัวด้วยนะครับ",
                    "ลาก่อนครับ ผมจะอยู่ที่นี่เมื่อคุณต้องการครับ"
                ],
                'default': [
                    "เข้าใจแล้วครับ มีอะไรให้ช่วยอีกไหมครับ",
                    "น่าสนใจครับ มีอะไรอื่นที่ต้องการช่วยเหลือไหมครับ",
                    "ผมอยู่ที่นี่เพื่อช่วยครับ อยากทราบอะไรเพิ่มเติมไหมครับ"
                ]
            }
        }
    
    def get_response(self, text: str, language: str = 'en') -> str:
        """สร้างการตอบสนองที่มีหน่วยความจำ"""
        import random
        
        text_lower = text.lower()
        lang_responses = self.responses.get(language, self.responses['en'])
        
        # Check for personalization
        user_name = self.memory.user_profile.get('name')
        preferred_language = self.memory.preferences.get('language', language)
        
        # Enhanced response generation with context
        recent_context = self.memory.get_recent_context(3)
        
        # Generate response based on context
        response = ""
        
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'สวัสดี']):
            response = random.choice(lang_responses['greetings'])
            if user_name:
                response = f"{response} {user_name}!"
        elif any(word in text_lower for word in ['time', 'เวลา', 'กี่โมง']):
            time_func = lang_responses['time']
            response = time_func() if callable(time_func) else random.choice(time_func)
        elif any(word in text_lower for word in ['date', 'วันที่', 'วันนี้']):
            date_func = lang_responses['date'] 
            response = date_func() if callable(date_func) else random.choice(date_func)
        elif any(word in text_lower for word in ['weather', 'อากาศ']):
            response = random.choice(lang_responses['weather'])
        elif any(word in text_lower for word in ['remember', 'memory', 'จำ', 'ความจำ']):
            response = random.choice(lang_responses['memory'])
        elif any(word in text_lower for word in ['thank', 'ขอบคุณ']):
            response = random.choice(lang_responses['thanks'])
        elif any(word in text_lower for word in ['bye', 'goodbye', 'ลาก่อน']):
            response = random.choice(lang_responses['goodbye'])
        else:
            response = random.choice(lang_responses['default'])
            
            # Add contextual awareness
            if recent_context:
                last_topic = recent_context[-1].get('user_input', '').lower()
                if 'time' in last_topic and 'time' not in text_lower:
                    if language == 'th':
                        response += " เกี่ยวกับเวลาที่เพิ่งถามไปหรือเปล่าครับ"
                    else:
                        response += " Are you asking about the time we just discussed?"
        
        return response

# Complete JARVIS System
class JarvisFinalComplete:
    """ระบบ JARVIS ที่สมบูรณ์"""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize memory
        self.memory = ConversationMemory()
        
        # Initialize components
        self.ai = MemoryEnhancedAI(self.memory)
        self.wake_word_detector = WakeWordDetector()
        
        # Voice components (from enhanced voice system)
        try:
            from jarvis_enhanced_voice import EnhancedSpeechRecognition, EnhancedTTS
            self.speech_recognition = EnhancedSpeechRecognition()
            self.tts = EnhancedTTS()
            self.voice_available = True
        except ImportError:
            self.logger.warning("⚠️ Voice components not available")
            self.voice_available = False
        
        # System state
        self.is_active = False
        self.is_listening = False
        self.wake_word_active = False
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "wake_word_detections": 0,
            "voice_interactions": 0,
            "text_interactions": 0,
            "start_time": time.time()
        }
        
        # Set up wake word callback
        if self.wake_word_detector.available:
            self.wake_word_detector.on_wake_word = self._on_wake_word_detected
        
        self.logger.info("✅ JARVIS Final Complete System initialized")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """โหลดการตั้งค่า"""
        try:
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            return {}
        except Exception as e:
            self.logger.error(f"❌ Config loading failed: {e}")
            return {}
    
    def start_system(self) -> bool:
        """เริ่มระบบ JARVIS"""
        if self.is_active:
            return True
        
        self.logger.info("🚀 Starting JARVIS Final Complete System...")
        
        try:
            self.is_active = True
            self.stats["start_time"] = time.time()
            
            # Start wake word detection
            if self.wake_word_detector.available:
                self.wake_word_detector.start_detection()
                self.wake_word_active = True
            
            self.logger.info("✅ JARVIS Final Complete System is active!")
            self.logger.info("🎙️ Say 'Hey JARVIS' to start conversation")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System startup failed: {e}")
            return False
    
    def stop_system(self):
        """หยุดระบบ JARVIS"""
        if not self.is_active:
            return
        
        self.logger.info("🛑 Stopping JARVIS Final Complete System...")
        
        # Stop wake word detection
        if self.wake_word_active:
            self.wake_word_detector.stop_detection()
            self.wake_word_active = False
        
        self.is_active = False
        self.memory._save_memory()
        
        self.logger.info("✅ JARVIS stopped and memory saved")
    
    def _on_wake_word_detected(self):
        """เมื่อตรวจพบคำปลุก"""
        self.logger.info("🚨 Wake word detected! Starting conversation...")
        self.stats["wake_word_detections"] += 1
        
        if self.voice_available:
            self._start_voice_conversation()
        else:
            # Fallback to text mode
            self.process_text_message("Hello JARVIS")
    
    def _start_voice_conversation(self, duration: float = 5.0):
        """เริ่มการสนทนาด้วยเสียง"""
        if not self.voice_available:
            self.logger.warning("⚠️ Voice not available")
            return
        
        self.logger.info(f"🎤 Listening for {duration} seconds...")
        self.is_listening = True
        
        try:
            # Record audio
            audio_data = sd.rec(int(duration * 16000), 
                              samplerate=16000, 
                              channels=1, 
                              dtype=np.float32)
            sd.wait()
            
            # Transcribe
            text, language, confidence = self.speech_recognition.transcribe_audio(
                audio_data.flatten(), 16000
            )
            
            if text and confidence > 0.3:
                self.logger.info(f"📝 Heard: '{text}' ({language})")
                self._process_interaction(text, language, "voice")
            else:
                self.logger.info("❌ Could not understand speech")
                
        except Exception as e:
            self.logger.error(f"❌ Voice conversation error: {e}")
        finally:
            self.is_listening = False
    
    def process_text_message(self, text: str, language: str = "auto"):
        """ประมวลผลข้อความ"""
        # Auto-detect language if needed
        if language == "auto":
            language = "th" if any(ord(c) > 127 for c in text) else "en"
        
        self._process_interaction(text, language, "text")
    
    def _process_interaction(self, text: str, language: str, mode: str):
        """ประมวลผลการโต้ตอบ"""
        self.logger.info(f"💬 Processing {mode} input: '{text}' ({language})")
        
        # Extract user information
        self._extract_user_info(text, language)
        
        # Generate response
        response = self.ai.get_response(text, language)
        
        # Add to memory
        self.memory.add_conversation(
            user_input=text,
            ai_response=response,
            language=language,
            context={"mode": mode, "timestamp": time.time()}
        )
        
        # Speak response if voice is available
        if self.voice_available and hasattr(self, 'tts'):
            self.tts.speak(response, language)
        else:
            print(f"🤖 JARVIS: {response}")
        
        # Update statistics
        self.stats["total_interactions"] += 1
        if mode == "voice":
            self.stats["voice_interactions"] += 1
        else:
            self.stats["text_interactions"] += 1
    
    def _extract_user_info(self, text: str, language: str):
        """สกัดข้อมูลผู้ใช้"""
        text_lower = text.lower()
        
        # Extract name
        if any(phrase in text_lower for phrase in ["my name is", "i'm", "call me", "ผมชื่อ", "ฉันชื่อ"]):
            words = text.split()
            for i, word in enumerate(words):
                if word.lower() in ["name", "is", "ชื่อ"] and i + 1 < len(words):
                    name = words[i + 1].strip('.,!?')
                    self.memory.update_user_profile("name", name)
                    self.logger.info(f"👤 Learned user name: {name}")
                    break
        
        # Extract language preference
        if language not in self.memory.preferences.get("languages", []):
            languages = self.memory.preferences.get("languages", [])
            languages.append(language)
            self.memory.update_preferences("languages", languages)
    
    def get_status(self) -> Dict[str, Any]:
        """สถานะระบบ"""
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "active": self.is_active,
            "listening": self.is_listening,
            "wake_word_active": self.wake_word_active,
            "voice_available": self.voice_available,
            "statistics": self.stats.copy(),
            "uptime_seconds": uptime,
            "memory_stats": {
                "conversations": len(self.memory.conversations),
                "user_profile": len(self.memory.user_profile),
                "preferences": len(self.memory.preferences)
            },
            "components": {
                "wake_word_detector": self.wake_word_detector.available,
                "speech_recognition": getattr(self, 'speech_recognition', None) is not None,
                "tts": getattr(self, 'tts', None) is not None,
                "memory": True,
                "ai": True
            }
        }
    
    def run_interactive(self):
        """เรียกใช้แบบ Interactive"""
        if not self.start_system():
            return
        
        print("\n🤖 JARVIS Final Complete System")
        print("===============================")
        print("Commands:")
        print("  chat <message>  - Send text message")
        print("  voice           - Start voice input")
        print("  status          - Show system status")
        print("  memory          - Show memory info")
        print("  profile         - Show user profile")
        print("  wake on/off     - Control wake word detection")
        print("  quit            - Exit")
        
        try:
            while self.is_active:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower().startswith('chat '):
                    message = user_input[5:]
                    self.process_text_message(message)
                elif user_input.lower() == 'voice':
                    if self.voice_available:
                        print("🎤 Listening for 5 seconds...")
                        self._start_voice_conversation(5.0)
                    else:
                        print("❌ Voice input not available")
                elif user_input.lower() == 'status':
                    status = self.get_status()
                    print(f"\n📊 System Status:")
                    for key, value in status.items():
                        print(f"   {key}: {value}")
                elif user_input.lower() == 'memory':
                    recent = self.memory.get_recent_context(5)
                    print(f"\n🧠 Recent Memory ({len(recent)} entries):")
                    for entry in recent:
                        timestamp = time.strftime("%H:%M:%S", time.localtime(entry['timestamp']))
                        print(f"   [{timestamp}] User: {entry['user_input']}")
                        print(f"   [{timestamp}] AI: {entry['ai_response']}")
                elif user_input.lower() == 'profile':
                    print(f"\n👤 User Profile:")
                    for key, value in self.memory.user_profile.items():
                        print(f"   {key}: {value}")
                    print(f"\n⚙️ Preferences:")
                    for key, value in self.memory.preferences.items():
                        print(f"   {key}: {value}")
                elif user_input.lower() == 'wake on':
                    if not self.wake_word_active and self.wake_word_detector.available:
                        self.wake_word_detector.start_detection()
                        self.wake_word_active = True
                        print("✅ Wake word detection enabled")
                    else:
                        print("⚠️ Wake word already active or not available")
                elif user_input.lower() == 'wake off':
                    if self.wake_word_active:
                        self.wake_word_detector.stop_detection()
                        self.wake_word_active = False
                        print("🔇 Wake word detection disabled")
                    else:
                        print("⚠️ Wake word already inactive")
                else:
                    self.process_text_message(user_input)
                    
        except KeyboardInterrupt:
            print("\n🛑 Interrupted by user")
        finally:
            self.stop_system()
            print("👋 Goodbye!")


def main():
    """ฟังก์ชันหลัก"""
    # ตั้งค่า logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("🤖 JARVIS Final Complete AI Assistant")
    print("=====================================")
    print("🎙️ Wake Word Detection: 'Hey JARVIS'")
    print("🧠 Memory: Conversation and user learning")
    print("🗣️ Voice: Speech recognition and synthesis")
    print("💬 Chat: Text-based interaction")
    
    # สร้างและเรียกใช้ JARVIS
    jarvis = JarvisFinalComplete()
    jarvis.run_interactive()


if __name__ == "__main__":
    main()