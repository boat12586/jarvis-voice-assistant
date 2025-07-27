#!/usr/bin/env python3
"""
🎙️ JARVIS Simple Voice Interface
สร้าง Voice Interface แบบง่ายโดยไม่ต้องใช้ TTS ที่ซับซ้อน

Version: 2.0.0 (2025 Edition)
"""

import sys
import os
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def create_simple_voice_system():
    """สร้างระบบ voice แบบง่ายใช้ libraries พื้นฐาน"""
    
    voice_dir = Path("src/voice")
    voice_dir.mkdir(exist_ok=True)
    
    # 1. Simple Speech Recognizer
    simple_recognizer = """#!/usr/bin/env python3
\"\"\"
🎤 Simple Speech Recognizer for JARVIS
ใช้ faster-whisper สำหรับการรู้จำเสียง
\"\"\"

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
    \"\"\"Simple speech recognizer using faster-whisper\"\"\"
    
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
        \"\"\"Initialize Whisper model\"\"\"
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
        \"\"\"Record from microphone and recognize speech\"\"\"
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
        \"\"\"Check if speech recognition is available\"\"\"
        return self.model is not None and AUDIO_AVAILABLE
    
    def get_audio_devices(self):
        \"\"\"Get available audio devices\"\"\"
        if not AUDIO_AVAILABLE:
            return []
        
        try:
            devices = sd.query_devices()
            return devices
        except Exception as e:
            self.logger.error(f"Failed to query audio devices: {e}")
            return []
"""
    
    # 2. Simple Text-to-Speech
    simple_tts = """#!/usr/bin/env python3
\"\"\"
🗣️ Simple Text-to-Speech for JARVIS
ใช้ระบบ TTS พื้นฐานของ OS
\"\"\"

import logging
import subprocess
import platform
from typing import Optional
import tempfile
import os

class SimpleTextToSpeech:
    \"\"\"Simple TTS using system commands\"\"\"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system = platform.system()
        self.available = self._check_availability()
        
        if self.available:
            self.logger.info(f"🗣️  TTS ready on {self.system}")
        else:
            self.logger.warning("🔇 TTS not available")
    
    def _check_availability(self) -> bool:
        \"\"\"Check if TTS is available on this system\"\"\"
        try:
            if self.system == "Linux":
                # Try espeak or festival
                result = subprocess.run(["which", "espeak"], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    return True
                    
                result = subprocess.run(["which", "festival"], 
                                      capture_output=True, text=True)
                return result.returncode == 0
                
            elif self.system == "Darwin":  # macOS
                return True  # macOS has built-in 'say' command
                
            elif self.system == "Windows":
                return True  # Windows has built-in SAPI
                
            return False
        except Exception as e:
            self.logger.error(f"TTS availability check failed: {e}")
            return False
    
    def speak(self, text: str, language: str = "th") -> bool:
        \"\"\"Speak the given text\"\"\"
        if not self.available or not text.strip():
            self.logger.warning("TTS not available or empty text")
            return False
        
        try:
            if self.system == "Linux":
                return self._speak_linux(text, language)
            elif self.system == "Darwin":
                return self._speak_macos(text, language)
            elif self.system == "Windows":
                return self._speak_windows(text, language)
            else:
                self.logger.error(f"TTS not supported on {self.system}")
                return False
                
        except Exception as e:
            self.logger.error(f"TTS failed: {e}")
            return False
    
    def _speak_linux(self, text: str, language: str) -> bool:
        \"\"\"Speak using Linux TTS\"\"\"
        try:
            # Try espeak first
            result = subprocess.run([
                "espeak", 
                "-s", "150",  # Speed
                "-v", "th" if language == "th" else "en",  # Voice
                text
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return True
            
            # Fallback to festival
            subprocess.run([
                "festival", "--tts"
            ], input=text, text=True, timeout=30)
            
            return True
            
        except subprocess.TimeoutExpired:
            self.logger.error("TTS timeout")
            return False
        except FileNotFoundError:
            self.logger.error("TTS command not found")
            return False
    
    def _speak_macos(self, text: str, language: str) -> bool:
        \"\"\"Speak using macOS say command\"\"\"
        try:
            voice = "Kanya" if language == "th" else "Alex"
            subprocess.run([
                "say", "-v", voice, "-r", "150", text
            ], timeout=30)
            return True
        except Exception as e:
            self.logger.error(f"macOS TTS failed: {e}")
            return False
    
    def _speak_windows(self, text: str, language: str) -> bool:
        \"\"\"Speak using Windows SAPI\"\"\"
        try:
            # Use PowerShell for Windows TTS
            ps_command = f'''
            Add-Type -AssemblyName System.Speech;
            $speak = New-Object System.Speech.Synthesis.SpeechSynthesizer;
            $speak.Speak("{text}");
            '''
            
            subprocess.run([
                "powershell", "-Command", ps_command
            ], timeout=30)
            return True
        except Exception as e:
            self.logger.error(f"Windows TTS failed: {e}")
            return False
    
    def is_available(self) -> bool:
        \"\"\"Check if TTS is available\"\"\"
        return self.available

# Fallback print-based TTS for testing
class PrintTTS:
    \"\"\"Fallback TTS that prints text instead of speaking\"\"\"
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🗣️  Using Print TTS (fallback)")
    
    def speak(self, text: str, language: str = "th") -> bool:
        \"\"\"Print text instead of speaking\"\"\"
        print(f"🗣️  JARVIS: {text}")
        return True
    
    def is_available(self) -> bool:
        return True
"""
    
    # 3. Voice Controller
    voice_controller = """#!/usr/bin/env python3
\"\"\"
🎮 Voice Controller for JARVIS
ควบคุมการทำงานของระบบเสียงทั้งหมด
\"\"\"

import logging
from typing import Optional, Callable
import threading
import time

from .speech_recognizer import SimpleSpeechRecognizer
from .text_to_speech import SimpleTextToSpeech, PrintTTS

class VoiceController:
    \"\"\"Main voice controller for JARVIS\"\"\"
    
    def __init__(self, use_fallback_tts: bool = False):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.speech_recognizer = SimpleSpeechRecognizer()
        
        if use_fallback_tts or not SimpleTextToSpeech().is_available():
            self.tts = PrintTTS()
        else:
            self.tts = SimpleTextToSpeech()
        
        # Voice settings
        self.wake_word = "hey jarvis"
        self.listening = False
        self.language = "th"
        
        # Callbacks
        self.on_command_received: Optional[Callable] = None
        self.on_wake_word_detected: Optional[Callable] = None
        
        self.logger.info("🎮 Voice Controller initialized")
    
    def set_command_callback(self, callback: Callable[[str], str]):
        \"\"\"Set callback for when voice command is received\"\"\"
        self.on_command_received = callback
    
    def set_wake_word_callback(self, callback: Callable):
        \"\"\"Set callback for when wake word is detected\"\"\"
        self.on_wake_word_detected = callback
    
    def listen_for_command(self, duration: float = 5.0) -> Optional[str]:
        \"\"\"Listen for a voice command\"\"\"
        if not self.speech_recognizer.is_available():
            self.logger.warning("🔇 Speech recognition not available")
            return None
        
        self.logger.info("🎤 Listening for command...")
        text = self.speech_recognizer.recognize_from_mic(duration)
        
        if text:
            self.logger.info(f"📝 Command received: {text}")
            return text.lower().strip()
        
        return None
    
    def speak_response(self, text: str, language: str = None) -> bool:
        \"\"\"Speak a response\"\"\"
        if not text:
            return False
            
        lang = language or self.language
        self.logger.info(f"🗣️  Speaking: {text}")
        
        return self.tts.speak(text, lang)
    
    def process_voice_command(self, command: str) -> str:
        \"\"\"Process a voice command and return response\"\"\"
        if not command:
            return "ไม่ได้ยินคำสั่งครับ"
        
        command = command.lower().strip()
        
        # Basic commands
        if "สวัสดี" in command or "hello" in command:
            return "สวัสดีครับ! ผมคือ JARVIS ยินดีที่ได้รู้จักครับ"
        
        elif "เวลา" in command or "time" in command:
            import datetime
            now = datetime.datetime.now()
            return f"ตอนนี้เวลา {now.strftime('%H:%M')} นาฬิกาครับ"
        
        elif "วันที่" in command or "date" in command:
            import datetime
            today = datetime.date.today()
            return f"วันนี้วันที่ {today.strftime('%d/%m/%Y')} ครับ"
        
        elif "ชื่อ" in command or "name" in command:
            return "ผมชื่อ JARVIS ครับ เป็น Voice Assistant ที่พร้อมช่วยเหลือคุณ"
        
        elif "ขอบคุณ" in command or "thank" in command:
            return "ยินดีครับ! มีอะไรให้ช่วยอีกไหมครับ"
        
        elif "ลาก่อน" in command or "goodbye" in command:
            return "ลาก่อนครับ! แล้วพบกันใหม่ครับ"
        
        else:
            # If callback is set, use it
            if self.on_command_received:
                try:
                    return self.on_command_received(command)
                except Exception as e:
                    self.logger.error(f"Command callback failed: {e}")
                    return "ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล"
            else:
                return f"ผมได้ยิน '{command}' แล้วครับ แต่ยังไม่เข้าใจคำสั่งนี้"
    
    def start_conversation(self):
        \"\"\"Start a voice conversation\"\"\"
        self.speak_response("สวัสดีครับ! ผมคือ JARVIS พร้อมรับคำสั่งจากคุณแล้วครับ")
        
        while True:
            self.speak_response("กรุณาพูดคำสั่งครับ")
            
            command = self.listen_for_command(duration=5.0)
            
            if not command:
                self.speak_response("ไม่ได้ยินครับ ลองใหม่อีกครั้งได้ครับ")
                continue
            
            if "หยุด" in command or "ปิด" in command or "stop" in command:
                self.speak_response("ลาก่อนครับ!")
                break
            
            response = self.process_voice_command(command)
            self.speak_response(response)
    
    def get_status(self) -> dict:
        \"\"\"Get voice system status\"\"\"
        return {
            "speech_recognition": self.speech_recognizer.is_available(),
            "text_to_speech": self.tts.is_available(),
            "language": self.language,
            "wake_word": self.wake_word
        }
"""
    
    # Write files
    files_to_write = [
        ("src/voice/speech_recognizer.py", simple_recognizer),
        ("src/voice/text_to_speech.py", simple_tts),
        ("src/voice/voice_controller.py", voice_controller)
    ]
    
    for file_path, content in files_to_write:
        full_path = Path(file_path)
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(full_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print(f"✅ Created: {file_path}")
    
    # Create __init__.py
    init_content = '''#!/usr/bin/env python3
"""
🎙️ JARVIS Voice Processing Module
"""

from .speech_recognizer import SimpleSpeechRecognizer
from .text_to_speech import SimpleTextToSpeech, PrintTTS
from .voice_controller import VoiceController

__all__ = [
    'SimpleSpeechRecognizer',
    'SimpleTextToSpeech', 
    'PrintTTS',
    'VoiceController'
]
'''
    
    with open("src/voice/__init__.py", 'w', encoding='utf-8') as f:
        f.write(init_content)
    
    print("✅ Created: src/voice/__init__.py")
    
    return True

if __name__ == "__main__":
    print("🎙️ Creating Simple Voice Interface for JARVIS...")
    
    if create_simple_voice_system():
        print("🎉 Simple Voice Interface created successfully!")
        print("📝 Next steps:")
        print("   1. Test voice components")
        print("   2. Integrate with JARVIS GUI")
        print("   3. Test full voice conversation")
    else:
        print("❌ Failed to create voice interface")