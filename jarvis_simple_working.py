#!/usr/bin/env python3
"""
🤖 JARVIS Simple Working Voice Assistant
พัฒนาต่อระบบ JARVIS ให้ทำงานได้จริง
"""

import logging
import time
import threading
import queue
from pathlib import Path
from typing import Dict, Any, Optional
import sounddevice as sd
import numpy as np
import yaml

# Basic AI Response System
class SimpleAI:
    """AI ตอบสนองอย่างง่าย"""
    
    def __init__(self):
        self.responses = {
            'en': {
                'hello': "Hello! I'm JARVIS, your voice assistant. How can I help you?",
                'time': f"The current time is {time.strftime('%H:%M:%S')}",
                'weather': "I don't have weather data access yet, but I'm working on it!",
                'goodbye': "Goodbye! Have a great day!",
                'default': "I understand. How can I assist you further?"
            },
            'th': {
                'hello': "สวัสดีครับ ผม JARVIS ผู้ช่วยเสียงของคุณ ต้องการให้ช่วยอะไรไหมครับ",
                'time': f"เวลาตอนนี้คือ {time.strftime('%H:%M:%S')} นาฬิกาครับ",
                'weather': "ตอนนี้ผมยังเข้าถึงข้อมูลสภาพอากาศไม่ได้ครับ แต่กำลังพัฒนาอยู่ครับ",
                'goodbye': "ลาก่อนครับ ขอให้มีวันที่ดีครับ",
                'default': "เข้าใจแล้วครับ มีอะไรให้ช่วยอีกไหมครับ"
            }
        }
    
    def get_response(self, text: str, language: str = 'en') -> str:
        """สร้างการตอบสนอง"""
        text_lower = text.lower()
        lang_responses = self.responses.get(language, self.responses['en'])
        
        # Simple keyword matching
        if any(word in text_lower for word in ['hello', 'hi', 'hey', 'สวัสดี']):
            return lang_responses['hello']
        elif any(word in text_lower for word in ['time', 'เวลา', 'กี่โมง']):
            return lang_responses['time']
        elif any(word in text_lower for word in ['weather', 'อากาศ', 'สภาพอากาศ']):
            return lang_responses['weather']
        elif any(word in text_lower for word in ['bye', 'goodbye', 'ลาก่อน', 'บาย']):
            return lang_responses['goodbye']
        else:
            return lang_responses['default']

# Voice Activity Detection
class SimpleVoiceDetector:
    """ตรวจจับเสียงพูดอย่างง่าย"""
    
    def __init__(self, sample_rate=16000, threshold=0.01):
        self.sample_rate = sample_rate
        self.threshold = threshold
        self.is_speaking = False
        
    def detect_speech(self, audio_data):
        """ตรวจสอบว่ามีการพูดหรือไม่"""
        # คำนวณระดับเสียง (RMS)
        rms = np.sqrt(np.mean(audio_data**2))
        return rms > self.threshold

# Text-to-Speech (Mock)
class SimpleTTS:
    """ระบบแปลงข้อความเป็นเสียงแบบง่าย"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def speak(self, text: str, language: str = 'en'):
        """พูดข้อความ (mock)"""
        self.logger.info(f"🗣️ JARVIS says ({language}): {text}")
        # TODO: เพิ่ม TTS จริงในอนาคต
        print(f"🤖 JARVIS: {text}")

# Main JARVIS System
class JarvisSimpleWorking:
    """ระบบ JARVIS ที่ทำงานได้จริง"""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.logger = logging.getLogger(__name__)
        
        # โหลดการตั้งค่า
        self.config = self._load_config(config_path)
        
        # เริ่มต้นส่วนประกอบ
        self.ai = SimpleAI()
        self.voice_detector = SimpleVoiceDetector()
        self.tts = SimpleTTS()
        
        # สถานะระบบ
        self.is_active = False
        self.is_listening = False
        self.conversation_active = False
        
        # Audio settings
        self.sample_rate = 16000
        self.chunk_size = 1024
        self.audio_queue = queue.Queue()
        
        # Statistics
        self.stats = {
            "interactions": 0,
            "start_time": time.time()
        }
        
        self.logger.info("✅ JARVIS Simple Working System initialized")
    
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
        
        self.logger.info("🚀 Starting JARVIS Simple System...")
        
        try:
            # ตรวจสอบอุปกรณ์เสียง
            devices = sd.query_devices()
            self.logger.info(f"📱 Found {len(devices)} audio devices")
            
            self.is_active = True
            self.stats["start_time"] = time.time()
            
            self.logger.info("✅ JARVIS is now active!")
            self.logger.info("💬 Type 'listen' to start voice conversation")
            self.logger.info("💬 Type 'chat <message>' to send text message")
            self.logger.info("💬 Type 'quit' to exit")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System startup failed: {e}")
            return False
    
    def stop_system(self):
        """หยุดระบบ JARVIS"""
        if not self.is_active:
            return
        
        self.logger.info("🛑 Stopping JARVIS Simple System...")
        
        self.is_active = False
        self.is_listening = False
        self.conversation_active = False
        
        self.logger.info("✅ JARVIS stopped")
    
    def start_listening(self):
        """เริ่มฟังเสียง"""
        if self.is_listening:
            self.logger.warning("⚠️ Already listening")
            return
        
        self.logger.info("🎤 Starting to listen...")
        self.is_listening = True
        
        try:
            # เริ่ม recording thread
            self.record_thread = threading.Thread(target=self._record_audio)
            self.record_thread.daemon = True
            self.record_thread.start()
            
            self.logger.info("✅ Listening started - speak now!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to start listening: {e}")
            self.is_listening = False
    
    def stop_listening(self):
        """หยุดฟังเสียง"""
        self.logger.info("🔇 Stopping listening...")
        self.is_listening = False
    
    def _record_audio(self):
        """บันทึกเสียง"""
        def audio_callback(indata, frames, time, status):
            if status:
                self.logger.warning(f"Audio status: {status}")
            
            if self.is_listening:
                # ตรวจสอบว่ามีเสียงพูดหรือไม่
                if self.voice_detector.detect_speech(indata.flatten()):
                    self.audio_queue.put(indata.copy())
        
        try:
            with sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                callback=audio_callback,
                blocksize=self.chunk_size
            ):
                while self.is_listening:
                    time.sleep(0.1)
                    
                    # ประมวลผลเสียงที่บันทึกได้
                    if not self.audio_queue.empty():
                        self._process_audio_queue()
                        
        except Exception as e:
            self.logger.error(f"❌ Audio recording error: {e}")
            self.is_listening = False
    
    def _process_audio_queue(self):
        """ประมวลผลเสียงที่บันทึกได้"""
        audio_chunks = []
        
        # รวบรวมข้อมูลเสียง
        while not self.audio_queue.empty():
            try:
                chunk = self.audio_queue.get_nowait()
                audio_chunks.append(chunk)
            except queue.Empty:
                break
        
        if audio_chunks:
            # จำลองการแปลงเสียงเป็นข้อความ (Mock Speech Recognition)
            self.logger.info("🔍 Processing speech...")
            
            # สำหรับการทดสอบ - จำลองการรู้จำเสียง
            mock_text = "Hello JARVIS"
            mock_language = "en"
            
            self._handle_recognized_speech(mock_text, mock_language)
    
    def _handle_recognized_speech(self, text: str, language: str):
        """จัดการข้อความที่รู้จำได้"""
        self.logger.info(f"📝 Recognized: '{text}' ({language})")
        
        # สร้างการตอบสนอง
        response = self.ai.get_response(text, language)
        
        # แสดงการตอบสนอง
        self.tts.speak(response, language)
        
        # อัพเดทสถิติ
        self.stats["interactions"] += 1
        
        # หยุดฟังหลังจากตอบ
        self.stop_listening()
    
    def process_text_message(self, text: str, language: str = "en"):
        """ประมวลผลข้อความ"""
        self.logger.info(f"💬 Processing text: '{text}'")
        
        # สร้างการตอบสนอง
        response = self.ai.get_response(text, language)
        
        # แสดงการตอบสนอง
        self.tts.speak(response, language)
        
        # อัพเดทสถิติ
        self.stats["interactions"] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """สถานะระบบ"""
        uptime = time.time() - self.stats["start_time"]
        
        return {
            "active": self.is_active,
            "listening": self.is_listening,
            "conversation_active": self.conversation_active,
            "interactions": self.stats["interactions"],
            "uptime_seconds": uptime
        }
    
    def run_interactive(self):
        """เรียกใช้แบบ Interactive"""
        if not self.start_system():
            return
        
        print("\n🎙️ JARVIS Interactive Mode")
        print("Commands:")
        print("  listen          - Start voice listening")
        print("  chat <message>  - Send text message")
        print("  status          - Show system status")
        print("  quit            - Exit")
        
        try:
            while self.is_active:
                user_input = input("\n> ").strip()
                
                if not user_input:
                    continue
                
                if user_input.lower() == 'quit':
                    break
                elif user_input.lower() == 'listen':
                    self.start_listening()
                    print("🎤 Listening... (will auto-stop after processing)")
                    time.sleep(5)  # Listen for 5 seconds
                    self.stop_listening()
                elif user_input.lower().startswith('chat '):
                    message = user_input[5:]
                    self.process_text_message(message)
                elif user_input.lower() == 'status':
                    status = self.get_status()
                    print(f"📊 Status: {status}")
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
    
    print("🤖 JARVIS Simple Working Voice Assistant")
    print("======================================")
    
    # สร้างและเรียกใช้ JARVIS
    jarvis = JarvisSimpleWorking()
    jarvis.run_interactive()


if __name__ == "__main__":
    main()