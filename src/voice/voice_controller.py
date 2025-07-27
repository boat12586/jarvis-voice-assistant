#!/usr/bin/env python3
"""
🎮 Voice Controller for JARVIS
ควบคุมการทำงานของระบบเสียงทั้งหมด
"""

import logging
from typing import Optional, Callable
import threading
import time

from .speech_recognizer import SimpleSpeechRecognizer
from .text_to_speech import SimpleTextToSpeech, PrintTTS
from .simple_wake_word import SimpleWakeWordDetector
from .advanced_command_system import AdvancedCommandSystem

class VoiceController:
    """Main voice controller for JARVIS"""
    
    def __init__(self, use_fallback_tts: bool = False):
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.speech_recognizer = SimpleSpeechRecognizer()
        
        if use_fallback_tts or not SimpleTextToSpeech().is_available():
            self.tts = PrintTTS()
        else:
            self.tts = SimpleTextToSpeech()
            
        # Initialize wake word detector
        self.wake_word_detector = SimpleWakeWordDetector()
        self.wake_word_detector.on_wake_word = self._on_wake_word_detected
        
        # Initialize advanced command system
        self.command_system = AdvancedCommandSystem()
        self.logger.info("🎤 Advanced Command System integrated")
        
        # Voice settings
        self.wake_word = "hey jarvis"
        self.listening = False
        self.language = "th"
        self.voice_detected = False  # Track voice detection state
        
        # Callbacks
        self.on_command_received: Optional[Callable] = None
        self.on_wake_word_detected: Optional[Callable] = None
        
        self.logger.info("🎮 Voice Controller initialized")
    
    def set_command_callback(self, callback: Callable[[str], str]):
        """Set callback for when voice command is received"""
        self.on_command_received = callback
    
    def set_wake_word_callback(self, callback: Callable):
        """Set callback for when wake word is detected"""
        self.on_wake_word_detected = callback
    
    def listen_for_command(self, duration: float = 5.0) -> Optional[str]:
        """Listen for a voice command"""
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
        """Speak a response"""
        if not text:
            return False
            
        lang = language or self.language
        self.logger.info(f"🗣️  Speaking: {text}")
        
        return self.tts.speak(text, lang)
    
    def process_voice_command(self, command: str) -> str:
        """Process a voice command and return response"""
        if not command:
            return "ไม่ได้ยินคำสั่งครับ"
        
        command_clean = command.strip()
        self.logger.info(f"🎤 Processing command: '{command_clean}'")
        
        # Detect language
        language = self._detect_command_language(command_clean)
        
        try:
            # Use advanced command system first
            command_match = self.command_system.process_voice_input(command_clean, language)
            
            if command_match:
                self.logger.info(f"✅ Command matched: {command_match.command.command_id} (confidence: {command_match.confidence:.2f})")
                
                # Execute the command
                result = self.command_system.execute_command(command_match)
                
                if result.get('success'):
                    response = result.get('result', {}).get('response', 'คำสั่งดำเนินการเสร็จสิ้น')
                    
                    # Handle special actions
                    if result.get('result', {}).get('action') == 'shutdown':
                        # Signal shutdown to application
                        if hasattr(self, 'on_shutdown_requested'):
                            self.on_shutdown_requested()
                    
                    return response
                else:
                    error = result.get('error', 'เกิดข้อผิดพลาดในการประมวลผล')
                    self.logger.error(f"Command execution error: {error}")
                    return f"ขออภัยครับ {error}"
            
            # Fallback to legacy command processing
            return self._process_legacy_command(command_clean, language)
            
        except Exception as e:
            self.logger.error(f"Command processing error: {e}")
            if language == "th":
                return "ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล"
            else:
                return "Sorry, an error occurred while processing your command"
    
    def _detect_command_language(self, command: str) -> str:
        """Detect command language"""
        # Count Thai characters
        thai_chars = sum(1 for char in command if 0x0E00 <= ord(char) <= 0x0E7F)
        if thai_chars > len(command) * 0.3:
            return "th"
        return "en"
    
    def _process_legacy_command(self, command: str, language: str) -> str:
        """Process legacy commands as fallback"""
        command_lower = command.lower()
        
        # Basic legacy commands
        if "สวัสดี" in command_lower or "hello" in command_lower:
            return "สวัสดีครับ! ผมคือ JARVIS ยินดีที่ได้รู้จักครับ"
        
        elif "ขอบคุณ" in command_lower or "thank" in command_lower:
            return "ยินดีครับ! มีอะไรให้ช่วยอีกไหมครับ"
        
        # If external callback is set, use it
        if self.on_command_received:
            try:
                return self.on_command_received(command)
            except Exception as e:
                self.logger.error(f"Legacy command callback failed: {e}")
                if language == "th":
                    return "ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล"
                else:
                    return "Sorry, an error occurred while processing your command"
        
        # Default fallback
        if language == "th":
            return f"ผมได้ยิน '{command}' แล้วครับ แต่ยังไม่เข้าใจคำสั่งนี้ ลองพูด 'ช่วยเหลือ' เพื่อดูคำสั่งที่ใช้ได้"
        else:
            return f"I heard '{command}' but I don't understand this command. Try saying 'help' to see available commands"
    
    def start_conversation(self):
        """Start a voice conversation"""
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
        """Get voice system status"""
        return {
            "speech_recognition": self.speech_recognizer.is_available(),
            "text_to_speech": self.tts.is_available(),
            "wake_word_detection": self.wake_word_detector.get_status(),
            "command_system": self.command_system.get_statistics(),
            "available_commands": len(self.command_system.commands),
            "language": self.language,
            "wake_word": self.wake_word,
            "voice_detected": self.voice_detected,
            "listening": self.listening
        }
    
    def shutdown(self):
        """Shutdown voice controller and cleanup resources"""
        self.logger.info("🛑 Shutting down Voice Controller...")
        self.listening = False
        self.voice_detected = False
        
        # Stop wake word detection
        self.stop_wake_word_detection()
        
        # Cleanup speech recognizer if it has cleanup method
        if hasattr(self.speech_recognizer, 'cleanup'):
            try:
                self.speech_recognizer.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up speech recognizer: {e}")
        
        # Cleanup TTS if it has cleanup method
        if hasattr(self.tts, 'cleanup'):
            try:
                self.tts.cleanup()
            except Exception as e:
                self.logger.error(f"Error cleaning up TTS: {e}")
        
        self.logger.info("✅ Voice Controller shutdown complete")
    
    def start_listening(self):
        """Start listening for voice input"""
        self.listening = True
        self.voice_detected = False
        self.logger.info("🎤 Started listening for voice input")
    
    def stop_listening(self):
        """Stop listening for voice input"""
        self.listening = False
        self.voice_detected = False
        self.wake_word_detector.stop_listening()
        self.logger.info("🔇 Stopped listening for voice input")
    
    def _on_wake_word_detected(self, text: str, confidence: float):
        """Handle wake word detection"""
        self.logger.info(f"🚨 Wake word detected: '{text}' (confidence: {confidence:.2f})")
        self.voice_detected = True
        
        # Trigger wake word callback if set
        if self.on_wake_word_detected:
            self.on_wake_word_detected()
        
        # Start listening for command after wake word
        self.speak_response("ครับ มีอะไรให้ช่วยไหมครับ")
        
        # Listen for command
        command = self.listen_for_command(duration=8.0)
        if command:
            response = self.process_voice_command(command)
            self.speak_response(response)
        else:
            self.speak_response("ไม่ได้ยินคำสั่งครับ ลองเรียก JARVIS ใหม่ได้ครับ")
    
    def start_wake_word_detection(self):
        """Start wake word detection"""
        if self.wake_word_detector.start_listening():
            self.logger.info("🎤 Wake word detection started")
            return True
        else:
            self.logger.error("❌ Failed to start wake word detection")
            return False
    
    def stop_wake_word_detection(self):
        """Stop wake word detection"""
        self.wake_word_detector.stop_listening()
        self.logger.info("🛑 Wake word detection stopped")
