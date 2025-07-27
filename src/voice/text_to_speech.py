#!/usr/bin/env python3
"""
🗣️ Simple Text-to-Speech for JARVIS
ใช้ระบบ TTS พื้นฐานของ OS
"""

import logging
import subprocess
import platform
from typing import Optional
import tempfile
import os

class SimpleTextToSpeech:
    """Simple TTS using system commands"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.system = platform.system()
        self.available = self._check_availability()
        
        if self.available:
            self.logger.info(f"🗣️  TTS ready on {self.system}")
        else:
            self.logger.warning("🔇 TTS not available")
    
    def _check_availability(self) -> bool:
        """Check if TTS is available on this system"""
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
        """Speak the given text"""
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
        """Speak using Linux TTS"""
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
        """Speak using macOS say command"""
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
        """Speak using Windows SAPI"""
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
        """Check if TTS is available"""
        return self.available

# Fallback print-based TTS for testing
class PrintTTS:
    """Fallback TTS that prints text instead of speaking"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.logger.info("🗣️  Using Print TTS (fallback)")
    
    def speak(self, text: str, language: str = "th") -> bool:
        """Print text instead of speaking"""
        print(f"🗣️  JARVIS: {text}")
        return True
    
    def is_available(self) -> bool:
        return True
