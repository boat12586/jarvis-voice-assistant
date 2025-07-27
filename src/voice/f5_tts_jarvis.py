#!/usr/bin/env python3
"""
🗣️ F5-TTS JARVIS Voice Synthesis
ระบบสังเคราะห์เสียง JARVIS ด้วย F5-TTS และเอฟเฟกต์เสียงขั้นสูง
"""

import logging
import torch
import numpy as np
import soundfile as sf
import tempfile
import os
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import time
import threading
import queue

# Audio processing libraries
try:
    import librosa
    import sounddevice as sd
    from scipy import signal
    from scipy.io import wavfile
    AUDIO_PROCESSING_AVAILABLE = True
except ImportError:
    AUDIO_PROCESSING_AVAILABLE = False
    logging.warning("Audio processing libraries not available")

# F5-TTS related imports (fallback if not available)
try:
    # Try to import F5-TTS (if installed)
    from f5_tts import F5TTS
    F5_TTS_AVAILABLE = True
except ImportError:
    F5_TTS_AVAILABLE = False
    logging.warning("F5-TTS not available - using alternative TTS")


class JarvisVoiceEffects:
    """เอฟเฟกต์เสียง JARVIS"""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(__name__)
    
    def apply_jarvis_effects(self, audio: np.ndarray, effects_config: Dict[str, float] = None) -> np.ndarray:
        """ใส่เอฟเฟกต์เสียง JARVIS"""
        if not AUDIO_PROCESSING_AVAILABLE:
            return audio
        
        effects_config = effects_config or {
            'metallic': 0.3,
            'reverb': 0.4,
            'bass_boost': 0.2,
            'clarity': 0.3,
            'robotic': 0.2
        }
        
        processed_audio = audio.copy()
        
        try:
            # 1. Metallic effect (ring modulation)
            if effects_config.get('metallic', 0) > 0:
                processed_audio = self._apply_metallic_effect(
                    processed_audio, effects_config['metallic']
                )
            
            # 2. Bass boost
            if effects_config.get('bass_boost', 0) > 0:
                processed_audio = self._apply_bass_boost(
                    processed_audio, effects_config['bass_boost']
                )
            
            # 3. Clarity enhancement
            if effects_config.get('clarity', 0) > 0:
                processed_audio = self._apply_clarity(
                    processed_audio, effects_config['clarity']
                )
            
            # 4. Reverb
            if effects_config.get('reverb', 0) > 0:
                processed_audio = self._apply_reverb(
                    processed_audio, effects_config['reverb']
                )
            
            # 5. Robotic effect (vocoder-like)
            if effects_config.get('robotic', 0) > 0:
                processed_audio = self._apply_robotic_effect(
                    processed_audio, effects_config['robotic']
                )
            
            # Normalize to prevent clipping
            processed_audio = self._normalize_audio(processed_audio)
            
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"❌ Effect processing failed: {e}")
            return audio  # Return original on error
    
    def _apply_metallic_effect(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """เอฟเฟกต์โลหะ (ring modulation)"""
        t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
        modulator_freq = 800  # Hz - characteristic JARVIS frequency
        modulator = np.sin(2 * np.pi * modulator_freq * t)
        
        # Apply ring modulation with intensity control
        metallic_audio = audio * (1 + intensity * modulator)
        return metallic_audio
    
    def _apply_bass_boost(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """เพิ่มเสียงเบส"""
        # Low-pass filter for bass frequencies
        nyquist = self.sample_rate // 2
        cutoff = 200  # Hz
        b, a = signal.butter(2, cutoff / nyquist, btype='low')
        
        bass = signal.filtfilt(b, a, audio)
        return audio + intensity * bass
    
    def _apply_clarity(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """เพิ่มความชัดเจน"""
        # High-pass filter for clarity
        nyquist = self.sample_rate // 2
        cutoff = 2000  # Hz
        b, a = signal.butter(2, cutoff / nyquist, btype='high')
        
        clarity = signal.filtfilt(b, a, audio)
        return audio + intensity * 0.3 * clarity
    
    def _apply_reverb(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """เอฟเฟกต์ห้อง"""
        # Simple reverb using delay and feedback
        delay_samples = int(0.1 * self.sample_rate)  # 100ms delay
        reverb_audio = np.zeros_like(audio)
        
        for i in range(len(audio)):
            reverb_audio[i] = audio[i]
            if i >= delay_samples:
                reverb_audio[i] += intensity * 0.3 * reverb_audio[i - delay_samples]
        
        return reverb_audio
    
    def _apply_robotic_effect(self, audio: np.ndarray, intensity: float) -> np.ndarray:
        """เอฟเฟกต์หุ่นยนต์"""
        # Frequency domain processing for robotic effect
        fft = np.fft.fft(audio)
        
        # Emphasize certain frequency bands
        freqs = np.fft.fftfreq(len(audio), 1/self.sample_rate)
        
        # Create harmonic emphasis pattern
        for i, freq in enumerate(freqs):
            if 300 <= abs(freq) <= 3000:  # Voice frequency range
                if abs(freq) % 100 < 50:  # Emphasize multiples of 100Hz
                    fft[i] *= (1 + intensity)
        
        return np.real(np.fft.ifft(fft))
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """ปรับระดับเสียงให้เหมาะสม"""
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.8  # Prevent clipping with 0.8 headroom
        return audio


class F5TTSJarvis:
    """ระบบสังเคราะห์เสียง JARVIS ด้วย F5-TTS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # TTS settings
        self.sample_rate = self.config.get('sample_rate', 22050)
        self.device = self.config.get('device', 'cpu')
        self.model_path = self.config.get('model_path', 'models/f5_tts')
        
        # JARVIS voice settings
        self.voice_effects = JarvisVoiceEffects(self.sample_rate)
        self.jarvis_effects_config = self.config.get('jarvis_effects', {
            'metallic': 0.3,
            'reverb': 0.4,
            'bass_boost': 0.2,
            'clarity': 0.3,
            'robotic': 0.2
        })
        
        # Models
        self.f5_model = None
        self.fallback_tts = None
        
        # Audio output
        self.audio_queue = queue.Queue()
        self.is_speaking = False
        
        self._initialize_tts()
    
    def _initialize_tts(self):
        """เริ่มต้นระบบ TTS"""
        self.logger.info("🗣️ Initializing JARVIS voice synthesis...")
        
        # Try F5-TTS first
        if F5_TTS_AVAILABLE:
            try:
                self._initialize_f5_tts()
            except Exception as e:
                self.logger.warning(f"⚠️ F5-TTS initialization failed: {e}")
        
        # Fallback to system TTS
        if not self.f5_model:
            self._initialize_fallback_tts()
    
    def _initialize_f5_tts(self):
        """เริ่มต้น F5-TTS"""
        try:
            model_path = Path(self.model_path)
            if model_path.exists():
                self.f5_model = F5TTS.from_pretrained(str(model_path))
                self.logger.info("✅ F5-TTS model loaded")
            else:
                self.logger.warning(f"⚠️ F5-TTS model not found at {model_path}")
                
        except Exception as e:
            self.logger.error(f"❌ F5-TTS loading failed: {e}")
            self.f5_model = None
    
    def _initialize_fallback_tts(self):
        """เริ่มต้น TTS สำรอง"""
        try:
            # Try gTTS for better quality
            try:
                from gtts import gTTS
                self.fallback_tts = 'gtts'
                self.logger.info("✅ gTTS fallback ready")
                return
            except ImportError:
                pass
            
            # Try pyttsx3
            try:
                import pyttsx3
                engine = pyttsx3.init()
                # Configure for JARVIS-like voice
                voices = engine.getProperty('voices')
                for voice in voices:
                    if 'english' in voice.name.lower() and 'male' in voice.name.lower():
                        engine.setProperty('voice', voice.id)
                        break
                
                engine.setProperty('rate', 180)  # Slightly slower
                engine.setProperty('volume', 0.9)
                
                self.fallback_tts = engine
                self.logger.info("✅ pyttsx3 fallback ready")
                return
            except ImportError:
                pass
            
            # System command fallback
            self.fallback_tts = 'system'
            self.logger.info("✅ System TTS fallback ready")
            
        except Exception as e:
            self.logger.error(f"❌ Fallback TTS initialization failed: {e}")
    
    def synthesize_speech(
        self, 
        text: str, 
        language: str = 'en',
        apply_effects: bool = True,
        save_path: Optional[str] = None
    ) -> Optional[np.ndarray]:
        """สังเคราะห์เสียงพูด"""
        
        if not text.strip():
            return None
        
        self.logger.info(f"🗣️ Synthesizing: {text[:50]}...")
        start_time = time.time()
        
        try:
            # Generate base audio
            audio = None
            
            if self.f5_model:
                audio = self._synthesize_f5_tts(text, language)
            
            if audio is None:
                audio = self._synthesize_fallback(text, language)
            
            if audio is None:
                self.logger.error("❌ All TTS methods failed")
                return None
            
            # Apply JARVIS effects
            if apply_effects and AUDIO_PROCESSING_AVAILABLE:
                audio = self.voice_effects.apply_jarvis_effects(
                    audio, self.jarvis_effects_config
                )
            
            # Save if requested
            if save_path:
                sf.write(save_path, audio, self.sample_rate)
                self.logger.debug(f"💾 Audio saved to {save_path}")
            
            synthesis_time = time.time() - start_time
            self.logger.info(f"✅ Speech synthesized in {synthesis_time:.2f}s")
            
            return audio
            
        except Exception as e:
            self.logger.error(f"❌ Speech synthesis failed: {e}")
            return None
    
    def _synthesize_f5_tts(self, text: str, language: str) -> Optional[np.ndarray]:
        """ใช้ F5-TTS สังเคราะห์เสียง"""
        if not self.f5_model:
            return None
        
        try:
            # Generate with F5-TTS
            audio = self.f5_model.synthesize(
                text=text,
                voice_preset="jarvis",  # If available
                language=language
            )
            
            # Convert to numpy array if needed
            if torch.is_tensor(audio):
                audio = audio.cpu().numpy()
            
            return audio
            
        except Exception as e:
            self.logger.error(f"❌ F5-TTS synthesis failed: {e}")
            return None
    
    def _synthesize_fallback(self, text: str, language: str) -> Optional[np.ndarray]:
        """ใช้ TTS สำรอง"""
        try:
            if self.fallback_tts == 'gtts':
                return self._synthesize_gtts(text, language)
            elif hasattr(self.fallback_tts, 'say'):  # pyttsx3
                return self._synthesize_pyttsx3(text)
            elif self.fallback_tts == 'system':
                return self._synthesize_system(text)
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Fallback synthesis failed: {e}")
            return None
    
    def _synthesize_gtts(self, text: str, language: str) -> Optional[np.ndarray]:
        """ใช้ gTTS สังเคราะห์เสียง"""
        try:
            from gtts import gTTS
            
            # Convert language code
            lang_code = 'th' if language == 'th' else 'en'
            
            tts = gTTS(text=text, lang=lang_code, slow=False)
            
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_file:
                tts.save(tmp_file.name)
                
                # Load and convert to numpy
                audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return audio
            
        except Exception as e:
            self.logger.error(f"❌ gTTS synthesis failed: {e}")
            return None
    
    def _synthesize_pyttsx3(self, text: str) -> Optional[np.ndarray]:
        """ใช้ pyttsx3 สังเคราะห์เสียง"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                self.fallback_tts.save_to_file(text, tmp_file.name)
                self.fallback_tts.runAndWait()
                
                # Load audio
                audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return audio
            
        except Exception as e:
            self.logger.error(f"❌ pyttsx3 synthesis failed: {e}")
            return None
    
    def _synthesize_system(self, text: str) -> Optional[np.ndarray]:
        """ใช้ระบบ TTS"""
        try:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                # Use espeak on Linux
                import subprocess
                subprocess.run([
                    'espeak', '-w', tmp_file.name, '-s', '160', '-p', '30', text
                ], check=True)
                
                # Load audio
                audio, sr = librosa.load(tmp_file.name, sr=self.sample_rate)
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return audio
            
        except Exception as e:
            self.logger.error(f"❌ System TTS failed: {e}")
            return None
    
    def speak(self, text: str, language: str = 'en', blocking: bool = False):
        """พูดข้อความ"""
        if not AUDIO_PROCESSING_AVAILABLE:
            self.logger.warning("⚠️ Audio playback not available")
            return
        
        if self.is_speaking and not blocking:
            self.logger.warning("⚠️ Already speaking")
            return
        
        def _speak_thread():
            try:
                self.is_speaking = True
                
                # Synthesize audio
                audio = self.synthesize_speech(text, language)
                
                if audio is not None:
                    # Play audio
                    sd.play(audio, samplerate=self.sample_rate)
                    sd.wait()  # Wait for playback to finish
                
            except Exception as e:
                self.logger.error(f"❌ Speech playback failed: {e}")
            finally:
                self.is_speaking = False
        
        if blocking:
            _speak_thread()
        else:
            threading.Thread(target=_speak_thread, daemon=True).start()
    
    def get_status(self) -> Dict[str, Any]:
        """สถานะระบบ TTS"""
        return {
            "f5_tts_available": F5_TTS_AVAILABLE,
            "f5_model_loaded": self.f5_model is not None,
            "fallback_tts": str(type(self.fallback_tts).__name__) if self.fallback_tts else None,
            "audio_processing": AUDIO_PROCESSING_AVAILABLE,
            "is_speaking": self.is_speaking,
            "sample_rate": self.sample_rate,
            "effects_config": self.jarvis_effects_config
        }


def test_f5_tts_jarvis():
    """ทดสอบระบบเสียง JARVIS"""
    print("🧪 Testing F5-TTS JARVIS Voice System...")
    
    # Create TTS system
    config = {
        'sample_rate': 22050,
        'device': 'cpu',
        'jarvis_effects': {
            'metallic': 0.4,
            'reverb': 0.3,
            'bass_boost': 0.2,
            'clarity': 0.4,
            'robotic': 0.3
        }
    }
    
    jarvis_tts = F5TTSJarvis(config)
    
    # Show status
    status = jarvis_tts.get_status()
    print(f"📊 TTS Status: {status}")
    
    # Test synthesis
    test_phrases = [
        ("Good morning, sir. I am JARVIS, your personal assistant.", "en"),
        ("สวัสดีครับ ผมคือจาร์วิส ผู้ช่วยส่วนตัวของท่าน", "th"),
        ("How may I assist you today?", "en"),
        ("มีอะไรให้ผมช่วยเหลือวันนี้ไหมครับ", "th")
    ]
    
    for i, (text, lang) in enumerate(test_phrases):
        print(f"\n🗣️ Testing phrase {i+1}: {text[:50]}...")
        
        # Synthesize (don't play, just test synthesis)
        audio = jarvis_tts.synthesize_speech(
            text, 
            language=lang,
            apply_effects=True,
            save_path=f"test_jarvis_voice_{i+1}.wav"
        )
        
        if audio is not None:
            print(f"   ✅ Synthesis successful ({len(audio)} samples)")
        else:
            print(f"   ❌ Synthesis failed")
    
    print("\n✅ F5-TTS JARVIS test completed!")
    return jarvis_tts


if __name__ == "__main__":
    test_f5_tts_jarvis()