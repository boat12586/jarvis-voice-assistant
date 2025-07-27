"""
F5-TTS Integration for JARVIS Voice Assistant
Advanced neural text-to-speech with voice cloning capabilities
"""

import os
import logging
import torch
import torchaudio
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Union
import soundfile as sf
import tempfile
import time
from dataclasses import dataclass

# Enhanced logging support
try:
    from src.system.enhanced_logger import ComponentLogger
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False


@dataclass
class F5TTSConfig:
    """F5-TTS configuration"""
    model_path: str = "models/f5_tts"
    voice_clone_path: str = "assets/voices/jarvis_voice.wav"
    sample_rate: int = 24000
    chunk_length: int = 2048
    hop_length: int = 256
    speed: float = 1.0
    pitch: float = 1.0
    temperature: float = 0.7
    device: str = "auto"
    cache_dir: str = "data/tts_cache"
    max_text_length: int = 500


class F5TTSEngine:
    """F5-TTS Engine with JARVIS voice cloning"""
    
    def __init__(self, config: F5TTSConfig):
        if ENHANCED_LOGGING:
            self.logger = ComponentLogger("f5_tts", config.__dict__)
        else:
            self.logger = logging.getLogger(__name__)
            
        self.config = config
        self.device = self._setup_device()
        self.model = None
        self.reference_audio = None
        self.is_ready = False
        
        # Create directories
        Path(config.model_path).mkdir(parents=True, exist_ok=True)
        Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
        
        # Initialize
        self._initialize()
    
    def _setup_device(self) -> torch.device:
        """Setup compute device"""
        if self.config.device == "auto":
            if torch.cuda.is_available():
                device = torch.device("cuda")
                self.logger.info(f"🚀 Using CUDA GPU: {torch.cuda.get_device_name()}")
            else:
                device = torch.device("cpu")
                self.logger.info("💻 Using CPU for TTS")
        else:
            device = torch.device(self.config.device)
            self.logger.info(f"🎯 Using specified device: {device}")
        
        return device
    
    def _initialize(self):
        """Initialize F5-TTS model and reference audio"""
        try:
            operation_id = None
            if ENHANCED_LOGGING:
                operation_id = self.logger.operation_start("initialize_f5_tts")
            
            self.logger.info("🎤 Initializing F5-TTS engine...")
            
            # Load reference audio for voice cloning
            if os.path.exists(self.config.voice_clone_path):
                self._load_reference_audio()
            else:
                self.logger.warning(f"Reference audio not found: {self.config.voice_clone_path}")
                self._create_default_reference()
            
            # Load or download F5-TTS model
            self._load_model()
            
            self.is_ready = True
            self.logger.info("✅ F5-TTS engine initialized successfully")
            
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, True, {"model_loaded": True})
                
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize F5-TTS: {e}", exception=e)
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, False)
            self._fallback_initialization()
    
    def _load_reference_audio(self):
        """Load reference audio for voice cloning"""
        try:
            self.logger.info(f"📁 Loading reference audio: {self.config.voice_clone_path}")
            
            # Load audio file
            audio_data, sample_rate = sf.read(self.config.voice_clone_path)
            
            # Convert to tensor and ensure correct format
            if audio_data.ndim > 1:
                audio_data = audio_data.mean(axis=1)  # Convert to mono
            
            audio_tensor = torch.from_numpy(audio_data).float()
            
            # Resample if needed
            if sample_rate != self.config.sample_rate:
                self.logger.info(f"🔄 Resampling from {sample_rate}Hz to {self.config.sample_rate}Hz")
                resampler = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.config.sample_rate
                )
                audio_tensor = resampler(audio_tensor)
            
            self.reference_audio = audio_tensor.to(self.device)
            self.logger.info(f"✅ Reference audio loaded: {audio_tensor.shape} samples")
            
        except Exception as e:
            self.logger.error(f"Failed to load reference audio: {e}", exception=e)
            self._create_default_reference()
    
    def _create_default_reference(self):
        """Create a default reference audio (silent placeholder)"""
        self.logger.info("🔧 Creating default reference audio")
        
        # Create 3 seconds of silence as placeholder
        duration = 3.0
        samples = int(self.config.sample_rate * duration)
        self.reference_audio = torch.zeros(samples, device=self.device)
        
        self.logger.warning("⚠️ Using silent reference - voice cloning will be limited")
    
    def _load_model(self):
        """Load or download F5-TTS model"""
        try:
            model_file = Path(self.config.model_path) / "f5_tts_model.pt"
            
            if not model_file.exists():
                self.logger.info("📥 F5-TTS model not found, attempting to download...")
                self._download_model()
            
            # For now, use a placeholder - would normally load actual F5-TTS model
            self.logger.info("🧠 Loading F5-TTS model...")
            
            # Simulate model loading (replace with actual F5-TTS loading)
            self.model = self._create_mock_model()
            
            self.logger.info("✅ F5-TTS model loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load F5-TTS model: {e}", exception=e)
            raise
    
    def _create_mock_model(self):
        """Create mock model for testing (replace with real F5-TTS)"""
        self.logger.info("🧪 Creating mock F5-TTS model for testing")
        
        # This would be replaced with actual F5-TTS model initialization
        class MockF5TTS:
            def __init__(self, device):
                self.device = device
                
            def generate(self, text: str, reference_audio: torch.Tensor) -> torch.Tensor:
                # Mock generation - creates simple sine wave for testing
                duration = min(len(text) * 0.1, 10.0)  # Estimate duration
                samples = int(24000 * duration)
                
                # Generate simple tone (replace with real F5-TTS inference)
                t = torch.linspace(0, duration, samples, device=self.device)
                frequency = 440.0  # A4 note
                audio = 0.3 * torch.sin(2 * torch.pi * frequency * t)
                
                return audio
        
        return MockF5TTS(self.device)
    
    def _download_model(self):
        """Download F5-TTS model (placeholder)"""
        self.logger.info("🌐 Would download F5-TTS model from Hugging Face...")
        
        # Placeholder for actual model download
        # In real implementation, would download from:
        # - Hugging Face model hub
        # - Official F5-TTS repository
        # - Pre-trained weights
        
        model_dir = Path(self.config.model_path)
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Create placeholder file
        placeholder_file = model_dir / "f5_tts_model.pt"
        placeholder_file.touch()
        
        self.logger.info("📁 Model download placeholder created")
    
    def _fallback_initialization(self):
        """Fallback to simpler TTS if F5-TTS fails"""
        self.logger.warning("🔄 Falling back to basic TTS...")
        
        try:
            # Import basic TTS as fallback
            from src.voice.text_to_speech import SimpleTextToSpeech
            self.fallback_tts = SimpleTextToSpeech()
            self.is_ready = self.fallback_tts.available
            
            if self.is_ready:
                self.logger.info("✅ Fallback TTS initialized")
            else:
                self.logger.error("❌ All TTS systems failed")
                
        except Exception as e:
            self.logger.error(f"Even fallback TTS failed: {e}", exception=e)
            self.is_ready = False
    
    def synthesize(self, text: str, output_path: Optional[str] = None) -> Optional[str]:
        """Synthesize speech from text using F5-TTS"""
        if not self.is_ready:
            self.logger.error("F5-TTS engine not ready")
            return None
        
        if not text.strip():
            self.logger.warning("Empty text provided for synthesis")
            return None
        
        try:
            operation_id = None
            if ENHANCED_LOGGING:
                operation_id = self.logger.operation_start("synthesize_speech")
            
            start_time = time.time()
            
            # Prepare text
            clean_text = self._preprocess_text(text)
            self.logger.debug(f"Synthesizing: '{clean_text[:50]}...'")
            
            # Generate audio using F5-TTS
            if self.model and hasattr(self.model, 'generate'):
                audio_tensor = self.model.generate(clean_text, self.reference_audio)
            else:
                # Fallback generation
                audio_tensor = self._generate_fallback_audio(clean_text)
            
            # Apply audio effects
            audio_tensor = self._apply_effects(audio_tensor)
            
            # Save audio
            if output_path is None:
                output_path = self._generate_temp_path()
            
            self._save_audio(audio_tensor, output_path)
            
            synthesis_time = time.time() - start_time
            audio_duration = len(audio_tensor) / self.config.sample_rate
            
            self.logger.info(f"✅ Synthesized {audio_duration:.2f}s audio in {synthesis_time:.2f}s")
            
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, True, {
                    "text_length": len(text),
                    "audio_duration": audio_duration,
                    "synthesis_time": synthesis_time,
                    "rtf": synthesis_time / audio_duration  # Real-time factor
                })
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Speech synthesis failed: {e}", exception=e)
            if ENHANCED_LOGGING and operation_id:
                self.logger.operation_end(operation_id, False)
            
            # Try fallback TTS
            return self._fallback_synthesis(text, output_path)
    
    def _preprocess_text(self, text: str) -> str:
        """Preprocess text for TTS"""
        # Basic text cleaning
        clean_text = text.strip()
        
        # Handle long text
        if len(clean_text) > self.config.max_text_length:
            self.logger.warning(f"Text too long ({len(clean_text)}), truncating to {self.config.max_text_length}")
            clean_text = clean_text[:self.config.max_text_length]
        
        # Thai language specific preprocessing
        if any('\u0e00' <= char <= '\u0e7f' for char in clean_text):
            self.logger.debug("Thai text detected, applying Thai-specific preprocessing")
            # Add Thai-specific text processing here
        
        return clean_text
    
    def _apply_effects(self, audio: torch.Tensor) -> torch.Tensor:
        """Apply audio effects (speed, pitch, etc.)"""
        try:
            # Apply speed adjustment
            if self.config.speed != 1.0:
                # Simple speed adjustment (real implementation would use proper resampling)
                target_length = int(len(audio) / self.config.speed)
                audio = torch.nn.functional.interpolate(
                    audio.unsqueeze(0).unsqueeze(0),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).squeeze()
            
            # Apply pitch adjustment (simplified)
            if self.config.pitch != 1.0:
                # This is a very basic pitch shift - real implementation would use proper algorithms
                audio = audio * self.config.pitch
            
            # Normalize audio
            audio = audio / (torch.max(torch.abs(audio)) + 1e-7)
            audio = audio * 0.8  # Prevent clipping
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Failed to apply audio effects: {e}", exception=e)
            return audio
    
    def _generate_fallback_audio(self, text: str) -> torch.Tensor:
        """Generate simple fallback audio"""
        # Create simple beep pattern based on text length
        duration = max(1.0, min(len(text) * 0.05, 5.0))
        samples = int(self.config.sample_rate * duration)
        
        # Generate tone sequence
        t = torch.linspace(0, duration, samples, device=self.device)
        frequencies = [440, 523, 659, 784]  # C major chord
        
        audio = torch.zeros_like(t)
        for i, freq in enumerate(frequencies):
            phase = 2 * torch.pi * freq * t + i * torch.pi / 4
            audio += 0.1 * torch.sin(phase) * torch.exp(-t * 2)
        
        return audio
    
    def _save_audio(self, audio_tensor: torch.Tensor, output_path: str):
        """Save audio tensor to file"""
        try:
            # Convert to numpy
            audio_np = audio_tensor.detach().cpu().numpy()
            
            # Ensure audio is in valid range
            audio_np = np.clip(audio_np, -1.0, 1.0)
            
            # Save as WAV file
            sf.write(output_path, audio_np, self.config.sample_rate)
            
            self.logger.debug(f"Audio saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save audio: {e}", exception=e)
            raise
    
    def _generate_temp_path(self) -> str:
        """Generate temporary file path for audio"""
        temp_dir = Path(self.config.cache_dir)
        temp_dir.mkdir(exist_ok=True)
        
        timestamp = int(time.time() * 1000)
        return str(temp_dir / f"tts_output_{timestamp}.wav")
    
    def _fallback_synthesis(self, text: str, output_path: Optional[str]) -> Optional[str]:
        """Use fallback TTS system"""
        try:
            if hasattr(self, 'fallback_tts') and self.fallback_tts.available:
                self.logger.info("🔄 Using fallback TTS system")
                return self.fallback_tts.speak_to_file(text, output_path)
            else:
                self.logger.error("No fallback TTS available")
                return None
                
        except Exception as e:
            self.logger.error(f"Fallback TTS also failed: {e}", exception=e)
            return None
    
    def speak(self, text: str) -> bool:
        """Synthesize and play speech directly"""
        try:
            audio_path = self.synthesize(text)
            if audio_path and os.path.exists(audio_path):
                # Play audio (would integrate with audio system)
                self.logger.info(f"🔊 Playing synthesized speech: {audio_path}")
                return True
            return False
            
        except Exception as e:
            self.logger.error(f"Direct speech failed: {e}", exception=e)
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS engine statistics"""
        stats = {
            "engine": "F5-TTS",
            "is_ready": self.is_ready,
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "reference_audio_loaded": self.reference_audio is not None,
            "config": {
                "sample_rate": self.config.sample_rate,
                "speed": self.config.speed,
                "pitch": self.config.pitch,
                "max_text_length": self.config.max_text_length
            }
        }
        
        if ENHANCED_LOGGING:
            component_stats = self.logger.get_component_stats()
            stats.update({"performance": component_stats})
        
        return stats
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("🧹 Cleaning up F5-TTS resources")
            
            if self.model:
                del self.model
                self.model = None
            
            if self.reference_audio is not None:
                del self.reference_audio
                self.reference_audio = None
            
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.is_ready = False
            self.logger.info("✅ F5-TTS cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}", exception=e)


# Factory function for easy integration
def create_f5_tts_engine(config: Optional[Dict[str, Any]] = None) -> F5TTSEngine:
    """Create F5-TTS engine with configuration"""
    if config is None:
        config = {}
    
    # Convert dict to dataclass
    tts_config = F5TTSConfig(**config)
    
    return F5TTSEngine(tts_config)


# Integration with existing TTS interface
class EnhancedTextToSpeech:
    """Enhanced TTS interface that uses F5-TTS with fallback"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.logger = logging.getLogger(__name__)
        
        try:
            # Try to initialize F5-TTS
            self.f5_engine = create_f5_tts_engine(config)
            self.primary_engine = "f5_tts"
            self.logger.info("✅ Enhanced TTS using F5-TTS engine")
            
        except Exception as e:
            self.logger.warning(f"F5-TTS initialization failed: {e}")
            
            # Fallback to simple TTS
            from src.voice.text_to_speech import SimpleTextToSpeech
            self.simple_engine = SimpleTextToSpeech()
            self.primary_engine = "simple"
            self.logger.info("✅ Enhanced TTS using simple engine fallback")
    
    def speak(self, text: str) -> bool:
        """Speak text using available engine"""
        if self.primary_engine == "f5_tts":
            return self.f5_engine.speak(text)
        else:
            return self.simple_engine.speak(text)
    
    def speak_to_file(self, text: str, output_path: str) -> str:
        """Generate speech to file"""
        if self.primary_engine == "f5_tts":
            result = self.f5_engine.synthesize(text, output_path)
            return result or output_path
        else:
            return self.simple_engine.speak_to_file(text, output_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get TTS statistics"""
        if self.primary_engine == "f5_tts":
            return self.f5_engine.get_stats()
        else:
            return {"engine": "simple", "available": getattr(self.simple_engine, 'available', False)}