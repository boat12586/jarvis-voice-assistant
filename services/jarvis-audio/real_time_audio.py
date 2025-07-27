"""
Real-time Audio Processing Service for Jarvis v2.0
Handles real-time streaming TTS/STT with advanced audio processing
"""

import asyncio
import logging
import numpy as np
import sounddevice as sd
import soundfile as sf
import queue
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import io
import wave
import json
import websockets
import base64
from datetime import datetime
import torch
import whisper
from transformers import pipeline
import edge_tts
import tempfile
import os

logger = logging.getLogger(__name__)

class AudioFormat(Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"

class StreamingState(Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

@dataclass
class AudioConfig:
    """Audio processing configuration with performance optimizations"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 512  # Reduced for lower latency
    buffer_size: int = 2048  # Reduced buffer size
    format: AudioFormat = AudioFormat.WAV
    
    # STT Configuration - Optimized for performance
    stt_model: str = "tiny"  # Faster model for real-time
    stt_language: str = "auto"
    stt_threshold: float = 0.3  # Lower threshold for better responsiveness
    stt_chunk_duration: float = 0.5  # Reduced for faster processing
    
    # TTS Configuration
    tts_voice: str = "en-US-AriaNeural"
    tts_rate: str = "+10%"  # Slightly faster speech
    tts_volume: str = "+0%"
    tts_pitch: str = "+0Hz"
    
    # Real-time Processing - Performance optimized
    enable_vad: bool = True  # Voice Activity Detection
    enable_noise_reduction: bool = True
    enable_echo_cancellation: bool = True
    enable_auto_gain: bool = True
    
    # Streaming Configuration - Low latency settings
    streaming_chunk_ms: int = 50  # Reduced for lower latency
    max_silence_ms: int = 800  # Shorter silence detection
    min_speech_ms: int = 200  # Reduced minimum speech duration
    
    # Performance Optimization Settings
    processing_threads: int = 2  # Multi-threading for performance
    enable_gpu_acceleration: bool = True  # Use GPU if available
    max_processing_queue_size: int = 10  # Limit queue size to prevent lag
    enable_adaptive_buffering: bool = True  # Dynamic buffer sizing

@dataclass
class AudioChunk:
    """Audio data chunk for streaming"""
    data: np.ndarray
    timestamp: float
    sample_rate: int
    is_speech: bool = False
    confidence: float = 0.0
    sequence_id: int = 0

@dataclass
class STTResult:
    """Speech-to-text result"""
    text: str
    confidence: float
    language: str
    timestamp: float
    is_final: bool = False
    processing_time: float = 0.0

@dataclass
class TTSResult:
    """Text-to-speech result"""
    audio_data: np.ndarray
    sample_rate: int
    duration: float
    text: str
    voice: str
    timestamp: float

class VoiceActivityDetector:
    """Real-time voice activity detection with performance optimizations"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.threshold = config.stt_threshold
        self.min_speech_frames = int(config.min_speech_ms * config.sample_rate / 1000)
        self.max_silence_frames = int(config.max_silence_ms * config.sample_rate / 1000)
        
        # VAD state
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.speech_buffer = []
        
        # Energy calculation - optimized
        self.energy_history = []
        self.energy_history_size = 30  # Reduced for better performance
        
        # Performance optimizations
        self.spectral_features = np.zeros(4)  # Cache for spectral features
        self.adaptive_threshold = config.stt_threshold
        self.noise_level = 0.0
        self.speech_confidence = 0.0
        
        # Pre-computed constants for performance
        self.frame_energy_weight = 0.7
        self.spectral_energy_weight = 0.3
        
    def process_chunk(self, audio_chunk: np.ndarray) -> bool:
        """Process audio chunk and detect voice activity with optimizations"""
        # Fast RMS energy calculation using vectorized operations
        rms = np.sqrt(np.mean(audio_chunk ** 2))
        
        # Calculate additional features for better detection
        if len(audio_chunk) > 256:  # Only for larger chunks to avoid overhead
            # Zero crossing rate (indicates speech vs noise)
            zero_crossings = np.sum(np.diff(np.signbit(audio_chunk)))
            zcr = zero_crossings / len(audio_chunk)
            
            # Spectral centroid approximation (frequency characteristics)
            fft_magnitude = np.abs(np.fft.rfft(audio_chunk))
            spectral_centroid = np.sum(fft_magnitude * np.arange(len(fft_magnitude))) / np.sum(fft_magnitude)
            
            # Combine features for better accuracy
            speech_score = (rms * self.frame_energy_weight + 
                           zcr * 0.2 + 
                           min(spectral_centroid / len(fft_magnitude), 1.0) * 0.1)
        else:
            speech_score = rms
            zcr = 0.0
        
        # Update energy history with exponential moving average for efficiency
        if len(self.energy_history) >= self.energy_history_size:
            self.energy_history[:-1] = self.energy_history[1:]
            self.energy_history[-1] = rms
        else:
            self.energy_history.append(rms)
        
        # Adaptive threshold with noise estimation
        if len(self.energy_history) > 5:
            # Use percentile for noise floor estimation (more robust)
            noise_floor = np.percentile(self.energy_history, 20)
            self.noise_level = 0.9 * self.noise_level + 0.1 * noise_floor
            self.adaptive_threshold = max(self.threshold, self.noise_level * 2.5)
        else:
            self.adaptive_threshold = self.threshold
        
        # Enhanced voice activity detection with confidence scoring
        if speech_score > self.adaptive_threshold:
            self.speech_frames += len(audio_chunk)
            self.silence_frames = 0
            
            # Calculate confidence based on signal strength
            self.speech_confidence = min(1.0, speech_score / (self.adaptive_threshold * 2))
            
            if self.speech_frames >= self.min_speech_frames:
                self.is_speaking = True
        else:
            self.silence_frames += len(audio_chunk)
            self.speech_confidence *= 0.9  # Decay confidence during silence
            
            if self.is_speaking and self.silence_frames >= self.max_silence_frames:
                self.is_speaking = False
                self.speech_frames = 0
        
        return self.is_speaking
    
    def get_speech_confidence(self) -> float:
        """Get current speech confidence score"""
        return self.speech_confidence
    
    def get_noise_level(self) -> float:
        """Get current noise level estimation"""
        return self.noise_level
    
    def reset(self):
        """Reset VAD state"""
        self.speech_frames = 0
        self.silence_frames = 0
        self.is_speaking = False
        self.speech_buffer.clear()

class RealTimeSTT:
    """Real-time Speech-to-Text processor with multi-threading optimizations"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.model = None
        self.processing_queue = queue.Queue(maxsize=config.max_processing_queue_size)
        self.result_callbacks = []
        self.is_running = False
        self.vad = VoiceActivityDetector(config)
        
        # Audio buffer for accumulation with circular buffer for efficiency
        self.audio_buffer = []
        self.buffer_lock = threading.Lock()
        
        # Multi-threading for performance
        self.processing_threads = []
        self.thread_pool_size = config.processing_threads
        
        # Performance metrics
        self.processing_times = []
        self.average_processing_time = 0.0
        
        # GPU acceleration flag
        self.use_gpu = config.enable_gpu_acceleration and torch.cuda.is_available()
        
        # Pre-allocated arrays for performance
        self.temp_audio_buffer = np.zeros(config.sample_rate * 2)  # 2 seconds buffer
        
    async def initialize(self):
        """Initialize STT model with GPU optimization"""
        try:
            device = "cuda" if self.use_gpu else "cpu"
            logger.info(f"Loading Whisper model: {self.config.stt_model} on {device}")
            
            # Load model with device specification
            self.model = whisper.load_model(self.config.stt_model, device=device)
            
            if self.use_gpu and torch.cuda.is_available():
                # Warm up GPU model
                dummy_audio = np.random.randn(16000).astype(np.float32)
                _ = self.model.transcribe(dummy_audio, task="transcribe")
                logger.info("GPU model warmed up successfully")
            
            logger.info(f"STT model loaded successfully on {device}")
            return True
        except Exception as e:
            logger.error(f"Failed to load STT model: {e}")
            return False
    
    def add_result_callback(self, callback: Callable[[STTResult], None]):
        """Add callback for STT results"""
        self.result_callbacks.append(callback)
    
    def start_streaming(self):
        """Start real-time STT processing with multi-threading"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start multiple processing threads for better performance
        for i in range(self.thread_pool_size):
            thread = threading.Thread(
                target=self._processing_loop, 
                name=f"STT-Worker-{i}",
                daemon=True
            )
            thread.start()
            self.processing_threads.append(thread)
        
        logger.info(f"Real-time STT started with {self.thread_pool_size} threads")
    
    def stop_streaming(self):
        """Stop real-time STT processing"""
        self.is_running = False
        
        # Join all processing threads
        for thread in self.processing_threads:
            thread.join(timeout=1.0)  # Timeout to prevent hanging
        
        self.processing_threads.clear()
        logger.info("Real-time STT stopped")
    
    def process_audio_chunk(self, chunk: AudioChunk):
        """Process incoming audio chunk with adaptive buffering"""
        # Voice activity detection with confidence
        is_speech = self.vad.process_chunk(chunk.data)
        chunk.is_speech = is_speech
        chunk.confidence = self.vad.get_speech_confidence()
        
        # Adaptive processing based on queue size and processing load
        if is_speech and chunk.confidence > 0.3:  # Only process confident speech
            with self.buffer_lock:
                self.audio_buffer.append(chunk)
                
                # Adaptive buffer size based on processing load
                required_chunks = self._calculate_optimal_buffer_size()
                
                if len(self.audio_buffer) >= required_chunks:
                    self._trigger_processing()
        
        # Drop old chunks if buffer gets too large (prevent memory buildup)
        elif len(self.audio_buffer) > self.config.max_processing_queue_size * 2:
            with self.buffer_lock:
                self.audio_buffer = self.audio_buffer[-self.config.max_processing_queue_size:]
    
    def _calculate_optimal_buffer_size(self) -> int:
        """Calculate optimal buffer size based on current processing load"""
        base_chunks = int(self.config.stt_chunk_duration * self.config.sample_rate / self.config.chunk_size)
        
        # Adjust based on average processing time
        if self.average_processing_time > 0.5:  # If processing is slow
            return max(base_chunks // 2, 2)  # Use smaller chunks for faster response
        elif self.average_processing_time < 0.1:  # If processing is fast
            return min(base_chunks * 2, 10)  # Use larger chunks for better accuracy
        else:
            return base_chunks
    
    def _trigger_processing(self):
        """Trigger STT processing of buffered audio with optimization"""
        if not self.audio_buffer:
            return
        
        try:
            # Combine audio chunks efficiently
            total_length = sum(len(chunk.data) for chunk in self.audio_buffer)
            combined_audio = np.empty(total_length, dtype=np.float32)
            
            offset = 0
            for chunk in self.audio_buffer:
                chunk_len = len(chunk.data)
                combined_audio[offset:offset + chunk_len] = chunk.data
                offset += chunk_len
            
            timestamp = self.audio_buffer[0].timestamp
            confidence = np.mean([chunk.confidence for chunk in self.audio_buffer])
            
            # Clear buffer
            self.audio_buffer.clear()
            
            # Add to processing queue with non-blocking put
            try:
                self.processing_queue.put_nowait((combined_audio, timestamp, confidence))
            except queue.Full:
                # If queue is full, drop the oldest item and add new one
                try:
                    self.processing_queue.get_nowait()
                    self.processing_queue.put_nowait((combined_audio, timestamp, confidence))
                except queue.Empty:
                    pass
                
        except Exception as e:
            logger.error(f"Error in trigger processing: {e}")
            self.audio_buffer.clear()
    
    def _processing_loop(self):
        """Main processing loop for STT with performance optimizations"""
        while self.is_running:
            try:
                # Get audio from queue with timeout
                audio_data, timestamp, input_confidence = self.processing_queue.get(timeout=1.0)
                
                # Skip processing if audio is too short or low confidence
                if len(audio_data) < self.config.sample_rate * 0.1 or input_confidence < 0.2:
                    continue
                
                # Process with Whisper
                start_time = time.time()
                
                # Optimize Whisper parameters for real-time processing
                result = self.model.transcribe(
                    audio_data,
                    language=None if self.config.stt_language == "auto" else self.config.stt_language,
                    task="transcribe",
                    verbose=False,  # Reduce logging overhead
                    fp16=self.use_gpu,  # Use FP16 for faster GPU processing
                    temperature=0,  # Deterministic for consistency
                    best_of=1,  # Single pass for speed
                    beam_size=1,  # Fastest beam search
                    patience=1.0,  # Reduce patience for speed
                    condition_on_previous_text=False,  # Faster processing
                    suppress_tokens=[-1]  # Suppress silence tokens
                )
                
                processing_time = time.time() - start_time
                
                # Update performance metrics
                self.processing_times.append(processing_time)
                if len(self.processing_times) > 10:
                    self.processing_times.pop(0)
                self.average_processing_time = np.mean(self.processing_times)
                
                # Extract text and confidence
                text = result.get("text", "").strip()
                segments = result.get("segments", [])
                
                if text and segments and len(text) > 1:  # Filter out single characters
                    # Calculate average confidence from segments
                    if segments:
                        avg_logprob = np.mean([seg.get("avg_logprob", -2.0) for seg in segments])
                        confidence = max(0.0, min(1.0, np.exp(avg_logprob)))
                    else:
                        confidence = input_confidence
                    
                    # Combine with input confidence for final score
                    final_confidence = (confidence + input_confidence) / 2
                    
                    # Only process high-confidence results for real-time
                    if final_confidence > 0.4:
                        # Create result
                        stt_result = STTResult(
                            text=text,
                            confidence=final_confidence,
                            language=result.get("language", "unknown"),
                            timestamp=timestamp,
                            is_final=True,
                            processing_time=processing_time
                        )
                        
                        # Call callbacks asynchronously for better performance
                        for callback in self.result_callbacks:
                            try:
                                callback(stt_result)
                            except Exception as e:
                                logger.error(f"STT callback error: {e}")
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"STT processing error: {e}")
                
    def get_performance_metrics(self) -> Dict[str, float]:
        """Get current performance metrics"""
        return {
            "average_processing_time": self.average_processing_time,
            "queue_size": self.processing_queue.qsize(),
            "buffer_size": len(self.audio_buffer),
            "gpu_enabled": self.use_gpu
        }

class RealTimeTTS:
    """Real-time Text-to-Speech processor"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.synthesis_queue = queue.Queue()
        self.result_callbacks = []
        self.is_running = False
        
        # Processing thread
        self.processing_thread = None
        
    def add_result_callback(self, callback: Callable[[TTSResult], None]):
        """Add callback for TTS results"""
        self.result_callbacks.append(callback)
    
    def start_streaming(self):
        """Start real-time TTS processing"""
        if self.is_running:
            return
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.start()
        logger.info("Real-time TTS started")
    
    def stop_streaming(self):
        """Stop real-time TTS processing"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join()
        logger.info("Real-time TTS stopped")
    
    async def synthesize_text(self, text: str, voice: str = None, user_id: str = None) -> TTSResult:
        """Synthesize text to speech"""
        timestamp = time.time()
        voice = voice or self.config.tts_voice
        
        try:
            # Create TTS communication
            communicate = edge_tts.Communicate(text, voice)
            
            # Generate audio
            audio_data = b""
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_data += chunk["data"]
            
            # Convert to numpy array
            if audio_data:
                # Save to temporary file to get audio data
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_file.write(audio_data)
                    temp_file.flush()
                    
                    # Load audio data
                    audio_array, sample_rate = sf.read(temp_file.name)
                    
                    # Cleanup
                    os.unlink(temp_file.name)
                
                # Ensure correct sample rate
                if sample_rate != self.config.sample_rate:
                    # Simple resampling (can be improved)
                    audio_array = self._resample_audio(audio_array, sample_rate, self.config.sample_rate)
                    sample_rate = self.config.sample_rate
                
                # Calculate duration
                duration = len(audio_array) / sample_rate
                
                result = TTSResult(
                    audio_data=audio_array,
                    sample_rate=sample_rate,
                    duration=duration,
                    text=text,
                    voice=voice,
                    timestamp=timestamp
                )
                
                # Call callbacks
                for callback in self.result_callbacks:
                    try:
                        callback(result)
                    except Exception as e:
                        logger.error(f"TTS callback error: {e}")
                
                return result
            
        except Exception as e:
            logger.error(f"TTS synthesis error: {e}")
            return None
    
    def _resample_audio(self, audio: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
        """Simple audio resampling"""
        if from_rate == to_rate:
            return audio
        
        # Simple linear interpolation resampling
        duration = len(audio) / from_rate
        new_length = int(duration * to_rate)
        
        old_indices = np.linspace(0, len(audio) - 1, new_length)
        new_audio = np.interp(old_indices, np.arange(len(audio)), audio)
        
        return new_audio.astype(audio.dtype)
    
    def _processing_loop(self):
        """Main processing loop for TTS"""
        while self.is_running:
            try:
                # Get synthesis request from queue
                text, voice, user_id = self.synthesis_queue.get(timeout=1.0)
                
                # Process TTS (run async in thread)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                result = loop.run_until_complete(self.synthesize_text(text, voice, user_id))
                loop.close()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"TTS processing error: {e}")

class AudioStreamManager:
    """Manages real-time audio streaming"""
    
    def __init__(self, config: AudioConfig):
        self.config = config
        self.stt = RealTimeSTT(config)
        self.tts = RealTimeTTS(config)
        
        # Audio I/O
        self.input_stream = None
        self.output_stream = None
        self.is_streaming = False
        
        # Audio buffers
        self.input_buffer = queue.Queue()
        self.output_buffer = queue.Queue()
        
        # State management
        self.state = StreamingState.IDLE
        self.current_user_id = None
        self.session_id = None
        
        # Callbacks
        self.stt_callbacks = []
        self.tts_callbacks = []
        self.state_callbacks = []
        
        # Metrics
        self.metrics = {
            "chunks_processed": 0,
            "stt_requests": 0,
            "tts_requests": 0,
            "errors": 0,
            "start_time": time.time()
        }
    
    async def initialize(self):
        """Initialize audio stream manager"""
        try:
            # Initialize STT
            await self.stt.initialize()
            
            # Set up callbacks
            self.stt.add_result_callback(self._on_stt_result)
            self.tts.add_result_callback(self._on_tts_result)
            
            logger.info("Audio stream manager initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize audio stream manager: {e}")
            return False
    
    def add_stt_callback(self, callback: Callable[[STTResult], None]):
        """Add STT result callback"""
        self.stt_callbacks.append(callback)
    
    def add_tts_callback(self, callback: Callable[[TTSResult], None]):
        """Add TTS result callback"""
        self.tts_callbacks.append(callback)
    
    def add_state_callback(self, callback: Callable[[StreamingState], None]):
        """Add state change callback"""
        self.state_callbacks.append(callback)
    
    def start_streaming(self, user_id: str, session_id: str):
        """Start audio streaming for user"""
        if self.is_streaming:
            return
        
        try:
            self.current_user_id = user_id
            self.session_id = session_id
            self.is_streaming = True
            
            # Start STT and TTS
            self.stt.start_streaming()
            self.tts.start_streaming()
            
            # Start audio I/O
            self._start_audio_io()
            
            self._set_state(StreamingState.LISTENING)
            logger.info(f"Audio streaming started for user {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to start audio streaming: {e}")
            self._set_state(StreamingState.ERROR)
    
    def stop_streaming(self):
        """Stop audio streaming"""
        if not self.is_streaming:
            return
        
        try:
            self.is_streaming = False
            
            # Stop audio I/O
            self._stop_audio_io()
            
            # Stop STT and TTS
            self.stt.stop_streaming()
            self.tts.stop_streaming()
            
            self._set_state(StreamingState.IDLE)
            logger.info("Audio streaming stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop audio streaming: {e}")
    
    def _start_audio_io(self):
        """Start audio input/output streams"""
        try:
            # Input stream for STT
            self.input_stream = sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                callback=self._audio_input_callback,
                dtype=np.float32
            )
            
            # Output stream for TTS
            self.output_stream = sd.OutputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                blocksize=self.config.chunk_size,
                callback=self._audio_output_callback,
                dtype=np.float32
            )
            
            self.input_stream.start()
            self.output_stream.start()
            
            logger.info("Audio I/O streams started")
            
        except Exception as e:
            logger.error(f"Failed to start audio I/O: {e}")
            raise
    
    def _stop_audio_io(self):
        """Stop audio input/output streams"""
        try:
            if self.input_stream:
                self.input_stream.stop()
                self.input_stream.close()
                self.input_stream = None
            
            if self.output_stream:
                self.output_stream.stop()
                self.output_stream.close()
                self.output_stream = None
            
            logger.info("Audio I/O streams stopped")
            
        except Exception as e:
            logger.error(f"Failed to stop audio I/O: {e}")
    
    def _audio_input_callback(self, indata, frames, time, status):
        """Audio input callback for STT"""
        if status:
            logger.warning(f"Audio input status: {status}")
        
        if self.is_streaming and self.state == StreamingState.LISTENING:
            try:
                # Create audio chunk
                chunk = AudioChunk(
                    data=indata.copy().flatten(),
                    timestamp=time.inputBufferAdcTime,
                    sample_rate=self.config.sample_rate,
                    sequence_id=self.metrics["chunks_processed"]
                )
                
                # Process through STT
                self.stt.process_audio_chunk(chunk)
                
                self.metrics["chunks_processed"] += 1
                
            except Exception as e:
                logger.error(f"Audio input processing error: {e}")
                self.metrics["errors"] += 1
    
    def _audio_output_callback(self, outdata, frames, time, status):
        """Audio output callback for TTS"""
        if status:
            logger.warning(f"Audio output status: {status}")
        
        try:
            if not self.output_buffer.empty():
                # Get audio data from buffer
                audio_data = self.output_buffer.get_nowait()
                
                # Ensure correct length
                if len(audio_data) >= frames:
                    outdata[:] = audio_data[:frames].reshape(-1, self.config.channels)
                else:
                    # Pad with zeros if too short
                    padded = np.zeros((frames, self.config.channels), dtype=np.float32)
                    padded[:len(audio_data)] = audio_data.reshape(-1, self.config.channels)
                    outdata[:] = padded
            else:
                # No audio data, output silence
                outdata.fill(0)
                
        except queue.Empty:
            outdata.fill(0)
        except Exception as e:
            logger.error(f"Audio output processing error: {e}")
            outdata.fill(0)
    
    def _on_stt_result(self, result: STTResult):
        """Handle STT result"""
        self.metrics["stt_requests"] += 1
        
        # Call callbacks
        for callback in self.stt_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"STT callback error: {e}")
    
    def _on_tts_result(self, result: TTSResult):
        """Handle TTS result"""
        self.metrics["tts_requests"] += 1
        
        # Add audio to output buffer
        try:
            # Split audio into chunks for streaming
            chunk_size = self.config.chunk_size
            audio_data = result.audio_data
            
            for i in range(0, len(audio_data), chunk_size):
                chunk = audio_data[i:i + chunk_size]
                self.output_buffer.put(chunk)
            
            self._set_state(StreamingState.SPEAKING)
            
        except Exception as e:
            logger.error(f"TTS output error: {e}")
        
        # Call callbacks
        for callback in self.tts_callbacks:
            try:
                callback(result)
            except Exception as e:
                logger.error(f"TTS callback error: {e}")
    
    def _set_state(self, new_state: StreamingState):
        """Set streaming state"""
        if self.state != new_state:
            self.state = new_state
            
            # Call state callbacks
            for callback in self.state_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.error(f"State callback error: {e}")
    
    async def synthesize_speech(self, text: str, voice: str = None):
        """Synthesize speech for current user"""
        if not self.is_streaming:
            return
        
        self._set_state(StreamingState.PROCESSING)
        
        try:
            result = await self.tts.synthesize_text(text, voice, self.current_user_id)
            return result
        except Exception as e:
            logger.error(f"Speech synthesis error: {e}")
            self._set_state(StreamingState.ERROR)
            return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get streaming metrics"""
        current_time = time.time()
        uptime = current_time - self.metrics["start_time"]
        
        return {
            **self.metrics,
            "uptime": uptime,
            "chunks_per_second": self.metrics["chunks_processed"] / uptime if uptime > 0 else 0,
            "current_state": self.state.value,
            "current_user": self.current_user_id,
            "session_id": self.session_id,
            "is_streaming": self.is_streaming
        }