"""
Voice Pipeline Optimization
Faster processing and better quality
"""

import numpy as np
import torch
import logging
from typing import Optional, Tuple, List
import time

class VoiceOptimizer:
    """Optimizes voice processing pipeline"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.vad_model = None
        self.audio_buffer = []
        
        # Performance optimization settings
        self.enable_caching = True
        self.enable_batching = True
        self.max_batch_size = 5
        
        # Audio optimization settings
        self.noise_reduction_enabled = True
        self.auto_gain_control = True
        self.dynamic_range_compression = True
        
        # Processing cache
        self._audio_cache = {}
        self._cache_size_limit = 100
        
    def initialize_vad(self):
        """Initialize Voice Activity Detection"""
        try:
            # Use simple energy-based VAD for now
            self.logger.info("Initialized simple VAD")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize VAD: {e}")
            return False
    
    def detect_voice_activity(self, audio: np.ndarray, 
                            threshold: float = 0.01) -> bool:
        """Simple voice activity detection"""
        # Calculate RMS energy
        rms = np.sqrt(np.mean(audio ** 2))
        return rms > threshold
    
    def chunk_audio(self, audio: np.ndarray, 
                   chunk_size: int = 1024,
                   overlap: int = 256) -> List[np.ndarray]:
        """Split audio into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(audio):
            end = min(start + chunk_size, len(audio))
            chunks.append(audio[start:end])
            start += chunk_size - overlap
        
        return chunks
    
    def preprocess_audio(self, audio: np.ndarray, 
                        sample_rate: int = 16000,
                        enable_optimizations: bool = True) -> np.ndarray:
        """Enhanced audio preprocessing for better recognition"""
        try:
            # Input validation
            if audio is None or len(audio) == 0:
                return np.array([])
            
            # Create cache key for this audio
            audio_hash = hash(audio.tobytes()) if self.enable_caching else None
            
            # Check cache first
            if self.enable_caching and audio_hash in self._audio_cache:
                self.logger.debug("Using cached audio preprocessing result")
                return self._audio_cache[audio_hash]
            
            start_time = time.time()
            
            # 1. Normalize audio with safety checks
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val
            else:
                # Silent audio, return as-is
                return audio
            
            if enable_optimizations:
                # 2. Auto Gain Control (AGC)
                if self.auto_gain_control:
                    audio = self._apply_auto_gain_control(audio)
                
                # 3. Noise reduction
                if self.noise_reduction_enabled:
                    audio = self._apply_noise_reduction(audio, sample_rate)
                
                # 4. Dynamic range compression
                if self.dynamic_range_compression:
                    audio = self._apply_dynamic_range_compression(audio)
            
            # 5. Final normalization
            max_val = np.max(np.abs(audio))
            if max_val > 0:
                audio = audio / max_val * 0.95  # Leave some headroom
            
            processing_time = time.time() - start_time
            self.logger.debug(f"Audio preprocessing took {processing_time:.3f}s")
            
            # Cache result
            if self.enable_caching and audio_hash:
                self._manage_cache()
                self._audio_cache[audio_hash] = audio.copy()
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            # Return normalized original audio as fallback
            max_val = np.max(np.abs(audio))
            return audio / max_val if max_val > 0 else audio
    
    def _apply_auto_gain_control(self, audio: np.ndarray, 
                               target_rms: float = 0.1) -> np.ndarray:
        """Apply automatic gain control"""
        try:
            # Calculate RMS in overlapping windows
            window_size = 1024
            overlap = 512
            
            if len(audio) < window_size:
                # Too short for windowed AGC, use simple method
                current_rms = np.sqrt(np.mean(audio ** 2))
                if current_rms > 0:
                    gain = target_rms / current_rms
                    # Limit gain to prevent excessive amplification
                    gain = min(gain, 10.0)
                    return audio * gain
                return audio
            
            # Windowed AGC
            output = np.zeros_like(audio)
            for i in range(0, len(audio) - window_size, overlap):
                window = audio[i:i + window_size]
                window_rms = np.sqrt(np.mean(window ** 2))
                
                if window_rms > 0:
                    gain = target_rms / window_rms
                    gain = min(gain, 5.0)  # Limit gain
                    output[i:i + window_size] += window * gain
                else:
                    output[i:i + window_size] += window
            
            return output
            
        except Exception as e:
            self.logger.warning(f"AGC failed: {e}")
            return audio
    
    def _apply_noise_reduction(self, audio: np.ndarray, 
                             sample_rate: int) -> np.ndarray:
        """Enhanced noise reduction"""
        try:
            if len(audio) < sample_rate // 20:  # Less than 50ms
                return audio
            
            # Remove DC offset
            audio = audio - np.mean(audio)
            
            # Enhanced high-pass filter (removes low-frequency noise)
            # Butterworth-like filter approximation
            cutoff_freq = 80  # Hz
            rc = 1.0 / (2 * np.pi * cutoff_freq)
            dt = 1.0 / sample_rate
            alpha = dt / (rc + dt)
            
            filtered = np.zeros_like(audio)
            filtered[0] = audio[0] * alpha
            
            for i in range(1, len(audio)):
                filtered[i] = alpha * audio[i] + (1 - alpha) * filtered[i-1]
            
            # Subtract low-pass component to get high-pass result
            high_passed = audio - filtered
            
            # Spectral subtraction for additional noise reduction
            high_passed = self._spectral_subtraction(high_passed, sample_rate)
            
            return high_passed
            
        except Exception as e:
            self.logger.warning(f"Noise reduction failed: {e}")
            return audio
    
    def _spectral_subtraction(self, audio: np.ndarray, 
                            sample_rate: int,
                            alpha: float = 2.0) -> np.ndarray:
        """Simple spectral subtraction for noise reduction"""
        try:
            # Simple implementation - can be improved with FFT
            # For now, just apply a gentle low-pass filter to remove high-freq noise
            
            # Moving average filter
            window_size = max(3, sample_rate // 8000)  # Adaptive window size
            
            if len(audio) < window_size:
                return audio
            
            # Apply smoothing
            smoothed = np.convolve(audio, np.ones(window_size) / window_size, mode='same')
            
            # Mix original and smoothed
            return 0.7 * audio + 0.3 * smoothed
            
        except Exception as e:
            self.logger.warning(f"Spectral subtraction failed: {e}")
            return audio
    
    def _apply_dynamic_range_compression(self, audio: np.ndarray,
                                       threshold: float = 0.5,
                                       ratio: float = 4.0) -> np.ndarray:
        """Apply dynamic range compression"""
        try:
            # Simple compressor
            compressed = np.zeros_like(audio)
            
            for i, sample in enumerate(audio):
                abs_sample = abs(sample)
                
                if abs_sample > threshold:
                    # Compress above threshold
                    excess = abs_sample - threshold
                    compressed_excess = excess / ratio
                    new_level = threshold + compressed_excess
                    compressed[i] = np.sign(sample) * new_level
                else:
                    compressed[i] = sample
            
            return compressed
            
        except Exception as e:
            self.logger.warning(f"Dynamic range compression failed: {e}")
            return audio
    
    def _manage_cache(self):
        """Manage cache size"""
        if len(self._audio_cache) >= self._cache_size_limit:
            # Remove oldest entries (simple FIFO)
            keys_to_remove = list(self._audio_cache.keys())[:10]
            for key in keys_to_remove:
                del self._audio_cache[key]
    
    def optimize_for_whisper(self, audio: np.ndarray, 
                           sample_rate: int = 16000) -> np.ndarray:
        """Enhanced Whisper optimization with preprocessing"""
        try:
            # Apply enhanced preprocessing first
            audio = self.preprocess_audio(audio, sample_rate)
            
            # Whisper expects 16kHz, mono, 30-second chunks max
            target_length = sample_rate * 30  # 30 seconds max
            
            if len(audio) > target_length:
                # Use voice activity detection to find the best segment
                best_segment = self._find_best_audio_segment(audio, target_length, sample_rate)
                audio = best_segment
            
            # Pad if too short but ensure minimum viable length
            min_length = sample_rate // 2  # 0.5 seconds minimum
            if len(audio) < min_length:
                padding = min_length - len(audio)
                audio = np.pad(audio, (0, padding), mode='constant')
            
            return audio
            
        except Exception as e:
            self.logger.error(f"Whisper optimization failed: {e}")
            return audio
    
    def _find_best_audio_segment(self, audio: np.ndarray, 
                               target_length: int,
                               sample_rate: int) -> np.ndarray:
        """Find the best audio segment using voice activity detection"""
        try:
            # Analyze audio in chunks to find voice activity
            chunk_size = sample_rate  # 1 second chunks
            voice_scores = []
            
            for i in range(0, len(audio) - chunk_size, chunk_size // 2):
                chunk = audio[i:i + chunk_size]
                
                # Calculate voice activity score
                rms = np.sqrt(np.mean(chunk ** 2))
                zero_crossing_rate = np.sum(np.diff(np.sign(chunk)) != 0) / len(chunk)
                
                # Combine metrics for voice activity score
                voice_score = rms * (1 + zero_crossing_rate)
                voice_scores.append((i, voice_score))
            
            if not voice_scores:
                # Fallback: take the end of audio
                return audio[-target_length:]
            
            # Find the segment with highest voice activity
            voice_scores.sort(key=lambda x: x[1], reverse=True)
            best_start = voice_scores[0][0]
            
            # Extract segment around the best start point
            segment_start = max(0, best_start - target_length // 4)
            segment_end = min(len(audio), segment_start + target_length)
            
            if segment_end - segment_start < target_length:
                # Adjust if segment is too short
                segment_start = max(0, segment_end - target_length)
            
            return audio[segment_start:segment_end]
            
        except Exception as e:
            self.logger.warning(f"Best segment selection failed: {e}")
            # Fallback: take the end of audio
            return audio[-target_length:]
    
    def batch_process_audio(self, audio_list: List[np.ndarray],
                          sample_rate: int = 16000) -> List[np.ndarray]:
        """Process multiple audio samples in batch for efficiency"""
        if not self.enable_batching or len(audio_list) <= 1:
            # Process individually
            return [self.preprocess_audio(audio, sample_rate) for audio in audio_list]
        
        try:
            start_time = time.time()
            processed_audio = []
            
            # Process in batches
            batch_size = min(self.max_batch_size, len(audio_list))
            
            for i in range(0, len(audio_list), batch_size):
                batch = audio_list[i:i + batch_size]
                batch_results = []
                
                for audio in batch:
                    try:
                        processed = self.preprocess_audio(audio, sample_rate, enable_optimizations=True)
                        batch_results.append(processed)
                    except Exception as e:
                        self.logger.warning(f"Failed to process audio in batch: {e}")
                        batch_results.append(audio)  # Fallback to original
                
                processed_audio.extend(batch_results)
            
            processing_time = time.time() - start_time
            self.logger.info(f"Batch processed {len(audio_list)} audio samples in {processing_time:.3f}s")
            
            return processed_audio
            
        except Exception as e:
            self.logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            return [self.preprocess_audio(audio, sample_rate) for audio in audio_list]
    
    def get_performance_stats(self) -> dict:
        """Get performance statistics"""
        return {
            "cache_enabled": self.enable_caching,
            "cache_size": len(self._audio_cache),
            "cache_limit": self._cache_size_limit,
            "batching_enabled": self.enable_batching,
            "max_batch_size": self.max_batch_size,
            "noise_reduction": self.noise_reduction_enabled,
            "auto_gain_control": self.auto_gain_control,
            "dynamic_range_compression": self.dynamic_range_compression
        }
    
    def clear_cache(self):
        """Clear processing cache"""
        self._audio_cache.clear()
        self.logger.info("Audio processing cache cleared")
    
    def update_settings(self, settings: dict):
        """Update optimizer settings"""
        if "enable_caching" in settings:
            self.enable_caching = settings["enable_caching"]
        if "enable_batching" in settings:
            self.enable_batching = settings["enable_batching"]
        if "max_batch_size" in settings:
            self.max_batch_size = max(1, settings["max_batch_size"])
        if "noise_reduction_enabled" in settings:
            self.noise_reduction_enabled = settings["noise_reduction_enabled"]
        if "auto_gain_control" in settings:
            self.auto_gain_control = settings["auto_gain_control"]
        if "dynamic_range_compression" in settings:
            self.dynamic_range_compression = settings["dynamic_range_compression"]
        
        self.logger.info("Voice optimizer settings updated")

# Global voice optimizer
voice_optimizer = VoiceOptimizer()
