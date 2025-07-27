"""
Data models for Jarvis Real-time Audio Service
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class AudioFormat(str, Enum):
    WAV = "wav"
    MP3 = "mp3"
    FLAC = "flac"
    OGG = "ogg"

class StreamingState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"
    ERROR = "error"

class AudioRequest(BaseModel):
    """Base audio request model"""
    user_id: str
    session_id: Optional[str] = None
    language: Optional[str] = "en"
    
class AudioResponse(BaseModel):
    """Base audio response model"""
    session_id: str
    user_id: str
    timestamp: datetime
    processing_time: float
    success: bool = True
    error: Optional[str] = None

class VoiceRequest(AudioRequest):
    """Voice processing request"""
    audio_data: bytes = Field(..., description="Raw audio data")
    sample_rate: int = Field(16000, description="Audio sample rate")
    channels: int = Field(1, description="Number of audio channels")
    format: AudioFormat = Field(AudioFormat.WAV, description="Audio format")
    
class VoiceResponse(AudioResponse):
    """Voice processing response"""
    text: str = Field(..., description="Transcribed text")
    confidence: float = Field(..., description="Confidence score")
    language: str = Field(..., description="Detected language")
    is_final: bool = Field(True, description="Is this a final transcription")

class TTSRequest(BaseModel):
    """Text-to-speech request"""
    text: str = Field(..., description="Text to synthesize")
    voice: Optional[str] = Field(None, description="Voice to use for synthesis")
    user_id: str
    session_id: Optional[str] = None
    
class TTSResponse(AudioResponse):
    """Text-to-speech response"""
    text: str = Field(..., description="Original text")
    voice: str = Field(..., description="Voice used for synthesis")
    duration: float = Field(..., description="Audio duration in seconds")
    sample_rate: int = Field(..., description="Audio sample rate")
    audio_data: Optional[bytes] = Field(None, description="Audio data (if requested)")

class StreamingConfig(BaseModel):
    """Configuration for real-time audio streaming"""
    sample_rate: Optional[int] = Field(16000, description="Audio sample rate")
    channels: Optional[int] = Field(1, description="Number of audio channels")
    chunk_size: Optional[int] = Field(1024, description="Audio chunk size")
    buffer_size: Optional[int] = Field(4096, description="Buffer size")
    format: Optional[AudioFormat] = Field(AudioFormat.WAV, description="Audio format")
    
    # STT Configuration
    stt_model: Optional[str] = Field("base", description="Whisper model size")
    stt_language: Optional[str] = Field("auto", description="STT language")
    stt_threshold: Optional[float] = Field(0.5, description="Voice activity threshold")
    stt_chunk_duration: Optional[float] = Field(1.0, description="STT chunk duration")
    
    # TTS Configuration
    tts_voice: Optional[str] = Field("en-US-AriaNeural", description="TTS voice")
    tts_rate: Optional[str] = Field("+0%", description="TTS rate")
    tts_volume: Optional[str] = Field("+0%", description="TTS volume")
    tts_pitch: Optional[str] = Field("+0Hz", description="TTS pitch")
    
    # Real-time Processing
    enable_vad: Optional[bool] = Field(True, description="Enable voice activity detection")
    enable_noise_reduction: Optional[bool] = Field(True, description="Enable noise reduction")
    enable_echo_cancellation: Optional[bool] = Field(True, description="Enable echo cancellation")
    enable_auto_gain: Optional[bool] = Field(True, description="Enable automatic gain control")
    
    # Streaming Configuration
    streaming_chunk_ms: Optional[int] = Field(100, description="Streaming chunk size in ms")
    max_silence_ms: Optional[int] = Field(1000, description="Maximum silence duration")
    min_speech_ms: Optional[int] = Field(300, description="Minimum speech duration")

class AudioSession(BaseModel):
    """Audio session information"""
    session_id: str
    user_id: str
    state: StreamingState
    config: StreamingConfig
    created_at: datetime
    last_activity: datetime
    metrics: Dict[str, Any] = {}
    
class AudioMetrics(BaseModel):
    """Audio processing metrics"""
    session_id: str
    user_id: str
    
    # Processing metrics
    chunks_processed: int = 0
    stt_requests: int = 0
    tts_requests: int = 0
    errors: int = 0
    
    # Performance metrics
    avg_processing_time: float = 0.0
    avg_stt_confidence: float = 0.0
    total_audio_duration: float = 0.0
    
    # System metrics
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    network_usage: float = 0.0
    
    # Timestamps
    start_time: datetime
    last_update: datetime
    uptime: float = 0.0

class StreamingStatus(BaseModel):
    """Current streaming status"""
    session_id: str
    user_id: str
    state: StreamingState
    is_streaming: bool
    current_activity: Optional[str] = None
    last_stt_text: Optional[str] = None
    last_tts_text: Optional[str] = None
    timestamp: datetime

class AudioChunkData(BaseModel):
    """Audio chunk data for streaming"""
    data: bytes
    timestamp: float
    sample_rate: int
    sequence_id: int
    is_speech: bool = False
    confidence: float = 0.0

class STTResultData(BaseModel):
    """Speech-to-text result data"""
    text: str
    confidence: float
    language: str
    timestamp: float
    is_final: bool = False
    processing_time: float = 0.0
    session_id: str
    user_id: str

class TTSResultData(BaseModel):
    """Text-to-speech result data"""
    text: str
    voice: str
    duration: float
    sample_rate: int
    timestamp: float
    session_id: str
    user_id: str

class WebSocketMessage(BaseModel):
    """WebSocket message structure"""
    type: str
    data: Dict[str, Any] = {}
    timestamp: Optional[datetime] = None
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class AudioError(BaseModel):
    """Audio processing error"""
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: datetime
    session_id: Optional[str] = None
    user_id: Optional[str] = None

class AudioServiceHealth(BaseModel):
    """Audio service health status"""
    status: str = "healthy"
    services: Dict[str, str] = {}
    active_sessions: int = 0
    total_users: int = 0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    uptime_seconds: int = 0
    timestamp: datetime
    version: str = "2.0.0"

class AudioConfigRequest(BaseModel):
    """Request to update audio configuration"""
    sample_rate: Optional[int] = None
    chunk_size: Optional[int] = None
    stt_model: Optional[str] = None
    tts_voice: Optional[str] = None
    enable_vad: Optional[bool] = None
    enable_noise_reduction: Optional[bool] = None
    streaming_chunk_ms: Optional[int] = None
    max_silence_ms: Optional[int] = None
    min_speech_ms: Optional[int] = None

class AudioConfigResponse(BaseModel):
    """Response with current audio configuration"""
    config: StreamingConfig
    applied_at: datetime
    session_id: str
    user_id: str