"""
Real-time Audio Service for Jarvis v2.0
FastAPI service for real-time TTS/STT with WebSocket support
"""

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
import logging
import uvicorn
import json
import asyncio
from datetime import datetime
import uuid
from contextlib import asynccontextmanager

from real_time_audio import (
    AudioStreamManager, AudioConfig, AudioFormat, StreamingState,
    STTResult, TTSResult, AudioChunk
)
from models import (
    AudioRequest, AudioResponse, StreamingConfig, AudioMetrics,
    VoiceRequest, VoiceResponse, AudioSession
)

# Security
security = HTTPBearer(auto_error=False)

# Global instances
audio_manager = None
active_sessions = {}  # user_id -> AudioStreamManager
connection_manager = None

# Startup tracking
server_start_time = datetime.now()

class AudioConnectionManager:
    """Manage WebSocket connections for real-time audio"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, str] = {}  # user_id -> session_id
        
    async def connect(self, websocket: WebSocket, user_id: str, session_id: str):
        """Connect user to audio streaming"""
        await websocket.accept()
        self.active_connections[user_id] = websocket
        self.user_sessions[user_id] = session_id
        logging.info(f"Audio WebSocket connected: {user_id}")
        
    def disconnect(self, user_id: str):
        """Disconnect user from audio streaming"""
        if user_id in self.active_connections:
            del self.active_connections[user_id]
        if user_id in self.user_sessions:
            del self.user_sessions[user_id]
        logging.info(f"Audio WebSocket disconnected: {user_id}")
        
    async def send_personal_message(self, message: str, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_text(message)
            except Exception as e:
                logging.error(f"Failed to send message to {user_id}: {e}")
                self.disconnect(user_id)
                
    async def send_audio_data(self, audio_data: bytes, user_id: str):
        """Send audio data to specific user"""
        if user_id in self.active_connections:
            try:
                await self.active_connections[user_id].send_bytes(audio_data)
            except Exception as e:
                logging.error(f"Failed to send audio to {user_id}: {e}")
                self.disconnect(user_id)
                
    def get_session_id(self, user_id: str) -> Optional[str]:
        """Get session ID for user"""
        return self.user_sessions.get(user_id)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global audio_manager, connection_manager
    try:
        logging.info("Starting Jarvis Real-time Audio Service...")
        
        # Initialize connection manager
        connection_manager = AudioConnectionManager()
        
        # Initialize default audio config
        default_config = AudioConfig(
            sample_rate=16000,
            channels=1,
            chunk_size=1024,
            buffer_size=4096,
            format=AudioFormat.WAV,
            stt_model="base",
            tts_voice="en-US-AriaNeural",
            enable_vad=True,
            enable_noise_reduction=True,
            streaming_chunk_ms=100,
            max_silence_ms=1000,
            min_speech_ms=300
        )
        
        logging.info("Audio service initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize audio service: {e}")
        raise
    
    yield
    
    # Shutdown
    try:
        # Stop all active audio sessions
        for user_id, manager in active_sessions.items():
            manager.stop_streaming()
        active_sessions.clear()
        
        logging.info("Audio service shutdown complete")
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")

# FastAPI App
app = FastAPI(
    title="Jarvis Real-time Audio Service",
    description="Real-time TTS/STT processing with WebSocket streaming",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Helper Functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[str]:
    """Get current user from token (simplified for demo)"""
    if not credentials:
        return None
    return credentials.credentials  # Token is just user_id for demo

async def require_user(user_id: str = Depends(get_current_user)) -> str:
    """Require authenticated user"""
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id

def get_or_create_audio_manager(user_id: str, config: AudioConfig = None) -> AudioStreamManager:
    """Get or create audio manager for user"""
    if user_id not in active_sessions:
        if config is None:
            config = AudioConfig()
        
        manager = AudioStreamManager(config)
        active_sessions[user_id] = manager
        
        # Set up callbacks for WebSocket notifications
        manager.add_stt_callback(lambda result: handle_stt_result(user_id, result))
        manager.add_tts_callback(lambda result: handle_tts_result(user_id, result))
        manager.add_state_callback(lambda state: handle_state_change(user_id, state))
        
    return active_sessions[user_id]

def handle_stt_result(user_id: str, result: STTResult):
    """Handle STT result and send to user"""
    if connection_manager:
        message = {
            "type": "stt_result",
            "data": {
                "text": result.text,
                "confidence": result.confidence,
                "language": result.language,
                "is_final": result.is_final,
                "processing_time": result.processing_time,
                "timestamp": result.timestamp
            }
        }
        asyncio.create_task(connection_manager.send_personal_message(
            json.dumps(message), user_id
        ))

def handle_tts_result(user_id: str, result: TTSResult):
    """Handle TTS result and send audio to user"""
    if connection_manager:
        # Send metadata first
        message = {
            "type": "tts_result",
            "data": {
                "text": result.text,
                "voice": result.voice,
                "duration": result.duration,
                "sample_rate": result.sample_rate,
                "timestamp": result.timestamp
            }
        }
        asyncio.create_task(connection_manager.send_personal_message(
            json.dumps(message), user_id
        ))
        
        # Send audio data
        audio_bytes = result.audio_data.tobytes()
        asyncio.create_task(connection_manager.send_audio_data(
            audio_bytes, user_id
        ))

def handle_state_change(user_id: str, state: StreamingState):
    """Handle streaming state change"""
    if connection_manager:
        message = {
            "type": "state_change",
            "data": {
                "state": state.value,
                "timestamp": datetime.now().isoformat()
            }
        }
        asyncio.create_task(connection_manager.send_personal_message(
            json.dumps(message), user_id
        ))

# API Routes

@app.get("/")
async def root():
    return {
        "service": "Jarvis Real-time Audio Service",
        "version": "2.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v2/audio/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "active_sessions": len(active_sessions),
        "timestamp": datetime.now().isoformat(),
        "uptime": int((datetime.now() - server_start_time).total_seconds())
    }

@app.post("/api/v2/audio/session/start")
async def start_audio_session(
    config: Optional[StreamingConfig] = None,
    user_id: str = Depends(require_user)
):
    """Start real-time audio session"""
    try:
        # Create audio config
        audio_config = AudioConfig()
        if config:
            if config.sample_rate:
                audio_config.sample_rate = config.sample_rate
            if config.chunk_size:
                audio_config.chunk_size = config.chunk_size
            if config.stt_model:
                audio_config.stt_model = config.stt_model
            if config.tts_voice:
                audio_config.tts_voice = config.tts_voice
            if config.enable_vad is not None:
                audio_config.enable_vad = config.enable_vad
        
        # Get or create audio manager
        manager = get_or_create_audio_manager(user_id, audio_config)
        
        # Initialize if needed
        if not await manager.initialize():
            raise HTTPException(status_code=500, detail="Failed to initialize audio processing")
        
        # Generate session ID
        session_id = str(uuid.uuid4())
        
        # Start streaming
        manager.start_streaming(user_id, session_id)
        
        return {
            "session_id": session_id,
            "user_id": user_id,
            "status": "started",
            "config": {
                "sample_rate": audio_config.sample_rate,
                "chunk_size": audio_config.chunk_size,
                "stt_model": audio_config.stt_model,
                "tts_voice": audio_config.tts_voice
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to start audio session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/audio/session/stop")
async def stop_audio_session(user_id: str = Depends(require_user)):
    """Stop real-time audio session"""
    try:
        if user_id in active_sessions:
            manager = active_sessions[user_id]
            manager.stop_streaming()
            del active_sessions[user_id]
            
            return {
                "status": "stopped",
                "user_id": user_id,
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(status_code=404, detail="No active session found")
            
    except Exception as e:
        logger.error(f"Failed to stop audio session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v2/audio/tts")
async def text_to_speech(
    request: Dict[str, Any],
    user_id: str = Depends(require_user)
):
    """Convert text to speech"""
    try:
        text = request.get("text", "")
        voice = request.get("voice")
        
        if not text:
            raise HTTPException(status_code=400, detail="Text is required")
        
        # Get or create audio manager
        manager = get_or_create_audio_manager(user_id)
        
        # Initialize if needed
        if not await manager.initialize():
            raise HTTPException(status_code=500, detail="Failed to initialize audio processing")
        
        # Synthesize speech
        result = await manager.synthesize_speech(text, voice)
        
        if result:
            return {
                "success": True,
                "text": result.text,
                "voice": result.voice,
                "duration": result.duration,
                "sample_rate": result.sample_rate,
                "timestamp": result.timestamp
            }
        else:
            raise HTTPException(status_code=500, detail="Speech synthesis failed")
            
    except Exception as e:
        logger.error(f"TTS error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v2/audio/session/metrics")
async def get_session_metrics(user_id: str = Depends(require_user)):
    """Get audio session metrics"""
    try:
        if user_id not in active_sessions:
            raise HTTPException(status_code=404, detail="No active session found")
        
        manager = active_sessions[user_id]
        metrics = manager.get_metrics()
        
        return {
            "user_id": user_id,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/audio/{user_id}")
async def websocket_audio_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time audio streaming"""
    session_id = str(uuid.uuid4())
    
    await connection_manager.connect(websocket, user_id, session_id)
    
    try:
        while True:
            # Receive data from client
            data = await websocket.receive()
            
            if "text" in data:
                # Handle text messages (control commands)
                message = json.loads(data["text"])
                await handle_websocket_message(message, user_id, session_id)
                
            elif "bytes" in data:
                # Handle binary data (audio chunks)
                audio_data = data["bytes"]
                await handle_audio_data(audio_data, user_id, session_id)
                
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
        
        # Stop audio session if active
        if user_id in active_sessions:
            active_sessions[user_id].stop_streaming()
            del active_sessions[user_id]
            
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(user_id)

async def handle_websocket_message(message: Dict[str, Any], user_id: str, session_id: str):
    """Handle WebSocket text messages"""
    try:
        message_type = message.get("type")
        
        if message_type == "start_streaming":
            # Start audio streaming
            config_data = message.get("config", {})
            config = AudioConfig(**config_data) if config_data else AudioConfig()
            
            manager = get_or_create_audio_manager(user_id, config)
            await manager.initialize()
            manager.start_streaming(user_id, session_id)
            
            await connection_manager.send_personal_message(
                json.dumps({"type": "streaming_started", "session_id": session_id}),
                user_id
            )
            
        elif message_type == "stop_streaming":
            # Stop audio streaming
            if user_id in active_sessions:
                active_sessions[user_id].stop_streaming()
                del active_sessions[user_id]
                
            await connection_manager.send_personal_message(
                json.dumps({"type": "streaming_stopped"}),
                user_id
            )
            
        elif message_type == "synthesize_speech":
            # Synthesize speech
            text = message.get("text", "")
            voice = message.get("voice")
            
            if text and user_id in active_sessions:
                manager = active_sessions[user_id]
                await manager.synthesize_speech(text, voice)
                
        elif message_type == "ping":
            # Ping/pong for keepalive
            await connection_manager.send_personal_message(
                json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                user_id
            )
            
    except Exception as e:
        logger.error(f"Error handling WebSocket message: {e}")
        await connection_manager.send_personal_message(
            json.dumps({"type": "error", "message": str(e)}),
            user_id
        )

async def handle_audio_data(audio_data: bytes, user_id: str, session_id: str):
    """Handle incoming audio data"""
    try:
        if user_id not in active_sessions:
            return
        
        manager = active_sessions[user_id]
        
        # Convert bytes to numpy array
        import numpy as np
        audio_array = np.frombuffer(audio_data, dtype=np.float32)
        
        # Create audio chunk
        chunk = AudioChunk(
            data=audio_array,
            timestamp=datetime.now().timestamp(),
            sample_rate=manager.config.sample_rate
        )
        
        # Process through STT
        manager.stt.process_audio_chunk(chunk)
        
    except Exception as e:
        logger.error(f"Error processing audio data: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8001,
        reload=True,
        log_level="info"
    )