"""
Jarvis Mobile Gateway v2.0
Mobile-optimized API gateway for iOS and Android applications
"""

from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
import logging
import uvicorn
import httpx
import json
import asyncio
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager
import base64
import hashlib
import jwt
from cryptography.fernet import Fernet

# Import models
from models import (
    MobileAuthRequest, MobileAuthResponse, MobileTokenRefresh,
    MobileVoiceRequest, MobileVoiceResponse, MobileSessionRequest,
    MobileNotificationRequest, MobileDeviceInfo, MobileUserPreferences,
    MobileCommandRequest, MobileCommandResponse, MobileHealthResponse,
    MobileSyncRequest, MobileSyncResponse
)

# Security
security = HTTPBearer(auto_error=False)

# Global instances
core_service_client = None
audio_service_client = None
notification_service = None
device_registry = {}
active_sessions = {}

# Configuration
CORE_SERVICE_URL = "http://jarvis-core:8000"
AUDIO_SERVICE_URL = "http://jarvis-audio:8001"
JWT_SECRET = "your_mobile_jwt_secret_here"
ENCRYPTION_KEY = Fernet.generate_key()
cipher_suite = Fernet(ENCRYPTION_KEY)

# Startup tracking
server_start_time = datetime.now()

class MobileNotificationService:
    """Handle mobile push notifications"""
    
    def __init__(self):
        self.fcm_key = None  # Firebase Cloud Messaging key
        self.apns_key = None  # Apple Push Notification Service key
        self.registered_devices = {}
        
    def register_device(self, user_id: str, device_info: MobileDeviceInfo):
        """Register mobile device for push notifications"""
        self.registered_devices[user_id] = {
            "device_id": device_info.device_id,
            "platform": device_info.platform,
            "push_token": device_info.push_token,
            "app_version": device_info.app_version,
            "registered_at": datetime.now().isoformat()
        }
        
    async def send_notification(self, user_id: str, title: str, message: str, data: Dict[str, Any] = None):
        """Send push notification to user's device"""
        if user_id not in self.registered_devices:
            return False
            
        device = self.registered_devices[user_id]
        
        # Mock notification sending - implement with actual FCM/APNS
        notification = {
            "title": title,
            "message": message,
            "data": data or {},
            "device_id": device["device_id"],
            "platform": device["platform"],
            "sent_at": datetime.now().isoformat()
        }
        
        logging.info(f"Sending notification to {user_id}: {title}")
        return True

class MobileSessionManager:
    """Manage mobile app sessions"""
    
    def __init__(self):
        self.sessions = {}
        
    def create_session(self, user_id: str, device_id: str) -> str:
        """Create new mobile session"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "device_id": device_id,
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "is_active": True,
            "voice_session_id": None,
            "sync_data": {}
        }
        return session_id
        
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session info"""
        return self.sessions.get(session_id)
        
    def update_activity(self, session_id: str):
        """Update session activity"""
        if session_id in self.sessions:
            self.sessions[session_id]["last_activity"] = datetime.now()
            
    def cleanup_expired_sessions(self):
        """Remove expired sessions"""
        current_time = datetime.now()
        expired_sessions = []
        
        for session_id, session in self.sessions.items():
            if current_time - session["last_activity"] > timedelta(hours=24):
                expired_sessions.append(session_id)
                
        for session_id in expired_sessions:
            del self.sessions[session_id]
            
        return len(expired_sessions)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global core_service_client, audio_service_client, notification_service, session_manager
    try:
        logging.info("Starting Jarvis Mobile Gateway...")
        
        # Initialize HTTP clients
        core_service_client = httpx.AsyncClient(base_url=CORE_SERVICE_URL)
        audio_service_client = httpx.AsyncClient(base_url=AUDIO_SERVICE_URL)
        
        # Initialize services
        notification_service = MobileNotificationService()
        session_manager = MobileSessionManager()
        
        # Start background tasks
        asyncio.create_task(cleanup_task())
        
        logging.info("Mobile Gateway initialized successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize mobile gateway: {e}")
        raise
    
    yield
    
    # Shutdown
    if core_service_client:
        await core_service_client.aclose()
    if audio_service_client:
        await audio_service_client.aclose()
    logging.info("Mobile Gateway shutdown complete")

# FastAPI App
app = FastAPI(
    title="Jarvis Mobile Gateway v2.0",
    description="Mobile-optimized API gateway for Jarvis Voice Assistant",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware for mobile apps
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for mobile apps
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
def generate_mobile_token(user_id: str, device_id: str) -> str:
    """Generate JWT token for mobile app"""
    payload = {
        "user_id": user_id,
        "device_id": device_id,
        "issued_at": datetime.now().timestamp(),
        "expires_at": (datetime.now() + timedelta(days=30)).timestamp()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")

def verify_mobile_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify JWT token from mobile app"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=["HS256"])
        if payload["expires_at"] < datetime.now().timestamp():
            return None
        return payload
    except Exception:
        return None

async def get_current_mobile_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[Dict[str, Any]]:
    """Get current mobile user from token"""
    if not credentials:
        return None
    
    payload = verify_mobile_token(credentials.credentials)
    if not payload:
        return None
        
    return {
        "user_id": payload["user_id"],
        "device_id": payload["device_id"],
        "token": credentials.credentials
    }

async def require_mobile_user(user: Dict[str, Any] = Depends(get_current_mobile_user)) -> Dict[str, Any]:
    """Require authenticated mobile user"""
    if not user:
        raise HTTPException(status_code=401, detail="Mobile authentication required")
    return user

async def cleanup_task():
    """Background cleanup task"""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            if 'session_manager' in globals():
                expired_count = session_manager.cleanup_expired_sessions()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired mobile sessions")
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "Jarvis Mobile Gateway v2.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v2/mobile/health", response_model=MobileHealthResponse)
async def health_check():
    """Mobile gateway health check"""
    services_status = {
        "mobile_gateway": "healthy",
        "core_service": "unknown",
        "audio_service": "unknown",
        "notification_service": "healthy"
    }
    
    # Check core service
    try:
        response = await core_service_client.get("/api/v2/health")
        services_status["core_service"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services_status["core_service"] = "unhealthy"
    
    # Check audio service
    try:
        response = await audio_service_client.get("/api/v2/audio/health")
        services_status["audio_service"] = "healthy" if response.status_code == 200 else "unhealthy"
    except Exception:
        services_status["audio_service"] = "unhealthy"
    
    overall_status = "healthy" if all(s != "unhealthy" for s in services_status.values()) else "degraded"
    
    return MobileHealthResponse(
        status=overall_status,
        services=services_status,
        active_sessions=len(active_sessions),
        registered_devices=len(device_registry),
        timestamp=datetime.now(),
        version="2.0.0"
    )

# Authentication Endpoints
@app.post("/api/v2/mobile/auth/login", response_model=MobileAuthResponse)
async def mobile_login(request: MobileAuthRequest):
    """Mobile app authentication"""
    try:
        # Create guest user in core service
        response = await core_service_client.post("/api/v2/guest")
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to create user session")
        
        user_data = response.json()
        user_id = user_data["user_id"]
        
        # Register device
        device_registry[user_id] = request.device_info
        notification_service.register_device(user_id, request.device_info)
        
        # Generate mobile token
        mobile_token = generate_mobile_token(user_id, request.device_info.device_id)
        
        # Create mobile session
        session_id = session_manager.create_session(user_id, request.device_info.device_id)
        
        return MobileAuthResponse(
            success=True,
            user_id=user_id,
            mobile_token=mobile_token,
            session_id=session_id,
            expires_at=datetime.now() + timedelta(days=30),
            user_info={
                "username": user_data["username"],
                "role": user_data["role"],
                "preferences": user_data.get("preferences", {})
            }
        )
        
    except Exception as e:
        logger.error(f"Mobile login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")

@app.post("/api/v2/mobile/auth/refresh", response_model=MobileAuthResponse)
async def refresh_mobile_token(request: MobileTokenRefresh):
    """Refresh mobile authentication token"""
    try:
        # Verify current token
        payload = verify_mobile_token(request.current_token)
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        user_id = payload["user_id"]
        device_id = payload["device_id"]
        
        # Generate new token
        new_token = generate_mobile_token(user_id, device_id)
        
        # Get user info from core service
        headers = {"Authorization": f"Bearer {user_id}"}
        response = await core_service_client.get(f"/api/v2/users/{user_id}", headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user info")
        
        user_data = response.json()
        
        return MobileAuthResponse(
            success=True,
            user_id=user_id,
            mobile_token=new_token,
            session_id=request.session_id,
            expires_at=datetime.now() + timedelta(days=30),
            user_info={
                "username": user_data["username"],
                "role": user_data["role"],
                "preferences": user_data.get("preferences", {})
            }
        )
        
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(status_code=500, detail="Token refresh failed")

# Voice Processing Endpoints
@app.post("/api/v2/mobile/voice/process", response_model=MobileVoiceResponse)
async def process_mobile_voice(request: MobileVoiceRequest, user: Dict[str, Any] = Depends(require_mobile_user)):
    """Process voice input from mobile app"""
    try:
        # Decode base64 audio data
        audio_data = base64.b64decode(request.audio_data)
        
        # Start audio session if needed
        headers = {"Authorization": f"Bearer {user['user_id']}"}
        audio_session_response = await audio_service_client.post(
            "/api/v2/audio/session/start",
            headers=headers
        )
        
        if audio_session_response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to start audio session")
        
        # Process TTS if text provided
        if request.text:
            tts_payload = {
                "text": request.text,
                "voice": request.voice_settings.voice if request.voice_settings else "en-US-AriaNeural"
            }
            
            tts_response = await audio_service_client.post(
                "/api/v2/audio/tts",
                json=tts_payload,
                headers=headers
            )
            
            if tts_response.status_code == 200:
                tts_data = tts_response.json()
                
                return MobileVoiceResponse(
                    success=True,
                    session_id=request.session_id,
                    response_text=request.text,
                    response_audio=None,  # Base64 encoded audio would go here
                    processing_time=tts_data.get("processing_time", 0.0),
                    confidence=1.0,
                    language=request.language or "en"
                )
        
        # For now, return success without actual audio processing
        return MobileVoiceResponse(
            success=True,
            session_id=request.session_id,
            response_text="Voice processing completed",
            response_audio=None,
            processing_time=0.1,
            confidence=0.95,
            language=request.language or "en"
        )
        
    except Exception as e:
        logger.error(f"Mobile voice processing error: {e}")
        raise HTTPException(status_code=500, detail="Voice processing failed")

# Command Processing Endpoints
@app.post("/api/v2/mobile/command", response_model=MobileCommandResponse)
async def process_mobile_command(request: MobileCommandRequest, user: Dict[str, Any] = Depends(require_mobile_user)):
    """Process command from mobile app"""
    try:
        # Forward command to core service
        headers = {"Authorization": f"Bearer {user['user_id']}"}
        chat_payload = {
            "message": request.command,
            "user_id": user["user_id"],
            "session_id": request.session_id,
            "language": request.language or "en"
        }
        
        response = await core_service_client.post(
            "/api/v2/chat",
            json=chat_payload,
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Command processing failed")
        
        chat_data = response.json()
        
        # Send push notification if requested
        if request.send_notification:
            await notification_service.send_notification(
                user["user_id"],
                "Command Completed",
                f"'{request.command}' executed successfully",
                {"command": request.command, "response": chat_data["response"]}
            )
        
        return MobileCommandResponse(
            success=True,
            command=request.command,
            response=chat_data["response"],
            session_id=chat_data["session_id"],
            confidence=chat_data.get("confidence", 1.0),
            processing_time=chat_data.get("processing_time", 0.0),
            metadata=chat_data.get("metadata", {})
        )
        
    except Exception as e:
        logger.error(f"Mobile command processing error: {e}")
        raise HTTPException(status_code=500, detail="Command processing failed")

# Session Management
@app.post("/api/v2/mobile/session")
async def create_mobile_session(request: MobileSessionRequest, user: Dict[str, Any] = Depends(require_mobile_user)):
    """Create new mobile session"""
    try:
        session_id = session_manager.create_session(user["user_id"], user["device_id"])
        
        # Update session in active sessions
        active_sessions[session_id] = {
            "user_id": user["user_id"],
            "device_id": user["device_id"],
            "session_type": request.session_type,
            "created_at": datetime.now().isoformat(),
            "preferences": request.preferences.dict() if request.preferences else {}
        }
        
        return {
            "success": True,
            "session_id": session_id,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Mobile session creation error: {e}")
        raise HTTPException(status_code=500, detail="Session creation failed")

@app.get("/api/v2/mobile/session/{session_id}")
async def get_mobile_session(session_id: str, user: Dict[str, Any] = Depends(require_mobile_user)):
    """Get mobile session info"""
    try:
        session = session_manager.get_session(session_id)
        
        if not session or session["user_id"] != user["user_id"]:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session_id,
            "user_id": session["user_id"],
            "device_id": session["device_id"],
            "created_at": session["created_at"].isoformat(),
            "last_activity": session["last_activity"].isoformat(),
            "is_active": session["is_active"]
        }
        
    except Exception as e:
        logger.error(f"Mobile session retrieval error: {e}")
        raise HTTPException(status_code=500, detail="Session retrieval failed")

# Notification Endpoints
@app.post("/api/v2/mobile/notifications/register")
async def register_for_notifications(request: MobileNotificationRequest, user: Dict[str, Any] = Depends(require_mobile_user)):
    """Register device for push notifications"""
    try:
        device_info = MobileDeviceInfo(
            device_id=user["device_id"],
            platform=request.platform,
            push_token=request.push_token,
            app_version=request.app_version,
            os_version=request.os_version
        )
        
        notification_service.register_device(user["user_id"], device_info)
        
        return {
            "success": True,
            "message": "Device registered for notifications",
            "registered_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Notification registration error: {e}")
        raise HTTPException(status_code=500, detail="Notification registration failed")

@app.post("/api/v2/mobile/notifications/test")
async def test_notification(user: Dict[str, Any] = Depends(require_mobile_user)):
    """Send test notification"""
    try:
        success = await notification_service.send_notification(
            user["user_id"],
            "Test Notification",
            "This is a test notification from Jarvis!",
            {"test": True}
        )
        
        return {
            "success": success,
            "message": "Test notification sent" if success else "Failed to send notification"
        }
        
    except Exception as e:
        logger.error(f"Test notification error: {e}")
        raise HTTPException(status_code=500, detail="Test notification failed")

# Sync Endpoints
@app.post("/api/v2/mobile/sync", response_model=MobileSyncResponse)
async def sync_mobile_data(request: MobileSyncRequest, user: Dict[str, Any] = Depends(require_mobile_user)):
    """Sync mobile app data with server"""
    try:
        # Get user preferences from core service
        headers = {"Authorization": f"Bearer {user['user_id']}"}
        response = await core_service_client.get(f"/api/v2/users/{user['user_id']}", headers=headers)
        
        if response.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to get user data")
        
        user_data = response.json()
        
        # Update session sync data
        session = session_manager.get_session(request.session_id)
        if session:
            session["sync_data"] = {
                "last_sync": datetime.now().isoformat(),
                "client_version": request.client_version,
                "sync_preferences": request.preferences.dict() if request.preferences else {}
            }
        
        return MobileSyncResponse(
            success=True,
            last_sync=datetime.now(),
            server_preferences=user_data.get("preferences", {}),
            sync_conflicts=[],  # No conflicts for now
            server_version="2.0.0"
        )
        
    except Exception as e:
        logger.error(f"Mobile sync error: {e}")
        raise HTTPException(status_code=500, detail="Sync failed")

# Device Management
@app.get("/api/v2/mobile/devices")
async def get_user_devices(user: Dict[str, Any] = Depends(require_mobile_user)):
    """Get user's registered devices"""
    try:
        devices = []
        if user["user_id"] in device_registry:
            device_info = device_registry[user["user_id"]]
            devices.append({
                "device_id": device_info.device_id,
                "platform": device_info.platform,
                "app_version": device_info.app_version,
                "os_version": device_info.os_version,
                "last_active": datetime.now().isoformat()
            })
        
        return {
            "devices": devices,
            "total_devices": len(devices)
        }
        
    except Exception as e:
        logger.error(f"Device listing error: {e}")
        raise HTTPException(status_code=500, detail="Device listing failed")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8002,
        reload=True,
        log_level="info"
    )