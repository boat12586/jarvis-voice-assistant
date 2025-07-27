"""
Pydantic models for Jarvis v2.0 Core Service
Enhanced with multi-user support and session isolation
"""

from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional, Any, Literal
from datetime import datetime
from enum import Enum
import uuid

# User Management Models
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    GUEST = "guest"

class UserStatus(str, Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    SUSPENDED = "suspended"

class User(BaseModel):
    user_id: str
    username: str
    email: Optional[str] = None
    role: UserRole = UserRole.USER
    status: UserStatus = UserStatus.ACTIVE
    created_at: datetime
    last_login: Optional[datetime] = None
    preferences: Dict[str, Any] = {}
    session_limit: int = 5  # Max concurrent sessions per user

class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    email: Optional[str] = None
    role: UserRole = UserRole.USER
    preferences: Dict[str, Any] = {}

    @validator('username')
    def username_alphanumeric(cls, v):
        if not v.replace('_', '').replace('-', '').isalnum():
            raise ValueError('Username must be alphanumeric (can include _ and -)')
        return v

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[str] = None
    role: Optional[UserRole] = None
    status: Optional[UserStatus] = None
    preferences: Optional[Dict[str, Any]] = None
    session_limit: Optional[int] = None

# Session Management Models
class SessionStatus(str, Enum):
    ACTIVE = "active"
    IDLE = "idle"
    EXPIRED = "expired"
    TERMINATED = "terminated"

class SessionType(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    API = "api"
    VOICE_ONLY = "voice_only"

class SessionInfo(BaseModel):
    session_id: str
    user_id: str
    session_type: SessionType = SessionType.WEB
    status: SessionStatus = SessionStatus.ACTIVE
    created_at: datetime
    last_activity: datetime
    expires_at: datetime
    context_length: int = 0
    metadata: Dict[str, Any] = {}
    
    # Session isolation data
    conversation_history: List[str] = []
    voice_settings: Dict[str, Any] = {}
    ui_preferences: Dict[str, Any] = {}

class SessionCreate(BaseModel):
    user_id: str
    session_type: SessionType = SessionType.WEB
    expires_in_hours: int = Field(default=24, ge=1, le=168)  # 1 hour to 1 week
    metadata: Dict[str, Any] = {}

# Chat and Voice Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000)
    user_id: str
    session_id: Optional[str] = None
    language: Optional[str] = "en"
    context: Optional[Dict[str, Any]] = {}

class ChatResponse(BaseModel):
    response: str
    session_id: str
    user_id: str
    timestamp: datetime
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    language: str = "en"
    metadata: Dict[str, Any] = {}

class VoiceRequest(BaseModel):
    audio_data: str  # Base64 encoded
    user_id: str
    session_id: Optional[str] = None
    format: str = Field(default="wav", pattern="^(wav|mp3|ogg|flac)$")
    sample_rate: int = Field(default=16000, ge=8000, le=48000)
    language: Optional[str] = "en"

class VoiceResponse(BaseModel):
    transcribed_text: Optional[str] = None
    response_text: Optional[str] = None
    audio_response: Optional[str] = None  # Base64 encoded
    session_id: str
    user_id: str
    timestamp: datetime
    confidence: float = Field(..., ge=0.0, le=1.0)
    processing_time: float
    language: str = "en"

# System Models
class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    services: Dict[str, str]
    timestamp: datetime
    version: str
    uptime_seconds: int = 0

class SystemStats(BaseModel):
    active_connections: int
    active_sessions: int
    total_users: int
    redis_connected: bool
    memory_usage_mb: float
    cpu_usage_percent: float
    uptime_seconds: int
    timestamp: datetime

class UserConnectionInfo(BaseModel):
    user_id: str
    username: str
    connection_count: int
    active_sessions: List[str]
    last_activity: datetime
    connection_type: str  # websocket, api, both

# WebSocket Message Models
class WebSocketMessage(BaseModel):
    type: Literal["chat", "voice", "ping", "pong", "status", "error", "notification"]
    data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))

class WebSocketResponse(BaseModel):
    type: str
    data: Any
    session_id: Optional[str] = None
    user_id: str
    timestamp: datetime
    message_id: str

# Admin Models
class AdminUserList(BaseModel):
    users: List[User]
    total_count: int
    page: int
    page_size: int

class AdminSessionList(BaseModel):
    sessions: List[SessionInfo]
    total_count: int
    page: int
    page_size: int

class SystemConfiguration(BaseModel):
    max_users: int = 1000
    max_sessions_per_user: int = 5
    session_timeout_hours: int = 24
    enable_guest_access: bool = True
    rate_limit_requests_per_minute: int = 60
    maintenance_mode: bool = False

# Error Models
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    error_code: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    session_id: Optional[str] = None
    user_id: Optional[str] = None

# Authentication Models
class AuthToken(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    user_id: str
    session_id: str

class AuthRequest(BaseModel):
    username: str
    password: Optional[str] = None  # For future authentication implementation
    session_type: SessionType = SessionType.WEB

# Pagination Models
class PaginationParams(BaseModel):
    page: int = Field(default=1, ge=1)
    page_size: int = Field(default=20, ge=1, le=100)
    sort_by: Optional[str] = None
    sort_order: Literal["asc", "desc"] = "desc"

# Activity Logging Models
class ActivityLog(BaseModel):
    log_id: str
    user_id: str
    session_id: Optional[str] = None
    activity_type: str  # chat, voice, login, logout, error
    description: str
    timestamp: datetime
    metadata: Dict[str, Any] = {}
    ip_address: Optional[str] = None