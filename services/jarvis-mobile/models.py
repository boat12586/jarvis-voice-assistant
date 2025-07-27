"""
Data models for Jarvis Mobile Gateway v2.0
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

class MobilePlatform(str, Enum):
    IOS = "ios"
    ANDROID = "android"

class SessionType(str, Enum):
    VOICE = "voice"
    CHAT = "chat"
    COMMAND = "command"

class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class MobileDeviceInfo(BaseModel):
    """Mobile device information"""
    device_id: str = Field(..., description="Unique device identifier")
    platform: MobilePlatform = Field(..., description="Mobile platform")
    push_token: Optional[str] = Field(None, description="Push notification token")
    app_version: str = Field(..., description="App version")
    os_version: str = Field(..., description="OS version")
    device_model: Optional[str] = Field(None, description="Device model")
    screen_size: Optional[str] = Field(None, description="Screen dimensions")
    timezone: Optional[str] = Field("UTC", description="Device timezone")

class MobileVoiceSettings(BaseModel):
    """Voice processing settings for mobile"""
    voice: str = Field("en-US-AriaNeural", description="TTS voice")
    language: str = Field("en", description="Recognition language")
    sample_rate: int = Field(16000, description="Audio sample rate")
    enable_noise_reduction: bool = Field(True, description="Enable noise reduction")
    voice_activity_detection: bool = Field(True, description="Enable VAD")
    auto_gain_control: bool = Field(True, description="Enable AGC")

class MobileUserPreferences(BaseModel):
    """User preferences for mobile app"""
    language: str = Field("en", description="Preferred language")
    theme: str = Field("dark", description="UI theme")
    notifications_enabled: bool = Field(True, description="Enable notifications")
    voice_settings: MobileVoiceSettings = Field(default_factory=MobileVoiceSettings)
    auto_sync: bool = Field(True, description="Enable automatic sync")
    offline_mode: bool = Field(False, description="Enable offline mode")

# Authentication Models
class MobileAuthRequest(BaseModel):
    """Mobile authentication request"""
    device_info: MobileDeviceInfo = Field(..., description="Device information")
    preferences: Optional[MobileUserPreferences] = Field(None, description="User preferences")
    biometric_auth: bool = Field(False, description="Use biometric authentication")

class MobileAuthResponse(BaseModel):
    """Mobile authentication response"""
    success: bool = Field(..., description="Authentication success")
    user_id: str = Field(..., description="User identifier")
    mobile_token: str = Field(..., description="Mobile JWT token")
    session_id: str = Field(..., description="Session identifier")
    expires_at: datetime = Field(..., description="Token expiration time")
    user_info: Dict[str, Any] = Field(..., description="User information")
    features: List[str] = Field(default_factory=list, description="Available features")

class MobileTokenRefresh(BaseModel):
    """Mobile token refresh request"""
    current_token: str = Field(..., description="Current JWT token")
    session_id: str = Field(..., description="Session identifier")
    device_id: str = Field(..., description="Device identifier")

# Voice Processing Models
class MobileVoiceRequest(BaseModel):
    """Mobile voice processing request"""
    session_id: str = Field(..., description="Session identifier")
    audio_data: Optional[str] = Field(None, description="Base64 encoded audio data")
    text: Optional[str] = Field(None, description="Text for TTS")
    voice_settings: Optional[MobileVoiceSettings] = Field(None, description="Voice settings")
    language: Optional[str] = Field("en", description="Language")
    format: str = Field("wav", description="Audio format")
    streaming: bool = Field(False, description="Enable streaming")

class MobileVoiceResponse(BaseModel):
    """Mobile voice processing response"""
    success: bool = Field(..., description="Processing success")
    session_id: str = Field(..., description="Session identifier")
    response_text: Optional[str] = Field(None, description="Recognized or synthesized text")
    response_audio: Optional[str] = Field(None, description="Base64 encoded audio response")
    processing_time: float = Field(..., description="Processing time in seconds")
    confidence: float = Field(..., description="Confidence score")
    language: str = Field(..., description="Detected/used language")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

# Command Processing Models
class MobileCommandRequest(BaseModel):
    """Mobile command processing request"""
    session_id: str = Field(..., description="Session identifier")
    command: str = Field(..., description="Command text")
    language: Optional[str] = Field("en", description="Command language")
    context: Dict[str, Any] = Field(default_factory=dict, description="Command context")
    send_notification: bool = Field(False, description="Send completion notification")

class MobileCommandResponse(BaseModel):
    """Mobile command processing response"""
    success: bool = Field(..., description="Command success")
    command: str = Field(..., description="Executed command")
    response: str = Field(..., description="Command response")
    session_id: str = Field(..., description="Session identifier")
    confidence: float = Field(..., description="Confidence score")
    processing_time: float = Field(..., description="Processing time")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Command metadata")

# Session Management Models
class MobileSessionRequest(BaseModel):
    """Mobile session creation request"""
    session_type: SessionType = Field(SessionType.CHAT, description="Session type")
    preferences: Optional[MobileUserPreferences] = Field(None, description="Session preferences")
    context: Dict[str, Any] = Field(default_factory=dict, description="Session context")

class MobileSessionInfo(BaseModel):
    """Mobile session information"""
    session_id: str = Field(..., description="Session identifier")
    user_id: str = Field(..., description="User identifier")
    device_id: str = Field(..., description="Device identifier")
    session_type: SessionType = Field(..., description="Session type")
    created_at: datetime = Field(..., description="Creation time")
    last_activity: datetime = Field(..., description="Last activity time")
    is_active: bool = Field(..., description="Session active status")
    preferences: Dict[str, Any] = Field(default_factory=dict, description="Session preferences")

# Notification Models
class MobileNotificationRequest(BaseModel):
    """Mobile notification registration request"""
    platform: MobilePlatform = Field(..., description="Mobile platform")
    push_token: str = Field(..., description="Push notification token")
    app_version: str = Field(..., description="App version")
    os_version: str = Field(..., description="OS version")
    notification_preferences: Dict[str, bool] = Field(default_factory=dict, description="Notification preferences")

class MobileNotificationMessage(BaseModel):
    """Mobile notification message"""
    title: str = Field(..., description="Notification title")
    body: str = Field(..., description="Notification body")
    priority: NotificationPriority = Field(NotificationPriority.NORMAL, description="Notification priority")
    data: Dict[str, Any] = Field(default_factory=dict, description="Additional data")
    actions: List[str] = Field(default_factory=list, description="Available actions")

# Sync Models
class MobileSyncRequest(BaseModel):
    """Mobile data sync request"""
    session_id: str = Field(..., description="Session identifier")
    last_sync: Optional[datetime] = Field(None, description="Last sync time")
    client_version: str = Field(..., description="Client version")
    preferences: Optional[MobileUserPreferences] = Field(None, description="Client preferences")
    sync_data: Dict[str, Any] = Field(default_factory=dict, description="Data to sync")

class MobileSyncResponse(BaseModel):
    """Mobile data sync response"""
    success: bool = Field(..., description="Sync success")
    last_sync: datetime = Field(..., description="Server sync time")
    server_preferences: Dict[str, Any] = Field(default_factory=dict, description="Server preferences")
    sync_conflicts: List[str] = Field(default_factory=list, description="Sync conflicts")
    server_version: str = Field(..., description="Server version")
    sync_data: Dict[str, Any] = Field(default_factory=dict, description="Synced data")

# Health and Status Models
class MobileHealthResponse(BaseModel):
    """Mobile gateway health response"""
    status: str = Field(..., description="Overall health status")
    services: Dict[str, str] = Field(..., description="Service statuses")
    active_sessions: int = Field(..., description="Active mobile sessions")
    registered_devices: int = Field(..., description="Registered devices")
    timestamp: datetime = Field(..., description="Health check timestamp")
    version: str = Field(..., description="Gateway version")

# Analytics Models
class MobileAnalyticsEvent(BaseModel):
    """Mobile analytics event"""
    event_type: str = Field(..., description="Event type")
    user_id: str = Field(..., description="User identifier")
    device_id: str = Field(..., description="Device identifier")
    session_id: str = Field(..., description="Session identifier")
    timestamp: datetime = Field(default_factory=datetime.now, description="Event timestamp")
    properties: Dict[str, Any] = Field(default_factory=dict, description="Event properties")

class MobileUsageStats(BaseModel):
    """Mobile usage statistics"""
    total_users: int = Field(..., description="Total users")
    active_users: int = Field(..., description="Active users")
    total_sessions: int = Field(..., description="Total sessions")
    avg_session_duration: float = Field(..., description="Average session duration")
    total_commands: int = Field(..., description="Total commands executed")
    total_voice_requests: int = Field(..., description="Total voice requests")
    platform_breakdown: Dict[str, int] = Field(default_factory=dict, description="Platform usage")

# Configuration Models
class MobileConfigResponse(BaseModel):
    """Mobile configuration response"""
    features: List[str] = Field(..., description="Available features")
    limits: Dict[str, int] = Field(default_factory=dict, description="Usage limits")
    settings: Dict[str, Any] = Field(default_factory=dict, description="Configuration settings")
    version: str = Field(..., description="Configuration version")

# Error Models
class MobileErrorResponse(BaseModel):
    """Mobile error response"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    code: int = Field(..., description="Error code")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

# Batch Processing Models
class MobileBatchRequest(BaseModel):
    """Mobile batch processing request"""
    requests: List[Dict[str, Any]] = Field(..., description="Batch requests")
    session_id: str = Field(..., description="Session identifier")
    parallel: bool = Field(False, description="Process in parallel")

class MobileBatchResponse(BaseModel):
    """Mobile batch processing response"""
    success: bool = Field(..., description="Batch success")
    responses: List[Dict[str, Any]] = Field(..., description="Individual responses")
    failed_requests: List[int] = Field(default_factory=list, description="Failed request indices")
    processing_time: float = Field(..., description="Total processing time")

# Offline Models
class MobileOfflineRequest(BaseModel):
    """Mobile offline request"""
    request_id: str = Field(..., description="Request identifier")
    request_type: str = Field(..., description="Request type")
    request_data: Dict[str, Any] = Field(..., description="Request data")
    created_at: datetime = Field(default_factory=datetime.now, description="Request creation time")

class MobileOfflineResponse(BaseModel):
    """Mobile offline response"""
    success: bool = Field(..., description="Processing success")
    processed_requests: List[str] = Field(..., description="Processed request IDs")
    failed_requests: List[str] = Field(default_factory=list, description="Failed request IDs")
    sync_time: datetime = Field(..., description="Sync completion time")