"""
Jarvis Voice Assistant v2.0 - Core Service
FastAPI-based microservice orchestrator
"""

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Dict, List, Optional, Any
import logging
import uvicorn
import redis
import json
import asyncio
from datetime import datetime, timedelta
import uuid
from contextlib import asynccontextmanager

# Import our enhanced models and managers
from models import (
    User, UserCreate, UserUpdate, UserRole, UserStatus,
    SessionInfo, SessionCreate, SessionType, SessionStatus,
    ChatRequest, ChatResponse, VoiceRequest, VoiceResponse,
    HealthResponse, SystemStats, WebSocketMessage, WebSocketResponse,
    ErrorResponse, PaginationParams, AdminUserList, AdminSessionList
)
from user_manager import UserManager
from connection_manager import ConnectionManager
from plugin_integration import JarvisPluginManager

# Security
security = HTTPBearer(auto_error=False)

# Global instances - will be initialized in lifespan
redis_client = None
user_manager = None
connection_manager = None
plugin_manager = None

# Startup tracking
server_start_time = datetime.now()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global redis_client, user_manager, connection_manager, plugin_manager
    try:
        # Initialize Redis
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        logging.info("Connected to Redis")
        
        # Initialize managers
        user_manager = UserManager(redis_client)
        connection_manager = ConnectionManager(user_manager)
        
        # Initialize plugin manager
        plugin_manager = JarvisPluginManager(user_manager, connection_manager, redis_client)
        await plugin_manager.initialize()
        
        # Start background tasks
        asyncio.create_task(cleanup_task())
        
        logging.info("Jarvis v2.0 Core Service started successfully")
        
    except Exception as e:
        logging.error(f"Failed to initialize services: {e}")
        raise
    
    yield
    
    # Shutdown
    if redis_client:
        redis_client.close()
    logging.info("Jarvis v2.0 Core Service shutdown complete")

# FastAPI App
app = FastAPI(
    title="Jarvis Voice Assistant v2.0 - Core Service",
    description="Microservices orchestrator for Jarvis Voice Assistant",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly for production
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
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Optional[User]:
    """Get current user from token (simplified for demo - implement proper JWT)"""
    if not credentials:
        return None
    # For demo, token is just user_id
    return user_manager.get_user(credentials.credentials) if user_manager else None

async def require_user(user: User = Depends(get_current_user)) -> User:
    """Require authenticated user"""
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user

async def require_admin(user: User = Depends(require_user)) -> User:
    """Require admin user"""
    if user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Admin access required")
    return user

async def cleanup_task():
    """Background task for cleanup"""
    while True:
        try:
            await asyncio.sleep(300)  # Run every 5 minutes
            if user_manager and connection_manager:
                # Cleanup expired sessions
                user_manager.cleanup_expired_sessions()
                # Cleanup inactive connections
                await connection_manager.cleanup_inactive_connections()
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def get_redis_key(prefix: str, identifier: str) -> str:
    """Generate Redis key"""
    return f"jarvis:v2:{prefix}:{identifier}"

async def store_session(session_id: str, user_id: str, data: Dict[str, Any]):
    """Store session data in Redis"""
    try:
        if redis_client:
            key = get_redis_key("session", session_id)
            session_data = {
                "session_id": session_id,
                "user_id": user_id,
                "created_at": datetime.now().isoformat(),
                "last_activity": datetime.now().isoformat(),
                "data": data
            }
            redis_client.setex(key, 3600, json.dumps(session_data))  # 1 hour expiry
    except Exception as e:
        logger.error(f"Failed to store session: {e}")

async def get_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Get session data from Redis"""
    try:
        if redis_client:
            key = get_redis_key("session", session_id)
            data = redis_client.get(key)
            if data:
                return json.loads(data)
    except Exception as e:
        logger.error(f"Failed to get session: {e}")
    return None

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    return {
        "service": "Jarvis Voice Assistant Core v2.0",
        "status": "running",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/v2/health", response_model=HealthResponse)
async def health_check():
    services_status = {
        "core": "healthy",
        "redis": "healthy" if redis_client and redis_client.ping() else "unhealthy",
        "audio": "unknown",  # TODO: Check audio service
        "ai": "unknown",     # TODO: Check AI service
        "web": "unknown"     # TODO: Check web service
    }
    
    overall_status = "healthy" if all(s in ["healthy", "unknown"] for s in services_status.values()) else "degraded"
    
    return HealthResponse(
        status=overall_status,
        services=services_status,
        timestamp=datetime.now(),
        version="2.0.0"
    )

@app.post("/api/v2/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    start_time = datetime.now()
    
    try:
        # Generate session ID if not provided
        session_id = request.session_id or generate_session_id()
        
        # Store session info
        await store_session(session_id, request.user_id, {
            "last_message": request.message,
            "language": request.language
        })
        
        # Process through plugins first
        plugin_response = None
        if plugin_manager:
            plugin_response = await plugin_manager.process_chat_message(request, "")
        
        # Use plugin response if available, otherwise use default
        if plugin_response:
            response = plugin_response
        else:
            # Default echo response
            response_text = f"Echo: {request.message}"
            confidence = 0.95
            processing_time = (datetime.now() - start_time).total_seconds()
            
            response = ChatResponse(
                response=response_text,
                session_id=session_id,
                timestamp=datetime.now(),
                confidence=confidence,
                processing_time=processing_time
            )
        
        # Send via WebSocket if connected
        await connection_manager.send_personal_message(
            json.dumps({
                "type": "chat_response",
                "data": response.dict()
            }),
            request.user_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v2/voice")
async def voice_endpoint(request: VoiceRequest):
    try:
        # Generate session ID if not provided
        session_id = request.session_id or generate_session_id()
        
        # Forward to audio service for processing
        # TODO: Implement actual audio service integration
        return {
            "session_id": session_id,
            "status": "processing",
            "message": "Voice processing forwarded to audio service",
            "audio_service_url": "http://localhost:8001/api/v2/audio/session/start"
        }
        
    except Exception as e:
        logger.error(f"Voice endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/api/v2/sessions/{session_id}", response_model=SessionInfo)
async def get_session_info(session_id: str):
    session_data = await get_session(session_id)
    
    if not session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionInfo(
        session_id=session_id,
        user_id=session_data["user_id"],
        created_at=datetime.fromisoformat(session_data["created_at"]),
        last_activity=datetime.fromisoformat(session_data["last_activity"]),
        status="active",
        context_length=len(session_data.get("data", {}))
    )

@app.delete("/api/v2/sessions/{session_id}")
async def delete_session(session_id: str):
    if redis_client:
        key = get_redis_key("session", session_id)
        deleted = redis_client.delete(key)
        if deleted:
            return {"message": "Session deleted successfully"}
    
    raise HTTPException(status_code=404, detail="Session not found")

@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await connection_manager.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Handle different message types
            if message_data.get("type") == "chat":
                # Process chat message
                chat_request = ChatRequest(
                    message=message_data["message"],
                    user_id=user_id,
                    session_id=message_data.get("session_id")
                )
                response = await chat_endpoint(chat_request)
                
            elif message_data.get("type") == "ping":
                await connection_manager.send_personal_message(
                    json.dumps({"type": "pong", "timestamp": datetime.now().isoformat()}),
                    user_id
                )
                
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(user_id)

# User Management Endpoints
@app.post("/api/v2/users", response_model=User)
async def create_user(user_data: UserCreate, admin: User = Depends(require_admin)):
    try:
        return user_manager.create_user(user_data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Create user error: {e}")
        raise HTTPException(status_code=500, detail="Failed to create user")

@app.get("/api/v2/users/me", response_model=User)
async def get_current_user_info(user: User = Depends(require_user)):
    return user

@app.get("/api/v2/users/{user_id}", response_model=User)
async def get_user_by_id(user_id: str, current_user: User = Depends(require_user)):
    # Users can only see their own profile, admins can see all
    if user_id != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    user = user_manager.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return user

@app.put("/api/v2/users/{user_id}", response_model=User)
async def update_user(user_id: str, updates: UserUpdate, current_user: User = Depends(require_user)):
    # Users can only update their own profile, admins can update all
    if user_id != current_user.user_id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    # Non-admins cannot change role
    if current_user.role != UserRole.ADMIN and updates.role is not None:
        raise HTTPException(status_code=403, detail="Cannot change user role")
    
    updated_user = user_manager.update_user(user_id, updates)
    if not updated_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return updated_user

# Session Management Endpoints
@app.post("/api/v2/sessions", response_model=SessionInfo)
async def create_session(session_data: SessionCreate, user: User = Depends(require_user)):
    # Users can only create sessions for themselves
    if session_data.user_id != user.user_id and user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Access denied")
    
    session = user_manager.create_session(session_data)
    if not session:
        raise HTTPException(status_code=400, detail="Failed to create session")
    
    return session

@app.get("/api/v2/sessions", response_model=List[SessionInfo])
async def get_user_sessions(user: User = Depends(require_user)):
    return user_manager.get_user_active_sessions(user.user_id)

# Admin endpoints
@app.get("/api/v2/admin/stats", response_model=SystemStats)
async def get_system_stats(admin: User = Depends(require_admin)):
    connection_stats = connection_manager.get_system_stats() if connection_manager else {}
    uptime = int((datetime.now() - server_start_time).total_seconds())
    
    return SystemStats(
        active_connections=connection_stats.get("total_connections", 0),
        active_sessions=connection_stats.get("active_sessions", 0),
        total_users=user_manager.get_total_user_count(),
        redis_connected=redis_client is not None and redis_client.ping(),
        memory_usage_mb=0.0,  # TODO: Implement memory monitoring
        cpu_usage_percent=0.0,  # TODO: Implement CPU monitoring
        uptime_seconds=uptime,
        timestamp=datetime.now()
    )

@app.get("/api/v2/admin/users", response_model=AdminUserList)
async def list_users(admin: User = Depends(require_admin), pagination: PaginationParams = Depends()):
    users = user_manager.list_users(pagination.page, pagination.page_size)
    total_count = user_manager.get_total_user_count()
    
    return AdminUserList(
        users=users,
        total_count=total_count,
        page=pagination.page,
        page_size=pagination.page_size
    )

@app.get("/api/v2/admin/sessions", response_model=AdminSessionList)
async def list_all_sessions(admin: User = Depends(require_admin), pagination: PaginationParams = Depends()):
    all_sessions = user_manager.get_all_active_sessions()
    
    # Apply pagination
    start = (pagination.page - 1) * pagination.page_size
    end = start + pagination.page_size
    sessions = all_sessions[start:end]
    
    return AdminSessionList(
        sessions=sessions,
        total_count=len(all_sessions),
        page=pagination.page,
        page_size=pagination.page_size
    )

@app.get("/api/v2/admin/connections")
async def get_active_connections(admin: User = Depends(require_admin)):
    if not connection_manager:
        return {"error": "Connection manager not available"}
    
    return connection_manager.get_system_stats()

@app.delete("/api/v2/admin/users/{user_id}")
async def delete_user(user_id: str, admin: User = Depends(require_admin)):
    # Prevent self-deletion
    if user_id == admin.user_id:
        raise HTTPException(status_code=400, detail="Cannot delete your own account")
    
    if user_manager.delete_user(user_id):
        # Disconnect all user connections
        if connection_manager:
            await connection_manager.disconnect_user(user_id)
        return {"message": "User deleted successfully"}
    else:
        raise HTTPException(status_code=404, detail="User not found")

@app.post("/api/v2/admin/cleanup")
async def trigger_cleanup(admin: User = Depends(require_admin)):
    expired_sessions = user_manager.cleanup_expired_sessions()
    inactive_connections = 0
    
    if connection_manager:
        inactive_connections = await connection_manager.cleanup_inactive_connections()
    
    return {
        "message": "Cleanup completed",
        "expired_sessions": expired_sessions,
        "inactive_connections": inactive_connections
    }

# Plugin Management Endpoints
@app.get("/api/v2/plugins")
async def list_plugins(user: User = Depends(require_user)):
    """List all plugins and their commands"""
    if not plugin_manager:
        return {"error": "Plugin system not available"}
    
    plugins = plugin_manager.list_plugins()
    commands = plugin_manager.get_available_commands()
    stats = plugin_manager.get_plugin_stats()
    
    return {
        "plugins": plugins,
        "commands": commands,
        "stats": stats
    }

@app.get("/api/v2/plugins/{plugin_name}")
async def get_plugin_info(plugin_name: str, user: User = Depends(require_user)):
    """Get detailed information about a specific plugin"""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin system not available")
    
    plugin_info = plugin_manager.get_plugin_info(plugin_name)
    if not plugin_info:
        raise HTTPException(status_code=404, detail="Plugin not found")
    
    return plugin_info

@app.get("/api/v2/help/commands")
async def get_command_help(user: User = Depends(require_user)):
    """Get help information for all available commands"""
    if not plugin_manager:
        return {"commands": {}}
    
    commands = plugin_manager.get_available_commands()
    plugins = plugin_manager.list_plugins()
    
    # Build detailed command help
    command_help = {}
    for command, description in commands.items():
        # Find which plugin provides this command
        plugin_name = None
        plugin_version = None
        
        for plugin in plugins:
            plugin_commands = plugin.get("commands", [])
            if command in plugin_commands:
                plugin_name = plugin.get("name")
                plugin_version = plugin.get("metadata", {}).get("version")
                break
        
        command_help[command] = {
            "plugin": plugin_name,
            "description": description,
            "usage": f"/{command} [args]",
            "plugin_version": plugin_version
        }
    
    return {"commands": command_help}

# Admin Plugin Management
@app.post("/api/v2/admin/plugins/{plugin_name}/load")
async def load_plugin(plugin_name: str, admin: User = Depends(require_admin)):
    """Load a specific plugin"""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin system not available")
    
    success = await plugin_manager.load_plugin(plugin_name)
    if success:
        return {"message": f"Plugin '{plugin_name}' loaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to load plugin '{plugin_name}'")

@app.post("/api/v2/admin/plugins/{plugin_name}/unload")
async def unload_plugin(plugin_name: str, admin: User = Depends(require_admin)):
    """Unload a specific plugin"""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin system not available")
    
    success = await plugin_manager.unload_plugin(plugin_name)
    if success:
        return {"message": f"Plugin '{plugin_name}' unloaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to unload plugin '{plugin_name}'")

@app.post("/api/v2/admin/plugins/{plugin_name}/reload")
async def reload_plugin(plugin_name: str, admin: User = Depends(require_admin)):
    """Reload a specific plugin"""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin system not available")
    
    success = await plugin_manager.reload_plugin(plugin_name)
    if success:
        return {"message": f"Plugin '{plugin_name}' reloaded successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to reload plugin '{plugin_name}'")

@app.put("/api/v2/admin/plugins/{plugin_name}/config")
async def update_plugin_config(plugin_name: str, config: Dict[str, Any], admin: User = Depends(require_admin)):
    """Update plugin configuration"""
    if not plugin_manager:
        raise HTTPException(status_code=503, detail="Plugin system not available")
    
    success = await plugin_manager.update_plugin_config(plugin_name, config)
    if success:
        return {"message": f"Plugin '{plugin_name}' configuration updated successfully"}
    else:
        raise HTTPException(status_code=400, detail=f"Failed to update configuration for plugin '{plugin_name}'")

# Guest user creation (for demo purposes)
@app.post("/api/v2/guest", response_model=User)
async def create_guest_user():
    guest_data = UserCreate(
        username=f"guest_{uuid.uuid4().hex[:8]}",
        role=UserRole.GUEST
    )
    
    try:
        return user_manager.create_user(guest_data)
    except Exception as e:
        logger.error(f"Failed to create guest user: {e}")
        raise HTTPException(status_code=500, detail="Failed to create guest user")

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )