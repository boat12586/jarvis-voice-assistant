"""
Multi-User Manager for Jarvis v2.0
Handles user authentication, session management, and isolation
"""

import uuid
import json
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set
import redis
import logging
from models import (
    User, UserCreate, UserUpdate, UserRole, UserStatus,
    SessionInfo, SessionCreate, SessionStatus, SessionType,
    ActivityLog, SystemConfiguration
)

logger = logging.getLogger(__name__)

class UserManager:
    """Manages users, sessions, and isolation for multi-user support"""
    
    def __init__(self, redis_client: redis.Redis):
        self.redis_client = redis_client
        self.config = SystemConfiguration()
        
        # Redis key prefixes
        self.USER_PREFIX = "jarvis:v2:user"
        self.SESSION_PREFIX = "jarvis:v2:session"
        self.USER_SESSIONS_PREFIX = "jarvis:v2:user_sessions"
        self.ACTIVITY_PREFIX = "jarvis:v2:activity"
        self.CONFIG_PREFIX = "jarvis:v2:config"
        
        # In-memory caches for performance
        self.active_sessions: Dict[str, SessionInfo] = {}
        self.user_session_map: Dict[str, Set[str]] = {}  # user_id -> session_ids
        
        # Initialize default admin user
        self._ensure_default_admin()
    
    def _ensure_default_admin(self):
        """Create default admin user if none exists"""
        try:
            admin_user_id = "admin_default"
            if not self.get_user(admin_user_id):
                admin_user = User(
                    user_id=admin_user_id,
                    username="admin",
                    email="admin@jarvis.local",
                    role=UserRole.ADMIN,
                    status=UserStatus.ACTIVE,
                    created_at=datetime.now(),
                    session_limit=10,
                    preferences={
                        "theme": "dark",
                        "language": "en",
                        "voice_enabled": True
                    }
                )
                self.create_user_from_model(admin_user)
                logger.info("Default admin user created")
        except Exception as e:
            logger.error(f"Failed to create default admin user: {e}")
    
    def _get_redis_key(self, prefix: str, identifier: str) -> str:
        """Generate Redis key with prefix"""
        return f"{prefix}:{identifier}"
    
    def _generate_user_id(self) -> str:
        """Generate unique user ID"""
        return f"user_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
    
    def _generate_session_id(self) -> str:
        """Generate unique session ID"""
        return f"sess_{uuid.uuid4().hex[:8]}_{int(datetime.now().timestamp())}"
    
    # User Management
    def create_user(self, user_data: UserCreate) -> User:
        """Create a new user"""
        try:
            # Check if username already exists
            if self.get_user_by_username(user_data.username):
                raise ValueError(f"Username '{user_data.username}' already exists")
            
            # Check user limit
            if self.get_total_user_count() >= self.config.max_users:
                raise ValueError("Maximum user limit reached")
            
            user_id = self._generate_user_id()
            user = User(
                user_id=user_id,
                username=user_data.username,
                email=user_data.email,
                role=user_data.role,
                status=UserStatus.ACTIVE,
                created_at=datetime.now(),
                preferences=user_data.preferences
            )
            
            return self.create_user_from_model(user)
            
        except Exception as e:
            logger.error(f"Failed to create user: {e}")
            raise
    
    def create_user_from_model(self, user: User) -> User:
        """Create user from User model"""
        try:
            user_key = self._get_redis_key(self.USER_PREFIX, user.user_id)
            username_key = self._get_redis_key(f"{self.USER_PREFIX}:username", user.username)
            
            # Store user data
            user_data = user.dict()
            user_data['created_at'] = user.created_at.isoformat()
            if user.last_login:
                user_data['last_login'] = user.last_login.isoformat()
            
            self.redis_client.setex(user_key, 86400 * 30, json.dumps(user_data))  # 30 days
            self.redis_client.setex(username_key, 86400 * 30, user.user_id)  # Username lookup
            
            # Initialize user sessions set
            user_sessions_key = self._get_redis_key(self.USER_SESSIONS_PREFIX, user.user_id)
            self.redis_client.delete(user_sessions_key)  # Clear any existing sessions
            
            # Log activity
            self._log_activity(user.user_id, None, "user_created", f"User {user.username} created")
            
            logger.info(f"User created: {user.username} ({user.user_id})")
            return user
            
        except Exception as e:
            logger.error(f"Failed to create user from model: {e}")
            raise
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        try:
            user_key = self._get_redis_key(self.USER_PREFIX, user_id)
            user_data = self.redis_client.get(user_key)
            
            if not user_data:
                return None
            
            data = json.loads(user_data)
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            if data.get('last_login'):
                data['last_login'] = datetime.fromisoformat(data['last_login'])
            
            return User(**data)
            
        except Exception as e:
            logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """Get user by username"""
        try:
            username_key = self._get_redis_key(f"{self.USER_PREFIX}:username", username)
            user_id = self.redis_client.get(username_key)
            
            if not user_id:
                return None
            
            return self.get_user(user_id)
            
        except Exception as e:
            logger.error(f"Failed to get user by username {username}: {e}")
            return None
    
    def update_user(self, user_id: str, updates: UserUpdate) -> Optional[User]:
        """Update user information"""
        try:
            user = self.get_user(user_id)
            if not user:
                return None
            
            # Apply updates
            update_data = updates.dict(exclude_unset=True)
            for field, value in update_data.items():
                setattr(user, field, value)
            
            # Save updated user
            return self.create_user_from_model(user)
            
        except Exception as e:
            logger.error(f"Failed to update user {user_id}: {e}")
            return None
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user and all associated sessions"""
        try:
            user = self.get_user(user_id)
            if not user:
                return False
            
            # Terminate all user sessions
            self.terminate_all_user_sessions(user_id)
            
            # Delete user data
            user_key = self._get_redis_key(self.USER_PREFIX, user_id)
            username_key = self._get_redis_key(f"{self.USER_PREFIX}:username", user.username)
            user_sessions_key = self._get_redis_key(self.USER_SESSIONS_PREFIX, user_id)
            
            self.redis_client.delete(user_key, username_key, user_sessions_key)
            
            # Log activity
            self._log_activity(user_id, None, "user_deleted", f"User {user.username} deleted")
            
            logger.info(f"User deleted: {user.username} ({user_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    def get_total_user_count(self) -> int:
        """Get total number of users"""
        try:
            user_keys = self.redis_client.keys(f"{self.USER_PREFIX}:user_*")
            return len(user_keys)
        except Exception as e:
            logger.error(f"Failed to get user count: {e}")
            return 0
    
    def list_users(self, page: int = 1, page_size: int = 20) -> List[User]:
        """List users with pagination"""
        try:
            user_keys = self.redis_client.keys(f"{self.USER_PREFIX}:user_*")
            start = (page - 1) * page_size
            end = start + page_size
            
            users = []
            for key in user_keys[start:end]:
                user_id = key.split(':')[-1]
                user = self.get_user(user_id)
                if user:
                    users.append(user)
            
            return users
            
        except Exception as e:
            logger.error(f"Failed to list users: {e}")
            return []
    
    # Session Management
    def create_session(self, session_data: SessionCreate) -> Optional[SessionInfo]:
        """Create a new session for user"""
        try:
            user = self.get_user(session_data.user_id)
            if not user:
                raise ValueError(f"User {session_data.user_id} not found")
            
            if user.status != UserStatus.ACTIVE:
                raise ValueError(f"User {session_data.user_id} is not active")
            
            # Check session limit
            active_sessions = self.get_user_active_sessions(session_data.user_id)
            if len(active_sessions) >= user.session_limit:
                raise ValueError(f"User {session_data.user_id} has reached session limit")
            
            session_id = self._generate_session_id()
            now = datetime.now()
            expires_at = now + timedelta(hours=session_data.expires_in_hours)
            
            session = SessionInfo(
                session_id=session_id,
                user_id=session_data.user_id,
                session_type=session_data.session_type,
                status=SessionStatus.ACTIVE,
                created_at=now,
                last_activity=now,
                expires_at=expires_at,
                metadata=session_data.metadata,
                conversation_history=[],
                voice_settings={},
                ui_preferences={}
            )
            
            # Store session
            session_key = self._get_redis_key(self.SESSION_PREFIX, session_id)
            session_data_dict = session.dict()
            session_data_dict['created_at'] = session.created_at.isoformat()
            session_data_dict['last_activity'] = session.last_activity.isoformat()
            session_data_dict['expires_at'] = session.expires_at.isoformat()
            
            ttl = int(session_data.expires_in_hours * 3600)  # Convert to seconds
            self.redis_client.setex(session_key, ttl, json.dumps(session_data_dict))
            
            # Add to user sessions
            user_sessions_key = self._get_redis_key(self.USER_SESSIONS_PREFIX, session_data.user_id)
            self.redis_client.sadd(user_sessions_key, session_id)
            self.redis_client.expire(user_sessions_key, ttl)
            
            # Update in-memory cache
            self.active_sessions[session_id] = session
            if session_data.user_id not in self.user_session_map:
                self.user_session_map[session_data.user_id] = set()
            self.user_session_map[session_data.user_id].add(session_id)
            
            # Update user last login
            user.last_login = now
            self.create_user_from_model(user)
            
            # Log activity
            self._log_activity(session_data.user_id, session_id, "session_created", 
                             f"Session created for {user.username}")
            
            logger.info(f"Session created: {session_id} for user {session_data.user_id}")
            return session
            
        except Exception as e:
            logger.error(f"Failed to create session: {e}")
            return None
    
    def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID"""
        try:
            # Check in-memory cache first
            if session_id in self.active_sessions:
                session = self.active_sessions[session_id]
                if session.expires_at > datetime.now():
                    return session
                else:
                    # Session expired, remove from cache
                    self._remove_session_from_cache(session_id)
            
            # Load from Redis
            session_key = self._get_redis_key(self.SESSION_PREFIX, session_id)
            session_data = self.redis_client.get(session_key)
            
            if not session_data:
                return None
            
            data = json.loads(session_data)
            data['created_at'] = datetime.fromisoformat(data['created_at'])
            data['last_activity'] = datetime.fromisoformat(data['last_activity'])
            data['expires_at'] = datetime.fromisoformat(data['expires_at'])
            
            session = SessionInfo(**data)
            
            # Check if expired
            if session.expires_at <= datetime.now():
                self.terminate_session(session_id)
                return None
            
            # Update cache
            self.active_sessions[session_id] = session
            if session.user_id not in self.user_session_map:
                self.user_session_map[session.user_id] = set()
            self.user_session_map[session.user_id].add(session_id)
            
            return session
            
        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None
    
    def update_session_activity(self, session_id: str) -> bool:
        """Update session last activity timestamp"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            session.last_activity = datetime.now()
            
            # Update in Redis
            session_key = self._get_redis_key(self.SESSION_PREFIX, session_id)
            session_data = session.dict()
            session_data['created_at'] = session.created_at.isoformat()
            session_data['last_activity'] = session.last_activity.isoformat()
            session_data['expires_at'] = session.expires_at.isoformat()
            
            # Calculate remaining TTL
            remaining_time = session.expires_at - datetime.now()
            ttl = max(int(remaining_time.total_seconds()), 60)  # At least 60 seconds
            
            self.redis_client.setex(session_key, ttl, json.dumps(session_data))
            
            # Update cache
            self.active_sessions[session_id] = session
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session activity {session_id}: {e}")
            return False
    
    def terminate_session(self, session_id: str) -> bool:
        """Terminate a session"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            # Update session status
            session.status = SessionStatus.TERMINATED
            
            # Remove from Redis
            session_key = self._get_redis_key(self.SESSION_PREFIX, session_id)
            user_sessions_key = self._get_redis_key(self.USER_SESSIONS_PREFIX, session.user_id)
            
            self.redis_client.delete(session_key)
            self.redis_client.srem(user_sessions_key, session_id)
            
            # Remove from cache
            self._remove_session_from_cache(session_id)
            
            # Log activity
            self._log_activity(session.user_id, session_id, "session_terminated", 
                             "Session terminated")
            
            logger.info(f"Session terminated: {session_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to terminate session {session_id}: {e}")
            return False
    
    def terminate_all_user_sessions(self, user_id: str) -> int:
        """Terminate all sessions for a user"""
        try:
            sessions = self.get_user_active_sessions(user_id)
            terminated_count = 0
            
            for session in sessions:
                if self.terminate_session(session.session_id):
                    terminated_count += 1
            
            logger.info(f"Terminated {terminated_count} sessions for user {user_id}")
            return terminated_count
            
        except Exception as e:
            logger.error(f"Failed to terminate all sessions for user {user_id}: {e}")
            return 0
    
    def get_user_active_sessions(self, user_id: str) -> List[SessionInfo]:
        """Get all active sessions for a user"""
        try:
            user_sessions_key = self._get_redis_key(self.USER_SESSIONS_PREFIX, user_id)
            session_ids = self.redis_client.smembers(user_sessions_key)
            
            active_sessions = []
            for session_id in session_ids:
                session = self.get_session(session_id)
                if session and session.status == SessionStatus.ACTIVE:
                    active_sessions.append(session)
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Failed to get active sessions for user {user_id}: {e}")
            return []
    
    def get_all_active_sessions(self) -> List[SessionInfo]:
        """Get all active sessions in the system"""
        try:
            session_keys = self.redis_client.keys(f"{self.SESSION_PREFIX}:sess_*")
            active_sessions = []
            
            for key in session_keys:
                session_id = key.split(':')[-1]
                session = self.get_session(session_id)
                if session and session.status == SessionStatus.ACTIVE:
                    active_sessions.append(session)
            
            return active_sessions
            
        except Exception as e:
            logger.error(f"Failed to get all active sessions: {e}")
            return []
    
    def _remove_session_from_cache(self, session_id: str):
        """Remove session from in-memory cache"""
        if session_id in self.active_sessions:
            session = self.active_sessions[session_id]
            user_id = session.user_id
            
            del self.active_sessions[session_id]
            
            if user_id in self.user_session_map:
                self.user_session_map[user_id].discard(session_id)
                if not self.user_session_map[user_id]:
                    del self.user_session_map[user_id]
    
    # Activity Logging
    def _log_activity(self, user_id: str, session_id: Optional[str], 
                     activity_type: str, description: str, 
                     metadata: Dict = None):
        """Log user activity"""
        try:
            log_id = f"log_{uuid.uuid4().hex[:8]}"
            activity = ActivityLog(
                log_id=log_id,
                user_id=user_id,
                session_id=session_id,
                activity_type=activity_type,
                description=description,
                timestamp=datetime.now(),
                metadata=metadata or {}
            )
            
            log_key = self._get_redis_key(self.ACTIVITY_PREFIX, log_id)
            log_data = activity.dict()
            log_data['timestamp'] = activity.timestamp.isoformat()
            
            # Store with 7 days TTL
            self.redis_client.setex(log_key, 86400 * 7, json.dumps(log_data))
            
        except Exception as e:
            logger.error(f"Failed to log activity: {e}")
    
    # Session Isolation Helpers
    def update_session_context(self, session_id: str, context_type: str, 
                              context_data: Dict[str, Any]) -> bool:
        """Update session-specific context (conversation, voice settings, etc.)"""
        try:
            session = self.get_session(session_id)
            if not session:
                return False
            
            if context_type == "conversation":
                session.conversation_history = context_data.get("history", [])
                session.context_length = len(session.conversation_history)
            elif context_type == "voice":
                session.voice_settings.update(context_data)
            elif context_type == "ui":
                session.ui_preferences.update(context_data)
            
            # Save updated session
            session_key = self._get_redis_key(self.SESSION_PREFIX, session_id)
            session_data = session.dict()
            session_data['created_at'] = session.created_at.isoformat()
            session_data['last_activity'] = session.last_activity.isoformat()
            session_data['expires_at'] = session.expires_at.isoformat()
            
            remaining_time = session.expires_at - datetime.now()
            ttl = max(int(remaining_time.total_seconds()), 60)
            
            self.redis_client.setex(session_key, ttl, json.dumps(session_data))
            self.active_sessions[session_id] = session
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update session context {session_id}: {e}")
            return False
    
    def get_session_context(self, session_id: str, context_type: str) -> Dict[str, Any]:
        """Get session-specific context"""
        try:
            session = self.get_session(session_id)
            if not session:
                return {}
            
            if context_type == "conversation":
                return {
                    "history": session.conversation_history,
                    "context_length": session.context_length
                }
            elif context_type == "voice":
                return session.voice_settings
            elif context_type == "ui":
                return session.ui_preferences
            else:
                return {}
                
        except Exception as e:
            logger.error(f"Failed to get session context {session_id}: {e}")
            return {}
    
    # Cleanup
    def cleanup_expired_sessions(self) -> int:
        """Clean up expired sessions"""
        try:
            cleaned_count = 0
            current_time = datetime.now()
            
            # Check all sessions in cache
            for session_id, session in list(self.active_sessions.items()):
                if session.expires_at <= current_time:
                    self.terminate_session(session_id)
                    cleaned_count += 1
            
            # Also check Redis keys that might not be in cache
            session_keys = self.redis_client.keys(f"{self.SESSION_PREFIX}:sess_*")
            for key in session_keys:
                session_id = key.split(':')[-1]
                if session_id not in self.active_sessions:
                    session = self.get_session(session_id)
                    if session and session.expires_at <= current_time:
                        self.terminate_session(session_id)
                        cleaned_count += 1
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} expired sessions")
            
            return cleaned_count
            
        except Exception as e:
            logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0