#!/usr/bin/env python3
"""
👥 User Manager for JARVIS Multi-User Support
ระบบจัดการผู้ใช้สำหรับ JARVIS แบบ Multi-User
"""

import logging
import time
import uuid
import hashlib
from typing import Dict, Any, Optional, List, Set, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from datetime import datetime, timedelta


class UserRole(Enum):
    """บทบาทของผู้ใช้"""
    ADMIN = "admin"           # ผู้ดูแลระบบ
    USER = "user"            # ผู้ใช้ทั่วไป
    GUEST = "guest"          # แขก
    FAMILY = "family"        # สมาชิกครอบครัว
    CHILD = "child"          # เด็ก (จำกัดการเข้าถึง)


class UserStatus(Enum):
    """สถานะของผู้ใช้"""
    ACTIVE = "active"        # กำลังใช้งาน
    IDLE = "idle"           # ไม่ได้ใช้งาน
    OFFLINE = "offline"      # ออฟไลน์
    BLOCKED = "blocked"      # ถูกบล็อก


@dataclass
class UserPreferences:
    """การตั้งค่าของผู้ใช้"""
    language: str = "th"                    # ภาษาหลัก
    voice_language: str = "th"              # ภาษาสำหรับการพูด
    response_style: str = "friendly"        # สไตล์การตอบ
    max_response_length: int = 500          # ความยาวการตอบสูงสุด
    enable_notifications: bool = True       # การแจ้งเตือน
    privacy_level: str = "medium"          # ระดับความเป็นส่วนตัว
    preferred_topics: List[str] = field(default_factory=list)
    blocked_topics: List[str] = field(default_factory=list)
    custom_commands: Dict[str, str] = field(default_factory=dict)


@dataclass
class UserSession:
    """เซสชันของผู้ใช้"""
    session_id: str
    user_id: str
    start_time: float
    last_activity: float
    ip_address: str = "unknown"
    device_info: str = "unknown"
    conversation_count: int = 0
    active: bool = True


@dataclass
class User:
    """ข้อมูลผู้ใช้"""
    user_id: str
    username: str
    display_name: str
    role: UserRole
    status: UserStatus
    created_at: float
    last_seen: float
    preferences: UserPreferences = field(default_factory=UserPreferences)
    voice_profile: Optional[str] = None     # ลายเสียงสำหรับ voice recognition
    authentication_data: Dict[str, Any] = field(default_factory=dict)
    conversation_history_id: Optional[str] = None
    active_sessions: List[str] = field(default_factory=list)
    total_interactions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        data = asdict(self)
        data['role'] = self.role.value
        data['status'] = self.status.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        """สร้างจาก dictionary"""
        data = data.copy()
        data['role'] = UserRole(data['role'])
        data['status'] = UserStatus(data['status'])
        if 'preferences' in data:
            data['preferences'] = UserPreferences(**data['preferences'])
        return cls(**data)


class UserManager:
    """ระบบจัดการผู้ใช้สำหรับ JARVIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # User storage
        self.users: Dict[str, User] = {}
        self.sessions: Dict[str, UserSession] = {}
        self.username_to_user_id: Dict[str, str] = {}
        
        # Settings
        self.max_users = self.config.get('max_users', 100)
        self.session_timeout = self.config.get('session_timeout', 3600)  # 1 hour
        self.enable_voice_recognition = self.config.get('enable_voice_recognition', True)
        self.require_authentication = self.config.get('require_authentication', False)
        self.auto_create_users = self.config.get('auto_create_users', True)
        
        # Voice recognition
        self.voice_profiles: Dict[str, str] = {}  # voice_signature -> user_id
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks
        self.user_callbacks: Dict[str, List[Callable]] = {
            'user_login': [],
            'user_logout': [],
            'user_created': [],
            'user_updated': [],
            'session_started': [],
            'session_ended': []
        }
        
        # Cleanup thread
        self.cleanup_active = False
        self.cleanup_thread = None
        
        # Create default admin user if none exists
        self._create_default_admin()
        
        self.logger.info("👥 User Manager initialized")
    
    def _create_default_admin(self):
        """สร้างผู้ดูแลระบบเริ่มต้น"""
        if not self.users:
            admin_user = self.create_user(
                username="admin",
                display_name="System Administrator",
                role=UserRole.ADMIN,
                auto_login=True
            )
            self.logger.info(f"✅ Default admin user created: {admin_user.user_id}")
    
    def create_user(self, username: str, display_name: str, 
                   role: UserRole = UserRole.USER, 
                   preferences: UserPreferences = None,
                   auto_login: bool = False) -> Optional[User]:
        """สร้างผู้ใช้ใหม่"""
        with self.lock:
            # Check if username already exists
            if username in self.username_to_user_id:
                self.logger.warning(f"⚠️ Username already exists: {username}")
                return None
            
            # Check user limit
            if len(self.users) >= self.max_users:
                self.logger.warning(f"⚠️ Maximum users reached: {self.max_users}")
                return None
            
            # Generate user ID
            user_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Create user
            user = User(
                user_id=user_id,
                username=username,
                display_name=display_name,
                role=role,
                status=UserStatus.OFFLINE,
                created_at=current_time,
                last_seen=current_time,
                preferences=preferences or UserPreferences()
            )
            
            # Store user
            self.users[user_id] = user
            self.username_to_user_id[username] = user_id
            
            # Auto-login if requested
            if auto_login:
                session = self.create_session(user_id)
                if session:
                    user.status = UserStatus.ACTIVE
            
            # Trigger callbacks
            self._trigger_callbacks('user_created', user)
            
            self.logger.info(f"✅ User created: {username} ({role.value})")
            return user
    
    def authenticate_user(self, username: str, password: str = None, 
                         voice_signature: str = None) -> Optional[User]:
        """ยืนยันตัวตนผู้ใช้"""
        with self.lock:
            # Get user by username
            if username not in self.username_to_user_id:
                if self.auto_create_users:
                    # Auto-create user if enabled
                    return self.create_user(username, username, UserRole.USER, auto_login=True)
                else:
                    self.logger.warning(f"⚠️ User not found: {username}")
                    return None
            
            user_id = self.username_to_user_id[username]
            user = self.users[user_id]
            
            # Check if user is blocked
            if user.status == UserStatus.BLOCKED:
                self.logger.warning(f"⚠️ User blocked: {username}")
                return None
            
            # Authentication logic
            if self.require_authentication:
                # Password authentication
                if password:
                    stored_hash = user.authentication_data.get('password_hash')
                    if stored_hash:
                        password_hash = self._hash_password(password)
                        if password_hash != stored_hash:
                            self.logger.warning(f"⚠️ Invalid password for: {username}")
                            return None
                
                # Voice authentication
                if self.enable_voice_recognition and voice_signature:
                    if not self._verify_voice_signature(user_id, voice_signature):
                        self.logger.warning(f"⚠️ Voice authentication failed for: {username}")
                        return None
            
            return user
    
    def login_user(self, username: str, password: str = None, 
                  voice_signature: str = None, 
                  device_info: str = "unknown",
                  ip_address: str = "unknown") -> Optional[UserSession]:
        """ล็อกอินผู้ใช้"""
        user = self.authenticate_user(username, password, voice_signature)
        if not user:
            return None
        
        # Create session
        session = self.create_session(
            user.user_id, 
            device_info=device_info,
            ip_address=ip_address
        )
        
        if session:
            # Update user status
            user.status = UserStatus.ACTIVE
            user.last_seen = time.time()
            
            # Trigger callbacks
            self._trigger_callbacks('user_login', user, session)
            
            self.logger.info(f"✅ User logged in: {username}")
        
        return session
    
    def logout_user(self, session_id: str) -> bool:
        """ล็อกเอาต์ผู้ใช้"""
        with self.lock:
            if session_id not in self.sessions:
                return False
            
            session = self.sessions[session_id]
            user = self.users.get(session.user_id)
            
            if user:
                # Remove session from user
                if session_id in user.active_sessions:
                    user.active_sessions.remove(session_id)
                
                # Update status if no more active sessions
                if not user.active_sessions:
                    user.status = UserStatus.OFFLINE
                
                # Trigger callbacks
                self._trigger_callbacks('user_logout', user, session)
            
            # Remove session
            del self.sessions[session_id]
            
            self.logger.info(f"✅ User logged out: {session.user_id}")
            return True
    
    def create_session(self, user_id: str, device_info: str = "unknown",
                      ip_address: str = "unknown") -> Optional[UserSession]:
        """สร้างเซสชันใหม่"""
        with self.lock:
            if user_id not in self.users:
                return None
            
            # Generate session ID
            session_id = str(uuid.uuid4())
            current_time = time.time()
            
            # Create session
            session = UserSession(
                session_id=session_id,
                user_id=user_id,
                start_time=current_time,
                last_activity=current_time,
                ip_address=ip_address,
                device_info=device_info
            )
            
            # Store session
            self.sessions[session_id] = session
            
            # Add to user's active sessions
            user = self.users[user_id]
            user.active_sessions.append(session_id)
            
            # Trigger callbacks
            self._trigger_callbacks('session_started', user, session)
            
            self.logger.info(f"✅ Session created: {session_id} for user {user_id}")
            return session
    
    def get_user_by_session(self, session_id: str) -> Optional[User]:
        """ดึงผู้ใช้จาก session ID"""
        session = self.sessions.get(session_id)
        if session and session.active:
            return self.users.get(session.user_id)
        return None
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """ดึงผู้ใช้จาก username"""
        user_id = self.username_to_user_id.get(username)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def get_user_by_voice(self, voice_signature: str) -> Optional[User]:
        """ดึงผู้ใช้จากลายเสียง"""
        if not self.enable_voice_recognition:
            return None
        
        user_id = self.voice_profiles.get(voice_signature)
        if user_id:
            return self.users.get(user_id)
        return None
    
    def update_user_activity(self, session_id: str):
        """อัปเดตกิจกรรมของผู้ใช้"""
        with self.lock:
            session = self.sessions.get(session_id)
            if session:
                session.last_activity = time.time()
                session.conversation_count += 1
                
                # Update user
                user = self.users.get(session.user_id)
                if user:
                    user.last_seen = time.time()
                    user.total_interactions += 1
                    if user.status == UserStatus.IDLE:
                        user.status = UserStatus.ACTIVE
    
    def update_user_preferences(self, user_id: str, preferences: Dict[str, Any]) -> bool:
        """อัปเดตการตั้งค่าผู้ใช้"""
        with self.lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            # Update preferences
            for key, value in preferences.items():
                if hasattr(user.preferences, key):
                    setattr(user.preferences, key, value)
            
            # Trigger callbacks
            self._trigger_callbacks('user_updated', user)
            
            self.logger.info(f"✅ Preferences updated for user: {user_id}")
            return True
    
    def register_voice_profile(self, user_id: str, voice_signature: str) -> bool:
        """ลงทะเบียนลายเสียงของผู้ใช้"""
        with self.lock:
            if user_id not in self.users:
                return False
            
            # Remove existing voice profile for this user
            for sig, uid in list(self.voice_profiles.items()):
                if uid == user_id:
                    del self.voice_profiles[sig]
            
            # Add new voice profile
            self.voice_profiles[voice_signature] = user_id
            self.users[user_id].voice_profile = voice_signature
            
            self.logger.info(f"✅ Voice profile registered for user: {user_id}")
            return True
    
    def _verify_voice_signature(self, user_id: str, voice_signature: str) -> bool:
        """ตรวจสอบลายเสียง"""
        user = self.users.get(user_id)
        if not user or not user.voice_profile:
            return True  # Allow if no voice profile set
        
        # Simple voice signature matching (in real implementation, use ML)
        return user.voice_profile == voice_signature
    
    def _hash_password(self, password: str) -> str:
        """เข้ารหัสรหัสผ่าน"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def set_user_password(self, user_id: str, password: str) -> bool:
        """ตั้งรหัสผ่านผู้ใช้"""
        with self.lock:
            user = self.users.get(user_id)
            if not user:
                return False
            
            password_hash = self._hash_password(password)
            user.authentication_data['password_hash'] = password_hash
            
            self.logger.info(f"✅ Password set for user: {user_id}")
            return True
    
    def get_active_users(self) -> List[User]:
        """ดึงรายชื่อผู้ใช้ที่กำลังใช้งาน"""
        return [user for user in self.users.values() 
                if user.status == UserStatus.ACTIVE]
    
    def get_user_sessions(self, user_id: str) -> List[UserSession]:
        """ดึงเซสชันของผู้ใช้"""
        user = self.users.get(user_id)
        if not user:
            return []
        
        return [self.sessions[sid] for sid in user.active_sessions 
                if sid in self.sessions]
    
    def block_user(self, user_id: str, admin_user_id: str) -> bool:
        """บล็อกผู้ใช้"""
        with self.lock:
            user = self.users.get(user_id)
            admin_user = self.users.get(admin_user_id)
            
            if not user or not admin_user:
                return False
            
            if admin_user.role != UserRole.ADMIN:
                self.logger.warning(f"⚠️ Non-admin user tried to block: {admin_user_id}")
                return False
            
            user.status = UserStatus.BLOCKED
            
            # End all sessions
            for session_id in user.active_sessions[:]:
                self.logout_user(session_id)
            
            self.logger.info(f"🚫 User blocked: {user_id} by {admin_user_id}")
            return True
    
    def unblock_user(self, user_id: str, admin_user_id: str) -> bool:
        """ปลดบล็อกผู้ใช้"""
        with self.lock:
            user = self.users.get(user_id)
            admin_user = self.users.get(admin_user_id)
            
            if not user or not admin_user:
                return False
            
            if admin_user.role != UserRole.ADMIN:
                self.logger.warning(f"⚠️ Non-admin user tried to unblock: {admin_user_id}")
                return False
            
            user.status = UserStatus.OFFLINE
            
            self.logger.info(f"✅ User unblocked: {user_id} by {admin_user_id}")
            return True
    
    def start_cleanup_service(self):
        """เริ่มบริการทำความสะอาดอัตโนมัติ"""
        if self.cleanup_active:
            return
        
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        self.logger.info("🧹 User cleanup service started")
    
    def stop_cleanup_service(self):
        """หยุดบริการทำความสะอาด"""
        self.cleanup_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        self.logger.info("🛑 User cleanup service stopped")
    
    def _cleanup_loop(self):
        """ลูปทำความสะอาดอัตโนมัติ"""
        while self.cleanup_active:
            try:
                self._cleanup_expired_sessions()
                self._update_idle_users()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"❌ Cleanup error: {e}")
                time.sleep(60)
    
    def _cleanup_expired_sessions(self):
        """ทำความสะอาดเซสชันที่หมดอายุ"""
        current_time = time.time()
        expired_sessions = []
        
        with self.lock:
            for session_id, session in self.sessions.items():
                if current_time - session.last_activity > self.session_timeout:
                    expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            self.logout_user(session_id)
            self.logger.info(f"🧹 Expired session cleaned: {session_id}")
    
    def _update_idle_users(self):
        """อัปเดตสถานะผู้ใช้ที่ไม่ได้ใช้งาน"""
        current_time = time.time()
        idle_threshold = 300  # 5 minutes
        
        with self.lock:
            for user in self.users.values():
                if (user.status == UserStatus.ACTIVE and 
                    current_time - user.last_seen > idle_threshold):
                    user.status = UserStatus.IDLE
    
    def register_callback(self, event: str, callback: Callable):
        """ลงทะเบียน callback"""
        if event in self.user_callbacks:
            self.user_callbacks[event].append(callback)
    
    def _trigger_callbacks(self, event: str, *args):
        """เรียก callbacks"""
        for callback in self.user_callbacks.get(event, []):
            try:
                callback(*args)
            except Exception as e:
                self.logger.error(f"❌ Callback error for {event}: {e}")
    
    def get_user_stats(self) -> Dict[str, Any]:
        """ดึงสถิติผู้ใช้"""
        with self.lock:
            stats = {
                'total_users': len(self.users),
                'active_users': len([u for u in self.users.values() if u.status == UserStatus.ACTIVE]),
                'idle_users': len([u for u in self.users.values() if u.status == UserStatus.IDLE]),
                'offline_users': len([u for u in self.users.values() if u.status == UserStatus.OFFLINE]),
                'blocked_users': len([u for u in self.users.values() if u.status == UserStatus.BLOCKED]),
                'active_sessions': len(self.sessions),
                'role_distribution': {},
                'total_interactions': sum(u.total_interactions for u in self.users.values())
            }
            
            # Role distribution
            for role in UserRole:
                stats['role_distribution'][role.value] = len([
                    u for u in self.users.values() if u.role == role
                ])
            
            return stats
    
    def export_users(self) -> List[Dict[str, Any]]:
        """ส่งออกข้อมูลผู้ใช้"""
        with self.lock:
            return [user.to_dict() for user in self.users.values()]
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.stop_cleanup_service()
        
        with self.lock:
            # End all sessions
            for session_id in list(self.sessions.keys()):
                self.logout_user(session_id)
            
            # Clear callbacks
            for callback_list in self.user_callbacks.values():
                callback_list.clear()
        
        self.logger.info("🧹 User Manager cleaned up")


def test_user_manager():
    """ทดสอบ User Manager"""
    print("🧪 Testing User Manager")
    print("━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create user manager
    config = {
        'max_users': 50,
        'session_timeout': 10,  # Short timeout for testing
        'auto_create_users': True,
        'enable_voice_recognition': True
    }
    
    manager = UserManager(config)
    
    # Test user creation
    print("👤 Testing user creation...")
    user1 = manager.create_user("alice", "Alice Smith", UserRole.USER)
    user2 = manager.create_user("bob", "Bob Johnson", UserRole.FAMILY)
    
    if user1 and user2:
        print(f"   ✅ Users created: {user1.username}, {user2.username}")
    else:
        print("   ❌ User creation failed")
    
    # Test authentication and login
    print("\n🔐 Testing authentication...")
    session1 = manager.login_user("alice", device_info="iPhone 12")
    session2 = manager.login_user("bob", device_info="Android")
    
    if session1 and session2:
        print(f"   ✅ Users logged in: {len(manager.get_active_users())} active")
    else:
        print("   ❌ Login failed")
    
    # Test user preferences
    print("\n⚙️ Testing preferences...")
    preferences = {
        'language': 'en',
        'response_style': 'formal',
        'preferred_topics': ['technology', 'science']
    }
    
    if manager.update_user_preferences(user1.user_id, preferences):
        updated_user = manager.users[user1.user_id]
        print(f"   ✅ Preferences updated: {updated_user.preferences.language}")
    else:
        print("   ❌ Preferences update failed")
    
    # Test voice profile
    print("\n🎤 Testing voice profiles...")
    voice_sig = "voice_signature_alice_123"
    if manager.register_voice_profile(user1.user_id, voice_sig):
        voice_user = manager.get_user_by_voice(voice_sig)
        if voice_user:
            print(f"   ✅ Voice profile: {voice_user.username}")
        else:
            print("   ❌ Voice recognition failed")
    else:
        print("   ❌ Voice profile registration failed")
    
    # Test activity tracking
    print("\n📊 Testing activity tracking...")
    for i in range(5):
        manager.update_user_activity(session1.session_id)
    
    stats = manager.get_user_stats()
    print(f"   📈 Total interactions: {stats['total_interactions']}")
    print(f"   👥 Active users: {stats['active_users']}")
    print(f"   🔗 Active sessions: {stats['active_sessions']}")
    
    # Test admin functions
    print("\n👑 Testing admin functions...")
    admin_user = manager.get_user_by_username("admin")
    if admin_user and manager.block_user(user2.user_id, admin_user.user_id):
        print("   ✅ User blocked by admin")
        
        if manager.unblock_user(user2.user_id, admin_user.user_id):
            print("   ✅ User unblocked by admin")
    
    # Test cleanup
    print("\n🧹 Testing cleanup...")
    manager.start_cleanup_service()
    time.sleep(2)  # Let cleanup run
    manager.stop_cleanup_service()
    
    # Final stats
    final_stats = manager.get_user_stats()
    print(f"\n📊 Final Statistics:")
    print(f"   Total users: {final_stats['total_users']}")
    print(f"   Active users: {final_stats['active_users']}")
    print(f"   Total interactions: {final_stats['total_interactions']}")
    
    # Cleanup
    manager.cleanup()
    
    return manager


if __name__ == "__main__":
    test_user_manager()