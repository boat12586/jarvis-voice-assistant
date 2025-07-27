#!/usr/bin/env python3
"""
💬 Conversation Context Manager for JARVIS Multi-User Support
ระบบจัดการบริบทการสนทนาสำหรับ JARVIS แบบ Multi-User
"""

import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import threading
from datetime import datetime, timedelta


class MessageType(Enum):
    """ประเภทของข้อความ"""
    USER_INPUT = "user_input"           # ข้อความจากผู้ใช้
    AI_RESPONSE = "ai_response"         # การตอบสนองจาก AI
    SYSTEM = "system"                   # ข้อความระบบ
    COMMAND = "command"                 # คำสั่ง
    NOTIFICATION = "notification"       # การแจ้งเตือน


class ConversationMode(Enum):
    """โหมดการสนทนา"""
    SINGLE_USER = "single_user"         # ผู้ใช้เดียว
    MULTI_USER = "multi_user"          # หลายผู้ใช้
    GROUP_CHAT = "group_chat"          # แชทกลุ่ม
    PRESENTATION = "presentation"       # การนำเสนอ


@dataclass
class ConversationMessage:
    """ข้อความในการสนทนา"""
    message_id: str
    user_id: str
    username: str
    content: str
    message_type: MessageType
    timestamp: float
    session_id: str
    language: str = "th"
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    context_reference: Optional[str] = None  # อ้างอิงถึงข้อความก่อนหน้า
    
    def to_dict(self) -> Dict[str, Any]:
        """แปลงเป็น dictionary"""
        data = asdict(self)
        data['message_type'] = self.message_type.value
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConversationMessage':
        """สร้างจาก dictionary"""
        data = data.copy()
        data['message_type'] = MessageType(data['message_type'])
        return cls(**data)


@dataclass
class ConversationState:
    """สถานะของการสนทนา"""
    current_topic: Optional[str] = None
    active_users: List[str] = field(default_factory=list)
    waiting_for_input: List[str] = field(default_factory=list)
    last_ai_response: Optional[str] = None
    conversation_mood: str = "neutral"      # neutral, happy, serious, urgent
    context_level: str = "standard"         # minimal, standard, detailed
    language_preference: str = "th"
    ai_personality: str = "friendly"        # friendly, formal, technical, casual


class ConversationContext:
    """ระบบจัดการบริบทการสนทนาสำหรับ JARVIS"""
    
    def __init__(self, context_id: str, user_manager, config: Dict[str, Any] = None):
        self.context_id = context_id
        self.user_manager = user_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Conversation settings
        self.mode = ConversationMode(self.config.get('mode', 'single_user'))
        self.max_history_length = self.config.get('max_history_length', 1000)
        self.context_window_size = self.config.get('context_window_size', 50)
        self.auto_cleanup_interval = self.config.get('auto_cleanup_interval', 3600)  # 1 hour
        
        # Message storage
        self.messages: List[ConversationMessage] = []
        self.message_index: Dict[str, ConversationMessage] = {}
        
        # Conversation state
        self.state = ConversationState()
        self.created_at = time.time()
        self.last_activity = time.time()
        self.is_active = True
        
        # User tracking
        self.participant_users: Dict[str, Dict[str, Any]] = {}
        self.active_sessions: Dict[str, str] = {}  # session_id -> user_id
        
        # Context management
        self.conversation_topics: List[str] = []
        self.important_messages: List[str] = []  # message IDs
        self.pending_responses: Dict[str, Dict[str, Any]] = {}
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Callbacks
        self.message_callbacks: List[Callable] = []
        self.context_change_callbacks: List[Callable] = []
        
        self.logger.info(f"💬 Conversation context created: {context_id} ({self.mode.value})")
    
    def add_participant(self, user_id: str, session_id: str) -> bool:
        """เพิ่มผู้เข้าร่วมการสนทนา"""
        with self.lock:
            user = self.user_manager.get_user_by_session(session_id)
            if not user:
                return False
            
            # Add user to participants
            self.participant_users[user_id] = {
                'user_id': user_id,
                'username': user.username,
                'display_name': user.display_name,
                'joined_at': time.time(),
                'last_message': None,
                'message_count': 0,
                'preferences': asdict(user.preferences)
            }
            
            # Track active session
            self.active_sessions[session_id] = user_id
            
            # Update conversation state
            if user_id not in self.state.active_users:
                self.state.active_users.append(user_id)
            
            # Add system message
            self._add_system_message(
                f"👤 {user.display_name} เข้าร่วมการสนทนา",
                metadata={'event': 'user_joined', 'user_id': user_id}
            )
            
            self.logger.info(f"👤 User joined conversation: {user.username}")
            return True
    
    def remove_participant(self, user_id: str) -> bool:
        """ลบผู้เข้าร่วมการสนทนา"""
        with self.lock:
            if user_id not in self.participant_users:
                return False
            
            user_info = self.participant_users[user_id]
            
            # Remove from active users
            if user_id in self.state.active_users:
                self.state.active_users.remove(user_id)
            
            # Remove from waiting list
            if user_id in self.state.waiting_for_input:
                self.state.waiting_for_input.remove(user_id)
            
            # Remove active sessions
            sessions_to_remove = [
                sid for sid, uid in self.active_sessions.items() if uid == user_id
            ]
            for session_id in sessions_to_remove:
                del self.active_sessions[session_id]
            
            # Add system message
            self._add_system_message(
                f"👋 {user_info['display_name']} ออกจากการสนทนา",
                metadata={'event': 'user_left', 'user_id': user_id}
            )
            
            # Remove from participants
            del self.participant_users[user_id]
            
            self.logger.info(f"👋 User left conversation: {user_info['username']}")
            return True
    
    def add_message(self, user_id: str, content: str, message_type: MessageType = MessageType.USER_INPUT,
                   session_id: str = None, language: str = None, 
                   confidence: float = 1.0, metadata: Dict[str, Any] = None) -> ConversationMessage:
        """เพิ่มข้อความในการสนทนา"""
        with self.lock:
            # Get user info
            user_info = self.participant_users.get(user_id)
            if not user_info:
                raise ValueError(f"User {user_id} is not a participant in this conversation")
            
            # Create message
            message = ConversationMessage(
                message_id=str(uuid.uuid4()),
                user_id=user_id,
                username=user_info['username'],
                content=content,
                message_type=message_type,
                timestamp=time.time(),
                session_id=session_id or "",
                language=language or user_info['preferences'].get('language', 'th'),
                confidence=confidence,
                metadata=metadata or {}
            )
            
            # Add to storage
            self.messages.append(message)
            self.message_index[message.message_id] = message
            
            # Update user stats
            user_info['last_message'] = message.timestamp
            user_info['message_count'] += 1
            
            # Update conversation state
            self.last_activity = time.time()
            
            # Cleanup old messages if needed
            if len(self.messages) > self.max_history_length:
                self._cleanup_old_messages()
            
            # Detect topic changes
            if message_type == MessageType.USER_INPUT:
                self._analyze_topic_change(content)
            
            # Trigger callbacks
            self._trigger_message_callbacks(message)
            
            self.logger.debug(f"📝 Message added: {user_info['username']}: {content[:50]}...")
            return message
    
    def add_ai_response(self, content: str, responding_to_user: str = None, 
                       confidence: float = 1.0, metadata: Dict[str, Any] = None) -> ConversationMessage:
        """เพิ่มการตอบสนองจาก AI"""
        # Use system user for AI responses
        ai_message = ConversationMessage(
            message_id=str(uuid.uuid4()),
            user_id="jarvis_ai",
            username="JARVIS",
            content=content,
            message_type=MessageType.AI_RESPONSE,
            timestamp=time.time(),
            session_id="",
            language=self.state.language_preference,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        if responding_to_user:
            ai_message.metadata['responding_to'] = responding_to_user
        
        with self.lock:
            self.messages.append(ai_message)
            self.message_index[ai_message.message_id] = ai_message
            self.state.last_ai_response = ai_message.message_id
            self.last_activity = time.time()
        
        self._trigger_message_callbacks(ai_message)
        return ai_message
    
    def _add_system_message(self, content: str, metadata: Dict[str, Any] = None):
        """เพิ่มข้อความระบบ"""
        system_message = ConversationMessage(
            message_id=str(uuid.uuid4()),
            user_id="system",
            username="System",
            content=content,
            message_type=MessageType.SYSTEM,
            timestamp=time.time(),
            session_id="",
            metadata=metadata or {}
        )
        
        self.messages.append(system_message)
        self.message_index[system_message.message_id] = system_message
    
    def get_context_window(self, size: Optional[int] = None) -> List[ConversationMessage]:
        """ดึงหน้าต่างบริบทล่าสุด"""
        window_size = size or self.context_window_size
        with self.lock:
            return self.messages[-window_size:] if len(self.messages) > window_size else self.messages.copy()
    
    def get_user_messages(self, user_id: str, limit: int = 50) -> List[ConversationMessage]:
        """ดึงข้อความของผู้ใช้คนหนึ่ง"""
        with self.lock:
            user_messages = [msg for msg in self.messages if msg.user_id == user_id]
            return user_messages[-limit:] if len(user_messages) > limit else user_messages
    
    def get_message_by_id(self, message_id: str) -> Optional[ConversationMessage]:
        """ดึงข้อความจาก ID"""
        return self.message_index.get(message_id)
    
    def search_messages(self, query: str, message_type: MessageType = None, 
                       user_id: str = None, limit: int = 20) -> List[ConversationMessage]:
        """ค้นหาข้อความ"""
        results = []
        query_lower = query.lower()
        
        with self.lock:
            for message in reversed(self.messages):
                # Apply filters
                if message_type and message.message_type != message_type:
                    continue
                if user_id and message.user_id != user_id:
                    continue
                
                # Search in content
                if query_lower in message.content.lower():
                    results.append(message)
                    if len(results) >= limit:
                        break
        
        return results
    
    def update_conversation_state(self, updates: Dict[str, Any]):
        """อัปเดตสถานะการสนทนา"""
        with self.lock:
            for key, value in updates.items():
                if hasattr(self.state, key):
                    setattr(self.state, key, value)
            
            self._trigger_context_change_callbacks()
    
    def set_waiting_for_input(self, user_ids: List[str]):
        """ตั้งค่าการรอ input จากผู้ใช้"""
        with self.lock:
            self.state.waiting_for_input = [uid for uid in user_ids if uid in self.participant_users]
    
    def mark_important_message(self, message_id: str) -> bool:
        """ทำเครื่องหมายข้อความสำคัญ"""
        if message_id in self.message_index:
            with self.lock:
                if message_id not in self.important_messages:
                    self.important_messages.append(message_id)
            return True
        return False
    
    def get_important_messages(self) -> List[ConversationMessage]:
        """ดึงข้อความสำคัญ"""
        with self.lock:
            return [self.message_index[mid] for mid in self.important_messages 
                   if mid in self.message_index]
    
    def _analyze_topic_change(self, content: str):
        """วิเคราะห์การเปลี่ยนหัวข้อ"""
        # Simple topic detection based on keywords
        topic_keywords = {
            'weather': ['อากาศ', 'ฝน', 'แดด', 'หนาว', 'ร้อน', 'weather', 'rain', 'sunny'],
            'time': ['เวลา', 'วันนี้', 'พรุ่งนี้', 'เมื่อไหร่', 'time', 'today', 'tomorrow'],
            'food': ['อาหาร', 'กิน', 'หิว', 'ร้านอาหาร', 'food', 'eat', 'hungry', 'restaurant'],
            'technology': ['คอมพิวเตอร์', 'โทรศัพท์', 'อินเทอร์เน็ต', 'computer', 'phone', 'internet'],
            'music': ['เพลง', 'ดนตรี', 'เล่น', 'ฟัง', 'music', 'song', 'play', 'listen']
        }
        
        content_lower = content.lower()
        detected_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_topics.append(topic)
        
        if detected_topics:
            new_topic = detected_topics[0]  # Use first detected topic
            if new_topic != self.state.current_topic:
                old_topic = self.state.current_topic
                self.state.current_topic = new_topic
                
                # Add to topic history
                if new_topic not in self.conversation_topics:
                    self.conversation_topics.append(new_topic)
                
                self.logger.info(f"🔄 Topic changed: {old_topic} → {new_topic}")
    
    def _cleanup_old_messages(self):
        """ทำความสะอาดข้อความเก่า"""
        # Keep important messages and recent messages
        messages_to_keep = []
        important_ids = set(self.important_messages)
        
        # Keep last N messages
        recent_messages = self.messages[-self.context_window_size:]
        messages_to_keep.extend(recent_messages)
        
        # Keep important messages
        for message in self.messages:
            if message.message_id in important_ids:
                messages_to_keep.append(message)
        
        # Remove duplicates and sort by timestamp
        unique_messages = list({msg.message_id: msg for msg in messages_to_keep}.values())
        unique_messages.sort(key=lambda x: x.timestamp)
        
        # Update storage
        self.messages = unique_messages
        self.message_index = {msg.message_id: msg for msg in unique_messages}
        
        self.logger.info(f"🧹 Cleaned up messages: {len(unique_messages)} remaining")
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """ดึงสรุปการสนทนา"""
        with self.lock:
            total_messages = len(self.messages)
            user_message_counts = {}
            
            for message in self.messages:
                if message.user_id not in user_message_counts:
                    user_message_counts[message.user_id] = 0
                user_message_counts[message.user_id] += 1
            
            return {
                'context_id': self.context_id,
                'mode': self.mode.value,
                'created_at': self.created_at,
                'last_activity': self.last_activity,
                'duration_minutes': (time.time() - self.created_at) / 60,
                'total_messages': total_messages,
                'participants': len(self.participant_users),
                'active_users': len(self.state.active_users),
                'current_topic': self.state.current_topic,
                'conversation_topics': self.conversation_topics,
                'user_message_counts': user_message_counts,
                'important_messages_count': len(self.important_messages),
                'state': asdict(self.state),
                'is_active': self.is_active
            }
    
    def export_conversation(self, format: str = 'json') -> str:
        """ส่งออกการสนทนา"""
        with self.lock:
            if format == 'json':
                import json
                export_data = {
                    'summary': self.get_conversation_summary(),
                    'messages': [msg.to_dict() for msg in self.messages],
                    'participants': self.participant_users
                }
                return json.dumps(export_data, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
    
    def register_message_callback(self, callback: Callable[[ConversationMessage], None]):
        """ลงทะเบียน callback สำหรับข้อความใหม่"""
        self.message_callbacks.append(callback)
    
    def register_context_change_callback(self, callback: Callable[[], None]):
        """ลงทะเบียน callback สำหรับการเปลี่ยนบริบท"""
        self.context_change_callbacks.append(callback)
    
    def _trigger_message_callbacks(self, message: ConversationMessage):
        """เรียก callbacks สำหรับข้อความใหม่"""
        for callback in self.message_callbacks:
            try:
                callback(message)
            except Exception as e:
                self.logger.error(f"❌ Message callback error: {e}")
    
    def _trigger_context_change_callbacks(self):
        """เรียก callbacks สำหรับการเปลี่ยนบริบท"""
        for callback in self.context_change_callbacks:
            try:
                callback()
            except Exception as e:
                self.logger.error(f"❌ Context change callback error: {e}")
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        with self.lock:
            self.is_active = False
            self.message_callbacks.clear()
            self.context_change_callbacks.clear()
        
        self.logger.info(f"🧹 Conversation context cleaned up: {self.context_id}")


def test_conversation_context():
    """ทดสอบ Conversation Context"""
    print("🧪 Testing Conversation Context")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Mock user manager for testing
    class MockUserManager:
        def __init__(self):
            self.users = {
                'user1': type('User', (), {
                    'user_id': 'user1',
                    'username': 'alice',
                    'display_name': 'Alice Smith',
                    'preferences': type('Preferences', (), {
                        'language': 'th',
                        'response_style': 'friendly'
                    })()
                })(),
                'user2': type('User', (), {
                    'user_id': 'user2', 
                    'username': 'bob',
                    'display_name': 'Bob Johnson',
                    'preferences': type('Preferences', (), {
                        'language': 'en',
                        'response_style': 'formal'
                    })()
                })()
            }
            self.sessions = {'session1': 'user1', 'session2': 'user2'}
        
        def get_user_by_session(self, session_id):
            user_id = self.sessions.get(session_id)
            return self.users.get(user_id)
    
    mock_user_manager = MockUserManager()
    
    # Create conversation context
    context = ConversationContext(
        context_id="test_conversation",
        user_manager=mock_user_manager,
        config={
            'mode': 'multi_user',
            'max_history_length': 100,
            'context_window_size': 20
        }
    )
    
    # Test adding participants
    print("👥 Testing participant management...")
    success1 = context.add_participant('user1', 'session1')
    success2 = context.add_participant('user2', 'session2')
    
    if success1 and success2:
        print(f"   ✅ Participants added: {len(context.participant_users)}")
    else:
        print("   ❌ Failed to add participants")
    
    # Test message adding
    print("\n💬 Testing message management...")
    
    # User messages
    msg1 = context.add_message('user1', "สวัสดีครับ JARVIS", MessageType.USER_INPUT, 'session1')
    msg2 = context.add_message('user2', "Hello JARVIS, how are you?", MessageType.USER_INPUT, 'session2')
    
    # AI response
    ai_msg = context.add_ai_response("สวัสดีครับ Alice และ Bob! ผมสบายดีครับ", confidence=0.95)
    
    # User follows up
    msg3 = context.add_message('user1', "วันนี้อากาศเป็นยังไงบ้าง", MessageType.USER_INPUT, 'session1')
    
    print(f"   📝 Messages added: {len(context.messages)}")
    print(f"   🤖 AI responses: {1 if context.state.last_ai_response else 0}")
    
    # Test context window
    print("\n🔍 Testing context window...")
    recent_messages = context.get_context_window(3)
    print(f"   Recent messages: {len(recent_messages)}")
    for msg in recent_messages:
        print(f"      {msg.username}: {msg.content[:50]}...")
    
    # Test search
    print("\n🔎 Testing message search...")
    weather_messages = context.search_messages("อากาศ")
    greeting_messages = context.search_messages("สวัสดี")
    
    print(f"   Weather messages: {len(weather_messages)}")
    print(f"   Greeting messages: {len(greeting_messages)}")
    
    # Test important messages
    print("\n⭐ Testing important messages...")
    context.mark_important_message(msg1.message_id)
    important = context.get_important_messages()
    print(f"   Important messages: {len(important)}")
    
    # Test topic detection
    print("\n🏷️ Testing topic detection...")
    print(f"   Current topic: {context.state.current_topic}")
    print(f"   Topic history: {context.conversation_topics}")
    
    # Test conversation state
    print("\n📊 Testing conversation state...")
    context.update_conversation_state({
        'conversation_mood': 'happy',
        'ai_personality': 'friendly'
    })
    
    summary = context.get_conversation_summary()
    print(f"   Duration: {summary['duration_minutes']:.1f} minutes")
    print(f"   Total messages: {summary['total_messages']}")
    print(f"   Participants: {summary['participants']}")
    print(f"   Current topic: {summary['current_topic']}")
    
    # Test export
    print("\n💾 Testing conversation export...")
    exported = context.export_conversation('json')
    print(f"   Exported {len(exported)} characters of conversation data")
    
    # Test user removal
    print("\n👋 Testing participant removal...")
    removed = context.remove_participant('user2')
    if removed:
        print(f"   ✅ User removed. Remaining: {len(context.participant_users)}")
    
    # Cleanup
    context.cleanup()
    
    return context


if __name__ == "__main__":
    test_conversation_context()