#!/usr/bin/env python3
"""
🎭 Multi-User Conversation Manager for JARVIS
ระบบจัดการการสนทนาหลายผู้ใช้สำหรับ JARVIS
"""

import logging
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from enum import Enum

from .user_manager import UserManager, User, UserSession
from .conversation_context import ConversationContext, ConversationMessage, MessageType, ConversationMode


class ConversationPriority(Enum):
    """ระดับความสำคัญของการสนทนา"""
    LOW = "low"                 # ความสำคัญต่ำ
    NORMAL = "normal"           # ความสำคัญปกติ
    HIGH = "high"              # ความสำคัญสูง
    URGENT = "urgent"          # เร่งด่วน
    EMERGENCY = "emergency"    # ฉุกเฉิน


@dataclass
class ConversationSession:
    """เซสชันการสนทนา"""
    conversation_id: str
    context: ConversationContext
    priority: ConversationPriority
    created_at: float
    last_activity: float
    is_active: bool = True
    max_participants: int = 10
    auto_cleanup_timeout: float = 3600  # 1 hour


class MultiUserManager:
    """ระบบจัดการการสนทนาหลายผู้ใช้สำหรับ JARVIS"""
    
    def __init__(self, user_manager: UserManager, config: Dict[str, Any] = None):
        self.user_manager = user_manager
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Conversation management
        self.active_conversations: Dict[str, ConversationSession] = {}
        self.user_conversations: Dict[str, List[str]] = {}  # user_id -> conversation_ids
        self.default_conversation_config = self.config.get('default_conversation', {})
        
        # Settings
        self.max_concurrent_conversations = self.config.get('max_concurrent_conversations', 50)
        self.default_conversation_timeout = self.config.get('default_conversation_timeout', 3600)
        self.enable_conversation_history = self.config.get('enable_conversation_history', True)
        self.auto_create_conversations = self.config.get('auto_create_conversations', True)
        
        # AI Integration
        self.ai_response_callbacks: List[Callable] = []
        self.conversation_event_callbacks: List[Callable] = []
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Cleanup service
        self.cleanup_active = False
        self.cleanup_thread = None
        
        # Conversation routing
        self.conversation_router = self._create_conversation_router()
        
        # Start services
        self.start_cleanup_service()
        
        self.logger.info("🎭 Multi-User Manager initialized")
    
    def _create_conversation_router(self) -> Dict[str, Callable]:
        """สร้างระบบ routing การสนทนา"""
        return {
            'single_user': self._route_single_user,
            'multi_user': self._route_multi_user,
            'group_chat': self._route_group_chat,
            'emergency': self._route_emergency
        }
    
    def create_conversation(self, initiator_user_id: str, mode: ConversationMode = ConversationMode.SINGLE_USER,
                          priority: ConversationPriority = ConversationPriority.NORMAL,
                          max_participants: int = None, config: Dict[str, Any] = None) -> Optional[str]:
        """สร้างการสนทนาใหม่"""
        with self.lock:
            # Check limits
            if len(self.active_conversations) >= self.max_concurrent_conversations:
                self.logger.warning(f"⚠️ Maximum concurrent conversations reached: {self.max_concurrent_conversations}")
                return None
            
            # Generate conversation ID
            conversation_id = f"conv_{int(time.time())}_{str(uuid.uuid4())[:8]}"
            
            # Create conversation context
            context_config = self.default_conversation_config.copy()
            if config:
                context_config.update(config)
            
            context_config.update({
                'mode': mode.value,
                'max_participants': max_participants or 10
            })
            
            context = ConversationContext(
                context_id=conversation_id,
                user_manager=self.user_manager,
                config=context_config
            )
            
            # Create conversation session
            session = ConversationSession(
                conversation_id=conversation_id,
                context=context,
                priority=priority,
                created_at=time.time(),
                last_activity=time.time(),
                max_participants=max_participants or 10,
                auto_cleanup_timeout=self.default_conversation_timeout
            )
            
            # Register callbacks
            context.register_message_callback(self._on_message_received)
            context.register_context_change_callback(lambda: self._on_context_changed(conversation_id))
            
            # Store conversation
            self.active_conversations[conversation_id] = session
            
            # Add initiator user if provided
            if initiator_user_id:
                self.join_conversation(conversation_id, initiator_user_id)
            
            self.logger.info(f"🆕 Conversation created: {conversation_id} ({mode.value}, {priority.value})")
            return conversation_id
    
    def join_conversation(self, conversation_id: str, user_id: str, session_id: str = None) -> bool:
        """เข้าร่วมการสนทนา"""
        with self.lock:
            # Check if conversation exists
            if conversation_id not in self.active_conversations:
                # Auto-create if enabled
                if self.auto_create_conversations:
                    new_conv_id = self.create_conversation(user_id)
                    if new_conv_id:
                        conversation_id = new_conv_id
                    else:
                        return False
                else:
                    return False
            
            session = self.active_conversations[conversation_id]
            
            # Check participant limit
            if len(session.context.participant_users) >= session.max_participants:
                self.logger.warning(f"⚠️ Conversation participant limit reached: {session.max_participants}")
                return False
            
            # Get user session if not provided
            if not session_id:
                user = self.user_manager.users.get(user_id)
                if user and user.active_sessions:
                    session_id = user.active_sessions[0]  # Use first active session
            
            # Add user to conversation
            success = session.context.add_participant(user_id, session_id or "")
            
            if success:
                # Track user conversations
                if user_id not in self.user_conversations:
                    self.user_conversations[user_id] = []
                
                if conversation_id not in self.user_conversations[user_id]:
                    self.user_conversations[user_id].append(conversation_id)
                
                # Update activity
                session.last_activity = time.time()
                
                # Trigger callbacks
                self._trigger_conversation_event('user_joined', conversation_id, user_id)
                
                self.logger.info(f"👥 User joined conversation: {user_id} → {conversation_id}")
            
            return success
    
    def leave_conversation(self, conversation_id: str, user_id: str) -> bool:
        """ออกจากการสนทนา"""
        with self.lock:
            if conversation_id not in self.active_conversations:
                return False
            
            session = self.active_conversations[conversation_id]
            success = session.context.remove_participant(user_id)
            
            if success:
                # Remove from user conversations
                if user_id in self.user_conversations:
                    if conversation_id in self.user_conversations[user_id]:
                        self.user_conversations[user_id].remove(conversation_id)
                    
                    # Clean up empty entries
                    if not self.user_conversations[user_id]:
                        del self.user_conversations[user_id]
                
                # Update activity
                session.last_activity = time.time()
                
                # Check if conversation should be closed
                if not session.context.participant_users:
                    self._close_conversation(conversation_id, "no_participants")
                
                # Trigger callbacks
                self._trigger_conversation_event('user_left', conversation_id, user_id)
                
                self.logger.info(f"👋 User left conversation: {user_id} ← {conversation_id}")
            
            return success
    
    def send_message(self, conversation_id: str, user_id: str, content: str, 
                    message_type: MessageType = MessageType.USER_INPUT,
                    session_id: str = None, metadata: Dict[str, Any] = None) -> Optional[ConversationMessage]:
        """ส่งข้อความในการสนทนา"""
        with self.lock:
            if conversation_id not in self.active_conversations:
                return None
            
            session = self.active_conversations[conversation_id]
            
            # Check if user is participant
            if user_id not in session.context.participant_users:
                self.logger.warning(f"⚠️ User {user_id} not in conversation {conversation_id}")
                return None
            
            # Add message
            message = session.context.add_message(
                user_id=user_id,
                content=content,
                message_type=message_type,
                session_id=session_id,
                metadata=metadata
            )
            
            # Update activity
            session.last_activity = time.time()
            
            # Route message for AI processing
            self._route_message_for_ai_processing(conversation_id, message)
            
            return message
    
    def send_ai_response(self, conversation_id: str, content: str, 
                        responding_to_user: str = None, confidence: float = 1.0,
                        metadata: Dict[str, Any] = None) -> Optional[ConversationMessage]:
        """ส่งการตอบสนองจาก AI"""
        with self.lock:
            if conversation_id not in self.active_conversations:
                return None
            
            session = self.active_conversations[conversation_id]
            
            # Add AI response
            ai_message = session.context.add_ai_response(
                content=content,
                responding_to_user=responding_to_user,
                confidence=confidence,
                metadata=metadata
            )
            
            # Update activity
            session.last_activity = time.time()
            
            return ai_message
    
    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """ดึงการสนทนา"""
        session = self.active_conversations.get(conversation_id)
        return session.context if session else None
    
    def get_user_conversations(self, user_id: str) -> List[str]:
        """ดึงรายการการสนทนาของผู้ใช้"""
        return self.user_conversations.get(user_id, []).copy()
    
    def get_active_conversations(self) -> List[Dict[str, Any]]:
        """ดึงรายการการสนทนาที่กำลังใช้งาน"""
        with self.lock:
            conversations = []
            for conv_id, session in self.active_conversations.items():
                conversations.append({
                    'conversation_id': conv_id,
                    'mode': session.context.mode.value,
                    'priority': session.priority.value,
                    'participants': len(session.context.participant_users),
                    'messages': len(session.context.messages),
                    'created_at': session.created_at,
                    'last_activity': session.last_activity,
                    'current_topic': session.context.state.current_topic
                })
            
            # Sort by priority and activity
            conversations.sort(key=lambda x: (
                self._priority_sort_key(x['priority']),
                -x['last_activity']
            ))
            
            return conversations
    
    def _priority_sort_key(self, priority: str) -> int:
        """แปลงความสำคัญเป็นตัวเลขสำหรับการเรียง"""
        priority_order = {
            'emergency': 0,
            'urgent': 1,
            'high': 2,
            'normal': 3,
            'low': 4
        }
        return priority_order.get(priority, 3)
    
    def switch_user_conversation(self, user_id: str, target_conversation_id: str) -> bool:
        """เปลี่ยนการสนทนาของผู้ใช้"""
        with self.lock:
            # Check if target conversation exists
            if target_conversation_id not in self.active_conversations:
                return False
            
            # Check if user is already in target conversation
            target_session = self.active_conversations[target_conversation_id]
            if user_id in target_session.context.participant_users:
                return True  # Already in conversation
            
            # Get user's current conversations
            current_conversations = self.get_user_conversations(user_id)
            
            # Leave current conversations (except target)
            for conv_id in current_conversations:
                if conv_id != target_conversation_id:
                    self.leave_conversation(conv_id, user_id)
            
            # Join target conversation
            return self.join_conversation(target_conversation_id, user_id)
    
    def _route_message_for_ai_processing(self, conversation_id: str, message: ConversationMessage):
        """นำข้อความไปประมวลผลด้วย AI"""
        # Trigger AI response callbacks
        for callback in self.ai_response_callbacks:
            try:
                callback(conversation_id, message)
            except Exception as e:
                self.logger.error(f"❌ AI response callback error: {e}")
    
    # Conversation routing methods
    def _route_single_user(self, conversation_id: str, message: ConversationMessage):
        """จัดการการสนทนาผู้ใช้เดียว"""
        # Simple routing for single user
        pass
    
    def _route_multi_user(self, conversation_id: str, message: ConversationMessage):
        """จัดการการสนทนาหลายผู้ใช้"""
        # Handle multi-user dynamics
        session = self.active_conversations.get(conversation_id)
        if session:
            # Check if all users need notification
            context = session.context
            if len(context.state.active_users) > 1:
                # Multi-user conversation logic
                pass
    
    def _route_group_chat(self, conversation_id: str, message: ConversationMessage):
        """จัดการแชทกลุ่ม"""
        # Group chat specific logic
        pass
    
    def _route_emergency(self, conversation_id: str, message: ConversationMessage):
        """จัดการการสนทนาฉุกเฉิน"""
        # Emergency priority handling
        pass
    
    def _on_message_received(self, message: ConversationMessage):
        """ประมวลผลข้อความที่ได้รับ"""
        # Find conversation containing this message
        for conv_id, session in self.active_conversations.items():
            if message.message_id in session.context.message_index:
                # Route message based on conversation mode
                mode = session.context.mode.value
                if mode in self.conversation_router:
                    self.conversation_router[mode](conv_id, message)
                break
    
    def _on_context_changed(self, conversation_id: str):
        """ประมวลผลการเปลี่ยนบริบท"""
        session = self.active_conversations.get(conversation_id)
        if session:
            session.last_activity = time.time()
    
    def _close_conversation(self, conversation_id: str, reason: str = "manual"):
        """ปิดการสนทนา"""
        with self.lock:
            if conversation_id not in self.active_conversations:
                return
            
            session = self.active_conversations[conversation_id]
            
            # Remove all participants
            participant_ids = list(session.context.participant_users.keys())
            for user_id in participant_ids:
                self.leave_conversation(conversation_id, user_id)
            
            # Cleanup context
            session.context.cleanup()
            session.is_active = False
            
            # Remove from active conversations
            del self.active_conversations[conversation_id]
            
            # Trigger callbacks
            self._trigger_conversation_event('conversation_closed', conversation_id, None, {'reason': reason})
            
            self.logger.info(f"🚪 Conversation closed: {conversation_id} (reason: {reason})")
    
    def start_cleanup_service(self):
        """เริ่มบริการทำความสะอาด"""
        if self.cleanup_active:
            return
        
        self.cleanup_active = True
        self.cleanup_thread = threading.Thread(
            target=self._cleanup_loop,
            daemon=True
        )
        self.cleanup_thread.start()
        self.logger.info("🧹 Conversation cleanup service started")
    
    def stop_cleanup_service(self):
        """หยุดบริการทำความสะอาด"""
        self.cleanup_active = False
        if self.cleanup_thread:
            self.cleanup_thread.join(timeout=5)
        self.logger.info("🛑 Conversation cleanup service stopped")
    
    def _cleanup_loop(self):
        """ลูปทำความสะอาดการสนทนา"""
        while self.cleanup_active:
            try:
                current_time = time.time()
                conversations_to_close = []
                
                with self.lock:
                    for conv_id, session in self.active_conversations.items():
                        # Check timeout
                        if current_time - session.last_activity > session.auto_cleanup_timeout:
                            conversations_to_close.append((conv_id, "timeout"))
                        
                        # Check if no participants
                        elif not session.context.participant_users:
                            conversations_to_close.append((conv_id, "no_participants"))
                
                # Close expired conversations
                for conv_id, reason in conversations_to_close:
                    self._close_conversation(conv_id, reason)
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"❌ Cleanup error: {e}")
                time.sleep(60)
    
    def register_ai_response_callback(self, callback: Callable[[str, ConversationMessage], None]):
        """ลงทะเบียน callback สำหรับการตอบสนอง AI"""
        self.ai_response_callbacks.append(callback)
    
    def register_conversation_event_callback(self, callback: Callable[[str, str, str, Dict], None]):
        """ลงทะเบียน callback สำหรับเหตุการณ์การสนทนา"""
        self.conversation_event_callbacks.append(callback)
    
    def _trigger_conversation_event(self, event_type: str, conversation_id: str, 
                                  user_id: str = None, metadata: Dict[str, Any] = None):
        """เรียก callbacks สำหรับเหตุการณ์การสนทนา"""
        for callback in self.conversation_event_callbacks:
            try:
                callback(event_type, conversation_id, user_id, metadata or {})
            except Exception as e:
                self.logger.error(f"❌ Conversation event callback error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติการสนทนา"""
        with self.lock:
            stats = {
                'active_conversations': len(self.active_conversations),
                'total_participants': sum(len(s.context.participant_users) for s in self.active_conversations.values()),
                'total_messages': sum(len(s.context.messages) for s in self.active_conversations.values()),
                'conversations_by_mode': {},
                'conversations_by_priority': {},
                'average_participants_per_conversation': 0,
                'most_active_conversation': None
            }
            
            # Calculate distributions
            for session in self.active_conversations.values():
                mode = session.context.mode.value
                priority = session.priority.value
                
                stats['conversations_by_mode'][mode] = stats['conversations_by_mode'].get(mode, 0) + 1
                stats['conversations_by_priority'][priority] = stats['conversations_by_priority'].get(priority, 0) + 1
            
            # Calculate averages
            if stats['active_conversations'] > 0:
                stats['average_participants_per_conversation'] = stats['total_participants'] / stats['active_conversations']
                
                # Find most active conversation
                most_active = max(
                    self.active_conversations.items(),
                    key=lambda x: len(x[1].context.messages),
                    default=(None, None)
                )
                
                if most_active[0]:
                    stats['most_active_conversation'] = {
                        'conversation_id': most_active[0],
                        'messages': len(most_active[1].context.messages),
                        'participants': len(most_active[1].context.participant_users)
                    }
            
            return stats
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        self.stop_cleanup_service()
        
        # Close all conversations
        with self.lock:
            conversation_ids = list(self.active_conversations.keys())
            for conv_id in conversation_ids:
                self._close_conversation(conv_id, "cleanup")
        
        # Clear callbacks
        self.ai_response_callbacks.clear()
        self.conversation_event_callbacks.clear()
        
        self.logger.info("🧹 Multi-User Manager cleaned up")


def test_multi_user_manager():
    """ทดสอบ Multi-User Manager"""
    print("🧪 Testing Multi-User Manager")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create user manager
    from .user_manager import UserManager, UserRole
    
    user_manager = UserManager({
        'auto_create_users': True,
        'require_authentication': False
    })
    
    # Create test users
    user1 = user_manager.create_user("alice", "Alice Smith", UserRole.USER, auto_login=True)
    user2 = user_manager.create_user("bob", "Bob Johnson", UserRole.USER, auto_login=True)
    user3 = user_manager.create_user("charlie", "Charlie Brown", UserRole.FAMILY, auto_login=True)
    
    # Create multi-user manager
    config = {
        'max_concurrent_conversations': 10,
        'auto_create_conversations': True,
        'default_conversation': {
            'context_window_size': 20
        }
    }
    
    multi_manager = MultiUserManager(user_manager, config)
    
    # Test conversation creation
    print("🆕 Testing conversation creation...")
    conv1_id = multi_manager.create_conversation(
        user1.user_id, 
        ConversationMode.MULTI_USER, 
        ConversationPriority.NORMAL
    )
    conv2_id = multi_manager.create_conversation(
        user2.user_id,
        ConversationMode.SINGLE_USER,
        ConversationPriority.HIGH
    )
    
    if conv1_id and conv2_id:
        print(f"   ✅ Conversations created: {conv1_id[:12]}..., {conv2_id[:12]}...")
    else:
        print("   ❌ Conversation creation failed")
    
    # Test joining conversations
    print("\n👥 Testing conversation joining...")
    join1 = multi_manager.join_conversation(conv1_id, user2.user_id)
    join2 = multi_manager.join_conversation(conv1_id, user3.user_id)
    
    if join1 and join2:
        conv1 = multi_manager.get_conversation(conv1_id)
        print(f"   ✅ Users joined: {len(conv1.participant_users)} participants")
    else:
        print("   ❌ User joining failed")
    
    # Test messaging
    print("\n💬 Testing messaging...")
    msg1 = multi_manager.send_message(conv1_id, user1.user_id, "สวัสดีทุกคน!")
    msg2 = multi_manager.send_message(conv1_id, user2.user_id, "Hello everyone!")
    ai_resp = multi_manager.send_ai_response(conv1_id, "สวัสดีครับทุกคน! มีอะไรให้ช่วยเหลือไหมครับ?")
    
    if msg1 and msg2 and ai_resp:
        conv1 = multi_manager.get_conversation(conv1_id)
        print(f"   ✅ Messages sent: {len(conv1.messages)} total messages")
    else:
        print("   ❌ Messaging failed")
    
    # Test conversation switching
    print("\n🔄 Testing conversation switching...")
    switch_success = multi_manager.switch_user_conversation(user2.user_id, conv2_id)
    if switch_success:
        user2_conversations = multi_manager.get_user_conversations(user2.user_id)
        print(f"   ✅ User switched: {len(user2_conversations)} active conversations")
    else:
        print("   ❌ Conversation switching failed")
    
    # Test active conversations
    print("\n📊 Testing conversation listing...")
    active_convs = multi_manager.get_active_conversations()
    print(f"   Active conversations: {len(active_convs)}")
    for conv in active_convs:
        print(f"      {conv['conversation_id'][:12]}... ({conv['mode']}, {conv['participants']} users)")
    
    # Test statistics
    print("\n📈 Testing statistics...")
    stats = multi_manager.get_statistics()
    print(f"   Total conversations: {stats['active_conversations']}")
    print(f"   Total participants: {stats['total_participants']}")
    print(f"   Total messages: {stats['total_messages']}")
    print(f"   Avg participants: {stats['average_participants_per_conversation']:.1f}")
    
    # Test conversation details
    print("\n🔍 Testing conversation details...")
    if conv1_id:
        conv1 = multi_manager.get_conversation(conv1_id)
        summary = conv1.get_conversation_summary()
        print(f"   Conversation duration: {summary['duration_minutes']:.1f} minutes")
        print(f"   Current topic: {summary['current_topic']}")
        print(f"   Message counts: {summary['user_message_counts']}")
    
    # Cleanup
    multi_manager.cleanup()
    user_manager.cleanup()
    
    return multi_manager


if __name__ == "__main__":
    test_multi_user_manager()