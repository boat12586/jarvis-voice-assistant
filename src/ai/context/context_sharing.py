"""
Advanced Context Sharing System for JARVIS
ระบบแบ่งปันบริบทขั้นสูงระหว่าง AI Agents
"""

import logging
import json
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Set, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
from enum import Enum
import asyncio
import pickle
import hashlib

class ContextType(Enum):
    """ประเภทบริบท"""
    CONVERSATION = "conversation"           # บริบทการสนทนา
    TASK = "task"                          # บริบทงาน
    USER_PROFILE = "user_profile"          # บริบทโปรไฟล์ผู้ใช้
    KNOWLEDGE = "knowledge"                # บริบทความรู้
    EMOTIONAL = "emotional"                # บริบทอารมณ์
    TEMPORAL = "temporal"                  # บริบทเวลา
    SPATIAL = "spatial"                    # บริบทสถานที่
    SYSTEM_STATE = "system_state"          # บริบทสถานะระบบ

class ContextPriority(Enum):
    """ระดับความสำคัญของบริบท"""
    CRITICAL = "critical"                  # สำคัญมาก (ต้องแชร์ทันที)
    HIGH = "high"                         # สำคัญ (แชร์เร็ว)
    MEDIUM = "medium"                     # ปานกลาง (แชร์ปกติ)
    LOW = "low"                           # น้อย (แชร์ได้ช้า)
    BACKGROUND = "background"             # เบื้องหลัง (แชร์เมื่อว่าง)

class ContextScope(Enum):
    """ขอบเขตการแชร์บริบท"""
    GLOBAL = "global"                     # แชร์กับ agent ทั้งหมด
    DOMAIN_SPECIFIC = "domain_specific"   # แชร์ในโดเมนเดียวกัน
    TARGETED = "targeted"                 # แชร์กับ agent เฉพาะ
    USER_SPECIFIC = "user_specific"       # แชร์เฉพาะ user คนเดียว
    SESSION_SPECIFIC = "session_specific" # แชร์เฉพาะ session

@dataclass
class ContextPacket:
    """แพ็กเก็ตบริบท"""
    packet_id: str
    source_agent: str                     # agent ที่ส่ง
    target_agents: Optional[List[str]]    # agent ที่รับ (None = ทั้งหมด)
    context_type: ContextType
    priority: ContextPriority
    scope: ContextScope
    context_data: Dict[str, Any]          # ข้อมูลบริบท
    metadata: Dict[str, Any]              # ข้อมูลเพิ่มเติม
    created_at: datetime
    expires_at: Optional[datetime] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    requires_acknowledgment: bool = False
    dependencies: List[str] = None        # packet_ids ที่ต้องประมวลผลก่อน
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class ContextSubscription:
    """การสมัครรับบริบท"""
    subscription_id: str
    agent_id: str
    context_types: List[ContextType]      # ประเภทบริบทที่สนใจ
    priority_filter: Optional[ContextPriority] = None  # กรองตามความสำคัญ
    scope_filter: Optional[List[ContextScope]] = None  # กรองตามขอบเขต
    user_filter: Optional[List[str]] = None            # กรองตาม user
    callback: Optional[Callable] = None   # callback function
    is_active: bool = True
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.scope_filter is None:
            self.scope_filter = []

@dataclass
class ContextTransferRecord:
    """บันทึกการส่งบริบท"""
    transfer_id: str
    packet_id: str
    source_agent: str
    target_agent: str
    transfer_time: datetime
    status: str = "pending"  # pending, delivered, failed, expired
    retry_count: int = 0
    error_message: Optional[str] = None

class ContextTransferProtocol:
    """โปรโตคอลการส่งบริบทระหว่าง Agents"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Core components
        self.agent_registry: Dict[str, Dict[str, Any]] = {}  # agent_id -> agent_info
        self.context_queue: deque = deque()                  # คิวบริบทที่รอส่ง
        self.priority_queues: Dict[ContextPriority, deque] = {
            priority: deque() for priority in ContextPriority
        }
        
        # Subscriptions and routing
        self.subscriptions: Dict[str, ContextSubscription] = {}  # subscription_id -> subscription
        self.agent_subscriptions: Dict[str, List[str]] = defaultdict(list)  # agent_id -> subscription_ids
        self.context_routing_table: Dict[ContextType, Set[str]] = defaultdict(set)  # context_type -> agent_ids
        
        # Transfer tracking
        self.active_transfers: Dict[str, ContextTransferRecord] = {}
        self.transfer_history: Dict[str, List[ContextTransferRecord]] = defaultdict(list)
        self.acknowledgments: Dict[str, Set[str]] = {}  # packet_id -> set of agent_ids that acknowledged
        
        # Context storage
        self.context_cache: Dict[str, ContextPacket] = {}  # packet_id -> packet
        self.context_index: Dict[str, Set[str]] = defaultdict(set)  # index_key -> packet_ids
        
        # Performance settings
        self.max_queue_size = config.get("max_queue_size", 10000)
        self.max_retry_count = config.get("max_retry_count", 3)
        self.transfer_timeout = config.get("transfer_timeout", 30)  # seconds
        self.cleanup_interval = config.get("cleanup_interval", 300)  # 5 minutes
        self.batch_size = config.get("batch_size", 50)
        
        # Threading
        self._running = False
        self._transfer_thread = None
        self._cleanup_thread = None
        self._last_cleanup = time.time()
        
        # Data persistence
        self.data_dir = Path(config.get("context_dir", "data/context_sharing"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load persisted data
        self._load_persistent_data()
        
        self.logger.info("Context Transfer Protocol initialized")
    
    def register_agent(self, agent_id: str, agent_info: Dict[str, Any]) -> bool:
        """ลงทะเบียน agent"""
        self.agent_registry[agent_id] = {
            **agent_info,
            "registered_at": datetime.now(),
            "last_seen": datetime.now(),
            "is_active": True,
            "capabilities": agent_info.get("capabilities", []),
            "domains": agent_info.get("domains", [])
        }
        
        self.logger.info(f"Registered agent: {agent_id}")
        return True
    
    def unregister_agent(self, agent_id: str) -> bool:
        """ยกเลิกการลงทะเบียน agent"""
        if agent_id in self.agent_registry:
            self.agent_registry[agent_id]["is_active"] = False
            
            # Remove subscriptions
            if agent_id in self.agent_subscriptions:
                subscription_ids = self.agent_subscriptions[agent_id].copy()
                for sub_id in subscription_ids:
                    self.unsubscribe(sub_id)
            
            self.logger.info(f"Unregistered agent: {agent_id}")
            return True
        return False
    
    def subscribe_to_context(self, agent_id: str, context_types: List[ContextType],
                           priority_filter: Optional[ContextPriority] = None,
                           scope_filter: Optional[List[ContextScope]] = None,
                           user_filter: Optional[List[str]] = None,
                           callback: Optional[Callable] = None) -> str:
        """สมัครรับบริบท"""
        
        subscription_id = f"sub_{agent_id}_{int(time.time())}_{len(self.subscriptions)}"
        
        subscription = ContextSubscription(
            subscription_id=subscription_id,
            agent_id=agent_id,
            context_types=context_types,
            priority_filter=priority_filter,
            scope_filter=scope_filter or [],
            user_filter=user_filter,
            callback=callback
        )
        
        self.subscriptions[subscription_id] = subscription
        self.agent_subscriptions[agent_id].append(subscription_id)
        
        # Update routing table
        for context_type in context_types:
            self.context_routing_table[context_type].add(agent_id)
        
        self.logger.info(f"Agent {agent_id} subscribed to context types: {[ct.value for ct in context_types]}")
        return subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """ยกเลิกการสมัครรับบริบท"""
        if subscription_id not in self.subscriptions:
            return False
        
        subscription = self.subscriptions[subscription_id]
        agent_id = subscription.agent_id
        
        # Remove from agent subscriptions
        if agent_id in self.agent_subscriptions:
            self.agent_subscriptions[agent_id].remove(subscription_id)
        
        # Update routing table
        for context_type in subscription.context_types:
            self.context_routing_table[context_type].discard(agent_id)
        
        # Remove subscription
        del self.subscriptions[subscription_id]
        
        self.logger.info(f"Unsubscribed: {subscription_id}")
        return True
    
    def share_context(self, source_agent: str, context_type: ContextType,
                     context_data: Dict[str, Any], priority: ContextPriority = ContextPriority.MEDIUM,
                     scope: ContextScope = ContextScope.GLOBAL,
                     target_agents: Optional[List[str]] = None,
                     user_id: Optional[str] = None, session_id: Optional[str] = None,
                     expires_in_seconds: Optional[int] = None,
                     requires_ack: bool = False, metadata: Optional[Dict[str, Any]] = None) -> str:
        """แชร์บริบทกับ agents อื่น"""
        
        packet_id = self._generate_packet_id(source_agent, context_type)
        
        expires_at = None
        if expires_in_seconds:
            expires_at = datetime.now() + timedelta(seconds=expires_in_seconds)
        
        packet = ContextPacket(
            packet_id=packet_id,
            source_agent=source_agent,
            target_agents=target_agents,
            context_type=context_type,
            priority=priority,
            scope=scope,
            context_data=context_data,
            metadata=metadata or {},
            created_at=datetime.now(),
            expires_at=expires_at,
            user_id=user_id,
            session_id=session_id,
            requires_acknowledgment=requires_ack
        )
        
        # Store packet
        self.context_cache[packet_id] = packet
        
        # Index packet
        self._index_packet(packet)
        
        # Queue for distribution
        self.priority_queues[priority].append(packet)
        
        # Initialize acknowledgments if required
        if requires_ack:
            self.acknowledgments[packet_id] = set()
        
        self.logger.debug(f"Context shared: {packet_id} from {source_agent}")
        return packet_id
    
    def _generate_packet_id(self, source_agent: str, context_type: ContextType) -> str:
        """สร้าง packet ID"""
        timestamp = str(time.time())
        content = f"{source_agent}_{context_type.value}_{timestamp}"
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    def _index_packet(self, packet: ContextPacket):
        """จัดทำดัชนีแพ็กเก็ต"""
        packet_id = packet.packet_id
        
        # Index by type
        self.context_index[f"type:{packet.context_type.value}"].add(packet_id)
        
        # Index by source agent
        self.context_index[f"source:{packet.source_agent}"].add(packet_id)
        
        # Index by user
        if packet.user_id:
            self.context_index[f"user:{packet.user_id}"].add(packet_id)
        
        # Index by session
        if packet.session_id:
            self.context_index[f"session:{packet.session_id}"].add(packet_id)
        
        # Index by priority
        self.context_index[f"priority:{packet.priority.value}"].add(packet_id)
        
        # Index by scope
        self.context_index[f"scope:{packet.scope.value}"].add(packet_id)
    
    def get_contexts(self, query_filters: Dict[str, Any], limit: int = 100) -> List[ContextPacket]:
        """ดึงบริบทตามเงื่อนไข"""
        matching_packets = set()
        
        # Build query from filters
        for filter_type, filter_value in query_filters.items():
            if filter_type == "context_type" and isinstance(filter_value, ContextType):
                key = f"type:{filter_value.value}"
            elif filter_type == "source_agent":
                key = f"source:{filter_value}"
            elif filter_type == "user_id":
                key = f"user:{filter_value}"
            elif filter_type == "session_id":
                key = f"session:{filter_value}"
            elif filter_type == "priority" and isinstance(filter_value, ContextPriority):
                key = f"priority:{filter_value.value}"
            elif filter_type == "scope" and isinstance(filter_value, ContextScope):
                key = f"scope:{filter_value.value}"
            else:
                continue
            
            if key in self.context_index:
                if not matching_packets:
                    matching_packets = self.context_index[key].copy()
                else:
                    matching_packets &= self.context_index[key]
        
        # Get packets and filter expired ones
        results = []
        current_time = datetime.now()
        
        for packet_id in matching_packets:
            if packet_id in self.context_cache:
                packet = self.context_cache[packet_id]
                
                # Check if expired
                if packet.expires_at and current_time > packet.expires_at:
                    continue
                
                results.append(packet)
        
        # Sort by creation time (newest first)
        results.sort(key=lambda x: x.created_at, reverse=True)
        
        return results[:limit]
    
    def acknowledge_context(self, agent_id: str, packet_id: str) -> bool:
        """ยืนยันการรับบริบท"""
        if packet_id in self.acknowledgments:
            self.acknowledgments[packet_id].add(agent_id)
            self.logger.debug(f"Context acknowledged: {packet_id} by {agent_id}")
            return True
        return False
    
    def start_context_distribution(self):
        """เริ่มการแจกจ่ายบริบท"""
        if self._running:
            return
        
        self._running = True
        self._transfer_thread = threading.Thread(target=self._context_distribution_loop, daemon=True)
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        
        self._transfer_thread.start()
        self._cleanup_thread.start()
        
        self.logger.info("Context distribution started")
    
    def stop_context_distribution(self):
        """หยุดการแจกจ่ายบริบท"""
        self._running = False
        
        if self._transfer_thread and self._transfer_thread.is_alive():
            self._transfer_thread.join(timeout=5)
        
        if self._cleanup_thread and self._cleanup_thread.is_alive():
            self._cleanup_thread.join(timeout=5)
        
        self.logger.info("Context distribution stopped")
    
    def _context_distribution_loop(self):
        """วนรอบการแจกจ่ายบริบท"""
        while self._running:
            try:
                # Process packets by priority
                packets_processed = 0
                
                for priority in [ContextPriority.CRITICAL, ContextPriority.HIGH, 
                               ContextPriority.MEDIUM, ContextPriority.LOW, ContextPriority.BACKGROUND]:
                    queue = self.priority_queues[priority]
                    batch_count = 0
                    
                    while queue and batch_count < self.batch_size:
                        packet = queue.popleft()
                        
                        # Check if packet expired
                        if packet.expires_at and datetime.now() > packet.expires_at:
                            self._handle_expired_packet(packet)
                            continue
                        
                        # Distribute packet
                        self._distribute_packet(packet)
                        packets_processed += 1
                        batch_count += 1
                    
                    # Don't process lower priority if we're busy with higher priority
                    if batch_count >= self.batch_size:
                        break
                
                # Sleep based on activity
                if packets_processed == 0:
                    time.sleep(0.1)  # No activity
                else:
                    time.sleep(0.01)  # High activity
                    
            except Exception as e:
                self.logger.error(f"Error in context distribution loop: {e}")
                time.sleep(1)
    
    def _distribute_packet(self, packet: ContextPacket):
        """แจกจ่ายแพ็กเก็ตบริบท"""
        # Determine target agents
        target_agents = []
        
        if packet.target_agents:
            # Explicit targets
            target_agents = packet.target_agents
        else:
            # Find subscribers
            target_agents = self._find_subscribers(packet)
        
        # Create transfer records and distribute
        for target_agent in target_agents:
            if target_agent == packet.source_agent:
                continue  # Don't send back to source
            
            if not self._is_agent_active(target_agent):
                continue  # Skip inactive agents
            
            transfer_id = f"transfer_{packet.packet_id}_{target_agent}_{int(time.time())}"
            
            transfer_record = ContextTransferRecord(
                transfer_id=transfer_id,
                packet_id=packet.packet_id,
                source_agent=packet.source_agent,
                target_agent=target_agent,
                transfer_time=datetime.now()
            )
            
            self.active_transfers[transfer_id] = transfer_record
            
            # Attempt delivery
            success = self._deliver_context(target_agent, packet)
            
            if success:
                transfer_record.status = "delivered"
                self.transfer_history[target_agent].append(transfer_record)
                del self.active_transfers[transfer_id]
            else:
                transfer_record.status = "failed"
                transfer_record.retry_count = 1
    
    def _find_subscribers(self, packet: ContextPacket) -> List[str]:
        """หา subscribers ที่เหมาะสม"""
        subscribers = []
        
        # Get potential subscribers from routing table
        potential_agents = self.context_routing_table.get(packet.context_type, set())
        
        for agent_id in potential_agents:
            if self._should_deliver_to_agent(agent_id, packet):
                subscribers.append(agent_id)
        
        return subscribers
    
    def _should_deliver_to_agent(self, agent_id: str, packet: ContextPacket) -> bool:
        """ตรวจสอบว่าควรส่งบริบทให้ agent นี้หรือไม่"""
        if not self._is_agent_active(agent_id):
            return False
        
        # Check agent subscriptions
        if agent_id not in self.agent_subscriptions:
            return False
        
        subscription_ids = self.agent_subscriptions[agent_id]
        
        for sub_id in subscription_ids:
            if sub_id not in self.subscriptions:
                continue
                
            subscription = self.subscriptions[sub_id]
            
            if not subscription.is_active:
                continue
            
            # Check context type
            if packet.context_type not in subscription.context_types:
                continue
            
            # Check priority filter
            if (subscription.priority_filter and 
                self._priority_order(packet.priority) < self._priority_order(subscription.priority_filter)):
                continue
            
            # Check scope filter
            if (subscription.scope_filter and 
                packet.scope not in subscription.scope_filter):
                continue
            
            # Check user filter
            if (subscription.user_filter and packet.user_id and 
                packet.user_id not in subscription.user_filter):
                continue
            
            return True
        
        return False
    
    def _priority_order(self, priority: ContextPriority) -> int:
        """แปลงความสำคัญเป็นตัวเลข"""
        order = {
            ContextPriority.CRITICAL: 5,
            ContextPriority.HIGH: 4,
            ContextPriority.MEDIUM: 3,
            ContextPriority.LOW: 2,
            ContextPriority.BACKGROUND: 1
        }
        return order.get(priority, 0)
    
    def _is_agent_active(self, agent_id: str) -> bool:
        """ตรวจสอบว่า agent ยังใช้งานอยู่หรือไม่"""
        if agent_id not in self.agent_registry:
            return False
        
        agent_info = self.agent_registry[agent_id]
        return agent_info.get("is_active", False)
    
    def _deliver_context(self, target_agent: str, packet: ContextPacket) -> bool:
        """ส่งบริบทให้ agent"""
        try:
            # Find subscription callback
            callback = None
            
            if target_agent in self.agent_subscriptions:
                for sub_id in self.agent_subscriptions[target_agent]:
                    subscription = self.subscriptions.get(sub_id)
                    if (subscription and subscription.is_active and 
                        packet.context_type in subscription.context_types):
                        callback = subscription.callback
                        break
            
            if callback:
                # Use callback delivery
                result = callback(packet)
                if result:
                    self.logger.debug(f"Context delivered via callback: {packet.packet_id} -> {target_agent}")
                    return True
            else:
                # Default delivery - update agent's last_seen and log
                if target_agent in self.agent_registry:
                    self.agent_registry[target_agent]["last_seen"] = datetime.now()
                
                self.logger.debug(f"Context delivered: {packet.packet_id} -> {target_agent}")
                return True
            
        except Exception as e:
            self.logger.error(f"Failed to deliver context to {target_agent}: {e}")
        
        return False
    
    def _handle_expired_packet(self, packet: ContextPacket):
        """จัดการแพ็กเก็ตที่หมดอายุ"""
        # Remove from cache
        if packet.packet_id in self.context_cache:
            del self.context_cache[packet.packet_id]
        
        # Remove from indexes
        self._remove_from_indexes(packet.packet_id)
        
        # Remove acknowledgments
        if packet.packet_id in self.acknowledgments:
            del self.acknowledgments[packet.packet_id]
        
        self.logger.debug(f"Expired packet removed: {packet.packet_id}")
    
    def _remove_from_indexes(self, packet_id: str):
        """ลบจากดัชนีทั้งหมด"""
        for packet_set in self.context_index.values():
            packet_set.discard(packet_id)
    
    def _cleanup_loop(self):
        """วนรอบการทำความสะอาด"""
        while self._running:
            try:
                current_time = time.time()
                
                if current_time - self._last_cleanup > self.cleanup_interval:
                    self._perform_cleanup()
                    self._last_cleanup = current_time
                
                time.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                time.sleep(60)
    
    def _perform_cleanup(self):
        """ทำความสะอาดระบบ"""
        current_time = datetime.now()
        
        # Clean expired packets
        expired_packets = []
        for packet_id, packet in self.context_cache.items():
            if packet.expires_at and current_time > packet.expires_at:
                expired_packets.append(packet_id)
        
        for packet_id in expired_packets:
            packet = self.context_cache.get(packet_id)
            if packet:
                self._handle_expired_packet(packet)
        
        # Clean old transfer records
        for agent_id, records in self.transfer_history.items():
            cutoff_time = current_time - timedelta(hours=24)  # Keep 24 hours
            self.transfer_history[agent_id] = [
                r for r in records if r.transfer_time > cutoff_time
            ]
        
        # Retry failed transfers
        failed_transfers = [(tid, record) for tid, record in self.active_transfers.items() 
                           if record.status == "failed" and record.retry_count < self.max_retry_count]
        
        for transfer_id, record in failed_transfers:
            if record.packet_id in self.context_cache:
                packet = self.context_cache[record.packet_id]
                success = self._deliver_context(record.target_agent, packet)
                
                if success:
                    record.status = "delivered"
                    self.transfer_history[record.target_agent].append(record)
                    del self.active_transfers[transfer_id]
                else:
                    record.retry_count += 1
                    if record.retry_count >= self.max_retry_count:
                        record.status = "failed_max_retries"
                        del self.active_transfers[transfer_id]
        
        # Update agent last_seen status
        inactive_threshold = current_time - timedelta(minutes=30)
        for agent_id, agent_info in self.agent_registry.items():
            if agent_info.get("last_seen", current_time) < inactive_threshold:
                agent_info["is_active"] = False
        
        self.logger.debug("System cleanup completed")
    
    def get_context_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติระบบบริบท"""
        current_time = datetime.now()
        
        # Count packets by type
        packet_counts = defaultdict(int)
        expired_count = 0
        
        for packet in self.context_cache.values():
            packet_counts[packet.context_type.value] += 1
            if packet.expires_at and current_time > packet.expires_at:
                expired_count += 1
        
        # Count queue sizes
        queue_sizes = {
            priority.value: len(queue) 
            for priority, queue in self.priority_queues.items()
        }
        
        # Agent statistics
        active_agents = sum(1 for info in self.agent_registry.values() if info.get("is_active", False))
        total_agents = len(self.agent_registry)
        
        # Transfer statistics
        total_transfers = sum(len(records) for records in self.transfer_history.values())
        active_transfers = len(self.active_transfers)
        
        return {
            "context_cache_size": len(self.context_cache),
            "expired_packets": expired_count,
            "packet_distribution": dict(packet_counts),
            "queue_sizes": queue_sizes,
            "agents": {
                "total": total_agents,
                "active": active_agents,
                "subscriptions": len(self.subscriptions)
            },
            "transfers": {
                "total_completed": total_transfers,
                "active": active_transfers,
                "pending_acknowledgments": len(self.acknowledgments)
            },
            "index_size": {key: len(packet_set) for key, packet_set in self.context_index.items()}
        }
    
    def _save_persistent_data(self):
        """บันทึกข้อมูลที่ต้องเก็บ"""
        try:
            # Save agent registry
            registry_file = self.data_dir / "agent_registry.json"
            registry_data = {}
            for agent_id, info in self.agent_registry.items():
                info_copy = info.copy()
                info_copy["registered_at"] = info["registered_at"].isoformat()
                info_copy["last_seen"] = info["last_seen"].isoformat()
                registry_data[agent_id] = info_copy
            
            with open(registry_file, 'w', encoding='utf-8') as f:
                json.dump(registry_data, f, ensure_ascii=False, indent=2)
            
            # Save subscriptions
            subscriptions_file = self.data_dir / "subscriptions.pkl"
            with open(subscriptions_file, 'wb') as f:
                pickle.dump(self.subscriptions, f)
            
            # Save recent context cache (last 1 hour only)
            recent_cache = {}
            cutoff_time = datetime.now() - timedelta(hours=1)
            for packet_id, packet in self.context_cache.items():
                if packet.created_at > cutoff_time:
                    recent_cache[packet_id] = packet
            
            cache_file = self.data_dir / "recent_context_cache.pkl"
            with open(cache_file, 'wb') as f:
                pickle.dump(recent_cache, f)
            
            self.logger.info("Persistent data saved")
            
        except Exception as e:
            self.logger.error(f"Failed to save persistent data: {e}")
    
    def _load_persistent_data(self):
        """โหลดข้อมูลที่เก็บไว้"""
        try:
            # Load agent registry
            registry_file = self.data_dir / "agent_registry.json"
            if registry_file.exists():
                with open(registry_file, 'r', encoding='utf-8') as f:
                    registry_data = json.load(f)
                
                for agent_id, info in registry_data.items():
                    info["registered_at"] = datetime.fromisoformat(info["registered_at"])
                    info["last_seen"] = datetime.fromisoformat(info["last_seen"])
                    info["is_active"] = False  # Mark as inactive on startup
                    self.agent_registry[agent_id] = info
            
            # Load subscriptions
            subscriptions_file = self.data_dir / "subscriptions.pkl"
            if subscriptions_file.exists():
                with open(subscriptions_file, 'rb') as f:
                    loaded_subscriptions = pickle.load(f)
                
                # Rebuild subscriptions and routing
                for sub_id, subscription in loaded_subscriptions.items():
                    self.subscriptions[sub_id] = subscription
                    self.agent_subscriptions[subscription.agent_id].append(sub_id)
                    
                    for context_type in subscription.context_types:
                        self.context_routing_table[context_type].add(subscription.agent_id)
            
            # Load recent context cache
            cache_file = self.data_dir / "recent_context_cache.pkl"
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    recent_cache = pickle.load(f)
                
                # Re-index loaded packets
                for packet_id, packet in recent_cache.items():
                    self.context_cache[packet_id] = packet
                    self._index_packet(packet)
            
            self.logger.info("Persistent data loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load persistent data: {e}")
    
    def shutdown(self):
        """ปิดระบบ"""
        self.stop_context_distribution()
        self._save_persistent_data()
        self.logger.info("Context Transfer Protocol shutdown complete")