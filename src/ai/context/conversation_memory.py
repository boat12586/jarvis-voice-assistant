"""
Advanced Conversation Memory System for JARVIS
ระบบจดจำการสนทนาขั้นสูง - เก็บหน่วยความจำระยะสั้นและระยะยาว
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, deque
import pickle
import hashlib
from enum import Enum
import numpy as np

class MemoryType(Enum):
    """ประเภทหน่วยความจำ"""
    WORKING = "working"           # หน่วยความจำทำงาน (15-30 วินาที)
    SHORT_TERM = "short_term"     # หน่วยความจำระยะสั้น (15-30 นาที)
    LONG_TERM = "long_term"       # หน่วยความจำระยะยาว (วัน-สัปดาห์)
    EPISODIC = "episodic"         # หน่วยความจำเหตุการณ์ (เรื่องราวที่สำคัญ)
    SEMANTIC = "semantic"         # หน่วยความจำความหมาย (ข้อมูล/ความรู้)

class MemoryImportance(Enum):
    """ระดับความสำคัญของความทรงจำ"""
    CRITICAL = "critical"         # สำคัญมาก (เก็บถาวร)
    HIGH = "high"                # สำคัญ (เก็บนาน)
    MEDIUM = "medium"            # ปานกลาง (เก็บปกติ)
    LOW = "low"                  # น้อย (ลบได้)
    TEMPORARY = "temporary"       # ชั่วคราว (ลบเร็ว)

@dataclass
class MemoryFragment:
    """ส่วนย่อยของความทรงจำ"""
    fragment_id: str
    content: Dict[str, Any]
    memory_type: MemoryType
    importance: MemoryImportance
    timestamp: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    associated_topics: Set[str] = None
    emotional_weight: float = 0.0
    user_id: Optional[str] = None
    context_tags: Set[str] = None
    
    def __post_init__(self):
        if self.associated_topics is None:
            self.associated_topics = set()
        if self.context_tags is None:
            self.context_tags = set()
        if self.last_accessed is None:
            self.last_accessed = self.timestamp

@dataclass
class ConversationContext:
    """บริบทการสนทนา"""
    context_id: str
    user_id: str
    session_id: str
    conversation_topic: Optional[str]
    active_memories: List[str]  # fragment_ids
    context_summary: str
    importance_score: float
    created_at: datetime
    updated_at: datetime
    
class AdvancedConversationMemory:
    """ระบบจดจำการสนทนาขั้นสูง"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Memory storage
        self.working_memory: deque = deque(maxlen=config.get("working_memory_size", 10))
        self.short_term_memory: Dict[str, MemoryFragment] = {}
        self.long_term_memory: Dict[str, MemoryFragment] = {}
        self.episodic_memory: Dict[str, MemoryFragment] = {}
        self.semantic_memory: Dict[str, MemoryFragment] = {}
        
        # Context management
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.context_history: Dict[str, List[ConversationContext]] = defaultdict(list)
        
        # Memory indexing
        self.topic_index: Dict[str, Set[str]] = defaultdict(set)  # topic -> fragment_ids
        self.user_index: Dict[str, Set[str]] = defaultdict(set)   # user_id -> fragment_ids
        self.time_index: Dict[str, Set[str]] = defaultdict(set)   # date -> fragment_ids
        self.importance_index: Dict[MemoryImportance, Set[str]] = defaultdict(set)
        
        # Memory management parameters
        self.working_memory_duration = timedelta(seconds=config.get("working_memory_duration", 30))
        self.short_term_duration = timedelta(minutes=config.get("short_term_duration", 30))
        self.long_term_threshold = config.get("long_term_threshold", 0.7)
        self.max_short_term_size = config.get("max_short_term_size", 1000)
        self.max_long_term_size = config.get("max_long_term_size", 10000)
        
        # Data persistence
        self.data_dir = Path(config.get("memory_dir", "data/advanced_memory"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing memories
        self._load_persistent_memory()
        
        # Start memory management
        self._last_cleanup = time.time()
        self.cleanup_interval = config.get("cleanup_interval", 300)  # 5 minutes
        
        self.logger.info("Advanced Conversation Memory System initialized")
    
    def add_memory(self, content: Dict[str, Any], memory_type: MemoryType,
                   importance: MemoryImportance, user_id: Optional[str] = None,
                   topics: Optional[Set[str]] = None, tags: Optional[Set[str]] = None,
                   emotional_weight: float = 0.0) -> str:
        """เพิ่มความทรงจำใหม่"""
        
        fragment_id = self._generate_fragment_id(content)
        
        fragment = MemoryFragment(
            fragment_id=fragment_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            timestamp=datetime.now(),
            associated_topics=topics or set(),
            emotional_weight=emotional_weight,
            user_id=user_id,
            context_tags=tags or set()
        )
        
        # Store in appropriate memory store
        if memory_type == MemoryType.WORKING:
            self.working_memory.append(fragment)
        elif memory_type == MemoryType.SHORT_TERM:
            self.short_term_memory[fragment_id] = fragment
        elif memory_type == MemoryType.LONG_TERM:
            self.long_term_memory[fragment_id] = fragment
        elif memory_type == MemoryType.EPISODIC:
            self.episodic_memory[fragment_id] = fragment
        elif memory_type == MemoryType.SEMANTIC:
            self.semantic_memory[fragment_id] = fragment
        
        # Update indexes
        self._update_indexes(fragment)
        
        # Trigger memory consolidation if needed
        self._consolidate_memories()
        
        self.logger.debug(f"Added {memory_type.value} memory: {fragment_id}")
        return fragment_id
    
    def retrieve_memories(self, query: str, user_id: Optional[str] = None,
                         memory_types: Optional[List[MemoryType]] = None,
                         max_results: int = 10, 
                         time_range: Optional[Tuple[datetime, datetime]] = None) -> List[MemoryFragment]:
        """ดึงความทรงจำที่เกี่ยวข้อง"""
        
        if memory_types is None:
            memory_types = list(MemoryType)
        
        # Get all relevant memory stores
        all_memories = []
        
        for memory_type in memory_types:
            if memory_type == MemoryType.WORKING:
                all_memories.extend(list(self.working_memory))
            elif memory_type == MemoryType.SHORT_TERM:
                all_memories.extend(self.short_term_memory.values())
            elif memory_type == MemoryType.LONG_TERM:
                all_memories.extend(self.long_term_memory.values())
            elif memory_type == MemoryType.EPISODIC:
                all_memories.extend(self.episodic_memory.values())
            elif memory_type == MemoryType.SEMANTIC:
                all_memories.extend(self.semantic_memory.values())
        
        # Filter by user if specified
        if user_id:
            all_memories = [m for m in all_memories if m.user_id == user_id]
        
        # Filter by time range if specified
        if time_range:
            start_time, end_time = time_range
            all_memories = [m for m in all_memories 
                           if start_time <= m.timestamp <= end_time]
        
        # Score and rank memories
        scored_memories = []
        for memory in all_memories:
            score = self._calculate_memory_relevance(memory, query)
            if score > 0.1:  # Minimum relevance threshold
                scored_memories.append((score, memory))
        
        # Sort by relevance score and recency
        scored_memories.sort(key=lambda x: (x[0], x[1].timestamp), reverse=True)
        
        # Update access statistics
        retrieved_memories = []
        for score, memory in scored_memories[:max_results]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            retrieved_memories.append(memory)
        
        self.logger.debug(f"Retrieved {len(retrieved_memories)} memories for query: {query}")
        return retrieved_memories
    
    def create_conversation_context(self, user_id: str, session_id: str,
                                  topic: Optional[str] = None) -> ConversationContext:
        """สร้างบริบทการสนทนาใหม่"""
        
        context_id = f"ctx_{user_id}_{session_id}_{int(time.time())}"
        
        # Retrieve relevant memories for context
        relevant_memories = []
        if topic:
            query = f"topic:{topic}"
            relevant_memories = self.retrieve_memories(query, user_id, max_results=5)
        
        context = ConversationContext(
            context_id=context_id,
            user_id=user_id,
            session_id=session_id,
            conversation_topic=topic,
            active_memories=[m.fragment_id for m in relevant_memories],
            context_summary=self._generate_context_summary(relevant_memories),
            importance_score=self._calculate_context_importance(relevant_memories),
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.active_contexts[context_id] = context
        
        self.logger.info(f"Created conversation context: {context_id}")
        return context
    
    def update_conversation_context(self, context_id: str, new_memories: List[str],
                                  new_topic: Optional[str] = None):
        """อัปเดตบริบทการสนทนา"""
        
        if context_id not in self.active_contexts:
            self.logger.warning(f"Context not found: {context_id}")
            return
        
        context = self.active_contexts[context_id]
        
        # Add new memories to context
        context.active_memories.extend(new_memories)
        # Keep only recent memories (max 20)
        context.active_memories = context.active_memories[-20:]
        
        # Update topic if provided
        if new_topic:
            context.conversation_topic = new_topic
        
        # Regenerate summary
        active_memory_objects = []
        for memory_id in context.active_memories:
            memory = self._get_memory_by_id(memory_id)
            if memory:
                active_memory_objects.append(memory)
        
        context.context_summary = self._generate_context_summary(active_memory_objects)
        context.importance_score = self._calculate_context_importance(active_memory_objects)
        context.updated_at = datetime.now()
        
        self.logger.debug(f"Updated conversation context: {context_id}")
    
    def get_conversation_summary(self, user_id: str, session_id: str,
                               max_turns: int = 10) -> Dict[str, Any]:
        """สร้างสรุปการสนทนา"""
        
        # Find memories for this session
        session_memories = []
        for memory_store in [self.working_memory, self.short_term_memory.values(),
                           self.long_term_memory.values(), self.episodic_memory.values()]:
            for memory in memory_store:
                if (memory.user_id == user_id and 
                    memory.context_tags and 'session:' + session_id in memory.context_tags):
                    session_memories.append(memory)
        
        # Sort by timestamp
        session_memories.sort(key=lambda x: x.timestamp)
        
        # Take recent memories
        recent_memories = session_memories[-max_turns:]
        
        # Generate summary
        summary = {
            "session_id": session_id,
            "user_id": user_id,
            "total_turns": len(session_memories),
            "recent_turns": len(recent_memories),
            "topics_discussed": list(set().union(*[m.associated_topics for m in recent_memories])),
            "conversation_flow": [],
            "key_insights": [],
            "emotional_tone": self._analyze_emotional_tone(recent_memories),
            "duration": None
        }
        
        if session_memories:
            start_time = session_memories[0].timestamp
            end_time = session_memories[-1].timestamp
            summary["duration"] = (end_time - start_time).total_seconds()
        
        # Build conversation flow
        for memory in recent_memories:
            if 'user_message' in memory.content and 'assistant_response' in memory.content:
                summary["conversation_flow"].append({
                    "timestamp": memory.timestamp.isoformat(),
                    "user_message": memory.content['user_message'][:100] + "...",
                    "assistant_response": memory.content['assistant_response'][:100] + "...",
                    "topics": list(memory.associated_topics),
                    "importance": memory.importance.value
                })
        
        # Generate key insights
        summary["key_insights"] = self._extract_key_insights(recent_memories)
        
        return summary
    
    def _generate_fragment_id(self, content: Dict[str, Any]) -> str:
        """สร้าง ID สำหรับ memory fragment"""
        content_str = json.dumps(content, sort_keys=True)
        timestamp = str(time.time())
        return hashlib.md5(f"{content_str}_{timestamp}".encode()).hexdigest()[:12]
    
    def _update_indexes(self, fragment: MemoryFragment):
        """อัปเดต indexes สำหรับการค้นหา"""
        
        # Topic index
        for topic in fragment.associated_topics:
            self.topic_index[topic].add(fragment.fragment_id)
        
        # User index
        if fragment.user_id:
            self.user_index[fragment.user_id].add(fragment.fragment_id)
        
        # Time index (by date)
        date_key = fragment.timestamp.strftime("%Y-%m-%d")
        self.time_index[date_key].add(fragment.fragment_id)
        
        # Importance index
        self.importance_index[fragment.importance].add(fragment.fragment_id)
    
    def _calculate_memory_relevance(self, memory: MemoryFragment, query: str) -> float:
        """คำนวณความเกี่ยวข้องของความทรงจำกับ query"""
        
        score = 0.0
        query_lower = query.lower()
        
        # Topic matching
        for topic in memory.associated_topics:
            if topic.lower() in query_lower:
                score += 0.4
        
        # Content matching
        content_text = str(memory.content).lower()
        query_words = query_lower.split()
        matching_words = sum(1 for word in query_words if word in content_text)
        if query_words:
            score += 0.3 * (matching_words / len(query_words))
        
        # Tag matching
        for tag in memory.context_tags:
            if tag.lower() in query_lower:
                score += 0.2
        
        # Recency bonus
        age_days = (datetime.now() - memory.timestamp).days
        recency_bonus = max(0, 0.1 - (age_days * 0.01))
        score += recency_bonus
        
        # Importance bonus
        importance_weights = {
            MemoryImportance.CRITICAL: 0.3,
            MemoryImportance.HIGH: 0.2,
            MemoryImportance.MEDIUM: 0.1,
            MemoryImportance.LOW: 0.05,
            MemoryImportance.TEMPORARY: 0.0
        }
        score += importance_weights.get(memory.importance, 0.0)
        
        # Access frequency bonus
        if memory.access_count > 0:
            score += min(0.1, memory.access_count * 0.02)
        
        return min(1.0, score)
    
    def _generate_context_summary(self, memories: List[MemoryFragment]) -> str:
        """สร้างสรุปบริบทจาก memories"""
        
        if not memories:
            return "No context available"
        
        # Extract key topics
        all_topics = set()
        for memory in memories:
            all_topics.update(memory.associated_topics)
        
        # Extract key insights
        key_points = []
        for memory in memories:
            if memory.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]:
                if 'summary' in memory.content:
                    key_points.append(memory.content['summary'])
                elif 'user_message' in memory.content:
                    key_points.append(memory.content['user_message'][:100])
        
        summary_parts = []
        if all_topics:
            summary_parts.append(f"Topics: {', '.join(list(all_topics)[:5])}")
        if key_points:
            summary_parts.append(f"Key points: {'; '.join(key_points[:3])}")
        
        return " | ".join(summary_parts) if summary_parts else "General conversation"
    
    def _calculate_context_importance(self, memories: List[MemoryFragment]) -> float:
        """คำนวณความสำคัญของบริบท"""
        
        if not memories:
            return 0.0
        
        importance_values = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TEMPORARY: 0.2
        }
        
        total_score = sum(importance_values.get(m.importance, 0.0) for m in memories)
        return total_score / len(memories)
    
    def _get_memory_by_id(self, memory_id: str) -> Optional[MemoryFragment]:
        """ดึง memory fragment จาก ID"""
        
        # Search in all memory stores
        memory_stores = [
            self.short_term_memory,
            self.long_term_memory,
            self.episodic_memory,
            self.semantic_memory
        ]
        
        for store in memory_stores:
            if memory_id in store:
                return store[memory_id]
        
        # Search in working memory
        for memory in self.working_memory:
            if memory.fragment_id == memory_id:
                return memory
        
        return None
    
    def _analyze_emotional_tone(self, memories: List[MemoryFragment]) -> Dict[str, Any]:
        """วิเคราะห์โทนอารมณ์จากความทรงจำ"""
        
        if not memories:
            return {"dominant_emotion": "neutral", "emotional_intensity": 0.0}
        
        total_weight = sum(abs(m.emotional_weight) for m in memories if m.emotional_weight != 0.0)
        positive_weight = sum(m.emotional_weight for m in memories if m.emotional_weight > 0.0)
        negative_weight = abs(sum(m.emotional_weight for m in memories if m.emotional_weight < 0.0))
        
        if total_weight == 0:
            return {"dominant_emotion": "neutral", "emotional_intensity": 0.0}
        
        if positive_weight > negative_weight:
            dominant = "positive"
            intensity = positive_weight / total_weight
        elif negative_weight > positive_weight:
            dominant = "negative"
            intensity = negative_weight / total_weight
        else:
            dominant = "mixed"
            intensity = total_weight / len(memories)
        
        return {
            "dominant_emotion": dominant,
            "emotional_intensity": intensity,
            "positive_ratio": positive_weight / total_weight if total_weight > 0 else 0,
            "negative_ratio": negative_weight / total_weight if total_weight > 0 else 0
        }
    
    def _extract_key_insights(self, memories: List[MemoryFragment]) -> List[str]:
        """สกัดข้อมูลเชิงลึกจากความทรงจำ"""
        
        insights = []
        
        # User preference insights
        preferences = {}
        for memory in memories:
            if 'user_preference' in memory.content:
                pref = memory.content['user_preference']
                if pref not in preferences:
                    preferences[pref] = 0
                preferences[pref] += 1
        
        if preferences:
            top_pref = max(preferences.items(), key=lambda x: x[1])
            insights.append(f"User preference: {top_pref[0]} (mentioned {top_pref[1]} times)")
        
        # Learning patterns
        learning_topics = []
        for memory in memories:
            if memory.memory_type == MemoryType.SEMANTIC:
                learning_topics.extend(memory.associated_topics)
        
        if learning_topics:
            from collections import Counter
            topic_counts = Counter(learning_topics)
            top_topic = topic_counts.most_common(1)[0]
            insights.append(f"Learning focus: {top_topic[0]} ({top_topic[1]} instances)")
        
        # Conversation patterns
        question_count = sum(1 for m in memories if 'question' in str(m.content).lower())
        if question_count > len(memories) * 0.6:
            insights.append("User is in exploratory/learning mode (many questions)")
        
        return insights
    
    def _consolidate_memories(self):
        """จัดระเบียบความทรงจำ - ย้ายจาก short-term เป็น long-term"""
        
        current_time = datetime.now()
        
        # Move from working memory to short-term based on importance and time
        working_to_promote = []
        for memory in list(self.working_memory):
            age = current_time - memory.timestamp
            if (age > self.working_memory_duration or 
                memory.importance in [MemoryImportance.CRITICAL, MemoryImportance.HIGH]):
                working_to_promote.append(memory)
        
        for memory in working_to_promote:
            self.working_memory.remove(memory)
            if memory.importance != MemoryImportance.TEMPORARY:
                self.short_term_memory[memory.fragment_id] = memory
                memory.memory_type = MemoryType.SHORT_TERM
        
        # Move from short-term to long-term based on criteria
        short_to_promote = []
        for memory_id, memory in self.short_term_memory.items():
            age = current_time - memory.timestamp
            
            # Criteria for long-term storage
            should_promote = (
                memory.importance == MemoryImportance.CRITICAL or
                (memory.importance == MemoryImportance.HIGH and age > self.short_term_duration) or
                (memory.access_count > 3 and memory.emotional_weight != 0.0) or
                (age > self.short_term_duration * 2 and memory.importance == MemoryImportance.MEDIUM)
            )
            
            if should_promote:
                short_to_promote.append(memory_id)
        
        for memory_id in short_to_promote:
            memory = self.short_term_memory.pop(memory_id)
            self.long_term_memory[memory_id] = memory
            memory.memory_type = MemoryType.LONG_TERM
        
        # Clean up old temporary memories
        self._cleanup_temporary_memories()
        
        # Manage memory size limits
        self._manage_memory_limits()
    
    def _cleanup_temporary_memories(self):
        """ลบความทรงจำชั่วคราวที่เก่า"""
        
        current_time = datetime.now()
        cleanup_age = timedelta(hours=1)  # Remove temporary memories after 1 hour
        
        # Cleanup from short-term memory
        to_remove = []
        for memory_id, memory in self.short_term_memory.items():
            if (memory.importance == MemoryImportance.TEMPORARY and 
                current_time - memory.timestamp > cleanup_age):
                to_remove.append(memory_id)
        
        for memory_id in to_remove:
            del self.short_term_memory[memory_id]
            self._remove_from_indexes(memory_id)
    
    def _manage_memory_limits(self):
        """จัดการขีดจำกัดของความทรงจำ"""
        
        # Limit short-term memory size
        if len(self.short_term_memory) > self.max_short_term_size:
            # Remove least important and least accessed memories
            memories_by_score = []
            for memory in self.short_term_memory.values():
                score = self._calculate_memory_retention_score(memory)
                memories_by_score.append((score, memory.fragment_id))
            
            memories_by_score.sort()  # Lowest score first
            excess_count = len(self.short_term_memory) - self.max_short_term_size
            
            for i in range(excess_count):
                _, memory_id = memories_by_score[i]
                del self.short_term_memory[memory_id]
                self._remove_from_indexes(memory_id)
        
        # Limit long-term memory size
        if len(self.long_term_memory) > self.max_long_term_size:
            # Remove only LOW importance memories that haven't been accessed recently
            to_remove = []
            for memory_id, memory in self.long_term_memory.items():
                if (memory.importance == MemoryImportance.LOW and
                    memory.access_count == 0 and
                    (datetime.now() - memory.last_accessed).days > 30):
                    to_remove.append(memory_id)
                    
                    if len(to_remove) >= len(self.long_term_memory) - self.max_long_term_size:
                        break
            
            for memory_id in to_remove:
                del self.long_term_memory[memory_id]
                self._remove_from_indexes(memory_id)
    
    def _calculate_memory_retention_score(self, memory: MemoryFragment) -> float:
        """คำนวณคะแนนการเก็บความทรงจำ (สูง = เก็บ, ต่ำ = ลบ)"""
        
        importance_scores = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.3,
            MemoryImportance.TEMPORARY: 0.1
        }
        
        base_score = importance_scores.get(memory.importance, 0.5)
        
        # Access frequency bonus
        access_bonus = min(0.3, memory.access_count * 0.05)
        
        # Recency bonus
        days_since_access = (datetime.now() - memory.last_accessed).days
        recency_bonus = max(0, 0.2 - (days_since_access * 0.01))
        
        # Emotional weight bonus
        emotional_bonus = min(0.1, abs(memory.emotional_weight) * 0.1)
        
        return base_score + access_bonus + recency_bonus + emotional_bonus
    
    def _remove_from_indexes(self, memory_id: str):
        """ลบจาก indexes ทั้งหมด"""
        
        # Remove from topic index
        for topic_fragments in self.topic_index.values():
            topic_fragments.discard(memory_id)
        
        # Remove from user index
        for user_fragments in self.user_index.values():
            user_fragments.discard(memory_id)
        
        # Remove from time index
        for time_fragments in self.time_index.values():
            time_fragments.discard(memory_id)
        
        # Remove from importance index
        for importance_fragments in self.importance_index.values():
            importance_fragments.discard(memory_id)
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติของระบบความทรงจำ"""
        
        total_memories = (len(self.working_memory) + 
                         len(self.short_term_memory) + 
                         len(self.long_term_memory) +
                         len(self.episodic_memory) +
                         len(self.semantic_memory))
        
        return {
            "total_memories": total_memories,
            "memory_distribution": {
                "working": len(self.working_memory),
                "short_term": len(self.short_term_memory),
                "long_term": len(self.long_term_memory),
                "episodic": len(self.episodic_memory),
                "semantic": len(self.semantic_memory)
            },
            "active_contexts": len(self.active_contexts),
            "topics_indexed": len(self.topic_index),
            "users_indexed": len(self.user_index),
            "importance_distribution": {
                importance.value: len(fragments)
                for importance, fragments in self.importance_index.items()
            }
        }
    
    def _save_persistent_memory(self):
        """บันทึกความทรงจำที่สำคัญ"""
        
        try:
            # Save long-term memories
            long_term_file = self.data_dir / "long_term_memory.pkl"
            with open(long_term_file, 'wb') as f:
                pickle.dump(self.long_term_memory, f)
            
            # Save episodic memories
            episodic_file = self.data_dir / "episodic_memory.pkl"
            with open(episodic_file, 'wb') as f:
                pickle.dump(self.episodic_memory, f)
            
            # Save semantic memories
            semantic_file = self.data_dir / "semantic_memory.pkl"
            with open(semantic_file, 'wb') as f:
                pickle.dump(self.semantic_memory, f)
            
            # Save indexes
            indexes_file = self.data_dir / "memory_indexes.pkl"
            indexes_data = {
                'topic_index': dict(self.topic_index),
                'user_index': dict(self.user_index),
                'time_index': dict(self.time_index),
                'importance_index': {k.value: v for k, v in self.importance_index.items()}
            }
            with open(indexes_file, 'wb') as f:
                pickle.dump(indexes_data, f)
            
            self.logger.info("Persistent memory saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save persistent memory: {e}")
    
    def _load_persistent_memory(self):
        """โหลดความทรงจำที่เก็บไว้"""
        
        try:
            # Load long-term memories
            long_term_file = self.data_dir / "long_term_memory.pkl"
            if long_term_file.exists():
                with open(long_term_file, 'rb') as f:
                    self.long_term_memory = pickle.load(f)
            
            # Load episodic memories
            episodic_file = self.data_dir / "episodic_memory.pkl"
            if episodic_file.exists():
                with open(episodic_file, 'rb') as f:
                    self.episodic_memory = pickle.load(f)
            
            # Load semantic memories
            semantic_file = self.data_dir / "semantic_memory.pkl"
            if semantic_file.exists():
                with open(semantic_file, 'rb') as f:
                    self.semantic_memory = pickle.load(f)
            
            # Load indexes
            indexes_file = self.data_dir / "memory_indexes.pkl"
            if indexes_file.exists():
                with open(indexes_file, 'rb') as f:
                    indexes_data = pickle.load(f)
                    
                    self.topic_index = defaultdict(set, indexes_data.get('topic_index', {}))
                    self.user_index = defaultdict(set, indexes_data.get('user_index', {}))
                    self.time_index = defaultdict(set, indexes_data.get('time_index', {}))
                    
                    importance_data = indexes_data.get('importance_index', {})
                    self.importance_index = defaultdict(set)
                    for importance_str, fragments in importance_data.items():
                        importance = MemoryImportance(importance_str)
                        self.importance_index[importance] = set(fragments)
            
            self.logger.info("Persistent memory loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load persistent memory: {e}")
    
    def periodic_maintenance(self):
        """การบำรุงรักษาประจำ"""
        
        current_time = time.time()
        
        if current_time - self._last_cleanup > self.cleanup_interval:
            self._consolidate_memories()
            self._save_persistent_memory()
            self._last_cleanup = current_time
            
            self.logger.debug("Periodic memory maintenance completed")
    
    def shutdown(self):
        """ปิดระบบและบันทึกข้อมูล"""
        
        self._save_persistent_memory()
        
        # Archive active contexts
        context_file = self.data_dir / "active_contexts.json"
        try:
            contexts_data = []
            for context in self.active_contexts.values():
                context_dict = asdict(context)
                context_dict['created_at'] = context.created_at.isoformat()
                context_dict['updated_at'] = context.updated_at.isoformat()
                contexts_data.append(context_dict)
            
            with open(context_file, 'w', encoding='utf-8') as f:
                json.dump(contexts_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save contexts: {e}")
        
        self.logger.info("Advanced Conversation Memory System shutdown complete")