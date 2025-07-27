#!/usr/bin/env python3
"""
🧠 Advanced Conversation Memory System for JARVIS
ระบบความจำการสนทนาขั้นสูงที่ใช้ mxbai-embed-large และ DeepSeek-R1
"""

import logging
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime, timedelta
import sqlite3
import pickle

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    logging.warning("Sentence transformers not available")


@dataclass
class ConversationTurn:
    """หน่วยข้อมูลการสนทนาหนึ่งรอบ"""
    turn_id: str
    session_id: str
    timestamp: float
    user_input: str
    ai_response: str
    user_language: str
    intent: str
    entities: List[Dict[str, Any]]
    emotion: Optional[str] = None
    topic: Optional[str] = None
    user_satisfaction: Optional[float] = None
    response_time: Optional[float] = None
    context_used: Optional[List[str]] = None


@dataclass
class UserProfile:
    """โปรไฟล์ผู้ใช้"""
    user_id: str
    name: Optional[str]
    preferred_language: str
    interests: List[str]
    communication_style: str  # formal, casual, technical
    last_active: float
    total_conversations: int
    favorite_topics: List[str]
    learning_progress: Dict[str, Any]
    personality_traits: Dict[str, float]


@dataclass
class ConversationContext:
    """บริบทการสนทนา"""
    current_topic: Optional[str]
    topic_history: List[str]
    recent_entities: List[Dict[str, Any]]
    user_mood: Optional[str]
    conversation_flow: str  # question_answer, storytelling, problem_solving
    open_questions: List[str]
    previous_requests: List[str]


class AdvancedConversationMemory:
    """ระบบความจำการสนทนาขั้นสูง"""
    
    def __init__(self, config: Dict[str, Any] = None, data_dir: str = "data/conversation_memory"):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Database setup
        self.db_path = self.data_dir / "conversation_memory.db"
        self.embedding_model = None
        
        # Memory settings
        self.max_context_turns = self.config.get('max_context_turns', 10)
        self.max_memory_days = self.config.get('max_memory_days', 30)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.7)
        
        # Initialize components
        self._initialize_database()
        if EMBEDDING_AVAILABLE:
            self._initialize_embeddings()
        
        # Current session data
        self.current_session_id = str(uuid.uuid4())
        self.current_context = ConversationContext(
            current_topic=None,
            topic_history=[],
            recent_entities=[],
            user_mood=None,
            conversation_flow="question_answer",
            open_questions=[],
            previous_requests=[]
        )
        
        self.logger.info("🧠 Advanced Conversation Memory initialized")
    
    def _initialize_database(self):
        """เริ่มต้นฐานข้อมูล"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Conversation turns table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_turns (
                        turn_id TEXT PRIMARY KEY,
                        session_id TEXT,
                        timestamp REAL,
                        user_input TEXT,
                        ai_response TEXT,
                        user_language TEXT,
                        intent TEXT,
                        entities TEXT,
                        emotion TEXT,
                        topic TEXT,
                        user_satisfaction REAL,
                        response_time REAL,
                        context_used TEXT,
                        embedding BLOB
                    )
                ''')
                
                # User profiles table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS user_profiles (
                        user_id TEXT PRIMARY KEY,
                        name TEXT,
                        preferred_language TEXT,
                        interests TEXT,
                        communication_style TEXT,
                        last_active REAL,
                        total_conversations INTEGER,
                        favorite_topics TEXT,
                        learning_progress TEXT,
                        personality_traits TEXT
                    )
                ''')
                
                # Sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS conversation_sessions (
                        session_id TEXT PRIMARY KEY,
                        user_id TEXT,
                        start_time REAL,
                        end_time REAL,
                        total_turns INTEGER,
                        main_topics TEXT,
                        summary TEXT,
                        satisfaction_score REAL
                    )
                ''')
                
                conn.commit()
                
            self.logger.info("✅ Database initialized")
            
        except Exception as e:
            self.logger.error(f"❌ Database initialization failed: {e}")
    
    def _initialize_embeddings(self):
        """เริ่มต้นโมเดล embedding"""
        try:
            model_name = self.config.get('embedding_model', 'mixedbread-ai/mxbai-embed-large-v1')
            self.logger.info(f"🧠 Loading embedding model: {model_name}")
            
            self.embedding_model = SentenceTransformer(model_name)
            self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            
            self.logger.info(f"✅ Embedding model ready ({self.embedding_dim}D)")
            
        except Exception as e:
            self.logger.error(f"❌ Embedding model initialization failed: {e}")
            self.embedding_model = None
    
    def add_conversation_turn(
        self,
        user_input: str,
        ai_response: str,
        user_language: str = 'en',
        intent: str = 'unknown',
        entities: List[Dict[str, Any]] = None,
        emotion: str = None,
        topic: str = None,
        user_id: str = "default_user"
    ) -> str:
        """เพิ่มการสนทนาหนึ่งรอบ"""
        
        turn_id = str(uuid.uuid4())
        timestamp = time.time()
        entities = entities or []
        
        # Create conversation turn
        turn = ConversationTurn(
            turn_id=turn_id,
            session_id=self.current_session_id,
            timestamp=timestamp,
            user_input=user_input,
            ai_response=ai_response,
            user_language=user_language,
            intent=intent,
            entities=entities,
            emotion=emotion,
            topic=topic
        )
        
        # Generate embedding
        embedding = None
        if self.embedding_model:
            try:
                combined_text = f"{user_input} {ai_response}"
                embedding = self.embedding_model.encode([combined_text])[0]
            except Exception as e:
                self.logger.error(f"❌ Embedding generation failed: {e}")
        
        # Save to database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    INSERT INTO conversation_turns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    turn.turn_id,
                    turn.session_id,
                    turn.timestamp,
                    turn.user_input,
                    turn.ai_response,
                    turn.user_language,
                    turn.intent,
                    json.dumps(turn.entities),
                    turn.emotion,
                    turn.topic,
                    turn.user_satisfaction,
                    turn.response_time,
                    json.dumps(turn.context_used),
                    pickle.dumps(embedding) if embedding is not None else None
                ))
                conn.commit()
            
            # Update context
            self._update_context(turn)
            
            self.logger.debug(f"💾 Conversation turn saved: {turn_id}")
            return turn_id
            
        except Exception as e:
            self.logger.error(f"❌ Failed to save conversation turn: {e}")
            return ""
    
    def _update_context(self, turn: ConversationTurn):
        """อัพเดทบริบทการสนทนา"""
        # Update topic
        if turn.topic:
            self.current_context.current_topic = turn.topic
            if turn.topic not in self.current_context.topic_history:
                self.current_context.topic_history.append(turn.topic)
        
        # Update entities
        for entity in turn.entities:
            if entity not in self.current_context.recent_entities:
                self.current_context.recent_entities.append(entity)
        
        # Limit recent entities
        if len(self.current_context.recent_entities) > 20:
            self.current_context.recent_entities = self.current_context.recent_entities[-20:]
        
        # Update open questions
        if turn.intent == 'question' and '?' in turn.user_input:
            self.current_context.open_questions.append(turn.user_input)
        
        # Limit open questions
        if len(self.current_context.open_questions) > 5:
            self.current_context.open_questions = self.current_context.open_questions[-5:]
    
    def get_relevant_context(
        self,
        query: str,
        max_turns: int = None,
        similarity_threshold: float = None
    ) -> List[ConversationTurn]:
        """หาบริบทที่เกี่ยวข้องกับคำถาม"""
        
        max_turns = max_turns or self.max_context_turns
        similarity_threshold = similarity_threshold or self.similarity_threshold
        
        if not self.embedding_model:
            # Fallback: return recent turns
            return self._get_recent_turns(max_turns)
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            
            # Get all conversations with embeddings
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM conversation_turns 
                    WHERE embedding IS NOT NULL 
                    ORDER BY timestamp DESC 
                    LIMIT 100
                ''')
                
                rows = cursor.fetchall()
                
            relevant_turns = []
            
            for row in rows:
                try:
                    stored_embedding = pickle.loads(row[13])  # embedding column
                    similarity = np.dot(query_embedding, stored_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(stored_embedding)
                    )
                    
                    if similarity >= similarity_threshold:
                        turn = ConversationTurn(
                            turn_id=row[0],
                            session_id=row[1],
                            timestamp=row[2],
                            user_input=row[3],
                            ai_response=row[4],
                            user_language=row[5],
                            intent=row[6],
                            entities=json.loads(row[7]) if row[7] else [],
                            emotion=row[8],
                            topic=row[9],
                            user_satisfaction=row[10],
                            response_time=row[11],
                            context_used=json.loads(row[12]) if row[12] else None
                        )
                        relevant_turns.append((turn, similarity))
                        
                except Exception as e:
                    self.logger.error(f"❌ Error processing stored embedding: {e}")
                    continue
            
            # Sort by similarity and return top results
            relevant_turns.sort(key=lambda x: x[1], reverse=True)
            return [turn for turn, _ in relevant_turns[:max_turns]]
            
        except Exception as e:
            self.logger.error(f"❌ Context retrieval failed: {e}")
            return self._get_recent_turns(max_turns)
    
    def _get_recent_turns(self, max_turns: int) -> List[ConversationTurn]:
        """ดึงการสนทนาล่าสุด"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM conversation_turns 
                    ORDER BY timestamp DESC 
                    LIMIT ?
                ''', (max_turns,))
                
                rows = cursor.fetchall()
                
            turns = []
            for row in rows:
                turn = ConversationTurn(
                    turn_id=row[0],
                    session_id=row[1],
                    timestamp=row[2],
                    user_input=row[3],
                    ai_response=row[4],
                    user_language=row[5],
                    intent=row[6],
                    entities=json.loads(row[7]) if row[7] else [],
                    emotion=row[8],
                    topic=row[9],
                    user_satisfaction=row[10],
                    response_time=row[11],
                    context_used=json.loads(row[12]) if row[12] else None
                )
                turns.append(turn)
            
            return list(reversed(turns))  # Return in chronological order
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get recent turns: {e}")
            return []
    
    def get_conversation_summary(self, session_id: str = None) -> Dict[str, Any]:
        """สรุปการสนทนา"""
        session_id = session_id or self.current_session_id
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT * FROM conversation_turns 
                    WHERE session_id = ? 
                    ORDER BY timestamp
                ''', (session_id,))
                
                rows = cursor.fetchall()
            
            if not rows:
                return {"total_turns": 0}
            
            # Analyze conversation
            total_turns = len(rows)
            languages = set()
            topics = set()
            intents = []
            
            for row in rows:
                languages.add(row[5])  # user_language
                if row[9]:  # topic
                    topics.add(row[9])
                intents.append(row[6])  # intent
            
            # Calculate metrics
            start_time = rows[0][2]  # timestamp
            end_time = rows[-1][2]
            duration = end_time - start_time
            
            return {
                "session_id": session_id,
                "total_turns": total_turns,
                "duration_seconds": duration,
                "languages": list(languages),
                "topics": list(topics),
                "common_intents": list(set(intents)),
                "start_time": datetime.fromtimestamp(start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to generate summary: {e}")
            return {"error": str(e)}
    
    def get_user_insights(self, user_id: str = "default_user") -> Dict[str, Any]:
        """วิเคราะห์ผู้ใช้"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get user conversations
                cursor.execute('''
                    SELECT user_language, intent, topic, timestamp 
                    FROM conversation_turns 
                    WHERE session_id IN (
                        SELECT session_id FROM conversation_sessions 
                        WHERE user_id = ?
                    )
                    ORDER BY timestamp DESC
                    LIMIT 100
                ''', (user_id,))
                
                rows = cursor.fetchall()
            
            if not rows:
                return {"message": "No conversation data found"}
            
            # Analyze patterns
            languages = {}
            intents = {}
            topics = {}
            
            for row in rows:
                lang, intent, topic, timestamp = row
                
                languages[lang] = languages.get(lang, 0) + 1
                intents[intent] = intents.get(intent, 0) + 1
                if topic:
                    topics[topic] = topics.get(topic, 0) + 1
            
            return {
                "total_conversations": len(rows),
                "preferred_language": max(languages.items(), key=lambda x: x[1])[0] if languages else None,
                "common_intents": sorted(intents.items(), key=lambda x: x[1], reverse=True)[:5],
                "favorite_topics": sorted(topics.items(), key=lambda x: x[1], reverse=True)[:5],
                "last_active": datetime.fromtimestamp(rows[0][3]).isoformat() if rows else None
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to get user insights: {e}")
            return {"error": str(e)}
    
    def clear_old_conversations(self, days: int = None):
        """ล้างการสนทนาเก่า"""
        days = days or self.max_memory_days
        cutoff_time = time.time() - (days * 24 * 3600)
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Delete old turns
                cursor.execute('''
                    DELETE FROM conversation_turns 
                    WHERE timestamp < ?
                ''', (cutoff_time,))
                
                deleted = cursor.rowcount
                conn.commit()
                
            self.logger.info(f"🗑️ Cleaned up {deleted} old conversation turns")
            
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")


def test_conversation_memory():
    """ทดสอบระบบความจำการสนทนา"""
    print("🧪 Testing Advanced Conversation Memory...")
    
    memory = AdvancedConversationMemory()
    
    # Test adding conversations
    print("\n💬 Adding test conversations...")
    
    test_conversations = [
        ("สวัสดีครับ", "สวัสดีครับ มีอะไรให้ช่วยไหมครับ", "th", "greeting"),
        ("What time is it?", "It's currently 2:30 PM", "en", "question"),
        ("ช่วยหาข้อมูลเกี่ยวกับ AI", "ผมจะหาข้อมูลเกี่ยวกับ AI ให้ครับ", "th", "request"),
        ("Thank you", "You're welcome! Anything else I can help with?", "en", "gratitude"),
        ("ลาก่อนครับ", "ลาก่อนครับ ดีใจที่ได้ช่วยเหลือครับ", "th", "farewell")
    ]
    
    turn_ids = []
    for user_input, ai_response, language, intent in test_conversations:
        turn_id = memory.add_conversation_turn(
            user_input=user_input,
            ai_response=ai_response,
            user_language=language,
            intent=intent
        )
        turn_ids.append(turn_id)
        print(f"   ✅ Added: {user_input[:30]}...")
    
    # Test context retrieval
    print(f"\n🔍 Testing context retrieval...")
    context = memory.get_relevant_context("AI information", max_turns=3)
    print(f"   Found {len(context)} relevant turns")
    
    for turn in context:
        print(f"   - {turn.user_input[:50]}...")
    
    # Test conversation summary
    print(f"\n📊 Testing conversation summary...")
    summary = memory.get_conversation_summary()
    print(f"   Summary: {summary}")
    
    # Test user insights
    print(f"\n👤 Testing user insights...")
    insights = memory.get_user_insights()
    print(f"   Insights: {insights}")
    
    print("\n✅ Advanced Conversation Memory test completed!")


if __name__ == "__main__":
    test_conversation_memory()