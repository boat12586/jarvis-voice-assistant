"""
Conversation Memory System for JARVIS Voice Assistant
Maintains context across multi-turn conversations using DeepSeek-R1 and mxbai-embed-large
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np
from datetime import datetime, timedelta

@dataclass
class ConversationTurn:
    """Single conversation turn structure"""
    timestamp: float
    user_input: str
    user_language: str
    processed_input: str
    intent: str
    entities: Dict[str, Any]
    assistant_response: str
    response_language: str
    confidence: float
    context_used: List[str]
    embedding: Optional[List[float]] = None
    session_id: str = ""
    turn_id: str = ""

@dataclass
class ConversationContext:
    """Current conversation context"""
    session_id: str
    start_time: float
    last_activity: float
    turn_count: int
    active_topics: List[str]
    user_preferences: Dict[str, Any]
    language_preference: str
    conversation_tone: str
    current_intent: str
    pending_actions: List[str]

class ConversationMemorySystem(QObject):
    """Advanced conversation memory with semantic understanding"""
    
    # Signals
    context_updated = pyqtSignal(dict)
    memory_stored = pyqtSignal(str)  # turn_id
    context_retrieved = pyqtSignal(list)  # relevant turns
    session_started = pyqtSignal(str)  # session_id
    session_ended = pyqtSignal(str)  # session_id
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("conversation_memory", {})
        
        # Memory configuration
        self.max_turns_per_session = self.config.get("max_turns_per_session", 50)
        self.max_session_duration = self.config.get("max_session_duration", 3600)  # 1 hour
        self.context_window_size = self.config.get("context_window_size", 10)
        self.similarity_threshold = self.config.get("similarity_threshold", 0.7)
        
        # Storage
        self.memory_dir = Path(self.config.get("memory_dir", "data/conversation_memory"))
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Current session data
        self.current_session: Optional[ConversationContext] = None
        self.conversation_turns: List[ConversationTurn] = []
        self.session_cache: Dict[str, List[ConversationTurn]] = {}
        
        # Embedding model integration
        self.embeddings_model = None
        self._initialize_embeddings()
        
        # Thai language integration
        self.thai_processor = None
        self._initialize_thai_support()
        
    def _initialize_embeddings(self):
        """Initialize embedding model for semantic memory"""
        try:
            from sentence_transformers import SentenceTransformer
            
            model_name = self.config.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")
            self.embeddings_model = SentenceTransformer(model_name)
            self.logger.info(f"Conversation memory embeddings initialized: {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize embeddings for conversation memory: {e}")
    
    def _initialize_thai_support(self):
        """Initialize Thai language support"""
        try:
            # Try multiple import paths to handle different execution contexts
            thai_processor = None
            
            # Try relative import first
            try:
                from .thai_language_enhanced import ThaiLanguageProcessor
                thai_processor = ThaiLanguageProcessor
            except ImportError:
                pass
            
            # Try absolute import from src
            if thai_processor is None:
                try:
                    import sys
                    import os
                    src_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'src')
                    if src_path not in sys.path:
                        sys.path.insert(0, src_path)
                    from features.thai_language_enhanced import ThaiLanguageProcessor
                    thai_processor = ThaiLanguageProcessor
                except ImportError:
                    pass
            
            if thai_processor is not None:
                self.thai_processor = thai_processor(self.config)
                self.logger.info("Thai language support for conversation memory initialized")
            else:
                raise ImportError("Could not import ThaiLanguageProcessor")
                
        except Exception as e:
            self.logger.warning(f"Thai language support not available for conversation memory: {e}")
            self.thai_processor = None
    
    def start_session(self, user_id: str = "default", language_preference: str = "en") -> str:
        """Start a new conversation session"""
        try:
            session_id = f"{user_id}_{int(time.time())}"
            
            self.current_session = ConversationContext(
                session_id=session_id,
                start_time=time.time(),
                last_activity=time.time(),
                turn_count=0,
                active_topics=[],
                user_preferences={},
                language_preference=language_preference,
                conversation_tone="neutral",
                current_intent="greeting",
                pending_actions=[]
            )
            
            self.conversation_turns = []
            
            # Load previous session data if available
            self._load_user_preferences(user_id)
            
            self.session_started.emit(session_id)
            self.logger.info(f"Started conversation session: {session_id}")
            
            return session_id
            
        except Exception as e:
            self.logger.error(f"Failed to start session: {e}")
            self.error_occurred.emit(f"Session start failed: {e}")
            return ""
    
    def add_conversation_turn(self, 
                            user_input: str,
                            user_language: str,
                            processed_input: str,
                            intent: str,
                            entities: Dict[str, Any],
                            assistant_response: str,
                            response_language: str,
                            confidence: float) -> str:
        """Add a conversation turn to memory"""
        try:
            if not self.current_session:
                # Auto-start session if none exists
                self.start_session()
            
            turn_id = f"{self.current_session.session_id}_turn_{self.current_session.turn_count + 1}"
            
            # Generate embedding for semantic search
            embedding = None
            if self.embeddings_model:
                combined_text = f"{processed_input} {assistant_response}"
                embedding = self.embeddings_model.encode(combined_text).tolist()
            
            # Get relevant context
            context_used = self._get_relevant_context(processed_input, intent)
            
            # Create conversation turn
            turn = ConversationTurn(
                timestamp=time.time(),
                user_input=user_input,
                user_language=user_language,
                processed_input=processed_input,
                intent=intent,
                entities=entities,
                assistant_response=assistant_response,
                response_language=response_language,
                confidence=confidence,
                context_used=context_used,
                embedding=embedding,
                session_id=self.current_session.session_id,
                turn_id=turn_id
            )
            
            # Add to current conversation
            self.conversation_turns.append(turn)
            
            # Update session context
            self._update_session_context(turn)
            
            # Store to disk
            self._save_turn_to_disk(turn)
            
            self.memory_stored.emit(turn_id)
            self.logger.info(f"Added conversation turn: {turn_id}")
            
            return turn_id
            
        except Exception as e:
            self.logger.error(f"Failed to add conversation turn: {e}")
            self.error_occurred.emit(f"Memory storage failed: {e}")
            return ""
    
    def get_conversation_context(self, query: str = "", max_turns: int = None) -> List[ConversationTurn]:
        """Get relevant conversation context"""
        try:
            if not max_turns:
                max_turns = self.context_window_size
            
            if not query:
                # Return recent turns
                return self.conversation_turns[-max_turns:]
            
            # Semantic search for relevant turns
            relevant_turns = []
            
            if self.embeddings_model and query:
                query_embedding = self.embeddings_model.encode(query)
                
                # Calculate similarities
                similarities = []
                for turn in self.conversation_turns:
                    if turn.embedding:
                        similarity = np.dot(query_embedding, turn.embedding) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(turn.embedding)
                        )
                        similarities.append((turn, similarity))
                
                # Sort by similarity and get top results
                similarities.sort(key=lambda x: x[1], reverse=True)
                relevant_turns = [turn for turn, sim in similarities[:max_turns] 
                                if sim > self.similarity_threshold]
            
            # If no semantic matches, fall back to recent turns
            if not relevant_turns:
                relevant_turns = self.conversation_turns[-max_turns:]
            
            self.context_retrieved.emit([asdict(turn) for turn in relevant_turns])
            return relevant_turns
            
        except Exception as e:
            self.logger.error(f"Failed to get conversation context: {e}")
            return self.conversation_turns[-max_turns:] if self.conversation_turns else []
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary"""
        if not self.current_session:
            return {}
        
        try:
            # Calculate session statistics
            total_duration = time.time() - self.current_session.start_time
            avg_response_confidence = np.mean([turn.confidence for turn in self.conversation_turns]) if self.conversation_turns else 0
            
            # Extract key topics
            all_entities = {}
            for turn in self.conversation_turns:
                for entity_type, entities in turn.entities.items():
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    all_entities[entity_type].extend(entities if isinstance(entities, list) else [entities])
            
            # Count intents
            intent_counts = {}
            for turn in self.conversation_turns:
                intent = turn.intent
                intent_counts[intent] = intent_counts.get(intent, 0) + 1
            
            summary = {
                "session_id": self.current_session.session_id,
                "duration_minutes": total_duration / 60,
                "turn_count": len(self.conversation_turns),
                "languages_used": list(set([turn.user_language for turn in self.conversation_turns])),
                "avg_confidence": avg_response_confidence,
                "primary_intents": sorted(intent_counts.items(), key=lambda x: x[1], reverse=True)[:5],
                "extracted_entities": {k: list(set(v)) for k, v in all_entities.items()},
                "active_topics": self.current_session.active_topics,
                "conversation_tone": self.current_session.conversation_tone,
                "user_preferences": self.current_session.user_preferences
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to generate session summary: {e}")
            return {"error": str(e)}
    
    def _update_session_context(self, turn: ConversationTurn):
        """Update session context based on new turn"""
        try:
            if not self.current_session:
                return
            
            # Update basic metrics
            self.current_session.last_activity = time.time()
            self.current_session.turn_count += 1
            self.current_session.current_intent = turn.intent
            
            # Extract and update topics from entities
            for entity_type, entities in turn.entities.items():
                if entity_type in ["objects", "topics", "locations"]:
                    if isinstance(entities, list):
                        for entity in entities:
                            if entity not in self.current_session.active_topics:
                                self.current_session.active_topics.append(entity)
                    else:
                        if entities not in self.current_session.active_topics:
                            self.current_session.active_topics.append(entities)
            
            # Limit active topics to prevent memory bloat
            if len(self.current_session.active_topics) > 10:
                self.current_session.active_topics = self.current_session.active_topics[-10:]
            
            # Update language preference if consistently using a language
            recent_languages = [t.user_language for t in self.conversation_turns[-5:]]
            if len(set(recent_languages)) == 1 and recent_languages[0] != self.current_session.language_preference:
                self.current_session.language_preference = recent_languages[0]
            
            # Analyze conversation tone
            politeness_indicators = ["please", "thank you", "sorry", "กรุณา", "ขอบคุณ", "ขอโทษ"]
            if any(indicator in turn.user_input.lower() for indicator in politeness_indicators):
                self.current_session.conversation_tone = "polite"
            
            # Update user preferences based on Thai context
            if self.thai_processor and turn.user_language in ["th", "thai"]:
                thai_context = self.thai_processor.enhance_for_ai_processing(turn.user_input)
                if "formality_hint" in thai_context:
                    self.current_session.user_preferences["communication_style"] = "formal"
                if "tone_hint" in thai_context:
                    self.current_session.user_preferences["politeness_level"] = "high"
            
            self.context_updated.emit(asdict(self.current_session))
            
        except Exception as e:
            self.logger.error(f"Failed to update session context: {e}")
    
    def _get_relevant_context(self, current_input: str, current_intent: str) -> List[str]:
        """Get relevant context for current turn"""
        try:
            context = []
            
            # Always include last few turns
            recent_turns = self.conversation_turns[-3:] if len(self.conversation_turns) >= 3 else self.conversation_turns
            for turn in recent_turns:
                context.append(f"Previous: {turn.processed_input} -> {turn.assistant_response}")
            
            # Add current session topics
            if self.current_session and self.current_session.active_topics:
                context.append(f"Active topics: {', '.join(self.current_session.active_topics)}")
            
            # Add intent-specific context
            if current_intent == "information_request":
                # Look for related previous questions
                for turn in reversed(self.conversation_turns):
                    if turn.intent in ["information_request", "explanation_request"] and len(context) < 5:
                        context.append(f"Related question: {turn.processed_input}")
            
            return context[:5]  # Limit context to prevent token overflow
            
        except Exception as e:
            self.logger.error(f"Failed to get relevant context: {e}")
            return []
    
    def _save_turn_to_disk(self, turn: ConversationTurn):
        """Save conversation turn to disk"""
        try:
            # Create session directory
            session_dir = self.memory_dir / turn.session_id
            session_dir.mkdir(exist_ok=True)
            
            # Save individual turn
            turn_file = session_dir / f"{turn.turn_id}.json"
            with open(turn_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(turn), f, ensure_ascii=False, indent=2)
            
            # Update session index
            self._update_session_index(turn)
            
        except Exception as e:
            self.logger.error(f"Failed to save turn to disk: {e}")
    
    def _update_session_index(self, turn: ConversationTurn):
        """Update session index file"""
        try:
            index_file = self.memory_dir / f"{turn.session_id}_index.json"
            
            # Load existing index
            index_data = {}
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
            
            # Update index
            if "turns" not in index_data:
                index_data["turns"] = []
            
            turn_summary = {
                "turn_id": turn.turn_id,
                "timestamp": turn.timestamp,
                "intent": turn.intent,
                "user_language": turn.user_language,
                "confidence": turn.confidence
            }
            
            index_data["turns"].append(turn_summary)
            index_data["last_updated"] = time.time()
            index_data["turn_count"] = len(index_data["turns"])
            
            # Save updated index
            with open(index_file, 'w', encoding='utf-8') as f:
                json.dump(index_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to update session index: {e}")
    
    def _load_user_preferences(self, user_id: str):
        """Load user preferences from previous sessions"""
        try:
            prefs_file = self.memory_dir / f"user_preferences_{user_id}.json"
            
            if prefs_file.exists():
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    preferences = json.load(f)
                
                if self.current_session:
                    self.current_session.user_preferences.update(preferences)
                    self.logger.info(f"Loaded user preferences for {user_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to load user preferences: {e}")
    
    def end_session(self):
        """End current conversation session"""
        try:
            if not self.current_session:
                return
            
            session_id = self.current_session.session_id
            
            # Save session summary
            summary = self.get_session_summary()
            summary_file = self.memory_dir / f"{session_id}_summary.json"
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            
            # Save user preferences
            user_id = session_id.split('_')[0]
            self._save_user_preferences(user_id)
            
            # Clear current session
            self.session_cache[session_id] = self.conversation_turns.copy()
            self.current_session = None
            self.conversation_turns = []
            
            self.session_ended.emit(session_id)
            self.logger.info(f"Ended conversation session: {session_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to end session: {e}")
            self.error_occurred.emit(f"Session end failed: {e}")
    
    def _save_user_preferences(self, user_id: str):
        """Save user preferences to disk"""
        try:
            if self.current_session and self.current_session.user_preferences:
                prefs_file = self.memory_dir / f"user_preferences_{user_id}.json"
                
                with open(prefs_file, 'w', encoding='utf-8') as f:
                    json.dump(self.current_session.user_preferences, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Saved user preferences for {user_id}")
                
        except Exception as e:
            self.logger.error(f"Failed to save user preferences: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics"""
        try:
            stats = {
                "current_session_active": self.current_session is not None,
                "current_turn_count": len(self.conversation_turns),
                "cached_sessions": len(self.session_cache),
                "memory_directory": str(self.memory_dir),
                "embeddings_available": self.embeddings_model is not None,
                "thai_support_available": self.thai_processor is not None,
                "max_turns_per_session": self.max_turns_per_session,
                "context_window_size": self.context_window_size
            }
            
            if self.current_session:
                stats["current_session"] = {
                    "session_id": self.current_session.session_id,
                    "duration_minutes": (time.time() - self.current_session.start_time) / 60,
                    "language_preference": self.current_session.language_preference,
                    "conversation_tone": self.current_session.conversation_tone,
                    "active_topics_count": len(self.current_session.active_topics)
                }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to get memory stats: {e}")
            return {"error": str(e)}
    
    def cleanup_old_sessions(self, days_old: int = 30):
        """Clean up old conversation sessions"""
        try:
            cutoff_time = time.time() - (days_old * 24 * 3600)
            cleaned_count = 0
            
            for session_file in self.memory_dir.glob("*_summary.json"):
                try:
                    with open(session_file, 'r', encoding='utf-8') as f:
                        summary = json.load(f)
                    
                    session_start = summary.get("session_start_time", 0)
                    if session_start < cutoff_time:
                        # Remove session files
                        session_id = summary.get("session_id", "")
                        if session_id:
                            session_dir = self.memory_dir / session_id
                            if session_dir.exists():
                                import shutil
                                shutil.rmtree(session_dir)
                            
                            # Remove summary and index
                            session_file.unlink()
                            index_file = self.memory_dir / f"{session_id}_index.json"
                            if index_file.exists():
                                index_file.unlink()
                            
                            cleaned_count += 1
                            
                except Exception as e:
                    self.logger.warning(f"Error cleaning session file {session_file}: {e}")
            
            self.logger.info(f"Cleaned up {cleaned_count} old conversation sessions")
            return cleaned_count
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup old sessions: {e}")
            return 0