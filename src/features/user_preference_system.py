"""
User Preference Memory System for JARVIS Voice Assistant
Learns and adapts to individual user preferences across emotional and communication patterns
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
class UserProfile:
    """Comprehensive user profile with preferences and patterns"""
    user_id: str
    created_date: float
    last_updated: float
    
    # Communication preferences
    preferred_personality: str
    communication_style: str  # formal, casual, friendly
    response_length_preference: str  # brief, detailed, adaptive
    language_preference: str
    
    # Emotional patterns
    dominant_emotions: Dict[str, float]  # emotion -> frequency
    stress_triggers: List[str]
    comfort_responses: List[str]
    emotional_stability_score: float
    
    # Interaction patterns
    typical_interaction_times: List[str]  # hours of day
    conversation_topics: Dict[str, float]  # topic -> frequency
    question_types: Dict[str, float]  # type -> frequency
    average_session_length: float
    
    # Learning preferences
    prefers_examples: bool
    prefers_step_by_step: bool
    technical_level: str  # beginner, intermediate, advanced
    patience_level: float  # 0-1
    
    # Cultural and contextual
    cultural_context: str
    timezone: str
    accessibility_needs: List[str]
    
    # Feedback history
    feedback_scores: List[float]
    improvement_areas: Dict[str, float]
    positive_feedback_patterns: List[str]

@dataclass
class PreferenceUpdate:
    """Record of a preference update"""
    timestamp: float
    preference_type: str
    old_value: Any
    new_value: Any
    confidence: float
    source: str  # "explicit", "implicit", "inferred"
    
class UserPreferenceSystem(QObject):
    """Advanced user preference learning and adaptation system"""
    
    # Signals
    preference_learned = pyqtSignal(str, str, dict)  # user_id, preference_type, data
    profile_updated = pyqtSignal(str, dict)  # user_id, profile_summary
    pattern_detected = pyqtSignal(str, str, dict)  # user_id, pattern_type, pattern_data
    recommendation_ready = pyqtSignal(str, dict)  # user_id, recommendations
    preference_conflict_detected = pyqtSignal(str, dict)  # user_id, conflict_info
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("user_preferences", {})
        
        # Storage
        self.preferences_dir = Path(self.config.get("preferences_dir", "data/user_preferences"))
        self.preferences_dir.mkdir(parents=True, exist_ok=True)
        
        # User profiles
        self.user_profiles: Dict[str, UserProfile] = {}
        self.preference_history: Dict[str, List[PreferenceUpdate]] = {}
        
        # Learning configuration
        self.learning_rate = self.config.get("learning_rate", 0.1)
        self.confidence_threshold = self.config.get("confidence_threshold", 0.7)
        self.pattern_detection_threshold = self.config.get("pattern_detection_threshold", 5)
        self.max_history_per_user = self.config.get("max_history_per_user", 200)
        
        # Pattern detection
        self.interaction_patterns: Dict[str, List[Dict[str, Any]]] = {}
        self.emotional_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        # Load existing profiles
        self._load_existing_profiles()
        
    def _load_existing_profiles(self):
        """Load existing user profiles from storage"""
        try:
            for profile_file in self.preferences_dir.glob("profile_*.json"):
                user_id = profile_file.stem.replace("profile_", "")
                with open(profile_file, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                
                self.user_profiles[user_id] = UserProfile(**profile_data)
                
            # Load preference history
            for history_file in self.preferences_dir.glob("history_*.json"):
                user_id = history_file.stem.replace("history_", "")
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                self.preference_history[user_id] = [
                    PreferenceUpdate(**update) for update in history_data
                ]
                
            self.logger.info(f"Loaded {len(self.user_profiles)} user profiles")
            
        except Exception as e:
            self.logger.error(f"Failed to load existing profiles: {e}")
    
    def create_or_get_user_profile(self, user_id: str, initial_context: Optional[Dict[str, Any]] = None) -> UserProfile:
        """Create a new user profile or get existing one"""
        try:
            if user_id in self.user_profiles:
                return self.user_profiles[user_id]
            
            # Create new profile with defaults
            profile = UserProfile(
                user_id=user_id,
                created_date=time.time(),
                last_updated=time.time(),
                
                # Default communication preferences
                preferred_personality="friendly",
                communication_style="adaptive",
                response_length_preference="adaptive",
                language_preference=initial_context.get("language", "en") if initial_context else "en",
                
                # Default emotional patterns
                dominant_emotions={},
                stress_triggers=[],
                comfort_responses=[],
                emotional_stability_score=0.5,
                
                # Default interaction patterns
                typical_interaction_times=[],
                conversation_topics={},
                question_types={},
                average_session_length=0.0,
                
                # Default learning preferences
                prefers_examples=True,
                prefers_step_by_step=True,
                technical_level="intermediate",
                patience_level=0.7,
                
                # Default cultural context
                cultural_context=initial_context.get("cultural_context", "general") if initial_context else "general",
                timezone=initial_context.get("timezone", "UTC") if initial_context else "UTC",
                accessibility_needs=[],
                
                # Empty feedback history
                feedback_scores=[],
                improvement_areas={},
                positive_feedback_patterns=[]
            )
            
            # Apply initial context if provided
            if initial_context:
                self._apply_initial_context(profile, initial_context)
            
            self.user_profiles[user_id] = profile
            self.preference_history[user_id] = []
            
            # Save new profile
            self._save_user_profile(user_id)
            
            self.logger.info(f"Created new user profile: {user_id}")
            return profile
            
        except Exception as e:
            self.logger.error(f"Failed to create user profile: {e}")
            return self._create_default_profile(user_id)
    
    def _apply_initial_context(self, profile: UserProfile, context: Dict[str, Any]):
        """Apply initial context to new user profile"""
        try:
            # Language preference
            if "language" in context:
                profile.language_preference = context["language"]
                
                # Adjust personality for Thai users
                if context["language"] in ["th", "thai"]:
                    profile.preferred_personality = "friendly"  # Thai culture tends to prefer warm communication
                    profile.communication_style = "polite"
            
            # Time zone for interaction timing
            if "timezone" in context:
                profile.timezone = context["timezone"]
            
            # Initial personality preference
            if "preferred_personality" in context:
                profile.preferred_personality = context["preferred_personality"]
            
            # Technical level hints
            if "technical_background" in context:
                tech_bg = context["technical_background"].lower()
                if "beginner" in tech_bg or "new" in tech_bg:
                    profile.technical_level = "beginner"
                    profile.prefers_step_by_step = True
                elif "advanced" in tech_bg or "expert" in tech_bg:
                    profile.technical_level = "advanced"
                    profile.prefers_step_by_step = False
            
        except Exception as e:
            self.logger.error(f"Failed to apply initial context: {e}")
    
    def learn_from_interaction(self, user_id: str, interaction_data: Dict[str, Any]):
        """Learn user preferences from interaction data"""
        try:
            profile = self.create_or_get_user_profile(user_id)
            
            # Extract learning signals from interaction
            self._learn_communication_preferences(profile, interaction_data)
            self._learn_emotional_patterns(profile, interaction_data)
            self._learn_interaction_patterns(profile, interaction_data)
            self._learn_content_preferences(profile, interaction_data)
            
            # Update profile timestamp
            profile.last_updated = time.time()
            
            # Save updated profile
            self._save_user_profile(user_id)
            
            # Emit signal
            profile_summary = self._get_profile_summary(profile)
            self.profile_updated.emit(user_id, profile_summary)
            
        except Exception as e:
            self.logger.error(f"Failed to learn from interaction: {e}")
    
    def _learn_communication_preferences(self, profile: UserProfile, interaction: Dict[str, Any]):
        """Learn communication style preferences"""
        try:
            user_input = interaction.get("user_input", "")
            user_language = interaction.get("user_language", "en")
            response_feedback = interaction.get("response_feedback", {})
            conversation_context = interaction.get("conversation_context", {})
            
            # Analyze user's communication style from input
            input_lower = user_input.lower()
            
            # Detect formality preference
            formal_indicators = ["please", "could you", "would you mind", "i would appreciate", "kindly"]
            casual_indicators = ["hey", "yo", "sup", "what's up", "can you", "gimme"]
            
            formal_count = sum(1 for indicator in formal_indicators if indicator in input_lower)
            casual_count = sum(1 for indicator in casual_indicators if indicator in input_lower)
            
            if formal_count > casual_count and formal_count > 0:
                self._update_preference(profile, "communication_style", "formal", 0.7, "implicit")
            elif casual_count > formal_count and casual_count > 0:
                self._update_preference(profile, "communication_style", "casual", 0.7, "implicit")
            
            # Learn response length preference from feedback
            if response_feedback:
                feedback_type = response_feedback.get("type", "")
                if feedback_type == "too_verbose":
                    self._update_preference(profile, "response_length_preference", "brief", 0.8, "explicit")
                elif feedback_type == "too_brief":
                    self._update_preference(profile, "response_length_preference", "detailed", 0.8, "explicit")
            
            # Learn language preference
            if user_language != profile.language_preference:
                # Check if this is a consistent pattern
                recent_languages = [interaction.get("user_language", "en") for interaction in 
                                  self.interaction_patterns.get(profile.user_id, [])[-5:]]
                recent_languages.append(user_language)
                
                if recent_languages.count(user_language) > len(recent_languages) * 0.6:
                    self._update_preference(profile, "language_preference", user_language, 0.9, "implicit")
            
        except Exception as e:
            self.logger.error(f"Failed to learn communication preferences: {e}")
    
    def _learn_emotional_patterns(self, profile: UserProfile, interaction: Dict[str, Any]):
        """Learn emotional patterns and triggers"""
        try:
            emotion_context = interaction.get("emotion_context", {})
            if not emotion_context:
                return
            
            primary_emotion = emotion_context.get("primary_emotion", "neutral")
            stress_level = emotion_context.get("stress_level", 0.0)
            user_input = interaction.get("user_input", "")
            
            # Update dominant emotions
            if primary_emotion in profile.dominant_emotions:
                profile.dominant_emotions[primary_emotion] += 1
            else:
                profile.dominant_emotions[primary_emotion] = 1
            
            # Detect stress triggers
            if stress_level > 0.7:
                # Extract potential triggers from input
                stress_triggers = self._extract_stress_triggers(user_input)
                for trigger in stress_triggers:
                    if trigger not in profile.stress_triggers:
                        profile.stress_triggers.append(trigger)
                        self._update_preference(profile, "stress_trigger_detected", trigger, 0.8, "inferred")
            
            # Update emotional stability score
            emotion_variance = np.var(list(profile.dominant_emotions.values())) if profile.dominant_emotions else 0
            stability = max(0.0, 1.0 - (emotion_variance / 100))  # Normalize
            profile.emotional_stability_score = (profile.emotional_stability_score * 0.9) + (stability * 0.1)
            
            # Learn comfort responses
            if primary_emotion in ["sadness", "anxiety", "frustration"]:
                response_effectiveness = interaction.get("response_effectiveness", 0.5)
                if response_effectiveness > 0.7:
                    response_text = interaction.get("assistant_response", "")
                    comfort_pattern = self._extract_comfort_pattern(response_text)
                    if comfort_pattern and comfort_pattern not in profile.comfort_responses:
                        profile.comfort_responses.append(comfort_pattern)
            
        except Exception as e:
            self.logger.error(f"Failed to learn emotional patterns: {e}")
    
    def _learn_interaction_patterns(self, profile: UserProfile, interaction: Dict[str, Any]):
        """Learn interaction timing and behavior patterns"""
        try:
            timestamp = interaction.get("timestamp", time.time())
            session_length = interaction.get("session_length", 0.0)
            
            # Track interaction times
            hour = datetime.fromtimestamp(timestamp).hour
            hour_str = f"{hour:02d}:00"
            
            if hour_str not in profile.typical_interaction_times:
                profile.typical_interaction_times.append(hour_str)
            
            # Keep only most frequent interaction times (top 8)
            if len(profile.typical_interaction_times) > 8:
                # This is simplified - in reality you'd track frequency
                profile.typical_interaction_times = profile.typical_interaction_times[-8:]
            
            # Update average session length
            if session_length > 0:
                if profile.average_session_length == 0:
                    profile.average_session_length = session_length
                else:
                    profile.average_session_length = (profile.average_session_length * 0.8) + (session_length * 0.2)
            
            # Track conversation topics
            topics = interaction.get("extracted_topics", [])
            for topic in topics:
                if topic in profile.conversation_topics:
                    profile.conversation_topics[topic] += 1
                else:
                    profile.conversation_topics[topic] = 1
            
            # Track question types
            question_type = interaction.get("question_type", "general")
            if question_type in profile.question_types:
                profile.question_types[question_type] += 1
            else:
                profile.question_types[question_type] = 1
            
        except Exception as e:
            self.logger.error(f"Failed to learn interaction patterns: {e}")
    
    def _learn_content_preferences(self, profile: UserProfile, interaction: Dict[str, Any]):
        """Learn content and learning style preferences"""
        try:
            user_input = interaction.get("user_input", "")
            response_feedback = interaction.get("response_feedback", {})
            user_behavior = interaction.get("user_behavior", {})
            
            # Detect preference for examples
            if "example" in user_input.lower() or "show me" in user_input.lower():
                profile.prefers_examples = True
                self._update_preference(profile, "prefers_examples", True, 0.8, "explicit")
            
            # Detect preference for step-by-step instructions
            if any(phrase in user_input.lower() for phrase in ["step by step", "how to", "guide me", "walk me through"]):
                profile.prefers_step_by_step = True
                self._update_preference(profile, "prefers_step_by_step", True, 0.8, "explicit")
            
            # Learn technical level from feedback
            if response_feedback:
                feedback_type = response_feedback.get("type", "")
                if feedback_type == "too_technical":
                    if profile.technical_level == "advanced":
                        profile.technical_level = "intermediate"
                    elif profile.technical_level == "intermediate":
                        profile.technical_level = "beginner"
                    self._update_preference(profile, "technical_level", profile.technical_level, 0.9, "explicit")
                
                elif feedback_type == "too_simple":
                    if profile.technical_level == "beginner":
                        profile.technical_level = "intermediate"
                    elif profile.technical_level == "intermediate":
                        profile.technical_level = "advanced"
                    self._update_preference(profile, "technical_level", profile.technical_level, 0.9, "explicit")
            
            # Learn patience level from behavior
            if user_behavior:
                interruption_count = user_behavior.get("interruptions", 0)
                quick_responses = user_behavior.get("quick_responses", 0)
                
                if interruption_count > 2 or quick_responses > 3:
                    # User seems impatient
                    profile.patience_level = max(0.1, profile.patience_level - 0.1)
                    self._update_preference(profile, "patience_level", profile.patience_level, 0.6, "inferred")
                
        except Exception as e:
            self.logger.error(f"Failed to learn content preferences: {e}")
    
    def _extract_stress_triggers(self, text: str) -> List[str]:
        """Extract potential stress triggers from user input"""
        triggers = []
        text_lower = text.lower()
        
        # Common stress-inducing situations
        stress_patterns = {
            "deadline": ["deadline", "due date", "urgent", "asap", "time pressure"],
            "technical_issues": ["error", "broken", "not working", "problem", "issue", "bug"],
            "confusion": ["confused", "don't understand", "unclear", "lost", "help"],
            "workload": ["overwhelmed", "too much", "busy", "stress", "pressure"],
            "complexity": ["complicated", "complex", "difficult", "hard", "challenging"]
        }
        
        for trigger_type, keywords in stress_patterns.items():
            if any(keyword in text_lower for keyword in keywords):
                triggers.append(trigger_type)
        
        return triggers
    
    def _extract_comfort_pattern(self, response_text: str) -> Optional[str]:
        """Extract effective comfort response patterns"""
        response_lower = response_text.lower()
        
        comfort_patterns = [
            "I understand",
            "I'm here to help",
            "Let's work through this together",
            "Don't worry",
            "Take your time",
            "Step by step"
        ]
        
        for pattern in comfort_patterns:
            if pattern.lower() in response_lower:
                return pattern
        
        return None
    
    def _update_preference(self, profile: UserProfile, preference_type: str, new_value: Any, confidence: float, source: str):
        """Update a specific preference with tracking"""
        try:
            old_value = getattr(profile, preference_type, None)
            
            # Only update if confidence is high enough or value has changed significantly
            if confidence >= self.confidence_threshold or old_value != new_value:
                setattr(profile, preference_type, new_value)
                
                # Record the update
                update = PreferenceUpdate(
                    timestamp=time.time(),
                    preference_type=preference_type,
                    old_value=old_value,
                    new_value=new_value,
                    confidence=confidence,
                    source=source
                )
                
                if profile.user_id not in self.preference_history:
                    self.preference_history[profile.user_id] = []
                
                self.preference_history[profile.user_id].append(update)
                
                # Limit history size
                if len(self.preference_history[profile.user_id]) > self.max_history_per_user:
                    self.preference_history[profile.user_id] = self.preference_history[profile.user_id][-self.max_history_per_user:]
                
                # Emit learning signal
                self.preference_learned.emit(profile.user_id, preference_type, {
                    "old_value": old_value,
                    "new_value": new_value,
                    "confidence": confidence,
                    "source": source
                })
                
        except Exception as e:
            self.logger.error(f"Failed to update preference: {e}")
    
    def get_user_recommendations(self, user_id: str) -> Dict[str, Any]:
        """Get personalized recommendations for user"""
        try:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return {}
            
            recommendations = {
                "personality": self._recommend_personality(profile),
                "communication_style": self._recommend_communication_style(profile),
                "response_adaptations": self._recommend_response_adaptations(profile),
                "interaction_timing": self._recommend_interaction_timing(profile),
                "content_style": self._recommend_content_style(profile)
            }
            
            self.recommendation_ready.emit(user_id, recommendations)
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Failed to get user recommendations: {e}")
            return {}
    
    def _recommend_personality(self, profile: UserProfile) -> str:
        """Recommend optimal personality based on user patterns"""
        # Consider emotional patterns
        if profile.emotional_stability_score < 0.4:
            return "professional"  # More stable and supportive
        
        # Consider dominant emotions
        total_emotions = sum(profile.dominant_emotions.values())
        if total_emotions > 0:
            positive_emotions = ["joy", "happiness", "excitement", "satisfaction"]
            positive_count = sum(profile.dominant_emotions.get(emotion, 0) for emotion in positive_emotions)
            
            if positive_count / total_emotions > 0.6:
                return "friendly"  # Match positive energy
            elif "stress" in profile.stress_triggers or "anxiety" in profile.dominant_emotions:
                return "professional"  # Calming and supportive
        
        return profile.preferred_personality
    
    def _recommend_communication_style(self, profile: UserProfile) -> Dict[str, Any]:
        """Recommend communication style adaptations"""
        return {
            "formality": profile.communication_style,
            "verbosity": profile.response_length_preference,
            "use_examples": profile.prefers_examples,
            "step_by_step": profile.prefers_step_by_step,
            "technical_level": profile.technical_level,
            "patience_required": profile.patience_level < 0.5
        }
    
    def _recommend_response_adaptations(self, profile: UserProfile) -> Dict[str, Any]:
        """Recommend specific response adaptations"""
        adaptations = {}
        
        # Stress management
        if profile.stress_triggers:
            adaptations["stress_management"] = {
                "triggers_to_avoid": profile.stress_triggers,
                "comfort_responses": profile.comfort_responses,
                "use_calming_tone": True
            }
        
        # Emotional support
        if profile.emotional_stability_score < 0.5:
            adaptations["emotional_support"] = {
                "increase_empathy": True,
                "provide_reassurance": True,
                "use_supportive_language": True
            }
        
        # Learning style
        adaptations["learning_style"] = {
            "provide_examples": profile.prefers_examples,
            "break_down_steps": profile.prefers_step_by_step,
            "technical_level": profile.technical_level
        }
        
        return adaptations
    
    def _recommend_interaction_timing(self, profile: UserProfile) -> Dict[str, Any]:
        """Recommend optimal interaction timing"""
        return {
            "preferred_hours": profile.typical_interaction_times,
            "average_session_length": profile.average_session_length,
            "timezone": profile.timezone
        }
    
    def _recommend_content_style(self, profile: UserProfile) -> Dict[str, Any]:
        """Recommend content style based on preferences"""
        return {
            "topics_of_interest": list(profile.conversation_topics.keys()),
            "question_types": list(profile.question_types.keys()),
            "technical_level": profile.technical_level,
            "cultural_context": profile.cultural_context
        }
    
    def provide_explicit_feedback(self, user_id: str, feedback_type: str, feedback_data: Dict[str, Any]):
        """Process explicit feedback from user"""
        try:
            profile = self.create_or_get_user_profile(user_id)
            
            # Record feedback
            feedback_score = feedback_data.get("score", 0.5)
            profile.feedback_scores.append(feedback_score)
            
            # Limit feedback history
            if len(profile.feedback_scores) > 50:
                profile.feedback_scores = profile.feedback_scores[-50:]
            
            # Process specific feedback types
            if feedback_type == "personality_preference":
                preferred_personality = feedback_data.get("preferred_personality")
                if preferred_personality:
                    self._update_preference(profile, "preferred_personality", preferred_personality, 1.0, "explicit")
            
            elif feedback_type == "communication_style":
                style = feedback_data.get("style")
                if style:
                    self._update_preference(profile, "communication_style", style, 1.0, "explicit")
            
            elif feedback_type == "response_length":
                length_pref = feedback_data.get("length_preference")
                if length_pref:
                    self._update_preference(profile, "response_length_preference", length_pref, 1.0, "explicit")
            
            elif feedback_type == "technical_level":
                tech_level = feedback_data.get("technical_level")
                if tech_level:
                    self._update_preference(profile, "technical_level", tech_level, 1.0, "explicit")
            
            # Save updated profile
            self._save_user_profile(user_id)
            
        except Exception as e:
            self.logger.error(f"Failed to process explicit feedback: {e}")
    
    def detect_preference_conflicts(self, user_id: str) -> Dict[str, Any]:
        """Detect conflicts in learned preferences"""
        try:
            if user_id not in self.preference_history:
                return {}
            
            history = self.preference_history[user_id]
            conflicts = {}
            
            # Group updates by preference type
            preference_updates = {}
            for update in history[-20:]:  # Look at recent updates
                pref_type = update.preference_type
                if pref_type not in preference_updates:
                    preference_updates[pref_type] = []
                preference_updates[pref_type].append(update)
            
            # Detect conflicts (back-and-forth changes)
            for pref_type, updates in preference_updates.items():
                if len(updates) >= 3:
                    values = [update.new_value for update in updates]
                    if len(set(values)) > 1:  # Multiple different values
                        # Check if it's oscillating
                        recent_changes = len(set(values[-3:]))
                        if recent_changes > 1:
                            conflicts[pref_type] = {
                                "type": "oscillating",
                                "values": values[-3:],
                                "confidence_scores": [update.confidence for update in updates[-3:]],
                                "sources": [update.source for update in updates[-3:]]
                            }
            
            if conflicts:
                self.preference_conflict_detected.emit(user_id, conflicts)
            
            return conflicts
            
        except Exception as e:
            self.logger.error(f"Failed to detect preference conflicts: {e}")
            return {}
    
    def _save_user_profile(self, user_id: str):
        """Save user profile to storage"""
        try:
            profile = self.user_profiles.get(user_id)
            if not profile:
                return
            
            # Save profile
            profile_file = self.preferences_dir / f"profile_{user_id}.json"
            with open(profile_file, 'w', encoding='utf-8') as f:
                json.dump(asdict(profile), f, ensure_ascii=False, indent=2)
            
            # Save preference history
            if user_id in self.preference_history:
                history_file = self.preferences_dir / f"history_{user_id}.json"
                history_data = [asdict(update) for update in self.preference_history[user_id]]
                with open(history_file, 'w', encoding='utf-8') as f:
                    json.dump(history_data, f, ensure_ascii=False, indent=2)
            
        except Exception as e:
            self.logger.error(f"Failed to save user profile: {e}")
    
    def _get_profile_summary(self, profile: UserProfile) -> Dict[str, Any]:
        """Get profile summary for external use"""
        return {
            "user_id": profile.user_id,
            "preferred_personality": profile.preferred_personality,
            "communication_style": profile.communication_style,
            "language_preference": profile.language_preference,
            "emotional_stability": profile.emotional_stability_score,
            "dominant_emotions": profile.dominant_emotions,
            "technical_level": profile.technical_level,
            "interaction_count": len(self.preference_history.get(profile.user_id, [])),
            "last_updated": profile.last_updated
        }
    
    def _create_default_profile(self, user_id: str) -> UserProfile:
        """Create a default profile as fallback"""
        return UserProfile(
            user_id=user_id,
            created_date=time.time(),
            last_updated=time.time(),
            preferred_personality="friendly",
            communication_style="adaptive",
            response_length_preference="adaptive",
            language_preference="en",
            dominant_emotions={},
            stress_triggers=[],
            comfort_responses=[],
            emotional_stability_score=0.5,
            typical_interaction_times=[],
            conversation_topics={},
            question_types={},
            average_session_length=0.0,
            prefers_examples=True,
            prefers_step_by_step=True,
            technical_level="intermediate",
            patience_level=0.7,
            cultural_context="general",
            timezone="UTC",
            accessibility_needs=[],
            feedback_scores=[],
            improvement_areas={},
            positive_feedback_patterns=[]
        )
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            total_interactions = sum(len(history) for history in self.preference_history.values())
            
            return {
                "total_users": len(self.user_profiles),
                "total_interactions": total_interactions,
                "average_interactions_per_user": total_interactions / len(self.user_profiles) if self.user_profiles else 0,
                "learning_rate": self.learning_rate,
                "confidence_threshold": self.confidence_threshold,
                "storage_directory": str(self.preferences_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}