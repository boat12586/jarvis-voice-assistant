"""
Advanced User Profile System for JARVIS
ระบบโปรไฟล์ผู้ใช้ขั้นสูง - เรียนรู้ความชอบและนิสัยผู้ใช้
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import hashlib
from enum import Enum
import numpy as np

class PreferenceType(Enum):
    """ประเภทความชอบ"""
    COMMUNICATION_STYLE = "communication_style"  # สไตล์การสื่อสาร
    TOPIC_INTEREST = "topic_interest"           # หัวข้อที่สนใจ
    RESPONSE_LENGTH = "response_length"         # ความยาวการตอบ
    LANGUAGE_PREFERENCE = "language_preference" # ภาษาที่ชอบ
    TIME_PREFERENCE = "time_preference"         # เวลาที่ชอบใช้งาน
    FORMALITY_LEVEL = "formality_level"        # ระดับความเป็นทางการ
    LEARNING_STYLE = "learning_style"          # สไตล์การเรียนรู้
    ACTIVITY_PATTERN = "activity_pattern"       # รูปแบบกิจกรรม

class UserBehaviorType(Enum):
    """ประเภทพฤติกรรมผู้ใช้"""
    QUESTION_ASKING = "question_asking"         # การถามคำถาม
    TASK_EXECUTION = "task_execution"           # การทำงาน
    LEARNING_ACTIVITY = "learning_activity"     # การเรียนรู้
    CASUAL_CHAT = "casual_chat"                # การสนทนาทั่วไป
    PROBLEM_SOLVING = "problem_solving"         # การแก้ปัญหา
    CREATIVE_WORK = "creative_work"            # งานสร้างสรรค์
    ENTERTAINMENT = "entertainment"             # ความบันเทิง

class ConfidenceLevel(Enum):
    """ระดับความเชื่อมั่น"""
    VERY_LOW = 0.0
    LOW = 0.2
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9
    CERTAIN = 1.0

@dataclass
class UserPreference:
    """ความชอบของผู้ใช้"""
    preference_id: str
    user_id: str
    preference_type: PreferenceType
    preference_key: str                 # เช่น "response_style"
    preference_value: Any               # เช่น "detailed"
    confidence_score: float             # 0.0 - 1.0
    evidence_count: int                 # จำนวนหลักฐานที่สนับสนุน
    last_observed: datetime
    first_observed: datetime
    context_tags: Set[str] = None       # บริบทที่พบความชอบนี้
    seasonal_patterns: Dict[str, float] = None  # รูปแบบตามฤดูกาล
    
    def __post_init__(self):
        if self.context_tags is None:
            self.context_tags = set()
        if self.seasonal_patterns is None:
            self.seasonal_patterns = {}

@dataclass
class UserBehaviorPattern:
    """รูปแบบพฤติกรรมผู้ใช้"""
    pattern_id: str
    user_id: str
    behavior_type: UserBehaviorType
    pattern_description: str
    frequency_data: Dict[str, int]      # เช่น {"monday": 5, "tuesday": 3}
    time_patterns: Dict[str, int]       # เช่น {"morning": 8, "evening": 12}
    context_patterns: Dict[str, int]    # เช่น {"work": 10, "home": 5}
    confidence_score: float
    last_updated: datetime
    observations_count: int
    
@dataclass
class UserInteractionHistory:
    """ประวัติการโต้ตอบของผู้ใช้"""
    interaction_id: str
    user_id: str
    interaction_type: str               # "question", "command", "conversation"
    content_summary: str
    response_satisfaction: Optional[float] = None  # 0.0 - 1.0
    topics_discussed: Set[str] = None
    language_used: str = "th"
    session_duration: Optional[float] = None
    timestamp: datetime = None
    context_data: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.topics_discussed is None:
            self.topics_discussed = set()
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.context_data is None:
            self.context_data = {}

class UserProfileSystem:
    """ระบบโปรไฟล์ผู้ใช้ขั้นสูง"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # User data storage
        self.user_profiles: Dict[str, Dict[str, Any]] = {}  # user_id -> profile_data
        self.user_preferences: Dict[str, Dict[str, UserPreference]] = defaultdict(dict)  # user_id -> preferences
        self.user_behaviors: Dict[str, Dict[str, UserBehaviorPattern]] = defaultdict(dict)  # user_id -> behaviors
        self.interaction_history: Dict[str, List[UserInteractionHistory]] = defaultdict(list)  # user_id -> interactions
        
        # Learning parameters
        self.min_evidence_threshold = config.get("min_evidence_threshold", 3)
        self.confidence_decay_rate = config.get("confidence_decay_rate", 0.95)  # Per day
        self.max_history_size = config.get("max_history_size", 1000)
        self.learning_rate = config.get("learning_rate", 0.1)
        
        # Pattern recognition
        self.pattern_analyzers: Dict[str, Any] = {}
        self.preference_detectors: Dict[str, Any] = {}
        
        # Data persistence
        self.data_dir = Path(config.get("profile_dir", "data/user_profiles"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_pattern_analyzers()
        self._initialize_preference_detectors()
        self._load_user_profiles()
        
        # Maintenance
        self._last_maintenance = time.time()
        self.maintenance_interval = config.get("maintenance_interval", 3600)  # 1 hour
        
        self.logger.info("User Profile System initialized")
    
    def _initialize_pattern_analyzers(self):
        """เริ่มต้น pattern analyzers"""
        self.pattern_analyzers = {
            "time_patterns": self._analyze_time_patterns,
            "topic_patterns": self._analyze_topic_patterns,
            "communication_patterns": self._analyze_communication_patterns,
            "learning_patterns": self._analyze_learning_patterns
        }
    
    def _initialize_preference_detectors(self):
        """เริ่มต้น preference detectors"""
        self.preference_detectors = {
            PreferenceType.COMMUNICATION_STYLE: self._detect_communication_style,
            PreferenceType.TOPIC_INTEREST: self._detect_topic_interests,
            PreferenceType.RESPONSE_LENGTH: self._detect_response_length_preference,
            PreferenceType.LANGUAGE_PREFERENCE: self._detect_language_preference,
            PreferenceType.TIME_PREFERENCE: self._detect_time_preference,
            PreferenceType.FORMALITY_LEVEL: self._detect_formality_preference,
            PreferenceType.LEARNING_STYLE: self._detect_learning_style
        }
    
    def get_or_create_user_profile(self, user_id: str) -> Dict[str, Any]:
        """ดึงหรือสร้างโปรไฟล์ผู้ใช้"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {
                "user_id": user_id,
                "created_at": datetime.now(),
                "last_updated": datetime.now(),
                "total_interactions": 0,
                "preferred_name": None,
                "demographic_data": {},
                "personality_traits": {},
                "skill_levels": {},
                "goals_and_interests": [],
                "usage_statistics": {
                    "total_sessions": 0,
                    "total_duration": 0.0,
                    "favorite_times": [],
                    "most_used_features": []
                }
            }
            self.logger.info(f"Created new user profile: {user_id}")
        
        return self.user_profiles[user_id]
    
    def record_interaction(self, user_id: str, interaction_type: str, 
                         content_summary: str, context_data: Dict[str, Any] = None,
                         topics: Set[str] = None, language: str = "th",
                         satisfaction_score: Optional[float] = None) -> str:
        """บันทึกการโต้ตอบ"""
        
        interaction_id = self._generate_interaction_id(user_id)
        
        interaction = UserInteractionHistory(
            interaction_id=interaction_id,
            user_id=user_id,
            interaction_type=interaction_type,
            content_summary=content_summary,
            response_satisfaction=satisfaction_score,
            topics_discussed=topics or set(),
            language_used=language,
            timestamp=datetime.now(),
            context_data=context_data or {}
        )
        
        # Store interaction
        self.interaction_history[user_id].append(interaction)
        
        # Maintain history size limit
        if len(self.interaction_history[user_id]) > self.max_history_size:
            self.interaction_history[user_id] = self.interaction_history[user_id][-self.max_history_size:]
        
        # Update user profile
        profile = self.get_or_create_user_profile(user_id)
        profile["total_interactions"] += 1
        profile["last_updated"] = datetime.now()
        
        # Trigger learning
        self._learn_from_interaction(interaction)
        
        self.logger.debug(f"Recorded interaction for user {user_id}: {interaction_type}")
        return interaction_id
    
    def _learn_from_interaction(self, interaction: UserInteractionHistory):
        """เรียนรู้จากการโต้ตอบ"""
        user_id = interaction.user_id
        
        # Learn preferences
        for preference_type, detector in self.preference_detectors.items():
            try:
                new_preferences = detector(interaction)
                for pref in new_preferences:
                    self._update_user_preference(user_id, pref)
            except Exception as e:
                self.logger.error(f"Error in preference detection {preference_type}: {e}")
        
        # Learn behavior patterns
        self._update_behavior_patterns(interaction)
        
        # Update profile insights
        self._update_profile_insights(user_id)
    
    def _detect_communication_style(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับสไตล์การสื่อสาร"""
        preferences = []
        content = interaction.content_summary.lower()
        
        # Detect formality level
        formal_indicators = ["ครับ", "ค่ะ", "กรุณา", "please", "thank you", "excuse me"]
        casual_indicators = ["เฮ้", "ไง", "อะ", "ป่าว", "hey", "yeah", "nah"]
        
        formal_count = sum(1 for indicator in formal_indicators if indicator in content)
        casual_count = sum(1 for indicator in casual_indicators if indicator in content)
        
        if formal_count > casual_count:
            preferences.append(UserPreference(
                preference_id=f"comm_formal_{user_id}_{int(time.time())}",
                user_id=interaction.user_id,
                preference_type=PreferenceType.FORMALITY_LEVEL,
                preference_key="formality_level",
                preference_value="formal",
                confidence_score=min(0.8, formal_count * 0.2),
                evidence_count=1,
                last_observed=interaction.timestamp,
                first_observed=interaction.timestamp
            ))
        elif casual_count > formal_count:
            preferences.append(UserPreference(
                preference_id=f"comm_casual_{user_id}_{int(time.time())}",
                user_id=interaction.user_id,
                preference_type=PreferenceType.FORMALITY_LEVEL,
                preference_key="formality_level",
                preference_value="casual",
                confidence_score=min(0.8, casual_count * 0.2),
                evidence_count=1,
                last_observed=interaction.timestamp,
                first_observed=interaction.timestamp
            ))
        
        # Detect verbosity preference
        content_length = len(interaction.content_summary)
        if content_length > 200:
            preferences.append(UserPreference(
                preference_id=f"comm_verbose_{user_id}_{int(time.time())}",
                user_id=interaction.user_id,
                preference_type=PreferenceType.RESPONSE_LENGTH,
                preference_key="preferred_length",
                preference_value="detailed",
                confidence_score=0.6,
                evidence_count=1,
                last_observed=interaction.timestamp,
                first_observed=interaction.timestamp
            ))
        elif content_length < 50:
            preferences.append(UserPreference(
                preference_id=f"comm_brief_{user_id}_{int(time.time())}",
                user_id=interaction.user_id,
                preference_type=PreferenceType.RESPONSE_LENGTH,
                preference_key="preferred_length",
                preference_value="brief",
                confidence_score=0.6,
                evidence_count=1,
                last_observed=interaction.timestamp,
                first_observed=interaction.timestamp
            ))
        
        return preferences
    
    def _detect_topic_interests(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับความสนใจในหัวข้อ"""
        preferences = []
        
        for topic in interaction.topics_discussed:
            preferences.append(UserPreference(
                preference_id=f"topic_{topic}_{interaction.user_id}_{int(time.time())}",
                user_id=interaction.user_id,
                preference_type=PreferenceType.TOPIC_INTEREST,
                preference_key="interested_topic",
                preference_value=topic,
                confidence_score=0.7,
                evidence_count=1,
                last_observed=interaction.timestamp,
                first_observed=interaction.timestamp
            ))
        
        return preferences
    
    def _detect_response_length_preference(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับความชอบด้านความยาวของการตอบ"""
        preferences = []
        
        # Analyze satisfaction vs response characteristics
        if interaction.response_satisfaction is not None:
            context_data = interaction.context_data or {}
            response_length = context_data.get("response_length", "medium")
            
            if interaction.response_satisfaction > 0.7:
                preferences.append(UserPreference(
                    preference_id=f"length_{response_length}_{interaction.user_id}_{int(time.time())}",
                    user_id=interaction.user_id,
                    preference_type=PreferenceType.RESPONSE_LENGTH,
                    preference_key="preferred_response_length",
                    preference_value=response_length,
                    confidence_score=interaction.response_satisfaction,
                    evidence_count=1,
                    last_observed=interaction.timestamp,
                    first_observed=interaction.timestamp
                ))
        
        return preferences
    
    def _detect_language_preference(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับความชอบด้านภาษา"""
        preferences = []
        
        preferences.append(UserPreference(
            preference_id=f"lang_{interaction.language_used}_{interaction.user_id}_{int(time.time())}",
            user_id=interaction.user_id,
            preference_type=PreferenceType.LANGUAGE_PREFERENCE,
            preference_key="preferred_language",
            preference_value=interaction.language_used,
            confidence_score=0.8,
            evidence_count=1,
            last_observed=interaction.timestamp,
            first_observed=interaction.timestamp
        ))
        
        return preferences
    
    def _detect_time_preference(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับความชอบด้านเวลา"""
        preferences = []
        hour = interaction.timestamp.hour
        
        time_period = "morning" if 6 <= hour < 12 else \
                     "afternoon" if 12 <= hour < 18 else \
                     "evening" if 18 <= hour < 22 else "night"
        
        preferences.append(UserPreference(
            preference_id=f"time_{time_period}_{interaction.user_id}_{int(time.time())}",
            user_id=interaction.user_id,
            preference_type=PreferenceType.TIME_PREFERENCE,
            preference_key="active_time_period",
            preference_value=time_period,
            confidence_score=0.5,
            evidence_count=1,
            last_observed=interaction.timestamp,
            first_observed=interaction.timestamp
        ))
        
        return preferences
    
    def _detect_formality_preference(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับความชอบด้านความเป็นทางการ"""
        # This is handled in _detect_communication_style
        return []
    
    def _detect_learning_style(self, interaction: UserInteractionHistory) -> List[UserPreference]:
        """ตรวจจับสไตล์การเรียนรู้"""
        preferences = []
        
        if interaction.interaction_type == "learning_activity":
            content = interaction.content_summary.lower()
            context = interaction.context_data or {}
            
            # Detect learning preferences
            if "example" in content or "ตัวอย่าง" in content:
                preferences.append(UserPreference(
                    preference_id=f"learn_examples_{interaction.user_id}_{int(time.time())}",
                    user_id=interaction.user_id,
                    preference_type=PreferenceType.LEARNING_STYLE,
                    preference_key="learning_method",
                    preference_value="examples",
                    confidence_score=0.7,
                    evidence_count=1,
                    last_observed=interaction.timestamp,
                    first_observed=interaction.timestamp
                ))
            
            if "step by step" in content or "ทีละขั้น" in content:
                preferences.append(UserPreference(
                    preference_id=f"learn_stepwise_{interaction.user_id}_{int(time.time())}",
                    user_id=interaction.user_id,
                    preference_type=PreferenceType.LEARNING_STYLE,
                    preference_key="learning_method",
                    preference_value="step_by_step",
                    confidence_score=0.7,
                    evidence_count=1,
                    last_observed=interaction.timestamp,
                    first_observed=interaction.timestamp
                ))
        
        return preferences
    
    def _update_user_preference(self, user_id: str, new_preference: UserPreference):
        """อัปเดตความชอบของผู้ใช้"""
        pref_key = f"{new_preference.preference_type.value}_{new_preference.preference_key}_{new_preference.preference_value}"
        
        if pref_key in self.user_preferences[user_id]:
            # Update existing preference
            existing_pref = self.user_preferences[user_id][pref_key]
            existing_pref.evidence_count += 1
            existing_pref.last_observed = new_preference.last_observed
            
            # Update confidence using weighted average
            weight = min(1.0, existing_pref.evidence_count / 10)  # Max weight at 10 observations
            existing_pref.confidence_score = (
                existing_pref.confidence_score * (1 - self.learning_rate) +
                new_preference.confidence_score * self.learning_rate
            )
            existing_pref.confidence_score = min(1.0, existing_pref.confidence_score + weight * 0.1)
            
            # Merge context tags
            existing_pref.context_tags.update(new_preference.context_tags)
            
        else:
            # Add new preference
            self.user_preferences[user_id][pref_key] = new_preference
        
        self.logger.debug(f"Updated preference for {user_id}: {pref_key}")
    
    def _update_behavior_patterns(self, interaction: UserInteractionHistory):
        """อัปเดตรูปแบบพฤติกรรม"""
        user_id = interaction.user_id
        
        # Time pattern
        hour = interaction.timestamp.hour
        day_of_week = interaction.timestamp.strftime("%A").lower()
        time_period = "morning" if 6 <= hour < 12 else \
                     "afternoon" if 12 <= hour < 18 else \
                     "evening" if 18 <= hour < 22 else "night"
        
        # Determine behavior type from interaction
        behavior_type = self._classify_interaction_behavior(interaction)
        
        pattern_key = f"{behavior_type.value}_pattern"
        
        if pattern_key not in self.user_behaviors[user_id]:
            # Create new behavior pattern
            self.user_behaviors[user_id][pattern_key] = UserBehaviorPattern(
                pattern_id=f"pattern_{user_id}_{behavior_type.value}_{int(time.time())}",
                user_id=user_id,
                behavior_type=behavior_type,
                pattern_description=f"{behavior_type.value} behavior pattern",
                frequency_data={day_of_week: 1},
                time_patterns={time_period: 1},
                context_patterns={},
                confidence_score=0.1,
                last_updated=interaction.timestamp,
                observations_count=1
            )
        else:
            # Update existing pattern
            pattern = self.user_behaviors[user_id][pattern_key]
            pattern.frequency_data[day_of_week] = pattern.frequency_data.get(day_of_week, 0) + 1
            pattern.time_patterns[time_period] = pattern.time_patterns.get(time_period, 0) + 1
            pattern.observations_count += 1
            pattern.last_updated = interaction.timestamp
            
            # Update confidence
            pattern.confidence_score = min(1.0, pattern.observations_count / 50.0)
    
    def _classify_interaction_behavior(self, interaction: UserInteractionHistory) -> UserBehaviorType:
        """จำแนกประเภทพฤติกรรมจากการโต้ตอบ"""
        interaction_type = interaction.interaction_type.lower()
        content = interaction.content_summary.lower()
        
        if "?" in content or "ถาม" in content or interaction_type == "question":
            return UserBehaviorType.QUESTION_ASKING
        elif "learn" in content or "เรียน" in content or interaction_type == "learning":
            return UserBehaviorType.LEARNING_ACTIVITY
        elif interaction_type in ["command", "task"]:
            return UserBehaviorType.TASK_EXECUTION
        elif "problem" in content or "ปัญหา" in content:
            return UserBehaviorType.PROBLEM_SOLVING
        elif "create" in content or "สร้าง" in content:
            return UserBehaviorType.CREATIVE_WORK
        elif interaction_type == "entertainment":
            return UserBehaviorType.ENTERTAINMENT
        else:
            return UserBehaviorType.CASUAL_CHAT
    
    def get_user_preferences(self, user_id: str, 
                           preference_type: Optional[PreferenceType] = None,
                           min_confidence: float = 0.5) -> List[UserPreference]:
        """ดึงความชอบของผู้ใช้"""
        if user_id not in self.user_preferences:
            return []
        
        preferences = []
        for pref in self.user_preferences[user_id].values():
            if pref.confidence_score >= min_confidence:
                if preference_type is None or pref.preference_type == preference_type:
                    preferences.append(pref)
        
        # Sort by confidence score
        preferences.sort(key=lambda x: x.confidence_score, reverse=True)
        return preferences
    
    def get_user_behavior_patterns(self, user_id: str,
                                 behavior_type: Optional[UserBehaviorType] = None) -> List[UserBehaviorPattern]:
        """ดึงรูปแบบพฤติกรรมของผู้ใช้"""
        if user_id not in self.user_behaviors:
            return []
        
        patterns = []
        for pattern in self.user_behaviors[user_id].values():
            if behavior_type is None or pattern.behavior_type == behavior_type:
                patterns.append(pattern)
        
        # Sort by confidence score
        patterns.sort(key=lambda x: x.confidence_score, reverse=True)
        return patterns
    
    def get_personalized_recommendations(self, user_id: str) -> Dict[str, Any]:
        """สร้างคำแนะนำเฉพาะบุคคล"""
        profile = self.get_or_create_user_profile(user_id)
        preferences = self.get_user_preferences(user_id)
        patterns = self.get_user_behavior_patterns(user_id)
        
        recommendations = {
            "communication_style": {},
            "content_preferences": {},
            "optimal_interaction_times": [],
            "suggested_features": [],
            "learning_recommendations": []
        }
        
        # Communication style recommendations
        formality_prefs = [p for p in preferences if p.preference_type == PreferenceType.FORMALITY_LEVEL]
        if formality_prefs:
            best_formality = max(formality_prefs, key=lambda x: x.confidence_score)
            recommendations["communication_style"]["formality"] = best_formality.preference_value
        
        length_prefs = [p for p in preferences if p.preference_type == PreferenceType.RESPONSE_LENGTH]
        if length_prefs:
            best_length = max(length_prefs, key=lambda x: x.confidence_score)
            recommendations["communication_style"]["response_length"] = best_length.preference_value
        
        # Language preference
        lang_prefs = [p for p in preferences if p.preference_type == PreferenceType.LANGUAGE_PREFERENCE]
        if lang_prefs:
            best_lang = max(lang_prefs, key=lambda x: x.confidence_score)
            recommendations["communication_style"]["language"] = best_lang.preference_value
        
        # Topic interests
        topic_prefs = [p for p in preferences if p.preference_type == PreferenceType.TOPIC_INTEREST]
        topic_interests = [(p.preference_value, p.confidence_score) for p in topic_prefs]
        topic_interests.sort(key=lambda x: x[1], reverse=True)
        recommendations["content_preferences"]["interested_topics"] = topic_interests[:5]
        
        # Optimal times
        time_patterns = [p for p in patterns if p.behavior_type in [UserBehaviorType.QUESTION_ASKING, UserBehaviorType.LEARNING_ACTIVITY]]
        if time_patterns:
            for pattern in time_patterns:
                best_times = sorted(pattern.time_patterns.items(), key=lambda x: x[1], reverse=True)
                recommendations["optimal_interaction_times"].extend([time for time, count in best_times[:2]])
        
        # Learning recommendations
        learning_prefs = [p for p in preferences if p.preference_type == PreferenceType.LEARNING_STYLE]
        if learning_prefs:
            for pref in learning_prefs:
                recommendations["learning_recommendations"].append({
                    "method": pref.preference_value,
                    "confidence": pref.confidence_score
                })
        
        return recommendations
    
    def _analyze_time_patterns(self, user_id: str) -> Dict[str, Any]:
        """วิเคราะห์รูปแบบการใช้งานตามเวลา"""
        interactions = self.interaction_history.get(user_id, [])
        
        if not interactions:
            return {}
        
        # Group by hour, day of week, month
        hour_counts = Counter()
        day_counts = Counter()
        month_counts = Counter()
        
        for interaction in interactions:
            hour_counts[interaction.timestamp.hour] += 1
            day_counts[interaction.timestamp.strftime("%A")] += 1
            month_counts[interaction.timestamp.month] += 1
        
        return {
            "peak_hours": hour_counts.most_common(3),
            "active_days": day_counts.most_common(3),
            "seasonal_patterns": month_counts.most_common()
        }
    
    def _analyze_topic_patterns(self, user_id: str) -> Dict[str, Any]:
        """วิเคราะห์รูปแบบหัวข้อที่สนใจ"""
        interactions = self.interaction_history.get(user_id, [])
        
        topic_counts = Counter()
        topic_trends = defaultdict(list)
        
        for interaction in interactions:
            for topic in interaction.topics_discussed:
                topic_counts[topic] += 1
                topic_trends[topic].append(interaction.timestamp)
        
        # Calculate topic trends
        trending_topics = {}
        for topic, timestamps in topic_trends.items():
            if len(timestamps) >= 3:
                # Calculate recent activity (last 7 days)
                recent_count = sum(1 for ts in timestamps if (datetime.now() - ts).days <= 7)
                trending_topics[topic] = recent_count / len(timestamps)
        
        return {
            "popular_topics": topic_counts.most_common(10),
            "trending_topics": sorted(trending_topics.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _analyze_communication_patterns(self, user_id: str) -> Dict[str, Any]:
        """วิเคราะห์รูปแบบการสื่อสาร"""
        interactions = self.interaction_history.get(user_id, [])
        
        if not interactions:
            return {}
        
        # Analyze communication characteristics
        avg_content_length = np.mean([len(i.content_summary) for i in interactions])
        language_usage = Counter([i.language_used for i in interactions])
        
        # Satisfaction patterns
        satisfaction_scores = [i.response_satisfaction for i in interactions if i.response_satisfaction is not None]
        avg_satisfaction = np.mean(satisfaction_scores) if satisfaction_scores else None
        
        return {
            "average_content_length": avg_content_length,
            "language_distribution": dict(language_usage),
            "average_satisfaction": avg_satisfaction,
            "total_interactions": len(interactions)
        }
    
    def _analyze_learning_patterns(self, user_id: str) -> Dict[str, Any]:
        """วิเคราะห์รูปแบบการเรียนรู้"""
        interactions = [i for i in self.interaction_history.get(user_id, []) 
                       if i.interaction_type in ["learning_activity", "question"]]
        
        if not interactions:
            return {}
        
        # Learning frequency
        learning_frequency = len(interactions) / max(1, len(self.interaction_history.get(user_id, [])))
        
        # Learning topics
        learning_topics = Counter()
        for interaction in interactions:
            learning_topics.update(interaction.topics_discussed)
        
        return {
            "learning_frequency": learning_frequency,
            "learning_topics": learning_topics.most_common(5),
            "learning_sessions": len(interactions)
        }
    
    def _update_profile_insights(self, user_id: str):
        """อัปเดตข้อมูลเชิงลึกของโปรไฟล์"""
        profile = self.get_or_create_user_profile(user_id)
        
        # Run pattern analyses
        time_patterns = self._analyze_time_patterns(user_id)
        topic_patterns = self._analyze_topic_patterns(user_id)
        comm_patterns = self._analyze_communication_patterns(user_id)
        learning_patterns = self._analyze_learning_patterns(user_id)
        
        # Update profile with insights
        profile["usage_statistics"].update({
            "time_patterns": time_patterns,
            "topic_patterns": topic_patterns,
            "communication_patterns": comm_patterns,
            "learning_patterns": learning_patterns
        })
        
        profile["last_updated"] = datetime.now()
    
    def _generate_interaction_id(self, user_id: str) -> str:
        """สร้าง ID สำหรับการโต้ตอบ"""
        timestamp = str(time.time())
        return hashlib.md5(f"{user_id}_{timestamp}".encode()).hexdigest()[:12]
    
    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """สร้างสรุปโปรไฟล์ผู้ใช้"""
        profile = self.get_or_create_user_profile(user_id)
        preferences = self.get_user_preferences(user_id)
        patterns = self.get_user_behavior_patterns(user_id)
        recommendations = self.get_personalized_recommendations(user_id)
        
        return {
            "user_id": user_id,
            "profile_created": profile["created_at"].isoformat(),
            "last_updated": profile["last_updated"].isoformat(),
            "total_interactions": profile["total_interactions"],
            "preferences_count": len(preferences),
            "behavior_patterns_count": len(patterns),
            "top_preferences": [
                {
                    "type": p.preference_type.value,
                    "key": p.preference_key,
                    "value": p.preference_value,
                    "confidence": p.confidence_score
                }
                for p in preferences[:5]
            ],
            "recommendations": recommendations,
            "insights": profile.get("usage_statistics", {})
        }
    
    def periodic_maintenance(self):
        """การบำรุงรักษาประจำ"""
        current_time = time.time()
        
        if current_time - self._last_maintenance > self.maintenance_interval:
            # Decay confidence scores
            self._decay_confidence_scores()
            
            # Clean old interactions
            self._clean_old_interactions()
            
            # Save profiles
            self._save_user_profiles()
            
            self._last_maintenance = current_time
            self.logger.debug("User profile maintenance completed")
    
    def _decay_confidence_scores(self):
        """ลดคะแนนความเชื่อมั่นตามเวลา"""
        for user_id, preferences in self.user_preferences.items():
            for pref in preferences.values():
                days_since_observation = (datetime.now() - pref.last_observed).days
                if days_since_observation > 0:
                    decay_factor = self.confidence_decay_rate ** days_since_observation
                    pref.confidence_score *= decay_factor
    
    def _clean_old_interactions(self):
        """ลบการโต้ตอบเก่า"""
        cutoff_date = datetime.now() - timedelta(days=365)  # Keep 1 year
        
        for user_id, interactions in self.interaction_history.items():
            old_count = len(interactions)
            interactions[:] = [i for i in interactions if i.timestamp > cutoff_date]
            new_count = len(interactions)
            
            if old_count != new_count:
                self.logger.info(f"Cleaned {old_count - new_count} old interactions for user {user_id}")
    
    def _save_user_profiles(self):
        """บันทึกโปรไฟล์ผู้ใช้"""
        try:
            # Save profiles
            profiles_file = self.data_dir / "user_profiles.json"
            profiles_data = {}
            for user_id, profile in self.user_profiles.items():
                profile_copy = profile.copy()
                profile_copy["created_at"] = profile["created_at"].isoformat()
                profile_copy["last_updated"] = profile["last_updated"].isoformat()
                profiles_data[user_id] = profile_copy
            
            with open(profiles_file, 'w', encoding='utf-8') as f:
                json.dump(profiles_data, f, ensure_ascii=False, indent=2)
            
            # Save preferences
            preferences_file = self.data_dir / "user_preferences.pkl"
            with open(preferences_file, 'wb') as f:
                pickle.dump(self.user_preferences, f)
            
            # Save behavior patterns
            behaviors_file = self.data_dir / "user_behaviors.pkl"
            with open(behaviors_file, 'wb') as f:
                pickle.dump(self.user_behaviors, f)
            
            # Save recent interactions (last 30 days only)
            recent_interactions = {}
            cutoff_date = datetime.now() - timedelta(days=30)
            for user_id, interactions in self.interaction_history.items():
                recent_interactions[user_id] = [
                    i for i in interactions if i.timestamp > cutoff_date
                ]
            
            interactions_file = self.data_dir / "recent_interactions.pkl"
            with open(interactions_file, 'wb') as f:
                pickle.dump(recent_interactions, f)
            
            self.logger.info("User profiles saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save user profiles: {e}")
    
    def _load_user_profiles(self):
        """โหลดโปรไฟล์ผู้ใช้"""
        try:
            # Load profiles
            profiles_file = self.data_dir / "user_profiles.json"
            if profiles_file.exists():
                with open(profiles_file, 'r', encoding='utf-8') as f:
                    profiles_data = json.load(f)
                
                for user_id, profile in profiles_data.items():
                    profile["created_at"] = datetime.fromisoformat(profile["created_at"])
                    profile["last_updated"] = datetime.fromisoformat(profile["last_updated"])
                    self.user_profiles[user_id] = profile
            
            # Load preferences
            preferences_file = self.data_dir / "user_preferences.pkl"
            if preferences_file.exists():
                with open(preferences_file, 'rb') as f:
                    self.user_preferences = pickle.load(f)
            
            # Load behavior patterns
            behaviors_file = self.data_dir / "user_behaviors.pkl"
            if behaviors_file.exists():
                with open(behaviors_file, 'rb') as f:
                    self.user_behaviors = pickle.load(f)
            
            # Load recent interactions
            interactions_file = self.data_dir / "recent_interactions.pkl"
            if interactions_file.exists():
                with open(interactions_file, 'rb') as f:
                    self.interaction_history = pickle.load(f)
            
            self.logger.info("User profiles loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load user profiles: {e}")
    
    def shutdown(self):
        """ปิดระบบ"""
        self._save_user_profiles()
        self.logger.info("User Profile System shutdown complete")