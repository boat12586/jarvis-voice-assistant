"""
Learning System for Jarvis Voice Assistant
Handles language learning, lessons, and educational content
"""

import os
import json
import logging
import random
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


@dataclass
class LearningUnit:
    """Learning unit structure"""
    unit_id: str
    title: str
    description: str
    language: str
    difficulty: str  # beginner, intermediate, advanced
    category: str  # vocabulary, grammar, pronunciation, conversation
    content: Dict[str, Any]
    progress: float = 0.0
    completed: bool = False
    last_accessed: Optional[datetime] = None
    
    def __post_init__(self):
        if self.last_accessed is None:
            self.last_accessed = datetime.now()
        elif isinstance(self.last_accessed, str):
            self.last_accessed = datetime.fromisoformat(self.last_accessed)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['last_accessed'] = self.last_accessed.isoformat() if self.last_accessed else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningUnit':
        """Create from dictionary"""
        return cls(**data)


@dataclass
class LearningProgress:
    """Learning progress tracking"""
    user_id: str
    language: str
    total_units: int
    completed_units: int
    current_level: str
    total_time_spent: float  # minutes
    streak_days: int
    last_study_date: Optional[datetime] = None
    achievements: List[str] = None
    
    def __post_init__(self):
        if self.achievements is None:
            self.achievements = []
        if self.last_study_date is None:
            self.last_study_date = datetime.now()
        elif isinstance(self.last_study_date, str):
            self.last_study_date = datetime.fromisoformat(self.last_study_date)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['last_study_date'] = self.last_study_date.isoformat() if self.last_study_date else None
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LearningProgress':
        """Create from dictionary"""
        return cls(**data)


class LearningContent:
    """Learning content generator"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Learning content templates
        self.learning_templates = {
            "vocabulary": {
                "beginner": {
                    "thai": [
                        {
                            "title": "Basic Greetings",
                            "description": "Learn essential Thai greetings and polite expressions",
                            "words": [
                                {"thai": "สวัสดี", "english": "Hello", "pronunciation": "sa-wat-dee"},
                                {"thai": "ขอบคุณ", "english": "Thank you", "pronunciation": "kob-khun"},
                                {"thai": "ขอโทษ", "english": "Sorry", "pronunciation": "kor-tot"},
                                {"thai": "ไม่เป็นไร", "english": "It's okay", "pronunciation": "mai-pen-rai"},
                                {"thai": "ลาก่อน", "english": "Goodbye", "pronunciation": "la-gon"}
                            ]
                        },
                        {
                            "title": "Numbers 1-10",
                            "description": "Learn to count from 1 to 10 in Thai",
                            "words": [
                                {"thai": "หนึ่ง", "english": "One", "pronunciation": "neung"},
                                {"thai": "สอง", "english": "Two", "pronunciation": "song"},
                                {"thai": "สาม", "english": "Three", "pronunciation": "sam"},
                                {"thai": "สี่", "english": "Four", "pronunciation": "see"},
                                {"thai": "ห้า", "english": "Five", "pronunciation": "ha"},
                                {"thai": "หก", "english": "Six", "pronunciation": "hok"},
                                {"thai": "เจ็ด", "english": "Seven", "pronunciation": "jet"},
                                {"thai": "แปด", "english": "Eight", "pronunciation": "paet"},
                                {"thai": "เก้า", "english": "Nine", "pronunciation": "gao"},
                                {"thai": "สิบ", "english": "Ten", "pronunciation": "sip"}
                            ]
                        },
                        {
                            "title": "Family Members",
                            "description": "Learn Thai words for family members",
                            "words": [
                                {"thai": "พ่อ", "english": "Father", "pronunciation": "por"},
                                {"thai": "แม่", "english": "Mother", "pronunciation": "mae"},
                                {"thai": "ลูก", "english": "Child", "pronunciation": "luuk"},
                                {"thai": "พี่", "english": "Older sibling", "pronunciation": "pee"},
                                {"thai": "น้อง", "english": "Younger sibling", "pronunciation": "nong"},
                                {"thai": "ปู่", "english": "Grandfather (paternal)", "pronunciation": "puu"},
                                {"thai": "ย่า", "english": "Grandmother (paternal)", "pronunciation": "yaa"},
                                {"thai": "ตา", "english": "Grandfather (maternal)", "pronunciation": "taa"},
                                {"thai": "ยาย", "english": "Grandmother (maternal)", "pronunciation": "yaay"}
                            ]
                        }
                    ],
                    "english": [
                        {
                            "title": "Basic Greetings",
                            "description": "Learn essential English greetings and polite expressions",
                            "words": [
                                {"english": "Hello", "thai": "สวัสดี", "pronunciation": "heh-low"},
                                {"english": "Good morning", "thai": "อรุณสวัสดิ์", "pronunciation": "good mor-ning"},
                                {"english": "Good afternoon", "thai": "สวัสดีตอนบ่าย", "pronunciation": "good af-ter-noon"},
                                {"english": "Good evening", "thai": "สวัสดีตอนเย็น", "pronunciation": "good eve-ning"},
                                {"english": "Thank you", "thai": "ขอบคุณ", "pronunciation": "thank you"},
                                {"english": "You're welcome", "thai": "ยินดี", "pronunciation": "your wel-come"},
                                {"english": "Excuse me", "thai": "ขอโทษ", "pronunciation": "ex-cuse me"},
                                {"english": "I'm sorry", "thai": "ขอโทษ", "pronunciation": "I'm sor-ry"},
                                {"english": "Goodbye", "thai": "ลาก่อน", "pronunciation": "good-bye"}
                            ]
                        }
                    ]
                }
            },
            "grammar": {
                "beginner": {
                    "thai": [
                        {
                            "title": "Sentence Structure",
                            "description": "Learn basic Thai sentence patterns",
                            "rules": [
                                {
                                    "rule": "Subject + Verb + Object",
                                    "example_thai": "ฉัน กิน ข้าว",
                                    "example_english": "I eat rice",
                                    "explanation": "Basic Thai sentence follows Subject-Verb-Object pattern"
                                },
                                {
                                    "rule": "Polite particles",
                                    "example_thai": "ขอบคุณครับ/ค่ะ",
                                    "example_english": "Thank you (polite)",
                                    "explanation": "Add ครับ (male) or ค่ะ (female) for politeness"
                                }
                            ]
                        }
                    ],
                    "english": [
                        {
                            "title": "Present Simple Tense",
                            "description": "Learn how to form present simple sentences",
                            "rules": [
                                {
                                    "rule": "Subject + Verb + Object",
                                    "example_english": "I eat rice",
                                    "example_thai": "ฉัน กิน ข้าว",
                                    "explanation": "Use base form of verb for present simple"
                                },
                                {
                                    "rule": "Third person singular",
                                    "example_english": "She eats rice",
                                    "example_thai": "เธอ กิน ข้าว",
                                    "explanation": "Add -s or -es to verb for he/she/it"
                                }
                            ]
                        }
                    ]
                }
            },
            "conversation": {
                "beginner": {
                    "thai": [
                        {
                            "title": "At the Restaurant",
                            "description": "Learn how to order food in Thai",
                            "dialogues": [
                                {
                                    "situation": "Ordering food",
                                    "lines": [
                                        {"speaker": "Waiter", "thai": "สวัสดีครับ สั่งอะไรครับ", "english": "Hello, what would you like to order?"},
                                        {"speaker": "Customer", "thai": "ขอ ผัดไทย หนึ่งจาน", "english": "I'd like one plate of Pad Thai"},
                                        {"speaker": "Waiter", "thai": "เผ็ดมั้ยครับ", "english": "Spicy?"},
                                        {"speaker": "Customer", "thai": "เผ็ดนิดหน่อยครับ", "english": "A little spicy, please"},
                                        {"speaker": "Waiter", "thai": "ดื่มอะไรครับ", "english": "What would you like to drink?"},
                                        {"speaker": "Customer", "thai": "ขอ น้ำเปล่า หนึ่งแก้ว", "english": "I'd like one glass of water"}
                                    ]
                                }
                            ]
                        }
                    ],
                    "english": [
                        {
                            "title": "Introducing Yourself",
                            "description": "Learn how to introduce yourself in English",
                            "dialogues": [
                                {
                                    "situation": "Meeting someone new",
                                    "lines": [
                                        {"speaker": "Person A", "english": "Hello, I'm Sarah. Nice to meet you.", "thai": "สวัสดี ฉันชื่อซาร่า ยินดีที่ได้รู้จัก"},
                                        {"speaker": "Person B", "english": "Hi Sarah, I'm Tom. Nice to meet you too.", "thai": "สวัสดีซาร่า ฉันชื่อทอม ยินดีที่ได้รู้จักเหมือนกัน"},
                                        {"speaker": "Person A", "english": "Where are you from?", "thai": "คุณมาจากไหน"},
                                        {"speaker": "Person B", "english": "I'm from Thailand. How about you?", "thai": "ฉันมาจากไทย แล้วคุณล่ะ"},
                                        {"speaker": "Person A", "english": "I'm from the United States.", "thai": "ฉันมาจากสหรัฐอเมริกา"}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            },
            "pronunciation": {
                "beginner": {
                    "thai": [
                        {
                            "title": "Thai Tones",
                            "description": "Learn the 5 Thai tones and their pronunciation",
                            "tones": [
                                {
                                    "tone": "Mid tone (สามัญ)",
                                    "symbol": "ก",
                                    "description": "Flat, neutral tone",
                                    "example": "กา (crow)",
                                    "audio_hint": "Speak in a flat, even tone"
                                },
                                {
                                    "tone": "Low tone (เอก)",
                                    "symbol": "ก่",
                                    "description": "Start mid, drop to low",
                                    "example": "ก่า (branch)",
                                    "audio_hint": "Start normal, drop your voice"
                                },
                                {
                                    "tone": "Falling tone (โท)",
                                    "symbol": "ก้",
                                    "description": "Start high, fall to low",
                                    "example": "ก้า (old)",
                                    "audio_hint": "Start high, fall down like a question"
                                },
                                {
                                    "tone": "High tone (ตรี)",
                                    "symbol": "ก๊",
                                    "description": "High, sharp tone",
                                    "example": "ก๊า (gas)",
                                    "audio_hint": "High pitch, like surprise"
                                },
                                {
                                    "tone": "Rising tone (จัตวา)",
                                    "symbol": "ก๋",
                                    "description": "Start low, rise to high",
                                    "example": "ก๋า (galangal)",
                                    "audio_hint": "Start low, rise up like a question"
                                }
                            ]
                        }
                    ],
                    "english": [
                        {
                            "title": "English Sounds",
                            "description": "Learn difficult English sounds for Thai speakers",
                            "sounds": [
                                {
                                    "sound": "th",
                                    "examples": ["think", "three", "thunder"],
                                    "thai_approximation": "ซ + ส",
                                    "tip": "Put your tongue between your teeth and blow air"
                                },
                                {
                                    "sound": "v",
                                    "examples": ["very", "voice", "video"],
                                    "thai_approximation": "ฟ + ว",
                                    "tip": "Touch your top teeth to your bottom lip and vibrate"
                                },
                                {
                                    "sound": "r",
                                    "examples": ["red", "right", "rock"],
                                    "thai_approximation": "เร + รอ",
                                    "tip": "Curl your tongue back without touching the roof of your mouth"
                                }
                            ]
                        }
                    ]
                }
            }
        }
    
    def get_learning_units(self, language: str, difficulty: str, category: str) -> List[LearningUnit]:
        """Generate learning units for specified criteria"""
        units = []
        
        try:
            # Get templates for the specified criteria
            templates = self.learning_templates.get(category, {}).get(difficulty, {}).get(language, [])
            
            for i, template in enumerate(templates):
                unit_id = f"{language}_{difficulty}_{category}_{i}"
                
                unit = LearningUnit(
                    unit_id=unit_id,
                    title=template["title"],
                    description=template["description"],
                    language=language,
                    difficulty=difficulty,
                    category=category,
                    content=template
                )
                units.append(unit)
                
            self.logger.info(f"Generated {len(units)} learning units for {language} {difficulty} {category}")
            
        except Exception as e:
            self.logger.error(f"Failed to generate learning units: {e}")
        
        return units
    
    def generate_quiz(self, unit: LearningUnit, question_count: int = 5) -> Dict[str, Any]:
        """Generate a quiz for a learning unit"""
        quiz = {
            "unit_id": unit.unit_id,
            "title": f"Quiz: {unit.title}",
            "questions": []
        }
        
        try:
            if unit.category == "vocabulary":
                # Generate vocabulary quiz
                words = unit.content.get("words", [])
                if words:
                    selected_words = random.sample(words, min(question_count, len(words)))
                    
                    for word in selected_words:
                        if unit.language == "thai":
                            question = {
                                "type": "translation",
                                "question": f"What does '{word['thai']}' mean in English?",
                                "options": [word["english"]],
                                "correct_answer": word["english"],
                                "explanation": f"'{word['thai']}' ({word['pronunciation']}) means '{word['english']}'"
                            }
                        else:
                            question = {
                                "type": "translation",
                                "question": f"How do you say '{word['english']}' in Thai?",
                                "options": [word["thai"]],
                                "correct_answer": word["thai"],
                                "explanation": f"'{word['english']}' is '{word['thai']}' ({word['pronunciation']}) in Thai"
                            }
                        quiz["questions"].append(question)
                        
            elif unit.category == "grammar":
                # Generate grammar quiz
                rules = unit.content.get("rules", [])
                if rules:
                    selected_rules = random.sample(rules, min(question_count, len(rules)))
                    
                    for rule in selected_rules:
                        question = {
                            "type": "grammar",
                            "question": f"Complete the sentence: {rule['example_thai' if unit.language == 'thai' else 'example_english']}",
                            "options": [rule["rule"]],
                            "correct_answer": rule["rule"],
                            "explanation": rule["explanation"]
                        }
                        quiz["questions"].append(question)
                        
            elif unit.category == "conversation":
                # Generate conversation quiz
                dialogues = unit.content.get("dialogues", [])
                if dialogues:
                    for dialogue in dialogues[:question_count]:
                        lines = dialogue["lines"]
                        if lines:
                            random_line = random.choice(lines)
                            question = {
                                "type": "conversation",
                                "question": f"In the situation '{dialogue['situation']}', what does '{random_line['thai' if unit.language == 'thai' else 'english']}' mean?",
                                "options": [random_line['english' if unit.language == 'thai' else 'thai']],
                                "correct_answer": random_line['english' if unit.language == 'thai' else 'thai'],
                                "explanation": f"This is a common phrase used in {dialogue['situation']}"
                            }
                            quiz["questions"].append(question)
                            
            elif unit.category == "pronunciation":
                # Generate pronunciation quiz
                if unit.language == "thai":
                    tones = unit.content.get("tones", [])
                    if tones:
                        selected_tones = random.sample(tones, min(question_count, len(tones)))
                        
                        for tone in selected_tones:
                            question = {
                                "type": "pronunciation",
                                "question": f"What type of tone is '{tone['symbol']}'?",
                                "options": [tone["tone"]],
                                "correct_answer": tone["tone"],
                                "explanation": f"{tone['description']} - {tone['audio_hint']}"
                            }
                            quiz["questions"].append(question)
                else:
                    sounds = unit.content.get("sounds", [])
                    if sounds:
                        selected_sounds = random.sample(sounds, min(question_count, len(sounds)))
                        
                        for sound in selected_sounds:
                            question = {
                                "type": "pronunciation",
                                "question": f"How do you pronounce the '{sound['sound']}' sound?",
                                "options": [sound["tip"]],
                                "correct_answer": sound["tip"],
                                "explanation": f"Examples: {', '.join(sound['examples'])}"
                            }
                            quiz["questions"].append(question)
                            
        except Exception as e:
            self.logger.error(f"Failed to generate quiz: {e}")
        
        return quiz


class LearningSystem(QObject):
    """Main learning system controller"""
    
    # Signals
    lesson_ready = pyqtSignal(dict)  # lesson content
    quiz_ready = pyqtSignal(dict)  # quiz content
    progress_updated = pyqtSignal(dict)  # progress info
    achievement_unlocked = pyqtSignal(str)  # achievement name
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration
        self.supported_languages = config.get("supported_languages", ["thai", "english"])
        self.difficulty_levels = config.get("difficulty_levels", ["beginner", "intermediate", "advanced"])
        self.categories = config.get("categories", ["vocabulary", "grammar", "conversation", "pronunciation"])
        
        # Components
        self.content_generator = LearningContent()
        
        # Learning data
        self.learning_units: Dict[str, LearningUnit] = {}
        self.user_progress: Dict[str, LearningProgress] = {}
        self.current_session_start = None
        
        # Initialize
        self._initialize()
        
        self.logger.info("Learning system initialized")
    
    def _initialize(self):
        """Initialize learning system"""
        try:
            # Load user progress
            self._load_user_progress()
            
            # Generate initial learning units
            self._generate_initial_units()
            
            self.logger.info("Learning system ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize learning system: {e}")
            self.error_occurred.emit(f"Learning system initialization failed: {e}")
    
    def _generate_initial_units(self):
        """Generate initial learning units"""
        try:
            for language in self.supported_languages:
                for difficulty in self.difficulty_levels:
                    for category in self.categories:
                        units = self.content_generator.get_learning_units(language, difficulty, category)
                        
                        for unit in units:
                            self.learning_units[unit.unit_id] = unit
                            
            self.logger.info(f"Generated {len(self.learning_units)} learning units")
            
        except Exception as e:
            self.logger.error(f"Failed to generate initial units: {e}")
    
    def start_lesson(self, language: str, difficulty: str, category: str) -> Dict[str, Any]:
        """Start a learning lesson"""
        try:
            # Mark session start
            self.current_session_start = datetime.now()
            
            # Find suitable units
            suitable_units = [
                unit for unit in self.learning_units.values()
                if (unit.language == language and 
                    unit.difficulty == difficulty and 
                    unit.category == category and 
                    not unit.completed)
            ]
            
            if not suitable_units:
                # Generate new units if none available
                new_units = self.content_generator.get_learning_units(language, difficulty, category)
                if new_units:
                    suitable_units = new_units
                    # Add to learning units
                    for unit in new_units:
                        self.learning_units[unit.unit_id] = unit
                else:
                    raise ValueError(f"No learning units available for {language} {difficulty} {category}")
            
            # Select first available unit
            selected_unit = suitable_units[0]
            selected_unit.last_accessed = datetime.now()
            
            # Create lesson content
            lesson_content = {
                "unit_id": selected_unit.unit_id,
                "title": selected_unit.title,
                "description": selected_unit.description,
                "language": selected_unit.language,
                "difficulty": selected_unit.difficulty,
                "category": selected_unit.category,
                "content": selected_unit.content,
                "progress": selected_unit.progress
            }
            
            self.logger.info(f"Started lesson: {selected_unit.title}")
            
            # Emit signal
            self.lesson_ready.emit(lesson_content)
            
            return lesson_content
            
        except Exception as e:
            self.logger.error(f"Failed to start lesson: {e}")
            self.error_occurred.emit(f"Failed to start lesson: {e}")
            return None
    
    def generate_quiz(self, unit_id: str, question_count: int = 5) -> Dict[str, Any]:
        """Generate a quiz for a learning unit"""
        try:
            if unit_id not in self.learning_units:
                raise ValueError(f"Learning unit not found: {unit_id}")
            
            unit = self.learning_units[unit_id]
            quiz = self.content_generator.generate_quiz(unit, question_count)
            
            self.logger.info(f"Generated quiz for unit: {unit.title}")
            
            # Emit signal
            self.quiz_ready.emit(quiz)
            
            return quiz
            
        except Exception as e:
            self.logger.error(f"Failed to generate quiz: {e}")
            self.error_occurred.emit(f"Failed to generate quiz: {e}")
            return None
    
    def submit_quiz_results(self, unit_id: str, results: Dict[str, Any]) -> Dict[str, Any]:
        """Submit quiz results and update progress"""
        try:
            if unit_id not in self.learning_units:
                raise ValueError(f"Learning unit not found: {unit_id}")
            
            unit = self.learning_units[unit_id]
            
            # Calculate score
            total_questions = results.get("total_questions", 0)
            correct_answers = results.get("correct_answers", 0)
            score = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            
            # Update unit progress
            unit.progress = score
            if score >= 80:  # 80% threshold for completion
                unit.completed = True
            
            # Update user progress
            user_id = "default"  # In a real system, this would be user-specific
            if user_id not in self.user_progress:
                self.user_progress[user_id] = LearningProgress(
                    user_id=user_id,
                    language=unit.language,
                    total_units=len([u for u in self.learning_units.values() if u.language == unit.language]),
                    completed_units=0,
                    current_level=unit.difficulty,
                    total_time_spent=0,
                    streak_days=0
                )
            
            progress = self.user_progress[user_id]
            
            # Update completed units count
            progress.completed_units = len([
                u for u in self.learning_units.values() 
                if u.language == unit.language and u.completed
            ])
            
            # Update time spent
            if self.current_session_start:
                session_time = (datetime.now() - self.current_session_start).total_seconds() / 60
                progress.total_time_spent += session_time
            
            # Update streak
            today = datetime.now().date()
            if progress.last_study_date:
                last_study = progress.last_study_date.date()
                if today == last_study:
                    # Same day, no change
                    pass
                elif today == last_study + timedelta(days=1):
                    # Consecutive day, increase streak
                    progress.streak_days += 1
                else:
                    # Streak broken, reset
                    progress.streak_days = 1
            else:
                progress.streak_days = 1
            
            progress.last_study_date = datetime.now()
            
            # Check for achievements
            achievements = self._check_achievements(progress, unit, score)
            
            # Save progress
            self._save_user_progress()
            
            # Create response
            response = {
                "score": score,
                "passed": score >= 80,
                "progress": progress.to_dict(),
                "achievements": achievements,
                "unit_completed": unit.completed
            }
            
            self.logger.info(f"Quiz results submitted: {score}% for unit {unit.title}")
            
            # Emit signals
            self.progress_updated.emit(response)
            
            for achievement in achievements:
                self.achievement_unlocked.emit(achievement)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to submit quiz results: {e}")
            self.error_occurred.emit(f"Failed to submit quiz results: {e}")
            return None
    
    def _check_achievements(self, progress: LearningProgress, unit: LearningUnit, score: float) -> List[str]:
        """Check for new achievements"""
        achievements = []
        
        try:
            # First lesson completed
            if progress.completed_units == 1 and "first_lesson" not in progress.achievements:
                achievements.append("first_lesson")
                progress.achievements.append("first_lesson")
            
            # Perfect score
            if score == 100 and "perfect_score" not in progress.achievements:
                achievements.append("perfect_score")
                progress.achievements.append("perfect_score")
            
            # Study streak achievements
            if progress.streak_days >= 7 and "week_streak" not in progress.achievements:
                achievements.append("week_streak")
                progress.achievements.append("week_streak")
            
            if progress.streak_days >= 30 and "month_streak" not in progress.achievements:
                achievements.append("month_streak")
                progress.achievements.append("month_streak")
            
            # Time spent achievements
            if progress.total_time_spent >= 60 and "hour_spent" not in progress.achievements:  # 1 hour
                achievements.append("hour_spent")
                progress.achievements.append("hour_spent")
            
            if progress.total_time_spent >= 600 and "ten_hours_spent" not in progress.achievements:  # 10 hours
                achievements.append("ten_hours_spent")
                progress.achievements.append("ten_hours_spent")
            
            # Completion achievements
            if progress.completed_units >= 10 and "ten_lessons" not in progress.achievements:
                achievements.append("ten_lessons")
                progress.achievements.append("ten_lessons")
            
            if progress.completed_units >= progress.total_units * 0.5 and "half_complete" not in progress.achievements:
                achievements.append("half_complete")
                progress.achievements.append("half_complete")
            
            if progress.completed_units >= progress.total_units and "full_complete" not in progress.achievements:
                achievements.append("full_complete")
                progress.achievements.append("full_complete")
                
        except Exception as e:
            self.logger.error(f"Error checking achievements: {e}")
        
        return achievements
    
    def get_user_progress(self, user_id: str = "default") -> Dict[str, Any]:
        """Get user learning progress"""
        try:
            if user_id not in self.user_progress:
                return {
                    "user_id": user_id,
                    "language": "thai",
                    "total_units": 0,
                    "completed_units": 0,
                    "current_level": "beginner",
                    "total_time_spent": 0,
                    "streak_days": 0,
                    "achievements": []
                }
            
            return self.user_progress[user_id].to_dict()
            
        except Exception as e:
            self.logger.error(f"Failed to get user progress: {e}")
            return {}
    
    def get_available_lessons(self, language: str = None, difficulty: str = None, category: str = None) -> List[Dict[str, Any]]:
        """Get available lessons with filters"""
        try:
            lessons = []
            
            for unit in self.learning_units.values():
                # Apply filters
                if language and unit.language != language:
                    continue
                if difficulty and unit.difficulty != difficulty:
                    continue
                if category and unit.category != category:
                    continue
                
                lesson_info = {
                    "unit_id": unit.unit_id,
                    "title": unit.title,
                    "description": unit.description,
                    "language": unit.language,
                    "difficulty": unit.difficulty,
                    "category": unit.category,
                    "progress": unit.progress,
                    "completed": unit.completed
                }
                lessons.append(lesson_info)
            
            # Sort by progress (incomplete first, then by progress)
            lessons.sort(key=lambda x: (x["completed"], -x["progress"]))
            
            return lessons
            
        except Exception as e:
            self.logger.error(f"Failed to get available lessons: {e}")
            return []
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics"""
        try:
            total_units = len(self.learning_units)
            completed_units = len([u for u in self.learning_units.values() if u.completed])
            
            # Language distribution
            language_stats = {}
            for unit in self.learning_units.values():
                lang = unit.language
                if lang not in language_stats:
                    language_stats[lang] = {"total": 0, "completed": 0}
                language_stats[lang]["total"] += 1
                if unit.completed:
                    language_stats[lang]["completed"] += 1
            
            # Category distribution
            category_stats = {}
            for unit in self.learning_units.values():
                cat = unit.category
                if cat not in category_stats:
                    category_stats[cat] = {"total": 0, "completed": 0}
                category_stats[cat]["total"] += 1
                if unit.completed:
                    category_stats[cat]["completed"] += 1
            
            return {
                "total_units": total_units,
                "completed_units": completed_units,
                "completion_rate": (completed_units / total_units * 100) if total_units > 0 else 0,
                "supported_languages": self.supported_languages,
                "difficulty_levels": self.difficulty_levels,
                "categories": self.categories,
                "language_stats": language_stats,
                "category_stats": category_stats,
                "total_users": len(self.user_progress)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get learning stats: {e}")
            return {}
    
    def _load_user_progress(self):
        """Load user progress from file"""
        try:
            progress_file = Path(__file__).parent.parent.parent / "data" / "learning_progress.json"
            
            if progress_file.exists():
                with open(progress_file, 'r', encoding='utf-8') as f:
                    progress_data = json.load(f)
                
                for user_id, data in progress_data.items():
                    self.user_progress[user_id] = LearningProgress.from_dict(data)
                
                self.logger.info(f"Loaded progress for {len(self.user_progress)} users")
                
        except Exception as e:
            self.logger.error(f"Failed to load user progress: {e}")
    
    def _save_user_progress(self):
        """Save user progress to file"""
        try:
            progress_file = Path(__file__).parent.parent.parent / "data" / "learning_progress.json"
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            
            progress_data = {
                user_id: progress.to_dict() 
                for user_id, progress in self.user_progress.items()
            }
            
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(progress_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save user progress: {e}")
    
    def shutdown(self):
        """Shutdown learning system"""
        self.logger.info("Shutting down learning system")
        self._save_user_progress()
        self.logger.info("Learning system shutdown complete")