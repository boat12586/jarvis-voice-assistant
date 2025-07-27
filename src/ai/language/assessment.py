"""
Advanced Language Assessment System for JARVIS
ระบบประเมินภาษาอังกฤษขั้นสูง - สร้างหลักสูตรเฉพาะบุคคล
"""

import logging
import json
import time
import re
from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import hashlib
from enum import Enum
import numpy as np
import random

class LanguageSkill(Enum):
    """ทักษะทางภาษา"""
    LISTENING = "listening"                 # ฟัง
    SPEAKING = "speaking"                   # พูด
    READING = "reading"                     # อาน
    WRITING = "writing"                     # เขียน
    VOCABULARY = "vocabulary"               # คำศัพท์
    GRAMMAR = "grammar"                     # ไวยากรณ์
    PRONUNCIATION = "pronunciation"         # การออกเสียง
    COMPREHENSION = "comprehension"         # ความเข้าใจ

class ProficiencyLevel(Enum):
    """ระดับความสามารถ"""
    BEGINNER = "beginner"                   # เริ่มต้น (A1)
    ELEMENTARY = "elementary"               # พื้นฐาน (A2)
    INTERMEDIATE = "intermediate"           # ปานกลาง (B1)
    UPPER_INTERMEDIATE = "upper_intermediate"  # ปานกลางสูง (B2)
    ADVANCED = "advanced"                   # สูง (C1)
    PROFICIENT = "proficient"              # คล่องแคล่ว (C2)

class AssessmentType(Enum):
    """ประเภทการประเมิน"""
    PLACEMENT = "placement"                 # จัดระดับ
    DIAGNOSTIC = "diagnostic"               # วินิจฉัย
    PROGRESS = "progress"                   # ความก้าวหน้า
    ACHIEVEMENT = "achievement"             # ผลสัมฤทธิ์
    ADAPTIVE = "adaptive"                   # ปรับตัว

class QuestionType(Enum):
    """ประเภทคำถาม"""
    MULTIPLE_CHOICE = "multiple_choice"     # เลือกตอบ
    FILL_BLANK = "fill_blank"              # เติมคำ
    TRUE_FALSE = "true_false"              # จริง/เท็จ
    MATCHING = "matching"                   # จับคู่
    ORDERING = "ordering"                   # เรียงลำดับ
    OPEN_ENDED = "open_ended"              # อิสระ
    AUDIO_RESPONSE = "audio_response"       # ตอบด้วยเสียง
    CONVERSATION = "conversation"           # สนทนา

@dataclass
class AssessmentQuestion:
    """คำถามประเมิน"""
    question_id: str
    skill: LanguageSkill
    level: ProficiencyLevel
    question_type: QuestionType
    question_text: str
    options: Optional[List[str]] = None     # สำหรับ multiple choice
    correct_answers: List[str] = None       # คำตอบที่ถูก
    explanation: str = ""                   # คำอธิบาย
    points: int = 1                        # คะแนน
    time_limit: Optional[int] = None        # เวลาจำกัด (วินาที)
    difficulty_score: float = 0.5          # ความยาก 0-1
    tags: Set[str] = None                  # แท็ก
    audio_prompt: Optional[str] = None      # เสียงคำถาม
    image_prompt: Optional[str] = None      # รูปภาพคำถาม
    
    def __post_init__(self):
        if self.correct_answers is None:
            self.correct_answers = []
        if self.tags is None:
            self.tags = set()

@dataclass
class AssessmentResponse:
    """คำตอบของผู้เรียน"""
    response_id: str
    question_id: str
    user_id: str
    user_answer: Any                        # คำตอบผู้ใช้
    is_correct: bool
    score: float                           # คะแนนที่ได้ (0-1)
    response_time: float                   # เวลาใช้ตอบ (วินาที)
    timestamp: datetime
    confidence_level: Optional[float] = None  # ความมั่นใจ (0-1)
    attempt_number: int = 1                # ครั้งที่ตอบ
    feedback: str = ""                     # ข้อเสนอแนะ
    analysis: Dict[str, Any] = None        # การวิเคราะห์คำตอบ
    
    def __post_init__(self):
        if self.analysis is None:
            self.analysis = {}

@dataclass
class SkillAssessment:
    """ผลการประเมินทักษะ"""
    skill: LanguageSkill
    current_level: ProficiencyLevel
    confidence_score: float                 # ความมั่นใจในการประเมิน (0-1)
    strengths: List[str]                   # จุดแข็ง
    weaknesses: List[str]                  # จุดอัอน
    recommended_level: ProficiencyLevel    # ระดับที่แนะนำ
    next_milestones: List[str]             # เป้าหมายต่อไป
    estimated_study_hours: int             # ชั่วโมงเรียนที่แนะนำ

@dataclass
class LearningPath:
    """เส้นทางการเรียน"""
    path_id: str
    user_id: str
    target_level: ProficiencyLevel
    current_level: ProficiencyLevel
    focus_skills: List[LanguageSkill]      # ทักษะที่เน้น
    learning_objectives: List[str]         # วัตถุประสงค์
    recommended_activities: List[Dict[str, Any]]  # กิจกรรมแนะนำ
    estimated_duration: int                # ระยะเวลา (วัน)
    progress_milestones: List[Dict[str, Any]]  # เป้าหมายระหว่างทาง
    created_at: datetime
    updated_at: datetime

class LanguageAssessment:
    """ระบบประเมินภาษาขั้นสูง"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Question banks
        self.question_banks: Dict[LanguageSkill, Dict[ProficiencyLevel, List[AssessmentQuestion]]] = {}
        self.questions: Dict[str, AssessmentQuestion] = {}  # question_id -> question
        
        # Assessment data
        self.user_responses: Dict[str, List[AssessmentResponse]] = defaultdict(list)  # user_id -> responses
        self.user_assessments: Dict[str, Dict[LanguageSkill, SkillAssessment]] = defaultdict(dict)  # user_id -> skills
        self.learning_paths: Dict[str, LearningPath] = {}  # path_id -> path
        
        # Assessment algorithms
        self.scoring_algorithms: Dict[QuestionType, callable] = {}
        self.level_calculators: Dict[LanguageSkill, callable] = {}
        
        # Adaptive testing
        self.adaptive_engines: Dict[str, Any] = {}  # user_id -> engine state
        
        # Data persistence
        self.data_dir = Path(config.get("assessment_dir", "data/language_assessment"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self._initialize_scoring_algorithms()
        self._initialize_level_calculators()
        self._load_question_banks()
        self._load_assessment_data()
        
        self.logger.info("Language Assessment System initialized")
    
    def _initialize_scoring_algorithms(self):
        """เริ่มต้นอัลกอริทึมการให้คะแนน"""
        self.scoring_algorithms = {
            QuestionType.MULTIPLE_CHOICE: self._score_multiple_choice,
            QuestionType.FILL_BLANK: self._score_fill_blank,
            QuestionType.TRUE_FALSE: self._score_true_false,
            QuestionType.MATCHING: self._score_matching,
            QuestionType.ORDERING: self._score_ordering,
            QuestionType.OPEN_ENDED: self._score_open_ended,
            QuestionType.AUDIO_RESPONSE: self._score_audio_response,
            QuestionType.CONVERSATION: self._score_conversation
        }
    
    def _initialize_level_calculators(self):
        """เริ่มต้นตัวคำนวณระดับ"""
        self.level_calculators = {
            LanguageSkill.LISTENING: self._calculate_listening_level,
            LanguageSkill.SPEAKING: self._calculate_speaking_level,
            LanguageSkill.READING: self._calculate_reading_level,
            LanguageSkill.WRITING: self._calculate_writing_level,
            LanguageSkill.VOCABULARY: self._calculate_vocabulary_level,
            LanguageSkill.GRAMMAR: self._calculate_grammar_level,
            LanguageSkill.PRONUNCIATION: self._calculate_pronunciation_level,
            LanguageSkill.COMPREHENSION: self._calculate_comprehension_level
        }
    
    def create_placement_assessment(self, user_id: str, target_skills: List[LanguageSkill]) -> str:
        """สร้างการประเมินเพื่อจัดระดับ"""
        assessment_id = f"placement_{user_id}_{int(time.time())}"
        
        # Select questions across multiple levels
        assessment_questions = []
        
        for skill in target_skills:
            # Start with intermediate level questions
            questions_per_level = 3
            
            for level in ProficiencyLevel:
                if skill in self.question_banks and level in self.question_banks[skill]:
                    available_questions = self.question_banks[skill][level]
                    selected = random.sample(
                        available_questions, 
                        min(questions_per_level, len(available_questions))
                    )
                    assessment_questions.extend(selected)
        
        # Randomize question order
        random.shuffle(assessment_questions)
        
        # Initialize adaptive engine for this user
        self.adaptive_engines[user_id] = {
            "assessment_id": assessment_id,
            "assessment_type": AssessmentType.PLACEMENT,
            "target_skills": target_skills,
            "questions": [q.question_id for q in assessment_questions],
            "current_question_index": 0,
            "responses": [],
            "estimated_levels": {skill: ProficiencyLevel.INTERMEDIATE for skill in target_skills},
            "confidence_scores": {skill: 0.0 for skill in target_skills}
        }
        
        self.logger.info(f"Created placement assessment for {user_id}: {len(assessment_questions)} questions")
        return assessment_id
    
    def get_next_question(self, user_id: str) -> Optional[AssessmentQuestion]:
        """ดึงคำถามถัดไป"""
        if user_id not in self.adaptive_engines:
            return None
        
        engine = self.adaptive_engines[user_id]
        
        if engine["current_question_index"] >= len(engine["questions"]):
            return None  # Assessment completed
        
        question_id = engine["questions"][engine["current_question_index"]]
        return self.questions.get(question_id)
    
    def submit_response(self, user_id: str, question_id: str, user_answer: Any,
                       response_time: float, confidence_level: Optional[float] = None) -> AssessmentResponse:
        """ส่งคำตอบ"""
        if question_id not in self.questions:
            raise ValueError(f"Question not found: {question_id}")
        
        question = self.questions[question_id]
        
        # Score the response
        score, is_correct, analysis = self._score_response(question, user_answer)
        
        # Create response record
        response_id = f"resp_{user_id}_{question_id}_{int(time.time())}"
        
        response = AssessmentResponse(
            response_id=response_id,
            question_id=question_id,
            user_id=user_id,
            user_answer=user_answer,
            is_correct=is_correct,
            score=score,
            response_time=response_time,
            timestamp=datetime.now(),
            confidence_level=confidence_level,
            analysis=analysis
        )
        
        # Store response
        self.user_responses[user_id].append(response)
        
        # Update adaptive engine
        if user_id in self.adaptive_engines:
            engine = self.adaptive_engines[user_id]
            engine["responses"].append(response)
            engine["current_question_index"] += 1
            
            # Update level estimates
            self._update_level_estimates(user_id, response)
        
        # Generate feedback
        response.feedback = self._generate_feedback(question, response)
        
        self.logger.debug(f"Response submitted: {response_id}")
        return response
    
    def _score_response(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำตอบ"""
        scoring_func = self.scoring_algorithms.get(question.question_type)
        
        if not scoring_func:
            return 0.0, False, {"error": "No scoring algorithm for this question type"}
        
        return scoring_func(question, user_answer)
    
    def _score_multiple_choice(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำถามเลือกตอบ"""
        if not question.correct_answers:
            return 0.0, False, {"error": "No correct answers defined"}
        
        user_str = str(user_answer).strip().lower()
        correct_answers = [str(ans).strip().lower() for ans in question.correct_answers]
        
        is_correct = user_str in correct_answers
        score = 1.0 if is_correct else 0.0
        
        analysis = {
            "user_answer": user_answer,
            "correct_answers": question.correct_answers,
            "is_exact_match": user_str in correct_answers
        }
        
        return score, is_correct, analysis
    
    def _score_fill_blank(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำถามเติมคำ"""
        if not question.correct_answers:
            return 0.0, False, {"error": "No correct answers defined"}
        
        user_str = str(user_answer).strip().lower()
        
        # Check for exact matches
        exact_matches = []
        partial_matches = []
        
        for correct_answer in question.correct_answers:
            correct_str = str(correct_answer).strip().lower()
            
            if user_str == correct_str:
                exact_matches.append(correct_answer)
            elif user_str in correct_str or correct_str in user_str:
                partial_matches.append(correct_answer)
        
        if exact_matches:
            score = 1.0
            is_correct = True
        elif partial_matches:
            score = 0.5
            is_correct = False
        else:
            score = 0.0
            is_correct = False
        
        analysis = {
            "user_answer": user_answer,
            "exact_matches": exact_matches,
            "partial_matches": partial_matches,
            "score_breakdown": {
                "exact": 1.0 if exact_matches else 0.0,
                "partial": 0.5 if partial_matches else 0.0
            }
        }
        
        return score, is_correct, analysis
    
    def _score_true_false(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำถามจริง/เท็จ"""
        if not question.correct_answers:
            return 0.0, False, {"error": "No correct answers defined"}
        
        # Normalize user answer
        user_bool = None
        user_str = str(user_answer).strip().lower()
        
        if user_str in ["true", "t", "1", "yes", "y", "จริง", "ใช่"]:
            user_bool = True
        elif user_str in ["false", "f", "0", "no", "n", "เท็จ", "ไม่ใช่"]:
            user_bool = False
        
        # Check against correct answers
        correct_bool = None
        correct_str = str(question.correct_answers[0]).strip().lower()
        
        if correct_str in ["true", "t", "1", "yes", "y", "จริง", "ใช่"]:
            correct_bool = True
        elif correct_str in ["false", "f", "0", "no", "n", "เท็จ", "ไม่ใช่"]:
            correct_bool = False
        
        is_correct = user_bool is not None and user_bool == correct_bool
        score = 1.0 if is_correct else 0.0
        
        analysis = {
            "user_answer": user_answer,
            "user_boolean": user_bool,
            "correct_boolean": correct_bool,
            "interpretation_success": user_bool is not None
        }
        
        return score, is_correct, analysis
    
    def _score_matching(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำถามจับคู่"""
        # Expect user_answer to be a dict or list of pairs
        if not isinstance(user_answer, (dict, list)):
            return 0.0, False, {"error": "Invalid answer format for matching question"}
        
        # Convert to dict if needed
        if isinstance(user_answer, list):
            try:
                user_matches = dict(user_answer)
            except:
                return 0.0, False, {"error": "Cannot convert answer to matches"}
        else:
            user_matches = user_answer
        
        # Expected format in correct_answers: [{"left": "A", "right": "1"}, ...]
        correct_pairs = {}
        for answer in question.correct_answers:
            if isinstance(answer, dict) and "left" in answer and "right" in answer:
                correct_pairs[answer["left"]] = answer["right"]
        
        if not correct_pairs:
            return 0.0, False, {"error": "No correct pairs defined"}
        
        # Calculate score
        correct_matches = 0
        total_pairs = len(correct_pairs)
        
        for left, right in user_matches.items():
            if left in correct_pairs and str(correct_pairs[left]).lower() == str(right).lower():
                correct_matches += 1
        
        score = correct_matches / total_pairs if total_pairs > 0 else 0.0
        is_correct = score >= 0.8  # 80% threshold for "correct"
        
        analysis = {
            "user_matches": user_matches,
            "correct_pairs": correct_pairs,
            "correct_matches": correct_matches,
            "total_pairs": total_pairs,
            "accuracy": score
        }
        
        return score, is_correct, analysis
    
    def _score_ordering(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำถามเรียงลำดับ"""
        if not isinstance(user_answer, list):
            return 0.0, False, {"error": "Answer must be a list"}
        
        if not question.correct_answers:
            return 0.0, False, {"error": "No correct order defined"}
        
        correct_order = question.correct_answers[0] if isinstance(question.correct_answers[0], list) else question.correct_answers
        
        if len(user_answer) != len(correct_order):
            return 0.0, False, {"error": "Answer length doesn't match expected order"}
        
        # Calculate position-based score
        correct_positions = 0
        for i, item in enumerate(user_answer):
            if i < len(correct_order) and str(item).lower() == str(correct_order[i]).lower():
                correct_positions += 1
        
        score = correct_positions / len(correct_order)
        is_correct = score >= 0.8
        
        analysis = {
            "user_order": user_answer,
            "correct_order": correct_order,
            "correct_positions": correct_positions,
            "total_positions": len(correct_order),
            "position_accuracy": score
        }
        
        return score, is_correct, analysis
    
    def _score_open_ended(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำถามอิสระ"""
        # For open-ended questions, we need more sophisticated scoring
        # This is a simplified version - in practice, you'd use NLP techniques
        
        user_text = str(user_answer).strip().lower()
        
        if not question.correct_answers:
            # If no specific answers, check for length and coherence
            word_count = len(user_text.split())
            
            if word_count >= 10:  # Minimum response length
                score = 0.7  # Base score for adequate response
                is_correct = True
            else:
                score = 0.3
                is_correct = False
        else:
            # Check for key terms or concepts
            key_terms = []
            for answer in question.correct_answers:
                key_terms.extend(str(answer).lower().split())
            
            user_words = user_text.split()
            matching_terms = sum(1 for term in key_terms if term in user_words)
            
            score = min(1.0, matching_terms / max(1, len(key_terms)) * 1.5)
            is_correct = score >= 0.5
        
        analysis = {
            "user_text": user_answer,
            "word_count": len(user_text.split()),
            "character_count": len(user_text),
            "estimated_score": score,
            "scoring_method": "basic_text_analysis"
        }
        
        return score, is_correct, analysis
    
    def _score_audio_response(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนคำตอบเสียง"""
        # Placeholder for audio scoring - would integrate with speech recognition
        # and pronunciation analysis in a real implementation
        
        analysis = {
            "audio_file": user_answer,
            "scoring_method": "placeholder",
            "message": "Audio scoring not implemented in this version"
        }
        
        # Return neutral score for now
        return 0.5, True, analysis
    
    def _score_conversation(self, question: AssessmentQuestion, user_answer: Any) -> Tuple[float, bool, Dict[str, Any]]:
        """ให้คะแนนการสนทนา"""
        # Placeholder for conversation scoring
        # Would analyze conversation flow, vocabulary use, grammar, etc.
        
        analysis = {
            "conversation_data": user_answer,
            "scoring_method": "placeholder",
            "message": "Conversation scoring not implemented in this version"
        }
        
        return 0.5, True, analysis
    
    def _update_level_estimates(self, user_id: str, response: AssessmentResponse):
        """อัปเดตการประมาณระดับ"""
        if user_id not in self.adaptive_engines:
            return
        
        engine = self.adaptive_engines[user_id]
        question = self.questions.get(response.question_id)
        
        if not question:
            return
        
        skill = question.skill
        question_level = question.level
        
        # Simple adaptive algorithm
        current_estimate = engine["estimated_levels"][skill]
        current_confidence = engine["confidence_scores"][skill]
        
        # Adjust based on response
        if response.is_correct:
            # If correct, user might be at this level or higher
            if self._level_order(question_level) >= self._level_order(current_estimate):
                # Consider moving up
                new_estimate = self._get_next_level(question_level)
                engine["estimated_levels"][skill] = new_estimate or question_level
        else:
            # If incorrect, user might be below this level
            if self._level_order(question_level) <= self._level_order(current_estimate):
                # Consider moving down
                new_estimate = self._get_previous_level(question_level)
                engine["estimated_levels"][skill] = new_estimate or question_level
        
        # Update confidence
        response_confidence = 0.1 + (response.score * 0.1)  # 0.1 to 0.2
        engine["confidence_scores"][skill] = min(1.0, current_confidence + response_confidence)
    
    def _level_order(self, level: ProficiencyLevel) -> int:
        """แปลงระดับเป็นตัวเลข"""
        order = {
            ProficiencyLevel.BEGINNER: 1,
            ProficiencyLevel.ELEMENTARY: 2,
            ProficiencyLevel.INTERMEDIATE: 3,
            ProficiencyLevel.UPPER_INTERMEDIATE: 4,
            ProficiencyLevel.ADVANCED: 5,
            ProficiencyLevel.PROFICIENT: 6
        }
        return order.get(level, 3)
    
    def _get_next_level(self, level: ProficiencyLevel) -> Optional[ProficiencyLevel]:
        """ได้ระดับถัดไป"""
        levels = list(ProficiencyLevel)
        try:
            current_index = levels.index(level)
            return levels[current_index + 1] if current_index < len(levels) - 1 else None
        except ValueError:
            return None
    
    def _get_previous_level(self, level: ProficiencyLevel) -> Optional[ProficiencyLevel]:
        """ได้ระดับก่อนหน้า"""
        levels = list(ProficiencyLevel)
        try:
            current_index = levels.index(level)
            return levels[current_index - 1] if current_index > 0 else None
        except ValueError:
            return None
    
    def complete_assessment(self, user_id: str) -> Dict[str, Any]:
        """จบการประเมิน"""
        if user_id not in self.adaptive_engines:
            return {"error": "No active assessment found"}
        
        engine = self.adaptive_engines[user_id]
        responses = engine["responses"]
        
        if not responses:
            return {"error": "No responses recorded"}
        
        # Calculate final skill assessments
        skill_assessments = {}
        
        for skill in engine["target_skills"]:
            skill_responses = [r for r in responses if self.questions.get(r.question_id, {}).skill == skill]
            
            if skill_responses:
                assessment = self._calculate_skill_assessment(skill, skill_responses)
                skill_assessments[skill] = assessment
                self.user_assessments[user_id][skill] = assessment
        
        # Generate learning path
        learning_path = self._generate_learning_path(user_id, skill_assessments)
        
        # Clean up engine
        del self.adaptive_engines[user_id]
        
        result = {
            "user_id": user_id,
            "assessment_completed": datetime.now().isoformat(),
            "skill_assessments": {skill.value: asdict(assessment) for skill, assessment in skill_assessments.items()},
            "learning_path": asdict(learning_path) if learning_path else None,
            "total_questions": len([r for r in responses]),
            "overall_score": np.mean([r.score for r in responses]),
            "completion_time": sum(r.response_time for r in responses),
            "recommendations": self._generate_recommendations(skill_assessments)
        }
        
        self.logger.info(f"Assessment completed for {user_id}")
        return result
    
    def _calculate_skill_assessment(self, skill: LanguageSkill, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณผลการประเมินทักษะ"""
        if not responses:
            return SkillAssessment(
                skill=skill,
                current_level=ProficiencyLevel.BEGINNER,
                confidence_score=0.0,
                strengths=[],
                weaknesses=[],
                recommended_level=ProficiencyLevel.BEGINNER,
                next_milestones=[],
                estimated_study_hours=0
            )
        
        # Use specific calculator if available
        calculator = self.level_calculators.get(skill)
        if calculator:
            return calculator(responses)
        
        # Default calculation
        return self._default_skill_calculation(skill, responses)
    
    def _default_skill_calculation(self, skill: LanguageSkill, responses: List[AssessmentResponse]) -> SkillAssessment:
        """การคำนวณทักษะแบบเริ่มต้น"""
        # Calculate average score and determine level
        avg_score = np.mean([r.score for r in responses])
        
        # Simple level mapping based on score
        if avg_score >= 0.9:
            level = ProficiencyLevel.PROFICIENT
        elif avg_score >= 0.8:
            level = ProficiencyLevel.ADVANCED
        elif avg_score >= 0.7:
            level = ProficiencyLevel.UPPER_INTERMEDIATE
        elif avg_score >= 0.6:
            level = ProficiencyLevel.INTERMEDIATE
        elif avg_score >= 0.4:
            level = ProficiencyLevel.ELEMENTARY
        else:
            level = ProficiencyLevel.BEGINNER
        
        # Calculate confidence based on consistency
        scores = [r.score for r in responses]
        consistency = 1 - np.std(scores) if len(scores) > 1 else 1.0
        confidence = min(1.0, avg_score * consistency)
        
        # Identify strengths and weaknesses
        strengths = []
        weaknesses = []
        
        correct_responses = [r for r in responses if r.is_correct]
        incorrect_responses = [r for r in responses if not r.is_correct]
        
        if len(correct_responses) > len(incorrect_responses):
            strengths.append(f"Good performance in {skill.value}")
        
        if len(incorrect_responses) > 0:
            # Analyze common mistakes
            common_issues = self._analyze_common_mistakes(skill, incorrect_responses)
            weaknesses.extend(common_issues)
        
        # Generate next milestones
        next_milestones = self._generate_skill_milestones(skill, level)
        
        # Estimate study hours
        estimated_hours = self._estimate_study_hours(level, ProficiencyLevel.ADVANCED)
        
        return SkillAssessment(
            skill=skill,
            current_level=level,
            confidence_score=confidence,
            strengths=strengths,
            weaknesses=weaknesses,
            recommended_level=level,
            next_milestones=next_milestones,
            estimated_study_hours=estimated_hours
        )
    
    def _calculate_listening_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับทักษะฟัง"""
        # Specialized calculation for listening skills
        return self._default_skill_calculation(LanguageSkill.LISTENING, responses)
    
    def _calculate_speaking_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับทักษะพูด"""
        return self._default_skill_calculation(LanguageSkill.SPEAKING, responses)
    
    def _calculate_reading_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับทักษะอ่าน"""
        return self._default_skill_calculation(LanguageSkill.READING, responses)
    
    def _calculate_writing_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับทักษะเขียน"""
        return self._default_skill_calculation(LanguageSkill.WRITING, responses)
    
    def _calculate_vocabulary_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับคำศัพท์"""
        return self._default_skill_calculation(LanguageSkill.VOCABULARY, responses)
    
    def _calculate_grammar_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับไวยากรณ์"""
        return self._default_skill_calculation(LanguageSkill.GRAMMAR, responses)
    
    def _calculate_pronunciation_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับการออกเสียง"""
        return self._default_skill_calculation(LanguageSkill.PRONUNCIATION, responses)
    
    def _calculate_comprehension_level(self, responses: List[AssessmentResponse]) -> SkillAssessment:
        """คำนวณระดับความเข้าใจ"""
        return self._default_skill_calculation(LanguageSkill.COMPREHENSION, responses)
    
    def _analyze_common_mistakes(self, skill: LanguageSkill, incorrect_responses: List[AssessmentResponse]) -> List[str]:
        """วิเคราะห์ข้อผิดพลาดทั่วไป"""
        mistakes = []
        
        # Analyze patterns in incorrect responses
        question_types = [self.questions.get(r.question_id, {}).question_type for r in incorrect_responses]
        type_counts = Counter(question_types)
        
        for question_type, count in type_counts.most_common(3):
            if count > 1:
                mistakes.append(f"Difficulty with {question_type.value if question_type else 'unknown'} questions")
        
        return mistakes
    
    def _generate_skill_milestones(self, skill: LanguageSkill, current_level: ProficiencyLevel) -> List[str]:
        """สร้างเป้าหมายทักษะ"""
        milestones = []
        
        milestone_templates = {
            LanguageSkill.VOCABULARY: [
                "Learn 100 new words",
                "Master common phrasal verbs",
                "Understand academic vocabulary"
            ],
            LanguageSkill.GRAMMAR: [
                "Perfect present and past tenses",
                "Master conditional sentences",
                "Use advanced sentence structures"
            ],
            LanguageSkill.LISTENING: [
                "Understand native speaker conversations",
                "Follow academic lectures",
                "Comprehend different accents"
            ],
            LanguageSkill.SPEAKING: [
                "Speak fluently in conversations",
                "Give presentations confidently",
                "Express complex ideas clearly"
            ]
        }
        
        templates = milestone_templates.get(skill, ["Improve overall skill level"])
        
        # Select appropriate milestones based on current level
        level_index = self._level_order(current_level)
        if level_index < len(templates):
            milestones.extend(templates[level_index:level_index+2])
        
        return milestones
    
    def _estimate_study_hours(self, current_level: ProficiencyLevel, target_level: ProficiencyLevel) -> int:
        """ประมาณชั่วโมงเรียน"""
        level_hours = {
            ProficiencyLevel.BEGINNER: 0,
            ProficiencyLevel.ELEMENTARY: 200,
            ProficiencyLevel.INTERMEDIATE: 400,
            ProficiencyLevel.UPPER_INTERMEDIATE: 600,
            ProficiencyLevel.ADVANCED: 800,
            ProficiencyLevel.PROFICIENT: 1000
        }
        
        current_hours = level_hours.get(current_level, 0)
        target_hours = level_hours.get(target_level, 1000)
        
        return max(0, target_hours - current_hours)
    
    def _generate_learning_path(self, user_id: str, skill_assessments: Dict[LanguageSkill, SkillAssessment]) -> Optional[LearningPath]:
        """สร้างเส้นทางการเรียน"""
        if not skill_assessments:
            return None
        
        path_id = f"path_{user_id}_{int(time.time())}"
        
        # Determine overall current level (lowest skill level)
        current_levels = [assessment.current_level for assessment in skill_assessments.values()]
        overall_current = min(current_levels, key=lambda x: self._level_order(x))
        
        # Determine target level (one level above current, or Advanced if already high)
        target_level = self._get_next_level(overall_current) or ProficiencyLevel.ADVANCED
        
        # Identify focus skills (lowest performing skills)
        skill_scores = {
            skill: assessment.confidence_score 
            for skill, assessment in skill_assessments.items()
        }
        
        sorted_skills = sorted(skill_scores.items(), key=lambda x: x[1])
        focus_skills = [skill for skill, _ in sorted_skills[:3]]  # Top 3 areas needing improvement
        
        # Generate learning objectives
        objectives = []
        for skill in focus_skills:
            assessment = skill_assessments[skill]
            objectives.extend([
                f"Improve {skill.value} from {assessment.current_level.value} to {target_level.value}",
                *assessment.next_milestones[:2]
            ])
        
        # Generate recommended activities
        activities = self._generate_learning_activities(focus_skills, overall_current, target_level)
        
        # Calculate estimated duration
        avg_study_hours = np.mean([assessment.estimated_study_hours for assessment in skill_assessments.values()])
        estimated_days = max(30, int(avg_study_hours / 2))  # Assuming 2 hours per day
        
        # Generate progress milestones
        milestones = self._generate_progress_milestones(focus_skills, estimated_days)
        
        learning_path = LearningPath(
            path_id=path_id,
            user_id=user_id,
            target_level=target_level,
            current_level=overall_current,
            focus_skills=focus_skills,
            learning_objectives=objectives,
            recommended_activities=activities,
            estimated_duration=estimated_days,
            progress_milestones=milestones,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        self.learning_paths[path_id] = learning_path
        return learning_path
    
    def _generate_learning_activities(self, focus_skills: List[LanguageSkill], 
                                    current_level: ProficiencyLevel, 
                                    target_level: ProficiencyLevel) -> List[Dict[str, Any]]:
        """สร้างกิจกรรมการเรียนแนะนำ"""
        activities = []
        
        activity_templates = {
            LanguageSkill.VOCABULARY: [
                {"type": "flashcards", "description": "Daily vocabulary flashcards", "duration": 15, "frequency": "daily"},
                {"type": "word_games", "description": "Interactive word games", "duration": 20, "frequency": "3x/week"},
                {"type": "reading", "description": "Read articles with new vocabulary", "duration": 30, "frequency": "daily"}
            ],
            LanguageSkill.GRAMMAR: [
                {"type": "exercises", "description": "Grammar exercises and drills", "duration": 25, "frequency": "daily"},
                {"type": "writing_practice", "description": "Structured writing practice", "duration": 30, "frequency": "5x/week"},
                {"type": "error_analysis", "description": "Analyze and correct common errors", "duration": 20, "frequency": "2x/week"}
            ],
            LanguageSkill.LISTENING: [
                {"type": "podcasts", "description": "Listen to English podcasts", "duration": 30, "frequency": "daily"},
                {"type": "movies", "description": "Watch English movies with subtitles", "duration": 60, "frequency": "3x/week"},
                {"type": "conversations", "description": "Listen to native speaker conversations", "duration": 20, "frequency": "daily"}
            ],
            LanguageSkill.SPEAKING: [
                {"type": "conversation_practice", "description": "Practice conversations with AI", "duration": 30, "frequency": "daily"},
                {"type": "pronunciation_drills", "description": "Pronunciation exercises", "duration": 15, "frequency": "daily"},
                {"type": "presentations", "description": "Practice short presentations", "duration": 20, "frequency": "2x/week"}
            ]
        }
        
        for skill in focus_skills:
            if skill in activity_templates:
                skill_activities = activity_templates[skill]
                for activity in skill_activities:
                    activities.append({
                        **activity,
                        "skill": skill.value,
                        "level": current_level.value
                    })
        
        return activities[:10]  # Limit to 10 activities
    
    def _generate_progress_milestones(self, focus_skills: List[LanguageSkill], 
                                    duration_days: int) -> List[Dict[str, Any]]:
        """สร้างเป้าหมายความก้าวหน้า"""
        milestones = []
        
        # Create milestones at 25%, 50%, 75%, and 100% of duration
        milestone_points = [0.25, 0.5, 0.75, 1.0]
        milestone_names = ["First Quarter", "Halfway", "Three Quarters", "Completion"]
        
        for i, (point, name) in enumerate(zip(milestone_points, milestone_names)):
            day = int(duration_days * point)
            
            milestones.append({
                "milestone_id": f"milestone_{i+1}",
                "name": name,
                "target_day": day,
                "description": f"Complete {name.lower()} of learning plan",
                "success_criteria": [
                    f"Complete {int(point * 100)}% of vocabulary exercises",
                    f"Achieve {60 + (i * 10)}% accuracy on practice tests",
                    f"Demonstrate improvement in {focus_skills[i % len(focus_skills)].value}" if focus_skills else "Show overall improvement"
                ],
                "assessment_required": i == len(milestone_points) - 1  # Final milestone requires assessment
            })
        
        return milestones
    
    def _generate_recommendations(self, skill_assessments: Dict[LanguageSkill, SkillAssessment]) -> List[str]:
        """สร้างคำแนะนำ"""
        recommendations = []
        
        if not skill_assessments:
            return ["Complete a placement assessment to receive personalized recommendations"]
        
        # Find weakest and strongest skills
        skill_scores = {skill: assessment.confidence_score for skill, assessment in skill_assessments.items()}
        weakest_skill = min(skill_scores.items(), key=lambda x: x[1])
        strongest_skill = max(skill_scores.items(), key=lambda x: x[1])
        
        recommendations.append(f"Focus primarily on improving your {weakest_skill[0].value} skills")
        recommendations.append(f"Continue to maintain your strong {strongest_skill[0].value} abilities")
        
        # Level-specific recommendations
        avg_level = np.mean([self._level_order(assessment.current_level) for assessment in skill_assessments.values()])
        
        if avg_level < 3:  # Below intermediate
            recommendations.append("Focus on building fundamental vocabulary and basic grammar")
            recommendations.append("Practice with simple texts and short audio materials")
        elif avg_level < 5:  # Below advanced
            recommendations.append("Work on complex grammar structures and advanced vocabulary")
            recommendations.append("Engage with authentic materials like news articles and podcasts")
        else:  # Advanced level
            recommendations.append("Focus on nuanced language use and cultural understanding")
            recommendations.append("Practice with professional and academic contexts")
        
        return recommendations
    
    def _generate_feedback(self, question: AssessmentQuestion, response: AssessmentResponse) -> str:
        """สร้างข้อเสนอแนะ"""
        if response.is_correct:
            feedback = f"✅ Correct! "
            
            if response.response_time < 10:  # Quick response
                feedback += "Great response time too!"
            elif response.score == 1.0:
                feedback += "Perfect answer!"
            else:
                feedback += "Good work!"
                
        else:
            feedback = f"❌ Not quite right. "
            
            if question.correct_answers:
                feedback += f"The correct answer is: {', '.join(map(str, question.correct_answers))}"
            
            if question.explanation:
                feedback += f"\n\n💡 Explanation: {question.explanation}"
        
        # Add encouraging note
        if response.score >= 0.5:
            feedback += "\n\nYou're on the right track - keep practicing!"
        else:
            feedback += f"\n\nDon't worry - {question.skill.value} takes practice. You'll improve!"
        
        return feedback
    
    def _load_question_banks(self):
        """โหลด question banks"""
        # This would typically load from external files or databases
        # For now, create some sample questions
        self._create_sample_questions()
    
    def _create_sample_questions(self):
        """สร้างคำถามตัวอย่าง"""
        sample_questions = [
            # Vocabulary - Beginner
            AssessmentQuestion(
                question_id="vocab_beg_001",
                skill=LanguageSkill.VOCABULARY,
                level=ProficiencyLevel.BEGINNER,
                question_type=QuestionType.MULTIPLE_CHOICE,
                question_text="What does 'hello' mean in Thai?",
                options=["สวัสดี", "ลาก่อน", "ขอบคุณ", "ขอโทษ"],
                correct_answers=["สวัสดี"],
                explanation="'Hello' means 'สวัสดี' which is a common greeting in Thai.",
                points=1,
                difficulty_score=0.2,
                tags={"greeting", "basic_vocabulary"}
            ),
            
            # Grammar - Intermediate
            AssessmentQuestion(
                question_id="gram_int_001",
                skill=LanguageSkill.GRAMMAR,
                level=ProficiencyLevel.INTERMEDIATE,
                question_type=QuestionType.FILL_BLANK,
                question_text="If I _____ rich, I would travel around the world.",
                correct_answers=["were", "was"],
                explanation="Use 'were' (subjunctive mood) in unreal conditional sentences.",
                points=2,
                difficulty_score=0.6,
                tags={"conditionals", "subjunctive"}
            ),
            
            # Reading - Elementary
            AssessmentQuestion(
                question_id="read_elem_001",
                skill=LanguageSkill.READING,
                level=ProficiencyLevel.ELEMENTARY,
                question_type=QuestionType.TRUE_FALSE,
                question_text="Read: 'John likes to eat pizza every Friday.' Question: John eats pizza once a week.",
                correct_answers=["true"],
                explanation="The text says John eats pizza 'every Friday', which means once a week.",
                points=1,
                difficulty_score=0.3,
                tags={"reading_comprehension", "frequency"}
            )
        ]
        
        # Organize questions by skill and level
        for question in sample_questions:
            self.questions[question.question_id] = question
            
            if question.skill not in self.question_banks:
                self.question_banks[question.skill] = {}
            
            if question.level not in self.question_banks[question.skill]:
                self.question_banks[question.skill][question.level] = []
            
            self.question_banks[question.skill][question.level].append(question)
        
        self.logger.info(f"Loaded {len(sample_questions)} sample questions")
    
    def _load_assessment_data(self):
        """โหลดข้อมูลการประเมิน"""
        try:
            # Load user responses
            responses_file = self.data_dir / "user_responses.pkl"
            if responses_file.exists():
                with open(responses_file, 'rb') as f:
                    self.user_responses = pickle.load(f)
            
            # Load user assessments
            assessments_file = self.data_dir / "user_assessments.pkl"
            if assessments_file.exists():
                with open(assessments_file, 'rb') as f:
                    self.user_assessments = pickle.load(f)
            
            # Load learning paths
            paths_file = self.data_dir / "learning_paths.pkl"
            if paths_file.exists():
                with open(paths_file, 'rb') as f:
                    self.learning_paths = pickle.load(f)
            
            self.logger.info("Assessment data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load assessment data: {e}")
    
    def save_assessment_data(self):
        """บันทึกข้อมูลการประเมิน"""
        try:
            # Save user responses
            responses_file = self.data_dir / "user_responses.pkl"
            with open(responses_file, 'wb') as f:
                pickle.dump(self.user_responses, f)
            
            # Save user assessments
            assessments_file = self.data_dir / "user_assessments.pkl"
            with open(assessments_file, 'wb') as f:
                pickle.dump(self.user_assessments, f)
            
            # Save learning paths
            paths_file = self.data_dir / "learning_paths.pkl"
            with open(paths_file, 'wb') as f:
                pickle.dump(self.learning_paths, f)
            
            self.logger.info("Assessment data saved successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to save assessment data: {e}")
    
    def get_user_progress(self, user_id: str) -> Dict[str, Any]:
        """ดูความก้าวหน้าของผู้ใช้"""
        responses = self.user_responses.get(user_id, [])
        assessments = self.user_assessments.get(user_id, {})
        
        # Find user's learning path
        user_path = None
        for path in self.learning_paths.values():
            if path.user_id == user_id:
                user_path = path
                break
        
        return {
            "user_id": user_id,
            "total_responses": len(responses),
            "skills_assessed": list(assessments.keys()),
            "skill_levels": {skill.value: assessment.current_level.value for skill, assessment in assessments.items()},
            "overall_progress": self._calculate_overall_progress(user_id),
            "learning_path": asdict(user_path) if user_path else None,
            "recent_activity": [asdict(r) for r in responses[-10:]]  # Last 10 responses
        }
    
    def _calculate_overall_progress(self, user_id: str) -> Dict[str, Any]:
        """คำนวณความก้าวหน้ารวม"""
        responses = self.user_responses.get(user_id, [])
        
        if not responses:
            return {"message": "No assessment data available"}
        
        # Recent performance (last 20 responses)
        recent_responses = responses[-20:] if len(responses) > 20 else responses
        recent_avg = np.mean([r.score for r in recent_responses])
        
        # Overall performance
        overall_avg = np.mean([r.score for r in responses])
        
        # Progress trend
        if len(responses) >= 10:
            early_avg = np.mean([r.score for r in responses[:len(responses)//2]])
            later_avg = np.mean([r.score for r in responses[len(responses)//2:]])
            trend = "improving" if later_avg > early_avg else "stable" if abs(later_avg - early_avg) < 0.1 else "declining"
        else:
            trend = "insufficient_data"
        
        return {
            "overall_score": overall_avg,
            "recent_score": recent_avg,
            "trend": trend,
            "total_assessments": len(responses),
            "improvement": recent_avg - overall_avg if recent_avg > overall_avg else 0
        }
    
    def shutdown(self):
        """ปิดระบบ"""
        self.save_assessment_data()
        self.logger.info("Language Assessment System shutdown complete")