"""
Self-Improvement System for JARVIS
ระบบพัฒนาตนเองที่ทำให้ JARVIS เรียนรู้และปรับปรุงตัวเองได้
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from enum import Enum

class LearningType(Enum):
    """ประเภทการเรียนรู้"""
    PATTERN_RECOGNITION = "pattern_recognition"  # จดจำรูปแบบ
    USER_PREFERENCE = "user_preference"          # ความชอบของผู้ใช้
    CONVERSATION_FLOW = "conversation_flow"      # การไหลของการสนทนา
    KNOWLEDGE_GAP = "knowledge_gap"              # ช่องว่างความรู้
    RESPONSE_QUALITY = "response_quality"        # คุณภาพการตอบ

@dataclass
class LearningEvent:
    """เหตุการณ์การเรียนรู้"""
    event_id: str
    timestamp: datetime
    learning_type: LearningType
    source_data: Dict[str, Any]
    pattern_detected: str
    confidence: float
    impact_score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "learning_type": self.learning_type.value,
            "source_data": self.source_data,
            "pattern_detected": self.pattern_detected,
            "confidence": self.confidence,
            "impact_score": self.impact_score,
            "metadata": self.metadata
        }

@dataclass
class KnowledgeUpdate:
    """การอัปเดตความรู้"""
    update_id: str
    timestamp: datetime
    knowledge_domain: str
    update_type: str  # "addition", "modification", "correction"
    old_knowledge: Optional[str]
    new_knowledge: str
    evidence_strength: float
    source_reliability: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "update_id": self.update_id,
            "timestamp": self.timestamp.isoformat(),
            "knowledge_domain": self.knowledge_domain,
            "update_type": self.update_type,
            "old_knowledge": self.old_knowledge,
            "new_knowledge": self.new_knowledge,
            "evidence_strength": self.evidence_strength,
            "source_reliability": self.source_reliability
        }

class SelfImprovementSystem:
    """ระบบพัฒนาตนเองของ JARVIS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Learning components
        self.learning_events: List[LearningEvent] = []
        self.knowledge_updates: List[KnowledgeUpdate] = []
        self.user_patterns: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.conversation_patterns: Dict[str, Any] = defaultdict(list)
        self.response_feedback: Dict[str, List[float]] = defaultdict(list)
        
        # Performance tracking
        self.performance_metrics = {
            "response_accuracy": [],
            "user_satisfaction": [],
            "conversation_length": [],
            "topic_coverage": [],
            "response_time": []
        }
        
        # Learning parameters
        self.learning_rate = config.get("learning_rate", 0.1)
        self.confidence_threshold = config.get("confidence_threshold", 0.7)
        self.min_pattern_occurrences = config.get("min_pattern_occurrences", 3)
        
        # Data persistence
        self.data_dir = Path(config.get("data_dir", "data/self_improvement"))
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing data
        self._load_learning_data()
        
        self.logger.info("Self-Improvement System initialized")
    
    def learn_from_conversation(self, conversation_data: Dict[str, Any]) -> List[LearningEvent]:
        """เรียนรู้จากการสนทนา"""
        learning_events = []
        
        # Analyze user patterns
        user_pattern_event = self._analyze_user_patterns(conversation_data)
        if user_pattern_event:
            learning_events.append(user_pattern_event)
        
        # Analyze conversation flow
        flow_event = self._analyze_conversation_flow(conversation_data)
        if flow_event:
            learning_events.append(flow_event)
        
        # Detect knowledge gaps
        knowledge_gap_event = self._detect_knowledge_gaps(conversation_data)
        if knowledge_gap_event:
            learning_events.append(knowledge_gap_event)
        
        # Evaluate response quality
        quality_event = self._evaluate_response_quality(conversation_data)
        if quality_event:
            learning_events.append(quality_event)
        
        # Store learning events
        for event in learning_events:
            self.learning_events.append(event)
        
        # Apply immediate improvements
        self._apply_immediate_improvements(learning_events)
        
        self.logger.info(f"Learned {len(learning_events)} insights from conversation")
        return learning_events
    
    def _analyze_user_patterns(self, conversation_data: Dict[str, Any]) -> Optional[LearningEvent]:
        """วิเคราะห์รูปแบบผู้ใช้"""
        user_id = conversation_data.get("user_id")
        if not user_id:
            return None
        
        # Extract user behavior patterns
        patterns = {
            "preferred_language": conversation_data.get("language", "th"),
            "response_style_preference": self._infer_response_style(conversation_data),
            "topic_interests": conversation_data.get("topics", []),
            "question_complexity": self._analyze_question_complexity(conversation_data),
            "interaction_time": conversation_data.get("interaction_time"),
            "session_length": conversation_data.get("session_length", 0)
        }
        
        # Check if this is a significant pattern change
        if user_id in self.user_patterns:
            similarity = self._calculate_pattern_similarity(
                self.user_patterns[user_id], patterns
            )
            if similarity > 0.8:  # Not much change
                return None
        
        # Update user patterns
        self.user_patterns[user_id].update(patterns)
        
        return LearningEvent(
            event_id=f"user_pattern_{user_id}_{int(time.time())}",
            timestamp=datetime.now(),
            learning_type=LearningType.USER_PREFERENCE,
            source_data=conversation_data,
            pattern_detected=f"User preferences for {user_id}",
            confidence=0.8,
            impact_score=0.6,
            metadata={"patterns": patterns}
        )
    
    def _analyze_conversation_flow(self, conversation_data: Dict[str, Any]) -> Optional[LearningEvent]:
        """วิเคราะห์การไหลของการสนทนา"""
        history = conversation_data.get("conversation_history", [])
        if len(history) < 3:
            return None
        
        # Extract conversation patterns
        flow_patterns = []
        for i in range(len(history) - 1):
            current = history[i]
            next_msg = history[i + 1]
            
            pattern = {
                "user_intent": current.get("analysis", {}).get("intent"),
                "assistant_response_type": self._classify_response_type(current.get("assistant_response", "")),
                "user_follow_up": next_msg.get("analysis", {}).get("intent"),
                "satisfaction_indicator": self._infer_satisfaction(next_msg)
            }
            flow_patterns.append(pattern)
        
        # Check for recurring patterns
        pattern_key = str(flow_patterns[-1])  # Most recent pattern
        self.conversation_patterns[pattern_key].append({
            "timestamp": datetime.now(),
            "effectiveness": self._assess_pattern_effectiveness(flow_patterns)
        })
        
        return LearningEvent(
            event_id=f"flow_{int(time.time())}",
            timestamp=datetime.now(),
            learning_type=LearningType.CONVERSATION_FLOW,
            source_data=conversation_data,
            pattern_detected=pattern_key,
            confidence=0.7,
            impact_score=0.5,
            metadata={"flow_patterns": flow_patterns}
        )
    
    def _detect_knowledge_gaps(self, conversation_data: Dict[str, Any]) -> Optional[LearningEvent]:
        """ตรวจจับช่องว่างความรู้"""
        analysis = conversation_data.get("analysis", {})
        assistant_confidence = conversation_data.get("response_confidence", 1.0)
        
        # Low confidence indicates potential knowledge gap
        if assistant_confidence < 0.6:
            topics = analysis.get("topics", [])
            question_type = analysis.get("question_type", "general")
            
            gap_detected = {
                "topics": topics,
                "question_type": question_type,
                "confidence_score": assistant_confidence,
                "user_message": conversation_data.get("user_message", ""),
                "response_adequacy": self._assess_response_adequacy(conversation_data)
            }
            
            return LearningEvent(
                event_id=f"knowledge_gap_{int(time.time())}",
                timestamp=datetime.now(),
                learning_type=LearningType.KNOWLEDGE_GAP,
                source_data=conversation_data,
                pattern_detected=f"Knowledge gap in {topics}",
                confidence=1.0 - assistant_confidence,
                impact_score=0.8,
                metadata={"gap_details": gap_detected}
            )
        
        return None
    
    def _evaluate_response_quality(self, conversation_data: Dict[str, Any]) -> Optional[LearningEvent]:
        """ประเมินคุณภาพการตอบ"""
        response = conversation_data.get("assistant_response", "")
        user_message = conversation_data.get("user_message", "")
        
        quality_metrics = {
            "relevance": self._assess_relevance(user_message, response),
            "completeness": self._assess_completeness(user_message, response),
            "clarity": self._assess_clarity(response),
            "helpfulness": self._assess_helpfulness(conversation_data),
            "cultural_appropriateness": self._assess_cultural_appropriateness(response)
        }
        
        overall_quality = np.mean(list(quality_metrics.values()))
        
        # Store quality feedback
        response_id = f"response_{int(time.time())}"
        self.response_feedback[response_id].append(overall_quality)
        
        return LearningEvent(
            event_id=f"quality_{int(time.time())}",
            timestamp=datetime.now(),
            learning_type=LearningType.RESPONSE_QUALITY,
            source_data=conversation_data,
            pattern_detected=f"Response quality: {overall_quality:.2f}",
            confidence=0.9,
            impact_score=overall_quality,
            metadata={"quality_metrics": quality_metrics}
        )
    
    def _infer_response_style(self, conversation_data: Dict[str, Any]) -> str:
        """อนุมานรูปแบบการตอบที่ผู้ใช้ชอบ"""
        user_messages = [msg.get("user_message", "") for msg in conversation_data.get("conversation_history", [])]
        
        # Analyze user message characteristics
        avg_length = np.mean([len(msg.split()) for msg in user_messages if msg])
        
        if avg_length < 5:
            return "concise"
        elif avg_length > 15:
            return "detailed"
        else:
            return "balanced"
    
    def _analyze_question_complexity(self, conversation_data: Dict[str, Any]) -> str:
        """วิเคราะห์ความซับซ้อนของคำถาม"""
        history = conversation_data.get("conversation_history", [])
        complexities = []
        
        for msg in history:
            analysis = msg.get("analysis", {})
            complexity = analysis.get("complexity", "simple")
            complexities.append(complexity)
        
        if not complexities:
            return "simple"
        
        # Most common complexity
        complexity_counter = Counter(complexities)
        return complexity_counter.most_common(1)[0][0]
    
    def _calculate_pattern_similarity(self, pattern1: Dict[str, Any], 
                                    pattern2: Dict[str, Any]) -> float:
        """คำนวณความคล้ายคลึงของรูปแบบ"""
        common_keys = set(pattern1.keys()) & set(pattern2.keys())
        if not common_keys:
            return 0.0
        
        similarities = []
        for key in common_keys:
            if pattern1[key] == pattern2[key]:
                similarities.append(1.0)
            elif isinstance(pattern1[key], (int, float)) and isinstance(pattern2[key], (int, float)):
                # Numeric similarity
                diff = abs(pattern1[key] - pattern2[key])
                max_val = max(abs(pattern1[key]), abs(pattern2[key]), 1)
                similarities.append(1.0 - (diff / max_val))
            else:
                similarities.append(0.0)
        
        return np.mean(similarities)
    
    def _classify_response_type(self, response: str) -> str:
        """จำแนกประเภทการตอบ"""
        response_lower = response.lower()
        
        if any(q in response for q in ["?", "คำถาม", "question"]):
            return "question_response"
        elif any(w in response_lower for w in ["ขั้นตอน", "step", "วิธี", "how"]):
            return "instructional"
        elif any(w in response_lower for w in ["คือ", "หมายถึง", "means", "definition"]):
            return "explanatory"
        elif len(response.split()) < 10:
            return "brief"
        else:
            return "comprehensive"
    
    def _infer_satisfaction(self, user_message_data: Dict[str, Any]) -> float:
        """อนุมานความพอใจจากข้อความผู้ใช้"""
        message = user_message_data.get("user_message", "").lower()
        analysis = user_message_data.get("analysis", {})
        
        satisfaction_score = 0.5  # neutral
        
        # Positive indicators
        positive_words = ["ขอบคุณ", "ดี", "เยี่ยม", "เข้าใจ", "ชัด", "thank", "good", "great", "clear"]
        negative_words = ["ไม่เข้าใจ", "งง", "ผิด", "แย่", "confused", "wrong", "bad", "unclear"]
        
        positive_count = sum(1 for word in positive_words if word in message)
        negative_count = sum(1 for word in negative_words if word in message)
        
        satisfaction_score += (positive_count * 0.2) - (negative_count * 0.3)
        
        # Emotion analysis
        emotion = analysis.get("emotion", "neutral")
        if emotion == "positive":
            satisfaction_score += 0.3
        elif emotion == "negative":
            satisfaction_score -= 0.3
        
        return max(0.0, min(1.0, satisfaction_score))
    
    def _assess_pattern_effectiveness(self, flow_patterns: List[Dict[str, Any]]) -> float:
        """ประเมินประสิทธิผลของรูปแบบ"""
        if not flow_patterns:
            return 0.5
        
        # Look at satisfaction indicators in the flow
        satisfactions = []
        for pattern in flow_patterns:
            satisfaction = pattern.get("satisfaction_indicator", 0.5)
            satisfactions.append(satisfaction)
        
        return np.mean(satisfactions)
    
    def _assess_response_adequacy(self, conversation_data: Dict[str, Any]) -> float:
        """ประเมินความเพียงพอของการตอบ"""
        user_message = conversation_data.get("user_message", "")
        response = conversation_data.get("assistant_response", "")
        
        # Simple heuristics for adequacy
        user_length = len(user_message.split())
        response_length = len(response.split())
        
        # Response should be proportional to question complexity
        if user_length > 0:
            ratio = response_length / user_length
            # Optimal ratio is around 2-4 (response 2-4 times longer than question)
            if 1.5 <= ratio <= 5.0:
                return 0.8
            elif ratio < 1.5:
                return 0.4  # Too brief
            else:
                return 0.6  # Too verbose
        
        return 0.5
    
    def _assess_relevance(self, user_message: str, response: str) -> float:
        """ประเมินความเกี่ยวข้อง"""
        # Simple keyword matching approach
        user_words = set(user_message.lower().split())
        response_words = set(response.lower().split())
        
        if len(user_words) == 0:
            return 0.5
        
        overlap = len(user_words & response_words)
        relevance = overlap / len(user_words)
        
        return min(1.0, relevance + 0.3)  # Add baseline relevance
    
    def _assess_completeness(self, user_message: str, response: str) -> float:
        """ประเมินความครบถ้วน"""
        # Check if response addresses question indicators
        question_words = ["คือ", "ทำไม", "อะไร", "ยังไง", "what", "why", "how", "when", "where"]
        
        has_question = any(word in user_message.lower() for word in question_words)
        
        if has_question:
            # Response should provide answer indicators
            answer_indicators = ["คือ", "เพราะ", "วิธี", "is", "because", "method", "reason"]
            has_answer = any(word in response.lower() for word in answer_indicators)
            
            return 0.8 if has_answer else 0.4
        
        return 0.7  # Non-question interactions
    
    def _assess_clarity(self, response: str) -> float:
        """ประเมินความชัดเจน"""
        # Simple metrics for clarity
        sentences = response.split('.')
        if not sentences:
            return 0.3
        
        avg_sentence_length = np.mean([len(s.split()) for s in sentences if s.strip()])
        
        # Optimal sentence length is 8-20 words
        if 8 <= avg_sentence_length <= 20:
            return 0.9
        elif avg_sentence_length < 8:
            return 0.6  # Too brief
        else:
            return 0.7  # Too long
    
    def _assess_helpfulness(self, conversation_data: Dict[str, Any]) -> float:
        """ประเมินประโยชน์"""
        response = conversation_data.get("assistant_response", "")
        analysis = conversation_data.get("analysis", {})
        
        helpfulness_score = 0.5  # baseline
        
        # Check for helpful elements
        helpful_patterns = [
            r"ขั้นตอน", r"วิธี", r"step", r"method",
            r"ตัวอย่าง", r"example", 
            r"แนะนำ", r"suggest", r"recommend",
            r"เพิ่มเติม", r"more", r"additional"
        ]
        
        import re
        helpful_count = sum(1 for pattern in helpful_patterns 
                          if re.search(pattern, response.lower()))
        
        helpfulness_score += helpful_count * 0.1
        
        return min(1.0, helpfulness_score)
    
    def _assess_cultural_appropriateness(self, response: str) -> float:
        """ประเมินความเหมาะสมทางวัฒนธรรม"""
        # Check for Thai cultural elements when appropriate
        cultural_indicators = [
            "ครับ", "ค่ะ", "คะ", "นะครับ", "นะค่ะ"  # Polite particles
        ]
        
        has_cultural = any(indicator in response for indicator in cultural_indicators)
        
        # Check if Thai language is used
        thai_chars = len(re.findall(r'[\u0E00-\u0E7F]', response))
        total_chars = len(response)
        
        if total_chars > 0:
            thai_ratio = thai_chars / total_chars
            if thai_ratio > 0.5 and has_cultural:
                return 0.9  # Good cultural appropriateness
            elif thai_ratio > 0.5:
                return 0.7  # Thai but missing some cultural elements
            else:
                return 0.8  # English responses are culturally neutral
        
        return 0.8
    
    def _apply_immediate_improvements(self, learning_events: List[LearningEvent]):
        """ปรับปรุงทันที"""
        for event in learning_events:
            if event.confidence > self.confidence_threshold:
                self._apply_single_improvement(event)
    
    def _apply_single_improvement(self, event: LearningEvent):
        """ปรับปรุงจากการเรียนรู้เดี่ยว"""
        if event.learning_type == LearningType.USER_PREFERENCE:
            # Update user preference model
            self.logger.info(f"Applied user preference improvement: {event.pattern_detected}")
        
        elif event.learning_type == LearningType.RESPONSE_QUALITY:
            # Adjust response generation parameters
            quality_metrics = event.metadata.get("quality_metrics", {})
            
            if quality_metrics.get("clarity", 0.5) < 0.6:
                self.logger.info("Adjusting for clearer responses")
            
            if quality_metrics.get("completeness", 0.5) < 0.6:
                self.logger.info("Adjusting for more complete responses")
        
        elif event.learning_type == LearningType.KNOWLEDGE_GAP:
            # Flag knowledge areas for improvement
            topics = event.metadata.get("gap_details", {}).get("topics", [])
            self.logger.info(f"Identified knowledge gap in: {topics}")
    
    def generate_improvement_report(self) -> Dict[str, Any]:
        """สร้างรายงานการปรับปรุง"""
        now = datetime.now()
        
        # Analyze learning events from last 7 days
        recent_events = [
            event for event in self.learning_events
            if (now - event.timestamp).days <= 7
        ]
        
        # Count learning types
        learning_type_counts = Counter([event.learning_type.value for event in recent_events])
        
        # Calculate average confidence and impact
        avg_confidence = np.mean([event.confidence for event in recent_events]) if recent_events else 0
        avg_impact = np.mean([event.impact_score for event in recent_events]) if recent_events else 0
        
        # Identify top improvement areas
        improvement_areas = []
        
        # Knowledge gaps
        knowledge_gaps = [event for event in recent_events 
                         if event.learning_type == LearningType.KNOWLEDGE_GAP]
        if knowledge_gaps:
            topics = []
            for event in knowledge_gaps:
                gap_topics = event.metadata.get("gap_details", {}).get("topics", [])
                topics.extend(gap_topics)
            
            topic_counts = Counter(topics)
            improvement_areas.append({
                "area": "knowledge_gaps",
                "priority": "high",
                "details": dict(topic_counts.most_common(3))
            })
        
        # Response quality issues
        quality_events = [event for event in recent_events 
                         if event.learning_type == LearningType.RESPONSE_QUALITY]
        if quality_events:
            low_quality = [event for event in quality_events if event.impact_score < 0.6]
            if low_quality:
                improvement_areas.append({
                    "area": "response_quality",
                    "priority": "medium",
                    "details": f"{len(low_quality)} low quality responses detected"
                })
        
        return {
            "report_timestamp": now.isoformat(),
            "learning_summary": {
                "total_events": len(recent_events),
                "learning_type_distribution": dict(learning_type_counts),
                "average_confidence": avg_confidence,
                "average_impact": avg_impact
            },
            "improvement_areas": improvement_areas,
            "user_patterns_learned": len(self.user_patterns),
            "conversation_patterns_identified": len(self.conversation_patterns),
            "recommendations": self._generate_improvement_recommendations()
        }
    
    def _generate_improvement_recommendations(self) -> List[str]:
        """สร้างข้อแนะนำการปรับปรุง"""
        recommendations = []
        
        # Analyze patterns to generate recommendations
        if len(self.learning_events) > 10:
            quality_events = [e for e in self.learning_events 
                             if e.learning_type == LearningType.RESPONSE_QUALITY]
            
            if quality_events:
                avg_quality = np.mean([e.impact_score for e in quality_events])
                if avg_quality < 0.7:
                    recommendations.append("Focus on improving response quality through better context understanding")
        
        # Check knowledge gaps
        gap_events = [e for e in self.learning_events 
                     if e.learning_type == LearningType.KNOWLEDGE_GAP]
        if len(gap_events) > 3:
            recommendations.append("Expand knowledge base in frequently questioned topics")
        
        # User pattern insights
        if len(self.user_patterns) > 5:
            recommendations.append("Leverage user patterns for more personalized responses")
        
        return recommendations
    
    def _save_learning_data(self):
        """บันทึกข้อมูลการเรียนรู้"""
        # Save learning events
        events_file = self.data_dir / "learning_events.json"
        with open(events_file, 'w', encoding='utf-8') as f:
            events_data = [event.to_dict() for event in self.learning_events]
            json.dump(events_data, f, ensure_ascii=False, indent=2)
        
        # Save user patterns
        patterns_file = self.data_dir / "user_patterns.json"
        with open(patterns_file, 'w', encoding='utf-8') as f:
            json.dump(dict(self.user_patterns), f, ensure_ascii=False, indent=2)
        
        # Save conversation patterns
        conv_patterns_file = self.data_dir / "conversation_patterns.json"
        with open(conv_patterns_file, 'w', encoding='utf-8') as f:
            # Convert datetime objects for JSON serialization
            serializable_patterns = {}
            for pattern, data_list in self.conversation_patterns.items():
                serializable_patterns[pattern] = [
                    {
                        "timestamp": item["timestamp"].isoformat(),
                        "effectiveness": item["effectiveness"]
                    } for item in data_list
                ]
            json.dump(serializable_patterns, f, ensure_ascii=False, indent=2)
        
        self.logger.info("Learning data saved successfully")
    
    def _load_learning_data(self):
        """โหลดข้อมูลการเรียนรู้"""
        try:
            # Load learning events
            events_file = self.data_dir / "learning_events.json"
            if events_file.exists():
                with open(events_file, 'r', encoding='utf-8') as f:
                    events_data = json.load(f)
                    
                for event_dict in events_data:
                    event = LearningEvent(
                        event_id=event_dict["event_id"],
                        timestamp=datetime.fromisoformat(event_dict["timestamp"]),
                        learning_type=LearningType(event_dict["learning_type"]),
                        source_data=event_dict["source_data"],
                        pattern_detected=event_dict["pattern_detected"],
                        confidence=event_dict["confidence"],
                        impact_score=event_dict["impact_score"],
                        metadata=event_dict["metadata"]
                    )
                    self.learning_events.append(event)
            
            # Load user patterns
            patterns_file = self.data_dir / "user_patterns.json"
            if patterns_file.exists():
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                    self.user_patterns.update(patterns_data)
            
            # Load conversation patterns
            conv_patterns_file = self.data_dir / "conversation_patterns.json"
            if conv_patterns_file.exists():
                with open(conv_patterns_file, 'r', encoding='utf-8') as f:
                    patterns_data = json.load(f)
                    
                for pattern, data_list in patterns_data.items():
                    self.conversation_patterns[pattern] = [
                        {
                            "timestamp": datetime.fromisoformat(item["timestamp"]),
                            "effectiveness": item["effectiveness"]
                        } for item in data_list
                    ]
            
            self.logger.info("Learning data loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load learning data: {e}")
    
    def shutdown(self):
        """ปิดระบบและบันทึกข้อมูล"""
        self._save_learning_data()
        self.logger.info("Self-Improvement System shutdown complete")