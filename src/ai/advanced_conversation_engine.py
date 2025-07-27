"""
Advanced Conversation Engine for JARVIS
ระบบสนทนาขั้นสูงที่มีความเข้าใจบริบทลึกและการตอบสนองที่เฉลียวฉลาด
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import re
from enum import Enum

class ConversationMood(Enum):
    """โหมดการสนทนาของ JARVIS"""
    CASUAL = "casual"           # สบายๆ เป็นกันเอง
    PROFESSIONAL = "professional"  # เป็นทางการ
    EDUCATIONAL = "educational"  # โหมดสอน
    ANALYTICAL = "analytical"   # วิเคราะห์เชิงลึก
    CREATIVE = "creative"       # สร้างสรรค์
    SUPPORTIVE = "supportive"   # ให้กำลังใจ

class ResponseStyle(Enum):
    """รูปแบบการตอบ"""
    CONCISE = "concise"         # กระชับ
    DETAILED = "detailed"       # ละเอียด
    STORYTELLING = "storytelling"  # เล่าเรื่อง
    INTERACTIVE = "interactive"  # โต้ตอบ

@dataclass
class ConversationContext:
    """บริบทการสนทนา"""
    user_id: str
    session_id: str
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_topic: Optional[str] = None
    mood: ConversationMood = ConversationMood.CASUAL
    response_style: ResponseStyle = ResponseStyle.DETAILED
    language_preference: str = "th"
    expertise_level: str = "intermediate"  # beginner, intermediate, expert
    interests: List[str] = None
    last_interaction: Optional[datetime] = None
    conversation_depth: int = 0
    
    def __post_init__(self):
        if self.interests is None:
            self.interests = []
        if self.last_interaction is None:
            self.last_interaction = datetime.now()

@dataclass
class SmartResponse:
    """การตอบที่ฉลาด"""
    text: str
    confidence: float
    reasoning: str
    suggestions: List[str]
    follow_up_questions: List[str]
    mood_detected: Optional[str] = None
    learning_opportunities: List[str] = None
    
    def __post_init__(self):
        if self.learning_opportunities is None:
            self.learning_opportunities = []

class AdvancedConversationEngine:
    """เครื่องมือสนทนาขั้นสูงของ JARVIS"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Context management
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.conversation_patterns: Dict[str, Any] = {}
        self.personality_traits: Dict[str, float] = {
            "helpfulness": 0.95,
            "creativity": 0.8,
            "formality": 0.6,
            "empathy": 0.9,
            "technical_depth": 0.85,
            "cultural_awareness": 0.9
        }
        
        # Knowledge domains
        self.knowledge_domains = {
            "technology": ["AI", "programming", "software", "hardware", "cybersecurity"],
            "science": ["physics", "chemistry", "biology", "mathematics", "astronomy"],
            "culture": ["thai culture", "traditions", "festivals", "food", "language"],
            "business": ["entrepreneurship", "marketing", "management", "finance"],
            "lifestyle": ["health", "fitness", "travel", "entertainment", "education"]
        }
        
        # Load conversation patterns
        self._load_conversation_patterns()
        
        self.logger.info("Advanced Conversation Engine initialized")
    
    def _load_conversation_patterns(self):
        """โหลดรูปแบบการสนทนา"""
        patterns_file = Path("data/conversation_patterns.json")
        if patterns_file.exists():
            try:
                with open(patterns_file, 'r', encoding='utf-8') as f:
                    self.conversation_patterns = json.load(f)
            except Exception as e:
                self.logger.warning(f"Could not load conversation patterns: {e}")
                self.conversation_patterns = self._get_default_patterns()
        else:
            self.conversation_patterns = self._get_default_patterns()
    
    def _get_default_patterns(self) -> Dict[str, Any]:
        """รูปแบบการสนทนาเริ่มต้น"""
        return {
            "greetings": {
                "thai_morning": ["สวัสดีตอนเช้าครับ", "อรุณสวัสดิ์ครับ", "เช้าดีครับ"],
                "thai_afternoon": ["สวัสดีตอนบ่ายครับ", "บ่ายดีครับ"],
                "thai_evening": ["สวัสดีตอนเย็นครับ", "เย็นดีครับ", "ค่ำดีครับ"],
                "english": ["Hello!", "Hi there!", "Good to see you!"]
            },
            "acknowledgments": {
                "thai": ["เข้าใจแล้วครับ", "รับทราบครับ", "ครับผม", "ใช่ครับ"],
                "english": ["I understand", "Got it", "That makes sense", "Absolutely"]
            },
            "thinking_phrases": {
                "thai": ["ให้ผมคิดดูสักครู่นะครับ", "นี่เป็นคำถามที่น่าสนใจ", "ให้ผมวิเคราะห์ดู"],
                "english": ["Let me think about that", "That's an interesting question", "Let me analyze this"]
            }
        }
    
    def start_conversation(self, user_id: str, session_id: str, 
                          user_preferences: Optional[Dict[str, Any]] = None) -> ConversationContext:
        """เริ่มการสนทนาใหม่"""
        context = ConversationContext(
            user_id=user_id,
            session_id=session_id,
            conversation_history=[],
            user_preferences=user_preferences or {},
            last_interaction=datetime.now()
        )
        
        # Set preferences from user data
        if user_preferences:
            context.language_preference = user_preferences.get("language", "th")
            context.mood = ConversationMood(user_preferences.get("mood", "casual"))
            context.response_style = ResponseStyle(user_preferences.get("response_style", "detailed"))
            context.expertise_level = user_preferences.get("expertise_level", "intermediate")
            context.interests = user_preferences.get("interests", [])
        
        self.active_contexts[session_id] = context
        
        self.logger.info(f"Started conversation for user {user_id}, session {session_id}")
        return context
    
    def process_message(self, session_id: str, message: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> SmartResponse:
        """ประมวลผลข้อความและสร้างการตอบที่ฉลาด"""
        if session_id not in self.active_contexts:
            self.logger.warning(f"No active context for session {session_id}")
            # Create default context
            self.start_conversation("unknown", session_id)
        
        context = self.active_contexts[session_id]
        
        # Analyze message
        analysis = self._analyze_message(message, context)
        
        # Update context
        self._update_context(context, message, analysis)
        
        # Generate smart response
        response = self._generate_smart_response(context, message, analysis)
        
        # Add to conversation history
        context.conversation_history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "assistant_response": response.text,
            "analysis": analysis,
            "confidence": response.confidence
        })
        
        context.last_interaction = datetime.now()
        context.conversation_depth += 1
        
        return response
    
    def _analyze_message(self, message: str, context: ConversationContext) -> Dict[str, Any]:
        """วิเคราะห์ข้อความเชิงลึก"""
        analysis = {
            "language": "th" if self._is_thai(message) else "en",
            "intent": self._detect_intent(message),
            "emotion": self._detect_emotion(message),
            "complexity": self._assess_complexity(message),
            "topics": self._extract_topics(message),
            "question_type": self._classify_question(message),
            "urgency": self._assess_urgency(message),
            "context_references": self._find_context_references(message, context)
        }
        
        return analysis
    
    def _is_thai(self, text: str) -> bool:
        """ตรวจสอบว่าเป็นภาษาไทยหรือไม่"""
        thai_chars = re.findall(r'[\u0E00-\u0E7F]', text)
        return len(thai_chars) > len(text) * 0.3
    
    def _detect_intent(self, message: str) -> str:
        """ตรวจจับเจตนา"""
        message_lower = message.lower()
        
        # Question patterns
        question_patterns = [
            r'\?', r'คือ', r'ทำไม', r'อะไร', r'ยังไง', r'เป็นยังไง',
            r'what', r'why', r'how', r'when', r'where', r'who'
        ]
        if any(re.search(pattern, message_lower) for pattern in question_patterns):
            return "question"
        
        # Request patterns
        request_patterns = [
            r'ช่วย', r'กรุณา', r'ขอ', r'please', r'can you', r'could you'
        ]
        if any(re.search(pattern, message_lower) for pattern in request_patterns):
            return "request"
        
        # Greeting patterns
        greeting_patterns = [
            r'สวัสดี', r'หวัดดี', r'ว้าส', r'hello', r'hi', r'hey'
        ]
        if any(re.search(pattern, message_lower) for pattern in greeting_patterns):
            return "greeting"
        
        return "statement"
    
    def _detect_emotion(self, message: str) -> str:
        """ตรวจจับอารมณ์"""
        message_lower = message.lower()
        
        # Positive emotions
        positive_patterns = [
            r'ดี', r'เยี่ยม', r'สุดยอด', r'ชอบ', r'รัก', r'มีความสุข',
            r'good', r'great', r'awesome', r'love', r'happy', r'excellent'
        ]
        if any(re.search(pattern, message_lower) for pattern in positive_patterns):
            return "positive"
        
        # Negative emotions
        negative_patterns = [
            r'เศร้า', r'โกรธ', r'หงุดหงิด', r'ผิดหวัง', r'เครียด',
            r'sad', r'angry', r'frustrated', r'disappointed', r'stressed'
        ]
        if any(re.search(pattern, message_lower) for pattern in negative_patterns):
            return "negative"
        
        # Confused/uncertain
        confused_patterns = [
            r'งง', r'สับสน', r'ไม่เข้าใจ', r'confused', r'don\'t understand'
        ]
        if any(re.search(pattern, message_lower) for pattern in confused_patterns):
            return "confused"
        
        return "neutral"
    
    def _assess_complexity(self, message: str) -> str:
        """ประเมินความซับซ้อน"""
        word_count = len(message.split())
        
        if word_count < 5:
            return "simple"
        elif word_count < 15:
            return "moderate"
        else:
            return "complex"
    
    def _extract_topics(self, message: str) -> List[str]:
        """สกัดหัวข้อจากข้อความ"""
        topics = []
        message_lower = message.lower()
        
        for domain, keywords in self.knowledge_domains.items():
            for keyword in keywords:
                if keyword.lower() in message_lower:
                    topics.append(domain)
                    break
        
        return topics
    
    def _classify_question(self, message: str) -> str:
        """จำแนกประเภทคำถาม"""
        if not message.strip().endswith('?') and not any(q in message.lower() for q in ['คือ', 'ทำไม', 'อะไร', 'what', 'why', 'how']):
            return "not_question"
        
        # Factual questions
        factual_patterns = [r'คือ', r'what is', r'what are', r'define']
        if any(re.search(pattern, message.lower()) for pattern in factual_patterns):
            return "factual"
        
        # How-to questions
        howto_patterns = [r'ยังไง', r'อย่างไร', r'how to', r'how can']
        if any(re.search(pattern, message.lower()) for pattern in howto_patterns):
            return "how_to"
        
        # Why questions
        why_patterns = [r'ทำไม', r'เพราะอะไร', r'why']
        if any(re.search(pattern, message.lower()) for pattern in why_patterns):
            return "why"
        
        return "general"
    
    def _assess_urgency(self, message: str) -> str:
        """ประเมินความเร่งด่วน"""
        urgent_patterns = [
            r'ด่วน', r'เร่ง', r'urgent', r'emergency', r'asap', r'immediately'
        ]
        if any(re.search(pattern, message.lower()) for pattern in urgent_patterns):
            return "high"
        
        return "normal"
    
    def _find_context_references(self, message: str, context: ConversationContext) -> List[str]:
        """หาการอ้างอิงบริบทก่อนหน้า"""
        references = []
        
        # Check for references to previous conversation
        reference_patterns = [
            r'ที่พูดไป', r'เมื่อกี้', r'ที่แล้ว', r'เมื่อไหร่',
            r'previously', r'earlier', r'before', r'you said'
        ]
        
        if any(re.search(pattern, message.lower()) for pattern in reference_patterns):
            references.append("previous_conversation")
        
        # Check for topic continuation
        if context.current_topic and context.current_topic in message.lower():
            references.append("current_topic")
        
        return references
    
    def _update_context(self, context: ConversationContext, message: str, analysis: Dict[str, Any]):
        """อัปเดตบริบทการสนทนา"""
        # Update current topic
        if analysis["topics"]:
            context.current_topic = analysis["topics"][0]
        
        # Adjust mood based on emotion
        if analysis["emotion"] == "positive":
            context.mood = ConversationMood.CASUAL
        elif analysis["emotion"] == "negative":
            context.mood = ConversationMood.SUPPORTIVE
        elif analysis["emotion"] == "confused":
            context.mood = ConversationMood.EDUCATIONAL
        
        # Adjust response style based on complexity
        if analysis["complexity"] == "simple":
            context.response_style = ResponseStyle.CONCISE
        elif analysis["complexity"] == "complex":
            context.response_style = ResponseStyle.DETAILED
    
    def _generate_smart_response(self, context: ConversationContext, 
                               message: str, analysis: Dict[str, Any]) -> SmartResponse:
        """สร้างการตอบที่ฉลาด"""
        # Select appropriate response template
        template = self._select_response_template(context, analysis)
        
        # Generate base response
        base_response = self._generate_base_response(message, analysis, context)
        
        # Add personality and style
        styled_response = self._apply_personality_style(base_response, context, analysis)
        
        # Generate follow-up questions
        follow_ups = self._generate_follow_up_questions(analysis, context)
        
        # Generate suggestions
        suggestions = self._generate_suggestions(analysis, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(analysis)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(analysis, context)
        
        # Identify learning opportunities
        learning_opportunities = self._identify_learning_opportunities(analysis, context)
        
        return SmartResponse(
            text=styled_response,
            confidence=confidence,
            reasoning=reasoning,
            suggestions=suggestions,
            follow_up_questions=follow_ups,
            mood_detected=analysis["emotion"],
            learning_opportunities=learning_opportunities
        )
    
    def _select_response_template(self, context: ConversationContext, 
                                analysis: Dict[str, Any]) -> str:
        """เลือกเทมเพลตการตอบ"""
        if analysis["intent"] == "greeting":
            return "greeting"
        elif analysis["intent"] == "question":
            if analysis["question_type"] == "factual":
                return "factual_answer"
            elif analysis["question_type"] == "how_to":
                return "how_to_guide"
            else:
                return "general_answer"
        elif analysis["intent"] == "request":
            return "assistance"
        else:
            return "general_response"
    
    def _generate_base_response(self, message: str, analysis: Dict[str, Any], 
                              context: ConversationContext) -> str:
        """สร้างการตอบพื้นฐาน"""
        # This would integrate with your existing AI engine
        # For now, provide intelligent default responses
        
        if analysis["intent"] == "greeting":
            if context.language_preference == "th":
                greetings = self.conversation_patterns["greetings"]["thai_morning"]
                return f"{greetings[0]} ยินดีที่ได้พูดคุยกับคุณครับ! 😊"
            else:
                return "Hello! I'm delighted to chat with you! 😊"
        
        elif analysis["intent"] == "question":
            if context.language_preference == "th":
                thinking = self.conversation_patterns["thinking_phrases"]["thai"][0]
                return f"{thinking} 🤔\n\nเกี่ยวกับคำถาม '{message}' นี้..."
            else:
                return f"Let me think about that 🤔\n\nRegarding your question about '{message}'..."
        
        else:
            if context.language_preference == "th":
                return f"ผมเข้าใจสิ่งที่คุณพูดครับ เกี่ยวกับ '{message}' ผมคิดว่า..."
            else:
                return f"I understand what you're saying. Regarding '{message}', I think..."
    
    def _apply_personality_style(self, response: str, context: ConversationContext, 
                               analysis: Dict[str, Any]) -> str:
        """ปรับสไตล์ตามบุคลิกและโหมด"""
        # Add empathy for negative emotions
        if analysis["emotion"] == "negative":
            if context.language_preference == "th":
                response = "ผมเข้าใจความรู้สึกของคุณครับ 💙 " + response
            else:
                response = "I understand how you feel 💙 " + response
        
        # Add encouragement for learning opportunities
        if analysis["topics"] and context.mood == ConversationMood.EDUCATIONAL:
            if context.language_preference == "th":
                response += "\n\n🎯 นี่เป็นโอกาสที่ดีในการเรียนรู้เรื่องนี้เพิ่มเติมครับ!"
            else:
                response += "\n\n🎯 This is a great opportunity to learn more about this topic!"
        
        return response
    
    def _generate_follow_up_questions(self, analysis: Dict[str, Any], 
                                    context: ConversationContext) -> List[str]:
        """สร้างคำถามติดตาม"""
        questions = []
        
        if analysis["topics"]:
            topic = analysis["topics"][0]
            if context.language_preference == "th":
                questions.append(f"คุณอยากเรียนรู้เพิ่มเติมเกี่ยวกับ {topic} ไหมครับ?")
                questions.append(f"มีประสบการณ์เกี่ยวกับ {topic} มาก่อนไหมครับ?")
            else:
                questions.append(f"Would you like to learn more about {topic}?")
                questions.append(f"Do you have any experience with {topic}?")
        
        if analysis["intent"] == "question" and analysis["question_type"] == "how_to":
            if context.language_preference == "th":
                questions.append("คุณมีประสบการณ์พื้นฐานในเรื่องนี้อยู่แล้วหรือยังครับ?")
                questions.append("อยากให้อธิบายแบบง่ายๆ หรือละเอียดครับ?")
            else:
                questions.append("Do you already have some basic experience with this?")
                questions.append("Would you like a simple or detailed explanation?")
        
        return questions
    
    def _generate_suggestions(self, analysis: Dict[str, Any], 
                           context: ConversationContext) -> List[str]:
        """สร้างข้อเสนอแนะ"""
        suggestions = []
        
        if analysis["emotion"] == "confused":
            if context.language_preference == "th":
                suggestions.extend([
                    "ลองถามคำถามที่เฉพาะเจาะจงมากขึ้น",
                    "ขอยกตัวอย่างประกอบ",
                    "แยกย่อยคำถามเป็นส่วนเล็กๆ"
                ])
            else:
                suggestions.extend([
                    "Try asking more specific questions",
                    "Ask for examples",
                    "Break down your question into smaller parts"
                ])
        
        if analysis["topics"]:
            topic = analysis["topics"][0]
            if context.language_preference == "th":
                suggestions.append(f"ลองศึกษา {topic} เพิ่มเติมจากแหล่งข้อมูลที่เชื่อถือได้")
            else:
                suggestions.append(f"Explore more about {topic} from reliable sources")
        
        return suggestions
    
    def _calculate_confidence(self, analysis: Dict[str, Any]) -> float:
        """คำนวณความเชื่อมั่น"""
        base_confidence = 0.7
        
        # Increase confidence for clear intents
        if analysis["intent"] in ["greeting", "question"]:
            base_confidence += 0.1
        
        # Adjust for complexity
        if analysis["complexity"] == "simple":
            base_confidence += 0.15
        elif analysis["complexity"] == "complex":
            base_confidence -= 0.1
        
        # Adjust for emotion clarity
        if analysis["emotion"] != "neutral":
            base_confidence += 0.05
        
        return min(max(base_confidence, 0.0), 1.0)
    
    def _generate_reasoning(self, analysis: Dict[str, Any], 
                          context: ConversationContext) -> str:
        """สร้างเหตุผลการตอบ"""
        reasoning_parts = []
        
        reasoning_parts.append(f"Intent detected: {analysis['intent']}")
        reasoning_parts.append(f"Emotion: {analysis['emotion']}")
        
        if analysis["topics"]:
            reasoning_parts.append(f"Topics: {', '.join(analysis['topics'])}")
        
        reasoning_parts.append(f"Response style: {context.response_style.value}")
        reasoning_parts.append(f"Conversation mood: {context.mood.value}")
        
        return " | ".join(reasoning_parts)
    
    def _identify_learning_opportunities(self, analysis: Dict[str, Any], 
                                       context: ConversationContext) -> List[str]:
        """ระบุโอกาสการเรียนรู้"""
        opportunities = []
        
        if analysis["question_type"] == "how_to":
            opportunities.append("hands_on_practice")
            opportunities.append("step_by_step_tutorial")
        
        if analysis["topics"]:
            for topic in analysis["topics"]:
                opportunities.append(f"deep_dive_{topic}")
        
        if analysis["emotion"] == "confused":
            opportunities.append("clarification_session")
            opportunities.append("basic_concepts_review")
        
        return opportunities
    
    def get_conversation_summary(self, session_id: str) -> Dict[str, Any]:
        """สรุปการสนทนา"""
        if session_id not in self.active_contexts:
            return {"error": "Session not found"}
        
        context = self.active_contexts[session_id]
        
        return {
            "session_id": session_id,
            "user_id": context.user_id,
            "conversation_length": len(context.conversation_history),
            "current_topic": context.current_topic,
            "mood": context.mood.value,
            "response_style": context.response_style.value,
            "language": context.language_preference,
            "conversation_depth": context.conversation_depth,
            "last_interaction": context.last_interaction.isoformat(),
            "topics_covered": list(set([
                topic for msg in context.conversation_history 
                if msg.get("analysis", {}).get("topics") 
                for topic in msg["analysis"]["topics"]
            ])),
            "dominant_emotion": self._get_dominant_emotion(context)
        }
    
    def _get_dominant_emotion(self, context: ConversationContext) -> str:
        """หาอารมณ์หลักในการสนทนา"""
        emotions = []
        for msg in context.conversation_history:
            if msg.get("analysis", {}).get("emotion"):
                emotions.append(msg["analysis"]["emotion"])
        
        if not emotions:
            return "neutral"
        
        # Find most common emotion
        from collections import Counter
        emotion_counts = Counter(emotions)
        return emotion_counts.most_common(1)[0][0]
    
    def save_context(self, session_id: str):
        """บันทึกบริบทการสนทนา"""
        if session_id not in self.active_contexts:
            return
        
        context = self.active_contexts[session_id]
        context_file = Path(f"data/contexts/{session_id}.json")
        context_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert context to dict for JSON serialization
        context_dict = asdict(context)
        context_dict["last_interaction"] = context.last_interaction.isoformat()
        
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_dict, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Context saved for session {session_id}")
    
    def load_context(self, session_id: str) -> Optional[ConversationContext]:
        """โหลดบริบทการสนทนา"""
        context_file = Path(f"data/contexts/{session_id}.json")
        
        if not context_file.exists():
            return None
        
        try:
            with open(context_file, 'r', encoding='utf-8') as f:
                context_dict = json.load(f)
            
            # Convert back to ConversationContext
            context_dict["last_interaction"] = datetime.fromisoformat(context_dict["last_interaction"])
            context_dict["mood"] = ConversationMood(context_dict["mood"])
            context_dict["response_style"] = ResponseStyle(context_dict["response_style"])
            
            context = ConversationContext(**context_dict)
            self.active_contexts[session_id] = context
            
            self.logger.info(f"Context loaded for session {session_id}")
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to load context for session {session_id}: {e}")
            return None