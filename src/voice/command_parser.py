"""
Natural Voice Command Parser for JARVIS Voice Assistant
Interprets natural language commands in both Thai and English using DeepSeek-R1
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple, Union
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass, field
from enum import Enum
import json

class CommandType(Enum):
    """Types of voice commands"""
    QUESTION = "question"
    ACTION = "action"
    REQUEST = "request"
    CONVERSATION = "conversation"
    SYSTEM = "system"
    UNKNOWN = "unknown"

class CommandPriority(Enum):
    """Command priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class ParsedCommand:
    """Parsed voice command structure"""
    original_text: str
    cleaned_text: str
    command_type: CommandType
    intent: str
    entities: Dict[str, Any] = field(default_factory=dict)
    parameters: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    language: str = "en"
    priority: CommandPriority = CommandPriority.NORMAL
    requires_response: bool = True
    context_needed: bool = False
    suggested_actions: List[str] = field(default_factory=list)

class VoiceCommandParser(QObject):
    """Natural language voice command parser"""
    
    # Signals
    command_parsed = pyqtSignal(dict)
    intent_detected = pyqtSignal(str, float)
    entities_extracted = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("command_parser", {})
        
        # Command patterns and templates
        self.command_patterns = self._initialize_command_patterns()
        self.intent_templates = self._initialize_intent_templates()
        self.entity_patterns = self._initialize_entity_patterns()
        
        # Thai language integration
        self.thai_processor = None
        self._initialize_thai_support()
        
    def _initialize_command_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize command recognition patterns"""
        return {
            # Questions - More precise patterns
            "what_questions": {
                "patterns": [
                    r"\bwhat\s+(?:is|are|was|were|will be|can|could|do|does|did)\b",
                    r"\bอะไร(?:คือ|เป็น|ได้|ทำ)\b",
                    r"\b(?:tell me|explain)\s+(?:about|what)\b",
                    r"\b(?:บอก|อธิบาย)(?:เกี่ยวกับ|ว่า|ให้ฟัง)\b",
                    r"คือ(?:อะไร|ไร|ไหน)"
                ],
                "intent": "information_request",
                "type": CommandType.QUESTION,
                "priority": 0.9
            },
            
            "how_questions": {
                "patterns": [
                    r"\bhow\s+(?:do|does|did|can|could|will|to)\b",
                    r"(?:ทำ|ใช้)(?:อย่างไร|ยังไง|ไง)",
                    r"\bhow\b.*?(?:work|works|function)\b",
                    r"สอน.*?(?:ทำ|ใช้)",
                    r"(?:อย่างไร|ยังไง|ไง)"
                ],
                "intent": "how_to_request",
                "type": CommandType.QUESTION,
                "priority": 0.9
            },
            
            "why_questions": {
                "patterns": [
                    r"\bwhy\s+(?:is|are|was|were|do|does|did|can|could)\b",
                    r"ทำไม.*?(?:เป็น|คือ|ได้|ต้อง)",
                    r"เพราะ(?:อะไร|เหตุไร)",
                    r"(?:what|อะไร).*?(?:reason|เหตุผล)",
                    r"\bทำไม\b"
                ],
                "intent": "explanation_request", 
                "type": CommandType.QUESTION,
                "priority": 0.9
            },
            
            # Actions
            "action_commands": {
                "patterns": [
                    r"(?:please )?(?:can you |could you )?(?:help me )?(?:to )?(\w+)",
                    r"(?:กรุณา|ช่วย|โปรด)?(?:.*?)(?:ให้|หน่อย|ด้วย)",
                    r"(?:start|begin|เริ่ม|เปิด|ตั้ง)",
                    r"(?:stop|end|หยุด|ปิด|จบ)",
                    r"(?:open|close|save|delete|เปิด|ปิด|บันทึก|ลบ)"
                ],
                "intent": "action_request",
                "type": CommandType.ACTION
            },
            
            # Greetings and conversation
            "greetings": {
                "patterns": [
                    r"\b(?:hello|hi|hey|good morning|good afternoon|good evening)\b",
                    r"\b(?:สวัสดี|หวัดดี|ดี|เฮ้|ฮาย)\b",
                    r"\b(?:how are you|how's it going|what's up)\b",
                    r"\b(?:สบายดี|เป็นไง|ไง|อย่างไร)\b"
                ],
                "intent": "greeting",
                "type": CommandType.CONVERSATION,
                "priority": 0.95
            },
            
            # System commands
            "system_commands": {
                "patterns": [
                    r"\b(?:exit|quit|goodbye|bye|stop listening)\b",
                    r"\b(?:ออก|ปิด|หยุด|บาย|ลาก่อน)\b",
                    r"(?:volume|เสียง).*?(?:up|down|ขึ้น|ลง)",
                    r"\b(?:settings|config|ตั้งค่า|การตั้งค่า)\b"
                ],
                "intent": "system_control",
                "type": CommandType.SYSTEM,
                "priority": 0.8
            }
        }
    
    def _initialize_intent_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize intent classification templates"""
        return {
            "information_request": {
                "description": "User wants information about something",
                "examples": [
                    "What is artificial intelligence?",
                    "Tell me about machine learning",
                    "ปัญญาประดิษฐ์คืออะไร",
                    "บอกเกี่ยวกับ machine learning"
                ],
                "response_type": "informative",
                "requires_knowledge": True
            },
            
            "how_to_request": {
                "description": "User wants to know how to do something",
                "examples": [
                    "How do I use this feature?",
                    "How to set up voice recognition?",
                    "ใช้ฟีเจอร์นี้ยังไง",
                    "ตั้งค่าการรู้จำเสียงอย่างไร"
                ],
                "response_type": "instructional",
                "requires_knowledge": True
            },
            
            "explanation_request": {
                "description": "User wants an explanation of why something is",
                "examples": [
                    "Why is this important?",
                    "Why do we need AI?",
                    "ทำไมเรื่องนี้สำคัญ",
                    "ทำไมเราต้องมี AI"
                ],
                "response_type": "explanatory",
                "requires_knowledge": True
            },
            
            "action_request": {
                "description": "User wants to perform an action",
                "examples": [
                    "Start the application",
                    "Open the settings",
                    "เริ่มโปรแกรม",
                    "เปิดการตั้งค่า"
                ],
                "response_type": "action",
                "requires_execution": True
            },
            
            "greeting": {
                "description": "User is greeting or starting conversation",
                "examples": [
                    "Hello JARVIS",
                    "Good morning",
                    "สวัสดีจาร์วิส",
                    "อรุณสวัสดิ์"
                ],
                "response_type": "conversational",
                "requires_response": True
            },
            
            "system_control": {
                "description": "User wants to control system functions",
                "examples": [
                    "Turn up the volume",
                    "Change settings",
                    "เปิดเสียงให้ดังขึ้น",
                    "เปลี่ยนการตั้งค่า"
                ],
                "response_type": "system",
                "requires_execution": True
            }
        }
    
    def _initialize_entity_patterns(self) -> Dict[str, List[str]]:
        """Initialize entity extraction patterns"""
        return {
            "time": [
                r"\b(?:today|tomorrow|yesterday|tonight|morning|afternoon|evening)\b",
                r"\b(?:วันนี้|พรุ่งนี้|เมื่อวาน|คืนนี้|เช้า|บ่าย|เย็น)\b",
                r"\b(?:\d{1,2}:\d{2}(?:\s*(?:am|pm))?)\b",
                r"\b(?:at|ตอน|เวลา)\s+\d{1,2}(?::\d{2})?\b"
            ],
            
            "numbers": [
                r"\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b",
                r"\b(?:หนึ่ง|สอง|สาม|สี่|ห้า|หก|เจ็ด|แปด|เก้า|สิบ)\b",
                r"\b\d+\b"
            ],
            
            "actions": [
                r"\b(?:open|close|start|stop|create|delete|save|load)\b",
                r"\b(?:เปิด|ปิด|เริ่ม|หยุด|สร้าง|ลบ|บันทึก|โหลด)\b"
            ],
            
            "objects": [
                r"\b(?:file|document|application|program|window|folder)\b",
                r"\b(?:ไฟล์|เอกสาร|แอปพลิเคชัน|โปรแกรม|หน้าต่าง|โฟลเดอร์)\b"
            ],
            
            "locations": [
                r"\b(?:here|there|home|work|office|desktop)\b",
                r"\b(?:ที่นี่|ที่นั่น|บ้าน|ที่ทำงาน|สำนักงาน|เดสก์ท็อป)\b"
            ]
        }
    
    def _initialize_thai_support(self):
        """Initialize Thai language support"""
        try:
            # Try multiple import paths to handle different execution contexts
            thai_processor = None
            
            # Try relative import first (when running from src/)
            try:
                from ..features.thai_language_enhanced import ThaiLanguageProcessor
                thai_processor = ThaiLanguageProcessor
            except ImportError:
                pass
            
            # Try absolute import from src (when running from project root)
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
            
            # Try direct import with absolute path
            if thai_processor is None:
                try:
                    import sys
                    import os
                    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
                    src_path = os.path.join(project_root, 'src')
                    if src_path not in sys.path:
                        sys.path.insert(0, src_path)
                    from src.features.thai_language_enhanced import ThaiLanguageProcessor
                    thai_processor = ThaiLanguageProcessor
                except ImportError:
                    pass
            
            if thai_processor is not None:
                self.thai_processor = thai_processor(self.config)
                self.logger.info("Thai language support initialized successfully")
            else:
                raise ImportError("Could not import ThaiLanguageProcessor from any path")
                
        except Exception as e:
            self.logger.warning(f"Thai language support not available: {e}")
            self.thai_processor = None
    
    def parse_command(self, text: str, language: str = "auto") -> ParsedCommand:
        """Parse natural language command"""
        try:
            # Clean and normalize text
            cleaned_text = self._clean_text(text)
            
            # Detect language if auto
            if language == "auto":
                language = self._detect_language(cleaned_text)
            
            # Process Thai text if needed
            enhanced_context = None
            if language in ["th", "thai"] and self.thai_processor:
                enhanced_context = self.thai_processor.enhance_for_ai_processing(cleaned_text)
                cleaned_text = enhanced_context.get("processed_text", cleaned_text)
            
            # Extract intent
            intent, confidence = self._extract_intent(cleaned_text, language)
            
            # Determine command type
            command_type = self._determine_command_type(cleaned_text, intent)
            
            # Extract entities
            entities = self._extract_entities(cleaned_text, language)
            
            # Extract parameters
            parameters = self._extract_parameters(cleaned_text, intent, entities)
            
            # Determine priority
            priority = self._determine_priority(intent, entities, parameters)
            
            # Generate suggested actions
            suggested_actions = self._generate_suggested_actions(intent, entities, parameters)
            
            # Create parsed command
            parsed_command = ParsedCommand(
                original_text=text,
                cleaned_text=cleaned_text,
                command_type=command_type,
                intent=intent,
                entities=entities,
                parameters=parameters,
                confidence=confidence,
                language=language,
                priority=priority,
                requires_response=self._requires_response(intent),
                context_needed=self._needs_context(intent),
                suggested_actions=suggested_actions
            )
            
            # Add Thai context if available
            if enhanced_context:
                parsed_command.parameters["thai_context"] = enhanced_context
            
            # Emit signals
            self.command_parsed.emit(parsed_command.__dict__)
            self.intent_detected.emit(intent, confidence)
            self.entities_extracted.emit(entities)
            
            return parsed_command
            
        except Exception as e:
            self.logger.error(f"Command parsing failed: {e}")
            self.error_occurred.emit(f"Command parsing failed: {e}")
            
            return ParsedCommand(
                original_text=text,
                cleaned_text=text,
                command_type=CommandType.UNKNOWN,
                intent="unknown",
                confidence=0.0
            )
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize input text"""
        try:
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Remove common filler words in English
            fillers_en = ["um", "uh", "er", "ah", "like", "you know"]
            for filler in fillers_en:
                text = re.sub(rf'\b{filler}\b', '', text, flags=re.IGNORECASE)
            
            # Remove common filler words in Thai
            fillers_th = ["เอ่อ", "อืม", "คือ", "ก็"]
            for filler in fillers_th:
                text = text.replace(filler, '')
            
            # Clean up multiple spaces again
            text = re.sub(r'\s+', ' ', text.strip())
            
            return text
            
        except Exception as e:
            self.logger.error(f"Text cleaning failed: {e}")
            return text
    
    def _detect_language(self, text: str) -> str:
        """Detect primary language of text"""
        try:
            # Count Thai vs English characters
            thai_chars = len(re.findall(r'[ก-๙]', text))
            english_chars = len(re.findall(r'[A-Za-z]', text))
            
            if thai_chars > english_chars:
                return "th"
            elif english_chars > 0:
                return "en"
            else:
                return "unknown"
                
        except Exception:
            return "en"  # Default to English
    
    def _extract_intent(self, text: str, language: str) -> Tuple[str, float]:
        """Extract intent from text"""
        try:
            text_lower = text.lower()
            best_intent = "unknown"
            best_confidence = 0.0
            best_priority = 0.0
            
            # Check against patterns with priority
            for pattern_group, config in self.command_patterns.items():
                pattern_priority = config.get("priority", 0.5)
                
                for pattern in config["patterns"]:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        base_confidence = self._calculate_pattern_confidence(pattern, text_lower)
                        # Apply priority weighting
                        weighted_confidence = base_confidence * pattern_priority
                        
                        # Prefer higher priority patterns or higher confidence
                        if (weighted_confidence > best_confidence or 
                            (pattern_priority > best_priority and weighted_confidence > 0.5)):
                            best_confidence = weighted_confidence
                            best_priority = pattern_priority
                            best_intent = config["intent"]
            
            # Apply language-specific adjustments
            if language == "th" and best_confidence > 0:
                best_confidence *= 1.05  # Slight boost for Thai
            
            return best_intent, min(1.0, best_confidence)
            
        except Exception as e:
            self.logger.error(f"Intent extraction failed: {e}")
            return "unknown", 0.0
    
    def _determine_command_type(self, text: str, intent: str) -> CommandType:
        """Determine command type based on text and intent"""
        try:
            # Use intent mapping first
            intent_to_type = {
                "information_request": CommandType.QUESTION,
                "how_to_request": CommandType.QUESTION,
                "explanation_request": CommandType.QUESTION,
                "action_request": CommandType.ACTION,
                "greeting": CommandType.CONVERSATION,
                "system_control": CommandType.SYSTEM
            }
            
            if intent in intent_to_type:
                return intent_to_type[intent]
            
            # Fallback to text analysis
            text_lower = text.lower()
            
            # Question indicators
            question_words = ["what", "how", "why", "when", "where", "who", "which",
                             "อะไร", "ทำไม", "เมื่อไร", "ที่ไหน", "ใครน", "อย่างไร"]
            if any(word in text_lower for word in question_words):
                return CommandType.QUESTION
            
            # Action indicators
            action_words = ["start", "stop", "open", "close", "create", "delete",
                           "เริ่ม", "หยุด", "เปิด", "ปิด", "สร้าง", "ลบ"]
            if any(word in text_lower for word in action_words):
                return CommandType.ACTION
            
            return CommandType.CONVERSATION
            
        except Exception:
            return CommandType.UNKNOWN
    
    def _extract_entities(self, text: str, language: str) -> Dict[str, Any]:
        """Extract entities from text"""
        try:
            entities = {}
            
            for entity_type, patterns in self.entity_patterns.items():
                matches = []
                for pattern in patterns:
                    found = re.findall(pattern, text, re.IGNORECASE)
                    matches.extend(found)
                
                if matches:
                    entities[entity_type] = list(set(matches))  # Remove duplicates
            
            return entities
            
        except Exception as e:
            self.logger.error(f"Entity extraction failed: {e}")
            return {}
    
    def _extract_parameters(self, text: str, intent: str, entities: Dict[str, Any]) -> Dict[str, Any]:
        """Extract parameters based on intent and entities"""
        try:
            parameters = {}
            
            # Copy entities as base parameters
            parameters.update(entities)
            
            # Intent-specific parameter extraction
            if intent == "information_request":
                # Extract topic
                topic_patterns = [
                    r"(?:about|เกี่ยวกับ)\s+([^?.!]+)",
                    r"(?:what is|อะไรคือ)\s+([^?.!]+)",
                    r"(?:tell me|บอกฉัน)(?:\s+about)?\s+([^?.!]+)"
                ]
                
                for pattern in topic_patterns:
                    match = re.search(pattern, text, re.IGNORECASE)
                    if match:
                        parameters["topic"] = match.group(1).strip()
                        break
            
            elif intent == "action_request":
                # Extract action and target
                action_match = re.search(r"(open|close|start|stop|create|delete|เปิด|ปิด|เริ่ม|หยุด|สร้าง|ลบ)\s+([^?.!]+)", text, re.IGNORECASE)
                if action_match:
                    parameters["action"] = action_match.group(1).lower()
                    parameters["target"] = action_match.group(2).strip()
            
            elif intent == "system_control":
                # Extract system command
                if "volume" in text.lower() or "เสียง" in text:
                    parameters["system_type"] = "audio"
                    if "up" in text.lower() or "ขึ้น" in text:
                        parameters["direction"] = "increase"
                    elif "down" in text.lower() or "ลง" in text:
                        parameters["direction"] = "decrease"
                
                if "settings" in text.lower() or "ตั้งค่า" in text:
                    parameters["system_type"] = "settings"
            
            return parameters
            
        except Exception as e:
            self.logger.error(f"Parameter extraction failed: {e}")
            return {}
    
    def _determine_priority(self, intent: str, entities: Dict[str, Any], parameters: Dict[str, Any]) -> CommandPriority:
        """Determine command priority"""
        try:
            # System commands are high priority
            if intent == "system_control":
                return CommandPriority.HIGH
            
            # Actions are normal priority
            if intent == "action_request":
                return CommandPriority.NORMAL
            
            # Questions and conversation are low priority
            return CommandPriority.LOW
            
        except Exception:
            return CommandPriority.NORMAL
    
    def _generate_suggested_actions(self, intent: str, entities: Dict[str, Any], parameters: Dict[str, Any]) -> List[str]:
        """Generate suggested actions based on parsed command"""
        try:
            suggestions = []
            
            if intent == "information_request":
                suggestions.append("search_knowledge_base")
                suggestions.append("provide_explanation")
                
                if "topic" in parameters:
                    suggestions.append("find_related_topics")
            
            elif intent == "action_request":
                if "action" in parameters:
                    suggestions.append(f"execute_{parameters['action']}")
                
                if "target" in parameters:
                    suggestions.append("validate_target_exists")
            
            elif intent == "greeting":
                suggestions.append("respond_greeting")
                suggestions.append("offer_assistance")
            
            elif intent == "system_control":
                if parameters.get("system_type") == "audio":
                    suggestions.append("adjust_volume")
                elif parameters.get("system_type") == "settings":
                    suggestions.append("open_settings")
            
            return suggestions
            
        except Exception as e:
            self.logger.error(f"Suggestion generation failed: {e}")
            return []
    
    def _requires_response(self, intent: str) -> bool:
        """Check if intent requires a response"""
        no_response_intents = ["system_control"]
        return intent not in no_response_intents
    
    def _needs_context(self, intent: str) -> bool:
        """Check if intent needs additional context"""
        context_intents = ["information_request", "explanation_request", "how_to_request"]
        return intent in context_intents
    
    def _calculate_pattern_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence score for pattern match"""
        try:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if not matches:
                return 0.0
            
            # Base confidence
            confidence = 0.7
            
            # Adjust based on match quality
            total_match_length = sum(len(match) if isinstance(match, str) else len(' '.join(match)) for match in matches)
            text_length = len(text)
            
            if text_length > 0:
                coverage = total_match_length / text_length
                confidence += coverage * 0.3
            
            return min(1.0, confidence)
            
        except Exception:
            return 0.5
    
    def get_parser_stats(self) -> Dict[str, Any]:
        """Get parser statistics and status"""
        return {
            "command_patterns": len(self.command_patterns),
            "intent_templates": len(self.intent_templates),
            "entity_types": len(self.entity_patterns),
            "thai_support": self.thai_processor is not None,
            "supported_languages": ["en", "th"] if self.thai_processor else ["en"],
            "supported_command_types": [ct.value for ct in CommandType],
            "supported_intents": list(self.intent_templates.keys())
        }