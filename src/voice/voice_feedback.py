"""
Intelligent Voice Feedback and Confirmation System for JARVIS
Provides contextual audio feedback, confirmations, and intelligent response management
"""

import logging
import time
import random
from typing import Dict, Any, List, Optional, Callable, Union
from PyQt6.QtCore import QObject, pyqtSignal, QTimer
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    from ..voice.command_parser import ParsedCommand
    from ..ai.nlu_engine import NLUResult
except ImportError:
    # Fallback for direct execution
    ParsedCommand = None
    NLUResult = None


class FeedbackType(Enum):
    """Types of voice feedback"""
    ACKNOWLEDGMENT = "acknowledgment"
    CONFIRMATION = "confirmation"
    CLARIFICATION = "clarification"
    ERROR_FEEDBACK = "error_feedback"
    PROGRESS_UPDATE = "progress_update"
    COMPLETION = "completion"
    GREETING = "greeting"
    GOODBYE = "goodbye"
    THINKING = "thinking"
    PERSONALITY = "personality"


class FeedbackPriority(Enum):
    """Priority levels for feedback"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


class ConfirmationMode(Enum):
    """Confirmation modes"""
    NONE = "none"
    IMPLICIT = "implicit"
    EXPLICIT = "explicit"
    ALWAYS = "always"


@dataclass
class FeedbackMessage:
    """Voice feedback message structure"""
    message_id: str
    feedback_type: FeedbackType
    content: str
    language: str = "en"
    priority: FeedbackPriority = FeedbackPriority.NORMAL
    confirmation_required: bool = False
    timeout: float = 0.0
    context: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_time: float = field(default_factory=time.time)


class PersonalityEngine:
    """JARVIS personality and response generation"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + ".PersonalityEngine")
        self.config = config.get("personality", {})
        
        # Personality configuration
        self.personality_level = self.config.get("personality_level", 0.7)  # 0.0 to 1.0
        self.formality_level = self.config.get("formality_level", 0.6)
        self.helpfulness_level = self.config.get("helpfulness_level", 0.9)
        self.humor_level = self.config.get("humor_level", 0.3)
        
        # Response templates
        self.response_templates = self._initialize_response_templates()
        
        # Personality traits
        self.traits = {
            "intelligent": 0.95,
            "helpful": 0.90,
            "efficient": 0.85,
            "friendly": 0.75,
            "formal": 0.60,
            "witty": 0.40
        }
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, List[str]]]:
        """Initialize response templates for different situations"""
        return {
            "acknowledgment": {
                "en": [
                    "Understood, sir.",
                    "Acknowledged.",
                    "I'm on it.",
                    "Right away, sir.",
                    "Consider it done.",
                    "Processing your request.",
                    "I'll take care of that.",
                    "Certainly.",
                    "Of course."
                ],
                "th": [
                    "เข้าใจแล้วครับ",
                    "รับทราบครับ",
                    "กำลังดำเนินการครับ",
                    "ทันทีครับ",
                    "จัดการให้เรียบร้อยครับ",
                    "กำลังประมวลผลครับ",
                    "ผมจะจัดการให้ครับ",
                    "แน่นอนครับ",
                    "ได้เลยครับ"
                ]
            },
            
            "confirmation": {
                "en": [
                    "Would you like me to proceed with {action}?",
                    "Shall I {action}?",
                    "Do you want me to {action}?",
                    "Should I go ahead and {action}?",
                    "Confirm: {action}?",
                    "May I {action}?",
                    "Is it okay to {action}?",
                    "Ready to {action}. Proceed?"
                ],
                "th": [
                    "ต้องการให้ผม{action}หรือไม่ครับ?",
                    "ให้ผม{action}ไหมครับ?",
                    "อยากให้ผม{action}ใช่ไหมครับ?",
                    "ผมไป{action}เลยไหมครับ?",
                    "ยืนยัน: {action}?",
                    "ขออนุญาต{action}ครับ?",
                    "จะ{action}ได้ไหมครับ?",
                    "พร้อม{action}แล้วครับ ดำเนินการไหม?"
                ]
            },
            
            "clarification": {
                "en": [
                    "I'm not sure I understand. Could you clarify?",
                    "Could you be more specific?",
                    "I need more information to help you.",
                    "What exactly would you like me to do?",
                    "Can you rephrase that?",
                    "I'm having trouble understanding your request.",
                    "Could you provide more details?",
                    "I want to make sure I understand correctly."
                ],
                "th": [
                    "ผมไม่แน่ใจว่าเข้าใจ ช่วยอธิบายเพิ่มเติมครับ?",
                    "ช่วยอธิบายให้ชัดเจนหน่อยครับ?",
                    "ผมต้องการข้อมูลเพิ่มเติมเพื่อช่วยคุณครับ",
                    "คุณต้องการให้ผมทำอะไรเป็นที่แน่ครับ?",
                    "ช่วยพูดใหม่ได้ไหมครับ?",
                    "ผมมีปัญหาในการเข้าใจคำขอของคุณครับ",
                    "ช่วยให้รายละเอียดเพิ่มเติมครับ?",
                    "ผมอยากให้แน่ใจว่าเข้าใจถูกต้องครับ"
                ]
            },
            
            "error_feedback": {
                "en": [
                    "I apologize, but I encountered an error.",
                    "Something went wrong. Let me try again.",
                    "I'm experiencing a technical difficulty.",
                    "There seems to be a problem.",
                    "I'm having trouble with that request.",
                    "An error occurred while processing.",
                    "Let me troubleshoot this issue.",
                    "I need to resolve a technical issue first."
                ],
                "th": [
                    "ขออภัยครับ เกิดข้อผิดพลาดขึ้น",
                    "มีบางอย่างผิดพลาด ให้ผมลองใหม่ครับ",
                    "ผมประสบปัญหาทางเทคนิคครับ",
                    "ดูเหมือนจะมีปัญหาครับ",
                    "ผมมีปัญหาในการดำเนินการตามที่ขอครับ",
                    "เกิดข้อผิดพลาดระหว่างการประมวลผลครับ",
                    "ให้ผมแก้ไขปัญหานี้ก่อนครับ",
                    "ผมต้องแก้ไขปัญหาทางเทคนิคก่อนครับ"
                ]
            },
            
            "progress_update": {
                "en": [
                    "Working on it...",
                    "Processing...",
                    "Almost done...",
                    "Making progress...",
                    "Just a moment...",
                    "Nearly finished...",
                    "Still working...",
                    "Getting closer..."
                ],
                "th": [
                    "กำลังดำเนินการครับ...",
                    "กำลังประมวลผลครับ...",
                    "เกือบเสร็จแล้วครับ...",
                    "กำลังมีความคืบหนาครับ...",
                    "อีกสักครู่ครับ...",
                    "เกือบเสร็จแล้วครับ...",
                    "ยังคงดำเนินการอยู่ครับ...",
                    "ใกล้เสร็จแล้วครับ..."
                ]
            },
            
            "completion": {
                "en": [
                    "Task completed successfully.",
                    "Done, sir.",
                    "All finished.",
                    "Task accomplished.",
                    "Mission complete.",
                    "Successfully completed.",
                    "All set.",
                    "Task finished."
                ],
                "th": [
                    "ดำเนินการเสร็จสิ้นเรียบร้อยครับ",
                    "เสร็จแล้วครับ",
                    "เสร็จทั้งหมดแล้วครับ",
                    "ภารกิจสำเร็จครับ",
                    "ภารกิจสมบูรณ์ครับ",
                    "ดำเนินการสำเร็จครับ",
                    "พร้อมแล้วครับ",
                    "งานเสร็จแล้วครับ"
                ]
            },
            
            "greeting": {
                "en": [
                    "Good morning, sir.",
                    "Hello there.",
                    "Good to see you.",
                    "At your service.",
                    "How may I assist you today?",
                    "Ready to help.",
                    "Standing by.",
                    "What can I do for you?"
                ],
                "th": [
                    "สวัสดีตอนเช้าครับ",
                    "สวัสดีครับ",
                    "ยินดีที่ได้เจอครับ",
                    "พร้อมรับใช้ครับ",
                    "วันนี้มีอะไรให้ช่วยไหมครับ?",
                    "พร้อมช่วยเหลือครับ",
                    "พร้อมรับคำสั่งครับ",
                    "มีอะไรให้ช่วยไหมครับ?"
                ]
            },
            
            "thinking": {
                "en": [
                    "Let me think about that...",
                    "Analyzing...",
                    "Considering your request...",
                    "Evaluating options...",
                    "Processing the information...",
                    "Thinking...",
                    "One moment while I analyze this...",
                    "Let me process that..."
                ],
                "th": [
                    "ให้ผมคิดดูสักครู่ครับ...",
                    "กำลังวิเคราะห์ครับ...",
                    "กำลังพิจารณาคำขอของคุณครับ...",
                    "กำลังประเมินตัวเลือกครับ...",
                    "กำลังประมวลผลข้อมูลครับ...",
                    "กำลังคิดครับ...",
                    "อีกสักครู่ ให้ผมวิเคราะห์ก่อนครับ...",
                    "ให้ผมประมวลผลก่อนครับ..."
                ]
            }
        }
    
    def generate_response(self, feedback_type: FeedbackType, language: str = "en", context: Dict[str, Any] = None) -> str:
        """Generate a personality-appropriate response"""
        try:
            if context is None:
                context = {}
            
            templates = self.response_templates.get(feedback_type.value, {}).get(language, [])
            
            if not templates:
                # Fallback to English if language not available
                templates = self.response_templates.get(feedback_type.value, {}).get("en", [])
            
            if not templates:
                return "I understand."
            
            # Select template based on personality
            if self.personality_level > 0.8:
                # High personality - prefer more expressive responses
                template = random.choice(templates[-3:] if len(templates) > 3 else templates)
            elif self.personality_level > 0.5:
                # Medium personality - balanced selection
                template = random.choice(templates)
            else:
                # Low personality - prefer simple responses
                template = random.choice(templates[:3] if len(templates) > 3 else templates)
            
            # Apply context formatting
            if "{action}" in template and "action" in context:
                template = template.format(action=context["action"])
            
            # Apply personality modifications
            if self.formality_level > 0.7 and language == "en":
                if not any(formal in template.lower() for formal in ["sir", "please", "certainly"]):
                    if random.random() < 0.3:
                        template += ", sir."
            
            return template
            
        except Exception as e:
            self.logger.error(f"Response generation failed: {e}")
            return "I understand." if language == "en" else "เข้าใจแล้วครับ"
    
    def get_personality_metrics(self) -> Dict[str, float]:
        """Get current personality metrics"""
        return {
            "personality_level": self.personality_level,
            "formality_level": self.formality_level,
            "helpfulness_level": self.helpfulness_level,
            "humor_level": self.humor_level,
            "traits": self.traits.copy()
        }


class ConfirmationManager:
    """Manages confirmation requests and responses"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + ".ConfirmationManager")
        self.config = config.get("confirmation", {})
        
        # Confirmation settings
        self.default_mode = ConfirmationMode(self.config.get("default_mode", "implicit"))
        self.timeout_duration = self.config.get("timeout_duration", 10.0)
        self.auto_confirm_threshold = self.config.get("auto_confirm_threshold", 0.9)
        
        # Pending confirmations
        self.pending_confirmations: Dict[str, FeedbackMessage] = {}
        
        # Confirmation patterns
        self.confirmation_patterns = {
            "positive": {
                "en": ["yes", "yeah", "yep", "sure", "okay", "ok", "go ahead", "proceed", "do it", "continue"],
                "th": ["ใช่", "ครับ", "ค่ะ", "โอเค", "ตกลง", "ได้", "ทำต่อ", "ดำเนินการ", "ไปเลย"]
            },
            "negative": {
                "en": ["no", "nope", "cancel", "stop", "abort", "don't", "never mind", "skip"],
                "th": ["ไม่", "ไม่ใช่", "ยกเลิก", "หยุด", "อย่า", "ช่างเถอะ", "ข้าม", "ไม่ต้อง"]
            }
        }
    
    def requires_confirmation(self, command: Any, confidence: float) -> bool:
        """Determine if command requires confirmation"""
        try:
            # High confidence commands may not need confirmation
            if confidence > self.auto_confirm_threshold:
                return False
            
            # Check command type for confirmation requirements
            if hasattr(command, 'intent'):
                risky_intents = ["system_control", "file_deletion", "data_modification"]
                if command.intent in risky_intents:
                    return True
            
            # Check confirmation mode
            if self.default_mode == ConfirmationMode.ALWAYS:
                return True
            elif self.default_mode == ConfirmationMode.NONE:
                return False
            elif self.default_mode == ConfirmationMode.EXPLICIT:
                return confidence < 0.8
            else:  # IMPLICIT
                return confidence < 0.7
            
        except Exception as e:
            self.logger.error(f"Confirmation check failed: {e}")
            return False
    
    def create_confirmation_request(self, message_id: str, action: str, language: str = "en") -> str:
        """Create confirmation request message"""
        try:
            confirmation_templates = {
                "en": [
                    f"Would you like me to {action}?",
                    f"Shall I {action}?",
                    f"Do you want me to {action}?",
                    f"Should I proceed with {action}?",
                    f"Confirm: {action}?"
                ],
                "th": [
                    f"ต้องการให้ผม{action}หรือไม่ครับ?",
                    f"ให้ผม{action}ไหมครับ?",
                    f"อยากให้ผม{action}ใช่ไหมครับ?",
                    f"จะดำเนินการ{action}ไหมครับ?",
                    f"ยืนยัน: {action}?"
                ]
            }
            
            templates = confirmation_templates.get(language, confirmation_templates["en"])
            confirmation_message = random.choice(templates)
            
            # Store pending confirmation
            self.pending_confirmations[message_id] = {
                "action": action,
                "language": language,
                "timestamp": time.time(),
                "message": confirmation_message
            }
            
            return confirmation_message
            
        except Exception as e:
            self.logger.error(f"Confirmation request creation failed: {e}")
            return f"Confirm: {action}?"
    
    def process_confirmation_response(self, response_text: str, language: str = "en") -> Optional[bool]:
        """Process user's confirmation response"""
        try:
            response_lower = response_text.lower().strip()
            
            # Check positive patterns
            positive_patterns = self.confirmation_patterns["positive"].get(language, [])
            if any(pattern in response_lower for pattern in positive_patterns):
                return True
            
            # Check negative patterns
            negative_patterns = self.confirmation_patterns["negative"].get(language, [])
            if any(pattern in response_lower for pattern in negative_patterns):
                return False
            
            # Ambiguous response
            return None
            
        except Exception as e:
            self.logger.error(f"Confirmation response processing failed: {e}")
            return None
    
    def cleanup_expired_confirmations(self):
        """Remove expired confirmation requests"""
        try:
            current_time = time.time()
            expired_ids = [
                msg_id for msg_id, confirmation in self.pending_confirmations.items()
                if (current_time - confirmation["timestamp"]) > self.timeout_duration
            ]
            
            for msg_id in expired_ids:
                del self.pending_confirmations[msg_id]
                
            if expired_ids:
                self.logger.info(f"Cleaned up {len(expired_ids)} expired confirmations")
                
        except Exception as e:
            self.logger.error(f"Confirmation cleanup failed: {e}")


class VoiceFeedbackSystem(QObject):
    """Intelligent voice feedback and confirmation system"""
    
    # Signals
    feedback_generated = pyqtSignal(dict)  # Feedback message
    confirmation_requested = pyqtSignal(str, str)  # message_id, confirmation_text
    confirmation_received = pyqtSignal(str, bool)  # message_id, confirmed
    feedback_spoken = pyqtSignal(str)  # message_id
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("voice_feedback", {})
        
        # Core components
        self.personality_engine = PersonalityEngine(config)
        self.confirmation_manager = ConfirmationManager(config)
        
        # Feedback settings
        self.enabled = self.config.get("enabled", True)
        self.default_language = self.config.get("default_language", "en")
        self.feedback_delay = self.config.get("feedback_delay", 0.5)
        self.max_queue_size = self.config.get("max_queue_size", 10)
        
        # Message management
        self.feedback_queue: List[FeedbackMessage] = []
        self.message_history: List[FeedbackMessage] = []
        self.pending_confirmations: Dict[str, FeedbackMessage] = {}
        
        # Timers
        self.feedback_timer = QTimer()
        self.feedback_timer.timeout.connect(self._process_feedback_queue)
        
        self.cleanup_timer = QTimer()
        self.cleanup_timer.timeout.connect(self._cleanup_expired_messages)
        self.cleanup_timer.start(30000)  # Cleanup every 30 seconds
        
        # Statistics
        self.stats = {
            "messages_generated": 0,
            "confirmations_requested": 0,
            "confirmations_received": 0,
            "feedback_by_type": {ft.value: 0 for ft in FeedbackType}
        }
    
    def generate_feedback(self, feedback_type: FeedbackType, context: Dict[str, Any] = None, 
                         language: str = None, priority: FeedbackPriority = FeedbackPriority.NORMAL) -> str:
        """Generate intelligent feedback message"""
        try:
            if not self.enabled:
                return ""
            
            if context is None:
                context = {}
            
            if language is None:
                language = self.default_language
            
            # Generate message content
            content = self.personality_engine.generate_response(feedback_type, language, context)
            
            # Create feedback message
            message_id = f"fb_{int(time.time())}_{id(self)}"
            
            feedback_message = FeedbackMessage(
                message_id=message_id,
                feedback_type=feedback_type,
                content=content,
                language=language,
                priority=priority,
                context=context.copy()
            )
            
            # Add to queue
            self._add_to_queue(feedback_message)
            
            # Update statistics
            self.stats["messages_generated"] += 1
            self.stats["feedback_by_type"][feedback_type.value] += 1
            
            # Emit signal
            self.feedback_generated.emit(feedback_message.__dict__)
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Feedback generation failed: {e}")
            self.error_occurred.emit(f"Feedback generation: {e}")
            return ""
    
    def request_confirmation(self, action: str, context: Dict[str, Any] = None, 
                           language: str = None) -> str:
        """Request user confirmation for an action"""
        try:
            if language is None:
                language = self.default_language
            
            if context is None:
                context = {}
            
            # Create confirmation message
            message_id = f"conf_{int(time.time())}_{id(self)}"
            
            confirmation_text = self.confirmation_manager.create_confirmation_request(
                message_id, action, language
            )
            
            # Create feedback message
            feedback_message = FeedbackMessage(
                message_id=message_id,
                feedback_type=FeedbackType.CONFIRMATION,
                content=confirmation_text,
                language=language,
                priority=FeedbackPriority.HIGH,
                confirmation_required=True,
                timeout=self.confirmation_manager.timeout_duration,
                context=context.copy()
            )
            
            # Add to pending confirmations
            self.pending_confirmations[message_id] = feedback_message
            
            # Add to queue for speaking
            self._add_to_queue(feedback_message)
            
            # Update statistics
            self.stats["confirmations_requested"] += 1
            
            # Emit signal
            self.confirmation_requested.emit(message_id, confirmation_text)
            
            return message_id
            
        except Exception as e:
            self.logger.error(f"Confirmation request failed: {e}")
            self.error_occurred.emit(f"Confirmation request: {e}")
            return ""
    
    def process_confirmation_response(self, response_text: str, language: str = None) -> Optional[str]:
        """Process user's response to confirmation"""
        try:
            if language is None:
                language = self.default_language
            
            # Process the response
            confirmation_result = self.confirmation_manager.process_confirmation_response(response_text, language)
            
            if confirmation_result is None:
                # Ambiguous response - request clarification
                clarification_id = self.generate_feedback(
                    FeedbackType.CLARIFICATION,
                    {"response": response_text},
                    language,
                    FeedbackPriority.HIGH
                )
                return clarification_id
            
            # Find the most recent pending confirmation
            if self.pending_confirmations:
                latest_confirmation_id = max(
                    self.pending_confirmations.keys(),
                    key=lambda x: self.pending_confirmations[x].created_time
                )
                
                # Remove from pending
                confirmed_message = self.pending_confirmations.pop(latest_confirmation_id)
                
                # Update statistics
                self.stats["confirmations_received"] += 1
                
                # Emit confirmation result
                self.confirmation_received.emit(latest_confirmation_id, confirmation_result)
                
                # Generate appropriate feedback
                if confirmation_result:
                    feedback_id = self.generate_feedback(
                        FeedbackType.ACKNOWLEDGMENT,
                        {"action": "proceeding"},
                        language
                    )
                else:
                    feedback_id = self.generate_feedback(
                        FeedbackType.ACKNOWLEDGMENT,
                        {"action": "cancelled"},
                        language
                    )
                
                return feedback_id
            
            return None
            
        except Exception as e:
            self.logger.error(f"Confirmation response processing failed: {e}")
            self.error_occurred.emit(f"Confirmation processing: {e}")
            return None
    
    def provide_progress_update(self, progress: float, task: str, language: str = None) -> str:
        """Provide progress update feedback"""
        try:
            if language is None:
                language = self.default_language
            
            context = {
                "progress": progress,
                "task": task,
                "percentage": f"{int(progress * 100)}%"
            }
            
            return self.generate_feedback(
                FeedbackType.PROGRESS_UPDATE,
                context,
                language,
                FeedbackPriority.LOW
            )
            
        except Exception as e:
            self.logger.error(f"Progress update failed: {e}")
            return ""
    
    def announce_completion(self, task: str, language: str = None) -> str:
        """Announce task completion"""
        try:
            if language is None:
                language = self.default_language
            
            context = {"task": task}
            
            return self.generate_feedback(
                FeedbackType.COMPLETION,
                context,
                language,
                FeedbackPriority.NORMAL
            )
            
        except Exception as e:
            self.logger.error(f"Completion announcement failed: {e}")
            return ""
    
    def handle_error(self, error_message: str, language: str = None) -> str:
        """Handle error with appropriate feedback"""
        try:
            if language is None:
                language = self.default_language
            
            context = {"error": error_message}
            
            return self.generate_feedback(
                FeedbackType.ERROR_FEEDBACK,
                context,
                language,
                FeedbackPriority.HIGH
            )
            
        except Exception as e:
            self.logger.error(f"Error handling failed: {e}")
            return ""
    
    def _add_to_queue(self, message: FeedbackMessage):
        """Add message to feedback queue"""
        try:
            # Remove oldest messages if queue is full
            while len(self.feedback_queue) >= self.max_queue_size:
                oldest = self.feedback_queue.pop(0)
                self.logger.warning(f"Dropped feedback message: {oldest.message_id}")
            
            # Insert based on priority
            inserted = False
            for i, existing in enumerate(self.feedback_queue):
                if message.priority.value > existing.priority.value:
                    self.feedback_queue.insert(i, message)
                    inserted = True
                    break
            
            if not inserted:
                self.feedback_queue.append(message)
            
            # Start processing timer if not running
            if not self.feedback_timer.isActive():
                self.feedback_timer.start(int(self.feedback_delay * 1000))
                
        except Exception as e:
            self.logger.error(f"Queue addition failed: {e}")
    
    def _process_feedback_queue(self):
        """Process feedback queue"""
        try:
            if not self.feedback_queue:
                self.feedback_timer.stop()
                return
            
            # Get next message
            message = self.feedback_queue.pop(0)
            
            # Add to history
            self.message_history.append(message)
            
            # Emit for speaking
            self.feedback_spoken.emit(message.message_id)
            
            # Continue processing if more messages
            if self.feedback_queue:
                self.feedback_timer.start(int(self.feedback_delay * 1000))
            else:
                self.feedback_timer.stop()
                
        except Exception as e:
            self.logger.error(f"Queue processing failed: {e}")
    
    def _cleanup_expired_messages(self):
        """Cleanup expired messages and confirmations"""
        try:
            current_time = time.time()
            
            # Cleanup confirmation manager
            self.confirmation_manager.cleanup_expired_confirmations()
            
            # Cleanup pending confirmations
            expired_confirmations = [
                msg_id for msg_id, message in self.pending_confirmations.items()
                if (current_time - message.created_time) > message.timeout
            ]
            
            for msg_id in expired_confirmations:
                del self.pending_confirmations[msg_id]
                self.logger.info(f"Expired confirmation: {msg_id}")
            
            # Limit message history size
            if len(self.message_history) > 100:
                self.message_history = self.message_history[-50:]
                
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")
    
    def get_feedback_stats(self) -> Dict[str, Any]:
        """Get feedback system statistics"""
        return {
            "statistics": self.stats.copy(),
            "queue_size": len(self.feedback_queue),
            "pending_confirmations": len(self.pending_confirmations),
            "message_history": len(self.message_history),
            "personality_metrics": self.personality_engine.get_personality_metrics(),
            "configuration": {
                "enabled": self.enabled,
                "default_language": self.default_language,
                "feedback_delay": self.feedback_delay,
                "max_queue_size": self.max_queue_size
            }
        }
    
    def set_personality_level(self, level: float):
        """Set personality level (0.0 to 1.0)"""
        try:
            self.personality_engine.personality_level = max(0.0, min(1.0, level))
            self.logger.info(f"Personality level set to {level:.2f}")
            
        except Exception as e:
            self.logger.error(f"Personality level setting failed: {e}")
    
    def set_confirmation_mode(self, mode: str):
        """Set confirmation mode"""
        try:
            self.confirmation_manager.default_mode = ConfirmationMode(mode)
            self.logger.info(f"Confirmation mode set to {mode}")
            
        except Exception as e:
            self.logger.error(f"Confirmation mode setting failed: {e}")
    
    def clear_queue(self):
        """Clear feedback queue"""
        cleared = len(self.feedback_queue)
        self.feedback_queue.clear()
        self.feedback_timer.stop()
        self.logger.info(f"Cleared {cleared} messages from queue")
    
    def shutdown(self):
        """Shutdown feedback system"""
        self.logger.info("Shutting down voice feedback system")
        self.feedback_timer.stop()
        self.cleanup_timer.stop()
        self.clear_queue()
        self.pending_confirmations.clear()