"""
Natural Language Understanding Engine for JARVIS Voice Assistant
Advanced command interpretation with context awareness and semantic understanding
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import time

try:
    from ..voice.command_parser import ParsedCommand, CommandType, CommandPriority
    from .rag_system import RAGSystem
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent))
    from voice.command_parser import ParsedCommand, CommandType, CommandPriority
    try:
        from ai.rag_system import RAGSystem
    except ImportError:
        RAGSystem = None


class IntentCategory(Enum):
    """Intent categories for better organization"""
    INFORMATIONAL = "informational"
    OPERATIONAL = "operational" 
    CONVERSATIONAL = "conversational"
    SYSTEM = "system"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


class ContextType(Enum):
    """Types of context needed for command processing"""
    NONE = "none"
    KNOWLEDGE = "knowledge"
    CONVERSATION = "conversation"
    USER_PROFILE = "user_profile"
    SYSTEM_STATE = "system_state"
    EXTERNAL_DATA = "external_data"


@dataclass
class IntentDefinition:
    """Comprehensive intent definition"""
    name: str
    category: IntentCategory
    description: str
    patterns: List[str]
    context_required: List[ContextType]
    parameters: List[str] = field(default_factory=list)
    examples: List[str] = field(default_factory=list)
    confidence_threshold: float = 0.7
    priority: int = 50


@dataclass
class NLUResult:
    """Enhanced NLU processing result"""
    original_command: ParsedCommand
    refined_intent: str
    intent_confidence: float
    semantic_understanding: Dict[str, Any]
    context_requirements: List[ContextType]
    suggested_responses: List[str]
    processing_time: float
    complexity_score: float
    requires_followup: bool = False
    error_message: Optional[str] = None


class SemanticAnalyzer:
    """Semantic analysis for deeper understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + ".SemanticAnalyzer")
        self.config = config.get("semantic_analyzer", {})
        
        # Semantic patterns
        self.semantic_patterns = self._initialize_semantic_patterns()
        self.entity_relationships = self._initialize_entity_relationships()
        
    def _initialize_semantic_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Initialize semantic understanding patterns"""
        return {
            "causality": {
                "patterns": [
                    r"(?:because|since|due to|as a result of|owing to)",
                    r"(?:เพราะ|เนื่องจาก|ด้วยเหตุที่|เป็นผลมาจาก)"
                ],
                "weight": 0.8,
                "indicates": "causal_relationship"
            },
            
            "comparison": {
                "patterns": [
                    r"(?:compared to|versus|vs|better than|worse than|similar to)",
                    r"(?:เทียบกับ|เปรียบเทียบ|ดีกว่า|แย่กว่า|คล้ายกับ)"
                ],
                "weight": 0.7,
                "indicates": "comparative_analysis"
            },
            
            "temporal": {
                "patterns": [
                    r"(?:before|after|during|while|when|since|until)",
                    r"(?:ก่อน|หลัง|ระหว่าง|ตอน|เมื่อ|ตั้งแต่|จนกว่า)"
                ],
                "weight": 0.6,
                "indicates": "temporal_relationship"
            },
            
            "quantity": {
                "patterns": [
                    r"(?:how many|how much|amount of|number of|quantity)",
                    r"(?:เท่าไหร่|กี่|จำนวน|ปริมาณ)"
                ],
                "weight": 0.8,
                "indicates": "quantitative_query"
            },
            
            "location": {
                "patterns": [
                    r"(?:where|location|place|position|at|in|on)",
                    r"(?:ที่ไหน|สถานที่|ตำแหน่ง|ใน|บน|ที่)"
                ],
                "weight": 0.7,
                "indicates": "spatial_query"
            },
            
            "process": {
                "patterns": [
                    r"(?:how to|steps|process|procedure|method|way)",
                    r"(?:วิธี|ขั้นตอน|กระบวนการ|แนวทาง)"
                ],
                "weight": 0.9,
                "indicates": "procedural_query"
            },
            
            "explanation": {
                "patterns": [
                    r"(?:explain|describe|tell me about|what is|define)",
                    r"(?:อธิบาย|บรรยาย|บอกเกี่ยวกับ|คืออะไร|นิยาม)"
                ],
                "weight": 0.8,
                "indicates": "explanatory_query"
            }
        }
    
    def _initialize_entity_relationships(self) -> Dict[str, List[str]]:
        """Initialize entity relationship mappings"""
        return {
            "technology": ["AI", "machine learning", "deep learning", "neural networks", "algorithm"],
            "programming": ["code", "programming", "development", "software", "application"],
            "science": ["physics", "chemistry", "biology", "research", "experiment"],
            "business": ["company", "startup", "enterprise", "management", "strategy"],
            "education": ["learning", "teaching", "course", "study", "knowledge"],
            "health": ["medical", "health", "treatment", "diagnosis", "wellness"]
        }
    
    def analyze_semantics(self, text: str, language: str) -> Dict[str, Any]:
        """Perform semantic analysis on text"""
        try:
            analysis = {
                "semantic_indicators": [],
                "entity_categories": [],
                "complexity_factors": [],
                "relationship_types": [],
                "semantic_score": 0.0
            }
            
            text_lower = text.lower()
            
            # Analyze semantic patterns
            for pattern_name, pattern_config in self.semantic_patterns.items():
                for pattern in pattern_config["patterns"]:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        analysis["semantic_indicators"].append({
                            "type": pattern_config["indicates"],
                            "pattern": pattern_name,
                            "weight": pattern_config["weight"]
                        })
                        analysis["semantic_score"] += pattern_config["weight"]
            
            # Analyze entity categories
            for category, keywords in self.entity_relationships.items():
                if any(keyword.lower() in text_lower for keyword in keywords):
                    analysis["entity_categories"].append(category)
            
            # Calculate complexity
            complexity_factors = []
            
            # Sentence length complexity
            words = text.split()
            if len(words) > 15:
                complexity_factors.append("long_sentence")
            
            # Multiple questions
            if text.count('?') > 1:
                complexity_factors.append("multiple_questions")
            
            # Technical terms
            technical_patterns = [
                r"\b(?:algorithm|implementation|architecture|framework|methodology)\b",
                r"\b(?:อัลกอริทึม|การดำเนินการ|สถาปัตยกรรม|กรอบงาน|วิธีการ)\b"
            ]
            
            for pattern in technical_patterns:
                if re.search(pattern, text_lower):
                    complexity_factors.append("technical_terminology")
                    break
            
            analysis["complexity_factors"] = complexity_factors
            analysis["complexity_score"] = len(complexity_factors) * 0.2 + analysis["semantic_score"] * 0.1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Semantic analysis failed: {e}")
            return {"semantic_score": 0.0, "error": str(e)}


class ContextManager:
    """Manages context for command understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__ + ".ContextManager")
        self.config = config.get("context_manager", {})
        
        # Context storage
        self.conversation_context = []
        self.user_context = {}
        self.system_context = {}
        
        # Context limits
        self.max_conversation_turns = self.config.get("max_conversation_turns", 10)
        
    def add_conversation_turn(self, user_input: str, assistant_response: str, metadata: Dict[str, Any]):
        """Add conversation turn to context"""
        try:
            turn = {
                "timestamp": time.time(),
                "user_input": user_input,
                "assistant_response": assistant_response,
                "metadata": metadata
            }
            
            self.conversation_context.append(turn)
            
            # Limit context size
            if len(self.conversation_context) > self.max_conversation_turns:
                self.conversation_context.pop(0)
                
        except Exception as e:
            self.logger.error(f"Failed to add conversation turn: {e}")
    
    def get_relevant_context(self, intent: str, text: str) -> Dict[str, Any]:
        """Get relevant context for command processing"""
        try:
            context = {
                "conversation": [],
                "user_profile": {},
                "system_state": {},
                "relevant_history": []
            }
            
            # Get recent conversation
            if self.conversation_context:
                context["conversation"] = self.conversation_context[-3:]  # Last 3 turns
            
            # Search for relevant previous interactions
            text_lower = text.lower()
            for turn in self.conversation_context:
                if any(word in turn["user_input"].lower() for word in text_lower.split()):
                    context["relevant_history"].append(turn)
            
            # User context
            context["user_profile"] = self.user_context.copy()
            
            # System context
            context["system_state"] = self.system_context.copy()
            
            return context
            
        except Exception as e:
            self.logger.error(f"Context retrieval failed: {e}")
            return {}
    
    def update_user_context(self, key: str, value: Any):
        """Update user context"""
        self.user_context[key] = value
    
    def update_system_context(self, key: str, value: Any):
        """Update system context"""
        self.system_context[key] = value


class NaturalLanguageUnderstanding:
    """Advanced NLU Engine for JARVIS"""
    
    def __init__(self, config: Dict[str, Any], rag_system: Optional[RAGSystem] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config.get("nlu_engine", {})
        
        # Initialize components
        self.semantic_analyzer = SemanticAnalyzer(config)
        self.context_manager = ContextManager(config)
        self.rag_system = rag_system
        
        # Intent definitions
        self.intent_definitions = self._initialize_intent_definitions()
        
        # Processing statistics
        self.stats = {
            "commands_processed": 0,
            "successful_interpretations": 0,
            "average_processing_time": 0.0,
            "complexity_distribution": {"low": 0, "medium": 0, "high": 0}
        }
        
    def _initialize_intent_definitions(self) -> Dict[str, IntentDefinition]:
        """Initialize comprehensive intent definitions"""
        return {
            "deep_information_request": IntentDefinition(
                name="deep_information_request",
                category=IntentCategory.INFORMATIONAL,
                description="Complex information requests requiring knowledge synthesis",
                patterns=[
                    r"(?:explain|describe|analyze|compare|contrast).*(?:in detail|thoroughly|comprehensively)",
                    r"(?:อธิบาย|วิเคราะห์|เปรียบเทียบ).*(?:อย่างละเอียด|โดยละเอียด|อย่างครบถ้วน)"
                ],
                context_required=[ContextType.KNOWLEDGE, ContextType.CONVERSATION],
                examples=[
                    "Explain machine learning algorithms in detail",
                    "อธิบายอัลกอริทึมการเรียนรู้ของเครื่องอย่างละเอียด"
                ],
                confidence_threshold=0.8,
                priority=90
            ),
            
            "procedural_guidance": IntentDefinition(
                name="procedural_guidance",
                category=IntentCategory.OPERATIONAL,
                description="Step-by-step guidance for complex tasks",
                patterns=[
                    r"(?:how to|guide me|walk me through|step by step)",
                    r"(?:วิธี|แนะนำ|สอน|ขั้นตอน)"
                ],
                context_required=[ContextType.KNOWLEDGE, ContextType.USER_PROFILE],
                examples=[
                    "Guide me through setting up a neural network",
                    "สอนการตั้งค่าโครงข่ายประสาทเทียม"
                ]
            ),
            
            "creative_request": IntentDefinition(
                name="creative_request",
                category=IntentCategory.CREATIVE,
                description="Requests for creative content generation",
                patterns=[
                    r"(?:create|generate|make|design|write|compose)",
                    r"(?:สร้าง|ทำ|เขียน|ออกแบบ|แต่ง)"
                ],
                context_required=[ContextType.USER_PROFILE, ContextType.CONVERSATION],
                examples=[
                    "Create a Python script for data analysis",
                    "สร้างสคริปต์ Python สำหรับวิเคราะห์ข้อมูล"
                ]
            ),
            
            "analytical_request": IntentDefinition(
                name="analytical_request", 
                category=IntentCategory.ANALYTICAL,
                description="Requests for analysis, evaluation, or assessment",
                patterns=[
                    r"(?:analyze|evaluate|assess|review|examine|investigate)",
                    r"(?:วิเคราะห์|ประเมิน|ตรวจสอบ|ศึกษา|พิจารณา)"
                ],
                context_required=[ContextType.KNOWLEDGE, ContextType.EXTERNAL_DATA],
                examples=[
                    "Analyze the performance of this algorithm",
                    "วิเคราะห์ประสิทธิภาพของอัลกอริทึมนี้"
                ]
            ),
            
            "contextual_continuation": IntentDefinition(
                name="contextual_continuation",
                category=IntentCategory.CONVERSATIONAL,
                description="Commands that continue previous conversation",
                patterns=[
                    r"(?:continue|more|tell me more|elaborate|expand)",
                    r"(?:ต่อ|เพิ่มเติม|อีก|ขยาย|รายละเอียด)"
                ],
                context_required=[ContextType.CONVERSATION],
                examples=[
                    "Tell me more about that",
                    "อธิบายเพิ่มเติมหน่อย"
                ]
            )
        }
    
    def process_command(self, parsed_command: ParsedCommand) -> NLUResult:
        """Process command with advanced NLU"""
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing NLU for: {parsed_command.cleaned_text}")
            
            # Semantic analysis
            semantic_analysis = self.semantic_analyzer.analyze_semantics(
                parsed_command.cleaned_text, 
                parsed_command.language
            )
            
            # Refine intent using semantic understanding
            refined_intent, intent_confidence = self._refine_intent(
                parsed_command, 
                semantic_analysis
            )
            
            # Get relevant context
            context = self.context_manager.get_relevant_context(
                refined_intent,
                parsed_command.cleaned_text
            )
            
            # Determine context requirements
            context_requirements = self._determine_context_requirements(
                refined_intent,
                semantic_analysis
            )
            
            # Generate response suggestions
            suggested_responses = self._generate_response_suggestions(
                refined_intent,
                semantic_analysis,
                context
            )
            
            # Calculate complexity
            complexity_score = self._calculate_complexity(
                parsed_command,
                semantic_analysis
            )
            
            # Determine if followup is needed
            requires_followup = self._requires_followup(
                refined_intent,
                complexity_score,
                context_requirements
            )
            
            processing_time = time.time() - start_time
            
            # Create result
            result = NLUResult(
                original_command=parsed_command,
                refined_intent=refined_intent,
                intent_confidence=intent_confidence,
                semantic_understanding=semantic_analysis,
                context_requirements=context_requirements,
                suggested_responses=suggested_responses,
                processing_time=processing_time,
                complexity_score=complexity_score,
                requires_followup=requires_followup
            )
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"NLU processing failed: {e}")
            processing_time = time.time() - start_time
            
            return NLUResult(
                original_command=parsed_command,
                refined_intent=parsed_command.intent,
                intent_confidence=0.0,
                semantic_understanding={},
                context_requirements=[],
                suggested_responses=["I need more information to help you."],
                processing_time=processing_time,
                complexity_score=0.0,
                error_message=str(e)
            )
    
    def _refine_intent(self, command: ParsedCommand, semantic_analysis: Dict[str, Any]) -> Tuple[str, float]:
        """Refine intent classification using semantic analysis"""
        try:
            text_lower = command.cleaned_text.lower()
            best_intent = command.intent
            best_confidence = command.confidence
            
            # Check against advanced intent definitions
            for intent_name, definition in self.intent_definitions.items():
                pattern_matches = 0
                total_patterns = len(definition.patterns)
                
                for pattern in definition.patterns:
                    if re.search(pattern, text_lower, re.IGNORECASE):
                        pattern_matches += 1
                
                if pattern_matches > 0:
                    # Calculate confidence based on pattern match and semantic indicators
                    pattern_confidence = pattern_matches / total_patterns
                    semantic_boost = 0.0
                    
                    # Boost confidence based on semantic indicators
                    for indicator in semantic_analysis.get("semantic_indicators", []):
                        if self._intent_matches_semantic_type(intent_name, indicator["type"]):
                            semantic_boost += indicator["weight"] * 0.1
                    
                    total_confidence = pattern_confidence + semantic_boost
                    
                    if total_confidence > best_confidence:
                        best_intent = intent_name
                        best_confidence = min(1.0, total_confidence)
            
            return best_intent, best_confidence
            
        except Exception as e:
            self.logger.error(f"Intent refinement failed: {e}")
            return command.intent, command.confidence
    
    def _intent_matches_semantic_type(self, intent: str, semantic_type: str) -> bool:
        """Check if intent matches semantic type"""
        mappings = {
            "explanatory_query": ["deep_information_request", "analytical_request"],
            "procedural_query": ["procedural_guidance"],
            "causal_relationship": ["analytical_request", "deep_information_request"],
            "comparative_analysis": ["analytical_request"],
            "quantitative_query": ["analytical_request"]
        }
        
        return intent in mappings.get(semantic_type, [])
    
    def _determine_context_requirements(self, intent: str, semantic_analysis: Dict[str, Any]) -> List[ContextType]:
        """Determine what context is needed for processing"""
        try:
            requirements = []
            
            # Intent-based requirements
            if intent in self.intent_definitions:
                requirements.extend(self.intent_definitions[intent].context_required)
            
            # Semantic-based requirements
            for indicator in semantic_analysis.get("semantic_indicators", []):
                if indicator["type"] in ["comparative_analysis", "causal_relationship"]:
                    if ContextType.KNOWLEDGE not in requirements:
                        requirements.append(ContextType.KNOWLEDGE)
                
                elif indicator["type"] == "temporal_relationship":
                    if ContextType.CONVERSATION not in requirements:
                        requirements.append(ContextType.CONVERSATION)
            
            # Entity-based requirements
            entity_categories = semantic_analysis.get("entity_categories", [])
            if any(cat in ["technology", "science"] for cat in entity_categories):
                if ContextType.KNOWLEDGE not in requirements:
                    requirements.append(ContextType.KNOWLEDGE)
            
            return requirements
            
        except Exception as e:
            self.logger.error(f"Context requirement determination failed: {e}")
            return [ContextType.NONE]
    
    def _generate_response_suggestions(self, intent: str, semantic_analysis: Dict[str, Any], context: Dict[str, Any]) -> List[str]:
        """Generate intelligent response suggestions"""
        try:
            suggestions = []
            
            # Intent-based suggestions
            if intent == "deep_information_request":
                suggestions.extend([
                    "Provide comprehensive explanation with examples",
                    "Include technical details and practical applications",
                    "Offer related topics for further exploration"
                ])
            
            elif intent == "procedural_guidance":
                suggestions.extend([
                    "Break down into clear, numbered steps",
                    "Include prerequisites and requirements",
                    "Provide troubleshooting tips"
                ])
            
            elif intent == "creative_request":
                suggestions.extend([
                    "Generate creative content based on requirements",
                    "Offer multiple variations or options",
                    "Include implementation guidelines"
                ])
            
            elif intent == "analytical_request":
                suggestions.extend([
                    "Provide structured analysis with key findings",
                    "Include pros, cons, and recommendations",
                    "Support with data and evidence"
                ])
            
            elif intent == "contextual_continuation":
                suggestions.extend([
                    "Continue from previous conversation context",
                    "Expand on previously mentioned topics",
                    "Reference earlier discussion points"
                ])
            
            # Semantic-based suggestions
            for indicator in semantic_analysis.get("semantic_indicators", []):
                if indicator["type"] == "comparative_analysis":
                    suggestions.append("Provide detailed comparison table or analysis")
                elif indicator["type"] == "quantitative_query":
                    suggestions.append("Include specific numbers and statistics")
                elif indicator["type"] == "spatial_query":
                    suggestions.append("Provide location-specific information")
            
            # Default suggestions if none generated
            if not suggestions:
                suggestions = [
                    "Provide helpful and accurate information",
                    "Ask clarifying questions if needed",
                    "Offer additional assistance"
                ]
            
            return suggestions[:3]  # Limit to top 3 suggestions
            
        except Exception as e:
            self.logger.error(f"Response suggestion generation failed: {e}")
            return ["Provide helpful response"]
    
    def _calculate_complexity(self, command: ParsedCommand, semantic_analysis: Dict[str, Any]) -> float:
        """Calculate command complexity score"""
        try:
            complexity = 0.0
            
            # Base complexity from semantic analysis
            complexity += semantic_analysis.get("complexity_score", 0.0)
            
            # Text length factor
            word_count = len(command.cleaned_text.split())
            if word_count > 20:
                complexity += 0.3
            elif word_count > 10:
                complexity += 0.2
            elif word_count > 5:
                complexity += 0.1
            
            # Multiple questions
            if command.cleaned_text.count('?') > 1:
                complexity += 0.2
            
            # Technical entities
            entity_count = sum(len(entities) for entities in command.entities.values())
            complexity += entity_count * 0.1
            
            # Intent complexity
            complex_intents = ["deep_information_request", "analytical_request", "procedural_guidance"]
            if command.intent in complex_intents:
                complexity += 0.3
            
            return min(1.0, complexity)
            
        except Exception as e:
            self.logger.error(f"Complexity calculation failed: {e}")
            return 0.5
    
    def _requires_followup(self, intent: str, complexity_score: float, context_requirements: List[ContextType]) -> bool:
        """Determine if command requires followup"""
        try:
            # High complexity commands often need followup
            if complexity_score > 0.7:
                return True
            
            # Certain intents typically need followup
            followup_intents = ["deep_information_request", "procedural_guidance", "creative_request"]
            if intent in followup_intents:
                return True
            
            # Multiple context requirements suggest complexity
            if len(context_requirements) > 2:
                return True
            
            return False
            
        except Exception:
            return False
    
    def _update_stats(self, result: NLUResult):
        """Update processing statistics"""
        try:
            self.stats["commands_processed"] += 1
            
            if result.error_message is None:
                self.stats["successful_interpretations"] += 1
            
            # Update average processing time
            total_time = self.stats["average_processing_time"] * (self.stats["commands_processed"] - 1)
            self.stats["average_processing_time"] = (total_time + result.processing_time) / self.stats["commands_processed"]
            
            # Update complexity distribution
            if result.complexity_score < 0.3:
                self.stats["complexity_distribution"]["low"] += 1
            elif result.complexity_score < 0.7:
                self.stats["complexity_distribution"]["medium"] += 1
            else:
                self.stats["complexity_distribution"]["high"] += 1
                
        except Exception as e:
            self.logger.error(f"Stats update failed: {e}")
    
    def add_conversation_context(self, user_input: str, assistant_response: str, metadata: Dict[str, Any]):
        """Add conversation turn to context"""
        self.context_manager.add_conversation_turn(user_input, assistant_response, metadata)
    
    def get_nlu_stats(self) -> Dict[str, Any]:
        """Get NLU processing statistics"""
        return {
            "processing_stats": self.stats.copy(),
            "intent_definitions": len(self.intent_definitions),
            "semantic_patterns": len(self.semantic_analyzer.semantic_patterns),
            "context_turns": len(self.context_manager.conversation_context)
        }