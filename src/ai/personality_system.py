"""
Personality System for JARVIS Voice Assistant
Implements multiple personality profiles and context-aware response generation
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass, asdict
from pathlib import Path
import random

@dataclass
class PersonalityProfile:
    """Personality profile configuration"""
    name: str
    description: str
    traits: Dict[str, float]  # Trait scores (0-1)
    response_style: Dict[str, Any]
    emotional_tendencies: Dict[str, float]
    language_patterns: Dict[str, List[str]]
    context_adaptations: Dict[str, Dict[str, Any]]
    
@dataclass
class ResponseContext:
    """Context for response generation"""
    user_emotion: Optional[Dict[str, Any]]
    conversation_history: List[Dict[str, Any]]
    user_preferences: Dict[str, Any]
    current_personality: str
    situational_context: Dict[str, Any]
    
class PersonalitySystem(QObject):
    """Advanced personality system with emotional intelligence"""
    
    # Signals
    personality_changed = pyqtSignal(str)  # new_personality
    response_adapted = pyqtSignal(dict)  # adaptation_info
    personality_learned = pyqtSignal(str, dict)  # trait, change
    context_analyzed = pyqtSignal(dict)  # context_summary
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("personality_system", {})
        
        # Current personality state
        self.current_personality: str = self.config.get("default_personality", "friendly")
        self.personality_profiles: Dict[str, PersonalityProfile] = {}
        self.adaptive_traits: Dict[str, float] = {}
        
        # Learning and adaptation
        self.learning_enabled = self.config.get("enable_learning", True)
        self.adaptation_rate = self.config.get("adaptation_rate", 0.1)
        self.user_interaction_history: List[Dict[str, Any]] = []
        
        # Context awareness
        self.context_memory_length = self.config.get("context_memory_length", 20)
        self.emotional_context_weight = self.config.get("emotional_context_weight", 0.7)
        
        # Thai language support
        self.thai_personality_adaptations = {}
        
        # Initialize personality profiles
        self._initialize_personality_profiles()
        self._load_thai_adaptations()
        
    def _initialize_personality_profiles(self):
        """Initialize built-in personality profiles"""
        try:
            # Professional personality
            professional = PersonalityProfile(
                name="professional",
                description="Formal, knowledgeable, and efficient assistant",
                traits={
                    "formality": 0.9,
                    "enthusiasm": 0.4,
                    "empathy": 0.6,
                    "humor": 0.2,
                    "directness": 0.8,
                    "supportiveness": 0.7,
                    "creativity": 0.5,
                    "patience": 0.8
                },
                response_style={
                    "tone": "formal",
                    "verbosity": "concise",
                    "technical_detail": "high",
                    "personal_touch": "minimal",
                    "confidence_level": "high"
                },
                emotional_tendencies={
                    "baseline_valence": 0.1,
                    "emotional_reactivity": 0.3,
                    "empathy_expression": 0.6,
                    "optimism_bias": 0.4
                },
                language_patterns={
                    "greetings": ["Good morning", "Good afternoon", "Good evening"],
                    "affirmations": ["Certainly", "Indeed", "Absolutely", "Correct"],
                    "transitions": ["Furthermore", "Additionally", "Moreover", "Consequently"],
                    "closings": ["I hope this helps", "Please let me know if you need further assistance"]
                },
                context_adaptations={
                    "stressed_user": {
                        "empathy": 0.8,
                        "patience": 0.9,
                        "tone": "calming",
                        "response_speed": "measured"
                    },
                    "urgent_request": {
                        "directness": 0.9,
                        "verbosity": "minimal",
                        "confidence_level": "very_high"
                    }
                }
            )
            
            # Friendly personality
            friendly = PersonalityProfile(
                name="friendly",
                description="Warm, approachable, and conversational assistant",
                traits={
                    "formality": 0.3,
                    "enthusiasm": 0.8,
                    "empathy": 0.9,
                    "humor": 0.7,
                    "directness": 0.5,
                    "supportiveness": 0.9,
                    "creativity": 0.7,
                    "patience": 0.8
                },
                response_style={
                    "tone": "warm",
                    "verbosity": "conversational",
                    "technical_detail": "moderate",
                    "personal_touch": "high",
                    "confidence_level": "moderate"
                },
                emotional_tendencies={
                    "baseline_valence": 0.6,
                    "emotional_reactivity": 0.7,
                    "empathy_expression": 0.9,
                    "optimism_bias": 0.8
                },
                language_patterns={
                    "greetings": ["Hi there!", "Hello!", "Hey!", "Good to see you!"],
                    "affirmations": ["Absolutely!", "You bet!", "Sure thing!", "Of course!"],
                    "transitions": ["Also", "Plus", "And another thing", "Oh, and"],
                    "closings": ["Hope that helps!", "Let me know how it goes!", "Happy to help anytime!"]
                },
                context_adaptations={
                    "happy_user": {
                        "enthusiasm": 0.9,
                        "humor": 0.8,
                        "tone": "celebratory"
                    },
                    "sad_user": {
                        "empathy": 1.0,
                        "supportiveness": 1.0,
                        "tone": "comforting",
                        "humor": 0.2
                    }
                }
            )
            
            # Casual personality
            casual = PersonalityProfile(
                name="casual",
                description="Relaxed, informal, and laid-back assistant",
                traits={
                    "formality": 0.1,
                    "enthusiasm": 0.6,
                    "empathy": 0.7,
                    "humor": 0.8,
                    "directness": 0.7,
                    "supportiveness": 0.7,
                    "creativity": 0.8,
                    "patience": 0.6
                },
                response_style={
                    "tone": "casual",
                    "verbosity": "brief",
                    "technical_detail": "low",
                    "personal_touch": "moderate",
                    "confidence_level": "relaxed"
                },
                emotional_tendencies={
                    "baseline_valence": 0.4,
                    "emotional_reactivity": 0.5,
                    "empathy_expression": 0.7,
                    "optimism_bias": 0.6
                },
                language_patterns={
                    "greetings": ["Hey", "What's up?", "Yo", "Sup"],
                    "affirmations": ["Yeah", "Yep", "Sure", "Gotcha", "Cool"],
                    "transitions": ["So", "Anyway", "Also", "Oh yeah"],
                    "closings": ["Cool!", "Catch ya later", "No prob", "Easy!"]
                },
                context_adaptations={
                    "excited_user": {
                        "enthusiasm": 0.8,
                        "humor": 0.9,
                        "tone": "energetic"
                    },
                    "confused_user": {
                        "patience": 0.8,
                        "technical_detail": "very_low",
                        "supportiveness": 0.8
                    }
                }
            )
            
            # Store personalities
            self.personality_profiles["professional"] = professional
            self.personality_profiles["friendly"] = friendly
            self.personality_profiles["casual"] = casual
            
            self.logger.info(f"Initialized {len(self.personality_profiles)} personality profiles")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize personality profiles: {e}")
    
    def _load_thai_adaptations(self):
        """Load Thai language personality adaptations"""
        try:
            self.thai_personality_adaptations = {
                "professional": {
                    "greetings": ["สวัสดีครับ/ค่ะ", "ดีครับ/ค่ะ"],
                    "polite_particles": ["ครับ", "ค่ะ", "นะครับ", "นะคะ"],
                    "formal_pronouns": ["ผม", "ดิฉัน", "คุณ", "ท่าน"],
                    "response_modifiers": ["กรุณา", "โปรด", "ขอ"]
                },
                "friendly": {
                    "greetings": ["สวัสดีจ้า", "หวัดดี", "ดีจ้า"],
                    "casual_particles": ["จ้า", "นะ", "เนอะ", "ว่ะ"],
                    "informal_pronouns": ["กู", "มึง", "เรา", "คุณ"],
                    "friendly_expressions": ["เอ็นดู", "น่ารัก", "เก่ง", "ดีมาก"]
                },
                "casual": {
                    "greetings": ["ว่าไง", "เป็นไง", "สบายดี"],
                    "very_casual_particles": ["วะ", "ว่ะ", "เฟ้ย"],
                    "slang_expressions": ["เจ๋ง", "โอเค", "โดน", "แน่นอน"],
                    "casual_endings": ["แล้วกัน", "นะ", "เดี๋ยว"]
                }
            }
            
            self.logger.info("Thai personality adaptations loaded")
            
        except Exception as e:
            self.logger.error(f"Failed to load Thai adaptations: {e}")
    
    def set_personality(self, personality_name: str) -> bool:
        """Set current personality profile"""
        try:
            if personality_name not in self.personality_profiles:
                self.logger.warning(f"Unknown personality: {personality_name}")
                return False
            
            old_personality = self.current_personality
            self.current_personality = personality_name
            
            self.personality_changed.emit(personality_name)
            self.logger.info(f"Personality changed from {old_personality} to {personality_name}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set personality: {e}")
            return False
    
    def adapt_response_to_emotion(self, response: str, emotion_context: Dict[str, Any], 
                                 language: str = "en") -> str:
        """Adapt response based on detected emotions and personality"""
        try:
            if not emotion_context:
                return response
            
            current_profile = self.personality_profiles.get(self.current_personality)
            if not current_profile:
                return response
            
            # Get user's emotional state
            primary_emotion = emotion_context.get("primary_emotion", "neutral")
            valence = emotion_context.get("valence", 0.0)
            arousal = emotion_context.get("arousal", 0.0)
            stress_level = emotion_context.get("stress_level", 0.0)
            
            # Apply personality-specific adaptations
            adapted_response = self._apply_emotional_adaptations(
                response, primary_emotion, valence, arousal, stress_level, current_profile, language
            )
            
            # Apply language-specific adaptations
            if language in ["th", "thai"]:
                adapted_response = self._apply_thai_adaptations(adapted_response, primary_emotion)
            
            # Add personality-specific language patterns
            adapted_response = self._apply_language_patterns(adapted_response, current_profile, language)
            
            # Record adaptation for learning
            self._record_adaptation(emotion_context, adapted_response)
            
            return adapted_response
            
        except Exception as e:
            self.logger.error(f"Failed to adapt response to emotion: {e}")
            return response
    
    def _apply_emotional_adaptations(self, response: str, emotion: str, valence: float, 
                                   arousal: float, stress_level: float, 
                                   profile: PersonalityProfile, language: str) -> str:
        """Apply emotional adaptations based on personality"""
        try:
            adaptations = []
            
            # Handle stress
            if stress_level > 0.7:
                if profile.traits["empathy"] > 0.7:
                    if language == "en":
                        adaptations.append("I understand this might be stressful. ")
                    else:
                        adaptations.append("เข้าใจว่าอาจจะเครียดนะ ")
                
                if profile.traits["patience"] > 0.7:
                    response = response.replace("quickly", "carefully")
                    response = response.replace("fast", "step by step")
            
            # Handle sadness
            if emotion in ["sadness", "disappointment"] and valence < -0.3:
                if profile.traits["empathy"] > 0.6:
                    if language == "en":
                        adaptations.append("I'm sorry to hear that. ")
                    else:
                        adaptations.append("เสียใจด้วยนะ ")
                
                if profile.traits["supportiveness"] > 0.7:
                    if language == "en":
                        adaptations.append("I'm here to help you through this. ")
                    else:
                        adaptations.append("อยู่ข้างๆ ช่วยเหลือเสมอนะ ")
            
            # Handle anger
            if emotion in ["anger", "frustration"] and valence < -0.5:
                if profile.traits["patience"] > 0.6:
                    if language == "en":
                        adaptations.append("I understand your frustration. Let's work through this together. ")
                    else:
                        adaptations.append("เข้าใจความรู้สึกนะ มาช่วยกันแก้ไขกันเถอะ ")
                
                # Make response more calm and measured
                response = response.replace("!", ".")
                response = response.replace("exciting", "helpful")
            
            # Handle joy/happiness
            if emotion in ["joy", "happiness", "excitement"] and valence > 0.5:
                if profile.traits["enthusiasm"] > 0.6:
                    if language == "en":
                        adaptations.append("That's wonderful! ")
                    else:
                        adaptations.append("ดีมากเลยจ้า! ")
                
                if profile.traits["humor"] > 0.5 and arousal > 0.3:
                    # Add more energetic language
                    response = response.replace("good", "fantastic")
                    response = response.replace("nice", "amazing")
            
            # Handle fear/anxiety
            if emotion in ["fear", "anxiety"] and arousal > 0.5:
                if profile.traits["supportiveness"] > 0.7:
                    if language == "en":
                        adaptations.append("Don't worry, I'll help you with this step by step. ")
                    else:
                        adaptations.append("ไม่ต้องกังวลนะ จะช่วยทีละขั้นตอน ")
            
            # Combine adaptations with response
            if adaptations:
                adapted_response = "".join(adaptations) + response
            else:
                adapted_response = response
            
            return adapted_response
            
        except Exception as e:
            self.logger.error(f"Failed to apply emotional adaptations: {e}")
            return response
    
    def _apply_thai_adaptations(self, response: str, emotion: str) -> str:
        """Apply Thai language cultural adaptations"""
        try:
            personality_thai = self.thai_personality_adaptations.get(self.current_personality, {})
            
            # Add appropriate politeness particles
            if self.current_personality == "professional":
                particles = personality_thai.get("polite_particles", ["ครับ", "ค่ะ"])
                if not any(particle in response for particle in particles):
                    response += f" {random.choice(particles)}"
            
            elif self.current_personality == "friendly":
                particles = personality_thai.get("casual_particles", ["จ้า", "นะ"])
                if emotion in ["joy", "happiness"]:
                    response += f" {random.choice(particles)}"
            
            elif self.current_personality == "casual":
                particles = personality_thai.get("very_casual_particles", ["วะ", "นะ"])
                if emotion not in ["sadness", "anger"]:  # Avoid casual particles for serious emotions
                    response += f" {random.choice(particles)}"
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to apply Thai adaptations: {e}")
            return response
    
    def _apply_language_patterns(self, response: str, profile: PersonalityProfile, language: str) -> str:
        """Apply personality-specific language patterns"""
        try:
            if language not in ["en", "english"]:
                return response  # Skip for non-English for now
            
            patterns = profile.language_patterns
            
            # Add personality-specific transitions
            if profile.traits["formality"] > 0.7:
                response = response.replace(" Also,", " Furthermore,")
                response = response.replace(" And", " Additionally")
            elif profile.traits["formality"] < 0.3:
                response = response.replace(" Furthermore,", " Also,")
                response = response.replace(" Additionally", " And")
            
            # Adjust verbosity based on personality
            if profile.response_style["verbosity"] == "concise" and len(response) > 200:
                # Simplify long responses for concise personalities
                response = response.replace(" in order to", " to")
                response = response.replace(" due to the fact that", " because")
            
            elif profile.response_style["verbosity"] == "conversational":
                # Add conversational elements for friendly personalities
                if profile.traits["enthusiasm"] > 0.6:
                    response = response.replace("Yes,", "Absolutely!")
                    response = response.replace("No,", "Not really,")
            
            return response
            
        except Exception as e:
            self.logger.error(f"Failed to apply language patterns: {e}")
            return response
    
    def analyze_context_for_personality(self, context: ResponseContext) -> Dict[str, Any]:
        """Analyze context to suggest personality adaptations"""
        try:
            analysis = {
                "suggested_personality": self.current_personality,
                "adaptation_reasons": [],
                "trait_adjustments": {},
                "confidence": 0.5
            }
            
            # Analyze user emotion
            if context.user_emotion:
                emotion = context.user_emotion.get("primary_emotion", "neutral")
                stress_level = context.user_emotion.get("stress_level", 0.0)
                
                # Suggest personality based on emotion
                if stress_level > 0.7 or emotion in ["anger", "frustration"]:
                    analysis["suggested_personality"] = "professional"
                    analysis["adaptation_reasons"].append("High stress detected - professional tone recommended")
                    analysis["trait_adjustments"]["patience"] = 0.9
                    analysis["trait_adjustments"]["empathy"] = 0.8
                
                elif emotion in ["sadness", "disappointment"]:
                    analysis["suggested_personality"] = "friendly"
                    analysis["adaptation_reasons"].append("Emotional support needed - friendly tone recommended")
                    analysis["trait_adjustments"]["empathy"] = 1.0
                    analysis["trait_adjustments"]["supportiveness"] = 1.0
                
                elif emotion in ["joy", "excitement"]:
                    analysis["suggested_personality"] = "friendly"
                    analysis["adaptation_reasons"].append("Positive emotion detected - friendly tone matches energy")
                    analysis["trait_adjustments"]["enthusiasm"] = 0.9
                    analysis["trait_adjustments"]["humor"] = 0.8
            
            # Analyze conversation history
            if context.conversation_history:
                recent_interactions = context.conversation_history[-5:]
                
                # Check for formal language patterns
                formal_indicators = ["please", "could you", "would you mind", "I would appreciate"]
                formal_count = sum(1 for msg in recent_interactions 
                                 if any(indicator in msg.get("user_input", "").lower() 
                                       for indicator in formal_indicators))
                
                if formal_count > len(recent_interactions) * 0.6:
                    analysis["suggested_personality"] = "professional"
                    analysis["adaptation_reasons"].append("Formal language pattern detected")
                    analysis["confidence"] += 0.2
            
            # Analyze user preferences
            if context.user_preferences:
                preferred_style = context.user_preferences.get("communication_style", "")
                if preferred_style == "formal":
                    analysis["suggested_personality"] = "professional"
                    analysis["adaptation_reasons"].append("User preference for formal communication")
                    analysis["confidence"] += 0.3
                elif preferred_style == "casual":
                    analysis["suggested_personality"] = "casual"
                    analysis["adaptation_reasons"].append("User preference for casual communication")
                    analysis["confidence"] += 0.3
            
            # Situational context
            if context.situational_context:
                situation = context.situational_context.get("type", "")
                if situation in ["urgent", "emergency", "critical"]:
                    analysis["suggested_personality"] = "professional"
                    analysis["adaptation_reasons"].append("Urgent situation requires professional approach")
                    analysis["trait_adjustments"]["directness"] = 0.9
                    analysis["confidence"] += 0.3
                elif situation in ["casual", "entertainment", "fun"]:
                    analysis["suggested_personality"] = "casual"
                    analysis["adaptation_reasons"].append("Casual situation allows relaxed approach")
                    analysis["trait_adjustments"]["humor"] = 0.8
                    analysis["confidence"] += 0.2
            
            self.context_analyzed.emit(analysis)
            return analysis
            
        except Exception as e:
            self.logger.error(f"Failed to analyze context: {e}")
            return {"suggested_personality": self.current_personality, "adaptation_reasons": [], "confidence": 0.0}
    
    def learn_from_interaction(self, user_feedback: Dict[str, Any], response_context: Dict[str, Any]):
        """Learn and adapt personality based on user feedback"""
        if not self.learning_enabled:
            return
        
        try:
            feedback_type = user_feedback.get("type", "")
            feedback_score = user_feedback.get("score", 0.5)  # 0-1 scale
            
            current_profile = self.personality_profiles.get(self.current_personality)
            if not current_profile:
                return
            
            # Adjust traits based on feedback
            if feedback_type == "too_formal" and feedback_score < 0.3:
                # Reduce formality
                adjustment = -self.adaptation_rate
                self._adjust_trait("formality", adjustment)
                self.personality_learned.emit("formality", {"adjustment": adjustment, "reason": "too_formal_feedback"})
            
            elif feedback_type == "too_casual" and feedback_score < 0.3:
                # Increase formality
                adjustment = self.adaptation_rate
                self._adjust_trait("formality", adjustment)
                self.personality_learned.emit("formality", {"adjustment": adjustment, "reason": "too_casual_feedback"})
            
            elif feedback_type == "not_empathetic" and feedback_score < 0.3:
                # Increase empathy
                adjustment = self.adaptation_rate
                self._adjust_trait("empathy", adjustment)
                self.personality_learned.emit("empathy", {"adjustment": adjustment, "reason": "empathy_feedback"})
            
            elif feedback_type == "too_verbose" and feedback_score < 0.3:
                # Adjust verbosity
                if current_profile.response_style["verbosity"] == "conversational":
                    current_profile.response_style["verbosity"] = "concise"
                elif current_profile.response_style["verbosity"] == "detailed":
                    current_profile.response_style["verbosity"] = "conversational"
            
            # Record interaction for future learning
            self.user_interaction_history.append({
                "timestamp": time.time(),
                "personality": self.current_personality,
                "feedback": user_feedback,
                "context": response_context
            })
            
            # Limit history size
            if len(self.user_interaction_history) > 100:
                self.user_interaction_history = self.user_interaction_history[-100:]
            
        except Exception as e:
            self.logger.error(f"Failed to learn from interaction: {e}")
    
    def _adjust_trait(self, trait_name: str, adjustment: float):
        """Adjust a personality trait"""
        try:
            current_profile = self.personality_profiles.get(self.current_personality)
            if not current_profile or trait_name not in current_profile.traits:
                return
            
            old_value = current_profile.traits[trait_name]
            new_value = max(0.0, min(1.0, old_value + adjustment))
            current_profile.traits[trait_name] = new_value
            
            self.logger.info(f"Adjusted {trait_name} from {old_value:.2f} to {new_value:.2f}")
            
        except Exception as e:
            self.logger.error(f"Failed to adjust trait {trait_name}: {e}")
    
    def _record_adaptation(self, emotion_context: Dict[str, Any], adapted_response: str):
        """Record adaptation for analysis and learning"""
        try:
            adaptation_info = {
                "timestamp": time.time(),
                "personality": self.current_personality,
                "emotion_context": emotion_context,
                "adapted_response_length": len(adapted_response),
                "adaptation_applied": True
            }
            
            self.response_adapted.emit(adaptation_info)
            
        except Exception as e:
            self.logger.error(f"Failed to record adaptation: {e}")
    
    def get_personality_info(self) -> Dict[str, Any]:
        """Get current personality information"""
        try:
            current_profile = self.personality_profiles.get(self.current_personality)
            if not current_profile:
                return {}
            
            return {
                "current_personality": self.current_personality,
                "available_personalities": list(self.personality_profiles.keys()),
                "current_traits": current_profile.traits,
                "response_style": current_profile.response_style,
                "emotional_tendencies": current_profile.emotional_tendencies,
                "description": current_profile.description,
                "learning_enabled": self.learning_enabled,
                "interaction_history_count": len(self.user_interaction_history)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get personality info: {e}")
            return {}
    
    def create_custom_personality(self, name: str, base_personality: str, trait_adjustments: Dict[str, float]) -> bool:
        """Create a custom personality based on existing one"""
        try:
            if base_personality not in self.personality_profiles:
                self.logger.warning(f"Base personality {base_personality} not found")
                return False
            
            # Copy base personality
            base_profile = self.personality_profiles[base_personality]
            custom_profile = PersonalityProfile(
                name=name,
                description=f"Custom personality based on {base_personality}",
                traits=base_profile.traits.copy(),
                response_style=base_profile.response_style.copy(),
                emotional_tendencies=base_profile.emotional_tendencies.copy(),
                language_patterns=base_profile.language_patterns.copy(),
                context_adaptations=base_profile.context_adaptations.copy()
            )
            
            # Apply trait adjustments
            for trait, adjustment in trait_adjustments.items():
                if trait in custom_profile.traits:
                    custom_profile.traits[trait] = max(0.0, min(1.0, custom_profile.traits[trait] + adjustment))
            
            # Store custom personality
            self.personality_profiles[name] = custom_profile
            
            self.logger.info(f"Created custom personality: {name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to create custom personality: {e}")
            return False
    
    def save_personality_state(self, file_path: str):
        """Save current personality state to file"""
        try:
            state = {
                "current_personality": self.current_personality,
                "adaptive_traits": self.adaptive_traits,
                "interaction_history": self.user_interaction_history[-50:],  # Save last 50 interactions
                "personality_profiles": {}
            }
            
            # Save personality profiles
            for name, profile in self.personality_profiles.items():
                state["personality_profiles"][name] = asdict(profile)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Personality state saved to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save personality state: {e}")
    
    def load_personality_state(self, file_path: str):
        """Load personality state from file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            self.current_personality = state.get("current_personality", "friendly")
            self.adaptive_traits = state.get("adaptive_traits", {})
            self.user_interaction_history = state.get("interaction_history", [])
            
            # Load personality profiles
            for name, profile_data in state.get("personality_profiles", {}).items():
                self.personality_profiles[name] = PersonalityProfile(**profile_data)
            
            self.logger.info(f"Personality state loaded from {file_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load personality state: {e}")