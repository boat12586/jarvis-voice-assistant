"""
Emotional AI Engine for JARVIS Voice Assistant
Integrates emotion detection, personality system, and context-aware response generation
"""

import logging
import json
import time
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass, asdict

from .emotion_detection import EmotionDetectionSystem, EmotionResult, EmotionalContext
from .personality_system import PersonalitySystem, PersonalityProfile, ResponseContext

@dataclass
class EmotionalResponse:
    """Emotional AI response with full context"""
    original_response: str
    adapted_response: str
    emotion_context: Dict[str, Any]
    personality_used: str
    adaptations_applied: List[str]
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]

class EmotionalAIEngine(QObject):
    """Main emotional AI engine that coordinates all emotional intelligence components"""
    
    # Signals
    emotional_response_ready = pyqtSignal(dict)  # EmotionalResponse as dict
    emotion_state_changed = pyqtSignal(dict)  # emotional state
    personality_adapted = pyqtSignal(str, str)  # old_personality, new_personality
    user_preference_learned = pyqtSignal(str, dict)  # preference_type, data
    context_analysis_complete = pyqtSignal(dict)  # analysis results
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("emotional_ai", {})
        
        # Initialize components
        self.emotion_detector = EmotionDetectionSystem(config)
        self.personality_system = PersonalitySystem(config)
        
        # Current state
        self.current_emotional_context: Optional[EmotionalContext] = None
        self.conversation_context: List[Dict[str, Any]] = []
        self.user_preferences: Dict[str, Any] = {}
        
        # Configuration
        self.auto_personality_adaptation = self.config.get("auto_personality_adaptation", True)
        self.emotion_memory_length = self.config.get("emotion_memory_length", 20)
        self.response_cache: Dict[str, EmotionalResponse] = {}
        self.cache_max_size = self.config.get("cache_max_size", 50)
        
        # Connect signals
        self._connect_signals()
        
        # Load user preferences if available
        self._load_user_preferences()
        
    def _connect_signals(self):
        """Connect signals from emotion and personality systems"""
        try:
            # Emotion detection signals
            self.emotion_detector.emotion_detected.connect(self._on_emotion_detected)
            self.emotion_detector.emotional_context_updated.connect(self._on_emotional_context_updated)
            self.emotion_detector.mood_change_detected.connect(self._on_mood_change)
            self.emotion_detector.stress_alert.connect(self._on_stress_alert)
            
            # Personality system signals
            self.personality_system.personality_changed.connect(self._on_personality_changed)
            self.personality_system.context_analyzed.connect(self._on_context_analyzed)
            self.personality_system.personality_learned.connect(self._on_personality_learned)
            
        except Exception as e:
            self.logger.error(f"Failed to connect signals: {e}")
    
    def process_with_emotional_intelligence(self, 
                                          user_input: str,
                                          original_response: str,
                                          audio_data: Optional[bytes] = None,
                                          language: str = "en",
                                          user_id: str = "default",
                                          conversation_context: Optional[List[Dict[str, Any]]] = None) -> EmotionalResponse:
        """
        Process response with full emotional intelligence
        
        Args:
            user_input: User's text input
            original_response: Original AI response to enhance
            audio_data: Optional audio data for voice emotion detection
            language: Language of the conversation
            user_id: User identifier for personalization
            conversation_context: Recent conversation history
            
        Returns:
            EmotionalResponse with adapted response and full context
        """
        start_time = time.time()
        
        try:
            self.logger.info(f"Processing emotional AI response for user: {user_id}")
            
            # Convert audio data if provided
            audio_array = None
            if audio_data:
                audio_array = self._convert_audio_data(audio_data)
            
            # Detect emotions from user input
            emotion_result = self.emotion_detector.detect_emotion_combined(
                text=user_input,
                audio_data=audio_array,
                language=language
            )
            
            # Update conversation context
            if conversation_context:
                self.conversation_context = conversation_context[-self.emotion_memory_length:]
            
            # Create response context
            response_context = ResponseContext(
                user_emotion=self._emotion_result_to_dict(emotion_result),
                conversation_history=self.conversation_context,
                user_preferences=self.user_preferences.get(user_id, {}),
                current_personality=self.personality_system.current_personality,
                situational_context=self._analyze_situational_context(user_input, emotion_result)
            )
            
            # Analyze context for personality adaptation
            context_analysis = self.personality_system.analyze_context_for_personality(response_context)
            
            # Auto-adapt personality if enabled
            if self.auto_personality_adaptation and context_analysis.get("confidence", 0) > 0.7:
                suggested_personality = context_analysis.get("suggested_personality")
                if suggested_personality != self.personality_system.current_personality:
                    old_personality = self.personality_system.current_personality
                    self.personality_system.set_personality(suggested_personality)
                    self.personality_adapted.emit(old_personality, suggested_personality)
            
            # Adapt response using personality and emotion
            adapted_response = self.personality_system.adapt_response_to_emotion(
                response=original_response,
                emotion_context=self._emotion_result_to_dict(emotion_result),
                language=language
            )
            
            # Apply additional emotional enhancements
            enhanced_response = self._apply_emotional_enhancements(
                adapted_response, emotion_result, language
            )
            
            # Create emotional response object
            processing_time = time.time() - start_time
            emotional_response = EmotionalResponse(
                original_response=original_response,
                adapted_response=enhanced_response,
                emotion_context=self._emotion_result_to_dict(emotion_result),
                personality_used=self.personality_system.current_personality,
                adaptations_applied=self._get_applied_adaptations(original_response, enhanced_response),
                confidence=emotion_result.confidence,
                processing_time=processing_time,
                metadata={
                    "user_id": user_id,
                    "language": language,
                    "context_analysis": context_analysis,
                    "audio_processed": audio_data is not None,
                    "timestamp": time.time()
                }
            )
            
            # Cache response
            self._cache_response(user_input, emotional_response)
            
            # Update user preferences based on interaction
            self._update_user_preferences(user_id, emotion_result, response_context)
            
            # Emit signal
            self.emotional_response_ready.emit(self._emotional_response_to_dict(emotional_response))
            
            return emotional_response
            
        except Exception as e:
            self.logger.error(f"Failed to process emotional AI response: {e}")
            # Return fallback response
            return EmotionalResponse(
                original_response=original_response,
                adapted_response=original_response,
                emotion_context={},
                personality_used="neutral",
                adaptations_applied=[],
                confidence=0.0,
                processing_time=time.time() - start_time,
                metadata={"error": str(e)}
            )
    
    def _convert_audio_data(self, audio_data: bytes) -> Optional[bytes]:
        """Convert audio data to appropriate format for emotion detection"""
        try:
            # For now, assume audio data is in the correct format
            # In a real implementation, you might need to convert formats
            return audio_data
        except Exception as e:
            self.logger.warning(f"Failed to convert audio data: {e}")
            return None
    
    def _analyze_situational_context(self, user_input: str, emotion_result: EmotionResult) -> Dict[str, Any]:
        """Analyze situational context from user input and emotions"""
        try:
            context = {
                "type": "normal",
                "urgency": "low",
                "complexity": "medium",
                "emotional_intensity": emotion_result.intensity
            }
            
            input_lower = user_input.lower()
            
            # Detect urgency
            urgent_keywords = ["urgent", "emergency", "asap", "immediately", "critical", "help", "problem"]
            if any(keyword in input_lower for keyword in urgent_keywords):
                context["urgency"] = "high"
                context["type"] = "urgent"
            
            # Detect complexity
            complex_keywords = ["complex", "detailed", "comprehensive", "analyze", "explain", "breakdown"]
            if any(keyword in input_lower for keyword in complex_keywords) or len(user_input) > 200:
                context["complexity"] = "high"
            
            # Detect casual context
            casual_keywords = ["chat", "fun", "joke", "casual", "relax", "entertainment"]
            if any(keyword in input_lower for keyword in casual_keywords):
                context["type"] = "casual"
            
            # Detect formal context
            formal_keywords = ["please", "would you", "could you", "i would appreciate", "formal"]
            if any(keyword in input_lower for keyword in formal_keywords):
                context["type"] = "formal"
            
            # Detect emotional support need
            support_keywords = ["sad", "worried", "anxious", "stressed", "depressed", "help me"]
            if any(keyword in input_lower for keyword in support_keywords) or emotion_result.valence < -0.5:
                context["type"] = "emotional_support"
            
            return context
            
        except Exception as e:
            self.logger.error(f"Failed to analyze situational context: {e}")
            return {"type": "normal", "urgency": "low", "complexity": "medium"}
    
    def _apply_emotional_enhancements(self, response: str, emotion_result: EmotionResult, language: str) -> str:
        """Apply additional emotional enhancements to response"""
        try:
            enhanced_response = response
            
            # Add emotional acknowledgment if needed
            if emotion_result.intensity > 0.7:
                if emotion_result.primary_emotion in ["sadness", "disappointment"]:
                    if language == "en":
                        enhanced_response = "I can sense this is important to you. " + enhanced_response
                    else:
                        enhanced_response = "รู้สึกว่าเรื่องนี้สำคัญกับคุณมาก " + enhanced_response
                
                elif emotion_result.primary_emotion in ["anger", "frustration"]:
                    if language == "en":
                        enhanced_response = "I understand your frustration. Let me help resolve this. " + enhanced_response
                    else:
                        enhanced_response = "เข้าใจความรู้สึกของคุณ ให้ช่วยแก้ไขเรื่องนี้นะ " + enhanced_response
                
                elif emotion_result.primary_emotion in ["joy", "excitement"]:
                    if language == "en":
                        enhanced_response = "I'm glad to hear your enthusiasm! " + enhanced_response
                    else:
                        enhanced_response = "ดีใจที่เห็นคุณกระตือรือร้นจัง! " + enhanced_response
            
            # Adjust response length based on emotional state
            if emotion_result.primary_emotion in ["anxiety", "stress"] and len(enhanced_response) > 300:
                # Simplify for anxious users
                enhanced_response = self._simplify_response(enhanced_response)
            
            return enhanced_response
            
        except Exception as e:
            self.logger.error(f"Failed to apply emotional enhancements: {e}")
            return response
    
    def _simplify_response(self, response: str) -> str:
        """Simplify response for users who might be overwhelmed"""
        try:
            # Break into simpler sentences
            simplified = response.replace("; however,", ". ")
            simplified = simplified.replace("; furthermore,", ". Also, ")
            simplified = simplified.replace(" due to the fact that", " because")
            simplified = simplified.replace(" in order to", " to")
            
            # Limit to essential information
            sentences = simplified.split(". ")
            if len(sentences) > 3:
                simplified = ". ".join(sentences[:3]) + "."
            
            return simplified
            
        except Exception:
            return response
    
    def _get_applied_adaptations(self, original: str, adapted: str) -> List[str]:
        """Identify what adaptations were applied to the response"""
        adaptations = []
        
        try:
            if len(adapted) > len(original):
                adaptations.append("emotional_acknowledgment")
            
            if original != adapted:
                if "sorry" in adapted.lower() and "sorry" not in original.lower():
                    adaptations.append("empathy_added")
                
                if "!" in adapted and "!" not in original:
                    adaptations.append("enthusiasm_added")
                
                if any(word in adapted.lower() for word in ["understand", "sense", "feel"]):
                    adaptations.append("emotional_validation")
                
                if "step by step" in adapted.lower() and "step by step" not in original.lower():
                    adaptations.append("stress_accommodation")
            
            return adaptations
            
        except Exception:
            return ["adaptation_applied"]
    
    def _update_user_preferences(self, user_id: str, emotion_result: EmotionResult, context: ResponseContext):
        """Update user preferences based on emotional patterns"""
        try:
            if user_id not in self.user_preferences:
                self.user_preferences[user_id] = {}
            
            user_prefs = self.user_preferences[user_id]
            
            # Track emotional patterns
            if "emotional_patterns" not in user_prefs:
                user_prefs["emotional_patterns"] = {}
            
            emotion = emotion_result.primary_emotion
            if emotion in user_prefs["emotional_patterns"]:
                user_prefs["emotional_patterns"][emotion] += 1
            else:
                user_prefs["emotional_patterns"][emotion] = 1
            
            # Track personality preferences
            if "preferred_personality" not in user_prefs:
                user_prefs["preferred_personality"] = {}
            
            current_personality = self.personality_system.current_personality
            if current_personality in user_prefs["preferred_personality"]:
                user_prefs["preferred_personality"][current_personality] += 1
            else:
                user_prefs["preferred_personality"][current_personality] = 1
            
            # Determine dominant emotional state
            total_emotions = sum(user_prefs["emotional_patterns"].values())
            if total_emotions > 10:  # After enough interactions
                dominant_emotion = max(user_prefs["emotional_patterns"], 
                                     key=user_prefs["emotional_patterns"].get)
                user_prefs["dominant_emotional_state"] = dominant_emotion
            
            # Determine preferred communication style
            if total_emotions > 15:
                preferred_personality = max(user_prefs["preferred_personality"],
                                          key=user_prefs["preferred_personality"].get)
                user_prefs["preferred_communication_style"] = preferred_personality
            
            # Save preferences
            self._save_user_preferences()
            
        except Exception as e:
            self.logger.error(f"Failed to update user preferences: {e}")
    
    def _cache_response(self, user_input: str, response: EmotionalResponse):
        """Cache emotional response for similar inputs"""
        try:
            cache_key = f"{user_input[:50]}_{response.personality_used}_{response.emotion_context.get('primary_emotion', 'neutral')}"
            
            self.response_cache[cache_key] = response
            
            # Limit cache size
            if len(self.response_cache) > self.cache_max_size:
                oldest_key = min(self.response_cache.keys(), 
                               key=lambda k: self.response_cache[k].metadata.get("timestamp", 0))
                del self.response_cache[oldest_key]
                
        except Exception as e:
            self.logger.error(f"Failed to cache response: {e}")
    
    def get_cached_response(self, user_input: str, current_emotion: str, current_personality: str) -> Optional[EmotionalResponse]:
        """Get cached response if available"""
        try:
            cache_key = f"{user_input[:50]}_{current_personality}_{current_emotion}"
            return self.response_cache.get(cache_key)
        except Exception:
            return None
    
    def _load_user_preferences(self):
        """Load user preferences from storage"""
        try:
            prefs_file = Path(self.config.get("preferences_file", "data/user_emotional_preferences.json"))
            if prefs_file.exists():
                with open(prefs_file, 'r', encoding='utf-8') as f:
                    self.user_preferences = json.load(f)
                self.logger.info("User emotional preferences loaded")
        except Exception as e:
            self.logger.warning(f"Could not load user preferences: {e}")
            self.user_preferences = {}
    
    def _save_user_preferences(self):
        """Save user preferences to storage"""
        try:
            prefs_file = Path(self.config.get("preferences_file", "data/user_emotional_preferences.json"))
            prefs_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(prefs_file, 'w', encoding='utf-8') as f:
                json.dump(self.user_preferences, f, ensure_ascii=False, indent=2)
        except Exception as e:
            self.logger.error(f"Failed to save user preferences: {e}")
    
    def _emotion_result_to_dict(self, emotion_result: EmotionResult) -> Dict[str, Any]:
        """Convert EmotionResult to dictionary"""
        return {
            "primary_emotion": emotion_result.primary_emotion,
            "confidence": emotion_result.confidence,
            "emotion_scores": emotion_result.emotion_scores,
            "valence": emotion_result.valence,
            "arousal": emotion_result.arousal,
            "intensity": emotion_result.intensity,
            "source": emotion_result.source,
            "timestamp": emotion_result.timestamp
        }
    
    def _emotional_response_to_dict(self, response: EmotionalResponse) -> Dict[str, Any]:
        """Convert EmotionalResponse to dictionary"""
        return {
            "original_response": response.original_response,
            "adapted_response": response.adapted_response,
            "emotion_context": response.emotion_context,
            "personality_used": response.personality_used,
            "adaptations_applied": response.adaptations_applied,
            "confidence": response.confidence,
            "processing_time": response.processing_time,
            "metadata": response.metadata
        }
    
    # Signal handlers
    def _on_emotion_detected(self, emotion_dict: Dict[str, Any]):
        """Handle emotion detection signal"""
        self.logger.info(f"Emotion detected: {emotion_dict.get('primary_emotion', 'unknown')}")
    
    def _on_emotional_context_updated(self, context_dict: Dict[str, Any]):
        """Handle emotional context update"""
        self.emotion_state_changed.emit(context_dict)
    
    def _on_mood_change(self, old_mood: str, new_mood: str):
        """Handle mood change detection"""
        self.logger.info(f"Mood change detected: {old_mood} -> {new_mood}")
    
    def _on_stress_alert(self, stress_level: float):
        """Handle stress alert"""
        self.logger.warning(f"High stress level detected: {stress_level:.2f}")
        # Could trigger automatic personality adaptation to more supportive mode
    
    def _on_personality_changed(self, new_personality: str):
        """Handle personality change"""
        self.logger.info(f"Personality changed to: {new_personality}")
    
    def _on_context_analyzed(self, analysis: Dict[str, Any]):
        """Handle context analysis completion"""
        self.context_analysis_complete.emit(analysis)
    
    def _on_personality_learned(self, trait: str, change: Dict[str, Any]):
        """Handle personality learning"""
        self.user_preference_learned.emit(trait, change)
    
    # Public API methods
    def set_personality(self, personality: str) -> bool:
        """Set current personality"""
        return self.personality_system.set_personality(personality)
    
    def get_current_personality(self) -> str:
        """Get current personality"""
        return self.personality_system.current_personality
    
    def get_available_personalities(self) -> List[str]:
        """Get list of available personalities"""
        return list(self.personality_system.personality_profiles.keys())
    
    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """Get summary of current emotional state"""
        emotion_summary = self.emotion_detector.get_emotional_state_summary()
        personality_info = self.personality_system.get_personality_info()
        
        return {
            "emotional_state": emotion_summary,
            "personality_info": personality_info,
            "user_preferences_count": len(self.user_preferences),
            "cache_size": len(self.response_cache),
            "auto_adaptation_enabled": self.auto_personality_adaptation
        }
    
    def reset_emotional_state(self):
        """Reset all emotional state"""
        self.emotion_detector.reset_emotional_state()
        self.conversation_context = []
        self.response_cache = {}
        self.logger.info("Emotional AI state reset")
    
    def enable_auto_personality_adaptation(self, enabled: bool):
        """Enable or disable automatic personality adaptation"""
        self.auto_personality_adaptation = enabled
        self.logger.info(f"Auto personality adaptation: {'enabled' if enabled else 'disabled'}")
    
    def provide_feedback(self, user_id: str, feedback_type: str, feedback_score: float, context: Dict[str, Any]):
        """Provide feedback for learning"""
        self.personality_system.learn_from_interaction(
            {"type": feedback_type, "score": feedback_score},
            context
        )