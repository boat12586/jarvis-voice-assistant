"""
Web Integration for Emotional AI System
Enhances the existing web app with emotional intelligence capabilities
"""

import logging
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path

from ai.emotional_ai_engine import EmotionalAIEngine
from features.user_preference_system import UserPreferenceSystem

class WebEmotionalIntegration:
    """Integration layer for emotional AI in web application"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize emotional AI components
        self.emotional_ai = EmotionalAIEngine(config)
        self.user_preferences = UserPreferenceSystem(config)
        
        # Session management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_timeout = config.get("session_timeout", 3600)  # 1 hour
        
        # Web-specific configuration
        self.enable_voice_emotion = config.get("enable_voice_emotion", False)
        self.enable_auto_personality = config.get("enable_auto_personality", True)
        self.enable_user_learning = config.get("enable_user_learning", True)
        
    def initialize_web_session(self, session_id: str, user_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Initialize a new web session with emotional AI"""
        try:
            # Create session data
            session_data = {
                "session_id": session_id,
                "start_time": time.time(),
                "last_activity": time.time(),
                "user_id": f"web_user_{session_id}",
                "conversation_history": [],
                "emotional_context": {},
                "current_personality": "friendly",
                "user_preferences": {}
            }
            
            # Apply user context if provided
            if user_context:
                session_data.update(user_context)
            
            # Get or create user profile
            user_profile = self.user_preferences.create_or_get_user_profile(
                session_data["user_id"], 
                user_context
            )
            
            # Set initial personality based on user profile
            session_data["current_personality"] = user_profile.preferred_personality
            self.emotional_ai.set_personality(user_profile.preferred_personality)
            
            # Store session
            self.active_sessions[session_id] = session_data
            
            self.logger.info(f"Initialized emotional AI web session: {session_id}")
            
            return {
                "status": "success",
                "session_id": session_id,
                "personality": session_data["current_personality"],
                "emotional_features": {
                    "emotion_detection": True,
                    "personality_adaptation": self.enable_auto_personality,
                    "user_learning": self.enable_user_learning,
                    "voice_emotion": self.enable_voice_emotion
                }
            }
            
        except Exception as e:
            self.logger.error(f"Failed to initialize web session: {e}")
            return {"status": "error", "message": str(e)}
    
    def process_web_message(self, session_id: str, message: str, original_response: str, 
                           language: str = "en", audio_data: Optional[bytes] = None) -> Dict[str, Any]:
        """Process web message with emotional AI enhancement"""
        try:
            # Update session activity
            session = self.active_sessions.get(session_id)
            if not session:
                session = self.initialize_web_session(session_id)["session_id"]
                session = self.active_sessions[session_id]
            
            session["last_activity"] = time.time()
            
            # Get user ID
            user_id = session["user_id"]
            
            # Process with emotional AI
            emotional_response = self.emotional_ai.process_with_emotional_intelligence(
                user_input=message,
                original_response=original_response,
                audio_data=audio_data,
                language=language,
                user_id=user_id,
                conversation_context=session["conversation_history"]
            )
            
            # Update session with emotional context
            session["emotional_context"] = emotional_response.emotion_context
            session["current_personality"] = emotional_response.personality_used
            
            # Add to conversation history
            conversation_turn = {
                "timestamp": time.time(),
                "user_input": message,
                "original_response": original_response,
                "enhanced_response": emotional_response.adapted_response,
                "emotion_context": emotional_response.emotion_context,
                "personality_used": emotional_response.personality_used,
                "adaptations": emotional_response.adaptations_applied
            }
            
            session["conversation_history"].append(conversation_turn)
            
            # Limit conversation history
            if len(session["conversation_history"]) > 20:
                session["conversation_history"] = session["conversation_history"][-20:]
            
            # Learn from interaction
            if self.enable_user_learning:
                interaction_data = {
                    "user_input": message,
                    "user_language": language,
                    "emotion_context": emotional_response.emotion_context,
                    "assistant_response": emotional_response.adapted_response,
                    "conversation_context": session["conversation_history"][-5:],
                    "timestamp": time.time(),
                    "session_length": time.time() - session["start_time"]
                }
                
                self.user_preferences.learn_from_interaction(user_id, interaction_data)
            
            # Prepare web response
            web_response = {
                "status": "success",
                "original_response": original_response,
                "enhanced_response": emotional_response.adapted_response,
                "emotional_analysis": {
                    "primary_emotion": emotional_response.emotion_context.get("primary_emotion", "neutral"),
                    "confidence": emotional_response.emotion_context.get("confidence", 0.5),
                    "valence": emotional_response.emotion_context.get("valence", 0.0),
                    "arousal": emotional_response.emotion_context.get("arousal", 0.0),
                    "intensity": emotional_response.emotion_context.get("intensity", 0.5)
                },
                "personality_info": {
                    "current_personality": emotional_response.personality_used,
                    "adaptations_applied": emotional_response.adaptations_applied,
                    "available_personalities": self.emotional_ai.get_available_personalities()
                },
                "processing_info": {
                    "processing_time": emotional_response.processing_time,
                    "confidence": emotional_response.confidence,
                    "audio_processed": audio_data is not None
                },
                "session_info": {
                    "session_id": session_id,
                    "conversation_turns": len(session["conversation_history"]),
                    "session_duration": time.time() - session["start_time"]
                }
            }
            
            return web_response
            
        except Exception as e:
            self.logger.error(f"Failed to process web message with emotional AI: {e}")
            return {
                "status": "error",
                "message": str(e),
                "fallback_response": original_response
            }
    
    def get_session_emotional_summary(self, session_id: str) -> Dict[str, Any]:
        """Get emotional summary for a session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            
            # Get emotional state summary
            emotional_summary = self.emotional_ai.get_emotional_state_summary()
            
            # Get user recommendations
            user_id = session["user_id"]
            recommendations = self.user_preferences.get_user_recommendations(user_id)
            
            # Analyze session patterns
            conversation_history = session["conversation_history"]
            session_analysis = self._analyze_session_patterns(conversation_history)
            
            return {
                "status": "success",
                "session_id": session_id,
                "emotional_state": emotional_summary,
                "user_recommendations": recommendations,
                "session_analysis": session_analysis,
                "current_personality": session["current_personality"],
                "session_duration": time.time() - session["start_time"]
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session emotional summary: {e}")
            return {"status": "error", "message": str(e)}
    
    def _analyze_session_patterns(self, conversation_history: list) -> Dict[str, Any]:
        """Analyze patterns in session conversation"""
        try:
            if not conversation_history:
                return {}
            
            # Emotion patterns
            emotions = [turn.get("emotion_context", {}).get("primary_emotion", "neutral") 
                       for turn in conversation_history]
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Personality usage
            personalities = [turn.get("personality_used", "friendly") for turn in conversation_history]
            personality_counts = {}
            for personality in personalities:
                personality_counts[personality] = personality_counts.get(personality, 0) + 1
            
            # Adaptation patterns
            all_adaptations = []
            for turn in conversation_history:
                all_adaptations.extend(turn.get("adaptations", []))
            
            adaptation_counts = {}
            for adaptation in all_adaptations:
                adaptation_counts[adaptation] = adaptation_counts.get(adaptation, 0) + 1
            
            # Calculate averages
            valences = [turn.get("emotion_context", {}).get("valence", 0.0) 
                       for turn in conversation_history]
            avg_valence = sum(valences) / len(valences) if valences else 0.0
            
            arousals = [turn.get("emotion_context", {}).get("arousal", 0.0) 
                       for turn in conversation_history]
            avg_arousal = sum(arousals) / len(arousals) if arousals else 0.0
            
            return {
                "total_turns": len(conversation_history),
                "emotion_distribution": emotion_counts,
                "personality_usage": personality_counts,
                "adaptation_frequency": adaptation_counts,
                "average_valence": round(avg_valence, 2),
                "average_arousal": round(avg_arousal, 2),
                "emotional_trend": self._calculate_emotional_trend(valences),
                "most_common_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral",
                "most_used_personality": max(personality_counts, key=personality_counts.get) if personality_counts else "friendly"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to analyze session patterns: {e}")
            return {}
    
    def _calculate_emotional_trend(self, valences: list) -> str:
        """Calculate emotional trend from valence values"""
        if len(valences) < 3:
            return "stable"
        
        # Look at recent trend
        recent_valences = valences[-5:] if len(valences) >= 5 else valences
        
        if len(recent_valences) < 2:
            return "stable"
        
        # Simple trend calculation
        trend_sum = 0
        for i in range(1, len(recent_valences)):
            if recent_valences[i] > recent_valences[i-1]:
                trend_sum += 1
            elif recent_valences[i] < recent_valences[i-1]:
                trend_sum -= 1
        
        if trend_sum > 0:
            return "improving"
        elif trend_sum < 0:
            return "declining"
        else:
            return "stable"
    
    def set_session_personality(self, session_id: str, personality: str) -> Dict[str, Any]:
        """Set personality for a specific session"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            
            # Validate personality
            available_personalities = self.emotional_ai.get_available_personalities()
            if personality not in available_personalities:
                return {
                    "status": "error", 
                    "message": f"Invalid personality. Available: {available_personalities}"
                }
            
            # Set personality
            success = self.emotional_ai.set_personality(personality)
            if success:
                session["current_personality"] = personality
                
                # Record explicit preference
                user_id = session["user_id"]
                self.user_preferences.provide_explicit_feedback(
                    user_id, 
                    "personality_preference", 
                    {"preferred_personality": personality}
                )
                
                return {
                    "status": "success",
                    "personality": personality,
                    "message": f"Personality changed to {personality}"
                }
            else:
                return {"status": "error", "message": "Failed to set personality"}
            
        except Exception as e:
            self.logger.error(f"Failed to set session personality: {e}")
            return {"status": "error", "message": str(e)}
    
    def provide_user_feedback(self, session_id: str, feedback_type: str, feedback_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process user feedback for learning"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            
            user_id = session["user_id"]
            
            # Process feedback through user preference system
            self.user_preferences.provide_explicit_feedback(user_id, feedback_type, feedback_data)
            
            # Also provide feedback to emotional AI for personality learning
            context = {
                "session_id": session_id,
                "conversation_history": session["conversation_history"][-5:],
                "current_personality": session["current_personality"]
            }
            
            feedback_score = feedback_data.get("score", 0.5)
            self.emotional_ai.provide_feedback(user_id, feedback_type, feedback_score, context)
            
            return {
                "status": "success",
                "message": "Feedback processed",
                "feedback_type": feedback_type
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process user feedback: {e}")
            return {"status": "error", "message": str(e)}
    
    def get_user_profile(self, session_id: str) -> Dict[str, Any]:
        """Get user profile information"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            
            user_id = session["user_id"]
            
            # Get user profile
            profile = self.user_preferences.user_profiles.get(user_id)
            if not profile:
                return {"status": "error", "message": "User profile not found"}
            
            # Get recommendations
            recommendations = self.user_preferences.get_user_recommendations(user_id)
            
            # Get preference conflicts
            conflicts = self.user_preferences.detect_preference_conflicts(user_id)
            
            return {
                "status": "success",
                "user_profile": {
                    "user_id": profile.user_id,
                    "preferred_personality": profile.preferred_personality,
                    "communication_style": profile.communication_style,
                    "language_preference": profile.language_preference,
                    "technical_level": profile.technical_level,
                    "emotional_stability": profile.emotional_stability_score,
                    "dominant_emotions": profile.dominant_emotions,
                    "created_date": profile.created_date,
                    "last_updated": profile.last_updated
                },
                "recommendations": recommendations,
                "preference_conflicts": conflicts,
                "interaction_count": len(self.user_preferences.preference_history.get(user_id, []))
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get user profile: {e}")
            return {"status": "error", "message": str(e)}
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        try:
            current_time = time.time()
            expired_sessions = []
            
            for session_id, session in self.active_sessions.items():
                if current_time - session["last_activity"] > self.session_timeout:
                    expired_sessions.append(session_id)
            
            for session_id in expired_sessions:
                del self.active_sessions[session_id]
                self.logger.info(f"Cleaned up expired session: {session_id}")
            
            return len(expired_sessions)
            
        except Exception as e:
            self.logger.error(f"Failed to cleanup expired sessions: {e}")
            return 0
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        try:
            emotional_stats = self.emotional_ai.get_emotional_state_summary()
            preference_stats = self.user_preferences.get_system_stats()
            
            return {
                "active_sessions": len(self.active_sessions),
                "emotional_ai_stats": emotional_stats,
                "user_preference_stats": preference_stats,
                "features_enabled": {
                    "voice_emotion": self.enable_voice_emotion,
                    "auto_personality": self.enable_auto_personality,
                    "user_learning": self.enable_user_learning
                },
                "session_timeout": self.session_timeout
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get system stats: {e}")
            return {}
    
    def reset_user_data(self, session_id: str) -> Dict[str, Any]:
        """Reset user data for privacy"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                return {"status": "error", "message": "Session not found"}
            
            user_id = session["user_id"]
            
            # Reset emotional state
            self.emotional_ai.reset_emotional_state()
            
            # Clear session data
            session["conversation_history"] = []
            session["emotional_context"] = {}
            
            return {
                "status": "success",
                "message": "User data reset successfully"
            }
            
        except Exception as e:
            self.logger.error(f"Failed to reset user data: {e}")
            return {"status": "error", "message": str(e)}