"""
Emotion Detection System for JARVIS Voice Assistant
Detects emotions from text and voice input using transformers and audio analysis
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass
import json
import time
from pathlib import Path

@dataclass
class EmotionResult:
    """Emotion detection result"""
    primary_emotion: str
    confidence: float
    emotion_scores: Dict[str, float]
    valence: float  # Positive/negative (-1 to 1)
    arousal: float  # Calm/excited (-1 to 1)
    intensity: float  # Weak/strong (0 to 1)
    source: str  # "text", "voice", or "combined"
    timestamp: float
    
@dataclass
class EmotionalContext:
    """Emotional context for conversation"""
    current_emotion: EmotionResult
    emotion_history: List[EmotionResult]
    mood_trend: str  # "improving", "declining", "stable"
    stress_level: float  # 0 to 1
    engagement_level: float  # 0 to 1
    emotional_stability: float  # 0 to 1

class EmotionDetectionSystem(QObject):
    """Advanced emotion detection with text and voice analysis"""
    
    # Signals
    emotion_detected = pyqtSignal(dict)  # EmotionResult as dict
    emotional_context_updated = pyqtSignal(dict)  # EmotionalContext as dict
    mood_change_detected = pyqtSignal(str, str)  # old_mood, new_mood
    stress_alert = pyqtSignal(float)  # stress_level
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("emotion_detection", {})
        
        # Emotion categories
        self.emotion_categories = {
            "positive": ["joy", "happiness", "excitement", "satisfaction", "love", "pride", "hope"],
            "negative": ["anger", "sadness", "fear", "disgust", "anxiety", "frustration", "disappointment"],
            "neutral": ["neutral", "calm", "contemplative", "focused", "curious"],
            "complex": ["surprise", "confusion", "nostalgia", "ambivalence", "anticipation"]
        }
        
        # Initialize models
        self.text_emotion_model = None
        self.voice_emotion_model = None
        self.sentiment_analyzer = None
        
        # Emotional state tracking
        self.current_emotional_context: Optional[EmotionalContext] = None
        self.emotion_history: List[EmotionResult] = []
        self.max_history_length = self.config.get("max_history_length", 50)
        
        # Voice analysis components
        self.voice_features_enabled = self.config.get("voice_analysis", True)
        self.prosody_analyzer = None
        
        # Language support
        self.supported_languages = ["en", "th"]
        
        # Initialize components
        self._initialize_text_emotion_detection()
        self._initialize_voice_emotion_detection()
        self._initialize_sentiment_analysis()
        
    def _initialize_text_emotion_detection(self):
        """Initialize text-based emotion detection"""
        try:
            from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
            
            # Use RoBERTa-based emotion detection model
            model_name = self.config.get("text_emotion_model", "j-hartmann/emotion-english-distilroberta-base")
            
            self.text_emotion_model = pipeline(
                "text-classification",
                model=model_name,
                tokenizer=model_name,
                return_all_scores=True,
                device=-1  # CPU by default
            )
            
            self.logger.info(f"Text emotion detection initialized: {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize text emotion detection: {e}")
            # Fallback to rule-based emotion detection
            self._initialize_rule_based_emotion_detection()
    
    def _initialize_rule_based_emotion_detection(self):
        """Initialize rule-based emotion detection as fallback"""
        try:
            self.emotion_keywords = {
                "joy": ["happy", "joy", "excited", "glad", "delighted", "pleased", "cheerful", "ดีใจ", "มีความสุข", "ยินดี"],
                "sadness": ["sad", "depressed", "down", "unhappy", "melancholy", "gloomy", "เศร้า", "หดหู่", "ไม่มีความสุข"],
                "anger": ["angry", "mad", "furious", "annoyed", "irritated", "upset", "โกรธ", "หงุดหงิด", "ขุ่นข้องใจ"],
                "fear": ["afraid", "scared", "worried", "anxious", "nervous", "terrified", "กลัว", "วิตกกังวล", "เครียด"],
                "surprise": ["surprised", "shocked", "amazed", "astonished", "stunned", "แปลกใจ", "ตกใจ", "ประหลาดใจ"],
                "disgust": ["disgusted", "revolted", "sick", "nauseous", "repulsed", "รังเกียจ", "คลื่นไส้"],
                "neutral": ["okay", "fine", "normal", "calm", "peaceful", "ปกติ", "สบายใจ", "เย็นใจ"]
            }
            
            self.logger.info("Rule-based emotion detection initialized as fallback")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize rule-based emotion detection: {e}")
    
    def _initialize_voice_emotion_detection(self):
        """Initialize voice-based emotion detection"""
        if not self.voice_features_enabled:
            return
            
        try:
            # Initialize librosa for audio feature extraction
            import librosa
            self.librosa = librosa
            
            # Voice emotion features to analyze
            self.voice_emotion_features = [
                "pitch_mean", "pitch_std", "pitch_range",
                "energy_mean", "energy_std",
                "speaking_rate", "pause_ratio",
                "spectral_centroid", "mfcc_features"
            ]
            
            self.logger.info("Voice emotion detection initialized")
            
        except ImportError:
            self.logger.warning("librosa not available, voice emotion detection disabled")
            self.voice_features_enabled = False
        except Exception as e:
            self.logger.warning(f"Could not initialize voice emotion detection: {e}")
            self.voice_features_enabled = False
    
    def _initialize_sentiment_analysis(self):
        """Initialize sentiment analysis for emotional context"""
        try:
            from transformers import pipeline
            
            # Use multilingual sentiment analysis
            model_name = self.config.get("sentiment_model", "cardiffnlp/twitter-roberta-base-sentiment-latest")
            
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model=model_name,
                tokenizer=model_name,
                device=-1  # CPU by default
            )
            
            self.logger.info(f"Sentiment analysis initialized: {model_name}")
            
        except Exception as e:
            self.logger.warning(f"Could not initialize sentiment analysis: {e}")
    
    def detect_emotion_from_text(self, text: str, language: str = "en") -> EmotionResult:
        """Detect emotion from text input"""
        try:
            emotion_scores = {}
            primary_emotion = "neutral"
            confidence = 0.5
            
            if self.text_emotion_model and language == "en":
                # Use transformer model for English
                results = self.text_emotion_model(text)
                
                for result in results[0]:  # First result contains all scores
                    emotion_scores[result['label'].lower()] = result['score']
                
                # Get primary emotion
                if emotion_scores:
                    primary_emotion = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[primary_emotion]
                    
            else:
                # Use rule-based detection
                emotion_scores = self._rule_based_emotion_detection(text, language)
                if emotion_scores:
                    primary_emotion = max(emotion_scores, key=emotion_scores.get)
                    confidence = emotion_scores[primary_emotion]
            
            # Calculate valence and arousal
            valence = self._calculate_valence(primary_emotion, emotion_scores)
            arousal = self._calculate_arousal(primary_emotion, emotion_scores)
            intensity = confidence
            
            emotion_result = EmotionResult(
                primary_emotion=primary_emotion,
                confidence=confidence,
                emotion_scores=emotion_scores,
                valence=valence,
                arousal=arousal,
                intensity=intensity,
                source="text",
                timestamp=time.time()
            )
            
            self.emotion_detected.emit(self._emotion_result_to_dict(emotion_result))
            return emotion_result
            
        except Exception as e:
            self.logger.error(f"Failed to detect emotion from text: {e}")
            return self._create_neutral_emotion_result("text")
    
    def detect_emotion_from_voice(self, audio_data: np.ndarray, sample_rate: int = 16000) -> EmotionResult:
        """Detect emotion from voice/audio input"""
        if not self.voice_features_enabled:
            return self._create_neutral_emotion_result("voice")
        
        try:
            # Extract voice features
            voice_features = self._extract_voice_features(audio_data, sample_rate)
            
            # Analyze emotional content from voice features
            emotion_scores = self._analyze_voice_emotions(voice_features)
            
            primary_emotion = "neutral"
            confidence = 0.5
            
            if emotion_scores:
                primary_emotion = max(emotion_scores, key=emotion_scores.get)
                confidence = emotion_scores[primary_emotion]
            
            # Calculate valence and arousal from voice
            valence = self._calculate_voice_valence(voice_features)
            arousal = self._calculate_voice_arousal(voice_features)
            intensity = confidence
            
            emotion_result = EmotionResult(
                primary_emotion=primary_emotion,
                confidence=confidence,
                emotion_scores=emotion_scores,
                valence=valence,
                arousal=arousal,
                intensity=intensity,
                source="voice",
                timestamp=time.time()
            )
            
            self.emotion_detected.emit(self._emotion_result_to_dict(emotion_result))
            return emotion_result
            
        except Exception as e:
            self.logger.error(f"Failed to detect emotion from voice: {e}")
            return self._create_neutral_emotion_result("voice")
    
    def detect_emotion_combined(self, text: str, audio_data: Optional[np.ndarray] = None, 
                              sample_rate: int = 16000, language: str = "en") -> EmotionResult:
        """Detect emotion from combined text and voice input"""
        try:
            # Get text emotion
            text_emotion = self.detect_emotion_from_text(text, language)
            
            # Get voice emotion if audio is available
            voice_emotion = None
            if audio_data is not None and self.voice_features_enabled:
                voice_emotion = self.detect_emotion_from_voice(audio_data, sample_rate)
            
            # Combine emotions
            if voice_emotion is not None:
                combined_emotion = self._combine_emotions(text_emotion, voice_emotion)
                combined_emotion.source = "combined"
            else:
                combined_emotion = text_emotion
            
            # Update emotional context
            self._update_emotional_context(combined_emotion)
            
            return combined_emotion
            
        except Exception as e:
            self.logger.error(f"Failed to detect combined emotion: {e}")
            return self._create_neutral_emotion_result("combined")
    
    def _rule_based_emotion_detection(self, text: str, language: str) -> Dict[str, float]:
        """Rule-based emotion detection using keywords"""
        try:
            text_lower = text.lower()
            emotion_scores = {}
            
            for emotion, keywords in self.emotion_keywords.items():
                score = 0.0
                for keyword in keywords:
                    if keyword in text_lower:
                        score += 1.0
                
                if score > 0:
                    # Normalize score
                    emotion_scores[emotion] = min(score / len(keywords), 1.0)
            
            # If no emotions detected, return neutral
            if not emotion_scores:
                emotion_scores["neutral"] = 0.7
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Rule-based emotion detection failed: {e}")
            return {"neutral": 0.5}
    
    def _extract_voice_features(self, audio_data: np.ndarray, sample_rate: int) -> Dict[str, float]:
        """Extract voice features for emotion analysis"""
        try:
            features = {}
            
            # Pitch features
            pitches, magnitudes = self.librosa.piptrack(y=audio_data, sr=sample_rate)
            pitch_values = pitches[magnitudes > np.percentile(magnitudes, 85)]
            
            if len(pitch_values) > 0:
                features["pitch_mean"] = np.mean(pitch_values)
                features["pitch_std"] = np.std(pitch_values)
                features["pitch_range"] = np.max(pitch_values) - np.min(pitch_values)
            else:
                features["pitch_mean"] = 0
                features["pitch_std"] = 0
                features["pitch_range"] = 0
            
            # Energy features
            energy = np.sum(audio_data ** 2)
            features["energy_mean"] = energy / len(audio_data)
            features["energy_std"] = np.std(audio_data ** 2)
            
            # Speaking rate (rough estimation)
            zero_crossings = self.librosa.zero_crossings(audio_data)
            features["speaking_rate"] = np.sum(zero_crossings) / len(audio_data)
            
            # Spectral features
            spectral_centroids = self.librosa.feature.spectral_centroid(y=audio_data, sr=sample_rate)[0]
            features["spectral_centroid"] = np.mean(spectral_centroids)
            
            # MFCC features
            mfccs = self.librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13)
            features["mfcc_mean"] = np.mean(mfccs)
            features["mfcc_std"] = np.std(mfccs)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Voice feature extraction failed: {e}")
            return {}
    
    def _analyze_voice_emotions(self, voice_features: Dict[str, float]) -> Dict[str, float]:
        """Analyze emotions from voice features"""
        try:
            emotion_scores = {}
            
            if not voice_features:
                return {"neutral": 0.5}
            
            # Simple heuristic-based emotion detection from voice
            pitch_mean = voice_features.get("pitch_mean", 0)
            pitch_std = voice_features.get("pitch_std", 0)
            energy_mean = voice_features.get("energy_mean", 0)
            speaking_rate = voice_features.get("speaking_rate", 0)
            
            # Happiness/Joy: Higher pitch, more variation, higher energy
            if pitch_mean > 150 and pitch_std > 20 and energy_mean > 0.01:
                emotion_scores["joy"] = 0.7
            
            # Sadness: Lower pitch, less variation, lower energy
            elif pitch_mean < 100 and pitch_std < 10 and energy_mean < 0.005:
                emotion_scores["sadness"] = 0.6
            
            # Anger: Higher pitch variation, higher energy, faster rate
            elif pitch_std > 30 and energy_mean > 0.015 and speaking_rate > 0.1:
                emotion_scores["anger"] = 0.6
            
            # Fear/Anxiety: Higher pitch, high variation, moderate energy
            elif pitch_mean > 130 and pitch_std > 25 and energy_mean > 0.008:
                emotion_scores["fear"] = 0.5
            
            # Default to neutral if no clear emotion
            else:
                emotion_scores["neutral"] = 0.6
            
            return emotion_scores
            
        except Exception as e:
            self.logger.error(f"Voice emotion analysis failed: {e}")
            return {"neutral": 0.5}
    
    def _calculate_valence(self, emotion: str, emotion_scores: Dict[str, float]) -> float:
        """Calculate valence (positive/negative) from emotion"""
        positive_emotions = ["joy", "happiness", "excitement", "satisfaction", "love", "pride", "hope"]
        negative_emotions = ["anger", "sadness", "fear", "disgust", "anxiety", "frustration", "disappointment"]
        
        if emotion in positive_emotions:
            return emotion_scores.get(emotion, 0.5)
        elif emotion in negative_emotions:
            return -emotion_scores.get(emotion, 0.5)
        else:
            return 0.0
    
    def _calculate_arousal(self, emotion: str, emotion_scores: Dict[str, float]) -> float:
        """Calculate arousal (calm/excited) from emotion"""
        high_arousal_emotions = ["anger", "excitement", "fear", "surprise", "anxiety"]
        low_arousal_emotions = ["sadness", "calm", "contentment", "boredom"]
        
        if emotion in high_arousal_emotions:
            return emotion_scores.get(emotion, 0.5)
        elif emotion in low_arousal_emotions:
            return -emotion_scores.get(emotion, 0.5)
        else:
            return 0.0
    
    def _calculate_voice_valence(self, voice_features: Dict[str, float]) -> float:
        """Calculate valence from voice features"""
        try:
            energy = voice_features.get("energy_mean", 0)
            pitch = voice_features.get("pitch_mean", 0)
            
            # Higher energy and moderate pitch typically indicate positive valence
            if energy > 0.01 and 120 < pitch < 180:
                return 0.6
            elif energy < 0.005 or pitch < 90:
                return -0.4
            else:
                return 0.0
                
        except Exception:
            return 0.0
    
    def _calculate_voice_arousal(self, voice_features: Dict[str, float]) -> float:
        """Calculate arousal from voice features"""
        try:
            energy = voice_features.get("energy_mean", 0)
            speaking_rate = voice_features.get("speaking_rate", 0)
            pitch_std = voice_features.get("pitch_std", 0)
            
            # High energy, fast speaking, high pitch variation indicate high arousal
            arousal_score = (energy * 100 + speaking_rate * 10 + pitch_std / 10) / 3
            return min(max(arousal_score - 0.5, -1.0), 1.0)
            
        except Exception:
            return 0.0
    
    def _combine_emotions(self, text_emotion: EmotionResult, voice_emotion: EmotionResult) -> EmotionResult:
        """Combine text and voice emotion results"""
        try:
            # Weight text and voice emotions
            text_weight = 0.6
            voice_weight = 0.4
            
            # Combine emotion scores
            combined_scores = {}
            all_emotions = set(text_emotion.emotion_scores.keys()) | set(voice_emotion.emotion_scores.keys())
            
            for emotion in all_emotions:
                text_score = text_emotion.emotion_scores.get(emotion, 0.0)
                voice_score = voice_emotion.emotion_scores.get(emotion, 0.0)
                combined_scores[emotion] = text_score * text_weight + voice_score * voice_weight
            
            # Get primary emotion
            primary_emotion = max(combined_scores, key=combined_scores.get)
            confidence = combined_scores[primary_emotion]
            
            # Combine valence, arousal, intensity
            valence = text_emotion.valence * text_weight + voice_emotion.valence * voice_weight
            arousal = text_emotion.arousal * text_weight + voice_emotion.arousal * voice_weight
            intensity = text_emotion.intensity * text_weight + voice_emotion.intensity * voice_weight
            
            return EmotionResult(
                primary_emotion=primary_emotion,
                confidence=confidence,
                emotion_scores=combined_scores,
                valence=valence,
                arousal=arousal,
                intensity=intensity,
                source="combined",
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Failed to combine emotions: {e}")
            return text_emotion
    
    def _update_emotional_context(self, emotion_result: EmotionResult):
        """Update emotional context with new emotion result"""
        try:
            # Add to history
            self.emotion_history.append(emotion_result)
            
            # Limit history size
            if len(self.emotion_history) > self.max_history_length:
                self.emotion_history = self.emotion_history[-self.max_history_length:]
            
            # Analyze mood trend
            mood_trend = self._analyze_mood_trend()
            
            # Calculate stress level
            stress_level = self._calculate_stress_level()
            
            # Calculate engagement level
            engagement_level = self._calculate_engagement_level()
            
            # Calculate emotional stability
            emotional_stability = self._calculate_emotional_stability()
            
            # Create emotional context
            self.current_emotional_context = EmotionalContext(
                current_emotion=emotion_result,
                emotion_history=self.emotion_history[-10:],  # Last 10 emotions
                mood_trend=mood_trend,
                stress_level=stress_level,
                engagement_level=engagement_level,
                emotional_stability=emotional_stability
            )
            
            # Emit signals
            context_dict = self._emotional_context_to_dict(self.current_emotional_context)
            self.emotional_context_updated.emit(context_dict)
            
            # Check for stress alert
            if stress_level > 0.7:
                self.stress_alert.emit(stress_level)
            
        except Exception as e:
            self.logger.error(f"Failed to update emotional context: {e}")
    
    def _analyze_mood_trend(self) -> str:
        """Analyze mood trend from recent emotions"""
        try:
            if len(self.emotion_history) < 3:
                return "stable"
            
            recent_valences = [e.valence for e in self.emotion_history[-5:]]
            
            if len(recent_valences) >= 3:
                trend = np.polyfit(range(len(recent_valences)), recent_valences, 1)[0]
                
                if trend > 0.1:
                    return "improving"
                elif trend < -0.1:
                    return "declining"
                else:
                    return "stable"
            
            return "stable"
            
        except Exception:
            return "stable"
    
    def _calculate_stress_level(self) -> float:
        """Calculate stress level from emotional history"""
        try:
            if not self.emotion_history:
                return 0.0
            
            # Look for stress indicators
            recent_emotions = self.emotion_history[-10:]
            stress_emotions = ["anger", "anxiety", "fear", "frustration"]
            
            stress_score = 0.0
            for emotion in recent_emotions:
                if emotion.primary_emotion in stress_emotions:
                    stress_score += emotion.confidence
                
                # High arousal with negative valence indicates stress
                if emotion.arousal > 0.5 and emotion.valence < -0.3:
                    stress_score += 0.3
            
            return min(stress_score / len(recent_emotions), 1.0)
            
        except Exception:
            return 0.0
    
    def _calculate_engagement_level(self) -> float:
        """Calculate engagement level from emotional activity"""
        try:
            if not self.emotion_history:
                return 0.5
            
            recent_emotions = self.emotion_history[-5:]
            
            # High arousal and varied emotions indicate high engagement
            avg_arousal = np.mean([abs(e.arousal) for e in recent_emotions])
            emotion_variety = len(set([e.primary_emotion for e in recent_emotions]))
            avg_confidence = np.mean([e.confidence for e in recent_emotions])
            
            engagement = (avg_arousal + (emotion_variety / 5) + avg_confidence) / 3
            return min(max(engagement, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability from variance in emotions"""
        try:
            if len(self.emotion_history) < 3:
                return 0.5
            
            recent_valences = [e.valence for e in self.emotion_history[-10:]]
            valence_std = np.std(recent_valences)
            
            # Lower variance indicates higher stability
            stability = max(1.0 - valence_std, 0.0)
            return stability
            
        except Exception:
            return 0.5
    
    def _create_neutral_emotion_result(self, source: str) -> EmotionResult:
        """Create a neutral emotion result"""
        return EmotionResult(
            primary_emotion="neutral",
            confidence=0.5,
            emotion_scores={"neutral": 0.5},
            valence=0.0,
            arousal=0.0,
            intensity=0.5,
            source=source,
            timestamp=time.time()
        )
    
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
    
    def _emotional_context_to_dict(self, context: EmotionalContext) -> Dict[str, Any]:
        """Convert EmotionalContext to dictionary"""
        return {
            "current_emotion": self._emotion_result_to_dict(context.current_emotion),
            "emotion_history": [self._emotion_result_to_dict(e) for e in context.emotion_history],
            "mood_trend": context.mood_trend,
            "stress_level": context.stress_level,
            "engagement_level": context.engagement_level,
            "emotional_stability": context.emotional_stability
        }
    
    def get_emotional_state_summary(self) -> Dict[str, Any]:
        """Get summary of current emotional state"""
        if not self.current_emotional_context:
            return {"status": "no_context", "message": "No emotional context available"}
        
        context = self.current_emotional_context
        
        return {
            "primary_emotion": context.current_emotion.primary_emotion,
            "confidence": context.current_emotion.confidence,
            "valence": context.current_emotion.valence,
            "arousal": context.current_emotion.arousal,
            "mood_trend": context.mood_trend,
            "stress_level": context.stress_level,
            "engagement_level": context.engagement_level,
            "emotional_stability": context.emotional_stability,
            "history_count": len(context.emotion_history),
            "timestamp": context.current_emotion.timestamp
        }
    
    def reset_emotional_state(self):
        """Reset emotional state and history"""
        self.emotion_history = []
        self.current_emotional_context = None
        self.logger.info("Emotional state reset")