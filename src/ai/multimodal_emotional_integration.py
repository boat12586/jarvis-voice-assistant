"""
Multimodal Emotional AI Integration for JARVIS Voice Assistant
Integrates emotion detection with multimodal AI for enhanced understanding
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import time
from dataclasses import dataclass

# Import emotional AI components
from .emotion_detection import EmotionDetectionSystem, EmotionResult, EmotionalContext
from .sentiment_analysis import SentimentAnalyzer

# Import multimodal components
from .multimodal_fusion_system import (
    MultimodalFusionSystem, 
    ModalityInput, 
    ModalityType,
    FusionResult
)


@dataclass
class EmotionalMultimodalContext:
    """Enhanced context combining emotional and multimodal data"""
    emotion_result: EmotionResult
    multimodal_result: FusionResult
    emotional_context: Optional[EmotionalContext]
    visual_emotion_cues: Dict[str, Any]
    text_emotion_alignment: float
    overall_emotional_confidence: float
    timestamp: datetime


class MultimodalEmotionalIntegration:
    """Integration system for emotional AI and multimodal processing"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Initialize emotional AI components
        emotion_config = config.get('emotion', {})
        self.emotion_system = EmotionDetectionSystem(emotion_config)
        self.sentiment_analyzer = SentimentAnalyzer(emotion_config)
        
        # Initialize multimodal system
        multimodal_config = config.get('multimodal', {})
        self.multimodal_system = MultimodalFusionSystem(multimodal_config)
        
        # Emotional-visual mappings
        self.emotion_visual_keywords = {
            'joy': ['smile', 'happy', 'celebration', 'party', 'laughter', 'bright'],
            'sadness': ['crying', 'tears', 'dark', 'gloomy', 'rain', 'alone'],
            'anger': ['fist', 'red', 'aggressive', 'fighting', 'confrontation'],
            'fear': ['hiding', 'running', 'scared', 'darkness', 'danger'],
            'surprise': ['wide eyes', 'open mouth', 'unexpected', 'shocking'],
            'disgust': ['rejecting', 'turning away', 'unpleasant', 'dirty'],
            'neutral': ['calm', 'peaceful', 'normal', 'everyday', 'routine']
        }
        
        # Integration weights
        self.integration_weights = config.get('integration_weights', {
            'text_emotion': 0.4,
            'visual_emotion': 0.3,
            'context_emotion': 0.2,
            'voice_emotion': 0.1
        })
        
        # Performance tracking
        self.integration_stats = {
            'integrations_performed': 0,
            'emotional_accuracy': 0.0,
            'visual_emotion_detections': 0,
            'text_emotion_alignments': 0
        }
    
    def process_multimodal_with_emotion(self, 
                                      modality_inputs: List[ModalityInput],
                                      user_id: Optional[str] = None,
                                      context: Optional[Dict[str, Any]] = None) -> EmotionalMultimodalContext:
        """Process multimodal input with emotional analysis"""
        
        try:
            # Perform standard multimodal fusion
            multimodal_result = self.multimodal_system.fuse_multimodal_input(
                modality_inputs, context=context
            )
            
            # Extract emotional information from different modalities
            emotional_data = self._extract_emotional_data(modality_inputs)
            
            # Detect emotions from text
            text_emotion = None
            if emotional_data['text']:
                text_emotion = self.emotion_system.detect_emotion_from_text(
                    emotional_data['text'],
                    language=emotional_data.get('language', 'en')
                )
            
            # Detect emotions from voice
            voice_emotion = None
            if emotional_data['voice_data'] is not None:
                voice_emotion = self.emotion_system.detect_emotion_from_voice(
                    emotional_data['voice_data'],
                    emotional_data.get('sample_rate', 16000)
                )
            
            # Analyze visual emotional cues
            visual_emotion_cues = {}
            if emotional_data['visual_data']:
                visual_emotion_cues = self._analyze_visual_emotion_cues(
                    emotional_data['visual_data'],
                    multimodal_result.detailed_analysis
                )
            
            # Combine emotions
            combined_emotion = self._combine_multimodal_emotions(
                text_emotion, voice_emotion, visual_emotion_cues
            )
            
            # Get current emotional context
            emotional_context = self.emotion_system.current_emotional_context
            
            # Calculate alignment scores
            text_emotion_alignment = self._calculate_text_emotion_alignment(
                text_emotion, multimodal_result
            )
            
            # Calculate overall emotional confidence
            overall_confidence = self._calculate_overall_emotional_confidence(
                combined_emotion, visual_emotion_cues, text_emotion_alignment
            )
            
            # Create integrated context
            integrated_context = EmotionalMultimodalContext(
                emotion_result=combined_emotion,
                multimodal_result=multimodal_result,
                emotional_context=emotional_context,
                visual_emotion_cues=visual_emotion_cues,
                text_emotion_alignment=text_emotion_alignment,
                overall_emotional_confidence=overall_confidence,
                timestamp=datetime.now()
            )
            
            # Update statistics
            self._update_integration_stats(integrated_context)
            
            return integrated_context
            
        except Exception as e:
            self.logger.error(f"Multimodal emotional integration failed: {e}")
            # Return fallback context
            return self._create_fallback_context(modality_inputs)
    
    def _extract_emotional_data(self, modality_inputs: List[ModalityInput]) -> Dict[str, Any]:
        """Extract emotional data from modality inputs"""
        
        emotional_data = {
            'text': '',
            'voice_data': None,
            'visual_data': None,
            'language': 'en',
            'sample_rate': 16000
        }
        
        for input_data in modality_inputs:
            if input_data.modality == ModalityType.TEXT:
                emotional_data['text'] = input_data.data
                
                # Detect language
                if self._contains_thai_text(input_data.data):
                    emotional_data['language'] = 'th'
            
            elif input_data.modality == ModalityType.VOICE:
                if isinstance(input_data.data, dict):
                    emotional_data['voice_data'] = input_data.data.get('audio_features')
                    emotional_data['sample_rate'] = input_data.data.get('sample_rate', 16000)
                elif isinstance(input_data.data, np.ndarray):
                    emotional_data['voice_data'] = input_data.data
            
            elif input_data.modality == ModalityType.VISION:
                emotional_data['visual_data'] = input_data.data
        
        return emotional_data
    
    def _analyze_visual_emotion_cues(self, visual_data: Union[str, Dict[str, Any]], 
                                   visual_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze emotional cues from visual content"""
        
        emotion_cues = {
            'detected_emotions': {},
            'confidence': 0.0,
            'visual_sentiment': 'neutral',
            'emotion_keywords_found': []
        }
        
        try:
            # Analyze image caption for emotional keywords
            if 'vision' in visual_analysis:
                vision_data = visual_analysis['vision']
                
                # Check caption for emotional keywords
                if 'caption' in vision_data:
                    caption = vision_data['caption'].lower()
                    emotion_cues['emotion_keywords_found'] = self._find_emotion_keywords_in_text(caption)
                
                # Analyze detected objects for emotional context
                if 'detected_objects' in vision_data:
                    objects = vision_data['detected_objects']
                    emotion_cues['object_emotions'] = self._analyze_object_emotions(objects)
                
                # Analyze OCR text for emotions
                if 'ocr_results' in vision_data and vision_data['ocr_results'].get('full_text'):
                    ocr_text = vision_data['ocr_results']['full_text']
                    ocr_emotion_keywords = self._find_emotion_keywords_in_text(ocr_text.lower())
                    emotion_cues['emotion_keywords_found'].extend(ocr_emotion_keywords)
            
            # Calculate visual sentiment based on findings
            emotion_cues['visual_sentiment'] = self._calculate_visual_sentiment(emotion_cues)
            
            # Calculate overall confidence
            if emotion_cues['emotion_keywords_found'] or emotion_cues.get('object_emotions'):
                emotion_cues['confidence'] = min(0.8, len(emotion_cues['emotion_keywords_found']) * 0.2 + 0.4)
            
        except Exception as e:
            self.logger.error(f"Visual emotion analysis failed: {e}")
        
        return emotion_cues
    
    def _find_emotion_keywords_in_text(self, text: str) -> List[Dict[str, Any]]:
        """Find emotion-related keywords in text"""
        
        found_keywords = []
        
        for emotion, keywords in self.emotion_visual_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    found_keywords.append({
                        'keyword': keyword,
                        'emotion': emotion,
                        'confidence': 0.7
                    })
        
        return found_keywords
    
    def _analyze_object_emotions(self, objects: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze emotional context from detected objects"""
        
        object_emotions = {}
        
        # Emotional mappings for common objects
        emotional_objects = {
            'person': {'joy': 0.3, 'neutral': 0.7},
            'smile': {'joy': 0.9},
            'face': {'neutral': 0.6, 'joy': 0.2, 'sadness': 0.2},
            'flower': {'joy': 0.6, 'neutral': 0.4},
            'food': {'joy': 0.4, 'neutral': 0.6},
            'animal': {'joy': 0.5, 'neutral': 0.5},
            'car': {'neutral': 0.8, 'excitement': 0.2},
            'weapon': {'fear': 0.7, 'anger': 0.3},
            'fire': {'fear': 0.6, 'danger': 0.4}
        }
        
        for obj in objects:
            obj_class = obj.get('class', '').lower()
            confidence = obj.get('confidence', 0.0)
            
            if obj_class in emotional_objects:
                for emotion, score in emotional_objects[obj_class].items():
                    if emotion not in object_emotions:
                        object_emotions[emotion] = 0.0
                    object_emotions[emotion] += score * confidence
        
        # Normalize scores
        total_score = sum(object_emotions.values())
        if total_score > 0:
            for emotion in object_emotions:
                object_emotions[emotion] /= total_score
        
        return object_emotions
    
    def _calculate_visual_sentiment(self, emotion_cues: Dict[str, Any]) -> str:
        """Calculate overall visual sentiment"""
        
        positive_indicators = 0
        negative_indicators = 0
        
        # Check emotion keywords
        for keyword_data in emotion_cues['emotion_keywords_found']:
            emotion = keyword_data['emotion']
            if emotion in ['joy', 'happiness', 'excitement', 'love']:
                positive_indicators += 1
            elif emotion in ['sadness', 'anger', 'fear', 'disgust']:
                negative_indicators += 1
        
        # Check object emotions
        object_emotions = emotion_cues.get('object_emotions', {})
        positive_emotions = object_emotions.get('joy', 0) + object_emotions.get('excitement', 0)
        negative_emotions = (object_emotions.get('fear', 0) + 
                           object_emotions.get('anger', 0) + 
                           object_emotions.get('sadness', 0))
        
        if positive_emotions > negative_emotions and positive_indicators > negative_indicators:
            return 'positive'
        elif negative_emotions > positive_emotions and negative_indicators > positive_indicators:
            return 'negative'
        else:
            return 'neutral'
    
    def _combine_multimodal_emotions(self, text_emotion: Optional[EmotionResult],
                                   voice_emotion: Optional[EmotionResult],
                                   visual_emotion_cues: Dict[str, Any]) -> EmotionResult:
        """Combine emotions from multiple modalities"""
        
        # Start with text emotion as base
        if text_emotion:
            combined_emotion = text_emotion
        else:
            # Create neutral emotion if no text emotion
            combined_emotion = EmotionResult(
                primary_emotion="neutral",
                confidence=0.5,
                emotion_scores={"neutral": 0.5},
                valence=0.0,
                arousal=0.0,
                intensity=0.5,
                source="multimodal",
                timestamp=time.time()
            )
        
        # Adjust based on visual emotion cues
        if visual_emotion_cues.get('confidence', 0) > 0.5:
            visual_sentiment = visual_emotion_cues['visual_sentiment']
            
            # Adjust valence based on visual sentiment
            if visual_sentiment == 'positive':
                combined_emotion.valence = min(1.0, combined_emotion.valence + 0.3)
            elif visual_sentiment == 'negative':
                combined_emotion.valence = max(-1.0, combined_emotion.valence - 0.3)
            
            # Incorporate visual emotion keywords
            for keyword_data in visual_emotion_cues.get('emotion_keywords_found', []):
                emotion = keyword_data['emotion']
                keyword_confidence = keyword_data['confidence']
                
                if emotion in combined_emotion.emotion_scores:
                    combined_emotion.emotion_scores[emotion] += keyword_confidence * 0.2
                else:
                    combined_emotion.emotion_scores[emotion] = keyword_confidence * 0.2
        
        # Incorporate voice emotion if available
        if voice_emotion:
            # Weight voice emotion influence
            voice_weight = 0.3
            combined_weight = 0.7
            
            # Adjust arousal based on voice emotion
            combined_emotion.arousal = (combined_emotion.arousal * combined_weight + 
                                      voice_emotion.arousal * voice_weight)
            
            # Merge emotion scores
            for emotion, score in voice_emotion.emotion_scores.items():
                if emotion in combined_emotion.emotion_scores:
                    combined_emotion.emotion_scores[emotion] = (
                        combined_emotion.emotion_scores[emotion] * combined_weight + 
                        score * voice_weight
                    )
                else:
                    combined_emotion.emotion_scores[emotion] = score * voice_weight
        
        # Recalculate primary emotion
        if combined_emotion.emotion_scores:
            primary_emotion = max(combined_emotion.emotion_scores, key=combined_emotion.emotion_scores.get)
            combined_emotion.primary_emotion = primary_emotion
            combined_emotion.confidence = combined_emotion.emotion_scores[primary_emotion]
        
        # Update source
        combined_emotion.source = "multimodal"
        combined_emotion.timestamp = time.time()
        
        return combined_emotion
    
    def _calculate_text_emotion_alignment(self, text_emotion: Optional[EmotionResult],
                                        multimodal_result: FusionResult) -> float:
        """Calculate alignment between text emotion and multimodal response"""
        
        if not text_emotion:
            return 0.5
        
        try:
            response_text = multimodal_result.fused_response.lower()
            primary_emotion = text_emotion.primary_emotion
            
            # Check if response acknowledges or reflects the emotion
            emotion_acknowledgments = {
                'joy': ['happy', 'great', 'wonderful', 'excellent', 'fantastic'],
                'sadness': ['sorry', 'sad', 'unfortunate', 'difficult', 'tough'],
                'anger': ['understand', 'frustrating', 'annoying', 'challenging'],
                'fear': ['safe', 'okay', 'calm', 'reassuring', 'don\'t worry'],
                'surprise': ['interesting', 'wow', 'unexpected', 'amazing'],
                'neutral': ['okay', 'fine', 'normal', 'standard']
            }
            
            if primary_emotion in emotion_acknowledgments:
                acknowledgment_words = emotion_acknowledgments[primary_emotion]
                alignment_score = sum(1 for word in acknowledgment_words if word in response_text)
                return min(alignment_score * 0.2, 1.0)
            
            return 0.5
            
        except Exception as e:
            self.logger.error(f"Text emotion alignment calculation failed: {e}")
            return 0.5
    
    def _calculate_overall_emotional_confidence(self, emotion_result: EmotionResult,
                                              visual_emotion_cues: Dict[str, Any],
                                              text_emotion_alignment: float) -> float:
        """Calculate overall confidence in emotional analysis"""
        
        # Base confidence from emotion detection
        base_confidence = emotion_result.confidence
        
        # Visual confirmation bonus
        visual_bonus = visual_emotion_cues.get('confidence', 0) * 0.2
        
        # Text alignment bonus
        alignment_bonus = text_emotion_alignment * 0.1
        
        # Combine confidence scores
        overall_confidence = min(1.0, base_confidence + visual_bonus + alignment_bonus)
        
        return overall_confidence
    
    def _contains_thai_text(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        return any('ก' <= char <= '๛' for char in text)
    
    def _update_integration_stats(self, context: EmotionalMultimodalContext):
        """Update integration statistics"""
        
        self.integration_stats['integrations_performed'] += 1
        
        # Update emotional accuracy based on confidence
        current_accuracy = self.integration_stats['emotional_accuracy']
        total_integrations = self.integration_stats['integrations_performed']
        new_accuracy = context.overall_emotional_confidence
        
        self.integration_stats['emotional_accuracy'] = (
            (current_accuracy * (total_integrations - 1) + new_accuracy) / total_integrations
        )
        
        # Update other stats
        if context.visual_emotion_cues.get('confidence', 0) > 0.5:
            self.integration_stats['visual_emotion_detections'] += 1
        
        if context.text_emotion_alignment > 0.6:
            self.integration_stats['text_emotion_alignments'] += 1
    
    def _create_fallback_context(self, modality_inputs: List[ModalityInput]) -> EmotionalMultimodalContext:
        """Create fallback context when integration fails"""
        
        # Create neutral emotion result
        neutral_emotion = EmotionResult(
            primary_emotion="neutral",
            confidence=0.5,
            emotion_scores={"neutral": 0.5},
            valence=0.0,
            arousal=0.0,
            intensity=0.5,
            source="fallback",
            timestamp=time.time()
        )
        
        # Create minimal multimodal result
        fallback_multimodal = FusionResult(
            fused_response="I understand your input but couldn't fully process the emotional context.",
            confidence=0.3,
            modalities_used=[input_data.modality for input_data in modality_inputs],
            fusion_strategy="fallback",
            reasoning="Integration system encountered an error",
            detailed_analysis={"error": "Fallback mode activated"},
            processing_time=0.0,
            timestamp=datetime.now()
        )
        
        return EmotionalMultimodalContext(
            emotion_result=neutral_emotion,
            multimodal_result=fallback_multimodal,
            emotional_context=None,
            visual_emotion_cues={},
            text_emotion_alignment=0.5,
            overall_emotional_confidence=0.3,
            timestamp=datetime.now()
        )
    
    def enhance_response_with_emotion(self, response: str, 
                                    emotional_context: EmotionalMultimodalContext) -> str:
        """Enhance response based on emotional context"""
        
        try:
            emotion = emotional_context.emotion_result.primary_emotion
            confidence = emotional_context.overall_emotional_confidence
            
            # Only enhance if confidence is high enough
            if confidence < 0.6:
                return response
            
            # Emotional response enhancements
            emotion_enhancements = {
                'joy': {
                    'prefix': "I'm glad to see you're in a good mood! ",
                    'suffix': " 😊",
                    'tone': 'enthusiastic'
                },
                'sadness': {
                    'prefix': "I sense you might be feeling down. ",
                    'suffix': " I'm here to help.",
                    'tone': 'supportive'
                },
                'anger': {
                    'prefix': "I understand you might be frustrated. ",
                    'suffix': " Let me help you with that.",
                    'tone': 'calm'
                },
                'fear': {
                    'prefix': "Don't worry, ",
                    'suffix': " Everything will be okay.",
                    'tone': 'reassuring'
                },
                'surprise': {
                    'prefix': "That's interesting! ",
                    'suffix': "",
                    'tone': 'engaged'
                }
            }
            
            if emotion in emotion_enhancements:
                enhancement = emotion_enhancements[emotion]
                enhanced_response = enhancement['prefix'] + response + enhancement['suffix']
                return enhanced_response
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response enhancement failed: {e}")
            return response
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get integration system statistics"""
        return self.integration_stats.copy()
    
    def get_emotional_recommendations(self, context: EmotionalMultimodalContext) -> List[str]:
        """Get recommendations based on emotional context"""
        
        recommendations = []
        emotion = context.emotion_result.primary_emotion
        confidence = context.overall_emotional_confidence
        
        if confidence < 0.5:
            return ["I need more context to better understand your emotional state."]
        
        emotion_recommendations = {
            'joy': [
                "Great to see you're happy! Would you like to share what's making you feel good?",
                "Your positive energy is wonderful. How can I help you maintain this mood?"
            ],
            'sadness': [
                "I'm here to listen if you'd like to talk about what's bothering you.",
                "Sometimes talking about difficult feelings can help. I'm here for you."
            ],
            'anger': [
                "I understand you're frustrated. Would you like to discuss what's causing this?",
                "Let's work together to address what's making you angry."
            ],
            'fear': [
                "It's okay to feel worried. Would you like to talk about your concerns?",
                "I'm here to help you feel more confident about whatever is worrying you."
            ],
            'neutral': [
                "How are you feeling today? I'm here to help with whatever you need.",
                "Is there anything specific you'd like to talk about or work on?"
            ]
        }
        
        if emotion in emotion_recommendations:
            recommendations = emotion_recommendations[emotion]
        
        return recommendations
    
    def shutdown(self):
        """Shutdown the emotional integration system"""
        self.logger.info("Shutting down multimodal emotional integration")
        
        # Shutdown subsystems
        if self.emotion_system:
            self.emotion_system.reset_emotional_state()
        
        if self.multimodal_system:
            self.multimodal_system.shutdown()
        
        # Clear statistics
        self.integration_stats = {
            'integrations_performed': 0,
            'emotional_accuracy': 0.0,
            'visual_emotion_detections': 0,
            'text_emotion_alignments': 0
        }
        
        self.logger.info("Multimodal emotional integration shutdown complete")