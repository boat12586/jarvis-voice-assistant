"""
Multimodal Fusion System for JARVIS Voice Assistant
Advanced fusion of Vision, Text, Voice, and Emotional AI
"""

import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

# Core components
from .multimodal_engine import MultimodalEngine
from .vision_models import VisionModelManager
from .ocr_system import AdvancedOCRSystem
from .visual_qa_system import VisualQASystem
from .video_analysis_system import VideoAnalysisSystem
from .emotion_detection import EmotionDetector
from .sentiment_analysis import SentimentAnalyzer


class ModalityType(Enum):
    """Types of modalities supported"""
    TEXT = "text"
    VOICE = "voice"
    VISION = "vision"
    EMOTION = "emotion"
    CONTEXT = "context"


@dataclass
class ModalityInput:
    """Input data for a specific modality"""
    modality: ModalityType
    data: Any
    confidence: float = 1.0
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class FusionResult:
    """Result of multimodal fusion"""
    fused_response: str
    confidence: float
    modalities_used: List[ModalityType]
    fusion_strategy: str
    reasoning: str
    detailed_analysis: Dict[str, Any]
    processing_time: float
    timestamp: datetime


class MultimodalFusionStrategy(Enum):
    """Fusion strategies for combining modalities"""
    WEIGHTED_AVERAGE = "weighted_average"
    ATTENTION_BASED = "attention_based"
    HIERARCHICAL = "hierarchical"
    CONTEXT_AWARE = "context_aware"
    ADAPTIVE = "adaptive"


class MultimodalFusionSystem:
    """Advanced multimodal fusion system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize subsystems
        self.vision_manager = None
        self.ocr_system = None
        self.vqa_system = None
        self.video_system = None
        self.emotion_detector = None
        self.sentiment_analyzer = None
        
        # Fusion parameters
        self.fusion_weights = config.get('fusion_weights', {
            'text': 0.3,
            'voice': 0.2,
            'vision': 0.3,
            'emotion': 0.1,
            'context': 0.1
        })
        
        # Context management
        self.conversation_context = []
        self.visual_context = {}
        self.emotional_context = {}
        
        # Performance tracking
        self.fusion_stats = {
            'total_fusions': 0,
            'successful_fusions': 0,
            'avg_confidence': 0.0,
            'modality_usage': {modality.value: 0 for modality in ModalityType},
            'fusion_strategies': {}
        }
        
        self._initialize_subsystems()
    
    def _initialize_subsystems(self):
        """Initialize all multimodal subsystems"""
        try:
            self.logger.info("Initializing multimodal fusion subsystems...")
            
            # Initialize vision components
            vision_config = self.config.get('vision', {})
            self.vision_manager = VisionModelManager(vision_config)
            self.vision_manager.load_clip_model()
            self.vision_manager.load_blip_models()
            
            # Initialize OCR system
            ocr_config = self.config.get('ocr', {})
            self.ocr_system = AdvancedOCRSystem(ocr_config)
            
            # Initialize VQA system
            vqa_config = self.config.get('vqa', {})
            self.vqa_system = VisualQASystem(vqa_config)
            
            # Initialize video analysis
            video_config = self.config.get('video', {})
            self.video_system = VideoAnalysisSystem(video_config)
            
            # Initialize emotional components
            try:
                emotion_config = self.config.get('emotion', {})
                self.emotion_detector = EmotionDetector(emotion_config)
                self.sentiment_analyzer = SentimentAnalyzer(emotion_config)
            except Exception as e:
                self.logger.warning(f"Emotional AI components not available: {e}")
            
            self.logger.info("Multimodal fusion system initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize fusion subsystems: {e}")
            raise
    
    def fuse_multimodal_input(self, 
                             inputs: List[ModalityInput],
                             fusion_strategy: MultimodalFusionStrategy = MultimodalFusionStrategy.ADAPTIVE,
                             context: Optional[Dict[str, Any]] = None) -> FusionResult:
        """Fuse multiple modality inputs into unified response"""
        
        start_time = time.time()
        
        try:
            # Validate inputs
            if not inputs:
                raise ValueError("No modality inputs provided")
            
            # Process each modality
            modality_results = {}
            modalities_used = []
            
            for input_data in inputs:
                try:
                    result = self._process_modality(input_data)
                    if result is not None:
                        modality_results[input_data.modality] = result
                        modalities_used.append(input_data.modality)
                except Exception as e:
                    self.logger.warning(f"Failed to process {input_data.modality.value}: {e}")
            
            if not modality_results:
                raise ValueError("No modalities could be processed successfully")
            
            # Apply fusion strategy
            fusion_result = self._apply_fusion_strategy(
                modality_results, fusion_strategy, context
            )
            
            # Create final result
            result = FusionResult(
                fused_response=fusion_result['response'],
                confidence=fusion_result['confidence'],
                modalities_used=modalities_used,
                fusion_strategy=fusion_strategy.value,
                reasoning=fusion_result['reasoning'],
                detailed_analysis=fusion_result['detailed_analysis'],
                processing_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
            # Update context and statistics
            self._update_context(result)
            self._update_fusion_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Multimodal fusion failed: {e}")
            return self._create_error_result(str(e), time.time() - start_time)
    
    def _process_modality(self, input_data: ModalityInput) -> Optional[Dict[str, Any]]:
        """Process individual modality input"""
        
        modality = input_data.modality
        data = input_data.data
        
        try:
            if modality == ModalityType.TEXT:
                return self._process_text_modality(data, input_data.metadata)
            
            elif modality == ModalityType.VISION:
                return self._process_vision_modality(data, input_data.metadata)
            
            elif modality == ModalityType.VOICE:
                return self._process_voice_modality(data, input_data.metadata)
            
            elif modality == ModalityType.EMOTION:
                return self._process_emotion_modality(data, input_data.metadata)
            
            elif modality == ModalityType.CONTEXT:
                return self._process_context_modality(data, input_data.metadata)
            
            else:
                self.logger.warning(f"Unknown modality type: {modality}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error processing {modality.value} modality: {e}")
            return None
    
    def _process_text_modality(self, text_data: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process text modality"""
        
        # Basic text analysis
        analysis = {
            'content': text_data,
            'word_count': len(text_data.split()),
            'character_count': len(text_data),
            'language': self._detect_language(text_data),
            'contains_question': '?' in text_data,
            'contains_visual_keywords': self._contains_visual_keywords(text_data)
        }
        
        # Sentiment analysis if available
        if self.sentiment_analyzer:
            try:
                sentiment = self.sentiment_analyzer.analyze_text(text_data)
                analysis['sentiment'] = sentiment
            except Exception as e:
                self.logger.warning(f"Sentiment analysis failed: {e}")
        
        # Intent detection
        analysis['intent'] = self._detect_intent(text_data)
        
        return {
            'type': 'text',
            'analysis': analysis,
            'confidence': 0.9,  # High confidence for text
            'importance': self._calculate_text_importance(analysis)
        }
    
    def _process_vision_modality(self, vision_data: Union[str, Dict[str, Any]], 
                                metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process vision modality (image or video)"""
        
        if isinstance(vision_data, str):
            # File path provided
            if vision_data.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                return self._process_video_data(vision_data, metadata)
            else:
                return self._process_image_data(vision_data, metadata)
        elif isinstance(vision_data, dict):
            # Pre-processed vision data
            return {
                'type': 'vision',
                'analysis': vision_data,
                'confidence': vision_data.get('confidence', 0.8),
                'importance': self._calculate_vision_importance(vision_data)
            }
        else:
            raise ValueError("Invalid vision data format")
    
    def _process_image_data(self, image_path: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process image data"""
        
        # Generate caption
        caption = self.vision_manager.generate_image_caption(image_path)
        
        # Perform OCR
        ocr_result = self.ocr_system.extract_text(image_path)
        
        # Object detection
        objects = self.vision_manager.detect_objects_yolo(image_path)
        
        # Get embeddings for similarity
        embeddings = self.vision_manager.get_image_embeddings(image_path)
        
        analysis = {
            'image_path': image_path,
            'caption': caption,
            'ocr_results': ocr_result,
            'detected_objects': objects,
            'has_text': bool(ocr_result.get('full_text', '')),
            'object_count': len(objects),
            'embeddings': embeddings.tolist() if len(embeddings) > 0 else []
        }
        
        return {
            'type': 'image',
            'analysis': analysis,
            'confidence': 0.85,
            'importance': self._calculate_vision_importance(analysis)
        }
    
    def _process_video_data(self, video_path: str, metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process video data"""
        
        # Analyze video
        video_analysis = self.video_system.analyze_video(
            video_path, 
            analysis_type="comprehensive"
        )
        
        analysis = {
            'video_path': video_path,
            'video_analysis': video_analysis,
            'scene_count': len(video_analysis.get('scenes', [])),
            'duration': video_analysis.get('video_metadata', {}).get('duration', 0),
            'has_scenes': len(video_analysis.get('scenes', [])) > 1
        }
        
        return {
            'type': 'video',
            'analysis': analysis,
            'confidence': 0.8,
            'importance': self._calculate_vision_importance(analysis)
        }
    
    def _process_voice_modality(self, voice_data: Union[str, Dict[str, Any]], 
                               metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process voice modality"""
        
        if isinstance(voice_data, str):
            # Transcribed text
            analysis = {
                'transcription': voice_data,
                'source': 'voice',
                'confidence': metadata.get('transcription_confidence', 0.8) if metadata else 0.8
            }
        elif isinstance(voice_data, dict):
            # Voice analysis data
            analysis = voice_data
        else:
            raise ValueError("Invalid voice data format")
        
        # Voice-specific analysis
        if self.emotion_detector and 'audio_features' in analysis:
            try:
                emotion = self.emotion_detector.analyze_voice(analysis['audio_features'])
                analysis['emotional_state'] = emotion
            except Exception as e:
                self.logger.warning(f"Voice emotion analysis failed: {e}")
        
        return {
            'type': 'voice',
            'analysis': analysis,
            'confidence': analysis.get('confidence', 0.7),
            'importance': self._calculate_voice_importance(analysis)
        }
    
    def _process_emotion_modality(self, emotion_data: Dict[str, Any], 
                                 metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process emotion modality"""
        
        analysis = {
            'emotional_state': emotion_data.get('emotion', 'neutral'),
            'confidence': emotion_data.get('confidence', 0.5),
            'valence': emotion_data.get('valence', 0.0),
            'arousal': emotion_data.get('arousal', 0.0),
            'context': emotion_data.get('context', '')
        }
        
        return {
            'type': 'emotion',
            'analysis': analysis,
            'confidence': analysis['confidence'],
            'importance': self._calculate_emotion_importance(analysis)
        }
    
    def _process_context_modality(self, context_data: Dict[str, Any], 
                                 metadata: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Process context modality"""
        
        analysis = {
            'conversation_history': context_data.get('conversation_history', []),
            'user_preferences': context_data.get('user_preferences', {}),
            'session_context': context_data.get('session_context', {}),
            'environmental_context': context_data.get('environmental_context', {})
        }
        
        return {
            'type': 'context',
            'analysis': analysis,
            'confidence': 0.6,  # Context has moderate confidence
            'importance': self._calculate_context_importance(analysis)
        }
    
    def _apply_fusion_strategy(self, modality_results: Dict[ModalityType, Dict[str, Any]], 
                              strategy: MultimodalFusionStrategy,
                              context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply fusion strategy to combine modality results"""
        
        if strategy == MultimodalFusionStrategy.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(modality_results)
        
        elif strategy == MultimodalFusionStrategy.ATTENTION_BASED:
            return self._attention_based_fusion(modality_results)
        
        elif strategy == MultimodalFusionStrategy.HIERARCHICAL:
            return self._hierarchical_fusion(modality_results)
        
        elif strategy == MultimodalFusionStrategy.CONTEXT_AWARE:
            return self._context_aware_fusion(modality_results, context)
        
        elif strategy == MultimodalFusionStrategy.ADAPTIVE:
            return self._adaptive_fusion(modality_results, context)
        
        else:
            # Default to weighted average
            return self._weighted_average_fusion(modality_results)
    
    def _weighted_average_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Simple weighted average fusion"""
        
        response_parts = []
        total_confidence = 0
        total_weight = 0
        detailed_analysis = {}
        
        for modality, result in modality_results.items():
            weight = self.fusion_weights.get(modality.value, 0.1)
            confidence = result['confidence']
            importance = result['importance']
            
            adjusted_weight = weight * importance
            total_weight += adjusted_weight
            total_confidence += confidence * adjusted_weight
            
            # Add content to response
            if modality == ModalityType.TEXT:
                response_parts.append(result['analysis']['content'])
            elif modality == ModalityType.VISION:
                if 'caption' in result['analysis']:
                    response_parts.append(f"I can see: {result['analysis']['caption']}")
                if result['analysis'].get('has_text'):
                    ocr_text = result['analysis']['ocr_results'].get('full_text', '')
                    if ocr_text:
                        response_parts.append(f"Text in image: {ocr_text}")
            elif modality == ModalityType.VOICE:
                if 'transcription' in result['analysis']:
                    response_parts.append(result['analysis']['transcription'])
            
            detailed_analysis[modality.value] = result['analysis']
        
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0
        
        return {
            'response': ' '.join(response_parts),
            'confidence': avg_confidence,
            'reasoning': 'Weighted average fusion of all modalities',
            'detailed_analysis': detailed_analysis
        }
    
    def _attention_based_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Attention-based fusion focusing on most relevant modalities"""
        
        # Calculate attention weights based on importance and confidence
        attention_weights = {}
        for modality, result in modality_results.items():
            importance = result['importance']
            confidence = result['confidence']
            attention_weights[modality] = importance * confidence
        
        # Normalize attention weights
        total_attention = sum(attention_weights.values())
        if total_attention > 0:
            for modality in attention_weights:
                attention_weights[modality] /= total_attention
        
        # Focus on top modalities
        sorted_modalities = sorted(attention_weights.items(), key=lambda x: x[1], reverse=True)
        top_modalities = sorted_modalities[:3]  # Focus on top 3
        
        response_parts = []
        confidence_sum = 0
        detailed_analysis = {}
        
        for modality, weight in top_modalities:
            result = modality_results[modality]
            confidence_sum += result['confidence'] * weight
            
            # Generate response based on modality
            if modality == ModalityType.VISION:
                analysis = result['analysis']
                if 'caption' in analysis:
                    response_parts.append(f"Visual analysis shows: {analysis['caption']}")
                if analysis.get('has_text'):
                    ocr_text = analysis['ocr_results'].get('full_text', '')
                    if ocr_text:
                        response_parts.append(f"I can read: {ocr_text}")
            elif modality == ModalityType.TEXT:
                response_parts.append(result['analysis']['content'])
            
            detailed_analysis[modality.value] = result['analysis']
        
        return {
            'response': ' '.join(response_parts),
            'confidence': confidence_sum,
            'reasoning': f'Attention-based fusion focusing on {[m.value for m, _ in top_modalities]}',
            'detailed_analysis': detailed_analysis
        }
    
    def _hierarchical_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Hierarchical fusion with modality priority"""
        
        # Define hierarchy: Vision > Text > Voice > Emotion > Context
        hierarchy = [
            ModalityType.VISION,
            ModalityType.TEXT, 
            ModalityType.VOICE,
            ModalityType.EMOTION,
            ModalityType.CONTEXT
        ]
        
        primary_modality = None
        for modality in hierarchy:
            if modality in modality_results:
                primary_modality = modality
                break
        
        if not primary_modality:
            return self._weighted_average_fusion(modality_results)
        
        primary_result = modality_results[primary_modality]
        response_parts = []
        detailed_analysis = {}
        
        # Start with primary modality
        if primary_modality == ModalityType.VISION:
            analysis = primary_result['analysis']
            if 'caption' in analysis:
                response_parts.append(analysis['caption'])
            if analysis.get('has_text'):
                ocr_text = analysis['ocr_results'].get('full_text', '')
                if ocr_text:
                    response_parts.append(f"Text content: {ocr_text}")
        elif primary_modality == ModalityType.TEXT:
            response_parts.append(primary_result['analysis']['content'])
        
        # Add supporting information from other modalities
        for modality, result in modality_results.items():
            if modality != primary_modality:
                if modality == ModalityType.EMOTION and 'emotional_state' in result['analysis']:
                    emotional_state = result['analysis']['emotional_state']
                    if emotional_state != 'neutral':
                        response_parts.append(f"(Emotional context: {emotional_state})")
            
            detailed_analysis[modality.value] = result['analysis']
        
        return {
            'response': ' '.join(response_parts),
            'confidence': primary_result['confidence'],
            'reasoning': f'Hierarchical fusion with {primary_modality.value} as primary',
            'detailed_analysis': detailed_analysis
        }
    
    def _context_aware_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]], 
                             context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Context-aware fusion considering conversation and user context"""
        
        # Analyze context to determine optimal fusion
        if context and 'previous_question' in context:
            prev_question = context['previous_question'].lower()
            
            # If previous question was about visual content, prioritize vision
            if any(word in prev_question for word in ['see', 'look', 'image', 'picture', 'video']):
                if ModalityType.VISION in modality_results:
                    return self._vision_prioritized_fusion(modality_results)
            
            # If previous question was about text, prioritize text/OCR
            elif any(word in prev_question for word in ['read', 'text', 'write', 'written']):
                return self._text_prioritized_fusion(modality_results)
        
        # Default to adaptive fusion
        return self._adaptive_fusion(modality_results, context)
    
    def _adaptive_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]], 
                        context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Adaptive fusion that chooses strategy based on input characteristics"""
        
        # Analyze input characteristics
        has_high_confidence_vision = (
            ModalityType.VISION in modality_results and 
            modality_results[ModalityType.VISION]['confidence'] > 0.8
        )
        
        has_text_query = (
            ModalityType.TEXT in modality_results and
            modality_results[ModalityType.TEXT]['analysis'].get('contains_question', False)
        )
        
        has_visual_keywords = (
            ModalityType.TEXT in modality_results and
            modality_results[ModalityType.TEXT]['analysis'].get('contains_visual_keywords', False)
        )
        
        # Choose strategy based on characteristics
        if has_high_confidence_vision and (has_visual_keywords or has_text_query):
            # Visual Q&A scenario
            return self._visual_qa_fusion(modality_results)
        elif len(modality_results) == 1:
            # Single modality - simple processing
            return self._single_modality_fusion(modality_results)
        elif has_text_query:
            # Question-focused fusion
            return self._question_focused_fusion(modality_results)
        else:
            # General multimodal fusion
            return self._attention_based_fusion(modality_results)
    
    def _visual_qa_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Specialized fusion for visual Q&A scenarios"""
        
        if ModalityType.TEXT not in modality_results or ModalityType.VISION not in modality_results:
            return self._weighted_average_fusion(modality_results)
        
        text_result = modality_results[ModalityType.TEXT]
        vision_result = modality_results[ModalityType.VISION]
        
        question = text_result['analysis']['content']
        
        # Use VQA system if available
        if self.vqa_system and 'image_path' in vision_result['analysis']:
            try:
                vqa_result = self.vqa_system.answer_question(
                    vision_result['analysis']['image_path'],
                    question
                )
                
                return {
                    'response': vqa_result['answer'],
                    'confidence': vqa_result['confidence'],
                    'reasoning': 'Visual Q&A specialized fusion',
                    'detailed_analysis': {
                        'vqa_result': vqa_result,
                        'text': text_result['analysis'],
                        'vision': vision_result['analysis']
                    }
                }
            except Exception as e:
                self.logger.warning(f"VQA fusion failed: {e}")
        
        # Fallback to manual fusion
        response_parts = []
        
        # Add vision information
        if 'caption' in vision_result['analysis']:
            response_parts.append(f"I can see: {vision_result['analysis']['caption']}")
        
        # Add OCR if relevant
        if vision_result['analysis'].get('has_text'):
            ocr_text = vision_result['analysis']['ocr_results'].get('full_text', '')
            if ocr_text and ('text' in question.lower() or 'read' in question.lower()):
                response_parts.append(f"Text content: {ocr_text}")
        
        return {
            'response': ' '.join(response_parts),
            'confidence': (text_result['confidence'] + vision_result['confidence']) / 2,
            'reasoning': 'Manual visual Q&A fusion',
            'detailed_analysis': {
                'text': text_result['analysis'],
                'vision': vision_result['analysis']
            }
        }
    
    def _single_modality_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Handle single modality input"""
        
        modality, result = next(iter(modality_results.items()))
        
        if modality == ModalityType.VISION:
            analysis = result['analysis']
            response_parts = []
            
            if 'caption' in analysis:
                response_parts.append(f"I can see: {analysis['caption']}")
            
            if analysis.get('has_text'):
                ocr_text = analysis['ocr_results'].get('full_text', '')
                if ocr_text:
                    response_parts.append(f"Text content: {ocr_text}")
            
            if analysis.get('detected_objects'):
                objects = [obj['class'] for obj in analysis['detected_objects'][:3]]
                if objects:
                    response_parts.append(f"Objects detected: {', '.join(objects)}")
            
            response = ' '.join(response_parts) if response_parts else "I can analyze this image."
            
        elif modality == ModalityType.TEXT:
            response = result['analysis']['content']
        
        else:
            response = f"Processed {modality.value} input successfully."
        
        return {
            'response': response,
            'confidence': result['confidence'],
            'reasoning': f'Single {modality.value} modality processing',
            'detailed_analysis': {modality.value: result['analysis']}
        }
    
    def _question_focused_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Fusion focused on answering questions"""
        
        # Prioritize modalities that can answer the question
        return self._attention_based_fusion(modality_results)
    
    def _vision_prioritized_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Fusion prioritizing vision modality"""
        
        if ModalityType.VISION not in modality_results:
            return self._weighted_average_fusion(modality_results)
        
        vision_result = modality_results[ModalityType.VISION]
        analysis = vision_result['analysis']
        
        response_parts = []
        
        if 'caption' in analysis:
            response_parts.append(analysis['caption'])
        
        if analysis.get('has_text'):
            ocr_text = analysis['ocr_results'].get('full_text', '')
            if ocr_text:
                response_parts.append(f"Text visible: {ocr_text}")
        
        # Add supporting information
        for modality, result in modality_results.items():
            if modality != ModalityType.VISION and modality == ModalityType.TEXT:
                if result['analysis'].get('contains_question'):
                    # This is likely a follow-up question about the image
                    question = result['analysis']['content']
                    response_parts.insert(0, f"Regarding your question '{question}': ")
        
        return {
            'response': ' '.join(response_parts),
            'confidence': vision_result['confidence'],
            'reasoning': 'Vision-prioritized fusion',
            'detailed_analysis': {modality.value: result['analysis'] for modality, result in modality_results.items()}
        }
    
    def _text_prioritized_fusion(self, modality_results: Dict[ModalityType, Dict[str, Any]]) -> Dict[str, Any]:
        """Fusion prioritizing text/OCR content"""
        
        response_parts = []
        highest_confidence = 0
        detailed_analysis = {}
        
        # Prioritize OCR text from vision
        if ModalityType.VISION in modality_results:
            vision_result = modality_results[ModalityType.VISION]
            if vision_result['analysis'].get('has_text'):
                ocr_text = vision_result['analysis']['ocr_results'].get('full_text', '')
                if ocr_text:
                    response_parts.append(f"Text content: {ocr_text}")
                    highest_confidence = max(highest_confidence, vision_result['confidence'])
            detailed_analysis['vision'] = vision_result['analysis']
        
        # Add direct text input
        if ModalityType.TEXT in modality_results:
            text_result = modality_results[ModalityType.TEXT]
            response_parts.append(text_result['analysis']['content'])
            highest_confidence = max(highest_confidence, text_result['confidence'])
            detailed_analysis['text'] = text_result['analysis']
        
        return {
            'response': ' '.join(response_parts),
            'confidence': highest_confidence,
            'reasoning': 'Text-prioritized fusion',
            'detailed_analysis': detailed_analysis
        }
    
    # Helper methods for calculating importance scores
    def _calculate_text_importance(self, analysis: Dict[str, Any]) -> float:
        """Calculate importance score for text modality"""
        importance = 0.5  # Base importance
        
        if analysis.get('contains_question'):
            importance += 0.3
        if analysis.get('contains_visual_keywords'):
            importance += 0.2
        if analysis['word_count'] > 10:
            importance += 0.1
        
        return min(1.0, importance)
    
    def _calculate_vision_importance(self, analysis: Dict[str, Any]) -> float:
        """Calculate importance score for vision modality"""
        importance = 0.6  # Base importance
        
        if analysis.get('has_text'):
            importance += 0.2
        if analysis.get('object_count', 0) > 0:
            importance += 0.1
        if 'caption' in analysis and len(analysis['caption']) > 20:
            importance += 0.1
        
        return min(1.0, importance)
    
    def _calculate_voice_importance(self, analysis: Dict[str, Any]) -> float:
        """Calculate importance score for voice modality"""
        importance = 0.4  # Base importance
        
        if analysis.get('confidence', 0) > 0.8:
            importance += 0.2
        if 'emotional_state' in analysis:
            importance += 0.1
        
        return min(1.0, importance)
    
    def _calculate_emotion_importance(self, analysis: Dict[str, Any]) -> float:
        """Calculate importance score for emotion modality"""
        importance = 0.3  # Base importance
        
        if analysis['confidence'] > 0.7:
            importance += 0.2
        if analysis['emotional_state'] != 'neutral':
            importance += 0.2
        
        return min(1.0, importance)
    
    def _calculate_context_importance(self, analysis: Dict[str, Any]) -> float:
        """Calculate importance score for context modality"""
        importance = 0.2  # Base importance
        
        if analysis.get('conversation_history'):
            importance += 0.2
        if analysis.get('user_preferences'):
            importance += 0.1
        
        return min(1.0, importance)
    
    # Utility methods
    def _detect_language(self, text: str) -> str:
        """Detect language of text"""
        thai_chars = sum(1 for char in text if 'ก' <= char <= '๛')
        english_chars = sum(1 for char in text if char.isascii() and char.isalpha())
        
        return "th" if thai_chars > english_chars else "en"
    
    def _contains_visual_keywords(self, text: str) -> bool:
        """Check if text contains visual keywords"""
        visual_keywords = ['see', 'look', 'show', 'image', 'picture', 'video', 'visual', 'watch', 'view']
        return any(keyword in text.lower() for keyword in visual_keywords)
    
    def _detect_intent(self, text: str) -> str:
        """Detect intent from text"""
        text_lower = text.lower()
        
        if '?' in text or any(word in text_lower for word in ['what', 'how', 'why', 'when', 'where', 'who']):
            return 'question'
        elif any(word in text_lower for word in ['show', 'display', 'see']):
            return 'request_visual'
        elif any(word in text_lower for word in ['read', 'text', 'written']):
            return 'request_text'
        else:
            return 'general'
    
    def _update_context(self, result: FusionResult):
        """Update conversation and context"""
        
        # Update conversation context
        context_entry = {
            'timestamp': result.timestamp,
            'modalities': [m.value for m in result.modalities_used],
            'response': result.fused_response,
            'confidence': result.confidence
        }
        
        self.conversation_context.append(context_entry)
        
        # Keep only recent context
        if len(self.conversation_context) > 10:
            self.conversation_context = self.conversation_context[-10:]
    
    def _update_fusion_stats(self, result: FusionResult):
        """Update fusion statistics"""
        
        self.fusion_stats['total_fusions'] += 1
        
        if result.confidence > 0.5:
            self.fusion_stats['successful_fusions'] += 1
        
        # Update average confidence
        current_avg = self.fusion_stats['avg_confidence']
        total = self.fusion_stats['total_fusions']
        new_avg = (current_avg * (total - 1) + result.confidence) / total
        self.fusion_stats['avg_confidence'] = new_avg
        
        # Update modality usage
        for modality in result.modalities_used:
            self.fusion_stats['modality_usage'][modality.value] += 1
        
        # Update strategy usage
        strategy = result.fusion_strategy
        self.fusion_stats['fusion_strategies'][strategy] = self.fusion_stats['fusion_strategies'].get(strategy, 0) + 1
    
    def _create_error_result(self, error_msg: str, processing_time: float) -> FusionResult:
        """Create error result"""
        return FusionResult(
            fused_response=f"I encountered an error while processing: {error_msg}",
            confidence=0.0,
            modalities_used=[],
            fusion_strategy="error",
            reasoning=f"Error occurred: {error_msg}",
            detailed_analysis={'error': error_msg},
            processing_time=processing_time,
            timestamp=datetime.now()
        )
    
    # Public interface methods
    def get_fusion_stats(self) -> Dict[str, Any]:
        """Get fusion system statistics"""
        return self.fusion_stats.copy()
    
    def get_conversation_context(self) -> List[Dict[str, Any]]:
        """Get recent conversation context"""
        return self.conversation_context.copy()
    
    def clear_context(self):
        """Clear conversation context"""
        self.conversation_context.clear()
        self.visual_context.clear()
        self.emotional_context.clear()
    
    def update_fusion_weights(self, new_weights: Dict[str, float]):
        """Update fusion weights"""
        self.fusion_weights.update(new_weights)
    
    def shutdown(self):
        """Shutdown fusion system"""
        self.logger.info("Shutting down multimodal fusion system")
        
        # Shutdown subsystems
        if self.vision_manager:
            self.vision_manager.clear_all_models()
        
        if self.vqa_system:
            self.vqa_system.shutdown()
        
        if self.video_system:
            self.video_system.shutdown()
        
        # Clear contexts
        self.clear_context()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Multimodal fusion system shutdown complete")