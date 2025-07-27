"""
Visual Question Answering System for JARVIS Voice Assistant
Advanced VQA with multiple model support and context awareness
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
import json
from PIL import Image
import cv2

# Model imports
from transformers import (
    BlipProcessor, BlipForQuestionAnswering,
    ViltProcessor, ViltForQuestionAnswering,
    AutoProcessor, AutoModelForQuestionAnswering,
    CLIPProcessor, CLIPModel
)

# Thai processing
import pythainlp
from pythainlp import word_tokenize


class VisualQASystem:
    """Advanced Visual Question Answering system"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.models = {}
        self.processors = {}
        
        # Context and memory
        self.conversation_history = []
        self.image_cache = {}
        
        # Performance tracking
        self.qa_stats = {
            'total_questions': 0,
            'successful_answers': 0,
            'avg_confidence': 0.0,
            'processing_times': []
        }
        
        # Question analysis patterns
        self.question_patterns = self._initialize_question_patterns()
        
        self.logger.info(f"Visual QA system using device: {self.device}")
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize VQA models"""
        try:
            # Load BLIP VQA model (primary)
            blip_model_name = self.config.get('blip_vqa_model', 'Salesforce/blip-vqa-base')
            self.logger.info(f"Loading BLIP VQA model: {blip_model_name}")
            
            self.processors['blip'] = BlipProcessor.from_pretrained(blip_model_name)
            self.models['blip'] = BlipForQuestionAnswering.from_pretrained(blip_model_name).to(self.device)
            
            # Load CLIP for contextual understanding
            clip_model_name = self.config.get('clip_model', 'openai/clip-vit-base-patch32')
            self.processors['clip'] = CLIPProcessor.from_pretrained(clip_model_name)
            self.models['clip'] = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            
            # Test models
            self._test_models()
            
            self.logger.info("Visual QA models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize VQA models: {e}")
    
    def _test_models(self):
        """Test loaded models"""
        try:
            # Create test image
            test_image = Image.new('RGB', (224, 224), color='blue')
            test_question = "What color is this image?"
            
            # Test BLIP VQA
            blip_inputs = self.processors['blip'](test_image, test_question, return_tensors="pt").to(self.device)
            with torch.no_grad():
                blip_outputs = self.models['blip'].generate(**blip_inputs, max_length=10)
                test_answer = self.processors['blip'].decode(blip_outputs[0], skip_special_tokens=True)
            
            self.logger.info(f"Model test successful - Answer: {test_answer}")
            
        except Exception as e:
            self.logger.warning(f"Model test failed: {e}")
    
    def _initialize_question_patterns(self) -> Dict[str, Any]:
        """Initialize question analysis patterns"""
        return {
            'what_questions': [
                'what', 'ขอ', 'คือ', 'อะไร', 'สิ่งไหน'
            ],
            'where_questions': [
                'where', 'ที่ไหน', 'ตรงไหน', 'สถานที่'
            ],
            'who_questions': [
                'who', 'ใคร', 'คนไหน'
            ],
            'how_questions': [
                'how', 'อย่างไร', 'ยังไง', 'วิธี'
            ],
            'when_questions': [
                'when', 'เมื่อไหร่', 'เวลา'
            ],
            'why_questions': [
                'why', 'ทำไม', 'เพราะ'
            ],
            'count_questions': [
                'how many', 'count', 'number', 'กี่', 'จำนวน'
            ],
            'color_questions': [
                'color', 'สี', 'สีอะไร'
            ],
            'size_questions': [
                'size', 'big', 'small', 'large', 'ขนาด', 'ใหญ่', 'เล็ก'
            ]
        }
    
    def answer_question(self, image_input: Union[str, Image.Image, np.ndarray], 
                       question: str, 
                       context: Optional[str] = None,
                       language: str = "auto") -> Dict[str, Any]:
        """Answer question about image"""
        
        start_time = time.time()
        
        try:
            # Prepare image
            image = self._prepare_image(image_input)
            if image is None:
                raise ValueError("Could not process image")
            
            # Detect language if auto
            if language == "auto":
                language = self._detect_question_language(question)
            
            # Analyze question type
            question_analysis = self._analyze_question(question, language)
            
            # Generate answer using appropriate strategy
            answer_result = self._generate_answer(image, question, question_analysis, context)
            
            # Post-process answer
            final_answer = self._post_process_answer(answer_result, question_analysis, language)
            
            # Create result
            result = {
                'question': question,
                'answer': final_answer['answer'],
                'confidence': final_answer['confidence'],
                'question_type': question_analysis['type'],
                'language': language,
                'processing_time': time.time() - start_time,
                'context_used': context is not None,
                'image_analysis': final_answer.get('image_analysis', {}),
                'reasoning': final_answer.get('reasoning', '')
            }
            
            # Update conversation history
            self._update_conversation_history(question, result)
            
            # Update statistics
            self._update_stats(result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error answering visual question: {e}")
            return self._create_error_result(question, str(e))
    
    def _prepare_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> Optional[Image.Image]:
        """Prepare image for processing"""
        try:
            if isinstance(image_input, str):
                # Check cache first
                if image_input in self.image_cache:
                    return self.image_cache[image_input]
                
                image = Image.open(image_input).convert("RGB")
                # Cache the image
                self.image_cache[image_input] = image
                return image
                
            elif isinstance(image_input, np.ndarray):
                return Image.fromarray(image_input)
            elif isinstance(image_input, Image.Image):
                return image_input.convert("RGB")
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error preparing image: {e}")
            return None
    
    def _detect_question_language(self, question: str) -> str:
        """Detect question language"""
        try:
            # Check for Thai characters
            thai_chars = sum(1 for char in question if 'ก' <= char <= '๛')
            english_chars = sum(1 for char in question if char.isascii() and char.isalpha())
            
            if thai_chars > english_chars:
                return "th"
            else:
                return "en"
                
        except Exception:
            return "en"
    
    def _analyze_question(self, question: str, language: str) -> Dict[str, Any]:
        """Analyze question to determine type and strategy"""
        question_lower = question.lower()
        
        analysis = {
            'type': 'general',
            'category': 'description',
            'keywords': [],
            'requires_counting': False,
            'requires_comparison': False,
            'requires_reasoning': False
        }
        
        # Detect question type
        for q_type, patterns in self.question_patterns.items():
            if any(pattern in question_lower for pattern in patterns):
                analysis['type'] = q_type
                break
        
        # Extract keywords
        if language == "th":
            try:
                words = word_tokenize(question, engine='newmm')
                analysis['keywords'] = [word for word in words if len(word) > 1]
            except:
                analysis['keywords'] = question.split()
        else:
            analysis['keywords'] = [word for word in question_lower.split() if len(word) > 2]
        
        # Detect special requirements
        counting_words = ['how many', 'count', 'number', 'กี่', 'จำนวน']
        comparison_words = ['compare', 'difference', 'similar', 'เปรียบเทียบ', 'ต่าง']
        reasoning_words = ['why', 'because', 'reason', 'ทำไม', 'เพราะ', 'สาเหตุ']
        
        analysis['requires_counting'] = any(word in question_lower for word in counting_words)
        analysis['requires_comparison'] = any(word in question_lower for word in comparison_words)
        analysis['requires_reasoning'] = any(word in question_lower for word in reasoning_words)
        
        return analysis
    
    def _generate_answer(self, image: Image.Image, question: str, 
                        question_analysis: Dict[str, Any], context: Optional[str]) -> Dict[str, Any]:
        """Generate answer using appropriate model and strategy"""
        
        # Primary answer using BLIP VQA
        blip_answer = self._answer_with_blip(image, question)
        
        # Enhance answer based on question type
        enhanced_answer = self._enhance_answer(image, question, blip_answer, question_analysis)
        
        # Add context if available
        if context:
            enhanced_answer = self._incorporate_context(enhanced_answer, context, question_analysis)
        
        return enhanced_answer
    
    def _answer_with_blip(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Generate answer using BLIP VQA model"""
        try:
            inputs = self.processors['blip'](image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.models['blip'].generate(
                    **inputs, 
                    max_length=50,
                    num_beams=5,
                    early_stopping=True
                )
                answer = self.processors['blip'].decode(outputs[0], skip_special_tokens=True)
            
            # Calculate confidence (simplified)
            confidence = min(0.9, len(answer.split()) * 0.1 + 0.5)
            
            return {
                'answer': answer,
                'confidence': confidence,
                'source': 'blip',
                'raw_output': outputs
            }
            
        except Exception as e:
            self.logger.error(f"BLIP VQA failed: {e}")
            return {
                'answer': "I cannot answer that question.",
                'confidence': 0.0,
                'source': 'error',
                'error': str(e)
            }
    
    def _enhance_answer(self, image: Image.Image, question: str, 
                       base_answer: Dict[str, Any], 
                       question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance answer based on question type"""
        
        enhanced = base_answer.copy()
        
        # Add image analysis for specific question types
        if question_analysis['requires_counting']:
            count_info = self._analyze_for_counting(image, question)
            enhanced['count_analysis'] = count_info
            if count_info['count'] is not None:
                enhanced['answer'] = f"{count_info['count']} {count_info['object_type']}"
        
        elif question_analysis['type'] in ['color_questions']:
            color_info = self._analyze_colors(image)
            enhanced['color_analysis'] = color_info
            if color_info['dominant_colors']:
                colors = ', '.join(color_info['dominant_colors'][:3])
                enhanced['answer'] = f"The main colors are {colors}"
        
        elif question_analysis['type'] in ['size_questions']:
            size_info = self._analyze_size_relationships(image)
            enhanced['size_analysis'] = size_info
        
        # Add reasoning for why questions
        if question_analysis['requires_reasoning']:
            reasoning = self._generate_reasoning(image, question, base_answer['answer'])
            enhanced['reasoning'] = reasoning
        
        return enhanced
    
    def _analyze_for_counting(self, image: Image.Image, question: str) -> Dict[str, Any]:
        """Analyze image for counting questions"""
        try:
            # Use CLIP to identify what to count
            object_keywords = ['person', 'people', 'car', 'animal', 'object', 'item']
            
            # Extract potential objects from question
            question_words = question.lower().split()
            potential_objects = [word for word in question_words if word in object_keywords or len(word) > 3]
            
            if not potential_objects:
                potential_objects = ['object']
            
            # Simple counting approach using CLIP similarity
            count_results = {}
            for obj in potential_objects[:3]:  # Limit to 3 objects
                similarity = self._calculate_object_presence(image, obj)
                count_results[obj] = similarity
            
            # Determine most likely object and rough count
            best_object = max(count_results.items(), key=lambda x: x[1])
            
            # Rough count estimation (simplified)
            estimated_count = int(best_object[1] * 5) if best_object[1] > 0.3 else 0
            
            return {
                'object_type': best_object[0],
                'confidence': best_object[1],
                'count': estimated_count,
                'analysis': count_results
            }
            
        except Exception as e:
            self.logger.error(f"Counting analysis failed: {e}")
            return {'object_type': 'unknown', 'confidence': 0.0, 'count': None}
    
    def _analyze_colors(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze dominant colors in image"""
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Reshape for color analysis
            pixels = img_array.reshape(-1, 3)
            
            # Find dominant colors using simple clustering
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=5, random_state=42)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert to color names (simplified)
            color_names = []
            for color in colors:
                name = self._rgb_to_color_name(color)
                color_names.append(name)
            
            return {
                'dominant_colors': color_names,
                'rgb_values': colors.tolist(),
                'color_count': len(colors)
            }
            
        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            return {'dominant_colors': ['unknown'], 'rgb_values': [], 'color_count': 0}
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB to approximate color name"""
        r, g, b = rgb
        
        # Simple color mapping
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 150 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 150 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 150:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        else:
            return "mixed"
    
    def _analyze_size_relationships(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze size relationships in image"""
        try:
            # Simple size analysis based on image dimensions and content
            width, height = image.size
            aspect_ratio = width / height
            
            return {
                'image_size': {'width': width, 'height': height},
                'aspect_ratio': aspect_ratio,
                'orientation': 'landscape' if width > height else 'portrait' if height > width else 'square'
            }
            
        except Exception as e:
            self.logger.error(f"Size analysis failed: {e}")
            return {}
    
    def _calculate_object_presence(self, image: Image.Image, object_name: str) -> float:
        """Calculate presence/similarity of object in image using CLIP"""
        try:
            text_query = f"a photo of {object_name}"
            
            inputs = self.processors['clip'](
                text=[text_query],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                similarity = torch.softmax(outputs.logits_per_image, dim=1)[0, 0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Object presence calculation failed: {e}")
            return 0.0
    
    def _generate_reasoning(self, image: Image.Image, question: str, answer: str) -> str:
        """Generate reasoning for why questions"""
        try:
            # Simple reasoning based on visual cues
            reasoning_parts = []
            
            # Analyze image content for reasoning
            if "because" not in answer.lower():
                reasoning_parts.append(f"Based on visual analysis of the image, {answer.lower()}")
            
            if len(reasoning_parts) == 0:
                reasoning_parts.append("The answer is based on visual elements visible in the image.")
            
            return " ".join(reasoning_parts)
            
        except Exception as e:
            self.logger.error(f"Reasoning generation failed: {e}")
            return "Unable to provide reasoning."
    
    def _incorporate_context(self, answer_result: Dict[str, Any], 
                           context: str, 
                           question_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Incorporate additional context into answer"""
        
        enhanced = answer_result.copy()
        
        # Add context to answer if relevant
        if context and len(context.strip()) > 0:
            enhanced['answer'] = f"{answer_result['answer']} {context}"
            enhanced['confidence'] = min(1.0, enhanced['confidence'] + 0.1)
        
        return enhanced
    
    def _post_process_answer(self, answer_result: Dict[str, Any], 
                           question_analysis: Dict[str, Any], 
                           language: str) -> Dict[str, Any]:
        """Post-process answer for better quality"""
        
        processed = answer_result.copy()
        answer = processed['answer']
        
        # Clean up answer
        answer = answer.strip()
        if answer and not answer[0].isupper():
            answer = answer[0].upper() + answer[1:]
        
        # Add period if missing
        if answer and answer[-1] not in '.!?':
            answer += '.'
        
        # Language-specific post-processing
        if language == "th":
            answer = self._post_process_thai_answer(answer)
        
        processed['answer'] = answer
        return processed
    
    def _post_process_thai_answer(self, answer: str) -> str:
        """Post-process Thai answers"""
        try:
            # Basic Thai text cleanup
            answer = answer.strip()
            # Add Thai-specific formatting if needed
            return answer
            
        except Exception as e:
            self.logger.error(f"Thai post-processing failed: {e}")
            return answer
    
    def _update_conversation_history(self, question: str, result: Dict[str, Any]):
        """Update conversation history"""
        history_entry = {
            'timestamp': time.time(),
            'question': question,
            'answer': result['answer'],
            'confidence': result['confidence'],
            'question_type': result['question_type']
        }
        
        self.conversation_history.append(history_entry)
        
        # Keep only recent history (last 10 interactions)
        if len(self.conversation_history) > 10:
            self.conversation_history = self.conversation_history[-10:]
    
    def _update_stats(self, result: Dict[str, Any]):
        """Update system statistics"""
        self.qa_stats['total_questions'] += 1
        
        if result['confidence'] > 0.5:
            self.qa_stats['successful_answers'] += 1
        
        # Update average confidence
        current_avg = self.qa_stats['avg_confidence']
        total = self.qa_stats['total_questions']
        new_avg = (current_avg * (total - 1) + result['confidence']) / total
        self.qa_stats['avg_confidence'] = new_avg
        
        # Track processing times
        self.qa_stats['processing_times'].append(result['processing_time'])
        if len(self.qa_stats['processing_times']) > 100:
            self.qa_stats['processing_times'] = self.qa_stats['processing_times'][-100:]
    
    def _create_error_result(self, question: str, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'question': question,
            'answer': 'I cannot answer that question due to an error.',
            'confidence': 0.0,
            'question_type': 'unknown',
            'language': 'en',
            'processing_time': 0.0,
            'error': error_msg,
            'success': False
        }
    
    def get_conversation_history(self) -> List[Dict[str, Any]]:
        """Get recent conversation history"""
        return self.conversation_history.copy()
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history.clear()
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        stats = self.qa_stats.copy()
        
        if stats['processing_times']:
            stats['avg_processing_time'] = sum(stats['processing_times']) / len(stats['processing_times'])
            stats['max_processing_time'] = max(stats['processing_times'])
            stats['min_processing_time'] = min(stats['processing_times'])
        
        stats['success_rate'] = (stats['successful_answers'] / stats['total_questions'] 
                                if stats['total_questions'] > 0 else 0)
        
        return stats
    
    def clear_image_cache(self):
        """Clear image cache"""
        self.image_cache.clear()
        
    def shutdown(self):
        """Shutdown VQA system"""
        self.logger.info("Shutting down Visual QA system")
        
        # Clear cache
        self.clear_image_cache()
        self.clear_conversation_history()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("Visual QA system shutdown complete")