"""
Feature Manager for Jarvis Voice Assistant
Manages specialized assistant features
"""

import logging
from typing import Dict, Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from .news_system import NewsSystem
from .translation_system import TranslationSystem
from .learning_system import LearningSystem
from .deep_question_system import DeepQuestionSystem
from .image_generation_system import ImageGenerationSystem


class FeatureManager(QObject):
    """Feature management system"""
    
    # Signals
    feature_executed = pyqtSignal(str, dict)  # feature_name, result
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Feature systems
        self.news_system: Optional[NewsSystem] = None
        self.translation_system: Optional[TranslationSystem] = None
        self.learning_system: Optional[LearningSystem] = None
        self.deep_question_system: Optional[DeepQuestionSystem] = None
        self.image_generation_system: Optional[ImageGenerationSystem] = None
        
        # Initialize features
        self._initialize_features()
        
        self.logger.info("Feature manager initialized")
    
    def _initialize_features(self):
        """Initialize feature systems"""
        try:
            # Initialize news system
            news_config = self.config.get("news", {})
            self.news_system = NewsSystem(news_config)
            
            # Initialize translation system
            translation_config = self.config.get("translation", {})
            self.translation_system = TranslationSystem(translation_config)
            
            # Initialize learning system
            learning_config = self.config.get("learning", {})
            self.learning_system = LearningSystem(learning_config)
            
            # Initialize deep question system
            deep_question_config = self.config.get("deep_question", {})
            self.deep_question_system = DeepQuestionSystem(deep_question_config)
            
            # Initialize image generation system
            image_generation_config = self.config.get("image_generation", {})
            self.image_generation_system = ImageGenerationSystem(image_generation_config)
            
            # Connect signals
            self._connect_signals()
            
            self.logger.info("Feature systems initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize features: {e}")
            self.error_occurred.emit(f"Feature initialization failed: {e}")
    
    def _connect_signals(self):
        """Connect signals from feature systems"""
        if self.news_system:
            self.news_system.news_ready.connect(self._on_news_ready)
            self.news_system.error_occurred.connect(self._on_news_error)
        
        if self.translation_system:
            self.translation_system.translation_ready.connect(self._on_translation_ready)
            self.translation_system.error_occurred.connect(self._on_translation_error)
        
        if self.learning_system:
            self.learning_system.lesson_ready.connect(self._on_lesson_ready)
            self.learning_system.quiz_ready.connect(self._on_quiz_ready)
            self.learning_system.error_occurred.connect(self._on_learning_error)
        
        if self.deep_question_system:
            self.deep_question_system.answer_ready.connect(self._on_deep_answer_ready)
            self.deep_question_system.error_occurred.connect(self._on_deep_question_error)
        
        if self.image_generation_system:
            self.image_generation_system.generation_complete.connect(self._on_image_generation_complete)
            self.image_generation_system.error_occurred.connect(self._on_image_generation_error)
    
    def execute_feature(self, feature_name: str, parameters: Dict[str, Any]):
        """Execute a specific feature"""
        self.logger.info(f"Executing feature: {feature_name} with params: {parameters}")
        
        try:
            if feature_name == "news":
                self._execute_news_feature(parameters)
            elif feature_name == "translate":
                self._execute_translate_feature(parameters)
            elif feature_name == "learn":
                self._execute_learn_feature(parameters)
            elif feature_name == "deep_question":
                self._execute_deep_question_feature(parameters)
            elif feature_name == "image_generation":
                self._execute_image_generation_feature(parameters)
            else:
                self.logger.warning(f"Unknown feature: {feature_name}")
                self.error_occurred.emit(f"Unknown feature: {feature_name}")
                
        except Exception as e:
            self.logger.error(f"Feature execution error: {e}")
            self.error_occurred.emit(f"Feature execution failed: {e}")
    
    def _execute_news_feature(self, parameters: Dict[str, Any]):
        """Execute news feature"""
        if not self.news_system:
            self.error_occurred.emit("News system not available")
            return
        
        try:
            language = parameters.get("language", "en")
            limit = parameters.get("limit", 5)
            
            # Get news text
            news_text = self.news_system.get_news_text(language, limit)
            
            # Get summaries for additional data
            summaries = self.news_system.get_news_summary(language, limit)
            
            result = {
                "news_text": news_text,
                "summaries": summaries,
                "language": language,
                "count": len(summaries)
            }
            
            self.feature_executed.emit("news", result)
            
        except Exception as e:
            self.logger.error(f"News feature error: {e}")
            self.error_occurred.emit(f"News feature failed: {e}")
    
    def _execute_translate_feature(self, parameters: Dict[str, Any]):
        """Execute translation feature"""
        if not self.translation_system:
            self.error_occurred.emit("Translation system not available")
            return
        
        try:
            text = parameters.get("text", "")
            source_lang = parameters.get("source_lang", "auto")
            target_lang = parameters.get("target_lang", "th")
            
            if not text:
                # Return system info if no text provided
                result = {
                    "message": "Translation system ready. Please provide text to translate.",
                    "supported_languages": self.translation_system.get_supported_languages(),
                    "example": "Try saying: 'Translate Hello to Thai'"
                }
                self.feature_executed.emit("translate", result)
                return
            
            # Perform translation
            translation_result = self.translation_system.translate_text(text, source_lang, target_lang)
            
            if translation_result:
                self.feature_executed.emit("translate", translation_result)
            else:
                self.error_occurred.emit("Translation failed")
                
        except Exception as e:
            self.logger.error(f"Translation feature error: {e}")
            self.error_occurred.emit(f"Translation feature failed: {e}")
    
    def _execute_learn_feature(self, parameters: Dict[str, Any]):
        """Execute language learning feature"""
        if not self.learning_system:
            self.error_occurred.emit("Learning system not available")
            return
        
        try:
            language = parameters.get("language", "thai")
            difficulty = parameters.get("difficulty", "beginner")
            category = parameters.get("category", "vocabulary")
            
            # Start a lesson
            lesson_result = self.learning_system.start_lesson(language, difficulty, category)
            
            if lesson_result:
                self.feature_executed.emit("learn", lesson_result)
            else:
                # Return available lessons info
                available_lessons = self.learning_system.get_available_lessons(language, difficulty, category)
                result = {
                    "message": "Learning system ready. Available lessons:",
                    "available_lessons": available_lessons[:5],  # Show first 5
                    "stats": self.learning_system.get_learning_stats()
                }
                self.feature_executed.emit("learn", result)
                
        except Exception as e:
            self.logger.error(f"Learning feature error: {e}")
            self.error_occurred.emit(f"Learning feature failed: {e}")
    
    def _execute_deep_question_feature(self, parameters: Dict[str, Any]):
        """Execute deep question feature"""
        if not self.deep_question_system:
            self.error_occurred.emit("Deep question system not available")
            return
        
        try:
            question = parameters.get("question", "")
            
            if not question:
                # Return system info if no question provided
                suggestions = self.deep_question_system.get_question_suggestions()
                result = {
                    "message": "Deep question system ready. You can ask complex questions about philosophy, science, technology, or society.",
                    "suggestions": suggestions[:5],  # Show first 5 suggestions
                    "stats": self.deep_question_system.get_system_stats()
                }
                self.feature_executed.emit("deep_question", result)
                return
            
            # Process the question
            answer_result = self.deep_question_system.process_question(question)
            
            if answer_result:
                self.feature_executed.emit("deep_question", answer_result)
            else:
                self.error_occurred.emit("Deep question processing failed")
                
        except Exception as e:
            self.logger.error(f"Deep question feature error: {e}")
            self.error_occurred.emit(f"Deep question feature failed: {e}")
    
    def _execute_image_generation_feature(self, parameters: Dict[str, Any]):
        """Execute image generation feature"""
        if not self.image_generation_system:
            self.error_occurred.emit("Image generation system not available")
            return
        
        try:
            prompt = parameters.get("prompt", "")
            style = parameters.get("style", "realistic")
            dimensions = parameters.get("dimensions", None)
            quality = parameters.get("quality", "medium")
            
            if not prompt:
                # Return system info if no prompt provided
                result = {
                    "message": "Image generation system ready. Please provide a description of what you'd like to create.",
                    "supported_styles": self.image_generation_system.get_supported_styles(),
                    "stats": self.image_generation_system.get_generation_stats(),
                    "example": "Try saying: 'Generate a beautiful sunset over mountains'"
                }
                self.feature_executed.emit("image_generation", result)
                return
            
            # Start image generation
            request_id = self.image_generation_system.generate_image(prompt, style, dimensions, quality)
            
            if request_id:
                result = {
                    "status": "Image generation started",
                    "request_id": request_id,
                    "prompt": prompt,
                    "style": style,
                    "message": "Image generation in progress. This may take a moment."
                }
                self.feature_executed.emit("image_generation", result)
            else:
                self.error_occurred.emit("Failed to start image generation")
                
        except Exception as e:
            self.logger.error(f"Image generation feature error: {e}")
            self.error_occurred.emit(f"Image generation feature failed: {e}")
    
    def _on_news_ready(self, summaries: list):
        """Handle news ready signal"""
        self.logger.info(f"News ready with {len(summaries)} summaries")
        # News is automatically available through get_news_text
    
    def _on_news_error(self, error_msg: str):
        """Handle news error"""
        self.logger.error(f"News system error: {error_msg}")
        # Don't propagate news errors to main system
    
    def _on_translation_ready(self, result: dict):
        """Handle translation ready signal"""
        self.logger.info(f"Translation ready: {result.get('translated_text', '')[:50]}...")
        # Translation results are handled in the execution method
    
    def _on_translation_error(self, error_msg: str):
        """Handle translation error"""
        self.logger.error(f"Translation system error: {error_msg}")
    
    def _on_lesson_ready(self, lesson: dict):
        """Handle lesson ready signal"""
        self.logger.info(f"Lesson ready: {lesson.get('title', 'Unknown')}")
        # Lesson results are handled in the execution method
    
    def _on_quiz_ready(self, quiz: dict):
        """Handle quiz ready signal"""
        self.logger.info(f"Quiz ready: {quiz.get('title', 'Unknown')}")
        # Quiz results could be used for interactive learning
    
    def _on_learning_error(self, error_msg: str):
        """Handle learning error"""
        self.logger.error(f"Learning system error: {error_msg}")
    
    def _on_deep_answer_ready(self, answer: dict):
        """Handle deep answer ready signal"""
        self.logger.info(f"Deep answer ready: {answer.get('question', '')[:50]}...")
        # Deep answers are handled in the execution method
    
    def _on_deep_question_error(self, error_msg: str):
        """Handle deep question error"""
        self.logger.error(f"Deep question system error: {error_msg}")
    
    def _on_image_generation_complete(self, result: dict):
        """Handle image generation complete signal"""
        self.logger.info(f"Image generation complete: {result.get('image_id', 'Unknown')}")
        # Image generation results could be displayed in UI
    
    def _on_image_generation_error(self, error_msg: str):
        """Handle image generation error"""
        self.logger.error(f"Image generation system error: {error_msg}")
    
    def get_feature_info(self) -> Dict[str, Any]:
        """Get feature system information"""
        info = {
            "available_features": [
                "news", "translate", "learn", "deep_question", "image_generation"
            ],
            "news_system": None
        }
        
        if self.news_system:
            info["news_system"] = self.news_system.get_stats()
        
        if self.translation_system:
            info["translation_system"] = self.translation_system.get_stats()
        
        if self.learning_system:
            info["learning_system"] = self.learning_system.get_learning_stats()
        
        if self.deep_question_system:
            info["deep_question_system"] = self.deep_question_system.get_system_stats()
        
        if self.image_generation_system:
            info["image_generation_system"] = self.image_generation_system.get_generation_stats()
        
        return info
    
    def shutdown(self):
        """Shutdown feature manager"""
        self.logger.info("Shutting down feature manager")
        
        if self.news_system:
            self.news_system.shutdown()
        
        if self.translation_system:
            self.translation_system.shutdown()
        
        if self.learning_system:
            self.learning_system.shutdown()
        
        if self.deep_question_system:
            self.deep_question_system.shutdown()
        
        if self.image_generation_system:
            self.image_generation_system.shutdown()
        
        self.logger.info("Feature manager shutdown complete")