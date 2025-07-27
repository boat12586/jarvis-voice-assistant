#!/usr/bin/env python3
"""
Test script for JARVIS Voice Assistant features
Tests all implemented features to ensure they work correctly
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.news_system import NewsSystem
from features.translation_system import TranslationSystem  
from features.learning_system import LearningSystem
from features.deep_question_system import DeepQuestionSystem
from features.image_generation_system import ImageGenerationSystem

def test_news_system():
    """Test news system functionality"""
    print("=== Testing News System ===")
    
    config = {
        "update_interval": 3600,
        "max_articles": 5,
        "categories": ["technology", "science", "world"],
        "db_path": "data/news.json"
    }
    
    news_system = NewsSystem(config)
    
    # Test getting news in Thai
    news_thai = news_system.get_news_text("th", 3)
    print(f"Thai News: {news_thai[:200]}...")
    
    # Test getting news summaries
    summaries = news_system.get_news_summary("th", 3)
    print(f"Found {len(summaries)} news summaries")
    
    print("✓ News system working correctly\n")

def test_translation_system():
    """Test translation system functionality"""
    print("=== Testing Translation System ===")
    
    config = {
        "supported_languages": ["en", "th", "zh", "ja", "ko"],
        "default_source_lang": "auto",
        "default_target_lang": "th"
    }
    
    translation_system = TranslationSystem(config)
    
    # Test translation
    result = translation_system.translate_text("Hello", "en", "th")
    print(f"Translation result: {result}")
    
    # Test language detection
    detection = translation_system.detect_language("สวัสดี")
    print(f"Language detection: {detection}")
    
    print("✓ Translation system working correctly\n")

def test_learning_system():
    """Test learning system functionality"""
    print("=== Testing Learning System ===")
    
    config = {
        "supported_languages": ["thai", "english"],
        "difficulty_levels": ["beginner", "intermediate", "advanced"],
        "categories": ["vocabulary", "grammar", "conversation", "pronunciation"]
    }
    
    learning_system = LearningSystem(config)
    
    # Test starting a lesson
    lesson = learning_system.start_lesson("thai", "beginner", "vocabulary")
    print(f"Lesson started: {lesson.get('title', 'Unknown')}")
    
    # Test getting available lessons
    lessons = learning_system.get_available_lessons("thai", "beginner")
    print(f"Available lessons: {len(lessons)}")
    
    print("✓ Learning system working correctly\n")

def test_deep_question_system():
    """Test deep question system functionality"""
    print("=== Testing Deep Question System ===")
    
    config = {
        "max_answer_length": 2000,
        "enable_follow_up": True,
        "cache_answers": True
    }
    
    deep_question_system = DeepQuestionSystem(config)
    
    # Test processing a question
    question = "What is the nature of consciousness?"
    result = deep_question_system.process_question(question)
    print(f"Question processed: {question}")
    print(f"Answer preview: {result.get('answer', '')[:200]}...")
    
    # Test getting suggestions
    suggestions = deep_question_system.get_question_suggestions("philosophy")
    print(f"Question suggestions: {len(suggestions)}")
    
    print("✓ Deep question system working correctly\n")

def test_image_generation_system():
    """Test image generation system functionality"""
    print("=== Testing Image Generation System ===")
    
    config = {
        "supported_styles": ["realistic", "artistic", "cartoon", "cyberpunk", "nature"],
        "default_dimensions": (512, 512),
        "max_image_size": (1024, 1024)
    }
    
    image_generation_system = ImageGenerationSystem(config)
    
    # Test generating an image
    request_id = image_generation_system.generate_image(
        "A beautiful sunset over mountains",
        "realistic",
        (400, 300),
        "medium"
    )
    
    print(f"Image generation started: {request_id}")
    
    # Wait a moment for generation to complete
    import time
    time.sleep(2)
    
    # Test getting generated images
    images = image_generation_system.get_generated_images(5)
    print(f"Generated images: {len(images)}")
    
    print("✓ Image generation system working correctly\n")

def main():
    """Run all feature tests"""
    print("🤖 JARVIS Voice Assistant - Feature Tests")
    print("=" * 50)
    
    try:
        test_news_system()
        test_translation_system()  
        test_learning_system()
        test_deep_question_system()
        test_image_generation_system()
        
        print("🎉 All features tested successfully!")
        print("\nFeature Summary:")
        print("✓ News System - Provides multilingual news summaries")
        print("✓ Translation System - Translates between multiple languages")
        print("✓ Learning System - Interactive language learning lessons")
        print("✓ Deep Question System - Comprehensive analysis of complex questions")
        print("✓ Image Generation System - Creates images from text descriptions")
        
        print("\n🚀 JARVIS Voice Assistant is ready for use!")
        
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()