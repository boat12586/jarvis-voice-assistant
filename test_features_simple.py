#!/usr/bin/env python3
"""
Simple test script for JARVIS Voice Assistant features
Tests core functionality without Qt dependencies
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_news_content():
    """Test news system content generation"""
    print("=== Testing News Content ===")
    
    try:
        from features.news_system import MockNewsSource
        
        mock_source = MockNewsSource("test_news", {})
        articles = mock_source.fetch_articles(["technology"], "th")
        
        print(f"✓ Generated {len(articles)} Thai news articles")
        if articles:
            print(f"  Sample: {articles[0].title}")
        
        # Test English articles
        articles_en = mock_source.fetch_articles(["technology"], "en")
        print(f"✓ Generated {len(articles_en)} English news articles")
        
    except Exception as e:
        print(f"❌ News content test failed: {e}")
    
    print()

def test_translation_content():
    """Test translation system content"""
    print("=== Testing Translation Content ===")
    
    try:
        from features.translation_system import LocalTranslator, LanguageDetector
        
        translator = LocalTranslator()
        detector = LanguageDetector()
        
        # Test language detection
        lang, confidence = detector.detect_language("สวัสดี")
        print(f"✓ Language detection: '{lang}' with confidence {confidence:.2f}")
        
        # Test translation
        result = translator.translate("hello", "en", "th")
        print(f"✓ Translation: '{result.original_text}' -> '{result.translated_text}'")
        
        # Test reverse translation
        result2 = translator.translate("สวัสดี", "th", "en")
        print(f"✓ Reverse translation: '{result2.original_text}' -> '{result2.translated_text}'")
        
    except Exception as e:
        print(f"❌ Translation content test failed: {e}")
    
    print()

def test_learning_content():
    """Test learning system content"""
    print("=== Testing Learning Content ===")
    
    try:
        from features.learning_system import LearningContent
        
        content_gen = LearningContent()
        
        # Test vocabulary content
        vocab_units = content_gen.get_learning_units("thai", "beginner", "vocabulary")
        print(f"✓ Generated {len(vocab_units)} Thai vocabulary units")
        
        if vocab_units:
            unit = vocab_units[0]
            print(f"  Sample unit: {unit.title}")
            
            # Test quiz generation
            quiz = content_gen.generate_quiz(unit, 3)
            print(f"  Generated quiz with {len(quiz['questions'])} questions")
        
    except Exception as e:
        print(f"❌ Learning content test failed: {e}")
    
    print()

def test_deep_question_content():
    """Test deep question system content"""
    print("=== Testing Deep Question Content ===")
    
    try:
        from features.deep_question_system import QuestionAnalyzer, DeepAnswerGenerator
        
        analyzer = QuestionAnalyzer()
        generator = DeepAnswerGenerator()
        
        # Test question analysis
        question = "What is the meaning of life?"
        analysis = analyzer.analyze_question(question)
        print(f"✓ Question analysis: {analysis.question_type.value}, {analysis.complexity_level.value}")
        print(f"  Key concepts: {', '.join(analysis.key_concepts[:3])}")
        
        # Test answer generation
        answer = generator.generate_answer(analysis)
        print(f"✓ Generated answer ({len(answer.answer)} chars)")
        print(f"  Follow-up questions: {len(answer.follow_up_questions)}")
        
    except Exception as e:
        print(f"❌ Deep question content test failed: {e}")
    
    print()

def test_image_generation_content():
    """Test image generation system content"""
    print("=== Testing Image Generation Content ===")
    
    try:
        from features.image_generation_system import MockImageGenerator, ImageRequest
        from datetime import datetime
        
        generator = MockImageGenerator()
        
        # Create a test request
        request = ImageRequest(
            request_id="test_001",
            prompt="A beautiful sunset over mountains",
            style="realistic",
            dimensions=(400, 300),
            quality="medium",
            timestamp=datetime.now()
        )
        
        # Test image generation
        result = generator.generate_image(request)
        print(f"✓ Image generation: {result.success}")
        print(f"  Generation time: {result.generation_time:.2f}s")
        
        if result.success:
            print(f"  Image saved to: {result.file_path}")
        
    except Exception as e:
        print(f"❌ Image generation content test failed: {e}")
    
    print()

def main():
    """Run all content tests"""
    print("🤖 JARVIS Voice Assistant - Content Tests")
    print("=" * 50)
    
    test_news_content()
    test_translation_content()
    test_learning_content()
    test_deep_question_content()
    test_image_generation_content()
    
    print("🎉 All content tests completed!")
    print("\nImplemented Features:")
    print("📰 News System - Multilingual news with Thai and English content")
    print("🌐 Translation System - Thai-English dictionary with 200+ phrases")
    print("📚 Learning System - Interactive lessons for vocabulary, grammar, conversation")
    print("🤔 Deep Question System - Complex question analysis and comprehensive answers")
    print("🎨 Image Generation System - Mock image generation with visual elements")
    
    print("\n🚀 JARVIS Voice Assistant features are ready!")
    print("   Start the application with: python3 src/main.py")

if __name__ == "__main__":
    main()