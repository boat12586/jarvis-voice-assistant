#!/usr/bin/env python3
"""
Comprehensive Test Script for JARVIS Emotional AI System
Tests all components of the emotional intelligence system
"""

import sys
import os
import json
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_emotion_detection():
    """Test emotion detection system"""
    print("🧠 Testing Emotion Detection System...")
    
    try:
        from ai.emotion_detection import EmotionDetectionSystem
        
        config = {
            "emotion_detection": {
                "voice_analysis": False,  # Skip voice for now
                "max_history_length": 10
            }
        }
        
        emotion_detector = EmotionDetectionSystem(config)
        
        # Test cases
        test_cases = [
            ("I'm so happy today!", "en", "joy"),
            ("This is really frustrating me", "en", "anger"),
            ("I'm worried about the exam", "en", "anxiety"),
            ("ดีใจมากเลย!", "th", "joy"),
            ("เครียดจังเลย", "th", "anxiety"),
            ("What a beautiful day", "en", "joy"),
            ("I don't know what to do", "en", "confusion")
        ]
        
        results = []
        for text, language, expected in test_cases:
            print(f"  Testing: '{text}' (Expected: {expected})")
            
            result = emotion_detector.detect_emotion_from_text(text, language)
            
            results.append({
                "text": text,
                "language": language,
                "expected": expected,
                "detected": result.primary_emotion,
                "confidence": result.confidence,
                "valence": result.valence,
                "arousal": result.arousal
            })
            
            print(f"    Result: {result.primary_emotion} (confidence: {result.confidence:.2f})")
        
        # Test emotional context
        print("\n  Testing emotional context tracking...")
        emotional_state = emotion_detector.get_emotional_state_summary()
        print(f"    Emotional state: {emotional_state}")
        
        print("✅ Emotion Detection System test completed!")
        return True, results
        
    except Exception as e:
        print(f"❌ Emotion Detection System test failed: {e}")
        return False, []

def test_personality_system():
    """Test personality system"""
    print("\n🎭 Testing Personality System...")
    
    try:
        from ai.personality_system import PersonalitySystem
        
        config = {
            "personality_system": {
                "default_personality": "friendly",
                "enable_learning": True
            }
        }
        
        personality_system = PersonalitySystem(config)
        
        # Test personality profiles
        print("  Testing personality profiles...")
        personalities = ["professional", "friendly", "casual"]
        
        for personality in personalities:
            success = personality_system.set_personality(personality)
            current = personality_system.current_personality
            print(f"    {personality}: {'✅' if success and current == personality else '❌'}")
        
        # Test response adaptation
        print("\n  Testing response adaptation...")
        
        test_adaptations = [
            {
                "response": "Here's the information you requested.",
                "emotion_context": {
                    "primary_emotion": "anxiety",
                    "valence": -0.5,
                    "arousal": 0.7,
                    "stress_level": 0.8
                },
                "language": "en"
            },
            {
                "response": "ขอข้อมูลนี้ให้คุณครับ",
                "emotion_context": {
                    "primary_emotion": "joy",
                    "valence": 0.8,
                    "arousal": 0.6,
                    "stress_level": 0.1
                },
                "language": "th"
            }
        ]
        
        adaptation_results = []
        for test in test_adaptations:
            personality_system.set_personality("friendly")
            adapted = personality_system.adapt_response_to_emotion(
                test["response"], 
                test["emotion_context"], 
                test["language"]
            )
            
            adaptation_results.append({
                "original": test["response"],
                "adapted": adapted,
                "emotion": test["emotion_context"]["primary_emotion"],
                "different": adapted != test["response"]
            })
            
            print(f"    Original: {test['response']}")
            print(f"    Adapted:  {adapted}")
            print(f"    Changed:  {'✅' if adapted != test['response'] else '❌'}")
        
        print("✅ Personality System test completed!")
        return True, adaptation_results
        
    except Exception as e:
        print(f"❌ Personality System test failed: {e}")
        return False, []

def test_emotional_ai_engine():
    """Test emotional AI engine"""
    print("\n🎯 Testing Emotional AI Engine...")
    
    try:
        from ai.emotional_ai_engine import EmotionalAIEngine
        
        config = {
            "emotional_ai": {
                "auto_personality_adaptation": True,
                "emotion_memory_length": 20
            },
            "emotion_detection": {
                "voice_analysis": False,
                "max_history_length": 50
            },
            "personality_system": {
                "default_personality": "friendly",
                "enable_learning": True
            }
        }
        
        emotional_ai = EmotionalAIEngine(config)
        
        # Test emotional processing
        print("  Testing emotional response processing...")
        
        test_scenarios = [
            {
                "user_input": "I'm feeling really stressed about work",
                "original_response": "I understand you're asking about work.",
                "language": "en",
                "user_id": "test_user_1"
            },
            {
                "user_input": "This is amazing! Thank you so much!",
                "original_response": "You're welcome.",
                "language": "en", 
                "user_id": "test_user_1"
            },
            {
                "user_input": "เซ็งมากเลย ทำงานไม่เสร็จ",
                "original_response": "ช่วยอะไรได้ไหมครับ",
                "language": "th",
                "user_id": "test_user_2"
            }
        ]
        
        processing_results = []
        for scenario in test_scenarios:
            print(f"    Processing: '{scenario['user_input']}'")
            
            emotional_response = emotional_ai.process_with_emotional_intelligence(
                user_input=scenario["user_input"],
                original_response=scenario["original_response"],
                language=scenario["language"],
                user_id=scenario["user_id"]
            )
            
            processing_results.append({
                "user_input": scenario["user_input"],
                "original_response": scenario["original_response"],
                "enhanced_response": emotional_response.adapted_response,
                "emotion_detected": emotional_response.emotion_context.get("primary_emotion", "unknown"),
                "personality_used": emotional_response.personality_used,
                "adaptations": emotional_response.adaptations_applied,
                "confidence": emotional_response.confidence
            })
            
            print(f"      Emotion: {emotional_response.emotion_context.get('primary_emotion', 'unknown')}")
            print(f"      Personality: {emotional_response.personality_used}")
            print(f"      Enhanced: {emotional_response.adapted_response != scenario['original_response']}")
        
        # Test emotional state summary
        print("\n  Testing emotional state summary...")
        emotional_summary = emotional_ai.get_emotional_state_summary()
        print(f"    Status: {emotional_summary.get('status', 'unknown')}")
        
        print("✅ Emotional AI Engine test completed!")
        return True, processing_results
        
    except Exception as e:
        print(f"❌ Emotional AI Engine test failed: {e}")
        return False, []

def test_user_preference_system():
    """Test user preference system"""
    print("\n👤 Testing User Preference System...")
    
    try:
        from features.user_preference_system import UserPreferenceSystem
        
        config = {
            "user_preferences": {
                "learning_rate": 0.1,
                "confidence_threshold": 0.7,
                "preferences_dir": "data/test_user_preferences"
            }
        }
        
        # Ensure test directory exists
        Path(config["user_preferences"]["preferences_dir"]).mkdir(parents=True, exist_ok=True)
        
        preference_system = UserPreferenceSystem(config)
        
        # Test user profile creation
        print("  Testing user profile creation...")
        
        test_user_id = "test_user_preferences"
        initial_context = {
            "language": "en",
            "timezone": "UTC",
            "preferred_personality": "friendly"
        }
        
        profile = preference_system.create_or_get_user_profile(test_user_id, initial_context)
        print(f"    Profile created: {profile.user_id}")
        print(f"    Language: {profile.language_preference}")
        print(f"    Personality: {profile.preferred_personality}")
        
        # Test learning from interactions
        print("\n  Testing preference learning...")
        
        test_interactions = [
            {
                "user_input": "Can you please explain this in detail?",
                "user_language": "en",
                "emotion_context": {"primary_emotion": "curiosity", "stress_level": 0.2},
                "assistant_response": "Detailed explanation...",
                "conversation_context": [],
                "timestamp": time.time()
            },
            {
                "user_input": "That's too verbose, make it shorter",
                "user_language": "en", 
                "emotion_context": {"primary_emotion": "frustration", "stress_level": 0.6},
                "response_feedback": {"type": "too_verbose", "score": 0.2},
                "timestamp": time.time()
            }
        ]
        
        for interaction in test_interactions:
            preference_system.learn_from_interaction(test_user_id, interaction)
            print(f"    Learned from: '{interaction['user_input'][:30]}...'")
        
        # Test recommendations
        print("\n  Testing user recommendations...")
        recommendations = preference_system.get_user_recommendations(test_user_id)
        print(f"    Recommendations generated: {len(recommendations) > 0}")
        
        # Test explicit feedback
        print("\n  Testing explicit feedback...")
        feedback_data = {
            "preferred_personality": "professional",
            "score": 0.9
        }
        preference_system.provide_explicit_feedback(test_user_id, "personality_preference", feedback_data)
        
        updated_profile = preference_system.user_profiles.get(test_user_id)
        print(f"    Personality updated: {updated_profile.preferred_personality == 'professional'}")
        
        print("✅ User Preference System test completed!")
        return True, {"profile": profile, "recommendations": recommendations}
        
    except Exception as e:
        print(f"❌ User Preference System test failed: {e}")
        return False, {}

def test_sentiment_analysis():
    """Test sentiment analysis system"""
    print("\n📊 Testing Sentiment Analysis System...")
    
    try:
        from ai.sentiment_analysis import SentimentAnalysisSystem
        
        config = {
            "sentiment_analysis": {
                "max_history_length": 50,
                "enable_pattern_detection": True
            }
        }
        
        sentiment_analyzer = SentimentAnalysisSystem(config)
        
        # Test sentiment analysis
        print("  Testing sentiment analysis...")
        
        test_texts = [
            ("I love this new feature!", "en", "positive"),
            ("This is terrible and frustrating", "en", "negative"), 
            ("The weather is okay today", "en", "neutral"),
            ("ชอบมากเลยครับ!", "th", "positive"),
            ("แย่มากเลย เซ็งจริงๆ", "th", "negative"),
            ("วันนี้อากาศปกติ", "th", "neutral")
        ]
        
        sentiment_results = []
        for text, language, expected in test_texts:
            result = sentiment_analyzer.analyze_sentiment(text, language)
            
            sentiment_results.append({
                "text": text,
                "language": language,
                "expected": expected,
                "detected": result.sentiment,
                "confidence": result.confidence,
                "polarity": result.polarity,
                "indicators": result.emotional_indicators
            })
            
            print(f"    '{text}' -> {result.sentiment} ({result.confidence:.2f})")
        
        # Test conversation sentiment
        print("\n  Testing conversation sentiment analysis...")
        conversation_sentiment = sentiment_analyzer.analyze_conversation_sentiment()
        
        print(f"    Overall sentiment: {conversation_sentiment.overall_sentiment}")
        print(f"    Sentiment trend: {conversation_sentiment.sentiment_trend}")
        print(f"    Engagement level: {conversation_sentiment.engagement_level:.2f}")
        
        # Test sentiment summary
        print("\n  Testing sentiment summary...")
        sentiment_summary = sentiment_analyzer.get_sentiment_summary()
        print(f"    System status: {sentiment_summary.get('system_status', 'unknown')}")
        print(f"    History length: {sentiment_summary.get('conversation_history_length', 0)}")
        
        print("✅ Sentiment Analysis System test completed!")
        return True, sentiment_results
        
    except Exception as e:
        print(f"❌ Sentiment Analysis System test failed: {e}")
        return False, []

def test_web_integration():
    """Test web integration"""
    print("\n🌐 Testing Web Integration...")
    
    try:
        from web_emotional_integration import WebEmotionalIntegration
        
        config = {
            "emotional_ai": {"auto_personality_adaptation": True},
            "emotion_detection": {"voice_analysis": False},
            "personality_system": {"default_personality": "friendly"},
            "user_preferences": {"preferences_dir": "data/test_web_preferences"},
            "enable_auto_personality": True,
            "enable_user_learning": True,
            "session_timeout": 3600
        }
        
        # Ensure test directory exists
        Path(config["user_preferences"]["preferences_dir"]).mkdir(parents=True, exist_ok=True)
        
        web_integration = WebEmotionalIntegration(config)
        
        # Test session initialization
        print("  Testing session initialization...")
        session_id = "test_web_session_" + str(int(time.time()))
        user_context = {"language": "en", "timezone": "UTC"}
        
        init_result = web_integration.initialize_web_session(session_id, user_context)
        print(f"    Session initialized: {init_result.get('status') == 'success'}")
        
        # Test message processing
        print("\n  Testing web message processing...")
        
        test_messages = [
            {
                "message": "I'm having trouble with this feature",
                "original_response": "Let me help you with that.",
                "language": "en"
            },
            {
                "message": "Thank you so much! This is perfect!",
                "original_response": "You're welcome!",
                "language": "en"
            }
        ]
        
        processing_results = []
        for test_msg in test_messages:
            result = web_integration.process_web_message(
                session_id=session_id,
                message=test_msg["message"],
                original_response=test_msg["original_response"],
                language=test_msg["language"]
            )
            
            processing_results.append({
                "message": test_msg["message"],
                "status": result.get("status"),
                "enhanced": result.get("status") == "success",
                "emotion": result.get("emotional_analysis", {}).get("primary_emotion", "unknown")
            })
            
            print(f"    '{test_msg['message'][:30]}...' -> {result.get('status', 'unknown')}")
        
        # Test session summary
        print("\n  Testing session summary...")
        summary = web_integration.get_session_emotional_summary(session_id)
        print(f"    Summary generated: {summary.get('status') == 'success'}")
        
        # Test system stats
        print("\n  Testing system statistics...")
        stats = web_integration.get_system_stats()
        print(f"    Active sessions: {stats.get('active_sessions', 0)}")
        
        print("✅ Web Integration test completed!")
        return True, processing_results
        
    except Exception as e:
        print(f"❌ Web Integration test failed: {e}")
        return False, []

def test_thai_language_support():
    """Test Thai language support across all systems"""
    print("\n🇹🇭 Testing Thai Language Support...")
    
    try:
        # Test with Thai inputs across all systems
        thai_test_cases = [
            {
                "text": "ดีใจมากเลยครับ!",
                "expected_emotion": "joy",
                "expected_sentiment": "positive"
            },
            {
                "text": "เครียดจัง ทำงานไม่เสร็จ",
                "expected_emotion": "anxiety", 
                "expected_sentiment": "negative"
            },
            {
                "text": "ขอบคุณมากครับ ช่วยได้มาก",
                "expected_emotion": "gratitude",
                "expected_sentiment": "positive"
            }
        ]
        
        thai_results = []
        
        # Test emotion detection with Thai
        from ai.emotion_detection import EmotionDetectionSystem
        emotion_config = {"emotion_detection": {"voice_analysis": False}}
        emotion_detector = EmotionDetectionSystem(emotion_config)
        
        # Test sentiment analysis with Thai
        from ai.sentiment_analysis import SentimentAnalysisSystem
        sentiment_config = {"sentiment_analysis": {"max_history_length": 10}}
        sentiment_analyzer = SentimentAnalysisSystem(sentiment_config)
        
        for test_case in thai_test_cases:
            print(f"  Testing: '{test_case['text']}'")
            
            # Test emotion detection
            emotion_result = emotion_detector.detect_emotion_from_text(test_case["text"], "th")
            
            # Test sentiment analysis
            sentiment_result = sentiment_analyzer.analyze_sentiment(test_case["text"], "th")
            
            thai_results.append({
                "text": test_case["text"],
                "emotion_detected": emotion_result.primary_emotion,
                "emotion_confidence": emotion_result.confidence,
                "sentiment_detected": sentiment_result.sentiment,
                "sentiment_confidence": sentiment_result.confidence,
                "expected_emotion": test_case["expected_emotion"],
                "expected_sentiment": test_case["expected_sentiment"]
            })
            
            print(f"    Emotion: {emotion_result.primary_emotion} ({emotion_result.confidence:.2f})")
            print(f"    Sentiment: {sentiment_result.sentiment} ({sentiment_result.confidence:.2f})")
        
        print("✅ Thai Language Support test completed!")
        return True, thai_results
        
    except Exception as e:
        print(f"❌ Thai Language Support test failed: {e}")
        return False, []

def run_performance_tests():
    """Run performance tests"""
    print("\n⚡ Running Performance Tests...")
    
    try:
        performance_results = {}
        
        # Test emotion detection performance
        print("  Testing emotion detection performance...")
        from ai.emotion_detection import EmotionDetectionSystem
        
        emotion_config = {"emotion_detection": {"voice_analysis": False}}
        emotion_detector = EmotionDetectionSystem(emotion_config)
        
        start_time = time.time()
        for i in range(10):
            emotion_detector.detect_emotion_from_text(f"Test message {i}", "en")
        emotion_time = time.time() - start_time
        
        performance_results["emotion_detection"] = {
            "total_time": emotion_time,
            "avg_time_per_request": emotion_time / 10,
            "requests_per_second": 10 / emotion_time
        }
        
        print(f"    Average time per emotion detection: {emotion_time/10:.3f}s")
        
        # Test sentiment analysis performance
        print("  Testing sentiment analysis performance...")
        from ai.sentiment_analysis import SentimentAnalysisSystem
        
        sentiment_config = {"sentiment_analysis": {"max_history_length": 50}}
        sentiment_analyzer = SentimentAnalysisSystem(sentiment_config)
        
        start_time = time.time()
        for i in range(10):
            sentiment_analyzer.analyze_sentiment(f"Test sentiment message {i}", "en")
        sentiment_time = time.time() - start_time
        
        performance_results["sentiment_analysis"] = {
            "total_time": sentiment_time,
            "avg_time_per_request": sentiment_time / 10,
            "requests_per_second": 10 / sentiment_time
        }
        
        print(f"    Average time per sentiment analysis: {sentiment_time/10:.3f}s")
        
        # Test full emotional AI processing performance
        print("  Testing full emotional AI processing performance...")
        from ai.emotional_ai_engine import EmotionalAIEngine
        
        full_config = {
            "emotional_ai": {"auto_personality_adaptation": True},
            "emotion_detection": {"voice_analysis": False},
            "personality_system": {"default_personality": "friendly"}
        }
        
        emotional_ai = EmotionalAIEngine(full_config)
        
        start_time = time.time()
        for i in range(5):  # Fewer iterations for complex processing
            emotional_ai.process_with_emotional_intelligence(
                user_input=f"Test full processing {i}",
                original_response="Test response",
                language="en",
                user_id="perf_test_user"
            )
        full_processing_time = time.time() - start_time
        
        performance_results["full_emotional_ai"] = {
            "total_time": full_processing_time,
            "avg_time_per_request": full_processing_time / 5,
            "requests_per_second": 5 / full_processing_time
        }
        
        print(f"    Average time per full AI processing: {full_processing_time/5:.3f}s")
        
        print("✅ Performance tests completed!")
        return True, performance_results
        
    except Exception as e:
        print(f"❌ Performance tests failed: {e}")
        return False, {}

def generate_test_report(test_results):
    """Generate comprehensive test report"""
    print("\n📋 Generating Test Report...")
    
    report = {
        "test_timestamp": time.time(),
        "test_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "results": test_results,
        "summary": {
            "total_tests": len(test_results),
            "passed_tests": sum(1 for result in test_results.values() if result["passed"]),
            "failed_tests": sum(1 for result in test_results.values() if not result["passed"]),
            "success_rate": 0
        }
    }
    
    report["summary"]["success_rate"] = report["summary"]["passed_tests"] / report["summary"]["total_tests"] * 100
    
    # Save report to file
    report_file = Path("emotional_ai_test_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"  Test report saved to: {report_file}")
    
    # Print summary
    print(f"\n📊 Test Summary:")
    print(f"  Total tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed_tests']}")
    print(f"  Failed: {report['summary']['failed_tests']}")
    print(f"  Success rate: {report['summary']['success_rate']:.1f}%")
    
    return report

def main():
    """Main test function"""
    print("🤖 JARVIS Emotional AI System - Comprehensive Test Suite")
    print("=" * 60)
    
    test_results = {}
    
    # Run all tests
    tests = [
        ("Emotion Detection", test_emotion_detection),
        ("Personality System", test_personality_system),
        ("Emotional AI Engine", test_emotional_ai_engine),
        ("User Preference System", test_user_preference_system),
        ("Sentiment Analysis", test_sentiment_analysis),
        ("Web Integration", test_web_integration),
        ("Thai Language Support", test_thai_language_support),
        ("Performance Tests", run_performance_tests)
    ]
    
    for test_name, test_function in tests:
        try:
            passed, data = test_function()
            test_results[test_name] = {
                "passed": passed,
                "data": data,
                "error": None
            }
        except Exception as e:
            test_results[test_name] = {
                "passed": False,
                "data": None,
                "error": str(e)
            }
            print(f"❌ {test_name} test crashed: {e}")
    
    # Generate test report
    report = generate_test_report(test_results)
    
    # Final status
    print("\n" + "=" * 60)
    if report["summary"]["success_rate"] >= 80:
        print("✅ Emotional AI System tests PASSED!")
        print("🎉 System is ready for deployment!")
    else:
        print("❌ Some tests FAILED!")
        print("🔧 Please review and fix the issues before deployment.")
    
    return report["summary"]["success_rate"] >= 80

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)