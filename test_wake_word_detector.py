#!/usr/bin/env python3
"""
Test script for Wake Word Detection
Tests pattern matching and basic wake word functionality
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice.wake_word_detector import WakeWordDetector, WakeWordConfig
from PyQt6.QtCore import QCoreApplication
import numpy as np

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def test_wake_word_patterns():
    """Test wake word pattern matching"""
    print("🎯 Testing Wake Word Pattern Matching...")
    
    config = WakeWordConfig.get_default_config()
    detector = WakeWordDetector(config)
    
    # Test wake word detection in text
    test_phrases = [
        {
            "text": "hey jarvis",
            "should_detect": True,
            "expected_phrase": "hey jarvis"
        },
        {
            "text": "hi jarvis",
            "should_detect": True,
            "expected_phrase": "hi jarvis"
        },
        {
            "text": "jarvis",
            "should_detect": True,
            "expected_phrase": "jarvis"
        },
        {
            "text": "สวัสดี จาร์วิส",
            "should_detect": True,
            "expected_phrase": "สวัสดี จาร์วิส"
        },
        {
            "text": "จาร์วิส",
            "should_detect": True,
            "expected_phrase": "จาร์วิส"
        },
        {
            "text": "hello world",
            "should_detect": False,
            "expected_phrase": None
        },
        {
            "text": "hey javis",  # Mispronunciation
            "should_detect": True,
            "expected_phrase": "jarvis"
        },
        {
            "text": "ok jarvis how are you",
            "should_detect": True,
            "expected_phrase": "jarvis"
        }
    ]
    
    results = []
    for test in test_phrases:
        print(f"\n  Testing: '{test['text']}'")
        
        # Test pattern matching
        detected = False
        detected_phrase = None
        confidence = 0.0
        
        # Check if any wake phrase matches
        for phrase in detector.wake_phrases:
            if detector._match_wake_phrase(test["text"].lower(), phrase.lower()):
                detected = True
                detected_phrase = phrase
                confidence = detector._calculate_confidence(test["text"].lower(), phrase.lower())
                break
        
        print(f"    Detected: {detected}")
        if detected:
            print(f"    Phrase: {detected_phrase}")
            print(f"    Confidence: {confidence:.3f}")
        
        # Check expectations
        expectation_met = (detected == test["should_detect"])
        if test["should_detect"] and detected:
            # For positive cases, check if detected phrase is reasonable
            expectation_met = confidence > 0.5
        
        results.append(expectation_met)
        print(f"    Expected: {test['should_detect']} → {'✅' if expectation_met else '❌'}")
    
    accuracy = sum(results) / len(results) * 100
    print(f"\n📊 Pattern Matching Accuracy: {accuracy:.1f}% ({sum(results)}/{len(results)})")
    return accuracy > 75

def test_confidence_calculation():
    """Test confidence calculation"""
    print("🎲 Testing Confidence Calculation...")
    
    config = WakeWordConfig.get_default_config()
    detector = WakeWordDetector(config)
    
    # Test confidence for different scenarios
    test_cases = [
        {
            "text": "hey jarvis",
            "wake_phrase": "hey jarvis",
            "expected_confidence": "> 0.8"  # Exact match should be high
        },
        {
            "text": "hey javis",
            "wake_phrase": "hey jarvis", 
            "expected_confidence": "> 0.6"  # Similar should be good
        },
        {
            "text": "hello there jarvis how are you",
            "wake_phrase": "jarvis",
            "expected_confidence": "> 0.5"  # Contains wake word
        },
        {
            "text": "jarvis please help me with something",
            "wake_phrase": "jarvis",
            "expected_confidence": "> 0.7"  # Starts with wake word
        },
        {
            "text": "completely different text",
            "wake_phrase": "jarvis",
            "expected_confidence": "< 0.3"  # Should be low
        }
    ]
    
    results = []
    for test in test_cases:
        confidence = detector._calculate_confidence(test["text"], test["wake_phrase"])
        
        print(f"  Text: '{test['text']}'")
        print(f"    Wake phrase: '{test['wake_phrase']}'")
        print(f"    Confidence: {confidence:.3f}")
        print(f"    Expected: {test['expected_confidence']}")
        
        # Check expectation
        if test["expected_confidence"].startswith(">"):
            threshold = float(test["expected_confidence"][2:])
            expectation_met = confidence > threshold
        else:  # "<"
            threshold = float(test["expected_confidence"][2:])
            expectation_met = confidence < threshold
        
        results.append(expectation_met)
        print(f"    Result: {'✅' if expectation_met else '❌'}")
        print()
    
    accuracy = sum(results) / len(results) * 100
    print(f"📊 Confidence Calculation Accuracy: {accuracy:.1f}%")
    return accuracy > 80

def test_similarity_matching():
    """Test word similarity matching"""
    print("🔍 Testing Word Similarity...")
    
    config = WakeWordConfig.get_default_config()
    detector = WakeWordDetector(config)
    
    # Test word similarity
    similarity_tests = [
        ("jarvis", "jarvis", True),  # Exact match
        ("jarvis", "javis", True),   # Common mispronunciation
        ("jarvis", "jarfis", True),  # Close variation
        ("jarvis", "hello", False),  # Different word
        ("hey", "hi", False),        # Different but similar meaning
        ("จาร์วิส", "จาร์วิส", True),  # Thai exact match
        ("jarvis", "จาร์วิส", False),  # Different languages
    ]
    
    results = []
    for word1, word2, expected in similarity_tests:
        similar = detector._words_similar(word1, word2)
        correct = similar == expected
        results.append(correct)
        
        print(f"  '{word1}' ~ '{word2}': {similar} (expected: {expected}) {'✅' if correct else '❌'}")
    
    accuracy = sum(results) / len(results) * 100
    print(f"\n📊 Similarity Matching Accuracy: {accuracy:.1f}%")
    return accuracy > 85

def test_configuration():
    """Test configuration validation"""
    print("⚙️ Testing Configuration...")
    
    # Test default config
    default_config = WakeWordConfig.get_default_config()
    print(f"  Default config valid: {WakeWordConfig.validate_config(default_config)}")
    
    # Test invalid configs
    invalid_configs = [
        {"wake_word": {"confidence_threshold": 1.5}},  # Invalid threshold
        {"wake_word": {"sample_rate": 12345}},         # Invalid sample rate
        {"wake_word": {}},                             # Missing fields
    ]
    
    validation_results = []
    for i, config in enumerate(invalid_configs):
        valid = WakeWordConfig.validate_config(config)
        validation_results.append(not valid)  # Should be invalid
        print(f"  Invalid config {i+1}: {not valid} {'✅' if not valid else '❌'}")
    
    # Test modified valid config
    valid_config = default_config.copy()
    valid_config["wake_word"]["confidence_threshold"] = 0.8
    valid_modified = WakeWordConfig.validate_config(valid_config)
    validation_results.append(valid_modified)
    print(f"  Modified valid config: {valid_modified} {'✅' if valid_modified else '❌'}")
    
    accuracy = sum(validation_results) / len(validation_results) * 100
    print(f"\n📊 Configuration Validation: {accuracy:.1f}%")
    return accuracy == 100

def test_detector_status():
    """Test detector status and initialization"""
    print("📊 Testing Detector Status...")
    
    config = WakeWordConfig.get_default_config()
    detector = WakeWordDetector(config)
    
    # Get status
    status = detector.get_status()
    
    print(f"  Is listening: {status['is_listening']}")
    print(f"  Wake phrases: {len(status['wake_phrases'])}")
    print(f"  Confidence threshold: {status['confidence_threshold']}")
    print(f"  Sample rate: {status['sample_rate']}")
    print(f"  Templates loaded: {status['templates_loaded']}")
    
    # Verify expected values
    checks = [
        status['is_listening'] == False,  # Should not be listening initially
        len(status['wake_phrases']) > 0,  # Should have wake phrases
        0 < status['confidence_threshold'] <= 1.0,  # Valid threshold
        status['sample_rate'] > 0,  # Valid sample rate
        status['templates_loaded'] > 0,  # Should have templates
    ]
    
    success = all(checks)
    print(f"\n  Status validation: {'✅' if success else '❌'}")
    return success

def test_performance():
    """Test wake word detection performance"""
    print("⚡ Testing Performance...")
    
    config = WakeWordConfig.get_default_config()
    detector = WakeWordDetector(config)
    
    # Test batch processing
    test_texts = [
        "hey jarvis",
        "jarvis help me",
        "สวัสดี จาร์วิส",
        "hello world",
        "how are you today"
    ] * 20  # 100 total tests
    
    start_time = time.time()
    
    detections = 0
    for text in test_texts:
        for phrase in detector.wake_phrases:
            if detector._match_wake_phrase(text.lower(), phrase.lower()):
                detections += 1
                break
    
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / len(test_texts)
    
    print(f"  Processed {len(test_texts)} texts in {total_time:.3f}s")
    print(f"  Average time per text: {avg_time*1000:.2f}ms")
    print(f"  Wake words detected: {detections}")
    print(f"  Detection rate: {detections/len(test_texts)*100:.1f}%")
    
    # Performance criteria
    performance_good = (
        avg_time < 0.01 and  # < 10ms per text
        total_time < 1.0     # Total < 1 second
    )
    
    print(f"  Performance: {'✅ Good' if performance_good else '❌ Needs improvement'}")
    return performance_good

def main():
    """Run all wake word detector tests"""
    print("🎙️ Wake Word Detector Testing")
    print("=" * 50)
    
    setup_logging()
    
    # Note: We won't test audio functionality in this basic test
    # as it requires microphone access and real-time audio processing
    
    try:
        results = []
        
        # Test pattern matching
        results.append(test_wake_word_patterns())
        print()
        
        # Test confidence calculation
        results.append(test_confidence_calculation())
        print()
        
        # Test similarity matching
        results.append(test_similarity_matching())
        print()
        
        # Test configuration
        results.append(test_configuration())
        print()
        
        # Test status
        results.append(test_detector_status())
        print()
        
        # Test performance
        results.append(test_performance())
        print()
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("=" * 50)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! Wake word detector is working excellently.")
        elif passed >= total * 0.8:
            print("🟡 Most tests passed. Wake word detector is working well.")
        else:
            print("❌ Several tests failed. Wake word detector needs attention.")
        
        print("\n💡 Note: Audio streaming tests require microphone access")
        print("   Use the main application to test real-time wake word detection")
        
        return passed >= total * 0.7
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)