#!/usr/bin/env python3
"""
Test script for Natural Language Understanding Engine
Tests advanced command interpretation and semantic analysis
"""

import sys
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from voice.command_parser import VoiceCommandParser
from ai.nlu_engine import NaturalLanguageUnderstanding, IntentCategory, ContextType
import yaml

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def load_test_config():
    """Load test configuration"""
    return {
        "command_parser": {
            "thai_support": True,
            "confidence_threshold": 0.5
        },
        "nlu_engine": {
            "enable_semantic_analysis": True,
            "max_context_turns": 10
        },
        "semantic_analyzer": {
            "complexity_threshold": 0.5
        },
        "context_manager": {
            "max_conversation_turns": 10
        }
    }

def test_semantic_analysis():
    """Test semantic analysis capabilities"""
    print("🧠 Testing Semantic Analysis...")
    
    config = load_test_config()
    nlu = NaturalLanguageUnderstanding(config)
    parser = VoiceCommandParser(config)
    
    # Test commands with different semantic complexity
    test_commands = [
        {
            "text": "Explain machine learning algorithms in detail",
            "expected_indicators": ["explanatory_query"],
            "expected_complexity": "medium"
        },
        {
            "text": "How does artificial intelligence compare to human intelligence?",
            "expected_indicators": ["comparative_analysis", "explanatory_query"],
            "expected_complexity": "high"
        },
        {
            "text": "Create a neural network for image recognition",
            "expected_indicators": [],
            "expected_complexity": "medium"
        },
        {
            "text": "อธิบายอัลกอริทึมการเรียนรู้ของเครื่องอย่างละเอียด",
            "expected_indicators": ["explanatory_query"],
            "expected_complexity": "medium"
        },
        {
            "text": "What causes overfitting in machine learning models and how can we prevent it?",
            "expected_indicators": ["causality", "explanatory_query"],
            "expected_complexity": "high"
        }
    ]
    
    results = []
    for test in test_commands:
        print(f"\n  Testing: '{test['text']}'")
        
        # Parse command first
        parsed = parser.parse_command(test["text"])
        
        # Process with NLU
        nlu_result = nlu.process_command(parsed)
        
        print(f"    Original Intent: {parsed.intent}")
        print(f"    Refined Intent: {nlu_result.refined_intent}")
        print(f"    Confidence: {nlu_result.intent_confidence:.3f}")
        print(f"    Complexity Score: {nlu_result.complexity_score:.3f}")
        print(f"    Processing Time: {nlu_result.processing_time*1000:.1f}ms")
        
        # Check semantic indicators
        semantic_indicators = nlu_result.semantic_understanding.get("semantic_indicators", [])
        found_indicators = [ind["type"] for ind in semantic_indicators]
        print(f"    Semantic Indicators: {found_indicators}")
        
        # Check complexity classification
        if nlu_result.complexity_score < 0.3:
            complexity_class = "low"
        elif nlu_result.complexity_score < 0.7:
            complexity_class = "medium"
        else:
            complexity_class = "high"
        print(f"    Complexity Class: {complexity_class}")
        
        # Context requirements
        print(f"    Context Required: {[ct.value for ct in nlu_result.context_requirements]}")
        
        # Response suggestions
        print(f"    Suggestions: {nlu_result.suggested_responses[:2]}")
        
        # Check if expectations met
        expectations_met = (
            complexity_class == test["expected_complexity"] and
            all(expected in found_indicators for expected in test["expected_indicators"])
        )
        
        results.append(expectations_met)
        print(f"    Expectations Met: {'✅' if expectations_met else '❌'}")
    
    accuracy = sum(results) / len(results) * 100
    print(f"\n📊 Semantic Analysis Accuracy: {accuracy:.1f}% ({sum(results)}/{len(results)})")
    return accuracy > 70

def test_intent_refinement():
    """Test intent refinement capabilities"""
    print("🎯 Testing Intent Refinement...")
    
    config = load_test_config()
    nlu = NaturalLanguageUnderstanding(config)
    parser = VoiceCommandParser(config)
    
    # Test commands that should be refined to more specific intents
    test_cases = [
        {
            "text": "Explain artificial intelligence algorithms in comprehensive detail",
            "original_expected": "information_request",
            "refined_expected": "deep_information_request"
        },
        {
            "text": "Guide me through building a machine learning model step by step",
            "original_expected": "how_to_request",
            "refined_expected": "procedural_guidance"
        },
        {
            "text": "Analyze the performance of different neural network architectures",
            "original_expected": "information_request",
            "refined_expected": "analytical_request"
        },
        {
            "text": "Create a Python script for data visualization",
            "original_expected": "action_request",
            "refined_expected": "creative_request"
        },
        {
            "text": "Tell me more about that concept you mentioned",
            "original_expected": "information_request",
            "refined_expected": "contextual_continuation"
        }
    ]
    
    results = []
    for test in test_cases:
        print(f"\n  Testing: '{test['text']}'")
        
        # Parse with basic parser
        parsed = parser.parse_command(test["text"])
        print(f"    Original Intent: {parsed.intent}")
        
        # Process with NLU
        nlu_result = nlu.process_command(parsed)
        print(f"    Refined Intent: {nlu_result.refined_intent}")
        print(f"    Confidence Improvement: {parsed.confidence:.3f} → {nlu_result.intent_confidence:.3f}")
        
        # Check if refinement worked
        refinement_success = (
            parsed.intent == test["original_expected"] and
            nlu_result.refined_intent == test["refined_expected"]
        )
        
        results.append(refinement_success)
        print(f"    Refinement Success: {'✅' if refinement_success else '❌'}")
    
    success_rate = sum(results) / len(results) * 100
    print(f"\n📊 Intent Refinement Success: {success_rate:.1f}% ({sum(results)}/{len(results)})")
    return success_rate > 60

def test_context_management():
    """Test context management and conversation tracking"""
    print("💭 Testing Context Management...")
    
    config = load_test_config()
    nlu = NaturalLanguageUnderstanding(config)
    parser = VoiceCommandParser(config)
    
    # Simulate conversation flow
    conversation = [
        ("What is machine learning?", "Machine learning is a subset of AI..."),
        ("How does it work?", "Machine learning works by training algorithms..."),
        ("Can you give me an example?", "A common example is image recognition..."),
        ("Tell me more about neural networks", "Neural networks are inspired by the brain..."),
        ("How do they compare to traditional algorithms?", "Neural networks differ from traditional algorithms...")
    ]
    
    print("  Building conversation context...")
    for i, (user_input, assistant_response) in enumerate(conversation):
        print(f"    Turn {i+1}: '{user_input}'")
        
        # Parse command
        parsed = parser.parse_command(user_input)
        
        # Process with NLU
        nlu_result = nlu.process_command(parsed)
        
        # Add to conversation context
        nlu.add_conversation_context(user_input, assistant_response, {
            "intent": nlu_result.refined_intent,
            "confidence": nlu_result.intent_confidence,
            "complexity": nlu_result.complexity_score
        })
        
        print(f"      Intent: {nlu_result.refined_intent}")
        print(f"      Context Required: {[ct.value for ct in nlu_result.context_requirements]}")
        print(f"      Requires Followup: {nlu_result.requires_followup}")
    
    # Test contextual continuation
    print("\n  Testing contextual continuation...")
    continuation_commands = [
        "Tell me more about that",
        "Can you elaborate?",
        "What else should I know?",
        "อธิบายเพิ่มเติมหน่อย"
    ]
    
    context_results = []
    for cmd in continuation_commands:
        parsed = parser.parse_command(cmd)
        nlu_result = nlu.process_command(parsed)
        
        # Check if contextual continuation was detected
        has_context = ContextType.CONVERSATION in nlu_result.context_requirements
        context_results.append(has_context)
        
        print(f"    '{cmd}' → Contextual: {'✅' if has_context else '❌'}")
    
    # Get NLU statistics
    stats = nlu.get_nlu_stats()
    print(f"\n  Context turns stored: {stats['context_turns']}")
    print(f"  Commands processed: {stats['processing_stats']['commands_processed']}")
    
    context_success = sum(context_results) / len(context_results) * 100
    print(f"\n📊 Context Detection Success: {context_success:.1f}%")
    return context_success > 50

def test_multilingual_support():
    """Test multilingual NLU capabilities"""
    print("🌐 Testing Multilingual Support...")
    
    config = load_test_config()
    nlu = NaturalLanguageUnderstanding(config)
    parser = VoiceCommandParser(config)
    
    # Test commands in both languages
    test_pairs = [
        {
            "english": "Explain artificial intelligence in detail",
            "thai": "อธิบายปัญญาประดิษฐ์อย่างละเอียด",
            "expected_intent": "deep_information_request"
        },
        {
            "english": "How do I create a neural network?",
            "thai": "สร้างโครงข่ายประสาทเทียมยังไง",
            "expected_intent": "procedural_guidance"
        },
        {
            "english": "Analyze this data for patterns",
            "thai": "วิเคราะห์ข้อมูลนี้หาแนวโน้ม",
            "expected_intent": "analytical_request"
        }
    ]
    
    results = []
    for pair in test_pairs:
        print(f"\n  Testing language pair:")
        
        # Test English
        en_parsed = parser.parse_command(pair["english"])
        en_result = nlu.process_command(en_parsed)
        print(f"    EN: '{pair['english']}'")
        print(f"        Intent: {en_result.refined_intent}, Confidence: {en_result.intent_confidence:.3f}")
        
        # Test Thai
        th_parsed = parser.parse_command(pair["thai"])
        th_result = nlu.process_command(th_parsed)
        print(f"    TH: '{pair['thai']}'")
        print(f"        Intent: {th_result.refined_intent}, Confidence: {th_result.intent_confidence:.3f}")
        
        # Check consistency
        consistent = (
            en_result.refined_intent == th_result.refined_intent or
            abs(en_result.intent_confidence - th_result.intent_confidence) < 0.3
        )
        
        results.append(consistent)
        print(f"    Consistency: {'✅' if consistent else '❌'}")
    
    consistency_rate = sum(results) / len(results) * 100
    print(f"\n📊 Multilingual Consistency: {consistency_rate:.1f}%")
    return consistency_rate > 60

def test_performance():
    """Test NLU performance and efficiency"""
    print("⚡ Testing Performance...")
    
    config = load_test_config()
    nlu = NaturalLanguageUnderstanding(config)
    parser = VoiceCommandParser(config)
    
    # Test commands of varying complexity
    test_commands = [
        "Hello JARVIS",  # Simple
        "What is machine learning?",  # Medium
        "Explain the mathematical foundations of deep learning neural networks including backpropagation algorithms and optimization techniques in comprehensive detail",  # Complex
    ] * 10  # 30 commands total
    
    processing_times = []
    
    print("  Processing 30 commands...")
    start_time = time.time()
    
    for i, cmd in enumerate(test_commands):
        parsed = parser.parse_command(cmd)
        nlu_result = nlu.process_command(parsed)
        processing_times.append(nlu_result.processing_time)
        
        if i % 10 == 9:  # Every 10 commands
            print(f"    Processed {i+1}/30 commands")
    
    total_time = time.time() - start_time
    avg_time = sum(processing_times) / len(processing_times)
    max_time = max(processing_times)
    min_time = min(processing_times)
    
    print(f"  Total processing time: {total_time:.3f}s")
    print(f"  Average per command: {avg_time*1000:.1f}ms")
    print(f"  Range: {min_time*1000:.1f}ms - {max_time*1000:.1f}ms")
    
    # Get final statistics
    stats = nlu.get_nlu_stats()
    print(f"  Commands processed: {stats['processing_stats']['commands_processed']}")
    print(f"  Success rate: {stats['processing_stats']['successful_interpretations']}/{stats['processing_stats']['commands_processed']}")
    print(f"  Complexity distribution: {stats['processing_stats']['complexity_distribution']}")
    
    # Performance criteria
    performance_good = (
        avg_time < 0.1 and  # Average < 100ms
        max_time < 0.5 and  # Max < 500ms
        stats['processing_stats']['successful_interpretations'] == stats['processing_stats']['commands_processed']
    )
    
    print(f"  Performance: {'✅ Good' if performance_good else '❌ Needs improvement'}")
    return performance_good

def main():
    """Run all NLU engine tests"""
    print("🧠 Natural Language Understanding Engine Testing")
    print("=" * 60)
    
    setup_logging()
    
    try:
        results = []
        
        # Test semantic analysis
        results.append(test_semantic_analysis())
        print()
        
        # Test intent refinement
        results.append(test_intent_refinement())
        print()
        
        # Test context management
        results.append(test_context_management())
        print()
        
        # Test multilingual support
        results.append(test_multilingual_support())
        print()
        
        # Test performance
        results.append(test_performance())
        print()
        
        # Summary
        passed = sum(results)
        total = len(results)
        
        print("=" * 60)
        print(f"🎯 Test Results: {passed}/{total} tests passed")
        
        if passed == total:
            print("✅ All tests passed! NLU engine is working excellently.")
        elif passed >= total * 0.8:
            print("🟡 Most tests passed. NLU engine is working well with minor issues.")
        else:
            print("❌ Several tests failed. NLU engine needs improvement.")
        
        print("\n🎉 Natural Language Understanding Engine testing completed!")
        return passed >= total * 0.6
        
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)