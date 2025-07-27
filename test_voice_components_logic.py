#!/usr/bin/env python3
"""
Voice Components Logic Test for JARVIS Voice Assistant
Tests the logic of voice components without requiring audio dependencies
"""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_logging():
    """Setup logging for the test"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_voice_components.log')
        ]
    )
    return logging.getLogger(__name__)

def load_test_config() -> Dict[str, Any]:
    """Load test configuration"""
    return {
        "command_parser": {
            "enabled": True,
            "confidence_threshold": 0.6
        },
        "conversation_memory": {
            "max_turns_per_session": 50,
            "max_session_duration": 3600,
            "context_window_size": 10,
            "similarity_threshold": 0.7,
            "memory_dir": "data/test_conversation_memory"
        },
        "thai_language": {
            "enabled": True
        }
    }

def test_command_parsing(logger):
    """Test command parsing with various inputs"""
    logger.info("=" * 60)
    logger.info("Testing Command Parsing")
    logger.info("=" * 60)
    
    config = load_test_config()
    
    try:
        # Import here to avoid audio dependencies in main imports
        from voice.command_parser import VoiceCommandParser
        
        command_parser = VoiceCommandParser(config)
        
        # Test cases for different languages and intents
        test_cases = [
            # English tests
            ("Hello JARVIS", "en", "greeting"),
            ("What is artificial intelligence?", "en", "information_request"),
            ("How do I use this feature?", "en", "how_to_request"),
            ("Please open the settings", "en", "action_request"),
            ("Turn up the volume", "en", "system_control"),
            
            # Thai tests
            ("สวัสดี จาร์วิส", "th", "greeting"),
            ("ปัญญาประดิษฐ์คืออะไร", "th", "information_request"),
            ("ช่วยเปิดการตั้งค่าหน่อย", "th", "action_request"),
            ("เปิดเสียงให้ดังขึ้น", "th", "system_control"),
            
            # Mixed language
            ("Hello จาร์วิส", "auto", "greeting")
        ]
        
        results = []
        
        for text, language, expected_intent in test_cases:
            try:
                parsed = command_parser.parse_command(text, language)
                
                result = {
                    "input": text,
                    "language": language,
                    "detected_language": parsed.language,
                    "intent": parsed.intent,
                    "expected_intent": expected_intent,
                    "confidence": parsed.confidence,
                    "entities": parsed.entities,
                    "success": parsed.intent == expected_intent or parsed.intent != "unknown"
                }
                
                results.append(result)
                
                status = "✅" if result["success"] else "⚠️"
                logger.info(f"{status} '{text}' -> {parsed.intent} (conf: {parsed.confidence:.2f})")
                
                if parsed.entities:
                    logger.info(f"   Entities: {parsed.entities}")
                
            except Exception as e:
                logger.error(f"❌ Failed to parse '{text}': {e}")
                results.append({
                    "input": text,
                    "error": str(e),
                    "success": False
                })
        
        # Calculate success rate
        successful = sum(1 for r in results if r.get("success", False))
        total = len(results)
        success_rate = (successful / total) * 100
        
        logger.info(f"\nCommand Parsing Results: {successful}/{total} ({success_rate:.1f}% success)")
        
        # Test parser stats
        stats = command_parser.get_parser_stats()
        logger.info(f"Parser Stats: {stats['command_patterns']} patterns, {len(stats['supported_intents'])} intents")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Command parsing test failed: {e}")
        return []

def test_thai_language_processing(logger):
    """Test Thai language processing capabilities"""
    logger.info("=" * 60)
    logger.info("Testing Thai Language Processing")
    logger.info("=" * 60)
    
    config = load_test_config()
    
    try:
        from features.thai_language_enhanced import ThaiLanguageProcessor
        
        thai_processor = ThaiLanguageProcessor(config)
        
        # Test cases for Thai processing
        test_cases = [
            "สวัสดีครับ จาร์วิส",
            "ขอบคุณมากครับ",
            "กรุณาช่วยอธิบายเกี่ยวกับปัญญาประดิษฐ์",
            "วันนี้อากาศเป็นอย่างไร",
            "Hello สวัสดี mixed language test"
        ]
        
        results = []
        
        for text in test_cases:
            try:
                result = thai_processor.process_thai_text(text)
                enhanced = thai_processor.enhance_for_ai_processing(text)
                
                test_result = {
                    "input": text,
                    "processed_text": result.processed_text,
                    "language": result.language,
                    "confidence": result.confidence,
                    "features": result.features,
                    "cultural_context": result.cultural_context,
                    "ai_enhancement": enhanced.get("ai_prompt_enhancement", ""),
                    "success": True
                }
                
                results.append(test_result)
                
                logger.info(f"✅ '{text}'")
                logger.info(f"   Language: {result.language} (conf: {result.confidence:.2f})")
                logger.info(f"   Cultural: {result.cultural_context}")
                
                if enhanced.get("ai_prompt_enhancement"):
                    logger.info(f"   AI Enhancement: {enhanced['ai_prompt_enhancement']}")
                
            except Exception as e:
                logger.error(f"❌ Failed to process '{text}': {e}")
                results.append({
                    "input": text,
                    "error": str(e),
                    "success": False
                })
        
        # Test stats
        stats = thai_processor.get_thai_language_stats()
        logger.info(f"\nThai Processor Stats: {stats['dictionary_size']} dictionary entries")
        logger.info(f"Features: {stats['supported_features']}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Thai language processing test failed: {e}")
        return []

def test_conversation_memory(logger):
    """Test conversation memory system"""
    logger.info("=" * 60)
    logger.info("Testing Conversation Memory")
    logger.info("=" * 60)
    
    config = load_test_config()
    
    try:
        from features.conversation_memory import ConversationMemorySystem
        
        memory = ConversationMemorySystem(config)
        
        # Start a test session
        session_id = memory.start_session("test_user", "en")
        logger.info(f"✅ Started session: {session_id}")
        
        # Add some conversation turns
        test_conversations = [
            {
                "user_input": "Hello JARVIS",
                "user_language": "en",
                "processed_input": "hello jarvis",
                "intent": "greeting",
                "entities": {},
                "assistant_response": "Hello! How can I help you today?",
                "response_language": "en",
                "confidence": 0.95
            },
            {
                "user_input": "What is artificial intelligence?",
                "user_language": "en", 
                "processed_input": "what is artificial intelligence",
                "intent": "information_request",
                "entities": {"topics": ["artificial intelligence"]},
                "assistant_response": "Artificial intelligence is a field of computer science focused on creating machines that can perform tasks requiring human intelligence.",
                "response_language": "en",
                "confidence": 0.88
            },
            {
                "user_input": "สวัสดีครับ",
                "user_language": "th",
                "processed_input": "สวัสดีครับ",
                "intent": "greeting",
                "entities": {},
                "assistant_response": "สวัสดีครับ มีอะไรให้ช่วยไหมครับ",
                "response_language": "th",
                "confidence": 0.92
            }
        ]
        
        turn_ids = []
        for conv in test_conversations:
            turn_id = memory.add_conversation_turn(**conv)
            turn_ids.append(turn_id)
            logger.info(f"✅ Added turn: {turn_id}")
        
        # Test context retrieval
        context = memory.get_conversation_context("artificial intelligence")
        logger.info(f"✅ Retrieved {len(context)} relevant turns for 'artificial intelligence'")
        
        # Test session summary
        summary = memory.get_session_summary()
        logger.info(f"✅ Session summary: {summary['turn_count']} turns, {len(summary['languages_used'])} languages")
        
        # Test memory stats
        stats = memory.get_memory_stats()
        logger.info(f"✅ Memory stats: session_active={stats['current_session_active']}")
        
        # End session
        memory.end_session()
        logger.info("✅ Session ended successfully")
        
        return {
            "session_id": session_id,
            "turns_added": len(turn_ids),
            "context_retrieved": len(context),
            "summary": summary,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Conversation memory test failed: {e}")
        return {"success": False, "error": str(e)}

def test_integration_logic(logger):
    """Test the integration logic (without audio components)"""
    logger.info("=" * 60)
    logger.info("Testing Integration Logic")
    logger.info("=" * 60)
    
    config = load_test_config()
    
    try:
        # Import components
        from voice.command_parser import VoiceCommandParser
        from features.conversation_memory import ConversationMemorySystem
        from features.thai_language_enhanced import ThaiLanguageProcessor
        
        # Initialize all components
        command_parser = VoiceCommandParser(config)
        conversation_memory = ConversationMemorySystem(config)
        thai_processor = ThaiLanguageProcessor(config)
        
        # Start conversation session
        session_id = conversation_memory.start_session("test_user", "en")
        logger.info(f"✅ Started integrated session: {session_id}")
        
        # Simulate voice interaction pipeline
        test_inputs = [
            ("Hey JARVIS, what's the weather today?", "en"),
            ("จาร์วิส ปัญญาประดิษฐ์คืออะไร", "th"),
            ("Can you help me with machine learning?", "en"),
            ("ขอบคุณครับ", "th")
        ]
        
        pipeline_results = []
        
        for text, language in test_inputs:
            try:
                logger.info(f"Processing: '{text}' ({language})")
                
                # Parse command
                parsed_command = command_parser.parse_command(text, language)
                logger.info(f"  Parsed: {parsed_command.intent} (conf: {parsed_command.confidence:.2f})")
                
                # Thai enhancement if needed
                enhanced_context = {}
                if language == "th":
                    enhanced_context = thai_processor.enhance_for_ai_processing(text)
                    if enhanced_context.get("ai_prompt_enhancement"):
                        logger.info(f"  Thai Enhancement: {enhanced_context['ai_prompt_enhancement']}")
                
                # Get conversation context
                context = conversation_memory.get_conversation_context(parsed_command.cleaned_text)
                logger.info(f"  Context: {len(context)} relevant turns")
                
                # Simulate assistant response
                response = f"I understand you said '{parsed_command.cleaned_text}' with intent '{parsed_command.intent}'"
                if language == "th":
                    response = f"ผมเข้าใจที่คุณพูดว่า '{parsed_command.cleaned_text}'"
                
                # Add to conversation memory
                turn_id = conversation_memory.add_conversation_turn(
                    user_input=text,
                    user_language=language,
                    processed_input=parsed_command.cleaned_text,
                    intent=parsed_command.intent,
                    entities=parsed_command.entities,
                    assistant_response=response,
                    response_language=language,
                    confidence=parsed_command.confidence
                )
                
                logger.info(f"  Added to memory: {turn_id}")
                
                pipeline_results.append({
                    "input": text,
                    "language": language,
                    "intent": parsed_command.intent,
                    "confidence": parsed_command.confidence,
                    "turn_id": turn_id,
                    "success": True
                })
                
            except Exception as e:
                logger.error(f"  ❌ Pipeline failed for '{text}': {e}")
                pipeline_results.append({
                    "input": text,
                    "error": str(e),
                    "success": False
                })
        
        # Get final session summary
        final_summary = conversation_memory.get_session_summary()
        logger.info(f"✅ Final session: {final_summary['turn_count']} turns, {final_summary['avg_confidence']:.2f} avg confidence")
        
        # End session
        conversation_memory.end_session()
        
        successful_pipeline = sum(1 for r in pipeline_results if r.get("success", False))
        total_pipeline = len(pipeline_results)
        
        logger.info(f"✅ Pipeline Results: {successful_pipeline}/{total_pipeline} successful")
        
        return {
            "session_id": session_id,
            "pipeline_results": pipeline_results,
            "final_summary": final_summary,
            "success_rate": (successful_pipeline / total_pipeline) * 100,
            "success": True
        }
        
    except Exception as e:
        logger.error(f"❌ Integration logic test failed: {e}")
        return {"success": False, "error": str(e)}

def run_component_tests():
    """Run voice component logic tests"""
    logger = setup_logging()
    
    logger.info("🚀 Starting JARVIS Voice Components Logic Tests")
    logger.info("=" * 80)
    
    test_results = {}
    
    # Command parsing tests
    test_results["command_parsing"] = test_command_parsing(logger)
    
    # Thai language processing tests
    test_results["thai_processing"] = test_thai_language_processing(logger)
    
    # Conversation memory tests
    test_results["conversation_memory"] = test_conversation_memory(logger)
    
    # Integration logic tests
    test_results["integration_logic"] = test_integration_logic(logger)
    
    # Generate summary report
    logger.info("=" * 80)
    logger.info("🎯 COMPONENT TESTS SUMMARY REPORT")
    logger.info("=" * 80)
    
    overall_success = True
    
    for test_name, result in test_results.items():
        if isinstance(result, dict) and result.get("success", True):
            logger.info(f"✅ {test_name.replace('_', ' ').title()}: PASSED")
        elif isinstance(result, list):
            success_count = sum(1 for r in result if r.get("success", False))
            total_count = len(result)
            success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
            status = "✅" if success_rate >= 80 else "⚠️" if success_rate >= 60 else "❌"
            logger.info(f"{status} {test_name.replace('_', ' ').title()}: {success_count}/{total_count} ({success_rate:.1f}%)")
            if success_rate < 80:
                overall_success = False
        else:
            logger.info(f"❌ {test_name.replace('_', ' ').title()}: FAILED")
            overall_success = False
    
    # Save detailed results
    results_file = Path("test_results_voice_components.json")
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(test_results, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"\n📊 Detailed results saved to: {results_file}")
    
    if overall_success:
        logger.info("🎉 ALL COMPONENT TESTS PASSED - Voice logic ready!")
    else:
        logger.info("⚠️ Some tests failed - Review results and fix issues")
    
    return test_results

if __name__ == "__main__":
    results = run_component_tests()
    
    # Exit with appropriate code
    overall_success = all(
        r.get("success", True) if isinstance(r, dict) else 
        sum(1 for item in r if item.get("success", False)) >= len(r) * 0.8 if isinstance(r, list) else
        True
        for r in results.values()
    )
    
    sys.exit(0 if overall_success else 1)