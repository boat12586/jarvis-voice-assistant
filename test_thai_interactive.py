#!/usr/bin/env python3
"""
Interactive Thai Voice System Test for JARVIS
ทดสอบระบบเสียงภาษาไทยแบบโต้ตอบสำหรับ JARVIS
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def setup_test_config():
    """Setup test configuration"""
    return {
        "command_parser": {
            "enabled": True,
            "confidence_threshold": 0.6
        },
        "conversation_memory": {
            "max_turns_per_session": 50,
            "context_window_size": 10,
            "memory_dir": "data/test_thai_conversation"
        },
        "thai_language": {
            "enabled": True
        }
    }

def test_thai_voice_system():
    """Test Thai voice system interactively"""
    print("🎯 JARVIS Thai Voice System Test")
    print("ทดสอบระบบเสียงภาษาไทย JARVIS")
    print("=" * 60)
    
    try:
        # Import components
        from voice.command_parser import VoiceCommandParser
        from features.thai_language_enhanced import ThaiLanguageProcessor
        from features.conversation_memory import ConversationMemorySystem
        
        # Initialize components
        config = setup_test_config()
        command_parser = VoiceCommandParser(config)
        thai_processor = ThaiLanguageProcessor(config)
        conversation_memory = ConversationMemorySystem(config)
        
        print("✅ Components initialized successfully!")
        print("✅ คอมโพเนนต์เริ่มต้นเรียบร้อยแล้ว!")
        
        # Start conversation session
        session_id = conversation_memory.start_session("thai_test_user", "th")
        print(f"✅ Started session: {session_id}")
        
        # Test cases in Thai
        thai_test_cases = [
            "สวัสดีครับ จาร์วิส",
            "ปัญญาประดิษฐ์คืออะไร",
            "ช่วยอธิบายเกี่ยวกับ machine learning หน่อย",
            "วันนี้อากาศเป็นอย่างไร",
            "ขอบคุณมากครับ",
            "เปิดเสียงให้ดังขึ้น",
            "กรุณาช่วยตั้งค่าระบบ"
        ]
        
        print("\n🎤 Testing Thai Voice Commands:")
        print("🎤 ทดสอบคำสั่งเสียงภาษาไทย:")
        print("=" * 60)
        
        for i, thai_text in enumerate(thai_test_cases, 1):
            print(f"\n{i}. Testing: '{thai_text}'")
            print(f"{i}. ทดสอบ: '{thai_text}'")
            print("-" * 40)
            
            try:
                # Process Thai text
                thai_result = thai_processor.process_thai_text(thai_text)
                enhanced = thai_processor.enhance_for_ai_processing(thai_text)
                
                print(f"📝 Processed: {thai_result.processed_text}")
                print(f"🌏 Language: {thai_result.language} (confidence: {thai_result.confidence:.2f})")
                
                if thai_result.cultural_context:
                    print(f"🎭 Cultural: {thai_result.cultural_context}")
                
                # Parse command
                parsed = command_parser.parse_command(thai_text, "th")
                print(f"🎯 Intent: {parsed.intent} (confidence: {parsed.confidence:.2f})")
                
                if parsed.entities:
                    print(f"📊 Entities: {parsed.entities}")
                
                # AI Enhancement
                if enhanced.get("ai_prompt_enhancement"):
                    print(f"🤖 AI Enhancement: {enhanced['ai_prompt_enhancement']}")
                
                # Simulate response based on intent
                if parsed.intent == "greeting":
                    response = "สวัสดีครับ! มีอะไรให้ช่วยไหมครับ"
                elif parsed.intent == "information_request":
                    response = f"ผมจะอธิบายเกี่ยวกับ {thai_result.processed_text} ให้ฟังครับ"
                elif parsed.intent == "system_control":
                    response = "ปรับระบบตามที่ต้องการแล้วครับ"
                elif parsed.intent == "action_request":
                    response = "ดำเนินการตามที่ขอแล้วครับ"
                else:
                    response = "เข้าใจแล้วครับ"
                
                print(f"💬 JARVIS Response: {response}")
                
                # Add to conversation memory
                turn_id = conversation_memory.add_conversation_turn(
                    user_input=thai_text,
                    user_language="th",
                    processed_input=thai_result.processed_text,
                    intent=parsed.intent,
                    entities=parsed.entities,
                    assistant_response=response,
                    response_language="th",
                    confidence=parsed.confidence
                )
                
                print(f"💾 Saved to memory: {turn_id}")
                
            except Exception as e:
                print(f"❌ Error processing '{thai_text}': {e}")
        
        # Show conversation summary
        print("\n" + "=" * 60)
        print("📊 Conversation Summary / สรุปการสนทนา")
        print("=" * 60)
        
        summary = conversation_memory.get_session_summary()
        print(f"🆔 Session ID: {summary['session_id']}")
        print(f"⏱️ Duration: {summary['duration_minutes']:.2f} minutes")
        print(f"💬 Total turns: {summary['turn_count']}")
        print(f"🌐 Languages: {summary['languages_used']}")
        print(f"📈 Average confidence: {summary['avg_confidence']:.2f}")
        print(f"🎯 Primary intents: {summary['primary_intents']}")
        
        if summary.get('user_preferences'):
            print(f"👤 User preferences: {summary['user_preferences']}")
        
        # Test context retrieval
        print("\n📚 Testing Context Retrieval:")
        context = conversation_memory.get_conversation_context("ปัญญาประดิษฐ์")
        print(f"Found {len(context)} relevant turns for 'ปัญญาประดิษฐ์'")
        
        # End session
        conversation_memory.end_session()
        print("\n✅ Session ended successfully!")
        print("✅ สิ้นสุดเซสชันเรียบร้อยแล้ว!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def show_component_status():
    """Show status of all voice components"""
    print("\n🔧 Component Status / สถานะคอมโพเนนต์")
    print("=" * 60)
    
    try:
        config = setup_test_config()
        
        # Test Thai processor
        from features.thai_language_enhanced import ThaiLanguageProcessor
        thai_processor = ThaiLanguageProcessor(config)
        thai_stats = thai_processor.get_thai_language_stats()
        
        print(f"✅ Thai Language Processor:")
        print(f"   📚 Dictionary: {thai_stats['dictionary_size']} entries")
        print(f"   🎭 Cultural patterns: {sum(thai_stats['cultural_patterns'].values())} items")
        print(f"   ⚙️ Features: {len(thai_stats['supported_features'])} capabilities")
        
        # Test command parser
        from voice.command_parser import VoiceCommandParser
        command_parser = VoiceCommandParser(config)
        parser_stats = command_parser.get_parser_stats()
        
        print(f"✅ Command Parser:")
        print(f"   🎯 Patterns: {parser_stats['command_patterns']} groups")
        print(f"   💭 Intents: {len(parser_stats['supported_intents'])} types")
        print(f"   🌐 Languages: {parser_stats['supported_languages']}")
        
        # Test conversation memory
        from features.conversation_memory import ConversationMemorySystem
        memory = ConversationMemorySystem(config)
        memory_stats = memory.get_memory_stats()
        
        print(f"✅ Conversation Memory:")
        print(f"   🧠 Embeddings: {memory_stats['embeddings_available']}")
        print(f"   🇹🇭 Thai support: {memory_stats['thai_support_available']}")
        print(f"   💾 Max turns: {memory_stats['max_turns_per_session']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Component status check failed: {e}")
        return False

if __name__ == "__main__":
    print("🚀 JARVIS Thai Voice System Interactive Test")
    print("🚀 การทดสอบระบบเสียงภาษาไทย JARVIS แบบโต้ตอบ")
    print("=" * 70)
    
    # Show component status first
    if show_component_status():
        print("\n🎯 Running interactive test...")
        print("🎯 เริ่มการทดสอบแบบโต้ตอบ...")
        
        if test_thai_voice_system():
            print("\n🎉 Test completed successfully!")
            print("🎉 การทดสอบเสร็จสิ้นเรียบร้อย!")
        else:
            print("\n❌ Test failed")
            print("❌ การทดสอบล้มเหลว")
    else:
        print("\n❌ Component initialization failed")
        print("❌ การเริ่มต้นคอมโพเนนต์ล้มเหลว")