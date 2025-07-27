#!/usr/bin/env python3
"""
🧪 Test JARVIS Final Complete System
ทดสอบระบบ JARVIS ที่สมบูรณ์
"""

import logging
import time
from jarvis_final_complete import JarvisFinalComplete

def test_complete_system():
    """ทดสอบระบบที่สมบูรณ์"""
    
    # ตั้งค่า logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🧪 Testing JARVIS Final Complete System")
    print("=======================================")
    
    # สร้าง JARVIS
    jarvis = JarvisFinalComplete()
    
    # เริ่มระบบ
    print("\n1. 🚀 Starting complete system...")
    if jarvis.start_system():
        print("✅ Complete system started successfully!")
    else:
        print("❌ Failed to start complete system")
        return
    
    # ทดสอบการตอบสนองข้อความพื้นฐาน
    print("\n2. 💬 Testing basic text responses...")
    basic_tests = [
        "Hello JARVIS",
        "What time is it?",
        "What's the date?",
        "สวัสดีครับ JARVIS",
        "ช่วยบอกเวลาหน่อย",
        "My name is Alex",
        "Thank you JARVIS",
        "Tell me about your memory"
    ]
    
    for i, message in enumerate(basic_tests, 1):
        print(f"\n   Test {i}: '{message}'")
        jarvis.process_text_message(message)
        time.sleep(1)  # Brief pause
    
    # ทดสอบหน่วยความจำ
    print("\n3. 🧠 Testing memory system...")
    print("   Adding user profile information...")
    jarvis.memory.update_user_profile("age", 25)
    jarvis.memory.update_user_profile("location", "Bangkok")
    jarvis.memory.update_preferences("language", "th")
    
    # แสดงข้อมูลหน่วยความจำ
    recent_conversations = jarvis.memory.get_recent_context(3)
    print(f"   Recent conversations: {len(recent_conversations)}")
    print(f"   User profile: {jarvis.memory.user_profile}")
    print(f"   Preferences: {jarvis.memory.preferences}")
    
    # ทดสอบการตอบสนองที่มีบริบท
    print("\n4. 🎯 Testing contextual responses...")
    contextual_tests = [
        "Do you remember my name?",
        "What did we talk about earlier?",
        "ผมชื่ออะไรครับ"
    ]
    
    for message in contextual_tests:
        print(f"\n   Context test: '{message}'")
        jarvis.process_text_message(message)
        time.sleep(1)
    
    # แสดงสถานะระบบ
    print("\n5. 📊 Final system status:")
    status = jarvis.get_status()
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for sub_key, sub_value in value.items():
                print(f"     {sub_key}: {sub_value}")
        else:
            print(f"   {key}: {value}")
    
    # ทดสอบฟีเจอร์เพิ่มเติม
    print("\n6. 🎛️ Testing additional features...")
    
    # Wake word detector status
    if jarvis.wake_word_detector.available:
        print("   ✅ Wake word detection is available")
    else:
        print("   ❌ Wake word detection not available")
    
    # Voice system status
    if jarvis.voice_available:
        print("   ✅ Voice system is available")
    else:
        print("   ❌ Voice system not available")
    
    # Memory persistence test
    print("   💾 Testing memory persistence...")
    original_count = len(jarvis.memory.conversations)
    jarvis.memory._save_memory()
    print(f"   Saved {original_count} conversations to disk")
    
    # หยุดระบบ
    print("\n7. 🛑 Stopping complete system...")
    jarvis.stop_system()
    print("✅ Complete system stopped")
    
    print("\n🎉 JARVIS Final Complete System test finished!")
    print("\nSummary:")
    print(f"   💬 Total interactions: {status['statistics']['total_interactions']}")
    print(f"   🧠 Conversations in memory: {status['memory_stats']['conversations']}")
    print(f"   👤 User profile items: {status['memory_stats']['user_profile']}")
    print(f"   ⚙️ User preferences: {status['memory_stats']['preferences']}")
    print(f"   ⏱️ Uptime: {status['uptime_seconds']:.1f} seconds")


if __name__ == "__main__":
    test_complete_system()