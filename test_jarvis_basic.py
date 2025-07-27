#!/usr/bin/env python3
"""
🧪 Test JARVIS Basic Functionality
ทดสอบการทำงานพื้นฐานของ JARVIS
"""

import logging
from jarvis_simple_working import JarvisSimpleWorking

def test_basic_functionality():
    """ทดสอบการทำงานพื้นฐาน"""
    
    # ตั้งค่า logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("🧪 Testing JARVIS Basic Functionality")
    print("=====================================")
    
    # สร้าง JARVIS
    jarvis = JarvisSimpleWorking()
    
    # เริ่มระบบ
    print("\n1. Starting system...")
    if jarvis.start_system():
        print("✅ System started successfully!")
    else:
        print("❌ Failed to start system")
        return
    
    # ทดสอบการตอบสนองข้อความ
    print("\n2. Testing text responses...")
    test_messages = [
        ("Hello JARVIS", "en"),
        ("What time is it?", "en"),
        ("สวัสดีครับ", "th"),
        ("ช่วยบอกเวลาหน่อยครับ", "th"),
        ("Goodbye", "en")
    ]
    
    for message, language in test_messages:
        print(f"\nTesting: '{message}' ({language})")
        jarvis.process_text_message(message, language)
    
    # แสดงสถานะ
    print("\n3. System status:")
    status = jarvis.get_status()
    print(f"   Active: {status['active']}")
    print(f"   Interactions: {status['interactions']}")
    print(f"   Uptime: {status['uptime_seconds']:.1f} seconds")
    
    # หยุดระบบ
    print("\n4. Stopping system...")
    jarvis.stop_system()
    print("✅ System stopped")
    
    print("\n🎉 Basic functionality test completed!")

if __name__ == "__main__":
    test_basic_functionality()