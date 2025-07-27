#!/usr/bin/env python3
"""
🎤 JARVIS Voice Demo
สาธิตการทำงานของระบบเสียง JARVIS

Version: 2.0.0 (2025 Edition)
"""

import sys
import logging
from pathlib import Path

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def setup_demo_logging():
    """ตั้งค่า logging แบบง่าย"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s - %(message)s'
    )

def voice_demo():
    """สาธิตระบบเสียง JARVIS"""
    print("🎤 JARVIS Voice Demo")
    print("=" * 40)
    
    try:
        from voice import VoiceController
        
        # Initialize voice controller
        print("🔧 Initializing JARVIS Voice System...")
        voice = VoiceController(use_fallback_tts=True)
        
        # Show system status
        status = voice.get_status()
        print(f"\n📊 Voice System Status:")
        for key, value in status.items():
            emoji = "✅" if value else "❌"
            print(f"   {emoji} {key}: {value}")
        
        print(f"\n🎙️ Voice System Demo")
        print("=" * 40)
        
        # Test basic commands
        test_commands = [
            "สวัสดี",
            "ชื่อ",
            "เวลา", 
            "วันที่",
            "ขอบคุณ"
        ]
        
        print("🗣️  Testing voice commands:")
        for i, command in enumerate(test_commands, 1):
            print(f"\n{i}. Testing command: '{command}'")
            response = voice.process_voice_command(command)
            print(f"   🤖 JARVIS Response: {response}")
            
            # Speak the response
            voice.speak_response(response)
            
            # Small delay between commands
            import time
            time.sleep(1)
        
        # Test English commands
        print(f"\n🌍 Testing English commands:")
        english_commands = ["hello", "name", "time"]
        
        for i, command in enumerate(english_commands, 1):
            print(f"\n{i}. Testing English: '{command}'")
            response = voice.process_voice_command(command)
            print(f"   🤖 JARVIS Response: {response}")
            voice.speak_response(response)
            import time
            time.sleep(1)
        
        print(f"\n🎉 Voice Demo Complete!")
        print("✨ JARVIS Voice System is fully operational!")
        
        # Show final system info
        print(f"\n📋 System Summary:")
        print(f"   🎤 Speech Recognition: {'Ready' if status['speech_recognition'] else 'Not Available'}")
        print(f"   🗣️  Text-to-Speech: {'Ready' if status['text_to_speech'] else 'Not Available'}")
        print(f"   🌐 Language: {status['language']}")
        print(f"   🎯 Wake Word: {status['wake_word']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Voice demo failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def interactive_demo():
    """สาธิตแบบโต้ตอบ (ถ้ามี microphone)"""
    print(f"\n🎙️ Interactive Voice Demo")
    print("=" * 40)
    
    try:
        from voice import VoiceController
        
        voice = VoiceController(use_fallback_tts=True)
        
        if voice.get_status()['speech_recognition']:
            print("🎤 Microphone test available!")
            print("ℹ️  For full microphone testing, run: python3 run_jarvis_voice.py")
            
            # Test if we can initialize speech recognizer
            from voice import SimpleSpeechRecognizer
            recognizer = SimpleSpeechRecognizer()
            
            if recognizer.is_available():
                print("✅ Speech recognizer ready for microphone input")
                print("🔊 Audio devices available:")
                devices = recognizer.get_audio_devices()
                if devices:
                    for i, device in enumerate(devices[:3]):  # Show first 3
                        print(f"   {i+1}. {device.get('name', 'Unknown Device')}")
                else:
                    print("   No audio devices found")
            else:
                print("⚠️  Speech recognizer not available")
        else:
            print("ℹ️  Speech recognition not available in current environment")
            
        return True
        
    except Exception as e:
        print(f"❌ Interactive demo setup failed: {e}")
        return False

def main():
    """รันการสาธิต"""
    setup_demo_logging()
    
    print("🤖 JARVIS Voice Assistant v2.0 - Demo")
    print("=" * 50)
    
    success = True
    
    # Run voice demo
    if not voice_demo():
        success = False
    
    # Run interactive demo
    if not interactive_demo():
        success = False
    
    if success:
        print(f"\n🎉 All demos completed successfully!")
        print(f"🚀 Ready to run full JARVIS Voice Assistant:")
        print(f"   python3 run_jarvis_voice.py")
        return 0
    else:
        print(f"\n⚠️  Some demos failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())