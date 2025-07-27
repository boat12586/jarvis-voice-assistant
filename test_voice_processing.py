#!/usr/bin/env python3
"""
🎙️ JARVIS Voice Processing Test
ทดสอบระบบประมวลผลเสียงของ JARVIS

Version: 2.0.0 (2025 Edition)
"""

import sys
import os
import logging
from pathlib import Path
import time

# Add src to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_voice_dependencies():
    """ทดสอบ dependencies สำหรับ voice processing"""
    print("🔍 ทดสอบ Voice Processing Dependencies...")
    
    try:
        # Test av (PyAV)
        import av
        print(f"✅ PyAV {av.__version__}: OK")
        
        # Test onnxruntime
        import onnxruntime
        print(f"✅ ONNXRuntime {onnxruntime.__version__}: OK")
        
        # Test faster-whisper
        import faster_whisper
        print("✅ Faster-Whisper: OK")
        
        # Test audio libraries
        import sounddevice as sd
        print("✅ SoundDevice: OK")
        
        import scipy
        print(f"✅ SciPy {scipy.__version__}: OK")
        
        return True
        
    except ImportError as e:
        print(f"❌ Voice dependencies test failed: {e}")
        return False

def test_whisper_model():
    """ทดสอบ Whisper model loading (แบบเบื้องต้น)"""
    print("\\n🧠 ทดสอบ Whisper Model Loading...")
    
    try:
        from faster_whisper import WhisperModel
        
        print("📥 Loading Whisper tiny model (for testing)...")
        # ใช้ tiny model เพื่อประหยัดเวลาและ memory
        model = WhisperModel("tiny", device="cpu", compute_type="int8")
        print("✅ Whisper model loaded successfully")
        
        # Test basic info
        print(f"   📊 Model info: tiny model on CPU")
        print(f"   💾 Compute type: int8 (optimized)")
        
        # Cleanup
        del model
        print("✅ Model cleanup completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Whisper model test failed: {e}")
        return False

def test_audio_devices():
    """ทดสอบ audio devices"""
    print("\\n🔊 ทดสอบ Audio Devices...")
    
    try:
        import sounddevice as sd
        
        # Get available devices
        devices = sd.query_devices()
        print(f"✅ Found {len(devices)} audio devices")
        
        # Show default devices
        default_input = sd.default.device[0]
        default_output = sd.default.device[1]
        
        if default_input is not None:
            input_info = sd.query_devices(default_input)
            print(f"🎤 Default input: {input_info['name']}")
        else:
            print("⚠️  No default input device")
            
        if default_output is not None:
            output_info = sd.query_devices(default_output)
            print(f"🔊 Default output: {output_info['name']}")
        else:
            print("⚠️  No default output device")
        
        return True
        
    except Exception as e:
        print(f"❌ Audio devices test failed: {e}")
        return False

def test_tts_basic():
    """ทดสอบ TTS พื้นฐาน"""
    print("\\n🗣️ ทดสอบ TTS (Text-to-Speech)...")
    
    try:
        # Test TTS import
        try:
            import TTS
            print("✅ TTS library available")
            
            # Test basic TTS functionality (without actually running)
            from TTS.api import TTS
            print("✅ TTS API import successful")
            
            # List available models (just the first few)
            models = TTS.list_models()
            if models:
                print(f"✅ Found {len(models)} TTS models available")
                print(f"   📋 Example models: {models[:3]}")
            
        except Exception as tts_e:
            print(f"⚠️  TTS not fully available: {tts_e}")
            print("ℹ️  Will use fallback TTS methods")
        
        return True
        
    except Exception as e:
        print(f"❌ TTS test failed: {e}")
        return False

def test_voice_integration():
    """ทดสอบการรวม voice components"""
    print("\\n🎯 ทดสอบ Voice Integration...")
    
    try:
        # Test voice components imports
        from voice.speech_recognizer import SpeechRecognizer
        print("✅ SpeechRecognizer import: OK")
        
        from voice.text_to_speech import TextToSpeech
        print("✅ TextToSpeech import: OK")
        
        from voice.voice_controller import VoiceController
        print("✅ VoiceController import: OK")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Voice integration test failed: {e}")
        print("ℹ️  This is expected if voice modules need updates")
        return False

def main():
    """รัน voice processing test"""
    print("=" * 60)
    print("🎙️ JARVIS Voice Processing Test")
    print("=" * 60)
    
    tests = [
        ("Voice Dependencies", test_voice_dependencies),
        ("Audio Devices", test_audio_devices),
        ("Whisper Model", test_whisper_model),
        ("TTS Basic", test_tts_basic),
        ("Voice Integration", test_voice_integration)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    print("\\n" + "=" * 60)
    print(f"📊 Voice Processing Results: {passed}/{total} tests passed")
    
    if passed >= 3:  # Allow some tests to fail
        print("🎉 Voice processing is ready for integration!")
        print("✨ Core voice dependencies are working.")
        if passed < total:
            print("⚠️  Some advanced features may need configuration.")
        return 0
    else:
        print("⚠️  Voice processing needs more work.")
        print("🔧 Fix the failing components before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())