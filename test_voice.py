#!/usr/bin/env python3
"""
Voice System Test for Jarvis Voice Assistant
Tests speech recognition without UI dependencies
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_faster_whisper():
    """Test Faster-Whisper installation and model loading"""
    print("🎙️ Testing Faster-Whisper...")
    
    try:
        from faster_whisper import WhisperModel
        print("   ✅ Faster-Whisper imported successfully")
        
        # Test model initialization (small model for quick test)
        print("   📥 Loading Whisper base model...")
        model = WhisperModel("base", device="cpu", compute_type="int8")
        print("   ✅ Whisper model loaded successfully")
        
        # Test basic functionality with a dummy audio (silence)
        print("   🔍 Testing transcription capability...")
        import numpy as np
        # Create 1 second of silence for testing
        dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second at 16kHz
        
        segments, info = model.transcribe(dummy_audio, beam_size=1)
        print(f"   ✅ Transcription test completed (detected language: {info.language})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Faster-Whisper test failed: {e}")
        return False

def test_speech_recognizer():
    """Test our speech recognizer implementation"""
    print("\n🎯 Testing Speech Recognizer Implementation...")
    
    try:
        from voice.speech_recognizer import SimpleSpeechRecognizer
        print("   ✅ Speech recognizer imported successfully")
        
        # Test initialization
        recognizer = SimpleSpeechRecognizer()
        print("   ✅ Speech recognizer initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Speech recognizer test failed: {e}")
        return False

def test_audio_system():
    """Test audio system availability"""
    print("\n🔊 Testing Audio System...")
    
    try:
        import sounddevice as sd
        print("   ✅ Sounddevice imported successfully")
        
        # List audio devices
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        output_devices = [d for d in devices if d['max_output_channels'] > 0]
        
        print(f"   📱 Found {len(input_devices)} input devices")
        print(f"   📢 Found {len(output_devices)} output devices")
        
        if input_devices and output_devices:
            print("   ✅ Audio system ready")
            return True
        else:
            print("   ⚠️ Limited audio device availability")
            return False
            
    except Exception as e:
        print(f"   ❌ Audio system test failed: {e}")
        return False

def test_torch_gpu():
    """Test PyTorch GPU availability"""
    print("\n🎮 Testing PyTorch GPU Support...")
    
    try:
        import torch
        print(f"   ✅ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   ✅ CUDA available - devices: {torch.cuda.device_count()}")
            print(f"   🎮 Current device: {torch.cuda.get_device_name()}")
            return True
        else:
            print("   ⚠️ CUDA not available - using CPU")
            return False
            
    except Exception as e:
        print(f"   ❌ PyTorch test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 Jarvis Voice Assistant - Voice System Test")
    print("=" * 60)
    
    tests = [
        ("Faster-Whisper", test_faster_whisper),
        ("Speech Recognizer", test_speech_recognizer), 
        ("Audio System", test_audio_system),
        ("PyTorch GPU", test_torch_gpu)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error during {test_name} test: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 VOICE SYSTEM TEST SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Voice system ready!")
        print("\nNext steps:")
        print("1. Test AI engine with: python test_ai.py")
        print("2. Install additional dependencies if needed")
        print("3. Run full application: python src/main.py")
    elif passed >= total * 0.7:
        print("\n⚠️ Voice system mostly ready with some limitations")
    else:
        print("\n❌ Voice system not ready - address failed tests")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)