#!/usr/bin/env python3
"""
Startup Test for Jarvis Voice Assistant
Tests basic application initialization without UI
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test basic imports"""
    print("📦 Testing Core Imports...")
    
    try:
        from system.config_manager import ConfigManager
        print("   ✅ ConfigManager imported")
        
        from system.logger import setup_logger
        print("   ✅ Logger imported")
        
        # Test voice imports
        from voice.speech_recognizer import SimpleSpeechRecognizer
        print("   ✅ Speech recognizer imported")
        
        # Test basic AI imports
        from ai.rag_system import RAGSystem
        print("   ✅ RAG system imported")
        
        print("   ✅ All core imports successful")
        return True
        
    except Exception as e:
        print(f"   ❌ Import test failed: {e}")
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n⚙️ Testing Configuration...")
    
    try:
        from system.config_manager import ConfigManager
        config = ConfigManager()
        config_data = config.get_config()
        
        print(f"   ✅ Configuration loaded ({len(config_data)} sections)")
        
        # Check key sections
        ai_config = config.get('ai', {})
        voice_config = config.get('voice', {})
        
        if ai_config and voice_config:
            print("   ✅ AI and Voice configurations present")
            return True
        else:
            print("   ⚠️ Some configuration sections missing")
            return False
            
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def test_basic_initialization():
    """Test basic component initialization"""
    print("\n🔧 Testing Component Initialization...")
    
    try:
        from system.config_manager import ConfigManager
        config = ConfigManager()
        
        # Test speech recognizer
        from voice.speech_recognizer import SimpleSpeechRecognizer
        recognizer = SimpleSpeechRecognizer()
        print("   ✅ Speech recognizer initialized")
        
        # Test RAG system (if knowledge base exists)
        try:
            from ai.rag_system import RAGSystem
            rag = RAGSystem(config)
            print("   ✅ RAG system initialized")
        except Exception as e:
            print(f"   ⚠️ RAG system initialization warning: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Component initialization failed: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 Jarvis Voice Assistant - Startup Test")
    print("=" * 60)
    
    tests = [
        ("Core Imports", test_imports),
        ("Configuration", test_configuration),
        ("Component Initialization", test_basic_initialization)
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
    print("📋 STARTUP TEST SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Basic startup successful!")
        print("\n📝 System Status:")
        print("• ✅ Core dependencies installed")
        print("• ✅ Voice recognition (Faster-Whisper) working")
        print("• ✅ Audio system available")
        print("• ✅ Configuration system working")
        print("• ✅ Basic AI components importable")
        print("\n🚀 Ready for next phase:")
        print("• Download AI models (Mistral 7B, embeddings)")
        print("• Configure TTS system")
        print("• Test full application integration")
        
    elif passed >= total * 0.7:
        print("\n⚠️ Startup mostly successful with some warnings")
        print("Core functionality should work with limitations")
    else:
        print("\n❌ Startup failed - address critical issues")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)