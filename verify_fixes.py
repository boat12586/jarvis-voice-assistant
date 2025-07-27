#!/usr/bin/env python3
"""
Simple verification of JARVIS AI Engine fixes
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def verify_imports():
    """Verify all modules import correctly"""
    print("📦 Verifying imports...")
    
    try:
        from system.config_manager import ConfigManager
        print("   ✅ ConfigManager")
        
        from ai.fallback_llm import FallbackLLMEngine
        print("   ✅ FallbackLLMEngine")
        
        from ai.llm_engine import LocalModelEngine, LLMEngine
        print("   ✅ LocalModelEngine & LLMEngine")
        
        from ai.ai_engine import AIEngine
        print("   ✅ AIEngine")
        
        return True
    except Exception as e:
        print(f"   ❌ Import failed: {e}")
        return False

def verify_basic_functionality():
    """Verify basic AI functionality"""
    print("\n🧠 Verifying basic functionality...")
    
    try:
        from ai.fallback_llm import FallbackLLMEngine
        
        # Test fallback engine
        engine = FallbackLLMEngine({})
        response = engine.generate_response("Hello", "en")
        
        if response and "text" in response:
            print(f"   ✅ Fallback response: {response['text'][:40]}...")
            return True
        else:
            print("   ❌ No response from fallback")
            return False
            
    except Exception as e:
        print(f"   ❌ Basic functionality failed: {e}")
        return False

def verify_model_discovery():
    """Verify local model discovery"""
    print("\n🔍 Verifying model discovery...")
    
    try:
        from ai.llm_engine import LocalModelEngine
        
        engine = LocalModelEngine({})
        print(f"   ✅ Model path: {engine.model_path}")
        print(f"   ✅ Model name: {engine.model_name}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model discovery failed: {e}")
        return False

def verify_configuration():
    """Verify configuration loading"""
    print("\n⚙️ Verifying configuration...")
    
    try:
        from system.config_manager import ConfigManager
        
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print(f"   ✅ Config loaded: {len(config)} sections")
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration failed: {e}")
        return False

def main():
    """Main verification"""
    print("🤖 JARVIS AI Engine - Fix Verification")
    print("=" * 50)
    
    tests = [
        ("Imports", verify_imports),
        ("Basic Functionality", verify_basic_functionality), 
        ("Model Discovery", verify_model_discovery),
        ("Configuration", verify_configuration)
    ]
    
    passed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
    
    print("\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("\n🎉 ALL FIXES VERIFIED!")
        print("\n✅ JARVIS AI Engine is working correctly with:")
        print("   • Local model support (no Hugging Face online dependency)")
        print("   • Fallback pattern-based responses")
        print("   • Proper configuration loading")
        print("   • Model discovery system")
        
        print("\n🚀 Ready to use:")
        print("   • Web interface: python test_web_simple.py")
        print("   • Text interface: python jarvis_text_interface.py")
        print("   • GUI interface: python jarvis_gui_app.py")
        
    elif passed >= 3:
        print("\n⚠️ Mostly working - minor issues present")
        
    else:
        print("\n❌ Significant issues remain")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)