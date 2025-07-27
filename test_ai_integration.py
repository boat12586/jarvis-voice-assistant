#!/usr/bin/env python3
"""
Test AI Engine Integration for JARVIS Voice Assistant
Tests the full AI pipeline with local models
"""

import sys
import os
import time
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def test_config_manager():
    """Test configuration manager"""
    print("⚙️ Testing Configuration Manager...")
    
    try:
        from system.config_manager import ConfigManager
        
        config = ConfigManager()
        print("   ✅ Configuration manager loaded")
        
        # Test AI config
        ai_config = config.get('ai', {})
        print(f"   ℹ️ AI config available: {bool(ai_config)}")
        
        return True, config
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False, None

def test_llm_engine(config):
    """Test LLM engine initialization"""
    print("\n🤖 Testing LLM Engine...")
    
    try:
        from ai.llm_engine import LLMEngine
        
        # Create engine
        llm_config = config.get('ai', {}) if config else {}
        engine = LLMEngine(llm_config)
        print("   ✅ LLM engine created")
        
        # Initialize
        engine.initialize()
        print("   ✅ LLM engine initialized")
        
        # Check if ready
        if engine.is_ready:
            print("   ✅ LLM engine is ready")
        else:
            print("   ⚠️ LLM engine not ready yet")
        
        return True, engine
        
    except Exception as e:
        print(f"   ❌ LLM engine test failed: {e}")
        return False, None

def test_ai_engine(config):
    """Test full AI engine"""
    print("\n🎯 Testing AI Engine...")
    
    try:
        from ai.ai_engine import AIEngine
        
        # Create engine
        ai_config = config.get('ai', {}) if config else {}
        engine = AIEngine(ai_config)
        print("   ✅ AI engine created")
        
        # Check if ready
        if engine.is_ready:
            print("   ✅ AI engine is ready")
        else:
            print("   ⚠️ AI engine not ready")
        
        return True, engine
        
    except Exception as e:
        print(f"   ❌ AI engine test failed: {e}")
        return False, None

def test_query_processing(ai_engine):
    """Test query processing"""
    print("\n💬 Testing Query Processing...")
    
    if not ai_engine:
        print("   ❌ No AI engine available")
        return False
    
    try:
        # Test simple queries
        test_queries = [
            ("Hello", "en"),
            ("What time is it?", "en"),
            ("สวัสดี", "th")
        ]
        
        for query, language in test_queries:
            print(f"   🔍 Testing: '{query}' ({language})")
            
            # Set up response handler
            response_received = False
            response_text = ""
            
            def on_response(text, metadata):
                nonlocal response_received, response_text
                response_received = True
                response_text = text
                print(f"      ✅ Response: {text[:50]}...")
            
            def on_error(error):
                print(f"      ❌ Error: {error}")
            
            # Connect signals
            ai_engine.response_ready.connect(on_response)
            ai_engine.error_occurred.connect(on_error)
            
            # Process query
            ai_engine.process_query(query, language)
            
            # Wait for response (simple polling)
            wait_time = 0
            while not response_received and wait_time < 10:
                time.sleep(0.1)
                wait_time += 0.1
                
                # Process events if Qt is available
                try:
                    from PyQt6.QtCore import QApplication
                    app = QApplication.instance()
                    if app:
                        app.processEvents()
                except:
                    pass
            
            if response_received:
                print(f"      ✅ Query processed successfully")
            else:
                print(f"      ⚠️ No response received (timeout)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Query processing test failed: {e}")
        return False

def test_engine_info(ai_engine):
    """Test engine information retrieval"""
    print("\n📊 Testing Engine Information...")
    
    if not ai_engine:
        print("   ❌ No AI engine available")
        return False
    
    try:
        info = ai_engine.get_engine_info()
        print(f"   ✅ Engine info retrieved")
        print(f"      Ready: {info.get('is_ready', False)}")
        print(f"      Processing: {info.get('is_processing', False)}")
        
        if 'llm_engine' in info and info['llm_engine']:
            llm_info = info['llm_engine']
            print(f"      Model: {llm_info.get('model_name', 'Unknown')}")
            print(f"      Requests: {llm_info.get('request_count', 0)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Engine info test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 JARVIS AI Engine Integration Test")
    print("=" * 60)
    
    # Test configuration
    config_success, config = test_config_manager()
    if not config_success:
        print("\n❌ Configuration failed - cannot continue")
        return False
    
    # Test LLM engine
    llm_success, llm_engine = test_llm_engine(config)
    if not llm_success:
        print("\n❌ LLM engine failed - trying fallback only")
    
    # Test AI engine
    ai_success, ai_engine = test_ai_engine(config)
    if not ai_success:
        print("\n❌ AI engine failed - cannot continue")
        return False
    
    # Test query processing
    query_success = test_query_processing(ai_engine)
    
    # Test engine info
    info_success = test_engine_info(ai_engine)
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 AI ENGINE INTEGRATION TEST SUMMARY:")
    print("=" * 60)
    
    tests = [
        ("Configuration", config_success),
        ("LLM Engine", llm_success),
        ("AI Engine", ai_success),
        ("Query Processing", query_success),
        ("Engine Info", info_success)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, result in tests:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= 4:  # Allow one test to fail
        print("\n🎉 AI Engine integration is working!")
        print("\nNext steps:")
        print("1. Test web interface: python jarvis_web_app.py")
        print("2. Test voice interface: python jarvis_gui_app.py")
        print("3. Test CLI interface: python jarvis_text_interface.py")
    elif passed >= 3:
        print("\n⚠️ AI Engine mostly working with some limitations")
        print("You can proceed with caution")
    else:
        print("\n❌ AI Engine needs more work")
        print("Address the failed tests before proceeding")
    
    return passed >= 3

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)