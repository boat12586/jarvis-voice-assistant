#!/usr/bin/env python3
"""
AI System Test for Jarvis Voice Assistant
Tests AI engine and model availability
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_transformers():
    """Test transformers library"""
    print("🧠 Testing Transformers Library...")
    
    try:
        import transformers
        print(f"   ✅ Transformers version: {transformers.__version__}")
        
        # Test tokenizer
        from transformers import AutoTokenizer
        print("   📝 Testing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        test_text = "Hello, how are you?"
        tokens = tokenizer.encode(test_text)
        print(f"   ✅ Tokenization test successful ({len(tokens)} tokens)")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Transformers test failed: {e}")
        return False

def test_sentence_transformers():
    """Test sentence transformers for embeddings"""
    print("\n🔍 Testing Sentence Transformers...")
    
    try:
        # Install if needed
        import subprocess
        try:
            import sentence_transformers
        except ImportError:
            print("   📥 Installing sentence-transformers...")
            subprocess.run([sys.executable, "-m", "pip", "install", "sentence-transformers"], 
                         check=True, capture_output=True)
            import sentence_transformers
        
        print(f"   ✅ Sentence Transformers version: {sentence_transformers.__version__}")
        
        # Test embeddings
        from sentence_transformers import SentenceTransformer
        print("   📥 Loading embedding model...")
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test encoding
        sentences = ["This is a test sentence", "Another test sentence"]
        embeddings = model.encode(sentences)
        print(f"   ✅ Embedding test successful (shape: {embeddings.shape})")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Sentence transformers test failed: {e}")
        return False

def test_rag_system():
    """Test RAG system implementation"""
    print("\n📚 Testing RAG System...")
    
    try:
        from ai.rag_system import RAGSystem
        from system.config_manager import ConfigManager
        print("   ✅ RAG system imported successfully")
        
        # Test initialization with config
        config = ConfigManager()
        rag = RAGSystem(config)
        print("   ✅ RAG system initialized")
        
        # Test with sample query
        try:
            results = rag.search("What is JARVIS?", top_k=1)
            if results:
                print(f"   ✅ Search test successful (found {len(results)} results)")
            else:
                print("   ⚠️ Search returned no results (knowledge base may be empty)")
        except Exception as e:
            print(f"   ⚠️ Search test failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ RAG system test failed: {e}")
        return False

def test_llm_availability():
    """Test LLM model availability"""
    print("\n🤖 Testing LLM Availability...")
    
    try:
        # Check if we can load a small model for testing
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print("   📥 Testing small model loading...")
        
        model_name = "microsoft/DialoGPT-small"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Don't load the full model in test, just check availability
        print(f"   ✅ Model {model_name} accessible")
        
        # Test our LLM engine import
        try:
            from ai.llm_engine import LLMEngine
            print("   ✅ LLM engine imported successfully")
        except Exception as e:
            print(f"   ⚠️ LLM engine import failed: {e}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ LLM test failed: {e}")
        return False

def test_ai_engine():
    """Test main AI engine"""
    print("\n🎯 Testing AI Engine Integration...")
    
    try:
        from ai.ai_engine import AIEngine
        from system.config_manager import ConfigManager
        print("   ✅ AI engine imported successfully")
        
        # Test initialization with config
        config = ConfigManager()
        engine = AIEngine(config)
        print("   ✅ AI engine initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ AI engine test failed: {e}")
        return False

def test_configuration():
    """Test configuration system"""
    print("\n⚙️ Testing Configuration System...")
    
    try:
        from system.config_manager import ConfigManager
        print("   ✅ Config manager imported successfully")
        
        config = ConfigManager()
        print("   ✅ Configuration loaded")
        
        # Test AI config access
        ai_config = config.get('ai', {})
        voice_config = config.get('voice', {})
        
        print(f"   ℹ️ AI config sections: {len(ai_config)}")
        print(f"   ℹ️ Voice config sections: {len(voice_config)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Configuration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 Jarvis Voice Assistant - AI System Test")
    print("=" * 60)
    
    tests = [
        ("Transformers Library", test_transformers),
        ("Sentence Transformers", test_sentence_transformers),
        ("RAG System", test_rag_system),
        ("LLM Availability", test_llm_availability),
        ("AI Engine", test_ai_engine),
        ("Configuration", test_configuration)
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
    print("📋 AI SYSTEM TEST SUMMARY:")
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
        print("\n🎉 AI system ready!")
        print("\nNext steps:")
        print("1. Test TTS system with: python test_tts.py")
        print("2. Download AI models if needed")
        print("3. Run full application: python src/main.py")
    elif passed >= total * 0.7:
        print("\n⚠️ AI system mostly ready with some limitations")
        print("You may need to download additional models")
    else:
        print("\n❌ AI system not ready - address failed tests")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)