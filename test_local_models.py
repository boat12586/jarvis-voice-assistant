#!/usr/bin/env python3
"""
Test Local Models for JARVIS Voice Assistant
Simple test to verify local model loading works
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_model_discovery():
    """Test local model discovery"""
    print("🔍 Testing local model discovery...")
    
    try:
        models_base = Path(__file__).parent / "models"
        print(f"   Models directory: {models_base}")
        
        if not models_base.exists():
            print("   ❌ Models directory not found")
            return False
        
        # List available models
        for model_dir in models_base.iterdir():
            if model_dir.is_dir():
                print(f"   📁 Found model directory: {model_dir.name}")
                
                # Look for snapshots
                for root, dirs, files in os.walk(model_dir):
                    if "snapshots" in dirs:
                        snapshots_dir = Path(root) / "snapshots"
                        print(f"      📸 Snapshots found: {snapshots_dir}")
                        
                        for snapshot in snapshots_dir.iterdir():
                            if snapshot.is_dir():
                                config_file = snapshot / "config.json"
                                tokenizer_file = snapshot / "tokenizer_config.json"
                                
                                print(f"         📄 Snapshot: {snapshot.name}")
                                print(f"         ✅ Config: {config_file.exists()}")
                                print(f"         ✅ Tokenizer: {tokenizer_file.exists()}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Error during model discovery: {e}")
        return False

def test_fallback_engine():
    """Test fallback engine"""
    print("\n🔄 Testing fallback engine...")
    
    try:
        from ai.fallback_llm import FallbackLLMEngine
        
        # Create engine
        config = {}
        engine = FallbackLLMEngine(config)
        print("   ✅ Fallback engine created")
        
        # Test generation
        response = engine.generate_response("Hello, how are you?", "en")
        if response and "text" in response:
            print(f"   ✅ Response generated: {response['text'][:50]}...")
            return True
        else:
            print("   ❌ No response generated")
            return False
        
    except Exception as e:
        print(f"   ❌ Fallback engine test failed: {e}")
        return False

def test_transformers_basic():
    """Test basic transformers functionality"""
    print("\n🤖 Testing transformers with local model...")
    
    try:
        import torch
        from transformers import AutoTokenizer, AutoConfig
        
        # Find a local model
        models_base = Path(__file__).parent / "models"
        
        for model_dir in ["mistral-7b-instruct", "deepseek-r1-distill-llama-8b"]:
            model_path = models_base / model_dir
            
            if model_path.exists():
                # Look for snapshots
                for root, dirs, files in os.walk(model_path):
                    if "snapshots" in dirs:
                        snapshots_dir = Path(root) / "snapshots"
                        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                        
                        if snapshot_dirs:
                            latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                            
                            if (latest_snapshot / "config.json").exists():
                                print(f"   📁 Testing model at: {latest_snapshot}")
                                
                                try:
                                    # Test config loading
                                    config = AutoConfig.from_pretrained(str(latest_snapshot), local_files_only=True)
                                    print(f"   ✅ Config loaded: {config.model_type}")
                                    
                                    # Test tokenizer loading (if available)
                                    if (latest_snapshot / "tokenizer_config.json").exists():
                                        tokenizer = AutoTokenizer.from_pretrained(str(latest_snapshot), local_files_only=True)
                                        test_text = "Hello world"
                                        tokens = tokenizer.encode(test_text)
                                        print(f"   ✅ Tokenizer test: '{test_text}' -> {len(tokens)} tokens")
                                    
                                    return True
                                    
                                except Exception as e:
                                    print(f"   ⚠️ Model loading failed: {e}")
                                    continue
        
        print("   ❌ No working local models found")
        return False
        
    except Exception as e:
        print(f"   ❌ Transformers test failed: {e}")
        return False

def test_local_model_engine():
    """Test the local model engine"""
    print("\n⚙️ Testing local model engine...")
    
    try:
        from ai.llm_engine import LocalModelEngine
        
        config = {
            "quantization": "none",  # Disable quantization for testing
            "max_tokens": 50,
            "temperature": 0.7
        }
        
        engine = LocalModelEngine(config)
        print(f"   ✅ Engine created with model: {engine.model_name}")
        print(f"   📁 Model path: {engine.model_path}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Local model engine test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🤖 JARVIS Local Models Test")
    print("=" * 50)
    
    tests = [
        ("Model Discovery", test_model_discovery),
        ("Fallback Engine", test_fallback_engine),
        ("Transformers Basic", test_transformers_basic),
        ("Local Model Engine", test_local_model_engine)
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
    print("\n" + "=" * 50)
    print("📋 LOCAL MODELS TEST SUMMARY:")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed >= total * 0.75:
        print("\n🎉 Local models system is working!")
        print("\nRecommendations:")
        if passed < total:
            print("- Some models may need to be fully downloaded")
            print("- Try running with fallback mode if models fail to load")
        print("- Test with: python test_deepseek_simple.py")
    else:
        print("\n❌ Local models system needs attention")
        print("\nNext steps:")
        print("1. Check model downloads are complete")
        print("2. Verify transformers library is properly installed")
        print("3. Consider using fallback engine only")
    
    return passed >= total * 0.75

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)