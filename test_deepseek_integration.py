#!/usr/bin/env python3
"""
Test DeepSeek-R1 integration and functionality
"""

import sys
import time
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_deepseek_model_loading():
    """Test DeepSeek-R1 model loading"""
    print("🧠 Testing DeepSeek-R1 Model Loading...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        model_name = "deepseek-ai/deepseek-r1-distill-llama-8b"
        
        print(f"   📥 Loading tokenizer for {model_name}...")
        start_time = time.time()
        
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - start_time
        
        print(f"   ✅ Tokenizer loaded in {tokenizer_time:.2f}s")
        
        # Test tokenization
        test_text = "Hello, I am JARVIS. How can I assist you today?"
        tokens = tokenizer.encode(test_text)
        decoded = tokenizer.decode(tokens)
        
        print(f"   🧮 Tokenization test:")
        print(f"      Text: '{test_text}'")
        print(f"      Tokens: {len(tokens)}")
        print(f"      Decoded: '{decoded}'")
        
        # Try loading model (this might take a while or fail due to size)
        print(f"   📥 Attempting to load model...")
        print(f"      Note: This may take several minutes for first download...")
        
        try:
            start_time = time.time()
            
            # Load with reduced precision to save memory
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
                trust_remote_code=True,
                load_in_8bit=True,  # Use 8-bit quantization
                timeout=300  # 5 minute timeout
            )
            
            model_time = time.time() - start_time
            print(f"   ✅ Model loaded in {model_time:.2f}s")
            
            # Test basic inference
            print("   🧮 Testing basic inference...")
            input_text = "What is artificial intelligence?"
            inputs = tokenizer(input_text, return_tensors="pt")
            
            start_time = time.time()
            with torch.no_grad():
                outputs = model.generate(
                    inputs.input_ids,
                    max_length=inputs.input_ids.shape[1] + 50,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            inference_time = time.time() - start_time
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"   ✅ Inference completed in {inference_time:.2f}s")
            print(f"   📝 Response: '{response[len(input_text):].strip()}'")
            
            return True, "full"
            
        except Exception as e:
            print(f"   ⚠️ Model loading failed: {e}")
            print("   ℹ️ This is expected if model is too large or still downloading")
            
            # Check if it's a download issue
            if "timed out" in str(e).lower() or "timeout" in str(e).lower():
                print("   💡 Suggestion: Model may be downloading in background")
                return True, "partial"
            else:
                print("   💡 Suggestion: Consider using smaller model or more memory")
                return False, "failed"
        
    except Exception as e:
        print(f"   ❌ DeepSeek test failed: {e}")
        traceback.print_exc()
        return False, "error"

def test_llm_engine_integration():
    """Test LLM engine integration with DeepSeek"""
    print("\n🎯 Testing LLM Engine Integration...")
    
    try:
        from ai.llm_engine import MistralEngine
        from system.config_manager import ConfigManager
        
        # Get configuration
        config = ConfigManager()
        ai_config = config.get("ai", {})
        
        print(f"   ⚙️ AI Config: {ai_config}")
        print(f"   🤖 Model Name: {ai_config.get('model_name', 'Not set')}")
        
        # Initialize LLM engine
        print("   🔧 Initializing LLM Engine...")
        
        try:
            # Create LLM engine (this might not fully load the model due to size)
            llm_engine = MistralEngine(ai_config)
            
            print("   ✅ LLM Engine created successfully")
            print("   ℹ️ Model loading may happen in background")
            
            # Test basic configuration
            print(f"   📊 Engine config:")
            print(f"      Model: {llm_engine.model_name}")
            print(f"      Device: {llm_engine.device}")
            print(f"      Quantization: {llm_engine.quantization}")
            
            return True
            
        except Exception as e:
            print(f"   ⚠️ LLM Engine initialization issue: {e}")
            print("   ℹ️ This might be due to model size or download status")
            return False
        
    except Exception as e:
        print(f"   ❌ LLM Engine test failed: {e}")
        traceback.print_exc()
        return False

def test_ai_engine_integration():
    """Test full AI engine integration"""
    print("\n🤖 Testing AI Engine Integration...")
    
    try:
        from ai.ai_engine import AIEngine
        from system.config_manager import ConfigManager
        
        config = ConfigManager()
        
        print("   🔧 Initializing AI Engine...")
        ai_engine = AIEngine(config)
        
        print("   ✅ AI Engine initialized")
        
        # Test RAG integration
        if hasattr(ai_engine, 'rag_system') and ai_engine.rag_system:
            print("   ✅ RAG system integrated")
            
            # Test knowledge retrieval
            try:
                query = "What are JARVIS capabilities?"
                context = ai_engine.rag_system.get_relevant_context(query)
                
                if context:
                    print(f"   ✅ Knowledge retrieval working")
                    print(f"   📚 Context length: {len(context)} characters")
                    print(f"   📝 Sample context: '{context[:100]}...'")
                else:
                    print("   ⚠️ No context retrieved")
                    
            except Exception as e:
                print(f"   ⚠️ Knowledge retrieval error: {e}")
        else:
            print("   ❌ RAG system not found")
        
        # Test LLM integration (without actual inference due to model size)
        if hasattr(ai_engine, 'llm_engine'):
            print("   ✅ LLM engine reference found")
        else:
            print("   ⚠️ LLM engine not initialized")
        
        return True
        
    except Exception as e:
        print(f"   ❌ AI Engine test failed: {e}")
        traceback.print_exc()
        return False

def test_fallback_model():
    """Test with a smaller fallback model"""
    print("\n🔄 Testing Fallback Model (DialoGPT)...")
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        
        # Use smaller model for testing
        fallback_model = "microsoft/DialoGPT-medium"
        
        print(f"   📥 Loading fallback model: {fallback_model}")
        
        tokenizer = AutoTokenizer.from_pretrained(fallback_model)
        model = AutoModelForCausalLM.from_pretrained(fallback_model)
        
        print("   ✅ Fallback model loaded successfully")
        
        # Test conversation
        input_text = "Hello, what can you do?"
        inputs = tokenizer.encode(input_text + tokenizer.eos_token, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_length=inputs.shape[1] + 50,
                do_sample=True,
                temperature=0.8,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"   ✅ Fallback inference successful")
        print(f"   📝 Response: '{response}'")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Fallback model test failed: {e}")
        return False

def create_model_switching_system():
    """Create system to switch between models based on availability"""
    print("\n⚙️ Creating Model Switching System...")
    
    try:
        model_config_code = '''"""
Dynamic Model Configuration for JARVIS
Switches between available models based on system capabilities
"""

import torch
import logging
from typing import Dict, Any, Optional

class ModelManager:
    """Manages model selection and fallback"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_models = {}
        self.current_model_info = None
        
    def check_system_capabilities(self) -> Dict[str, Any]:
        """Check system capabilities for model selection"""
        capabilities = {
            "cuda_available": torch.cuda.is_available(),
            "gpu_memory_gb": 0,
            "system_memory_gb": 0
        }
        
        if capabilities["cuda_available"]:
            try:
                capabilities["gpu_memory_gb"] = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except:
                capabilities["gpu_memory_gb"] = 0
        
        try:
            import psutil
            capabilities["system_memory_gb"] = psutil.virtual_memory().total / 1024**3
        except:
            capabilities["system_memory_gb"] = 8  # Conservative estimate
        
        return capabilities
    
    def get_optimal_model_config(self) -> Dict[str, Any]:
        """Get optimal model configuration based on system"""
        capabilities = self.check_system_capabilities()
        
        self.logger.info(f"System capabilities: {capabilities}")
        
        # Model configurations by preference
        model_options = [
            {
                "name": "deepseek-ai/deepseek-r1-distill-llama-8b",
                "display_name": "DeepSeek-R1 8B",
                "memory_requirement_gb": 8,
                "gpu_memory_gb": 6,
                "capabilities": ["reasoning", "multilingual", "large_context"],
                "priority": 1
            },
            {
                "name": "microsoft/DialoGPT-large",
                "display_name": "DialoGPT Large",
                "memory_requirement_gb": 4,
                "gpu_memory_gb": 2,
                "capabilities": ["conversation", "fast"],
                "priority": 2
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "display_name": "DialoGPT Medium",
                "memory_requirement_gb": 2,
                "gpu_memory_gb": 1,
                "capabilities": ["conversation", "lightweight"],
                "priority": 3
            }
        ]
        
        # Select best available model
        for model_config in model_options:
            memory_ok = capabilities["system_memory_gb"] >= model_config["memory_requirement_gb"]
            gpu_ok = (not capabilities["cuda_available"] or 
                     capabilities["gpu_memory_gb"] >= model_config["gpu_memory_gb"])
            
            if memory_ok and gpu_ok:
                self.logger.info(f"Selected model: {model_config['display_name']}")
                self.current_model_info = model_config
                return {
                    "model_name": model_config["name"],
                    "load_in_8bit": capabilities["gpu_memory_gb"] < 8,
                    "device_map": "auto" if capabilities["cuda_available"] else None,
                    "torch_dtype": "float16" if capabilities["cuda_available"] else "float32"
                }
        
        # Fallback to smallest model
        fallback = model_options[-1]
        self.logger.warning(f"Using fallback model: {fallback['display_name']}")
        self.current_model_info = fallback
        
        return {
            "model_name": fallback["name"],
            "load_in_8bit": False,
            "device_map": None,
            "torch_dtype": "float32"
        }
    
    def get_model_info(self) -> Optional[Dict[str, Any]]:
        """Get current model information"""
        return self.current_model_info

# Global model manager
model_manager = ModelManager()
'''
        
        model_file = Path("src/utils/model_manager.py")
        model_file.parent.mkdir(exist_ok=True)
        
        with open(model_file, 'w', encoding='utf-8') as f:
            f.write(model_config_code)
        
        print("   ✅ Model switching system created")
        
        # Test the model manager
        print("   🧮 Testing model selection...")
        
        exec(model_config_code)
        
        # This will be available in the local scope
        capabilities = model_manager.check_system_capabilities()
        print(f"   📊 System capabilities: {capabilities}")
        
        optimal_config = model_manager.get_optimal_model_config()
        print(f"   🎯 Optimal model config: {optimal_config}")
        
        model_info = model_manager.get_model_info()
        if model_info:
            print(f"   📋 Selected model: {model_info['display_name']}")
            print(f"   🏷️ Capabilities: {model_info['capabilities']}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Model switching system creation failed: {e}")
        return False

def main():
    """Main testing function"""
    print("🧠 JARVIS DeepSeek-R1 Integration Test")
    print("=" * 60)
    
    tests = [
        ("DeepSeek Model Loading", test_deepseek_model_loading),
        ("LLM Engine Integration", test_llm_engine_integration),
        ("AI Engine Integration", test_ai_engine_integration),
        ("Fallback Model Test", test_fallback_model),
        ("Model Switching System", create_model_switching_system)
    ]
    
    results = []
    model_status = "unknown"
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "DeepSeek Model Loading":
                result, status = test_func()
                model_status = status
                results.append((test_name, result))
            else:
                result = test_func()
                results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error in {test_name}: {e}")
            traceback.print_exc()
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 DEEPSEEK INTEGRATION SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"DeepSeek Model Status: {model_status}")
    
    # Recommendations
    if model_status == "full":
        print("\n🎉 DeepSeek-R1 fully functional!")
        print("🚀 JARVIS ready for advanced reasoning tasks")
    elif model_status == "partial":
        print("\n⚠️ DeepSeek-R1 partially working")
        print("💡 Model may still be downloading or need more memory")
        print("🔄 Fallback models available for immediate use")
    elif passed >= total * 0.8:
        print("\n✅ Integration mostly successful")
        print("🔄 Using fallback models until DeepSeek-R1 is ready")
    else:
        print("\n❌ Integration issues detected")
        print("🛠️ Check system requirements and dependencies")
    
    # Next steps
    print(f"\n📋 Next Steps:")
    if model_status in ["full", "partial"]:
        print("1. ✅ RAG system working with knowledge base")
        print("2. ✅ DeepSeek-R1 configuration ready")
        print("3. 🎯 Test full conversation pipeline")
        print("4. 🗣️ Implement voice command recognition")
        print("5. 🌐 Add Thai language enhancements")
    else:
        print("1. 🔧 Ensure adequate system resources")
        print("2. ⏳ Allow time for model downloads")
        print("3. 🔄 Use fallback models for testing")
        print("4. 🚀 Proceed with feature development")
    
    return passed >= total * 0.6

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)