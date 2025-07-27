#!/usr/bin/env python3
"""
🧪 DeepSeek-R1 Model Benchmark & Test Script
"""

import sys
import time
import psutil
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_deepseek_status():
    """Test if DeepSeek-R1 model is properly downloaded and functional"""
    print("🔍 Testing DeepSeek-R1 Model Status...")
    
    model_name = "deepseek-ai/deepseek-r1-distill-llama-8b"
    
    try:
        print(f"📥 Loading tokenizer for {model_name}...")
        start_time = time.time()
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer_time = time.time() - start_time
        print(f"✅ Tokenizer loaded in {tokenizer_time:.2f}s")
        
        print(f"📥 Loading model for {model_name}...")
        start_time = time.time()
        
        # Check available memory
        memory = psutil.virtual_memory()
        print(f"💾 Available memory: {memory.available / (1024**3):.1f}GB")
        
        # Try loading with appropriate settings
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            low_cpu_mem_usage=True
        )
        model_time = time.time() - start_time
        print(f"✅ Model loaded in {model_time:.2f}s")
        
        # Test inference
        print("🧪 Testing inference...")
        test_prompt = "Hello, my name is JARVIS and I am"
        
        start_time = time.time()
        inputs = tokenizer(test_prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        inference_time = time.time() - start_time
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ Inference completed in {inference_time:.2f}s")
        print(f"🤖 Response: {response}")
        
        # Memory usage
        if torch.cuda.is_available():
            print(f"🎮 GPU Memory: {torch.cuda.memory_allocated() / (1024**3):.2f}GB allocated")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_integration_with_jarvis():
    """Test integration with JARVIS components"""
    print("\n🔗 Testing JARVIS Integration...")
    
    try:
        from ai.deepseek_integration import DeepSeekR1
        
        print("📦 Initializing DeepSeek-R1 integration...")
        deepseek = DeepSeekR1()
        
        print("🧪 Testing conversation...")
        response = deepseek.generate_response("สวัสดี ฉันชื่อ JARVIS")
        print(f"🤖 Thai Response: {response}")
        
        response = deepseek.generate_response("What are your capabilities?")
        print(f"🤖 English Response: {response}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration error: {e}")
        return False

def main():
    print("🚀 JARVIS DeepSeek-R1 Benchmark Starting...")
    print("=" * 60)
    
    # System info
    print(f"🖥️  System: {psutil.cpu_count()} CPUs, {psutil.virtual_memory().total / (1024**3):.1f}GB RAM")
    print(f"🎮 CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
    
    print("\n" + "=" * 60)
    
    # Test 1: Model status
    model_ok = test_deepseek_status()
    
    # Test 2: JARVIS integration
    integration_ok = test_integration_with_jarvis()
    
    print("\n" + "=" * 60)
    print("📊 BENCHMARK RESULTS:")
    print(f"🔹 DeepSeek-R1 Model: {'✅ Ready' if model_ok else '❌ Not Ready'}")
    print(f"🔹 JARVIS Integration: {'✅ Working' if integration_ok else '❌ Issues'}")
    
    if model_ok and integration_ok:
        print("\n🎉 All systems operational! JARVIS with DeepSeek-R1 is ready!")
    else:
        print("\n⚠️  Some issues detected. Check logs above.")

if __name__ == "__main__":
    main()