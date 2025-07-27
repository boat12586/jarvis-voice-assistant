#\!/usr/bin/env python3
"""
Simple DeepSeek-R1 integration test for JARVIS
"""

import sys
import logging
import torch
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def check_system():
    """Quick system check"""
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 System Check:")
    logger.info(f"  CUDA: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}")
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"  Memory: {gpu_memory:.1f} GB")
    
    return torch.cuda.is_available()

def test_tokenizer():
    """Test DeepSeek tokenizer"""
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import AutoTokenizer
        
        logger.info("🔤 Testing DeepSeek Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            "deepseek-ai/deepseek-r1-distill-llama-8b",
            trust_remote_code=True
        )
        
        # Test basic functionality
        text = "Hello JARVIS"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens, skip_special_tokens=True)
        
        logger.info(f"  ✅ Tokenizer works: {len(tokens)} tokens")
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Tokenizer failed: {e}")
        return False

def test_lightweight_model():
    """Test lightweight alternative"""
    logger = logging.getLogger(__name__)
    
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        logger.info("🪶 Testing DialoGPT (lightweight)...")
        
        tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-small")
        model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-small")
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test inference
        inputs = tokenizer("Hello", return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_length=20,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        logger.info(f"  ✅ Lightweight model works: {len(response)} chars")
        
        return True
        
    except Exception as e:
        logger.error(f"  ❌ Lightweight model failed: {e}")
        return False

def create_config(has_cuda, tokenizer_works, lightweight_works):
    """Create optimized config"""
    logger = logging.getLogger(__name__)
    
    try:
        config = {
            "ai": {
                "llm": {
                    "model_name": "deepseek-ai/deepseek-r1-distill-llama-8b" if has_cuda and tokenizer_works else "microsoft/DialoGPT-small",
                    "fallback_model": "microsoft/DialoGPT-small",
                    "quantization": "8bit" if has_cuda else "none",
                    "max_context_length": 4096 if has_cuda else 1024,
                    "temperature": 0.7,
                    "max_tokens": 256,
                    "device": "auto",
                    "timeout": 60
                }
            }
        }
        
        import yaml
        with open("config/ai_optimized.yaml", 'w') as f:
            yaml.dump(config, f, indent=2)
        
        logger.info("✅ Configuration created: config/ai_optimized.yaml")
        return True
        
    except Exception as e:
        logger.error(f"❌ Config creation failed: {e}")
        return False

def main():
    """Main test"""
    logger = setup_logging()
    
    logger.info("🧪 Quick DeepSeek Integration Test")
    logger.info("="*40)
    
    # Tests
    has_cuda = check_system()
    tokenizer_works = test_tokenizer()
    lightweight_works = test_lightweight_model()
    config_created = create_config(has_cuda, tokenizer_works, lightweight_works)
    
    # Summary
    logger.info("\n" + "="*40)
    logger.info("SUMMARY")
    logger.info("="*40)
    
    logger.info(f"🔍 CUDA: {'Available' if has_cuda else 'Not Available'}")
    logger.info(f"🔤 DeepSeek Tokenizer: {'Works' if tokenizer_works else 'Failed'}")
    logger.info(f"🪶 Lightweight Model: {'Works' if lightweight_works else 'Failed'}")
    logger.info(f"⚙️ Config: {'Created' if config_created else 'Failed'}")
    
    if has_cuda and tokenizer_works:
        logger.info("\n🎉 Ready for DeepSeek-R1\!")
        logger.info("💡 Use quantization for memory efficiency")
    elif lightweight_works:
        logger.info("\n💡 Use lightweight model for now")
        logger.info("☁️ Consider cloud deployment for DeepSeek")
    else:
        logger.info("\n⚠️ Check dependencies")
    
    return tokenizer_works or lightweight_works

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
EOF < /dev/null
