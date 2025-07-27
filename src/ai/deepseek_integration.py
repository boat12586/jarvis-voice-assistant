#!/usr/bin/env python3
"""
🧠 DeepSeek-R1 Integration for JARVIS
Advanced AI reasoning with fallback support
"""

import logging
from typing import Optional, Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pathlib import Path
import yaml
try:
    from .fallback_ai import FallbackAI
except ImportError:
    from fallback_ai import FallbackAI

logger = logging.getLogger(__name__)

class DeepSeekR1:
    """DeepSeek-R1 AI integration with fallback system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.model_name = "deepseek-ai/deepseek-r1-distill-llama-8b"
        self.model = None
        self.tokenizer = None
        self.fallback_ai = FallbackAI()
        self.is_model_ready = False
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Try to initialize main model
        self._try_initialize_model()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load AI configuration"""
        if config_path and Path(config_path).exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                return config.get('ai', {})
        
        # Default configuration
        return {
            'llm': {
                'model_name': self.model_name,
                'max_context_length': 8192,
                'temperature': 0.7,
                'max_tokens': 512,
                'quantization': '8bit',
                'device': 'auto',
                'timeout': 30
            }
        }
    
    def _try_initialize_model(self):
        """Try to initialize DeepSeek-R1 model"""
        try:
            logger.info(f"Attempting to load {self.model_name}...")
            
            # Quick availability check
            if not self._check_model_files():
                logger.warning("DeepSeek-R1 model files not complete, using fallback")
                return False
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else "cpu",
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            
            self.is_model_ready = True
            logger.info("✅ DeepSeek-R1 model loaded successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Could not load DeepSeek-R1: {e}. Using fallback AI.")
            self.is_model_ready = False
            return False
    
    def _check_model_files(self) -> bool:
        """Check if model files are completely downloaded"""
        try:
            import os
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            model_dir = f"models--{self.model_name.replace('/', '--')}"
            model_path = Path(cache_dir) / model_dir
            
            if not model_path.exists():
                return False
            
            # Check for essential files
            snapshots_dir = model_path / "snapshots"
            if not snapshots_dir.exists():
                return False
            
            # Find the latest snapshot
            snapshots = list(snapshots_dir.iterdir())
            if not snapshots:
                return False
            
            latest_snapshot = snapshots[0]  # Should be only one
            
            # Check for essential files
            required_files = ['config.json', 'tokenizer.json']
            for file_name in required_files:
                if not (latest_snapshot / file_name).exists():
                    return False
            
            # Check if safetensors files exist (might be symbolic links)
            safetensors_files = list(latest_snapshot.glob("*.safetensors"))
            index_file = latest_snapshot / "model.safetensors.index.json"
            
            # Either direct safetensors or index file should exist
            if not safetensors_files and not index_file.exists():
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking model files: {e}")
            return False
    
    def generate_response(self, text: str, context: Dict[str, Any] = None) -> str:
        """Generate AI response with fallback support"""
        try:
            # If main model is ready, use it
            if self.is_model_ready and self.model and self.tokenizer:
                return self._generate_with_deepseek(text, context)
            else:
                # Use fallback AI
                logger.info("Using fallback AI for response")
                return self.fallback_ai.generate_response(text, context)
                
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return self.fallback_ai.generate_response(text, context)
    
    def _generate_with_deepseek(self, text: str, context: Dict[str, Any] = None) -> str:
        """Generate response using DeepSeek-R1"""
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(text, context)
            
            # Tokenize
            inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=self.config['llm'].get('max_tokens', 512),
                    temperature=self.config['llm'].get('temperature', 0.7),
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the new content
            new_content = response[len(prompt):].strip()
            
            return new_content if new_content else "I understand, but I need a moment to process that."
            
        except Exception as e:
            logger.error(f"DeepSeek generation error: {e}")
            return self.fallback_ai.generate_response(text, context)
    
    def _prepare_prompt(self, text: str, context: Dict[str, Any] = None) -> str:
        """Prepare prompt for DeepSeek-R1"""
        # System prompt for JARVIS personality
        system_prompt = """You are JARVIS, an advanced AI assistant. You are:
- Helpful, intelligent, and sophisticated
- Capable of understanding both English and Thai
- Professional but friendly in tone
- Concise and direct in responses
- Knowledgeable about technology and general topics

Respond naturally and helpfully to the user's query."""
        
        # Add context if available
        if context:
            conversation_history = context.get('conversation_history', [])
            if conversation_history:
                recent_context = "\n".join([
                    f"User: {item.get('user', '')}\nJARVIS: {item.get('assistant', '')}" 
                    for item in conversation_history[-3:]  # Last 3 exchanges
                ])
                system_prompt += f"\n\nRecent conversation:\n{recent_context}"
        
        # Final prompt
        prompt = f"{system_prompt}\n\nUser: {text}\nJARVIS:"
        return prompt
    
    def is_available(self) -> bool:
        """Check if AI system is available (always true with fallback)"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get current model information"""
        if self.is_model_ready:
            return {
                "name": "DeepSeek-R1",
                "model": self.model_name,
                "status": "ready",
                "type": "transformer",
                "capabilities": ["conversation", "reasoning", "multilingual"],
                "languages": ["en", "th", "zh", "many_others"]
            }
        else:
            return self.fallback_ai.get_model_info()
    
    def reload_model(self) -> bool:
        """Try to reload the main model"""
        logger.info("Attempting to reload DeepSeek-R1 model...")
        return self._try_initialize_model()

def test_deepseek_integration():
    """Test DeepSeek integration with fallback"""
    print("🧪 Testing DeepSeek-R1 Integration...")
    
    ai = DeepSeekR1()
    
    print(f"🔹 Model info: {ai.get_model_info()}")
    print(f"🔹 Available: {ai.is_available()}")
    
    test_queries = [
        "Hello, I'm testing JARVIS",
        "What can you help me with?",
        "สวัสดี JARVIS ทำอะไรได้บ้าง",
        "Tell me about AI"
    ]
    
    for query in test_queries:
        response = ai.generate_response(query)
        print(f"\n🔹 Query: {query}")
        print(f"🤖 Response: {response}")
    
    print("\n✅ DeepSeek integration test completed")

if __name__ == "__main__":
    test_deepseek_integration()