"""
Local LLM Engine for Jarvis Voice Assistant
Handles Mistral 7B model with quantization and GPU acceleration
"""

import os
import logging
import json
from typing import Dict, Any, List, Optional, Generator
from dataclasses import dataclass
from pathlib import Path
import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex, QTimer


@dataclass
class LLMRequest:
    """LLM request structure"""
    prompt: str
    language: str
    max_tokens: int
    temperature: float
    system_prompt: Optional[str] = None
    conversation_history: Optional[List[Dict[str, str]]] = None


@dataclass
class LLMResponse:
    """LLM response structure"""
    text: str
    language: str
    confidence: float
    processing_time: float
    token_count: int
    model_info: Dict[str, Any]


class PromptTemplates:
    """Prompt templates for different interaction types"""
    
    JARVIS_SYSTEM_PROMPT = """You are J.A.R.V.I.S, Tony Stark's AI assistant from Iron Man. You are:
- Intelligent, sophisticated, and slightly formal
- Helpful and efficient in your responses
- Confident but not arrogant
- Capable of handling technical and general queries
- Multilingual (English and Thai)
- Privacy-focused (all processing is local)

Respond in the same language as the user's query. Keep responses concise but informative.
Use a slightly formal, professional tone that matches J.A.R.V.I.S's character."""

    CONVERSATION_TEMPLATE = """<|system|>
{system_prompt}
</s>
<|user|>
{user_message}
</s>
<|assistant|>
"""

    MULTILINGUAL_TEMPLATE = """<|system|>
{system_prompt}

Language Instructions:
- If user writes in English, respond in English
- If user writes in Thai, respond in Thai
- Detect language automatically from input
- Maintain consistent language throughout response
</s>
<|user|>
{user_message}
</s>
<|assistant|>
"""

    DEEP_THINKING_TEMPLATE = """<|system|>
{system_prompt}

Deep Analysis Mode:
- Provide thorough, analytical responses
- Consider multiple perspectives
- Show reasoning process
- Provide examples and context
- Be comprehensive but structured
</s>
<|user|>
{user_message}
</s>
<|assistant|>
"""

    @staticmethod
    def get_template(template_type: str, system_prompt: str, user_message: str) -> str:
        """Get formatted prompt template"""
        if template_type == "deep_thinking":
            return PromptTemplates.DEEP_THINKING_TEMPLATE.format(
                system_prompt=system_prompt,
                user_message=user_message
            )
        elif template_type == "multilingual":
            return PromptTemplates.MULTILINGUAL_TEMPLATE.format(
                system_prompt=system_prompt,
                user_message=user_message
            )
        else:
            return PromptTemplates.CONVERSATION_TEMPLATE.format(
                system_prompt=system_prompt,
                user_message=user_message
            )


class LocalModelEngine(QThread):
    """Local LLM inference engine using downloaded models"""
    
    # Signals
    response_ready = pyqtSignal(LLMResponse)
    response_error = pyqtSignal(str)
    loading_progress = pyqtSignal(str, int)  # status, percentage
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Find the best available local model
        self.model_path, self.model_name = self._find_best_local_model()
        self.quantization = config.get("quantization", "8bit")
        self.gpu_layers = config.get("gpu_layers", 35)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Model components
        self.tokenizer: Optional[AutoTokenizer] = None
        self.model: Optional[AutoModelForCausalLM] = None
        self.generation_pipeline = None
        
        # Request handling
        self.current_request: Optional[LLMRequest] = None
        self.mutex = QMutex()
        
        # Performance monitoring
        self.model_loaded = False
        self.gpu_memory_used = 0
        
        self.logger.info(f"Initialized with model: {self.model_name} at {self.model_path}")
    
    def _find_best_local_model(self) -> tuple:
        """Find the best available local model"""
        import os
        
        # Base models directory
        models_base = Path(__file__).parent.parent.parent / "models"
        
        # Priority order for models
        model_priorities = [
            "deepseek-r1-distill-llama-8b",
            "mistral-7b-instruct"
        ]
        
        for model_dir in model_priorities:
            model_path = models_base / model_dir
            
            if model_path.exists():
                # Look for huggingface cache structure
                for root, dirs, files in os.walk(model_path):
                    if "snapshots" in dirs:
                        snapshots_dir = Path(root) / "snapshots"
                        # Find the latest snapshot
                        snapshot_dirs = [d for d in snapshots_dir.iterdir() if d.is_dir()]
                        if snapshot_dirs:
                            latest_snapshot = max(snapshot_dirs, key=lambda x: x.stat().st_mtime)
                            # Check if this snapshot has required files
                            if (latest_snapshot / "config.json").exists():
                                self.logger.info(f"Found local model at: {latest_snapshot}")
                                return str(latest_snapshot), model_dir
        
        # Fallback: try to use any model files directly
        for model_dir in model_priorities:
            model_path = models_base / model_dir
            if (model_path / "config.json").exists():
                return str(model_path), model_dir
        
        # If no local models found, fall back to a simple model name
        self.logger.warning("No suitable local models found, will attempt to use fallback")
        return str(models_base / "fallback"), "microsoft/DialoGPT-medium"
        
    def load_model(self) -> bool:
        """Load local model with quantization"""
        try:
            self.loading_progress.emit("Loading tokenizer...", 10)
            
            # Load tokenizer from local path
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
                trust_remote_code=True,
                padding_side="left"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.loading_progress.emit("Configuring quantization...", 30)
            
            # Configure quantization
            quantization_config = None
            if self.quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    load_in_4bit=False,
                    bnb_8bit_compute_dtype=torch.float16,
                    bnb_8bit_use_double_quant=True,
                    bnb_8bit_quant_type="nf4"
                )
            elif self.quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            self.loading_progress.emit("Loading model...", 50)
            
            # Load model from local path
            if quantization_config:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    quantization_config=quantization_config,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_path,
                    local_files_only=True,
                    device_map="auto",
                    trust_remote_code=True,
                    torch_dtype=torch.float16
                )
            
            self.loading_progress.emit("Creating generation pipeline...", 80)
            
            # Create generation pipeline
            self.generation_pipeline = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device_map="auto",
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            
            self.loading_progress.emit("Model loaded successfully", 100)
            
            self.model_loaded = True
            self.logger.info("Mistral model loaded successfully")
            
            # Log GPU memory usage
            if torch.cuda.is_available():
                self.gpu_memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
                self.logger.info(f"GPU memory used: {self.gpu_memory_used:.2f} GB")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load Mistral model: {e}")
            self.response_error.emit(f"Model loading failed: {e}")
            return False
    
    def generate_response(self, request: LLMRequest):
        """Generate response for request"""
        self.current_request = request
        self.start()
    
    def run(self):
        """Run generation in thread"""
        try:
            if not self.model_loaded:
                if not self.load_model():
                    return
            
            if not self.current_request:
                self.response_error.emit("No request provided")
                return
            
            # Generate response
            response = self._generate_text(self.current_request)
            
            if response:
                self.response_ready.emit(response)
            else:
                self.response_error.emit("Failed to generate response")
                
        except Exception as e:
            self.logger.error(f"Generation error: {e}")
            self.response_error.emit(f"Generation failed: {e}")
    
    def _generate_text(self, request: LLMRequest) -> Optional[LLMResponse]:
        """Generate text using the model"""
        try:
            import time
            start_time = time.time()
            
            # Prepare prompt
            system_prompt = request.system_prompt or PromptTemplates.JARVIS_SYSTEM_PROMPT
            
            # Determine template type
            template_type = "conversation"
            if "deep" in request.prompt.lower() or "analyze" in request.prompt.lower():
                template_type = "deep_thinking"
            elif request.language != "en":
                template_type = "multilingual"
            
            # Format prompt
            formatted_prompt = PromptTemplates.get_template(
                template_type, system_prompt, request.prompt
            )
            
            self.logger.info(f"Generating response for: {request.prompt[:50]}...")
            
            # Generate response
            with torch.no_grad():
                outputs = self.generation_pipeline(
                    formatted_prompt,
                    max_new_tokens=request.max_tokens,
                    temperature=request.temperature,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id,
                    return_full_text=False
                )
            
            # Extract generated text
            generated_text = outputs[0]["generated_text"].strip()
            
            # Clean up response
            generated_text = self._clean_response(generated_text)
            
            # Calculate metrics
            processing_time = time.time() - start_time
            token_count = len(self.tokenizer.encode(generated_text))
            
            # Create response object
            response = LLMResponse(
                text=generated_text,
                language=request.language,
                confidence=0.85,  # Simplified confidence score
                processing_time=processing_time,
                token_count=token_count,
                model_info={
                    "model_name": self.model_name,
                    "quantization": self.quantization,
                    "gpu_memory_used": self.gpu_memory_used
                }
            )
            
            self.logger.info(f"Response generated in {processing_time:.2f}s ({token_count} tokens)")
            return response
            
        except Exception as e:
            self.logger.error(f"Text generation error: {e}")
            return None
    
    def _clean_response(self, text: str) -> str:
        """Clean up generated response"""
        # Remove common artifacts
        text = text.replace("<|assistant|>", "")
        text = text.replace("<|user|>", "")
        text = text.replace("</s>", "")
        text = text.replace("<s>", "")
        
        # Remove duplicate spaces
        text = " ".join(text.split())
        
        # Ensure proper ending
        if text and not text.endswith(('.', '!', '?', ':', ';')):
            text += '.'
        
        return text.strip()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        info = {
            "model_name": self.model_name,
            "quantization": self.quantization,
            "device": self.device,
            "model_loaded": self.model_loaded,
            "gpu_memory_used": self.gpu_memory_used
        }
        
        if torch.cuda.is_available():
            info.update({
                "gpu_available": True,
                "gpu_name": torch.cuda.get_device_name(0),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory / 1024**3,
                "gpu_memory_allocated": torch.cuda.memory_allocated() / 1024**3,
                "gpu_memory_reserved": torch.cuda.memory_reserved() / 1024**3
            })
        else:
            info["gpu_available"] = False
        
        return info
    
    def unload_model(self):
        """Unload model to free memory"""
        if self.model:
            del self.model
            self.model = None
        
        if self.tokenizer:
            del self.tokenizer
            self.tokenizer = None
        
        if self.generation_pipeline:
            del self.generation_pipeline
            self.generation_pipeline = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.model_loaded = False
        self.gpu_memory_used = 0
        
        self.logger.info("Model unloaded and memory cleared")


class LLMEngine(QObject):
    """Local LLM engine controller"""
    
    # Signals
    response_ready = pyqtSignal(LLMResponse)
    response_error = pyqtSignal(str)
    model_loading = pyqtSignal(str, int)  # status, percentage
    model_loaded = pyqtSignal()
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Default parameters
        self.default_max_tokens = config.get("max_tokens", 512)
        self.default_temperature = config.get("temperature", 0.7)
        
        # Model engines - start with fallback
        self.use_fallback = True
        
        try:
            from .fallback_llm import FallbackLLMEngine
            self.fallback_engine = FallbackLLMEngine(config)
            self.logger.info("Fallback LLM engine loaded")
        except Exception as e:
            self.logger.error(f"Failed to load fallback engine: {e}")
            self.fallback_engine = None
        
        # Try to load Local Model engine (optional)
        try:
            self.local_model_engine = LocalModelEngine(config)
            self.logger.info("Local model engine initialized")
        except Exception as e:
            self.logger.warning(f"Local model engine not available: {e}")
            self.local_model_engine = None
        
        # State - Fallback is always ready
        self.is_ready = True if self.fallback_engine else False
        self.current_request: Optional[LLMRequest] = None
        
        # Connect signals
        self._connect_signals()
        
        # Performance monitoring
        self.request_count = 0
        self.total_processing_time = 0
        
    def _connect_signals(self):
        """Connect signals from engines"""
        # Connect fallback engine
        if self.fallback_engine:
            self.fallback_engine.response_ready.connect(self._on_fallback_response)
            self.fallback_engine.error_occurred.connect(self._on_response_error)
        
        # Connect Local Model engine if available
        if self.local_model_engine:
            self.local_model_engine.response_ready.connect(self._on_response_ready)
            self.local_model_engine.response_error.connect(self._on_response_error)
            self.local_model_engine.loading_progress.connect(self._on_loading_progress)
        
    def initialize(self):
        """Initialize LLM engine"""
        self.logger.info("Initializing LLM engine...")
        
        # Try to load Local Model in background if available
        if self.local_model_engine:
            try:
                self.local_model_engine.start()
            except Exception as e:
                self.logger.warning(f"Failed to start Local Model engine: {e}")
        
        # Fallback is already ready
        if self.fallback_engine:
            self.logger.info("LLM engine ready with fallback mode")
        
    def process_query(self, query: str, language: str = "en", 
                     max_tokens: Optional[int] = None, 
                     temperature: Optional[float] = None,
                     system_prompt: Optional[str] = None) -> bool:
        """Process user query"""
        if not self.is_ready:
            self.logger.warning("LLM engine not ready")
            self.response_error.emit("AI engine not ready")
            return False
        
        try:
            self.logger.info(f"Processing query: {query[:50]}...")
            
            # Determine which engine to use
            if self.local_model_engine and not self.use_fallback:
                # Create request for Local Model
                request = LLMRequest(
                    prompt=query,
                    language=language,
                    max_tokens=max_tokens or self.default_max_tokens,
                    temperature=temperature or self.default_temperature,
                    system_prompt=system_prompt
                )
                
                self.local_model_engine.generate_response(request)
                self.current_request = request
            else:
                # Use fallback engine
                if self.fallback_engine:
                    self.fallback_engine.generate_response(
                        prompt=query,
                        language=language,
                        max_tokens=max_tokens,
                        temperature=temperature
                    )
                else:
                    self.response_error.emit("No engine available")
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            self.response_error.emit(f"Query processing failed: {e}")
            return False
    
    def _on_response_ready(self, response: LLMResponse):
        """Handle Local Model response ready"""
        self.logger.info(f"Local Model response ready: {response.text[:50]}...")
        
        # Update statistics
        self.request_count += 1
        self.total_processing_time += response.processing_time
        
        # Emit response
        self.response_ready.emit(response)
        self.current_request = None
    
    def _on_fallback_response(self, response_dict: Dict[str, Any]):
        """Handle fallback response ready"""
        self.logger.info(f"Fallback response ready: {response_dict.get('text', '')[:50]}...")
        
        # Convert dict to LLMResponse object
        response = LLMResponse(
            text=response_dict.get('text', ''),
            language=response_dict.get('language', 'en'),
            confidence=response_dict.get('confidence', 0.8),
            processing_time=response_dict.get('processing_time', 0.0),
            token_count=response_dict.get('token_count', 0),
            model_info=response_dict.get('model_info', {})
        )
        
        # Update statistics
        self.request_count += 1
        self.total_processing_time += response.processing_time
        
        # Emit response
        self.response_ready.emit(response)
        self.current_request = None
        
    def _on_response_error(self, error_msg: str):
        """Handle response error"""
        self.logger.error(f"Response error: {error_msg}")
        self.response_error.emit(error_msg)
        self.current_request = None
        
    def _on_loading_progress(self, status: str, percentage: int):
        """Handle loading progress"""
        self.logger.info(f"Loading progress: {status} ({percentage}%)")
        self.model_loading.emit(status, percentage)
        
        if percentage == 100:
            self.is_ready = True
            self.model_loaded.emit()
            self.logger.info("LLM engine ready")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        base_info = {}
        
        if self.local_model_engine:
            base_info = self.local_model_engine.get_model_info()
        elif self.fallback_engine:
            base_info = self.fallback_engine.get_model_info()
        
        base_info.update({
            "is_ready": self.is_ready,
            "request_count": self.request_count,
            "average_processing_time": (
                self.total_processing_time / self.request_count 
                if self.request_count > 0 else 0
            ),
            "default_max_tokens": self.default_max_tokens,
            "default_temperature": self.default_temperature
        })
        
        return base_info
    
    def set_parameters(self, max_tokens: Optional[int] = None, 
                      temperature: Optional[float] = None):
        """Set generation parameters"""
        if max_tokens is not None:
            self.default_max_tokens = max(1, min(2048, max_tokens))
        
        if temperature is not None:
            self.default_temperature = max(0.1, min(2.0, temperature))
        
        self.logger.info(f"Parameters updated: max_tokens={self.default_max_tokens}, temperature={self.default_temperature}")
    
    def shutdown(self):
        """Shutdown LLM engine"""
        self.logger.info("Shutting down LLM engine...")
        
        # Stop any running generation
        if self.local_model_engine and self.local_model_engine.isRunning():
            self.local_model_engine.quit()
            self.local_model_engine.wait()
        
        # Unload model
        if self.local_model_engine:
            self.local_model_engine.unload_model()
        
        self.is_ready = False
        self.logger.info("LLM engine shutdown complete")