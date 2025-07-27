"""
AI Engine for Jarvis Voice Assistant
Handles local LLM processing and RAG
"""

import logging
from typing import Dict, Any, Optional
from PyQt6.QtCore import QObject, pyqtSignal, QTimer

from .llm_engine import LLMEngine, LLMResponse
from .rag_factory import RAGSystemManager


class AIEngine(QObject):
    """AI processing engine with local LLM"""
    
    # Signals
    response_ready = pyqtSignal(str, dict)  # response, metadata
    error_occurred = pyqtSignal(str)
    model_loading = pyqtSignal(str, int)  # status, percentage
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Components
        self.llm_engine: Optional[LLMEngine] = None
        self.rag_manager: Optional[RAGSystemManager] = None
        self.rag_system = None  # Will be set by rag_manager
        
        # State
        self.is_ready = False
        self.is_processing = False
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize AI components"""
        try:
            # Initialize RAG system using factory
            rag_config = self.config.get("rag", {})
            self.rag_manager = RAGSystemManager(rag_config)
            
            if self.rag_manager.initialize():
                self.rag_system = self.rag_manager.get_system()
                self.logger.info(f"RAG system initialized successfully")
                
                # Load initial knowledge base
                self._load_initial_knowledge()
            else:
                self.logger.warning("Failed to initialize RAG system")
            
            # Initialize LLM engine
            self.llm_engine = LLMEngine(self.config)
            
            # Connect signals
            self._connect_signals()
            
            # Start initialization
            self.llm_engine.initialize()
            
            # Set ready state based on engine availability
            if self.llm_engine and self.llm_engine.is_ready:
                self.is_ready = True
                self.logger.info("AI engine initialized and ready")
            else:
                # If LLM engine has fallback, we're still ready
                if self.llm_engine and self.llm_engine.fallback_engine:
                    self.is_ready = True
                    self.logger.info("AI engine initialized with fallback mode")
                else:
                    self.logger.warning("AI engine initialized but not ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize AI engine: {e}")
            # Try to continue with fallback
            try:
                from .fallback_llm import FallbackLLMEngine
                fallback_config = self.config.get("fallback", {})
                fallback_engine = FallbackLLMEngine(fallback_config)
                self.is_ready = True
                self.logger.info("AI engine initialized with emergency fallback")
            except Exception as fallback_error:
                self.logger.error(f"Even fallback failed: {fallback_error}")
                self.error_occurred.emit(f"AI engine initialization failed: {e}")
    
    def _connect_signals(self):
        """Connect signals from components"""
        if self.llm_engine:
            self.llm_engine.response_ready.connect(self._on_llm_response)
            self.llm_engine.response_error.connect(self._on_llm_error)
            self.llm_engine.model_loading.connect(self._on_model_loading)
            self.llm_engine.model_loaded.connect(self._on_model_loaded)
        
        if self.rag_system:
            self.rag_system.error_occurred.connect(self._on_rag_error)
    
    def process_query(self, query: str, language: str = "en"):
        """Process user query"""
        if not self.is_ready:
            self.logger.warning("AI engine not ready")
            self.error_occurred.emit("AI engine not ready")
            return
        
        if self.is_processing:
            self.logger.warning("Already processing a query")
            self.error_occurred.emit("Already processing a query")
            return
        
        try:
            self.logger.info(f"Processing query: {query[:50]}... (language: {language})")
            self.is_processing = True
            
            # Determine query type and adjust parameters
            max_tokens = 512
            temperature = 0.7
            system_prompt = None
            
            # Adjust parameters based on query type
            if any(keyword in query.lower() for keyword in ["explain", "analyze", "detail", "comprehensive"]):
                max_tokens = 1024
                temperature = 0.6
            elif any(keyword in query.lower() for keyword in ["create", "generate", "write"]):
                max_tokens = 768
                temperature = 0.8
            elif any(keyword in query.lower() for keyword in ["quick", "brief", "short"]):
                max_tokens = 256
                temperature = 0.5
            
            # Get relevant context from RAG
            context = ""
            if self.rag_system and self.rag_system.is_ready:
                context = self.rag_system.get_context(query, max_context_length=1500)
            
            # Enhance system prompt with context
            if context:
                enhanced_system_prompt = f"""You are J.A.R.V.I.S, Tony Stark's AI assistant. Use the following context to help answer the user's question:

Context:
{context}

Instructions:
- Use the context information when relevant to the user's question
- If the context doesn't contain relevant information, use your general knowledge
- Always maintain the J.A.R.V.I.S personality: professional, intelligent, and helpful
- Respond in the same language as the user's question
- Be concise but informative"""
                system_prompt = enhanced_system_prompt
            
            # Process with LLM
            success = self.llm_engine.process_query(
                query=query,
                language=language,
                max_tokens=max_tokens,
                temperature=temperature,
                system_prompt=system_prompt
            )
            
            if not success:
                self.is_processing = False
                self.error_occurred.emit("Failed to start query processing")
                
        except Exception as e:
            self.logger.error(f"Failed to process query: {e}")
            self.error_occurred.emit(f"Query processing failed: {e}")
            self.is_processing = False
    
    def _on_llm_response(self, response: LLMResponse):
        """Handle LLM response"""
        try:
            self.logger.info(f"LLM response ready: {response.text[:50]}...")
            
            # Create metadata
            metadata = {
                "language": response.language,
                "confidence": response.confidence,
                "processing_time": response.processing_time,
                "token_count": response.token_count,
                "model_info": response.model_info
            }
            
            # Emit response
            self.response_ready.emit(response.text, metadata)
            
        except Exception as e:
            self.logger.error(f"Error handling LLM response: {e}")
            self.error_occurred.emit(f"Response handling failed: {e}")
        
        finally:
            self.is_processing = False
    
    def _on_llm_error(self, error_msg: str):
        """Handle LLM error"""
        self.logger.error(f"LLM error: {error_msg}")
        self.error_occurred.emit(f"LLM error: {error_msg}")
        self.is_processing = False
    
    def _on_model_loading(self, status: str, percentage: int):
        """Handle model loading progress"""
        self.logger.info(f"Model loading: {status} ({percentage}%)")
        self.model_loading.emit(status, percentage)
    
    def _on_model_loaded(self):
        """Handle model loaded"""
        self.is_ready = True
        self.logger.info("AI engine ready")
    
    def _on_rag_error(self, error_msg: str):
        """Handle RAG system error"""
        self.logger.error(f"RAG error: {error_msg}")
        # Don't emit error for RAG issues, just log them
    
    def _load_initial_knowledge(self):
        """Load initial knowledge base"""
        try:
            from pathlib import Path
            
            # Load knowledge base file using RAG manager
            knowledge_file = Path(__file__).parent.parent.parent / "data" / "knowledge_base.json"
            
            if self.rag_manager and knowledge_file.exists():
                success = self.rag_manager.reload_knowledge_base(str(knowledge_file))
                if success:
                    self.logger.info("Initial knowledge base loaded via RAG manager")
                else:
                    self.logger.warning("Failed to load knowledge base via RAG manager")
            else:
                if not knowledge_file.exists():
                    self.logger.warning(f"Knowledge base file not found: {knowledge_file}")
                if not self.rag_manager:
                    self.logger.warning("RAG manager not available for knowledge loading")
                
        except Exception as e:
            self.logger.error(f"Failed to load initial knowledge: {e}")
    
    def add_knowledge(self, content: str, metadata: Dict[str, Any] = None):
        """Add knowledge to RAG system"""
        if self.rag_system:
            return self.rag_system.add_document(content, metadata)
        return False
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine information"""
        info = {
            "is_ready": self.is_ready,
            "is_processing": self.is_processing,
            "llm_engine": None
        }
        
        if self.llm_engine:
            info["llm_engine"] = self.llm_engine.get_model_info()
        
        if self.rag_manager:
            info["rag_system"] = self.rag_manager.get_stats()
        
        return info
    
    def set_parameters(self, max_tokens: Optional[int] = None, 
                      temperature: Optional[float] = None):
        """Set AI parameters"""
        if self.llm_engine:
            self.llm_engine.set_parameters(max_tokens, temperature)
    
    def shutdown(self):
        """Shutdown AI engine"""
        self.logger.info("Shutting down AI engine")
        
        if self.llm_engine:
            self.llm_engine.shutdown()
        
        if self.rag_manager:
            self.rag_manager.shutdown()
        
        self.is_ready = False
        self.is_processing = False
        
        self.logger.info("AI engine shutdown complete")