"""
Enhanced RAG System with comprehensive error handling and logging
Improved version of the original RAG system with better diagnostics
"""

import os
import json
import logging
import pickle
import traceback
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex
import hashlib

class RAGError(Exception):
    """Custom exception for RAG system errors"""
    def __init__(self, message: str, error_type: str = "general", details: Dict = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}
        self.timestamp = time.time()

class RAGLogger:
    """Enhanced logging for RAG system"""
    
    def __init__(self, name: str, log_file: str = None):
        self.logger = logging.getLogger(name)
        self.log_file = log_file
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler if specified
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.setLevel(logging.DEBUG)
    
    def log_operation(self, operation: str, success: bool, details: Dict = None):
        """Log operation with structured information"""
        details = details or {}
        status = "SUCCESS" if success else "FAILED"
        message = f"{operation} - {status}"
        
        if details:
            detail_str = ", ".join(f"{k}={v}" for k, v in details.items())
            message += f" ({detail_str})"
        
        if success:
            self.logger.info(message)
        else:
            self.logger.error(message)
    
    def log_error(self, error: Exception, context: str = "", extra_info: Dict = None):
        """Log error with full context"""
        extra_info = extra_info or {}
        
        error_msg = f"ERROR in {context}: {str(error)}"
        self.logger.error(error_msg)
        
        if extra_info:
            for key, value in extra_info.items():
                self.logger.error(f"  {key}: {value}")
        
        # Log full traceback
        self.logger.error(f"Traceback: {traceback.format_exc()}")

@dataclass
class RAGConfig:
    """RAG system configuration with validation"""
    embedding_model: str = "all-MiniLM-L6-v2"
    vector_db_path: str = "data/vectordb"
    chunk_size: int = 256
    chunk_overlap: int = 25
    top_k: int = 10
    similarity_threshold: float = 0.2
    max_context_length: int = 2048
    max_documents: int = 10000
    embedding_dim: Optional[int] = None
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate configuration parameters"""
        errors = []
        
        if self.chunk_size <= 0:
            errors.append("chunk_size must be positive")
        
        if self.chunk_overlap < 0 or self.chunk_overlap >= self.chunk_size:
            errors.append("chunk_overlap must be non-negative and less than chunk_size")
        
        if self.top_k <= 0:
            errors.append("top_k must be positive")
        
        if not (0.0 <= self.similarity_threshold <= 1.0):
            errors.append("similarity_threshold must be between 0.0 and 1.0")
        
        if self.max_documents <= 0:
            errors.append("max_documents must be positive")
        
        return len(errors) == 0, errors

class EnhancedVectorStore:
    """Enhanced vector store with comprehensive error handling"""
    
    def __init__(self, config: RAGConfig, logger: RAGLogger):
        self.config = config
        self.logger = logger
        
        # Initialize properties
        self.embedding_dim = config.embedding_dim
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, Any] = {}
        self.document_ids: List[str] = []
        
        # Performance tracking
        self.operation_stats = {
            "documents_added": 0,
            "searches_performed": 0,
            "errors_occurred": 0,
            "last_operation_time": 0
        }
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index with error handling"""
        try:
            if not self.embedding_dim:
                raise RAGError("Embedding dimension not set", "initialization")
            
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            
            self.logger.log_operation(
                "Vector store initialization", 
                True,
                {"embedding_dim": self.embedding_dim, "index_type": "IndexFlatIP"}
            )
            
        except Exception as e:
            self.logger.log_error(e, "vector store initialization")
            raise RAGError(f"Failed to initialize vector store: {e}", "initialization")
    
    def add_documents(self, documents: List[Dict], embeddings: np.ndarray) -> bool:
        """Add documents with comprehensive error handling"""
        operation_start = time.time()
        
        try:
            # Validate inputs
            if not documents:
                raise RAGError("No documents provided", "validation")
            
            if embeddings is None or len(embeddings) == 0:
                raise RAGError("No embeddings provided", "validation")
            
            if len(documents) != len(embeddings):
                raise RAGError(
                    f"Document count ({len(documents)}) doesn't match embedding count ({len(embeddings)})",
                    "validation"
                )
            
            # Validate embedding dimensions
            if embeddings.shape[1] != self.embedding_dim:
                raise RAGError(
                    f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings.shape[1]}",
                    "dimension_mismatch"
                )
            
            # Prepare embeddings
            embeddings_array = embeddings.astype(np.float32)
            faiss.normalize_L2(embeddings_array)
            
            # Add to index
            start_index = self.index.ntotal
            self.index.add(embeddings_array)
            
            # Store documents
            for i, doc in enumerate(documents):
                doc_id = doc.get('id', f"doc_{start_index + i}")
                self.documents[doc_id] = doc
                self.document_ids.append(doc_id)
            
            # Update stats
            self.operation_stats["documents_added"] += len(documents)
            self.operation_stats["last_operation_time"] = time.time()
            
            operation_time = time.time() - operation_start
            
            self.logger.log_operation(
                "Document addition",
                True,
                {
                    "documents_added": len(documents),
                    "total_documents": self.index.ntotal,
                    "operation_time": f"{operation_time:.3f}s"
                }
            )
            
            return True
            
        except RAGError:
            self.operation_stats["errors_occurred"] += 1
            raise
        except Exception as e:
            self.operation_stats["errors_occurred"] += 1
            self.logger.log_error(e, "document addition", {
                "documents_count": len(documents) if documents else 0,
                "embeddings_shape": embeddings.shape if embeddings is not None else "None"
            })
            raise RAGError(f"Failed to add documents: {e}", "addition_error")
    
    def search(self, query_embedding: np.ndarray, top_k: int = None) -> List[Dict]:
        """Search with enhanced error handling and diagnostics"""
        operation_start = time.time()
        
        try:
            # Validate inputs
            if query_embedding is None:
                raise RAGError("No query embedding provided", "validation")
            
            if self.index.ntotal == 0:
                self.logger.logger.warning("No documents in vector store")
                return []
            
            # Prepare query
            query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
            
            if query_embedding.shape[1] != self.embedding_dim:
                raise RAGError(
                    f"Query embedding dimension mismatch: expected {self.embedding_dim}, got {query_embedding.shape[1]}",
                    "dimension_mismatch"
                )
            
            faiss.normalize_L2(query_embedding)
            
            # Perform search
            k = min(top_k or self.config.top_k, self.index.ntotal)
            scores, indices = self.index.search(query_embedding, k)
            
            # Process results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx == -1:  # Invalid index
                    continue
                
                if score < self.config.similarity_threshold:
                    continue
                
                if idx >= len(self.document_ids):
                    self.logger.logger.warning(f"Index {idx} out of range")
                    continue
                
                doc_id = self.document_ids[idx]
                if doc_id not in self.documents:
                    self.logger.logger.warning(f"Document {doc_id} not found")
                    continue
                
                results.append({
                    "document": self.documents[doc_id],
                    "score": float(score),
                    "relevance": float(score)
                })
            
            # Update stats
            self.operation_stats["searches_performed"] += 1
            self.operation_stats["last_operation_time"] = time.time()
            
            operation_time = time.time() - operation_start
            
            self.logger.log_operation(
                "Vector search",
                True,
                {
                    "results_found": len(results),
                    "search_k": k,
                    "operation_time": f"{operation_time:.3f}s"
                }
            )
            
            return results
            
        except RAGError:
            self.operation_stats["errors_occurred"] += 1
            raise
        except Exception as e:
            self.operation_stats["errors_occurred"] += 1
            self.logger.log_error(e, "vector search", {
                "query_shape": query_embedding.shape if query_embedding is not None else "None",
                "index_size": self.index.ntotal if self.index else "None"
            })
            raise RAGError(f"Search failed: {e}", "search_error")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        return {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal if self.index else 0,
            "embedding_dim": self.embedding_dim,
            "operations": self.operation_stats.copy(),
            "config": {
                "max_documents": self.config.max_documents,
                "similarity_threshold": self.config.similarity_threshold,
                "top_k": self.config.top_k
            }
        }

class EnhancedRAGSystem(QObject):
    """Enhanced RAG system with comprehensive error handling"""
    
    # Signals
    document_added = pyqtSignal(str)
    search_completed = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    status_updated = pyqtSignal(dict)
    
    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__()
        
        # Extract non-RAGConfig parameters
        log_file = config_dict.pop('log_file', None)
        
        # Initialize configuration
        self.config = RAGConfig(**config_dict)
        
        # Validate configuration
        config_valid, config_errors = self.config.validate()
        if not config_valid:
            raise RAGError(f"Invalid configuration: {', '.join(config_errors)}", "configuration")
        
        # Initialize logger
        log_file = log_file or "logs/rag_system.log"
        self.logger = RAGLogger(f"{__name__}.EnhancedRAGSystem", log_file)
        
        # Initialize components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.vector_store: Optional[EnhancedVectorStore] = None
        self.is_ready = False
        self.mutex = QMutex()
        
        # System stats
        self.system_stats = {
            "initialization_time": time.time(),
            "documents_processed": 0,
            "searches_performed": 0,
            "errors_encountered": 0
        }
        
        # Initialize system
        self._initialize_system()
    
    def _initialize_system(self):
        """Initialize the RAG system with comprehensive error handling"""
        try:
            self.logger.logger.info("Initializing Enhanced RAG System")
            
            # Initialize embedding model
            self._initialize_embedding_model()
            
            # Initialize vector store
            self._initialize_vector_store()
            
            # Try to load existing data
            self._load_existing_data()
            
            self.is_ready = True
            
            init_time = time.time() - self.system_stats["initialization_time"]
            
            self.logger.log_operation(
                "RAG system initialization",
                True,
                {
                    "model": self.config.embedding_model,
                    "embedding_dim": self.config.embedding_dim,
                    "initialization_time": f"{init_time:.3f}s"
                }
            )
            
            # Emit status update
            self.status_updated.emit(self.get_system_status())
            
        except Exception as e:
            self.system_stats["errors_encountered"] += 1
            self.logger.log_error(e, "system initialization")
            self.error_occurred.emit(f"RAG system initialization failed: {e}")
            raise
    
    def _initialize_embedding_model(self):
        """Initialize embedding model with fallback options"""
        try:
            self.logger.logger.info(f"Loading embedding model: {self.config.embedding_model}")
            
            # Try primary model
            try:
                self.embedding_model = SentenceTransformer(
                    self.config.embedding_model,
                    trust_remote_code=True,
                    device='cpu'
                )
                
                # Test model
                test_embedding = self.embedding_model.encode("Test sentence")
                self.config.embedding_dim = len(test_embedding)
                
                self.logger.log_operation(
                    "Primary embedding model load",
                    True,
                    {"model": self.config.embedding_model, "dimension": self.config.embedding_dim}
                )
                
            except Exception as primary_error:
                self.logger.logger.warning(f"Primary model failed: {primary_error}")
                
                # Try fallback model
                fallback_model = "all-MiniLM-L6-v2"
                self.logger.logger.info(f"Trying fallback model: {fallback_model}")
                
                self.embedding_model = SentenceTransformer(fallback_model, device='cpu')
                test_embedding = self.embedding_model.encode("Test sentence")
                self.config.embedding_dim = len(test_embedding)
                self.config.embedding_model = fallback_model
                
                self.logger.log_operation(
                    "Fallback embedding model load",
                    True,
                    {"model": fallback_model, "dimension": self.config.embedding_dim}
                )
                
        except Exception as e:
            raise RAGError(f"Failed to initialize embedding model: {e}", "model_initialization")
    
    def _initialize_vector_store(self):
        """Initialize vector store"""
        try:
            self.vector_store = EnhancedVectorStore(self.config, self.logger)
            
        except Exception as e:
            raise RAGError(f"Failed to initialize vector store: {e}", "vector_store_initialization")
    
    def _load_existing_data(self):
        """Load existing vector store data if available"""
        try:
            vector_db_path = Path(self.config.vector_db_path)
            index_file = vector_db_path.with_suffix('.index')
            docs_file = vector_db_path.with_suffix('.docs')
            
            if index_file.exists() and docs_file.exists():
                self.logger.logger.info("Loading existing vector store...")
                
                # Load FAISS index
                self.vector_store.index = faiss.read_index(str(index_file))
                
                # Load documents
                with open(docs_file, 'rb') as f:
                    data = pickle.load(f)
                    self.vector_store.documents = data.get("documents", {})
                    self.vector_store.document_ids = data.get("document_ids", [])
                
                self.logger.log_operation(
                    "Existing data load",
                    True,
                    {"documents_loaded": len(self.vector_store.documents)}
                )
            else:
                self.logger.logger.info("No existing vector store found, starting fresh")
                
        except Exception as e:
            self.logger.logger.warning(f"Failed to load existing data: {e}")
            # Continue with fresh store
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document with enhanced error handling"""
        try:
            self.mutex.lock()
            
            if not self.is_ready:
                raise RAGError("RAG system not ready", "system_not_ready")
            
            if not content or not content.strip():
                raise RAGError("Empty content provided", "validation")
            
            # Process document
            documents = self._process_document(content, metadata)
            
            if not documents:
                raise RAGError("No valid chunks created from document", "processing")
            
            # Generate embeddings
            embeddings = self._generate_embeddings([doc['content'] for doc in documents])
            
            # Add to vector store
            success = self.vector_store.add_documents(documents, embeddings)
            
            if success:
                self.system_stats["documents_processed"] += 1
                self.document_added.emit(documents[0]['id'])
                
                # Save data
                self._save_data()
            
            return success
            
        except RAGError:
            self.system_stats["errors_encountered"] += 1
            raise
        except Exception as e:
            self.system_stats["errors_encountered"] += 1
            self.logger.log_error(e, "document addition")
            self.error_occurred.emit(f"Document addition failed: {e}")
            return False
        finally:
            self.mutex.unlock()
    
    def search(self, query: str, top_k: int = None) -> List[Dict]:
        """Search with enhanced error handling"""
        try:
            if not self.is_ready:
                raise RAGError("RAG system not ready", "system_not_ready")
            
            if not query or not query.strip():
                raise RAGError("Empty query provided", "validation")
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k)
            
            self.system_stats["searches_performed"] += 1
            self.search_completed.emit(results)
            
            return results
            
        except RAGError:
            self.system_stats["errors_encountered"] += 1
            raise
        except Exception as e:
            self.system_stats["errors_encountered"] += 1
            self.logger.log_error(e, "search")
            self.error_occurred.emit(f"Search failed: {e}")
            return []
    
    def _process_document(self, content: str, metadata: Dict = None) -> List[Dict]:
        """Process document into chunks"""
        metadata = metadata or {}
        
        # Simple chunking for now
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        
        for i in range(0, len(content), chunk_size - overlap):
            chunk_content = content[i:i + chunk_size]
            
            if chunk_content.strip():
                chunk_id = f"chunk_{hashlib.md5(chunk_content.encode()).hexdigest()[:8]}"
                
                chunks.append({
                    'id': chunk_id,
                    'content': chunk_content.strip(),
                    'metadata': {**metadata, 'chunk_index': len(chunks)}
                })
        
        return chunks
    
    def _generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings with error handling"""
        try:
            embeddings = self.embedding_model.encode(texts)
            return np.array(embeddings)
            
        except Exception as e:
            raise RAGError(f"Failed to generate embeddings: {e}", "embedding_generation")
    
    def _save_data(self):
        """Save vector store data"""
        try:
            vector_db_path = Path(self.config.vector_db_path)
            vector_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save index
            faiss.write_index(self.vector_store.index, str(vector_db_path.with_suffix('.index')))
            
            # Save documents
            data = {
                "documents": self.vector_store.documents,
                "document_ids": self.vector_store.document_ids,
                "config": {
                    "embedding_dim": self.config.embedding_dim,
                    "model": self.config.embedding_model
                }
            }
            
            with open(vector_db_path.with_suffix('.docs'), 'wb') as f:
                pickle.dump(data, f)
            
        except Exception as e:
            self.logger.logger.warning(f"Failed to save data: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        status = {
            "is_ready": self.is_ready,
            "config": {
                "embedding_model": self.config.embedding_model,
                "embedding_dim": self.config.embedding_dim,
                "similarity_threshold": self.config.similarity_threshold
            },
            "system_stats": self.system_stats.copy(),
            "vector_store_stats": self.vector_store.get_stats() if self.vector_store else {}
        }
        
        # Calculate uptime
        status["uptime"] = time.time() - self.system_stats["initialization_time"]
        
        return status
    
    def get_diagnostics(self) -> Dict[str, Any]:
        """Get detailed diagnostics for troubleshooting"""
        diagnostics = {
            "system_status": self.get_system_status(),
            "configuration": {
                "embedding_model": self.config.embedding_model,
                "vector_db_path": self.config.vector_db_path,
                "chunk_size": self.config.chunk_size,
                "chunk_overlap": self.config.chunk_overlap,
                "similarity_threshold": self.config.similarity_threshold,
                "top_k": self.config.top_k
            },
            "performance": {
                "total_operations": (
                    self.system_stats["documents_processed"] + 
                    self.system_stats["searches_performed"]
                ),
                "error_rate": (
                    self.system_stats["errors_encountered"] / 
                    max(1, self.system_stats["documents_processed"] + self.system_stats["searches_performed"])
                )
            }
        }
        
        return diagnostics