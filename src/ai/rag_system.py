"""
RAG (Retrieval-Augmented Generation) System for Jarvis Voice Assistant
Handles local knowledge base and context retrieval
"""

import os
import json
import logging
import pickle
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex
import hashlib
import time

# Enhanced logging
try:
    from src.system.enhanced_logger import ComponentLogger
    ENHANCED_LOGGING = True
except ImportError:
    ENHANCED_LOGGING = False


@dataclass
class Document:
    """Document structure for RAG system"""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[np.ndarray] = None
    created_at: float = 0.0
    updated_at: float = 0.0
    
    def __post_init__(self):
        if self.created_at == 0.0:
            self.created_at = time.time()
        if self.updated_at == 0.0:
            self.updated_at = time.time()


@dataclass
class SearchResult:
    """Search result structure"""
    document: Document
    score: float
    relevance: float


class DocumentProcessor:
    """Document processing utilities"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[Document]:
        """Split text into chunks"""
        if not text.strip():
            return []
        
        metadata = metadata or {}
        chunks = []
        
        # Simple sentence-based chunking
        sentences = self._split_into_sentences(text)
        current_chunk = ""
        chunk_count = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.chunk_size and current_chunk:
                # Create chunk
                chunk_id = self._generate_chunk_id(current_chunk, chunk_count)
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    "chunk_index": chunk_count,
                    "chunk_type": "text",
                    "original_length": len(text),
                    "chunk_length": len(current_chunk)
                })
                
                chunks.append(Document(
                    id=chunk_id,
                    content=current_chunk.strip(),
                    metadata=chunk_metadata
                ))
                
                # Handle overlap
                if self.chunk_overlap > 0:
                    overlap_text = current_chunk[-self.chunk_overlap:] if len(current_chunk) > self.chunk_overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk = sentence
                    
                chunk_count += 1
            else:
                current_chunk += " " + sentence if current_chunk else sentence
        
        # Add final chunk
        if current_chunk.strip():
            chunk_id = self._generate_chunk_id(current_chunk, chunk_count)
            chunk_metadata = metadata.copy()
            chunk_metadata.update({
                "chunk_index": chunk_count,
                "chunk_type": "text",
                "original_length": len(text),
                "chunk_length": len(current_chunk)
            })
            
            chunks.append(Document(
                id=chunk_id,
                content=current_chunk.strip(),
                metadata=chunk_metadata
            ))
        
        self.logger.info(f"Created {len(chunks)} chunks from {len(text)} characters")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        import re
        
        # Simple sentence splitting (improved for Thai and English)
        sentence_endings = r'[.!?]|[।]|[។]'
        sentences = re.split(sentence_endings, text)
        
        # Clean and filter sentences
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Handle Thai text (basic support)
        thai_sentences = []
        for sentence in sentences:
            # Split long sentences for Thai
            if len(sentence) > 200 and self._contains_thai(sentence):
                # Simple splitting on Thai sentence markers
                thai_parts = re.split(r'[,;]', sentence)
                thai_sentences.extend([p.strip() for p in thai_parts if p.strip()])
            else:
                thai_sentences.append(sentence)
        
        return thai_sentences
    
    def _contains_thai(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        thai_range = range(0x0E00, 0x0E7F)
        return any(ord(char) in thai_range for char in text)
    
    def _generate_chunk_id(self, content: str, chunk_index: int) -> str:
        """Generate unique chunk ID"""
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"chunk_{chunk_index}_{content_hash}"


class VectorStore:
    """Vector storage and retrieval system with memory management"""
    
    def __init__(self, embedding_dim: int = 1024, index_type: str = "flat", max_documents: int = 10000):
        self.embedding_dim = embedding_dim
        self.index_type = index_type
        self.max_documents = max_documents
        self.logger = logging.getLogger(__name__)
        
        # FAISS index
        self.index: Optional[faiss.Index] = None
        self.documents: Dict[str, Document] = {}
        self.document_ids: List[str] = []
        
        # Memory management
        self._memory_pressure_threshold = 0.8  # 80% of max_documents
        self._cleanup_batch_size = 100
        self._last_cleanup_time = time.time()
        self._cleanup_interval = 300  # 5 minutes
        
        # Initialize index
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index"""
        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product for similarity
        elif self.index_type == "ivf":
            # IVF index for larger datasets
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, 100)
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
        
        self.logger.info(f"Initialized FAISS index: {self.index_type}")
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store with memory management"""
        if not documents:
            return
        
        # Check if we need cleanup before adding
        self._check_memory_pressure()
        
        embeddings = []
        docs_to_add = []
        
        for doc in documents:
            if doc.embedding is not None:
                # Check for memory limits
                if len(self.documents) >= self.max_documents:
                    self.logger.warning("Document limit reached, triggering cleanup")
                    self._cleanup_old_documents()
                    
                embeddings.append(doc.embedding)
                docs_to_add.append(doc)
            else:
                self.logger.warning(f"Document {doc.id} has no embedding, skipping")
        
        if embeddings:
            try:
                embeddings_array = np.array(embeddings).astype(np.float32)
                
                # Validate embedding dimensions
                if embeddings_array.shape[1] != self.embedding_dim:
                    raise ValueError(f"Embedding dimension mismatch: expected {self.embedding_dim}, got {embeddings_array.shape[1]}")
                
                # Normalize embeddings for cosine similarity
                faiss.normalize_L2(embeddings_array)
                
                # Train index if needed (before adding documents)
                if hasattr(self.index, 'is_trained') and not self.index.is_trained:
                    self.logger.info("Training FAISS index...")
                    self.index.train(embeddings_array)
                    self.logger.info("FAISS index training completed")
                
                # Add to index
                self.index.add(embeddings_array)
                
                # Store documents
                for doc in docs_to_add:
                    self.documents[doc.id] = doc
                    self.document_ids.append(doc.id)
                
                self.logger.info(f"Added {len(docs_to_add)} documents to vector store (total: {self.index.ntotal})")
                
            except Exception as e:
                import traceback
                self.logger.error(f"Failed to add documents to vector store: {e}")
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                
                # Cleanup on failure
                for doc in docs_to_add:
                    if doc.id in self.documents:
                        del self.documents[doc.id]
                    if doc.id in self.document_ids:
                        self.document_ids.remove(doc.id)
                raise
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               similarity_threshold: float = 0.7) -> List[SearchResult]:
        """Search for similar documents"""
        if self.index.ntotal == 0:
            self.logger.warning("No documents in vector store")
            return []
        
        # Normalize query embedding
        query_embedding = query_embedding.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if idx == -1:  # Invalid index
                continue
                
            # Check if idx is within valid range
            if idx >= len(self.document_ids):
                self.logger.warning(f"Index {idx} out of range for document_ids (size: {len(self.document_ids)})")
                continue
                
            # For inner product similarity, higher scores are better
            # Convert to similarity score (cosine similarity ranges from -1 to 1)
            similarity = float(score)
            
            if similarity < similarity_threshold:
                continue
            
            doc_id = self.document_ids[idx]
            if doc_id not in self.documents:
                self.logger.warning(f"Document ID {doc_id} not found in documents")
                continue
                
            document = self.documents[doc_id]
            
            # Calculate relevance (can be improved with more sophisticated scoring)
            relevance = float(score)
            
            results.append(SearchResult(
                document=document,
                score=float(score),
                relevance=relevance
            ))
        
        self.logger.info(f"Found {len(results)} relevant documents")
        return results
    
    def get_document(self, doc_id: str) -> Optional[Document]:
        """Get document by ID"""
        return self.documents.get(doc_id)
    
    def remove_document(self, doc_id: str) -> bool:
        """Remove document (note: FAISS doesn't support removal, needs rebuild)"""
        if doc_id in self.documents:
            del self.documents[doc_id]
            # Note: Would need to rebuild index for complete removal
            return True
        return False
    
    def _check_memory_pressure(self):
        """Check and handle memory pressure"""
        current_time = time.time()
        
        # Periodic cleanup
        if current_time - self._last_cleanup_time > self._cleanup_interval:
            self._cleanup_old_documents()
            self._last_cleanup_time = current_time
        
        # Check memory pressure
        if len(self.documents) > self.max_documents * self._memory_pressure_threshold:
            self.logger.warning(f"Memory pressure detected: {len(self.documents)}/{self.max_documents} documents")
            self._cleanup_old_documents()
    
    def _cleanup_old_documents(self):
        """Clean up old documents to free memory"""
        try:
            if len(self.documents) <= self._cleanup_batch_size:
                return
            
            # Sort documents by age (oldest first)
            sorted_docs = sorted(
                self.documents.items(),
                key=lambda x: x[1].created_at
            )
            
            # Remove oldest documents
            docs_to_remove = sorted_docs[:self._cleanup_batch_size]
            removed_count = 0
            
            for doc_id, doc in docs_to_remove:
                try:
                    # Remove from documents dict
                    del self.documents[doc_id]
                    
                    # Remove from document_ids list
                    if doc_id in self.document_ids:
                        self.document_ids.remove(doc_id)
                    
                    # Clear document embeddings to free memory
                    if hasattr(doc, 'embedding') and doc.embedding is not None:
                        del doc.embedding
                    
                    removed_count += 1
                    
                except Exception as e:
                    self.logger.error(f"Error removing document {doc_id}: {e}")
            
            self.logger.info(f"Cleaned up {removed_count} old documents, {len(self.documents)} remaining")
            
            # Note: FAISS index rebuild would be needed for complete cleanup
            # For now, we just clean the document storage
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def clear_all_documents(self):
        """Clear all documents and rebuild index"""
        try:
            # Clear document storage
            self.documents.clear()
            self.document_ids.clear()
            
            # Rebuild empty index
            self._initialize_index()
            
            self.logger.info("Cleared all documents and rebuilt index")
            
        except Exception as e:
            self.logger.error(f"Error clearing documents: {e}")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics"""
        import sys
        
        # Calculate approximate memory usage
        documents_size = sum(sys.getsizeof(doc) for doc in self.documents.values())
        embeddings_size = sum(
            doc.embedding.nbytes if doc.embedding is not None else 0 
            for doc in self.documents.values()
        )
        
        return {
            "total_documents": len(self.documents),
            "max_documents": self.max_documents,
            "memory_pressure": len(self.documents) / self.max_documents,
            "documents_memory_mb": documents_size / (1024 * 1024),
            "embeddings_memory_mb": embeddings_size / (1024 * 1024),
            "last_cleanup": self._last_cleanup_time,
            "cleanup_interval": self._cleanup_interval
        }

    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        base_stats = {
            "total_documents": len(self.documents),
            "index_size": self.index.ntotal,
            "embedding_dim": self.embedding_dim,
            "index_type": self.index_type
        }
        
        # Add memory stats
        base_stats.update(self.get_memory_stats())
        return base_stats
    
    def save(self, path: str):
        """Save vector store to disk"""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.index")
            
            # Save documents and metadata
            with open(f"{path}.docs", "wb") as f:
                pickle.dump({
                    "documents": self.documents,
                    "document_ids": self.document_ids,
                    "embedding_dim": self.embedding_dim,
                    "index_type": self.index_type
                }, f)
            
            self.logger.info(f"Saved vector store to {path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save vector store: {e}")
    
    def load(self, path: str) -> bool:
        """Load vector store from disk"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.index")
            
            # Load documents and metadata
            with open(f"{path}.docs", "rb") as f:
                data = pickle.load(f)
                self.documents = data["documents"]
                self.document_ids = data["document_ids"]
                self.embedding_dim = data["embedding_dim"]
                self.index_type = data["index_type"]
            
            self.logger.info(f"Loaded vector store from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load vector store: {e}")
            return False


class RAGSystem(QObject):
    """Main RAG system controller"""
    
    # Signals
    document_added = pyqtSignal(str)  # document_id
    search_completed = pyqtSignal(list)  # search_results
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration
        self.embedding_model_name = config.get("embedding_model", "mixedbread-ai/mxbai-embed-large-v1")
        self.vector_db_path = config.get("vector_db_path", "data/vectordb")
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.top_k = config.get("top_k", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        
        # Components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.document_processor = DocumentProcessor(self.chunk_size, self.chunk_overlap)
        self.vector_store: Optional[VectorStore] = None
        
        # State
        self.is_ready = False
        self.mutex = QMutex()
        
        # Initialize
        self._initialize()
    
    def __del__(self):
        """Cleanup resources when object is destroyed"""
        try:
            self.cleanup()
        except:
            pass  # Silent cleanup on destruction
    
    def cleanup(self):
        """Explicit cleanup method for memory management"""
        try:
            self.logger.info("Cleaning up RAG system resources...")
            
            # Save current state
            if self.vector_store and self.is_ready:
                try:
                    self.vector_store.save(self.vector_db_path)
                except Exception as e:
                    self.logger.error(f"Failed to save vector store during cleanup: {e}")
            
            # Clear vector store
            if self.vector_store:
                self.vector_store.clear_all_documents()
                self.vector_store = None
            
            # Clear embedding model
            if self.embedding_model:
                del self.embedding_model
                self.embedding_model = None
            
            # Clear document processor
            if hasattr(self, 'document_processor'):
                del self.document_processor
            
            self.is_ready = False
            self.logger.info("RAG system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during RAG system cleanup: {e}")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get current memory usage statistics"""
        if not self.vector_store:
            return {"error": "Vector store not initialized"}
        
        return self.vector_store.get_memory_stats()
    
    def force_cleanup(self):
        """Force cleanup of old documents"""
        if self.vector_store:
            self.vector_store._cleanup_old_documents()
            self.logger.info("Forced cleanup completed")
    
    def _initialize(self):
        """Initialize RAG system with comprehensive error handling"""
        try:
            self.logger.info(f"Initializing RAG system with model: {self.embedding_model_name}")
            
            # Create necessary directories
            os.makedirs(os.path.dirname(self.vector_db_path), exist_ok=True)
            
            # Load embedding model with detailed logging
            self.logger.info("Loading embedding model...")
            try:
                # Try multiple initialization methods for robustness
                self.embedding_model = SentenceTransformer(
                    self.embedding_model_name,
                    trust_remote_code=True,
                    device='cpu'  # Start with CPU, move to GPU if available later
                )
                
                # Test the model with a simple sentence to ensure it works
                test_embedding = self.embedding_model.encode("This is a test sentence.")
                embedding_dim = len(test_embedding)
                
                self.logger.info(f"Embedding model loaded successfully")
                self.logger.info(f"Model dimension: {embedding_dim}")
                self.logger.info(f"Test embedding shape: {test_embedding.shape}")
                
            except Exception as model_error:
                import traceback
                self.logger.error(f"Failed to load embedding model '{self.embedding_model_name}': {model_error}")
                self.logger.error(f"Model loading traceback: {traceback.format_exc()}")
                
                # Try fallback model
                fallback_model = "all-MiniLM-L6-v2"
                self.logger.info(f"Attempting to load fallback model: {fallback_model}")
                
                try:
                    self.embedding_model = SentenceTransformer(fallback_model, device='cpu')
                    test_embedding = self.embedding_model.encode("This is a test sentence.")
                    embedding_dim = len(test_embedding)
                    self.embedding_model_name = fallback_model  # Update config
                    
                    self.logger.warning(f"Using fallback embedding model: {fallback_model}")
                    self.logger.info(f"Fallback model dimension: {embedding_dim}")
                    
                except Exception as fallback_error:
                    self.logger.error(f"Fallback model also failed: {fallback_error}")
                    raise Exception(f"Both primary and fallback embedding models failed: {model_error}, {fallback_error}")
            
            # Initialize vector store with correct dimension
            self.logger.info(f"Initializing vector store with dimension: {embedding_dim}")
            self.vector_store = VectorStore(embedding_dim)
            
            # Try to load existing vector store with dimension compatibility check
            if os.path.exists(f"{self.vector_db_path}.index") and os.path.exists(f"{self.vector_db_path}.docs"):
                self.logger.info("Attempting to load existing vector store...")
                try:
                    if self.vector_store.load(self.vector_db_path):
                        # Verify dimension compatibility
                        if self.vector_store.embedding_dim == embedding_dim:
                            self.logger.info(f"Existing vector store loaded successfully with {self.vector_store.index.ntotal} documents")
                        else:
                            self.logger.warning(f"Dimension mismatch: stored {self.vector_store.embedding_dim} vs current {embedding_dim}")
                            self.logger.info("Rebuilding vector store with new dimensions...")
                            self.vector_store = VectorStore(embedding_dim)
                    else:
                        self.logger.warning("Failed to load existing vector store, starting fresh")
                        self.vector_store = VectorStore(embedding_dim)
                        
                except Exception as load_error:
                    self.logger.warning(f"Error loading existing vector store: {load_error}")
                    self.logger.info("Starting with fresh vector store")
                    self.vector_store = VectorStore(embedding_dim)
            else:
                self.logger.info("No existing vector store found, starting fresh")
            
            self.is_ready = True
            total_docs = self.vector_store.index.ntotal if self.vector_store else 0
            self.logger.info(f"RAG system initialized successfully with {total_docs} documents")
            
            # Log system configuration
            self.logger.info(f"RAG Configuration:")
            self.logger.info(f"  - Model: {self.embedding_model_name}")
            self.logger.info(f"  - Embedding dimension: {embedding_dim}")
            self.logger.info(f"  - Chunk size: {self.chunk_size}")
            self.logger.info(f"  - Chunk overlap: {self.chunk_overlap}")
            self.logger.info(f"  - Similarity threshold: {self.similarity_threshold}")
            self.logger.info(f"  - Vector DB path: {self.vector_db_path}")
            
        except Exception as e:
            import traceback
            error_msg = f"Failed to initialize RAG system: {e}"
            full_traceback = traceback.format_exc()
            
            self.logger.error(error_msg)
            self.logger.error(f"Full traceback: {full_traceback}")
            
            # Emit error signal
            self.error_occurred.emit(error_msg)
            
            # Set system to not ready
            self.is_ready = False
            
            # Try to provide helpful error information
            if "No module named" in str(e):
                self.logger.error("Missing dependencies detected. Please install required packages:")
                self.logger.error("  pip install sentence-transformers faiss-cpu torch")
            elif "CUDA" in str(e) or "GPU" in str(e):
                self.logger.error("GPU-related error detected. The system will try to use CPU fallback.")
            elif "disk" in str(e).lower() or "space" in str(e).lower():
                self.logger.error("Disk space or permission error detected. Check available disk space and write permissions.")
            
            raise e
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to knowledge base"""
        if not self.is_ready:
            self.logger.warning("RAG system not ready for document addition")
            return False
        
        if not content or not content.strip():
            self.logger.warning("Empty content provided for document addition")
            return False
        
        try:
            # Use QMutex properly
            self.mutex.lock()
            try:
                self.logger.debug(f"Processing document: {len(content)} characters")
                
                # Process document into chunks
                chunks = self.document_processor.chunk_text(content, metadata)
                
                if not chunks:
                    self.logger.warning("No chunks created from document")
                    return False
                
                self.logger.debug(f"Created {len(chunks)} chunks from document")
                
                # Generate embeddings for chunks with detailed error handling
                valid_chunks = []
                for i, chunk in enumerate(chunks):
                    try:
                        self.logger.debug(f"Generating embedding for chunk {i+1}/{len(chunks)}")
                        embedding = self.embedding_model.encode(chunk.content)
                        chunk.embedding = embedding
                        valid_chunks.append(chunk)
                        self.logger.debug(f"Generated embedding shape: {embedding.shape}")
                    except Exception as embed_error:
                        import traceback
                        self.logger.error(f"Failed to generate embedding for chunk {i+1}: {embed_error}")
                        self.logger.error(f"Chunk content length: {len(chunk.content)}")
                        self.logger.error(f"Traceback: {traceback.format_exc()}")
                        continue
                
                if not valid_chunks:
                    self.logger.error("No valid chunks with embeddings created")
                    return False
                
                self.logger.info(f"Generated embeddings for {len(valid_chunks)}/{len(chunks)} chunks")
                
                # Add to vector store
                self.logger.debug("Adding chunks to vector store...")
                self.vector_store.add_documents(valid_chunks)
                
                # Save vector store
                try:
                    self.logger.debug("Saving vector store...")
                    self.vector_store.save(self.vector_db_path)
                    self.logger.debug("Vector store saved successfully")
                except Exception as save_error:
                    self.logger.warning(f"Failed to save vector store: {save_error}")
                
                self.logger.info(f"Successfully added document with {len(valid_chunks)} chunks")
                self.document_added.emit(valid_chunks[0].id)
                
                return True
            finally:
                self.mutex.unlock()
                
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to add document: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.error_occurred.emit(f"Document addition failed: {e}")
            return False
    
    def search(self, query: str, top_k: Optional[int] = None, 
               similarity_threshold: Optional[float] = None) -> List[SearchResult]:
        """Search for relevant documents"""
        if not self.is_ready:
            self.logger.warning("RAG system not ready")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding,
                top_k=top_k or self.top_k,
                similarity_threshold=similarity_threshold or self.similarity_threshold
            )
            
            self.logger.info(f"Search query: '{query}' returned {len(results)} results")
            self.search_completed.emit(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            self.error_occurred.emit(f"Search failed: {e}")
            return []
    
    def get_context(self, query: str, max_context_length: int = 2000) -> str:
        """Get relevant context for query"""
        results = self.search(query)
        
        if not results:
            return ""
        
        # Combine relevant documents
        context_parts = []
        current_length = 0
        
        for result in results:
            doc_content = result.document.content
            
            # Check if adding this document would exceed max length
            if current_length + len(doc_content) > max_context_length:
                # Add partial content if possible
                remaining_length = max_context_length - current_length
                if remaining_length > 100:  # Only add if meaningful
                    context_parts.append(doc_content[:remaining_length] + "...")
                break
            
            context_parts.append(doc_content)
            current_length += len(doc_content)
        
        context = "\n\n".join(context_parts)
        self.logger.info(f"Generated context of {len(context)} characters from {len(context_parts)} documents")
        
        return context
    
    def add_knowledge_base(self, knowledge_data: Dict[str, Any]):
        """Add structured knowledge base data"""
        if not self.is_ready:
            self.logger.warning("RAG system not ready for knowledge base loading")
            return
            
        try:
            added_count = 0
            for category, items in knowledge_data.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            content = item.get("content", "")
                            if content.strip():  # Only add non-empty content
                                metadata = item.copy()
                                metadata["category"] = category
                                if self.add_document(content, metadata):
                                    added_count += 1
                elif isinstance(items, dict):
                    for key, value in items.items():
                        if isinstance(value, str) and value.strip():  # Only add non-empty strings
                            metadata = {"category": category, "key": key}
                            if self.add_document(value, metadata):
                                added_count += 1
            
            self.logger.info(f"Successfully added {added_count} documents from knowledge base with {len(knowledge_data)} categories")
            
        except Exception as e:
            self.logger.error(f"Failed to add knowledge base: {e}")
            if hasattr(self, 'error_occurred'):
                self.error_occurred.emit(f"Knowledge base addition failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics"""
        stats = {
            "is_ready": self.is_ready,
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold
        }
        
        if self.vector_store:
            stats.update(self.vector_store.get_stats())
        
        return stats
    
    def clear_knowledge_base(self):
        """Clear all documents from knowledge base"""
        try:
            if self.vector_store:
                embedding_dim = self.vector_store.embedding_dim
                self.vector_store = VectorStore(embedding_dim)
                self.vector_store.save(self.vector_db_path)
                
            self.logger.info("Knowledge base cleared")
            
        except Exception as e:
            self.logger.error(f"Failed to clear knowledge base: {e}")
            self.error_occurred.emit(f"Knowledge base clear failed: {e}")
    
    def shutdown(self):
        """Shutdown RAG system"""
        self.logger.info("Shutting down RAG system")
        
        # Save vector store
        if self.vector_store:
            self.vector_store.save(self.vector_db_path)
        
        self.is_ready = False
        self.logger.info("RAG system shutdown complete")