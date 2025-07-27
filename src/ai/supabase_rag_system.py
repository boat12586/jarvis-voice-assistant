"""
Supabase-based RAG (Retrieval-Augmented Generation) System for Jarvis Voice Assistant
Handles cloud knowledge base and context retrieval using Supabase PostgreSQL with vector extensions
"""

import os
import json
import logging
import uuid
import time
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np
from sentence_transformers import SentenceTransformer
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QMutex
import hashlib
import asyncio

try:
    from supabase import create_client, Client
    import psycopg2
    from psycopg2.extras import RealDictCursor
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logging.warning("Supabase dependencies not installed. Install with: pip install supabase psycopg2-binary")


@dataclass
class SupabaseDocument:
    """Document structure for Supabase RAG system"""
    document_id: str
    content: str
    metadata: Dict[str, Any]
    category: Optional[str] = None
    chunk_index: int = 0
    chunk_type: str = "text"
    original_length: Optional[int] = None
    chunk_length: Optional[int] = None
    embedding: Optional[List[float]] = None
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def __post_init__(self):
        if self.original_length is None:
            self.original_length = len(self.content)
        if self.chunk_length is None:
            self.chunk_length = len(self.content)


@dataclass
class SupabaseSearchResult:
    """Search result structure for Supabase RAG"""
    document: SupabaseDocument
    similarity_score: float
    relevance_score: float
    rank_position: int = 0


class SupabaseDocumentProcessor:
    """Document processing utilities for Supabase storage"""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.logger = logging.getLogger(__name__)
        
    def chunk_text(self, text: str, metadata: Dict[str, Any] = None) -> List[SupabaseDocument]:
        """Split text into chunks for Supabase storage"""
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
                
                chunks.append(SupabaseDocument(
                    document_id=chunk_id,
                    content=current_chunk.strip(),
                    metadata=chunk_metadata,
                    category=metadata.get("category"),
                    chunk_index=chunk_count,
                    chunk_type="text",
                    original_length=len(text),
                    chunk_length=len(current_chunk)
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
            
            chunks.append(SupabaseDocument(
                document_id=chunk_id,
                content=current_chunk.strip(),
                metadata=chunk_metadata,
                category=metadata.get("category"),
                chunk_index=chunk_count,
                chunk_type="text",
                original_length=len(text),
                chunk_length=len(current_chunk)
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


class SupabaseVectorStore:
    """Supabase vector storage and retrieval system"""
    
    def __init__(self, supabase_url: str, supabase_key: str, embedding_dim: int = 384):
        if not SUPABASE_AVAILABLE:
            raise ImportError("Supabase dependencies not available. Install with: pip install supabase psycopg2-binary")
        
        self.supabase_url = supabase_url
        self.supabase_key = supabase_key
        self.embedding_dim = embedding_dim
        self.logger = logging.getLogger(__name__)
        
        # Initialize Supabase client
        self.supabase: Client = create_client(supabase_url, supabase_key)
        
        # Test connection
        self._test_connection()
        
    def _test_connection(self):
        """Test Supabase connection"""
        try:
            # Simple query to test connection
            result = self.supabase.table('rag_categories').select('count').execute()
            self.logger.info("Supabase connection successful")
        except Exception as e:
            self.logger.error(f"Supabase connection failed: {e}")
            raise
    
    def add_documents(self, documents: List[SupabaseDocument]) -> bool:
        """Add documents to Supabase vector store"""
        if not documents:
            return True
        
        try:
            documents_data = []
            embeddings_data = []
            
            for doc in documents:
                if not doc.embedding:
                    self.logger.warning(f"Document {doc.document_id} has no embedding, skipping")
                    continue
                
                # Prepare document data
                doc_data = {
                    'document_id': doc.document_id,
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'category': doc.category,
                    'chunk_index': doc.chunk_index,
                    'chunk_type': doc.chunk_type,
                    'original_length': doc.original_length,
                    'chunk_length': doc.chunk_length
                }
                documents_data.append(doc_data)
                
                # Prepare embedding data
                embedding_data = {
                    'document_id': doc.document_id,
                    'embedding': doc.embedding
                }
                embeddings_data.append(embedding_data)
            
            # Insert documents
            if documents_data:
                doc_result = self.supabase.table('rag_documents').upsert(documents_data).execute()
                self.logger.info(f"Inserted {len(documents_data)} documents")
            
            # Insert embeddings
            if embeddings_data:
                emb_result = self.supabase.table('rag_embeddings').upsert(embeddings_data).execute()
                self.logger.info(f"Inserted {len(embeddings_data)} embeddings")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to add documents to Supabase: {e}")
            return False
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5, 
               similarity_threshold: float = 0.7) -> List[SupabaseSearchResult]:
        """Search for similar documents using Supabase vector similarity"""
        try:
            # Convert numpy array to list
            query_vector = query_embedding.tolist()
            
            # Use the custom function for vector similarity search
            result = self.supabase.rpc(
                'find_similar_documents',
                {
                    'query_embedding': query_vector,
                    'similarity_threshold': similarity_threshold,
                    'max_results': top_k
                }
            ).execute()
            
            search_results = []
            for i, row in enumerate(result.data):
                doc = SupabaseDocument(
                    document_id=row['document_id'],
                    content=row['content'],
                    metadata=row['metadata'] or {},
                    category=row['category'],
                    chunk_index=row['chunk_index']
                )
                
                search_result = SupabaseSearchResult(
                    document=doc,
                    similarity_score=float(row['similarity_score']),
                    relevance_score=float(row['similarity_score']),
                    rank_position=i + 1
                )
                search_results.append(search_result)
            
            self.logger.info(f"Found {len(search_results)} relevant documents")
            return search_results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def get_document(self, document_id: str) -> Optional[SupabaseDocument]:
        """Get document by ID"""
        try:
            result = self.supabase.table('rag_documents').select('*').eq('document_id', document_id).execute()
            
            if result.data:
                row = result.data[0]
                return SupabaseDocument(
                    document_id=row['document_id'],
                    content=row['content'],
                    metadata=row['metadata'] or {},
                    category=row['category'],
                    chunk_index=row['chunk_index'],
                    chunk_type=row['chunk_type'],
                    original_length=row['original_length'],
                    chunk_length=row['chunk_length'],
                    created_at=row['created_at'],
                    updated_at=row['updated_at']
                )
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get document {document_id}: {e}")
            return None
    
    def remove_document(self, document_id: str) -> bool:
        """Remove document and its embeddings"""
        try:
            # Remove embeddings first (due to foreign key constraint)
            self.supabase.table('rag_embeddings').delete().eq('document_id', document_id).execute()
            
            # Remove document
            self.supabase.table('rag_documents').delete().eq('document_id', document_id).execute()
            
            self.logger.info(f"Removed document {document_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to remove document {document_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics"""
        try:
            result = self.supabase.rpc('get_rag_statistics').execute()
            
            if result.data:
                stats = result.data[0]
                return {
                    'total_documents': stats['total_documents'],
                    'total_embeddings': stats['total_embeddings'],
                    'total_categories': stats['total_categories'],
                    'total_sessions': stats['total_sessions'],
                    'total_queries': stats['total_queries'],
                    'avg_query_results': stats['avg_query_results'],
                    'last_document_added': stats['last_document_added'],
                    'last_query_at': stats['last_query_at'],
                    'embedding_dim': self.embedding_dim
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Failed to get statistics: {e}")
            return {}
    
    def clear_all_documents(self, category: Optional[str] = None):
        """Clear all documents or documents in a specific category"""
        try:
            if category:
                # Clear specific category
                self.supabase.table('rag_embeddings').delete().eq('document_id', 
                    self.supabase.table('rag_documents').select('document_id').eq('category', category)
                ).execute()
                self.supabase.table('rag_documents').delete().eq('category', category).execute()
                self.logger.info(f"Cleared all documents in category: {category}")
            else:
                # Clear all documents
                self.supabase.table('rag_embeddings').delete().neq('id', 'none').execute()
                self.supabase.table('rag_documents').delete().neq('id', 'none').execute()
                self.logger.info("Cleared all documents")
                
        except Exception as e:
            self.logger.error(f"Failed to clear documents: {e}")


class SupabaseRAGSystem(QObject):
    """Main Supabase-based RAG system controller"""
    
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
        self.supabase_url = config.get("supabase_url")
        self.supabase_key = config.get("supabase_key")
        self.chunk_size = config.get("chunk_size", 512)
        self.chunk_overlap = config.get("chunk_overlap", 50)
        self.top_k = config.get("top_k", 5)
        self.similarity_threshold = config.get("similarity_threshold", 0.7)
        
        # Validate required config
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("Supabase URL and key are required in config")
        
        # Components
        self.embedding_model: Optional[SentenceTransformer] = None
        self.document_processor = SupabaseDocumentProcessor(self.chunk_size, self.chunk_overlap)
        self.vector_store: Optional[SupabaseVectorStore] = None
        
        # State
        self.is_ready = False
        self.mutex = QMutex()
        self.session_id = str(uuid.uuid4())
        
        # Initialize
        self._initialize()
    
    def _initialize(self):
        """Initialize Supabase RAG system"""
        try:
            if not SUPABASE_AVAILABLE:
                raise ImportError("Supabase dependencies not available")
            
            self.logger.info(f"Initializing Supabase RAG system with model: {self.embedding_model_name}")
            
            # Load embedding model
            self.logger.info("Loading embedding model...")
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
            embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
            self.logger.info(f"Embedding model loaded successfully, dimension: {embedding_dim}")
            
            # Initialize Supabase vector store
            self.logger.info("Initializing Supabase vector store...")
            self.vector_store = SupabaseVectorStore(
                self.supabase_url, 
                self.supabase_key, 
                embedding_dim
            )
            
            # Create session record
            self._create_session()
            
            self.is_ready = True
            stats = self.vector_store.get_stats()
            self.logger.info(f"Supabase RAG system initialized successfully with {stats.get('total_documents', 0)} documents")
            
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to initialize Supabase RAG system: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.error_occurred.emit(f"Supabase RAG initialization failed: {e}")
    
    def _create_session(self):
        """Create session record in Supabase"""
        try:
            session_data = {
                'session_id': self.session_id,
                'user_id': 'jarvis_user',  # Can be customized
                'query_count': 0
            }
            
            self.vector_store.supabase.table('rag_sessions').upsert(session_data).execute()
            self.logger.info(f"Created session: {self.session_id}")
            
        except Exception as e:
            self.logger.warning(f"Failed to create session record: {e}")
    
    def add_document(self, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Add document to Supabase knowledge base"""
        if not self.is_ready:
            self.logger.warning("Supabase RAG system not ready for document addition")
            return False
        
        if not content or not content.strip():
            self.logger.warning("Empty content provided for document addition")
            return False
        
        try:
            self.mutex.lock()
            try:
                self.logger.debug(f"Processing document: {len(content)} characters")
                
                # Process document into chunks
                chunks = self.document_processor.chunk_text(content, metadata)
                
                if not chunks:
                    self.logger.warning("No chunks created from document")
                    return False
                
                self.logger.debug(f"Created {len(chunks)} chunks from document")
                
                # Generate embeddings for chunks
                valid_chunks = []
                for i, chunk in enumerate(chunks):
                    try:
                        self.logger.debug(f"Generating embedding for chunk {i+1}/{len(chunks)}")
                        embedding = self.embedding_model.encode(chunk.content)
                        chunk.embedding = embedding.tolist()
                        valid_chunks.append(chunk)
                        self.logger.debug(f"Generated embedding shape: {embedding.shape}")
                    except Exception as embed_error:
                        self.logger.error(f"Failed to generate embedding for chunk {i+1}: {embed_error}")
                        continue
                
                if not valid_chunks:
                    self.logger.error("No valid chunks with embeddings created")
                    return False
                
                self.logger.info(f"Generated embeddings for {len(valid_chunks)}/{len(chunks)} chunks")
                
                # Add to Supabase vector store
                success = self.vector_store.add_documents(valid_chunks)
                
                if success:
                    self.logger.info(f"Successfully added document with {len(valid_chunks)} chunks")
                    self.document_added.emit(valid_chunks[0].document_id)
                    return True
                else:
                    self.logger.error("Failed to add documents to Supabase")
                    return False
                    
            finally:
                self.mutex.unlock()
                
        except Exception as e:
            import traceback
            self.logger.error(f"Failed to add document: {e}")
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            self.error_occurred.emit(f"Document addition failed: {e}")
            return False
    
    def search(self, query: str, top_k: Optional[int] = None, 
               similarity_threshold: Optional[float] = None) -> List[SupabaseSearchResult]:
        """Search for relevant documents in Supabase"""
        if not self.is_ready:
            self.logger.warning("Supabase RAG system not ready")
            return []
        
        try:
            start_time = time.time()
            
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)
            
            # Search vector store
            results = self.vector_store.search(
                query_embedding,
                top_k=top_k or self.top_k,
                similarity_threshold=similarity_threshold or self.similarity_threshold
            )
            
            # Log query
            response_time_ms = int((time.time() - start_time) * 1000)
            self._log_query(query, query_embedding, results, response_time_ms)
            
            self.logger.info(f"Search query: '{query}' returned {len(results)} results in {response_time_ms}ms")
            self.search_completed.emit(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            self.error_occurred.emit(f"Search failed: {e}")
            return []
    
    def _log_query(self, query: str, query_embedding: np.ndarray, results: List[SupabaseSearchResult], response_time_ms: int):
        """Log search query for analytics"""
        try:
            query_data = {
                'session_id': self.session_id,
                'query_text': query,
                'query_embedding': query_embedding.tolist(),
                'results_count': len(results),
                'similarity_threshold': self.similarity_threshold,
                'top_k': self.top_k,
                'response_time_ms': response_time_ms
            }
            
            query_result = self.vector_store.supabase.table('rag_query_logs').insert(query_data).execute()
            
            if query_result.data and results:
                query_log_id = query_result.data[0]['id']
                
                # Log search results
                search_results_data = []
                for result in results:
                    search_results_data.append({
                        'query_log_id': query_log_id,
                        'document_id': result.document.document_id,
                        'similarity_score': result.similarity_score,
                        'relevance_score': result.relevance_score,
                        'rank_position': result.rank_position
                    })
                
                self.vector_store.supabase.table('rag_search_results').insert(search_results_data).execute()
            
        except Exception as e:
            self.logger.warning(f"Failed to log query: {e}")
    
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
        """Add structured knowledge base data to Supabase"""
        if not self.is_ready:
            self.logger.warning("Supabase RAG system not ready for knowledge base loading")
            return
            
        try:
            added_count = 0
            for category, items in knowledge_data.items():
                if isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict):
                            content = item.get("content", "")
                            if content.strip():
                                metadata = item.copy()
                                metadata["category"] = category
                                if self.add_document(content, metadata):
                                    added_count += 1
                elif isinstance(items, dict):
                    for key, value in items.items():
                        if isinstance(value, str) and value.strip():
                            metadata = {"category": category, "key": key}
                            if self.add_document(value, metadata):
                                added_count += 1
            
            self.logger.info(f"Successfully added {added_count} documents from knowledge base with {len(knowledge_data)} categories")
            
        except Exception as e:
            self.logger.error(f"Failed to add knowledge base: {e}")
            self.error_occurred.emit(f"Knowledge base addition failed: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Supabase RAG system statistics"""
        base_stats = {
            "is_ready": self.is_ready,
            "embedding_model": self.embedding_model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "top_k": self.top_k,
            "similarity_threshold": self.similarity_threshold,
            "session_id": self.session_id,
            "supabase_url": self.supabase_url[:30] + "..." if self.supabase_url else None
        }
        
        if self.vector_store:
            base_stats.update(self.vector_store.get_stats())
        
        return base_stats
    
    def clear_knowledge_base(self, category: Optional[str] = None):
        """Clear all documents from Supabase knowledge base"""
        try:
            if self.vector_store:
                self.vector_store.clear_all_documents(category)
                self.logger.info(f"Knowledge base cleared{' for category: ' + category if category else ''}")
            
        except Exception as e:
            self.logger.error(f"Failed to clear knowledge base: {e}")
            self.error_occurred.emit(f"Knowledge base clear failed: {e}")
    
    def cleanup(self):
        """Cleanup resources"""
        try:
            self.logger.info("Cleaning up Supabase RAG system resources...")
            
            # Clear components
            if self.embedding_model:
                del self.embedding_model
                self.embedding_model = None
            
            if hasattr(self, 'document_processor'):
                del self.document_processor
            
            self.vector_store = None
            self.is_ready = False
            
            self.logger.info("Supabase RAG system cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error during Supabase RAG system cleanup: {e}")
    
    def shutdown(self):
        """Shutdown Supabase RAG system"""
        self.logger.info("Shutting down Supabase RAG system")
        self.cleanup()
        self.logger.info("Supabase RAG system shutdown complete")