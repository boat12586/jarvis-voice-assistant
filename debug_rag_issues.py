#!/usr/bin/env python3
"""
Debug RAG Issues for JARVIS Voice Assistant
Deep debugging of embedding and search problems
"""

import sys
import os
import traceback
import time
from pathlib import Path
import json
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Set up detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_embedding_model_directly():
    """Test embedding model directly"""
    print("🔍 Testing Embedding Model Directly...")
    
    try:
        print("   📥 Importing sentence-transformers...")
        from sentence_transformers import SentenceTransformer
        
        print("   📥 Loading mxbai-embed-large model...")
        start_time = time.time()
        
        # Try to load the model
        model = SentenceTransformer('mixedbread-ai/mxbai-embed-large-v1')
        load_time = time.time() - start_time
        
        print(f"   ✅ Model loaded in {load_time:.2f}s")
        
        # Test encoding
        test_texts = [
            "What is JARVIS?",
            "I am J.A.R.V.I.S, an AI assistant.",
            "ภาษาไทยทดสอบ"
        ]
        
        print("   🧮 Testing embeddings...")
        for i, text in enumerate(test_texts):
            try:
                start_time = time.time()
                embedding = model.encode(text)
                encode_time = time.time() - start_time
                
                print(f"      {i+1}. '{text}' → {embedding.shape} in {encode_time:.3f}s")
                print(f"         Sample values: {embedding[:5]}")
                
            except Exception as e:
                print(f"      ❌ Error encoding '{text}': {e}")
                return False
        
        print(f"   ✅ Embedding model working correctly!")
        print(f"   📊 Embedding dimension: {embedding.shape[0]}")
        
        return True, embedding.shape[0]
        
    except Exception as e:
        print(f"   ❌ Embedding model test failed: {e}")
        traceback.print_exc()
        return False, 0

def test_faiss_directly():
    """Test FAISS vector store directly"""
    print("\n📚 Testing FAISS Vector Store...")
    
    try:
        import faiss
        import numpy as np
        
        print("   📥 Testing FAISS operations...")
        
        # Create test vectors
        dimension = 1024  # mxbai-embed-large dimension
        n_vectors = 5
        
        # Generate random test vectors
        test_vectors = np.random.random((n_vectors, dimension)).astype(np.float32)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(test_vectors)
        
        print(f"   🧮 Created {n_vectors} test vectors of dimension {dimension}")
        
        # Create FAISS index
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Add vectors
        index.add(test_vectors)
        
        print(f"   ✅ Added vectors to FAISS index")
        print(f"   📊 Index size: {index.ntotal}")
        
        # Test search
        query_vector = test_vectors[0:1]  # Use first vector as query
        
        scores, indices = index.search(query_vector, 3)
        
        print(f"   🔍 Search results:")
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            print(f"      {i+1}. Index {idx}, Score: {score:.4f}")
        
        print("   ✅ FAISS working correctly!")
        return True
        
    except Exception as e:
        print(f"   ❌ FAISS test failed: {e}")
        traceback.print_exc()
        return False

def debug_rag_system_step_by_step():
    """Debug RAG system step by step"""
    print("\n🔬 Debugging RAG System Step by Step...")
    
    try:
        # Step 1: Import and initialize
        print("   🔸 Step 1: Importing RAG components...")
        from ai.rag_system import RAGSystem, Document
        from system.config_manager import ConfigManager
        
        config = ConfigManager()
        print("   ✅ Imports successful")
        
        # Step 2: Initialize RAG
        print("   🔸 Step 2: Initializing RAG system...")
        rag = RAGSystem(config)
        
        # Wait for initialization
        print("   ⏳ Waiting for RAG initialization...")
        time.sleep(5)
        
        if not rag.is_ready:
            print("   ❌ RAG system not ready")
            return False
        
        print("   ✅ RAG system initialized")
        
        # Step 3: Check embedding model
        print("   🔸 Step 3: Checking embedding model...")
        if hasattr(rag, 'embedding_model') and rag.embedding_model:
            print(f"   ✅ Embedding model loaded: {rag.embedding_model_name}")
            
            # Test embedding
            try:
                test_text = "Test embedding"
                embedding = rag.embedding_model.encode(test_text)
                print(f"   ✅ Embedding test successful: {embedding.shape}")
            except Exception as e:
                print(f"   ❌ Embedding test failed: {e}")
                return False
        else:
            print("   ❌ No embedding model found")
            return False
        
        # Step 4: Test document addition
        print("   🔸 Step 4: Testing document addition...")
        
        test_docs = [
            "JARVIS is an AI assistant created by Tony Stark.",
            "Voice recognition allows users to speak to the system.",
            "The system can understand both Thai and English languages."
        ]
        
        for i, content in enumerate(test_docs):
            try:
                print(f"      Adding document {i+1}: '{content[:30]}...'")
                
                # Add document directly
                result = rag.add_document(content, {"test_id": i, "category": "test"})
                
                if result:
                    print(f"      ✅ Document {i+1} added successfully")
                else:
                    print(f"      ❌ Document {i+1} failed to add")
                    
            except Exception as e:
                print(f"      ❌ Error adding document {i+1}: {e}")
                traceback.print_exc()
        
        # Step 5: Check vector store
        print("   🔸 Step 5: Checking vector store...")
        if hasattr(rag, 'vector_store') and rag.vector_store:
            stats = rag.vector_store.get_stats()
            print(f"   📊 Vector store stats: {stats}")
            
            if stats.get('total_documents', 0) > 0:
                print("   ✅ Documents in vector store")
            else:
                print("   ❌ No documents in vector store")
                return False
        else:
            print("   ❌ No vector store found")
            return False
        
        # Step 6: Test search
        print("   🔸 Step 6: Testing search...")
        
        test_queries = [
            "What is JARVIS?",
            "voice recognition",
            "Thai English"
        ]
        
        for query in test_queries:
            try:
                print(f"      Searching for: '{query}'")
                
                # Generate query embedding
                query_embedding = rag.embedding_model.encode(query)
                print(f"      Query embedding shape: {query_embedding.shape}")
                
                # Search vector store
                results = rag.vector_store.search(query_embedding, top_k=3, similarity_threshold=0.1)
                
                print(f"      Search results: {len(results)}")
                
                for j, result in enumerate(results):
                    print(f"         {j+1}. Score: {result.score:.4f}, Content: '{result.document.content[:50]}...'")
                
                if len(results) > 0:
                    print(f"      ✅ Search successful for '{query}'")
                else:
                    print(f"      ⚠️ No results for '{query}'")
                    
            except Exception as e:
                print(f"      ❌ Search failed for '{query}': {e}")
                traceback.print_exc()
        
        print("   ✅ RAG system debugging completed")
        return True
        
    except Exception as e:
        print(f"   ❌ RAG debugging failed: {e}")
        traceback.print_exc()
        return False

def fix_knowledge_base_loading():
    """Fix knowledge base loading with proper error handling"""
    print("\n🔧 Fixing Knowledge Base Loading...")
    
    try:
        # Load knowledge base file
        kb_file = Path("data/knowledge_base.json")
        
        if not kb_file.exists():
            print("   ❌ Knowledge base file not found")
            return False
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            kb_data = json.load(f)
        
        print(f"   ✅ Knowledge base loaded: {len(kb_data)} categories")
        
        # Initialize RAG system
        from ai.rag_system import RAGSystem
        from system.config_manager import ConfigManager
        
        config = ConfigManager()
        rag = RAGSystem(config)
        
        # Wait for initialization
        time.sleep(5)
        
        if not rag.is_ready:
            print("   ❌ RAG system not ready")
            return False
        
        print("   ✅ RAG system ready for knowledge loading")
        
        # Load knowledge with detailed logging
        added_count = 0
        failed_count = 0
        
        for category, items in kb_data.items():
            print(f"   📂 Processing category: {category}")
            
            if isinstance(items, dict):
                for key, value in items.items():
                    if isinstance(value, str) and value.strip():
                        try:
                            metadata = {
                                "category": category,
                                "key": key,
                                "source": "knowledge_base"
                            }
                            
                            # Add with detailed error handling
                            success = rag.add_document(value.strip(), metadata)
                            
                            if success:
                                added_count += 1
                                print(f"      ✅ Added: {key}")
                            else:
                                failed_count += 1
                                print(f"      ❌ Failed: {key}")
                                
                        except Exception as e:
                            failed_count += 1
                            print(f"      ❌ Error adding {key}: {e}")
            
            elif isinstance(items, list):
                for i, item in enumerate(items):
                    if isinstance(item, dict) and "content" in item:
                        content = item["content"]
                        if content.strip():
                            try:
                                metadata = {
                                    "category": category,
                                    "index": i,
                                    "source": "knowledge_base",
                                    **{k: v for k, v in item.items() if k != "content"}
                                }
                                
                                success = rag.add_document(content.strip(), metadata)
                                
                                if success:
                                    added_count += 1
                                    print(f"      ✅ Added item {i}")
                                else:
                                    failed_count += 1
                                    print(f"      ❌ Failed item {i}")
                                    
                            except Exception as e:
                                failed_count += 1
                                print(f"      ❌ Error adding item {i}: {e}")
        
        print(f"\n   📊 Knowledge Base Loading Summary:")
        print(f"      ✅ Successfully added: {added_count}")
        print(f"      ❌ Failed to add: {failed_count}")
        print(f"      📈 Success rate: {added_count/(added_count+failed_count)*100:.1f}%")
        
        # Test search after loading
        if added_count > 0:
            print("\n   🔍 Testing search after knowledge loading...")
            
            test_queries = [
                "What is JARVIS?",
                "voice recognition capabilities",
                "Thai language support"
            ]
            
            search_success = 0
            for query in test_queries:
                try:
                    results = rag.search(query, top_k=3)
                    if results:
                        search_success += 1
                        print(f"      ✅ '{query}' → {len(results)} results")
                        for result in results[:1]:
                            print(f"         • {result.document.content[:60]}...")
                    else:
                        print(f"      ⚠️ '{query}' → No results")
                        
                except Exception as e:
                    print(f"      ❌ Search error for '{query}': {e}")
            
            print(f"   📊 Search success: {search_success}/{len(test_queries)}")
            
            return search_success > 0
        else:
            print("   ❌ No documents loaded - cannot test search")
            return False
        
    except Exception as e:
        print(f"   ❌ Knowledge base fix failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main debugging function"""
    print("🔍 JARVIS RAG System Deep Debug")
    print("=" * 60)
    
    tests = [
        ("Embedding Model Direct", test_embedding_model_directly),
        ("FAISS Vector Store", test_faiss_directly),
        ("RAG System Step-by-Step", debug_rag_system_step_by_step),
        ("Knowledge Base Loading", fix_knowledge_base_loading)
    ]
    
    results = []
    embedding_dim = 0
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_name == "Embedding Model Direct":
                result, dim = test_func()
                embedding_dim = dim
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
    print("📋 DEBUG SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:25} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if embedding_dim > 0:
        print(f"Embedding dimension: {embedding_dim}")
    
    if passed == total:
        print("\n🎉 All RAG components working!")
        print("🚀 JARVIS knowledge system is ready")
    elif passed >= total * 0.7:
        print("\n⚠️ Most components working with some issues")
        print("System should be functional with limitations")
    else:
        print("\n❌ Critical RAG issues found")
        print("Requires immediate attention")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)