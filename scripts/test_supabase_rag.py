#!/usr/bin/env python3
"""
Test script for Supabase RAG system
Validates configuration, connection, and basic functionality
"""

import os
import sys
import json
import logging
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from config.supabase_config import (
    SupabaseConfig, 
    validate_supabase_config, 
    is_supabase_configured,
    create_sample_supabase_config,
    SETUP_INSTRUCTIONS
)
from ai.rag_factory import RAGFactory, RAGSystemManager


def setup_logging():
    """Setup logging for test script"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def test_configuration():
    """Test Supabase configuration"""
    print("=" * 60)
    print("TESTING SUPABASE CONFIGURATION")
    print("=" * 60)
    
    # Check if configured
    if not is_supabase_configured():
        print("❌ Supabase is not configured")
        print("\nTo configure Supabase:")
        print("1. Set environment variables:")
        print("   export SUPABASE_URL='https://your-project.supabase.co'")
        print("   export SUPABASE_ANON_KEY='your-anon-key'")
        print("\n2. Or create a configuration file:")
        config_file = create_sample_supabase_config()
        print(f"   Created sample config at: {config_file}")
        print("   Edit this file with your Supabase credentials")
        print("\n" + SETUP_INSTRUCTIONS)
        return False
    
    print("✅ Supabase is configured")
    
    # Validate configuration
    is_valid, message = validate_supabase_config()
    if not is_valid:
        print(f"❌ Configuration validation failed: {message}")
        return False
    
    print("✅ Configuration validation passed")
    
    # Display configuration (without sensitive data)
    config = SupabaseConfig()
    print(f"   Supabase URL: {config.get('supabase_url')}")
    print(f"   Embedding Model: {config.get('embedding_model')}")
    print(f"   Chunk Size: {config.get('chunk_size')}")
    print(f"   Top K: {config.get('top_k')}")
    print(f"   Similarity Threshold: {config.get('similarity_threshold')}")
    
    return True


def test_dependencies():
    """Test required dependencies"""
    print("\n" + "=" * 60)
    print("TESTING DEPENDENCIES")
    print("=" * 60)
    
    dependencies = [
        ("supabase", "Supabase Python client"),
        ("psycopg2", "PostgreSQL adapter"),
        ("sentence_transformers", "Sentence Transformers"),
        ("numpy", "NumPy"),
    ]
    
    all_available = True
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"✅ {description} is available")
        except ImportError:
            print(f"❌ {description} is missing")
            all_available = False
    
    if not all_available:
        print("\nTo install missing dependencies:")
        print("pip install supabase psycopg2-binary sentence-transformers numpy")
    
    return all_available


def test_rag_factory():
    """Test RAG factory functionality"""
    print("\n" + "=" * 60)
    print("TESTING RAG FACTORY")
    print("=" * 60)
    
    # Check available systems
    available = RAGFactory.get_available_systems()
    print(f"Available RAG systems:")
    print(f"   Local FAISS: {'✅' if available['local'] else '❌'}")
    print(f"   Supabase: {'✅' if available['supabase'] else '❌'}")
    
    if not available['supabase']:
        print("❌ Supabase RAG system is not available")
        return False
    
    # Test configuration validation
    test_config = {
        "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "similarity_threshold": 0.7
    }
    
    is_valid, message = RAGFactory.validate_config(test_config, "supabase")
    if not is_valid:
        print(f"❌ Configuration validation failed: {message}")
        return False
    
    print("✅ RAG factory configuration validation passed")
    return True


def test_supabase_connection():
    """Test Supabase connection and basic operations"""
    print("\n" + "=" * 60)
    print("TESTING SUPABASE CONNECTION")
    print("=" * 60)
    
    try:
        from ai.supabase_rag_system import SupabaseRAGSystem
        from config.supabase_config import get_supabase_rag_config
        
        # Create RAG system
        config = get_supabase_rag_config()
        rag_system = SupabaseRAGSystem(config)
        
        if not rag_system.is_ready:
            print("❌ Failed to initialize Supabase RAG system")
            return False
        
        print("✅ Supabase RAG system initialized successfully")
        
        # Test statistics
        stats = rag_system.get_stats()
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Total embeddings: {stats.get('total_embeddings', 0)}")
        print(f"   Total categories: {stats.get('total_categories', 0)}")
        
        # Cleanup
        rag_system.cleanup()
        print("✅ Connection test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Connection test failed: {e}")
        return False


def test_document_operations():
    """Test document addition and search operations"""
    print("\n" + "=" * 60)
    print("TESTING DOCUMENT OPERATIONS")
    print("=" * 60)
    
    try:
        # Initialize RAG system
        manager = RAGSystemManager({
            "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
            "chunk_size": 256,  # Smaller for testing
            "chunk_overlap": 25,
            "top_k": 3,
            "similarity_threshold": 0.5
        }, force_supabase=True)
        
        if not manager.initialize():
            print("❌ Failed to initialize RAG system")
            return False
        
        rag_system = manager.get_system()
        print("✅ RAG system initialized for testing")
        
        # Test document addition
        test_content = """
        JARVIS (Just A Rather Very Intelligent System) is an AI assistant created for helping users 
        with various tasks. It can understand natural language, provide information, and assist 
        with complex queries. JARVIS prioritizes user privacy and processes all data locally.
        """
        
        test_metadata = {
            "category": "test_category",
            "source": "test_script",
            "test_id": f"test_{int(time.time())}"
        }
        
        print("Adding test document...")
        success = rag_system.add_document(test_content, test_metadata)
        
        if not success:
            print("❌ Failed to add test document")
            manager.shutdown()
            return False
        
        print("✅ Test document added successfully")
        
        # Wait a moment for processing
        time.sleep(2)
        
        # Test search
        print("Testing search functionality...")
        search_results = rag_system.search("What is JARVIS?", top_k=2)
        
        if not search_results:
            print("❌ Search returned no results")
            manager.shutdown()
            return False
        
        print(f"✅ Search returned {len(search_results)} results")
        
        for i, result in enumerate(search_results):
            print(f"   Result {i+1}:")
            print(f"     Score: {result.similarity_score:.3f}")
            print(f"     Content: {result.document.content[:100]}...")
            print(f"     Category: {result.document.category}")
        
        # Test context generation
        print("Testing context generation...")
        context = rag_system.get_context("JARVIS AI assistant", max_context_length=500)
        
        if not context:
            print("❌ Context generation returned empty result")
        else:
            print(f"✅ Context generated ({len(context)} characters)")
            print(f"   Context preview: {context[:150]}...")
        
        # Get final statistics
        stats = rag_system.get_stats()
        print(f"\nFinal statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   System type: {stats.get('system_type', 'unknown')}")
        
        # Cleanup
        manager.shutdown()
        print("✅ Document operations test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Document operations test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_knowledge_base_loading():
    """Test loading knowledge base from JSON"""
    print("\n" + "=" * 60)
    print("TESTING KNOWLEDGE BASE LOADING")
    print("=" * 60)
    
    knowledge_base_path = "data/knowledge_base.json"
    
    if not os.path.exists(knowledge_base_path):
        print(f"❌ Knowledge base file not found: {knowledge_base_path}")
        return False
    
    try:
        # Initialize RAG system
        manager = RAGSystemManager({
            "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
        }, force_supabase=True)
        
        if not manager.initialize():
            print("❌ Failed to initialize RAG system")
            return False
        
        print("✅ RAG system initialized")
        
        # Load knowledge base
        print("Loading knowledge base...")
        success = manager.reload_knowledge_base(knowledge_base_path)
        
        if not success:
            print("❌ Failed to load knowledge base")
            manager.shutdown()
            return False
        
        print("✅ Knowledge base loaded successfully")
        
        # Test search with knowledge base content
        print("Testing search with knowledge base content...")
        search_results = manager.get_system().search("What can JARVIS do?", top_k=3)
        
        if search_results:
            print(f"✅ Found {len(search_results)} relevant results")
            for i, result in enumerate(search_results[:2]):
                print(f"   Result {i+1}: {result.document.content[:100]}... (score: {result.similarity_score:.3f})")
        else:
            print("❌ No search results found")
        
        # Get final statistics
        stats = manager.get_stats()
        print(f"\nKnowledge base statistics:")
        print(f"   Total documents: {stats.get('total_documents', 0)}")
        print(f"   Total categories: {stats.get('total_categories', 0)}")
        
        manager.shutdown()
        print("✅ Knowledge base loading test completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Knowledge base loading test failed: {e}")
        return False


def main():
    """Main test function"""
    setup_logging()
    
    print("SUPABASE RAG SYSTEM TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Configuration", test_configuration),
        ("Dependencies", test_dependencies),
        ("RAG Factory", test_rag_factory),
        ("Supabase Connection", test_supabase_connection),
        ("Document Operations", test_document_operations),
        ("Knowledge Base Loading", test_knowledge_base_loading),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                failed += 1
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            failed += 1
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Tests passed: {passed}")
    print(f"Tests failed: {failed}")
    print(f"Total tests: {passed + failed}")
    
    if failed == 0:
        print("🎉 ALL TESTS PASSED!")
        return 0
    else:
        print("💥 SOME TESTS FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())