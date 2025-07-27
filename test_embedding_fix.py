#!/usr/bin/env python3
"""
Simple test to verify mxbai-embed-large integration is working
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

def test_basic_embedding():
    """Test basic embedding functionality"""
    print("🧪 Testing mxbai-embed-large Integration")
    print("=" * 50)
    
    try:
        from ai.rag_system import RAGSystem
        from system.config_manager import ConfigManager
        
        # Load config
        print("📋 Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        rag_config = config.get("rag", {})
        
        print(f"   📊 Config: {rag_config.get('embedding_model', 'NOT FOUND')}")
        
        # Initialize RAG
        print("🧠 Initializing RAG system...")
        rag = RAGSystem(rag_config)
        
        # Wait for ready
        import time
        for i in range(10):  # 20 second timeout
            if rag.is_ready:
                break
            time.sleep(2)
            print(f"   ⏳ Waiting... {(i+1)*2}s")
        
        if not rag.is_ready:
            print("❌ RAG system not ready")
            return False
            
        # Check stats
        stats = rag.get_stats()
        print(f"✅ RAG System Ready!")
        print(f"   📊 Model: {stats['embedding_model']}")
        print(f"   📊 Dimensions: {stats['embedding_dim']}")
        print(f"   📊 Documents: {stats['total_documents']}")
        
        # Test basic embedding
        print("🧪 Testing document addition...")
        test_doc = "This is a test document for JARVIS AI assistant."
        
        try:
            doc_id = rag.add_document(test_doc, {"source": "test"})
            print(f"   ✅ Document added: {doc_id}")
            
            # Test search
            print("🔍 Testing search...")
            results = rag.search("JARVIS test", top_k=1)
            
            if results:
                print(f"   ✅ Search working: {len(results)} results")
                print(f"      • Score: {results[0].score:.3f}")
                print(f"      • Content: {results[0].document.content[:50]}...")
            else:
                print("   ⚠️ No search results")
                
        except Exception as e:
            print(f"   ❌ Document test failed: {e}")
            return False
            
        print("\n🎉 mxbai-embed-large integration working!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_basic_embedding()
    sys.exit(0 if success else 1)