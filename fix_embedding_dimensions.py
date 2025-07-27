#!/usr/bin/env python3
"""
Fix embedding dimension mismatch by resetting vector database
and recreating with correct dimensions for mxbai-embed-large (1024)
"""

import os
import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

from ai.rag_system import RAGSystem
from utils.memory_manager import MemoryManager
from system.config_manager import ConfigManager

def main():
    """Fix embedding dimension mismatch"""
    print("🔧 Fixing embedding dimension mismatch...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize components
        print("📋 Loading configuration...")
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        print("💾 Checking memory...")
        import psutil
        memory = psutil.virtual_memory()
        print(f"   RAM: {memory.used/1024**3:.1f}/{memory.total/1024**3:.1f} GB")
        
        # Remove old vector database files
        print("🗑️ Removing old vector database files...")
        vector_files = [
            "data/vectordb.index",
            "data/vectordb.docs", 
            "data/vectordb.pkl"
        ]
        
        for file_path in vector_files:
            if os.path.exists(file_path):
                os.remove(file_path)
                print(f"   ✅ Removed {file_path}")
            else:
                print(f"   ➖ {file_path} not found")
        
        # Initialize RAG system with new dimensions
        print("🧠 Initializing RAG system with mxbai-embed-large...")
        rag_config = config.get("rag", {})  # RAG config is at root level
        print(f"   📋 RAG config: {rag_config}")
        print(f"   🔍 Embedding model: {rag_config.get('embedding_model', 'NOT FOUND')}")
        rag_system = RAGSystem(rag_config)
        
        # Wait for initialization
        print("⏳ Waiting for model loading...")
        import time
        max_wait = 60  # 60 seconds timeout
        wait_time = 0
        
        while not rag_system.is_ready and wait_time < max_wait:
            time.sleep(2)
            wait_time += 2
            print(f"   ⏳ Waiting... {wait_time}s/{max_wait}s")
        
        if not rag_system.is_ready:
            print("❌ RAG system failed to initialize within timeout")
            return False
            
        print("✅ RAG system initialized successfully!")
        
        # Check embedding dimensions
        stats = rag_system.get_stats()
        print(f"📊 Embedding dimensions: {stats['embedding_dim']}")
        print(f"📊 Model: {stats['embedding_model']}")
        
        if stats['embedding_dim'] == 1024:
            print("✅ Correct dimensions (1024) confirmed!")
        else:
            print(f"⚠️ Expected 1024 dimensions, got {stats['embedding_dim']}")
            
        # Test basic functionality
        print("🧪 Testing basic embedding generation...")
        test_result = rag_system.search("test query", top_k=1)
        print(f"   Search test: {'✅ Working' if test_result else '⚠️ No results (expected for empty DB)'}")
        
        print("\n🎉 Embedding dimension fix completed!")
        print("📋 Next steps:")
        print("   1. Run python3 test_knowledge.py to reload knowledge base")
        print("   2. Test search functionality")
        print("   3. Verify system integration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during fix: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)