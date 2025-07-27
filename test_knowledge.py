#!/usr/bin/env python3
"""
Knowledge Base Test for Jarvis Voice Assistant
Tests knowledge base loading and search functionality
"""

import sys
import os
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_knowledge_file():
    """Test knowledge base file"""
    print("📚 Testing Knowledge Base File...")
    
    try:
        knowledge_file = Path(__file__).parent / "data" / "knowledge_base.json"
        
        if not knowledge_file.exists():
            print(f"   ❌ Knowledge file not found: {knowledge_file}")
            return False
        
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
        
        print(f"   ✅ Knowledge file loaded: {len(knowledge_data)} categories")
        
        # Display categories
        for category in knowledge_data.keys():
            items = knowledge_data[category]
            if isinstance(items, dict):
                print(f"      • {category}: {len(items)} items")
            elif isinstance(items, list):
                print(f"      • {category}: {len(items)} items")
            else:
                print(f"      • {category}: {type(items)}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Knowledge file test failed: {e}")
        return False

def test_rag_system_direct():
    """Test RAG system directly with knowledge base"""
    print("\n🧠 Testing RAG System Direct Loading...")
    
    try:
        from ai.rag_system import RAGSystem
        from system.config_manager import ConfigManager
        
        # Initialize
        config_manager = ConfigManager()
        config = config_manager.get_config()
        rag_config = config.get("rag", {})
        rag = RAGSystem(rag_config)
        
        # Wait for initialization
        import time
        time.sleep(2)
        
        if not rag.is_ready:
            print("   ❌ RAG system not ready")
            return False
        
        print("   ✅ RAG system initialized")
        
        # Load knowledge base manually
        knowledge_file = Path(__file__).parent / "data" / "knowledge_base.json"
        
        with open(knowledge_file, 'r', encoding='utf-8') as f:
            knowledge_data = json.load(f)
        
        print(f"   📥 Loading {len(knowledge_data)} knowledge categories...")
        
        # Add knowledge manually
        added_count = 0
        for category, items in knowledge_data.items():
            if isinstance(items, dict):
                for key, value in items.items():
                    if isinstance(value, str) and value.strip():
                        metadata = {"category": category, "key": key}
                        if rag.add_document(value, metadata):
                            added_count += 1
                            
        print(f"   ✅ Added {added_count} documents to knowledge base")
        
        # Test search
        test_queries = [
            "What is JARVIS?",
            "voice recognition",
            "capabilities"
        ]
        
        search_results = 0
        for query in test_queries:
            results = rag.search(query, top_k=2)
            if results:
                print(f"   🔍 '{query}' → Found {len(results)} results")
                search_results += len(results)
            else:
                print(f"   ⚠️ '{query}' → No results")
        
        if search_results > 0:
            print(f"   ✅ Search system working: {search_results} total results")
            return True
        else:
            print("   ❌ No search results found")
            return False
        
    except Exception as e:
        print(f"   ❌ RAG system test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_ai_engine_knowledge():
    """Test AI engine knowledge loading"""
    print("\n🤖 Testing AI Engine Knowledge Loading...")
    
    try:
        from ai.ai_engine import AIEngine
        from system.config_manager import ConfigManager
        
        config = ConfigManager()
        ai_engine = AIEngine(config)
        
        # Wait for initialization
        import time
        time.sleep(3)
        
        if not ai_engine.rag_system or not ai_engine.rag_system.is_ready:
            print("   ❌ AI engine RAG system not ready")
            return False
        
        print("   ✅ AI engine initialized")
        
        # Get stats
        stats = ai_engine.rag_system.get_stats()
        print(f"   📊 RAG Stats: {stats}")
        
        # Force reload knowledge
        ai_engine._load_initial_knowledge()
        
        time.sleep(2)
        
        # Test search again
        results = ai_engine.rag_system.search("JARVIS capabilities", top_k=3)
        if results:
            print(f"   ✅ Knowledge search working: {len(results)} results")
            for result in results[:2]:
                print(f"      • {result.document.content[:100]}...")
            return True
        else:
            print("   ❌ Knowledge search not working")
            return False
        
    except Exception as e:
        print(f"   ❌ AI engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    print("🤖 Jarvis Voice Assistant - Knowledge Base Test")
    print("=" * 60)
    
    tests = [
        ("Knowledge File", test_knowledge_file),
        ("RAG System Direct", test_rag_system_direct),
        ("AI Engine Knowledge", test_ai_engine_knowledge)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Error during {test_name} test: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("📋 KNOWLEDGE BASE TEST SUMMARY:")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 Knowledge base fully functional!")
    elif passed >= total * 0.7:
        print("\n⚠️ Knowledge base mostly working with some issues")
    else:
        print("\n❌ Knowledge base needs attention")
    
    return passed >= total * 0.7

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)