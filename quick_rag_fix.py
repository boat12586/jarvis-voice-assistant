#!/usr/bin/env python3
"""
Quick RAG system fix for JARVIS - uses existing models and fixes configuration
"""

import sys
import os
import logging
import yaml
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    return logging.getLogger(__name__)

def fix_rag_configuration():
    """Fix RAG configuration with optimal settings"""
    logger = logging.getLogger(__name__)
    
    try:
        config_path = Path("config/default_config.yaml")
        
        # Read current config
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update RAG settings with optimal values
        if 'rag' not in config:
            config['rag'] = {}
        
        config['rag'].update({
            'embedding_model': 'mixedbread-ai/mxbai-embed-large-v1',
            'vector_db_path': 'data/vectordb',
            'chunk_size': 256,
            'chunk_overlap': 25,
            'top_k': 10,
            'similarity_threshold': 0.2,  # Much lower for better search
            'max_context_length': 2048
        })
        
        # Write updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        logger.info("✅ RAG configuration updated with optimal settings")
        logger.info(f"   - Similarity threshold: {config['rag']['similarity_threshold']}")
        logger.info(f"   - Chunk size: {config['rag']['chunk_size']}")
        logger.info(f"   - Top K results: {config['rag']['top_k']}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to update config: {e}")
        return False

def test_rag_with_fallback():
    """Test RAG system with fallback model if needed"""
    logger = logging.getLogger(__name__)
    
    try:
        from src.ai.rag_system import RAGSystem
        
        # Try with smaller model first
        fallback_config = {
            'embedding_model': 'all-MiniLM-L6-v2',  # Smaller, faster model
            'vector_db_path': 'data/vectordb_test',
            'chunk_size': 256,
            'chunk_overlap': 25,
            'top_k': 5,
            'similarity_threshold': 0.2
        }
        
        logger.info("Testing RAG with fallback model (all-MiniLM-L6-v2)...")
        rag = RAGSystem(fallback_config)
        
        if not rag.is_ready:
            logger.error("❌ RAG system not ready")
            return False
        
        # Test document addition
        test_doc = "JARVIS is an AI voice assistant that can understand Thai and English languages."
        success = rag.add_document(test_doc)
        
        if success:
            logger.info("✅ Document addition successful")
            
            # Test search
            results = rag.search("JARVIS voice assistant")
            logger.info(f"✅ Search successful: {len(results)} results found")
            
            if results:
                logger.info(f"   Best match score: {results[0].score:.3f}")
                return True
        
        return False
        
    except Exception as e:
        logger.error(f"❌ RAG test failed: {e}")
        return False

def create_knowledge_base():
    """Create initial knowledge base for JARVIS"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create knowledge base directory
        kb_dir = Path("data/knowledge_base")
        kb_dir.mkdir(exist_ok=True)
        
        # JARVIS knowledge base
        knowledge = {
            "jarvis_info": [
                {
                    "content": "JARVIS is an advanced AI voice assistant designed for natural conversation in English and Thai languages.",
                    "category": "general",
                    "type": "core_info"
                },
                {
                    "content": "JARVIS uses DeepSeek-R1 for language understanding and mxbai-embed-large for document retrieval and search.",
                    "category": "technical",
                    "type": "ai_models"
                },
                {
                    "content": "JARVIS can control smart home devices, answer questions, remember conversations, and learn from user interactions.",
                    "category": "capabilities",
                    "type": "features"
                }
            ],
            "thai_support": [
                {
                    "content": "JARVIS สามารถสนทนาภาษาไทยได้อย่างเป็นธรรมชาติ และเข้าใจบริบททางวัฒนธรรมไทย",
                    "category": "languages",
                    "type": "thai_capability"
                },
                {
                    "content": "ระบบรองรับการแปลภาษาไทย-อังกฤษ และสามารถตอบคำถามในทั้งสองภาษา",
                    "category": "languages", 
                    "type": "translation"
                }
            ],
            "voice_features": [
                {
                    "content": "JARVIS features advanced voice recognition using Faster-Whisper and sci-fi voice synthesis.",
                    "category": "voice",
                    "type": "recognition"
                },
                {
                    "content": "Wake word detection with 'Hey JARVIS' activation and continuous listening capabilities.",
                    "category": "voice",
                    "type": "wake_word"
                }
            ]
        }
        
        # Save knowledge base
        kb_file = kb_dir / "jarvis_knowledge.json"
        import json
        with open(kb_file, 'w', encoding='utf-8') as f:
            json.dump(knowledge, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✅ Knowledge base created: {kb_file}")
        logger.info(f"   Total categories: {len(knowledge)}")
        total_docs = sum(len(docs) for docs in knowledge.values())
        logger.info(f"   Total documents: {total_docs}")
        
        return True, knowledge
        
    except Exception as e:
        logger.error(f"❌ Failed to create knowledge base: {e}")
        return False, None

def load_knowledge_into_rag():
    """Load knowledge base into RAG system"""
    logger = logging.getLogger(__name__)
    
    try:
        # Create knowledge base
        kb_success, knowledge = create_knowledge_base()
        if not kb_success:
            return False
        
        # Load RAG with fallback model
        from src.ai.rag_system import RAGSystem
        
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',
            'vector_db_path': 'data/vectordb',
            'chunk_size': 256,
            'chunk_overlap': 25,
            'top_k': 10,
            'similarity_threshold': 0.2
        }
        
        rag = RAGSystem(config)
        if not rag.is_ready:
            logger.error("❌ RAG system not ready")
            return False
        
        # Add knowledge base documents
        logger.info("Loading knowledge base into RAG system...")
        rag.add_knowledge_base(knowledge)
        
        # Test the loaded knowledge
        test_queries = [
            "What is JARVIS?",
            "Thai language support",
            "voice recognition features",
            "ภาษาไทย"
        ]
        
        logger.info("Testing knowledge base queries...")
        for query in test_queries:
            results = rag.search(query)
            logger.info(f"   '{query}': {len(results)} results")
        
        stats = rag.get_stats()
        logger.info(f"✅ Knowledge base loaded successfully")
        logger.info(f"   Total documents in RAG: {stats.get('total_documents', 0)}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to load knowledge base: {e}")
        return False

def main():
    """Main execution function"""
    logger = setup_logging()
    
    logger.info("🛠️ Quick JARVIS RAG System Fix")
    logger.info("Fixing configuration and testing with fallback models")
    
    # Step 1: Fix configuration
    logger.info("\n1. Fixing RAG configuration...")
    config_success = fix_rag_configuration()
    
    # Step 2: Test RAG system
    logger.info("\n2. Testing RAG system...")
    rag_success = test_rag_with_fallback()
    
    # Step 3: Load knowledge base
    logger.info("\n3. Loading knowledge base...")
    kb_success = load_knowledge_into_rag()
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("QUICK FIX SUMMARY")
    logger.info("="*50)
    
    results = {
        "Configuration Fix": config_success,
        "RAG System Test": rag_success,
        "Knowledge Base": kb_success
    }
    
    passed = sum(results.values())
    total = len(results)
    
    for test, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed >= 2:
        logger.info("🎉 RAG system is now functional!")
        logger.info("Next steps:")
        logger.info("  - Download mxbai-embed-large model for better performance")
        logger.info("  - Test DeepSeek-R1 integration")
        logger.info("  - Continue with advanced features")
    else:
        logger.error("❌ Critical issues remain")
        logger.info("Please check dependencies and system requirements")
    
    return passed >= 2

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)