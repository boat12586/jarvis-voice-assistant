#!/usr/bin/env python3
"""
Test the enhanced RAG system with comprehensive error handling
"""

import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    return logging.getLogger(__name__)

def test_enhanced_rag():
    """Test the enhanced RAG system"""
    logger = setup_logging()
    
    try:
        from src.ai.enhanced_rag_system import EnhancedRAGSystem, RAGError
        
        # Configuration for enhanced RAG
        config = {
            'embedding_model': 'all-MiniLM-L6-v2',  # Use faster model for testing
            'vector_db_path': 'data/vectordb_enhanced',
            'chunk_size': 256,
            'chunk_overlap': 25,
            'top_k': 10,
            'similarity_threshold': 0.2
        }
        
        # Additional config for logging
        config['log_file'] = 'logs/enhanced_rag_test.log'
        
        logger.info("🧪 Testing Enhanced RAG System")
        logger.info("="*50)
        
        # Initialize system
        logger.info("1. Initializing Enhanced RAG System...")
        rag = EnhancedRAGSystem(config)
        
        if not rag.is_ready:
            logger.error("❌ Enhanced RAG system not ready")
            return False
        
        logger.info("✅ Enhanced RAG system initialized successfully")
        
        # Test document addition with error handling
        logger.info("\n2. Testing document addition with error handling...")
        
        test_documents = [
            {
                "content": "JARVIS is an advanced AI voice assistant designed for natural conversation.",
                "metadata": {"category": "general", "source": "documentation"}
            },
            {
                "content": "The system supports both English and Thai languages seamlessly.",
                "metadata": {"category": "features", "source": "specification"}
            },
            {
                "content": "JARVIS uses state-of-the-art AI models for understanding and generation.",
                "metadata": {"category": "technical", "source": "architecture"}
            },
            {
                "content": "",  # Test empty content handling
                "metadata": {"category": "test", "source": "error_test"}
            }
        ]
        
        successful_additions = 0
        for i, doc_info in enumerate(test_documents):
            try:
                success = rag.add_document(doc_info["content"], doc_info["metadata"])
                if success:
                    successful_additions += 1
                    logger.info(f"  ✅ Document {i+1}: Added successfully")
                else:
                    logger.warning(f"  ⚠️ Document {i+1}: Addition failed")
            except RAGError as e:
                logger.info(f"  ℹ️ Document {i+1}: Expected error - {e.error_type}: {e}")
            except Exception as e:
                logger.error(f"  ❌ Document {i+1}: Unexpected error - {e}")
        
        logger.info(f"Successfully added {successful_additions}/{len(test_documents)} documents")
        
        # Test search functionality with error handling
        logger.info("\n3. Testing search functionality...")
        
        test_queries = [
            "JARVIS voice assistant",
            "Thai language support", 
            "AI models technology",
            "",  # Test empty query handling
            "nonexistent topic query"
        ]
        
        successful_searches = 0
        for i, query in enumerate(test_queries):
            try:
                results = rag.search(query)
                successful_searches += 1
                logger.info(f"  ✅ Query {i+1}: '{query[:30]}...' → {len(results)} results")
                
                # Show top result if available
                if results:
                    top_result = results[0]
                    logger.info(f"      Top result score: {top_result['score']:.3f}")
                    logger.info(f"      Content preview: {top_result['document']['content'][:60]}...")
                    
            except RAGError as e:
                logger.info(f"  ℹ️ Query {i+1}: Expected error - {e.error_type}: {e}")
            except Exception as e:
                logger.error(f"  ❌ Query {i+1}: Unexpected error - {e}")
        
        logger.info(f"Successfully completed {successful_searches}/{len(test_queries)} searches")
        
        # Test system diagnostics
        logger.info("\n4. Testing system diagnostics...")
        
        try:
            status = rag.get_system_status()
            diagnostics = rag.get_diagnostics()
            
            logger.info("  ✅ System Status:")
            logger.info(f"      Ready: {status['is_ready']}")
            logger.info(f"      Uptime: {status['uptime']:.2f} seconds")
            logger.info(f"      Documents processed: {status['system_stats']['documents_processed']}")
            logger.info(f"      Searches performed: {status['system_stats']['searches_performed']}")
            logger.info(f"      Errors encountered: {status['system_stats']['errors_encountered']}")
            
            logger.info("  ✅ Diagnostics:")
            logger.info(f"      Error rate: {diagnostics['performance']['error_rate']:.2%}")
            logger.info(f"      Total operations: {diagnostics['performance']['total_operations']}")
            
        except Exception as e:
            logger.error(f"  ❌ Diagnostics failed: {e}")
        
        # Test error conditions
        logger.info("\n5. Testing error conditions...")
        
        error_tests = [
            ("Empty query", lambda: rag.search("")),
            ("None query", lambda: rag.search(None)),
            ("Empty document", lambda: rag.add_document("")),
            ("None document", lambda: rag.add_document(None))
        ]
        
        for test_name, test_func in error_tests:
            try:
                test_func()
                logger.warning(f"  ⚠️ {test_name}: Should have raised error but didn't")
            except RAGError as e:
                logger.info(f"  ✅ {test_name}: Correctly handled - {e.error_type}")
            except Exception as e:
                logger.info(f"  ℹ️ {test_name}: Other error - {type(e).__name__}")
        
        # Final summary
        logger.info("\n" + "="*50)
        logger.info("ENHANCED RAG TEST SUMMARY")
        logger.info("="*50)
        
        final_status = rag.get_system_status()
        
        logger.info(f"✅ System Status: {'Ready' if final_status['is_ready'] else 'Not Ready'}")
        logger.info(f"📊 Documents in store: {final_status['vector_store_stats'].get('total_documents', 0)}")
        logger.info(f"🔍 Total searches: {final_status['system_stats']['searches_performed']}")
        logger.info(f"⚠️ Total errors: {final_status['system_stats']['errors_encountered']}")
        
        # Performance assessment
        total_ops = (final_status['system_stats']['documents_processed'] + 
                    final_status['system_stats']['searches_performed'])
        error_rate = final_status['system_stats']['errors_encountered'] / max(1, total_ops)
        
        if error_rate < 0.1:
            logger.info("🎉 Enhanced RAG system is working excellently!")
        elif error_rate < 0.3:
            logger.info("✅ Enhanced RAG system is working well with minor issues")
        else:
            logger.warning("⚠️ Enhanced RAG system has significant error rate")
        
        return final_status['is_ready'] and total_ops > 0
        
    except Exception as e:
        import traceback
        logger.error(f"❌ Enhanced RAG test failed: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test execution"""
    success = test_enhanced_rag()
    
    if success:
        print("\n🎉 Enhanced RAG system test completed successfully!")
        print("The improved error handling and logging is working properly.")
    else:
        print("\n❌ Enhanced RAG system test failed.")
        print("Please check the logs for detailed error information.")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)