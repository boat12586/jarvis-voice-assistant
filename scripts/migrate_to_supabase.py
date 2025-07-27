#!/usr/bin/env python3
"""
Migration script to transfer existing RAG data from local FAISS to Supabase
Handles migration of documents, embeddings, and knowledge base data
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from ai.rag_system import RAGSystem, Document
from ai.supabase_rag_system import SupabaseRAGSystem, SupabaseDocument
from config.supabase_config import SupabaseConfig, validate_supabase_config


class RAGMigrator:
    """Handles migration from local RAG to Supabase RAG"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.setup_logging()
        
        # Load configurations
        self.supabase_config = SupabaseConfig(config_path)
        
        # Validate Supabase configuration
        is_valid, message = self.supabase_config.validate_config()
        if not is_valid:
            raise ValueError(f"Invalid Supabase configuration: {message}")
        
        self.logger.info("RAG Migration utility initialized")
    
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('migration.log')
            ]
        )
    
    def migrate_knowledge_base(self, knowledge_base_path: str = "data/knowledge_base.json") -> bool:
        """Migrate knowledge base from JSON file to Supabase"""
        try:
            if not os.path.exists(knowledge_base_path):
                self.logger.error(f"Knowledge base file not found: {knowledge_base_path}")
                return False
            
            # Load knowledge base data
            with open(knowledge_base_path, 'r', encoding='utf-8') as f:
                knowledge_data = json.load(f)
            
            self.logger.info(f"Loaded knowledge base with {len(knowledge_data)} categories")
            
            # Initialize Supabase RAG system
            supabase_rag = SupabaseRAGSystem(self.supabase_config.get_rag_config())
            
            if not supabase_rag.is_ready:
                self.logger.error("Failed to initialize Supabase RAG system")
                return False
            
            # Add knowledge base to Supabase
            self.logger.info("Starting knowledge base migration to Supabase...")
            supabase_rag.add_knowledge_base(knowledge_data)
            
            # Get statistics
            stats = supabase_rag.get_stats()
            self.logger.info(f"Migration completed. Total documents: {stats.get('total_documents', 0)}")
            
            supabase_rag.cleanup()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to migrate knowledge base: {e}")
            return False
    
    def migrate_local_vectordb(self, vectordb_path: str = "data/vectordb") -> bool:
        """Migrate existing local vector database to Supabase"""
        try:
            if not os.path.exists(f"{vectordb_path}.index"):
                self.logger.warning(f"Local vector database not found at: {vectordb_path}")
                return True  # Not an error, just no data to migrate
            
            self.logger.info("Starting migration from local vector database...")
            
            # Initialize local RAG system
            local_config = {
                "embedding_model": self.supabase_config.get("embedding_model"),
                "vector_db_path": vectordb_path,
                "chunk_size": self.supabase_config.get("chunk_size"),
                "chunk_overlap": self.supabase_config.get("chunk_overlap"),
                "top_k": self.supabase_config.get("top_k"),
                "similarity_threshold": self.supabase_config.get("similarity_threshold")
            }
            
            local_rag = RAGSystem(local_config)
            
            if not local_rag.is_ready:
                self.logger.error("Failed to initialize local RAG system")
                return False
            
            # Initialize Supabase RAG system
            supabase_rag = SupabaseRAGSystem(self.supabase_config.get_rag_config())
            
            if not supabase_rag.is_ready:
                self.logger.error("Failed to initialize Supabase RAG system")
                local_rag.cleanup()
                return False
            
            # Get local documents
            if not local_rag.vector_store or not local_rag.vector_store.documents:
                self.logger.info("No documents found in local vector database")
                local_rag.cleanup()
                supabase_rag.cleanup()
                return True
            
            migrated_count = 0
            total_docs = len(local_rag.vector_store.documents)
            
            self.logger.info(f"Found {total_docs} documents to migrate")
            
            # Migrate each document
            for doc_id, local_doc in local_rag.vector_store.documents.items():
                try:
                    # Convert local document to Supabase format
                    supabase_doc = SupabaseDocument(
                        document_id=local_doc.id,
                        content=local_doc.content,
                        metadata=local_doc.metadata,
                        category=local_doc.metadata.get("category"),
                        chunk_index=local_doc.metadata.get("chunk_index", 0),
                        chunk_type=local_doc.metadata.get("chunk_type", "text"),
                        original_length=local_doc.metadata.get("original_length"),
                        chunk_length=local_doc.metadata.get("chunk_length"),
                        embedding=local_doc.embedding.tolist() if local_doc.embedding is not None else None
                    )
                    
                    # Add to Supabase
                    if supabase_rag.vector_store.add_documents([supabase_doc]):
                        migrated_count += 1
                        if migrated_count % 10 == 0:
                            self.logger.info(f"Migrated {migrated_count}/{total_docs} documents")
                    
                except Exception as e:
                    self.logger.warning(f"Failed to migrate document {doc_id}: {e}")
                    continue
            
            self.logger.info(f"Successfully migrated {migrated_count}/{total_docs} documents")
            
            # Cleanup
            local_rag.cleanup()
            supabase_rag.cleanup()
            
            return migrated_count > 0
            
        except Exception as e:
            self.logger.error(f"Failed to migrate local vector database: {e}")
            return False
    
    def verify_migration(self) -> bool:
        """Verify that migration was successful"""
        try:
            self.logger.info("Verifying migration...")
            
            # Initialize Supabase RAG system
            supabase_rag = SupabaseRAGSystem(self.supabase_config.get_rag_config())
            
            if not supabase_rag.is_ready:
                self.logger.error("Failed to initialize Supabase RAG system for verification")
                return False
            
            # Get statistics
            stats = supabase_rag.get_stats()
            
            self.logger.info("Migration verification results:")
            self.logger.info(f"  Total documents: {stats.get('total_documents', 0)}")
            self.logger.info(f"  Total embeddings: {stats.get('total_embeddings', 0)}")
            self.logger.info(f"  Total categories: {stats.get('total_categories', 0)}")
            self.logger.info(f"  Last document added: {stats.get('last_document_added', 'N/A')}")
            
            # Test search functionality
            test_query = "What is JARVIS?"
            results = supabase_rag.search(test_query, top_k=3)
            
            self.logger.info(f"Test search for '{test_query}' returned {len(results)} results")
            
            if results:
                for i, result in enumerate(results[:2]):
                    self.logger.info(f"  Result {i+1}: {result.document.content[:100]}... (score: {result.similarity_score:.3f})")
            
            supabase_rag.cleanup()
            
            return stats.get('total_documents', 0) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to verify migration: {e}")
            return False
    
    def backup_local_data(self, backup_dir: str = "data/backup") -> bool:
        """Create backup of local data before migration"""
        try:
            import shutil
            from datetime import datetime
            
            os.makedirs(backup_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Backup vector database files
            vectordb_files = ["data/vectordb.index", "data/vectordb.docs"]
            for file_path in vectordb_files:
                if os.path.exists(file_path):
                    backup_path = os.path.join(backup_dir, f"{os.path.basename(file_path)}.{timestamp}")
                    shutil.copy2(file_path, backup_path)
                    self.logger.info(f"Backed up {file_path} to {backup_path}")
            
            # Backup knowledge base
            knowledge_base_path = "data/knowledge_base.json"
            if os.path.exists(knowledge_base_path):
                backup_path = os.path.join(backup_dir, f"knowledge_base.json.{timestamp}")
                shutil.copy2(knowledge_base_path, backup_path)
                self.logger.info(f"Backed up {knowledge_base_path} to {backup_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to backup local data: {e}")
            return False
    
    def run_full_migration(self, backup: bool = True) -> bool:
        """Run complete migration process"""
        try:
            self.logger.info("Starting full RAG migration to Supabase")
            
            # Create backup if requested
            if backup:
                if not self.backup_local_data():
                    self.logger.warning("Backup failed, but continuing with migration")
            
            # Migrate knowledge base
            if not self.migrate_knowledge_base():
                self.logger.error("Knowledge base migration failed")
                return False
            
            # Migrate local vector database (if exists)
            if not self.migrate_local_vectordb():
                self.logger.error("Local vector database migration failed")
                return False
            
            # Verify migration
            if not self.verify_migration():
                self.logger.error("Migration verification failed")
                return False
            
            self.logger.info("Full migration completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"Full migration failed: {e}")
            return False


def main():
    """Main migration script"""
    parser = argparse.ArgumentParser(description="Migrate RAG data from local to Supabase")
    parser.add_argument("--config", help="Path to Supabase configuration file")
    parser.add_argument("--knowledge-base", default="data/knowledge_base.json", 
                       help="Path to knowledge base JSON file")
    parser.add_argument("--vectordb", default="data/vectordb", 
                       help="Path to local vector database")
    parser.add_argument("--no-backup", action="store_true", 
                       help="Skip creating backup of local data")
    parser.add_argument("--verify-only", action="store_true", 
                       help="Only verify existing migration")
    parser.add_argument("--knowledge-only", action="store_true", 
                       help="Only migrate knowledge base")
    parser.add_argument("--vectordb-only", action="store_true", 
                       help="Only migrate vector database")
    
    args = parser.parse_args()
    
    try:
        # Create migrator
        migrator = RAGMigrator(args.config)
        
        # Run requested operation
        if args.verify_only:
            success = migrator.verify_migration()
        elif args.knowledge_only:
            success = migrator.migrate_knowledge_base(args.knowledge_base)
        elif args.vectordb_only:
            success = migrator.migrate_local_vectordb(args.vectordb)
        else:
            success = migrator.run_full_migration(backup=not args.no_backup)
        
        if success:
            print("Migration completed successfully!")
            sys.exit(0)
        else:
            print("Migration failed. Check migration.log for details.")
            sys.exit(1)
            
    except Exception as e:
        print(f"Migration error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()