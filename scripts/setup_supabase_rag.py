#!/usr/bin/env python3
"""
Setup script for Supabase RAG system
Automates the configuration and testing of Supabase integration
"""

import os
import sys
import json
import subprocess
from pathlib import Path
from typing import Dict, Any, Tuple

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))


def print_banner():
    """Print setup banner"""
    print("=" * 70)
    print("🚀 JARVIS SUPABASE RAG SYSTEM SETUP")
    print("=" * 70)
    print("This script will help you set up Supabase as the backend for")
    print("JARVIS's Retrieval-Augmented Generation (RAG) system.")
    print()


def check_dependencies() -> bool:
    """Check if required dependencies are installed"""
    print("📦 Checking dependencies...")
    
    dependencies = [
        ("supabase", "Supabase Python client"),
        ("psycopg2", "PostgreSQL adapter"),
        ("sentence_transformers", "Sentence Transformers"),
    ]
    
    missing = []
    
    for module, description in dependencies:
        try:
            __import__(module)
            print(f"  ✅ {description}")
        except ImportError:
            print(f"  ❌ {description}")
            missing.append(module)
    
    if missing:
        print(f"\n⚠️  Missing dependencies: {', '.join(missing)}")
        print("Install them with:")
        print("pip install supabase psycopg2-binary sentence-transformers")
        return False
    
    print("✅ All dependencies are installed!")
    return True


def get_supabase_credentials() -> Tuple[str, str]:
    """Get Supabase credentials from user"""
    print("\n🔐 Supabase Configuration")
    print("You'll need your Supabase project URL and API key.")
    print("Find these in your Supabase project settings > API")
    print()
    
    url = input("Enter your Supabase URL (https://xxx.supabase.co): ").strip()
    if not url.startswith("https://"):
        url = "https://" + url
    
    key = input("Enter your Supabase anon key: ").strip()
    
    return url, key


def validate_credentials(url: str, key: str) -> bool:
    """Validate Supabase credentials"""
    print("\n🔍 Validating credentials...")
    
    try:
        from supabase import create_client
        client = create_client(url, key)
        
        # Test connection with a simple query
        result = client.table('rag_categories').select('count').execute()
        print("✅ Credentials are valid!")
        return True
        
    except Exception as e:
        print(f"❌ Credential validation failed: {e}")
        print("Please check your URL and API key.")
        return False


def create_config_file(url: str, key: str) -> bool:
    """Create Supabase configuration file"""
    print("\n📝 Creating configuration file...")
    
    config = {
        "supabase_url": url,
        "supabase_key": key,
        "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
        "chunk_size": 512,
        "chunk_overlap": 50,
        "top_k": 5,
        "similarity_threshold": 0.7,
        "max_documents": 10000,
        "enable_logging": True,
        "log_queries": True,
        "auto_cleanup": True,
        "cleanup_interval": 3600
    }
    
    try:
        # Create config directory
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)
        
        # Write config file
        config_file = config_dir / "supabase.json"
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✅ Configuration saved to {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to create config file: {e}")
        return False


def setup_environment_variables(url: str, key: str):
    """Show environment variable setup instructions"""
    print("\n🌍 Environment Variables (Optional)")
    print("You can also set these environment variables instead of using the config file:")
    print()
    print(f"export SUPABASE_URL='{url}'")
    print(f"export SUPABASE_ANON_KEY='{key}'")
    print()
    print("Add these to your ~/.bashrc or ~/.zshrc for permanent setup.")


def check_database_schema() -> bool:
    """Check if database schema is set up"""
    print("\n🗄️  Checking database schema...")
    
    try:
        from config.supabase_config import SupabaseConfig
        from ai.supabase_rag_system import SupabaseVectorStore
        
        config = SupabaseConfig()
        rag_config = config.get_rag_config()
        
        if not rag_config.get('supabase_url') or not rag_config.get('supabase_key'):
            print("❌ Configuration not found")
            return False
        
        # Test database connection and schema
        vector_store = SupabaseVectorStore(
            rag_config['supabase_url'],
            rag_config['supabase_key']
        )
        
        stats = vector_store.get_stats()
        if 'total_documents' in stats:
            print("✅ Database schema is properly set up!")
            print(f"   Current documents: {stats.get('total_documents', 0)}")
            print(f"   Current categories: {stats.get('total_categories', 0)}")
            return True
        else:
            print("❌ Database schema not found")
            return False
            
    except Exception as e:
        print(f"❌ Schema check failed: {e}")
        return False


def show_schema_setup_instructions():
    """Show instructions for setting up database schema"""
    print("\n📋 Database Schema Setup Required")
    print("You need to run the SQL schema in your Supabase project:")
    print()
    print("1. Go to your Supabase project dashboard")
    print("2. Navigate to the SQL Editor")
    print("3. Copy and paste the contents of 'database/supabase_schema.sql'")
    print("4. Execute the SQL to create tables and functions")
    print()
    print("The schema file is located at:")
    print("   database/supabase_schema.sql")
    print()
    
    response = input("Press Enter after you've set up the schema, or 'skip' to continue: ").strip().lower()
    return response != 'skip'


def run_migration() -> bool:
    """Run migration from local RAG to Supabase"""
    print("\n🔄 Migration Options")
    print("Do you want to migrate existing RAG data to Supabase?")
    print("This will transfer your knowledge base and any existing vector data.")
    print()
    
    response = input("Migrate existing data? (y/N): ").strip().lower()
    
    if response in ['y', 'yes']:
        print("Running migration...")
        try:
            subprocess.run([
                sys.executable, 
                "scripts/migrate_to_supabase.py"
            ], check=True)
            print("✅ Migration completed successfully!")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Migration failed: {e}")
            return False
    else:
        print("Skipping migration.")
        return True


def run_tests() -> bool:
    """Run test suite"""
    print("\n🧪 Running Tests")
    print("Testing Supabase RAG system functionality...")
    
    try:
        subprocess.run([
            sys.executable, 
            "scripts/test_supabase_rag.py"
        ], check=True)
        print("✅ All tests passed!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Tests failed: {e}")
        return False


def show_completion_message():
    """Show completion message and next steps"""
    print("\n🎉 Setup Complete!")
    print("=" * 50)
    print("Your JARVIS Supabase RAG system is now configured!")
    print()
    print("Next steps:")
    print("1. Start JARVIS normally - it will automatically use Supabase")
    print("2. Test the system with voice commands or the web interface")
    print("3. Monitor your Supabase dashboard for usage and performance")
    print()
    print("Useful commands:")
    print("  - Test RAG system:      python scripts/test_supabase_rag.py")
    print("  - Migrate data:         python scripts/migrate_to_supabase.py")
    print("  - View configuration:   cat config/supabase.json")
    print()
    print("For troubleshooting, see: database/README.md")


def main():
    """Main setup function"""
    print_banner()
    
    # Step 1: Check dependencies
    if not check_dependencies():
        print("\nPlease install missing dependencies and try again.")
        return 1
    
    # Step 2: Get credentials
    try:
        url, key = get_supabase_credentials()
    except KeyboardInterrupt:
        print("\nSetup cancelled.")
        return 1
    
    # Step 3: Validate credentials
    if not validate_credentials(url, key):
        return 1
    
    # Step 4: Create config file
    if not create_config_file(url, key):
        return 1
    
    # Step 5: Show environment setup
    setup_environment_variables(url, key)
    
    # Step 6: Check database schema
    if not check_database_schema():
        if show_schema_setup_instructions():
            if not check_database_schema():
                print("❌ Schema still not found. Please set up the schema and try again.")
                return 1
        else:
            print("⚠️  Skipping schema check - make sure to set it up later!")
    
    # Step 7: Migration
    if not run_migration():
        print("⚠️  Migration failed, but setup can continue.")
    
    # Step 8: Run tests
    if not run_tests():
        print("⚠️  Tests failed, but setup is complete.")
    
    # Step 9: Show completion
    show_completion_message()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Setup failed with error: {e}")
        sys.exit(1)