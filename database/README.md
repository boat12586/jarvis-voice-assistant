# Supabase RAG System Database Setup

This directory contains the database schema and setup instructions for the Supabase-based RAG (Retrieval-Augmented Generation) system used by the JARVIS voice assistant.

## Overview

The Supabase RAG system replaces the local FAISS vector database with a cloud-based PostgreSQL database that includes vector similarity search capabilities. This provides better scalability, persistence, and multi-user support.

## Features

- **Vector Similarity Search**: Uses PostgreSQL's pgvector extension for efficient similarity search
- **Document Management**: Structured storage of documents with metadata and categorization
- **Query Logging**: Tracks search queries and results for analytics
- **Session Management**: Supports multiple user sessions and usage tracking
- **Automatic Cleanup**: Built-in data lifecycle management
- **Row-Level Security**: Configurable access control policies

## Prerequisites

1. **Supabase Account**: Sign up at [https://supabase.com](https://supabase.com)
2. **Python Dependencies**: Install required packages
   ```bash
   pip install supabase psycopg2-binary
   ```

## Setup Instructions

### 1. Create Supabase Project

1. Go to [https://supabase.com](https://supabase.com) and create a new project
2. Wait for the project to be fully provisioned
3. Note down your project URL and API keys from the project settings

### 2. Database Schema Setup

1. Go to your Supabase project dashboard
2. Navigate to the SQL Editor
3. Copy and paste the contents of `supabase_schema.sql` into the editor
4. Execute the SQL to create all necessary tables, indexes, and functions

### 3. Environment Configuration

Set up environment variables with your Supabase credentials:

```bash
export SUPABASE_URL="https://your-project-id.supabase.co"
export SUPABASE_ANON_KEY="your-anon-key-here"
export SUPABASE_SERVICE_KEY="your-service-key-here"  # Optional, for admin operations
```

Alternatively, create a configuration file:

```bash
# Create config directory if it doesn't exist
mkdir -p config

# Create configuration file
cat > config/supabase.json << EOF
{
  "supabase_url": "https://your-project-id.supabase.co",
  "supabase_key": "your-anon-key-here",
  "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
  "chunk_size": 512,
  "chunk_overlap": 50,
  "top_k": 5,
  "similarity_threshold": 0.7,
  "enable_logging": true,
  "log_queries": true
}
EOF
```

### 4. Test the Setup

Run the test script to verify everything is working:

```bash
python scripts/test_supabase_rag.py
```

### 5. Migrate Existing Data (Optional)

If you have existing RAG data from the local system, use the migration script:

```bash
python scripts/migrate_to_supabase.py
```

## Database Schema

### Core Tables

#### `rag_documents`
Stores document content and metadata:
- `id`: Primary key (UUID)
- `document_id`: Unique document identifier
- `content`: Document text content
- `metadata`: JSON metadata
- `category`: Document category
- `chunk_index`: Chunk position for multi-part documents
- Timestamps for creation and updates

#### `rag_embeddings`
Stores vector embeddings for similarity search:
- `id`: Primary key (UUID)
- `document_id`: References `rag_documents`
- `embedding`: Vector embedding (384 dimensions by default)
- `created_at`: Creation timestamp

#### `rag_categories`
Organizes documents into categories:
- `id`: Primary key (UUID)
- `name`: Category name
- `description`: Category description
- `metadata`: Additional category information

#### `rag_sessions`
Tracks user sessions:
- `id`: Primary key (UUID)
- `session_id`: Unique session identifier
- `user_id`: User identifier
- `query_count`: Number of queries in session
- Session timestamps

#### `rag_query_logs`
Logs all search queries:
- `id`: Primary key (UUID)
- `session_id`: References `rag_sessions`
- `query_text`: Search query text
- `query_embedding`: Query vector embedding
- `results_count`: Number of results returned
- Performance metrics

#### `rag_search_results`
Tracks search result details:
- `id`: Primary key (UUID)
- `query_log_id`: References `rag_query_logs`
- `document_id`: References `rag_documents`
- `similarity_score`: Similarity score
- `rank_position`: Result ranking

### Indexes

The schema includes optimized indexes for:
- Document ID lookups
- Category filtering
- Vector similarity search (HNSW index)
- Session and query tracking
- Metadata searching (GIN index)

### Functions

#### `find_similar_documents(query_embedding, similarity_threshold, max_results)`
Performs vector similarity search and returns relevant documents with scores.

#### `get_rag_statistics()`
Returns comprehensive statistics about the RAG system including document counts, query metrics, and usage patterns.

## Configuration Options

### RAG System Settings

- `embedding_model`: Sentence transformer model name
- `chunk_size`: Maximum size of document chunks (characters)
- `chunk_overlap`: Overlap between consecutive chunks
- `top_k`: Default number of results to return
- `similarity_threshold`: Minimum similarity score for results
- `enable_logging`: Whether to log queries and results
- `log_queries`: Whether to store query embeddings

### Performance Settings

- `max_documents`: Maximum number of documents to store
- `auto_cleanup`: Enable automatic cleanup of old data
- `cleanup_interval`: Cleanup interval in seconds

## Usage Examples

### Basic Usage

```python
from ai.rag_factory import RAGSystemManager

# Initialize RAG system (auto-selects Supabase if configured)
manager = RAGSystemManager({
    "embedding_model": "mixedbread-ai/mxbai-embed-large-v1",
    "top_k": 5,
    "similarity_threshold": 0.7
})

if manager.initialize():
    rag_system = manager.get_system()
    
    # Add document
    rag_system.add_document(
        "JARVIS is an AI assistant that helps users with various tasks.",
        {"category": "jarvis_info", "source": "user_manual"}
    )
    
    # Search for relevant documents
    results = rag_system.search("What is JARVIS?")
    
    # Get context for query
    context = rag_system.get_context("JARVIS capabilities")
    
    manager.shutdown()
```

### Force Supabase Usage

```python
from ai.rag_factory import RAGSystemManager

# Force use of Supabase RAG system
manager = RAGSystemManager(config, force_supabase=True)
```

### Load Knowledge Base

```python
# Load knowledge base from JSON file
manager.reload_knowledge_base("data/knowledge_base.json")
```

## Monitoring and Analytics

### Query Statistics

The system automatically tracks:
- Query frequency and patterns
- Response times
- Result relevance scores
- User session activity

### Access Statistics

```python
stats = rag_system.get_stats()
print(f"Total documents: {stats['total_documents']}")
print(f"Total queries: {stats['total_queries']}")
print(f"Average results per query: {stats['avg_query_results']}")
```

### Database Monitoring

Use Supabase dashboard to monitor:
- Database usage and performance
- API request patterns
- Storage usage
- Query performance

## Security Considerations

### Row-Level Security (RLS)

The schema includes RLS policies that can be customized based on your authentication system:

```sql
-- Example: Restrict access by user ID
CREATE POLICY "Users can only access their documents" ON rag_documents
    FOR ALL USING (metadata->>'user_id' = auth.uid()::text);
```

### API Key Management

- Use environment variables for API keys
- Never commit credentials to version control
- Use Supabase's anon key for client applications
- Use service key only for administrative operations

### Data Privacy

- Documents are stored in your Supabase instance
- Vector embeddings are computed locally before storage
- Query logs can be disabled if privacy is a concern

## Troubleshooting

### Common Issues

1. **Connection Failed**
   - Verify Supabase URL and API key
   - Check network connectivity
   - Ensure project is not paused

2. **Schema Errors**
   - Verify pgvector extension is enabled
   - Check for SQL syntax errors in schema
   - Ensure proper permissions

3. **Search Not Working**
   - Verify documents have embeddings
   - Check similarity threshold settings
   - Ensure vector index is created

4. **Performance Issues**
   - Monitor query performance in Supabase dashboard
   - Consider adjusting chunk size
   - Check vector index configuration

### Debug Mode

Enable debug logging:

```python
import logging
logging.getLogger('ai.supabase_rag_system').setLevel(logging.DEBUG)
```

### Support

For issues:
1. Check the test script output: `python scripts/test_supabase_rag.py`
2. Review logs in `migration.log` (if using migration script)
3. Check Supabase project logs and metrics
4. Verify all dependencies are installed correctly

## Migration from Local RAG

If migrating from the local FAISS-based system:

1. **Backup existing data**:
   ```bash
   cp -r data/vectordb* data/backup/
   ```

2. **Run migration script**:
   ```bash
   python scripts/migrate_to_supabase.py
   ```

3. **Verify migration**:
   ```bash
   python scripts/migrate_to_supabase.py --verify-only
   ```

4. **Update application configuration** to use Supabase RAG system

## Contributing

When modifying the schema:
1. Update `supabase_schema.sql`
2. Test changes with the test script
3. Update this README if needed
4. Consider backward compatibility