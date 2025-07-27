-- Supabase Schema for JARVIS RAG System
-- This schema defines the database structure for document storage and vector embeddings

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS uuid-ossp;

-- Documents table for storing knowledge base documents
CREATE TABLE IF NOT EXISTS rag_documents (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) UNIQUE NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}',
    category VARCHAR(100),
    chunk_index INTEGER DEFAULT 0,
    chunk_type VARCHAR(50) DEFAULT 'text',
    original_length INTEGER,
    chunk_length INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Vector embeddings table for semantic search
CREATE TABLE IF NOT EXISTS rag_embeddings (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id VARCHAR(255) NOT NULL REFERENCES rag_documents(document_id) ON DELETE CASCADE,
    embedding vector(384), -- Adjust dimension based on embedding model
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Knowledge categories table for organizing documents
CREATE TABLE IF NOT EXISTS rag_categories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Knowledge sessions table for tracking RAG system usage
CREATE TABLE IF NOT EXISTS rag_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) UNIQUE NOT NULL,
    user_id VARCHAR(255),
    query_count INTEGER DEFAULT 0,
    last_query_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Query logs table for tracking search queries and results
CREATE TABLE IF NOT EXISTS rag_query_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id VARCHAR(255) REFERENCES rag_sessions(session_id),
    query_text TEXT NOT NULL,
    query_embedding vector(384),
    results_count INTEGER DEFAULT 0,
    similarity_threshold FLOAT DEFAULT 0.7,
    top_k INTEGER DEFAULT 5,
    response_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Search results table for tracking what documents were retrieved
CREATE TABLE IF NOT EXISTS rag_search_results (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_log_id UUID REFERENCES rag_query_logs(id) ON DELETE CASCADE,
    document_id VARCHAR(255) REFERENCES rag_documents(document_id),
    similarity_score FLOAT NOT NULL,
    relevance_score FLOAT NOT NULL,
    rank_position INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_rag_documents_document_id ON rag_documents(document_id);
CREATE INDEX IF NOT EXISTS idx_rag_documents_category ON rag_documents(category);
CREATE INDEX IF NOT EXISTS idx_rag_documents_created_at ON rag_documents(created_at);
CREATE INDEX IF NOT EXISTS idx_rag_documents_metadata ON rag_documents USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_rag_embeddings_document_id ON rag_embeddings(document_id);
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_created_at ON rag_embeddings(created_at);

CREATE INDEX IF NOT EXISTS idx_rag_sessions_session_id ON rag_sessions(session_id);
CREATE INDEX IF NOT EXISTS idx_rag_sessions_user_id ON rag_sessions(user_id);

CREATE INDEX IF NOT EXISTS idx_rag_query_logs_session_id ON rag_query_logs(session_id);
CREATE INDEX IF NOT EXISTS idx_rag_query_logs_created_at ON rag_query_logs(created_at);

CREATE INDEX IF NOT EXISTS idx_rag_search_results_query_log_id ON rag_search_results(query_log_id);
CREATE INDEX IF NOT EXISTS idx_rag_search_results_document_id ON rag_search_results(document_id);

-- HNSW index for vector similarity search (requires pg_vector)
CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector_hnsw 
ON rag_embeddings USING hnsw (embedding vector_cosine_ops) 
WITH (m = 16, ef_construction = 64);

-- Alternative IVFFlat index for larger datasets
-- CREATE INDEX IF NOT EXISTS idx_rag_embeddings_vector_ivfflat 
-- ON rag_embeddings USING ivfflat (embedding vector_cosine_ops) 
-- WITH (lists = 100);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for automatic timestamp updates
CREATE TRIGGER update_rag_documents_updated_at 
    BEFORE UPDATE ON rag_documents 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rag_categories_updated_at 
    BEFORE UPDATE ON rag_categories 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rag_sessions_updated_at 
    BEFORE UPDATE ON rag_sessions 
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to find similar documents using vector similarity
CREATE OR REPLACE FUNCTION find_similar_documents(
    query_embedding vector(384),
    similarity_threshold float DEFAULT 0.7,
    max_results integer DEFAULT 5
)
RETURNS TABLE (
    document_id varchar(255),
    content text,
    metadata jsonb,
    similarity_score float,
    category varchar(100),
    chunk_index integer
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        d.document_id,
        d.content,
        d.metadata,
        1 - (e.embedding <=> query_embedding) as similarity_score,
        d.category,
        d.chunk_index
    FROM rag_embeddings e
    JOIN rag_documents d ON e.document_id = d.document_id
    WHERE 1 - (e.embedding <=> query_embedding) >= similarity_threshold
    ORDER BY e.embedding <=> query_embedding
    LIMIT max_results;
END;
$$ LANGUAGE plpgsql;

-- Function to get document statistics
CREATE OR REPLACE FUNCTION get_rag_statistics()
RETURNS TABLE (
    total_documents bigint,
    total_embeddings bigint,
    total_categories bigint,
    total_sessions bigint,
    total_queries bigint,
    avg_query_results float,
    last_document_added timestamp with time zone,
    last_query_at timestamp with time zone
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        (SELECT COUNT(*) FROM rag_documents) as total_documents,
        (SELECT COUNT(*) FROM rag_embeddings) as total_embeddings,
        (SELECT COUNT(*) FROM rag_categories) as total_categories,
        (SELECT COUNT(*) FROM rag_sessions) as total_sessions,
        (SELECT COUNT(*) FROM rag_query_logs) as total_queries,
        (SELECT AVG(results_count) FROM rag_query_logs) as avg_query_results,
        (SELECT MAX(created_at) FROM rag_documents) as last_document_added,
        (SELECT MAX(created_at) FROM rag_query_logs) as last_query_at;
END;
$$ LANGUAGE plpgsql;

-- Row Level Security (RLS) policies for multi-tenant support
ALTER TABLE rag_documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_embeddings ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_categories ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_query_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE rag_search_results ENABLE ROW LEVEL SECURITY;

-- Basic RLS policies (can be customized based on auth requirements)
CREATE POLICY "Users can access their own documents" ON rag_documents
    FOR ALL USING (true); -- Adjust based on auth system

CREATE POLICY "Users can access their own embeddings" ON rag_embeddings
    FOR ALL USING (true); -- Adjust based on auth system

CREATE POLICY "Users can access categories" ON rag_categories
    FOR ALL USING (true);

CREATE POLICY "Users can access their own sessions" ON rag_sessions
    FOR ALL USING (true); -- Adjust to filter by user_id

CREATE POLICY "Users can access their own queries" ON rag_query_logs
    FOR ALL USING (true); -- Adjust based on session ownership

CREATE POLICY "Users can access their own search results" ON rag_search_results
    FOR ALL USING (true);

-- Insert default categories
INSERT INTO rag_categories (name, description, metadata) VALUES
    ('jarvis_information', 'Core information about JARVIS assistant', '{"type": "system", "priority": "high"}'),
    ('technical_information', 'Technical details and capabilities', '{"type": "system", "priority": "high"}'),
    ('features', 'Available features and functionality', '{"type": "user_guide", "priority": "medium"}'),
    ('usage_tips', 'Tips for using JARVIS effectively', '{"type": "user_guide", "priority": "medium"}'),
    ('troubleshooting', 'Common issues and solutions', '{"type": "support", "priority": "medium"}'),
    ('thai_language_support', 'Thai language capabilities', '{"type": "localization", "priority": "medium"}')
ON CONFLICT (name) DO NOTHING;

-- Create a view for easy document retrieval with embeddings
CREATE OR REPLACE VIEW document_embeddings_view AS
SELECT 
    d.id,
    d.document_id,
    d.content,
    d.metadata,
    d.category,
    d.chunk_index,
    d.chunk_type,
    d.original_length,
    d.chunk_length,
    d.created_at,
    d.updated_at,
    e.embedding,
    e.id as embedding_id
FROM rag_documents d
LEFT JOIN rag_embeddings e ON d.document_id = e.document_id;

-- Comments for documentation
COMMENT ON TABLE rag_documents IS 'Stores document content and metadata for RAG system';
COMMENT ON TABLE rag_embeddings IS 'Stores vector embeddings for semantic similarity search';
COMMENT ON TABLE rag_categories IS 'Organizes documents into logical categories';
COMMENT ON TABLE rag_sessions IS 'Tracks user sessions and query patterns';
COMMENT ON TABLE rag_query_logs IS 'Logs all search queries for analytics and debugging';
COMMENT ON TABLE rag_search_results IS 'Tracks which documents were retrieved for each query';

COMMENT ON FUNCTION find_similar_documents IS 'Performs vector similarity search to find relevant documents';
COMMENT ON FUNCTION get_rag_statistics IS 'Returns comprehensive statistics about the RAG system';