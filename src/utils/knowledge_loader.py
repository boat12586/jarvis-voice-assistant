"""
Enhanced Knowledge Base Loader
Better document processing and indexing
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, List
import re

class EnhancedKnowledgeLoader:
    """Enhanced knowledge base loader with better processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def preprocess_text(self, text: str) -> str:
        """Preprocess text for better indexing"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Ensure proper sentence endings
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Clean up the text
        text = text.strip()
        
        return text
    
    def create_document_variants(self, content: str, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create multiple variants of each document for better retrieval"""
        variants = []
        
        # Original document
        variants.append({
            "content": self.preprocess_text(content),
            "metadata": metadata.copy()
        })
        
        # Question-based variant for Q&A
        if "key" in metadata:
            question_variant = f"Question: What is {metadata['key']}? Answer: {content}"
            variants.append({
                "content": self.preprocess_text(question_variant),
                "metadata": {**metadata, "type": "qa"}
            })
        
        # Summary variant for long content
        if len(content) > 200:
            sentences = content.split('. ')
            if len(sentences) > 2:
                summary = '. '.join(sentences[:2]) + '.'
                variants.append({
                    "content": self.preprocess_text(summary),
                    "metadata": {**metadata, "type": "summary"}
                })
        
        return variants
    
    def load_knowledge_base(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load and process knowledge base"""
        documents = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                kb_data = json.load(f)
            
            for category, items in kb_data.items():
                if isinstance(items, dict):
                    for key, value in items.items():
                        if isinstance(value, str) and value.strip():
                            metadata = {
                                "category": category,
                                "key": key,
                                "source": "knowledge_base"
                            }
                            
                            variants = self.create_document_variants(value, metadata)
                            documents.extend(variants)
                            
                elif isinstance(items, list):
                    for item in items:
                        if isinstance(item, dict) and "content" in item:
                            content = item["content"]
                            if content.strip():
                                metadata = {
                                    "category": category,
                                    "source": "knowledge_base",
                                    **item
                                }
                                
                                variants = self.create_document_variants(content, metadata)
                                documents.extend(variants)
            
            self.logger.info(f"Loaded {len(documents)} document variants from knowledge base")
            return documents
            
        except Exception as e:
            self.logger.error(f"Failed to load knowledge base: {e}")
            return []
