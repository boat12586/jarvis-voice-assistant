"""
Cache Manager for JARVIS Voice Assistant
Handles caching of news articles and other data to reduce API calls
"""

import os
import json
import time
import logging
import hashlib
from typing import Dict, Any, Optional, List
from pathlib import Path
from datetime import datetime, timedelta


class CacheManager:
    """Simple file-based cache manager with TTL support"""
    
    def __init__(self, cache_dir: str = "data/cache", default_ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.default_ttl = default_ttl
        self.logger = logging.getLogger(__name__)
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Memory cache for frequently accessed items
        self._memory_cache: Dict[str, Dict[str, Any]] = {}
        self._memory_cache_size = 100  # Max items in memory
        
    def _get_cache_key(self, key: str) -> str:
        """Generate a safe cache key"""
        # Create a hash of the key to avoid filesystem issues
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_cache_file(self, cache_key: str) -> Path:
        """Get the cache file path for a key"""
        return self.cache_dir / f"{cache_key}.json"
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL"""
        try:
            cache_key = self._get_cache_key(key)
            ttl = ttl or self.default_ttl
            
            cache_data = {
                "value": value,
                "timestamp": time.time(),
                "ttl": ttl,
                "expires_at": time.time() + ttl
            }
            
            # Store in memory cache
            if len(self._memory_cache) >= self._memory_cache_size:
                # Remove oldest item
                oldest_key = min(self._memory_cache.keys(), 
                               key=lambda k: self._memory_cache[k].get("timestamp", 0))
                del self._memory_cache[oldest_key]
            
            self._memory_cache[cache_key] = cache_data
            
            # Store in file cache
            cache_file = self._get_cache_file(cache_key)
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Cached item with key: {key[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set cache for key {key}: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """Get a value from cache"""
        try:
            cache_key = self._get_cache_key(key)
            
            # Check memory cache first
            if cache_key in self._memory_cache:
                cache_data = self._memory_cache[cache_key]
                if time.time() < cache_data["expires_at"]:
                    self.logger.debug(f"Cache hit (memory) for key: {key[:50]}...")
                    return cache_data["value"]
                else:
                    # Expired, remove from memory
                    del self._memory_cache[cache_key]
            
            # Check file cache
            cache_file = self._get_cache_file(cache_key)
            if not cache_file.exists():
                return None
            
            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)
            
            # Check if expired
            if time.time() >= cache_data["expires_at"]:
                # Remove expired file
                cache_file.unlink(missing_ok=True)
                return None
            
            # Add back to memory cache
            if len(self._memory_cache) < self._memory_cache_size:
                self._memory_cache[cache_key] = cache_data
            
            self.logger.debug(f"Cache hit (file) for key: {key[:50]}...")
            return cache_data["value"]
            
        except Exception as e:
            self.logger.error(f"Failed to get cache for key {key}: {e}")
            return None
    
    def has(self, key: str) -> bool:
        """Check if key exists in cache and is not expired"""
        return self.get(key) is not None
    
    def delete(self, key: str) -> bool:
        """Delete a key from cache"""
        try:
            cache_key = self._get_cache_key(key)
            
            # Remove from memory cache
            if cache_key in self._memory_cache:
                del self._memory_cache[cache_key]
            
            # Remove from file cache
            cache_file = self._get_cache_file(cache_key)
            cache_file.unlink(missing_ok=True)
            
            self.logger.debug(f"Deleted cache for key: {key[:50]}...")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete cache for key {key}: {e}")
            return False
    
    def clear_expired(self) -> int:
        """Clear all expired cache entries"""
        try:
            cleared_count = 0
            current_time = time.time()
            
            # Clear expired memory cache
            expired_memory_keys = [
                k for k, v in self._memory_cache.items()
                if current_time >= v["expires_at"]
            ]
            for key in expired_memory_keys:
                del self._memory_cache[key]
                cleared_count += 1
            
            # Clear expired file cache
            for cache_file in self.cache_dir.glob("*.json"):
                try:
                    with open(cache_file, 'r', encoding='utf-8') as f:
                        cache_data = json.load(f)
                    
                    if current_time >= cache_data["expires_at"]:
                        cache_file.unlink()
                        cleared_count += 1
                        
                except Exception as e:
                    self.logger.warning(f"Failed to check cache file {cache_file}: {e}")
                    # Remove corrupted cache files
                    cache_file.unlink(missing_ok=True)
                    cleared_count += 1
            
            if cleared_count > 0:
                self.logger.info(f"Cleared {cleared_count} expired cache entries")
            
            return cleared_count
            
        except Exception as e:
            self.logger.error(f"Failed to clear expired cache: {e}")
            return 0
    
    def clear_all(self) -> bool:
        """Clear all cache entries"""
        try:
            # Clear memory cache
            self._memory_cache.clear()
            
            # Clear file cache
            for cache_file in self.cache_dir.glob("*.json"):
                cache_file.unlink()
            
            self.logger.info("Cleared all cache entries")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to clear all cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        try:
            file_count = len(list(self.cache_dir.glob("*.json")))
            memory_count = len(self._memory_cache)
            
            # Calculate total size
            total_size = 0
            for cache_file in self.cache_dir.glob("*.json"):
                total_size += cache_file.stat().st_size
            
            return {
                "memory_cache_items": memory_count,
                "file_cache_items": file_count,
                "total_cache_size_bytes": total_size,
                "cache_directory": str(self.cache_dir),
                "default_ttl": self.default_ttl
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {}


class NewsCache:
    """Specialized cache for news articles"""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache = cache_manager
        self.logger = logging.getLogger(__name__)
    
    def cache_articles(self, source: str, category: str, language: str, 
                      articles: List[Dict[str, Any]], ttl: int = 1800) -> bool:
        """Cache news articles for a specific source/category/language combination"""
        cache_key = f"news_articles_{source}_{category}_{language}"
        
        # Convert articles to cacheable format
        cacheable_articles = []
        for article in articles:
            if hasattr(article, 'to_dict'):
                cacheable_articles.append(article.to_dict())
            elif isinstance(article, dict):
                cacheable_articles.append(article)
            else:
                self.logger.warning(f"Cannot cache article of type {type(article)}")
        
        return self.cache.set(cache_key, cacheable_articles, ttl)
    
    def get_cached_articles(self, source: str, category: str, language: str) -> Optional[List[Dict[str, Any]]]:
        """Get cached articles for a specific source/category/language combination"""
        cache_key = f"news_articles_{source}_{category}_{language}"
        return self.cache.get(cache_key)
    
    def cache_feed_data(self, feed_url: str, data: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache raw feed data"""
        cache_key = f"feed_data_{feed_url}"
        return self.cache.set(cache_key, data, ttl)
    
    def get_cached_feed_data(self, feed_url: str) -> Optional[Dict[str, Any]]:
        """Get cached feed data"""
        cache_key = f"feed_data_{feed_url}"
        return self.cache.get(cache_key)
    
    def cache_api_response(self, api_name: str, endpoint: str, params: Dict[str, Any], 
                          response: Dict[str, Any], ttl: int = 1800) -> bool:
        """Cache API response"""
        # Create cache key from API name, endpoint, and sorted parameters
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_key = f"api_{api_name}_{endpoint}_{param_str}"
        return self.cache.set(cache_key, response, ttl)
    
    def get_cached_api_response(self, api_name: str, endpoint: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get cached API response"""
        param_str = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_key = f"api_{api_name}_{endpoint}_{param_str}"
        return self.cache.get(cache_key)