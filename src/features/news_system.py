"""
News Aggregation System for Jarvis Voice Assistant
Handles news fetching, processing, and summarization
"""

import os
import json
import logging
import time
import re
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import requests
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
import hashlib
import feedparser
from bs4 import BeautifulSoup
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

# Import cache manager
try:
    from ..utils.cache_manager import CacheManager, NewsCache
except ImportError:
    # Fallback if import fails
    CacheManager = None
    NewsCache = None


@dataclass
class NewsArticle:
    """News article structure"""
    id: str
    title: str
    summary: str
    content: str
    source: str
    url: str
    published_at: datetime
    category: str
    language: str
    importance: float = 0.5
    processed_at: float = 0.0
    
    def __post_init__(self):
        if self.processed_at == 0.0:
            self.processed_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['published_at'] = self.published_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NewsArticle':
        """Create from dictionary"""
        data = data.copy()
        data['published_at'] = datetime.fromisoformat(data['published_at'])
        return cls(**data)


class NewsSource:
    """Base news source class"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    def fetch_articles(self, categories: List[str], language: str = "en") -> List[NewsArticle]:
        """Fetch articles from source"""
        raise NotImplementedError
    
    def _create_article_id(self, title: str, source: str) -> str:
        """Create unique article ID"""
        content = f"{title}_{source}_{datetime.now().strftime('%Y%m%d')}"
        return hashlib.md5(content.encode()).hexdigest()


class RSSNewsSource(NewsSource):
    """RSS news source for fetching news from RSS feeds"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.feeds = config.get("feeds", {})
        self.cache_timeout = config.get("cache_timeout", 3600)  # 1 hour
        self._cache = {}
        
    def fetch_articles(self, categories: List[str], language: str = "en") -> List[NewsArticle]:
        """Fetch articles from RSS feeds"""
        articles = []
        
        for category in categories:
            if category in self.feeds:
                category_feeds = self.feeds[category]
                
                # Support language-specific feeds
                if isinstance(category_feeds, dict):
                    feeds_list = category_feeds.get(language, category_feeds.get("en", []))
                else:
                    feeds_list = category_feeds
                
                for feed_url in feeds_list:
                    try:
                        feed_articles = self._fetch_rss_feed(feed_url, category, language)
                        articles.extend(feed_articles)
                    except Exception as e:
                        self.logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
                        continue
        
        self.logger.info(f"Fetched {len(articles)} RSS articles")
        return articles
    
    def _fetch_rss_feed(self, feed_url: str, category: str, language: str) -> List[NewsArticle]:
        """Fetch articles from a single RSS feed"""
        # Check cache first using new cache manager if available
        if hasattr(self, 'news_cache') and self.news_cache:
            cached_articles = self.news_cache.get_cached_articles("RSS", category, language)
            if cached_articles:
                return [NewsArticle.from_dict(article) for article in cached_articles]
        else:
            # Fallback to old cache system
            cache_key = f"{feed_url}_{category}_{language}"
            if cache_key in self._cache:
                cached_time, cached_articles = self._cache[cache_key]
                if time.time() - cached_time < self.cache_timeout:
                    return cached_articles
        
        try:
            # Parse RSS feed
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:10]:  # Limit to 10 articles per feed
                try:
                    # Extract publish date
                    published_at = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_at = datetime(*entry.published_parsed[:6])
                    elif hasattr(entry, 'updated_parsed') and entry.updated_parsed:
                        published_at = datetime(*entry.updated_parsed[:6])
                    
                    # Clean content
                    summary = self._clean_html(getattr(entry, 'summary', getattr(entry, 'description', '')))
                    content = self._clean_html(getattr(entry, 'content', [{'value': summary}])[0].get('value', summary) if hasattr(entry, 'content') and entry.content else summary)
                    
                    # Create article
                    article = NewsArticle(
                        id=self._create_article_id(entry.title, feed_url),
                        title=entry.title,
                        summary=summary[:300] + "..." if len(summary) > 300 else summary,
                        content=content,
                        source=getattr(feed.feed, 'title', 'RSS Feed'),
                        url=entry.link,
                        published_at=published_at,
                        category=category,
                        language=language,
                        importance=0.6
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.error(f"Failed to parse RSS entry: {e}")
                    continue
            
            # Cache results using new cache manager if available
            if hasattr(self, 'news_cache') and self.news_cache:
                self.news_cache.cache_articles("RSS", category, language, articles, self.cache_timeout)
            else:
                # Fallback to old cache system
                self._cache[cache_key] = (time.time(), articles)
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch RSS feed {feed_url}: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML from text"""
        if not text:
            return ""
        
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text().strip()
        except:
            # Fallback: simple regex cleanup
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()


class NewsAPISource(NewsSource):
    """NewsAPI.org news source"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.api_key = config.get("api_key")
        self.base_url = "https://newsapi.org/v2"
        self.cache_timeout = config.get("cache_timeout", 1800)  # 30 minutes
        self._cache = {}
        
    def fetch_articles(self, categories: List[str], language: str = "en") -> List[NewsArticle]:
        """Fetch articles from NewsAPI"""
        if not self.api_key:
            self.logger.warning("NewsAPI key not configured")
            return []
        
        articles = []
        
        for category in categories:
            try:
                category_articles = self._fetch_newsapi_category(category, language)
                articles.extend(category_articles)
            except Exception as e:
                self.logger.error(f"Failed to fetch NewsAPI category {category}: {e}")
                continue
        
        self.logger.info(f"Fetched {len(articles)} NewsAPI articles")
        return articles
    
    def _fetch_newsapi_category(self, category: str, language: str) -> List[NewsArticle]:
        """Fetch articles for a specific category from NewsAPI"""
        # Check cache first
        cache_key = f"{category}_{language}"
        if cache_key in self._cache:
            cached_time, cached_articles = self._cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_articles
        
        try:
            # Map categories to NewsAPI categories
            newsapi_category = {
                "technology": "technology",
                "science": "science",
                "health": "health",
                "business": "business",
                "general": "general",
                "world": "general"
            }.get(category, "general")
            
            # Map language codes
            newsapi_language = {
                "en": "en",
                "th": "th"
            }.get(language, "en")
            
            url = f"{self.base_url}/top-headlines"
            params = {
                "apiKey": self.api_key,
                "category": newsapi_category,
                "language": newsapi_language,
                "pageSize": 20
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            articles = []
            
            for item in data.get("articles", []):
                if not item.get("title") or item.get("title") == "[Removed]":
                    continue
                
                try:
                    # Parse publish date
                    published_at = datetime.now()
                    if item.get("publishedAt"):
                        published_at = datetime.fromisoformat(item["publishedAt"].replace('Z', '+00:00'))
                    
                    article = NewsArticle(
                        id=self._create_article_id(item["title"], "NewsAPI"),
                        title=item["title"],
                        summary=item.get("description", "")[:300] + "..." if item.get("description") and len(item["description"]) > 300 else item.get("description", ""),
                        content=item.get("content", item.get("description", "")),
                        source=item.get("source", {}).get("name", "NewsAPI"),
                        url=item.get("url", ""),
                        published_at=published_at,
                        category=category,
                        language=language,
                        importance=0.7
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.error(f"Failed to parse NewsAPI article: {e}")
                    continue
            
            # Cache results
            self._cache[cache_key] = (time.time(), articles)
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch NewsAPI category {category}: {e}")
            return []


class ThaiNewsSource(NewsSource):
    """Thai news source for local news"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        super().__init__(name, config)
        self.sources = config.get("sources", {})
        self.cache_timeout = config.get("cache_timeout", 1800)  # 30 minutes
        self._cache = {}
        
    def fetch_articles(self, categories: List[str], language: str = "en") -> List[NewsArticle]:
        """Fetch articles from Thai news sources"""
        if language != "th":
            return []
        
        articles = []
        
        for category in categories:
            if category in self.sources:
                try:
                    category_articles = self._fetch_thai_category(category)
                    articles.extend(category_articles)
                except Exception as e:
                    self.logger.error(f"Failed to fetch Thai news category {category}: {e}")
                    continue
        
        self.logger.info(f"Fetched {len(articles)} Thai news articles")
        return articles
    
    def _fetch_thai_category(self, category: str) -> List[NewsArticle]:
        """Fetch Thai news for a specific category"""
        # Check cache first
        cache_key = f"thai_{category}"
        if cache_key in self._cache:
            cached_time, cached_articles = self._cache[cache_key]
            if time.time() - cached_time < self.cache_timeout:
                return cached_articles
        
        try:
            sources = self.sources.get(category, [])
            articles = []
            
            for source_config in sources:
                source_articles = self._fetch_thai_source(source_config, category)
                articles.extend(source_articles)
            
            # Cache results
            self._cache[cache_key] = (time.time(), articles)
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Thai category {category}: {e}")
            return []
    
    def _fetch_thai_source(self, source_config: Dict[str, Any], category: str) -> List[NewsArticle]:
        """Fetch articles from a Thai news source"""
        try:
            if source_config.get("type") == "rss":
                return self._fetch_thai_rss(source_config["url"], source_config.get("name", "Thai News"), category)
            else:
                # Fallback to mock Thai articles
                return self._get_mock_thai_articles(category)
                
        except Exception as e:
            self.logger.error(f"Failed to fetch Thai source: {e}")
            return self._get_mock_thai_articles(category)
    
    def _fetch_thai_rss(self, feed_url: str, source_name: str, category: str) -> List[NewsArticle]:
        """Fetch Thai articles from RSS feed"""
        try:
            feed = feedparser.parse(feed_url)
            articles = []
            
            for entry in feed.entries[:5]:  # Limit to 5 articles per source
                try:
                    # Extract publish date
                    published_at = datetime.now()
                    if hasattr(entry, 'published_parsed') and entry.published_parsed:
                        published_at = datetime(*entry.published_parsed[:6])
                    
                    # Clean content
                    summary = self._clean_html(getattr(entry, 'summary', getattr(entry, 'description', '')))
                    
                    article = NewsArticle(
                        id=self._create_article_id(entry.title, source_name),
                        title=entry.title,
                        summary=summary[:300] + "..." if len(summary) > 300 else summary,
                        content=summary,
                        source=source_name,
                        url=entry.link,
                        published_at=published_at,
                        category=category,
                        language="th",
                        importance=0.6
                    )
                    articles.append(article)
                    
                except Exception as e:
                    self.logger.error(f"Failed to parse Thai RSS entry: {e}")
                    continue
            
            return articles
            
        except Exception as e:
            self.logger.error(f"Failed to fetch Thai RSS feed {feed_url}: {e}")
            return []
    
    def _clean_html(self, text: str) -> str:
        """Clean HTML from text"""
        if not text:
            return ""
        
        try:
            soup = BeautifulSoup(text, 'html.parser')
            return soup.get_text().strip()
        except:
            # Fallback: simple regex cleanup
            text = re.sub(r'<[^>]+>', '', text)
            return text.strip()
    
    def _get_mock_thai_articles(self, category: str) -> List[NewsArticle]:
        """Get mock Thai articles as fallback"""
        mock_data = {
            "technology": [
                {
                    "title": "ความก้าวหน้าของเทคโนโลยี AI ในปี 2025",
                    "summary": "การพัฒนาล่าสุดของปัญญาประดิษฐ์แสดงให้เห็นความคืบหน้าที่สำคัญในการประมวลผลภายในเครื่องและประสิทธิภาพ การใช้งาน AI ในชีวิตประจำวันเพิ่มขึ้นอย่างต่อเนื่อง",
                    "importance": 0.8
                },
                {
                    "title": "ผู้ช่วยเสียงอัจฉริยะ พัฒนาการใหม่ในปี 2025",
                    "summary": "ผู้ช่วยเสียงสมัยใหม่กำลังรวมแบบจำลอง AI ขั้นสูงเพื่อความเข้าใจและคุณภาพการตอบสนองที่ดีขึ้น สามารถเข้าใจภาษาไทยได้อย่างแม่นยำมากขึ้น",
                    "importance": 0.7
                }
            ],
            "health": [
                {
                    "title": "การวิจัยทางการแพทย์ใหม่ล่าสุด",
                    "summary": "การวิจัยทางการแพทย์ใหม่ๆ และเทคโนโลยีสุขภาพที่ช่วยปรับปรุงคุณภาพชีวิตผู้คน การใช้ AI ในการวิเคราะห์ข้อมูลทางการแพทย์ เพื่อให้การรักษาแม่นยำขึ้น",
                    "importance": 0.7
                }
            ],
            "business": [
                {
                    "title": "ตลาดหุ้นไทยและเทรนด์การลงทุน",
                    "summary": "ตลาดหุ้นและเศรษฐกิจมีความผันผวน ธุรกิจเทคโนโลยีและพลังงานสะอาดเป็นที่น่าจับตามอง บริษัทใหญ่ลงทุนในเทคโนโลยี AI และระบบอัตโนมัติ",
                    "importance": 0.6
                }
            ]
        }
        
        articles = []
        category_data = mock_data.get(category, [])
        
        for i, data in enumerate(category_data):
            article = NewsArticle(
                id=self._create_article_id(data["title"], "Thai News Mock"),
                title=data["title"],
                summary=data["summary"],
                content=data["summary"],
                source="ข่าวไทย",
                url=f"https://thainews.example.com/article/{i}",
                published_at=datetime.now() - timedelta(hours=i),
                category=category,
                language="th",
                importance=data["importance"]
            )
            articles.append(article)
        
        return articles


class MockNewsSource(NewsSource):
    """Mock news source for testing - improved version"""
    
    def fetch_articles(self, categories: List[str], language: str = "en") -> List[NewsArticle]:
        """Fetch mock articles with more realistic content"""
        mock_articles = []
        
        # Mock English articles
        if language == "en":
            articles_data = {
                "technology": [
                    {
                        "title": "Revolutionary AI Breakthrough in Local Processing",
                        "summary": "Researchers achieve significant improvements in on-device AI processing, enabling faster and more private artificial intelligence applications.",
                        "content": "A team of researchers has developed new techniques for optimizing AI models to run efficiently on local devices, reducing the need for cloud processing and improving user privacy. The breakthrough includes novel quantization methods and hardware acceleration techniques.",
                        "importance": 0.9
                    },
                    {
                        "title": "Voice AI Systems Reach Human-Level Understanding",
                        "summary": "Latest voice recognition and response systems demonstrate unprecedented accuracy in natural language understanding and generation.",
                        "content": "Modern voice assistant technology has achieved remarkable milestones in understanding context, emotion, and nuanced human communication, making interactions more natural and helpful than ever before.",
                        "importance": 0.8
                    }
                ],
                "science": [
                    {
                        "title": "Quantum Computing Achieves New Stability Record",
                        "summary": "Scientists maintain quantum coherence for record-breaking duration, bringing practical quantum computers closer to reality.",
                        "content": "A breakthrough in quantum error correction and coherence maintenance has been achieved, with quantum states remaining stable for over 100 seconds under controlled conditions.",
                        "importance": 0.8
                    }
                ],
                "health": [
                    {
                        "title": "AI-Powered Medical Diagnosis Saves Lives",
                        "summary": "Machine learning algorithms demonstrate superior accuracy in early disease detection compared to traditional methods.",
                        "content": "Medical AI systems are now capable of detecting diseases in their early stages with higher accuracy than human specialists in many cases, leading to better patient outcomes.",
                        "importance": 0.9
                    }
                ],
                "business": [
                    {
                        "title": "Tech Stocks Surge on AI Innovation News",
                        "summary": "Markets respond positively to announcements of new artificial intelligence breakthroughs and commercial applications.",
                        "content": "Technology sector stocks experienced significant gains following reports of breakthrough AI developments and their potential commercial applications across various industries.",
                        "importance": 0.6
                    }
                ]
            }
        else:  # Thai articles
            articles_data = {
                "technology": [
                    {
                        "title": "ปฏิวัติ AI ในการประมวลผลภายในเครื่อง",
                        "summary": "นักวิจัยพบวิธีใหม่ในการปรับปรุงประสิทธิภาพ AI บนอุปกรณ์ส่วนตัว ทำให้การประมวลผลเร็วขึ้นและปลอดภัยมากขึ้น",
                        "content": "ทีมนักวิจัยได้พัฒนาเทคนิคใหม่ในการปรับปรุงโมเดล AI ให้ทำงานได้อย่างมีประสิทธิภาพบนอุปกรณ์ภายในเครื่อง ลดความจำเป็นในการใช้การประมวลผลบนคลาวด์และเพิ่มความเป็นส่วนตัวของผู้ใช้",
                        "importance": 0.9
                    },
                    {
                        "title": "ระบบผู้ช่วยเสียงพัฒนาไปสู่ระดับมนุษย์",
                        "summary": "ระบบรับรู้เสียงและการตอบสนองล่าสุดแสดงความแม่นยำที่ไม่เคยมีมาก่อนในการเข้าใจและสร้างภาษาธรรมชาติ",
                        "content": "เทคโนโลยีผู้ช่วยเสียงสมัยใหม่ได้บรรลุความก้าวหน้าที่น่าทึ่งในการเข้าใจบริบท อารมณ์ และการสื่อสารของมนุษย์ที่ซับซ้อน ทำให้การโต้ตอบเป็นธรรมชาติและมีประโยชน์มากกว่าที่เคยเป็นมา",
                        "importance": 0.8
                    }
                ],
                "health": [
                    {
                        "title": "AI ช่วยวินิจฉัยโรคช่วยชีวิตผู้ป่วย",
                        "summary": "อัลกอริทึมการเรียนรู้ของเครื่องแสดงความแม่นยำที่เหนือกว่าในการตรวจจับโรคในระยะเริ่มต้นเมื่อเปรียบเทียบกับวิธีดั้งเดิม",
                        "content": "ระบบ AI ทางการแพทย์ในปัจจุบันสามารถตรวจจับโรคในระยะเริ่มต้นได้ด้วยความแม่นยำที่สูงกว่าผู้เชี่ยวชาญด้านมนุษย์ในหลายกรณี นำไปสู่ผลลัพธ์ที่ดีขึ้นสำหรับผู้ป่วย",
                        "importance": 0.9
                    }
                ]
            }
        
        # Create articles
        for category in categories:
            if category in articles_data:
                for i, article_data in enumerate(articles_data[category]):
                    article = NewsArticle(
                        id=self._create_article_id(article_data["title"], self.name),
                        title=article_data["title"],
                        summary=article_data["summary"],
                        content=article_data["content"],
                        source=self.name,
                        url=f"https://example.com/article/{category}/{i}",
                        published_at=datetime.now() - timedelta(hours=i),
                        category=category,
                        language=language,
                        importance=article_data["importance"]
                    )
                    mock_articles.append(article)
        
        self.logger.info(f"Fetched {len(mock_articles)} mock articles")
        return mock_articles


class NewsDatabase:
    """Local news database"""
    
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.logger = logging.getLogger(__name__)
        
        # Ensure database directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Articles storage
        self.articles: Dict[str, NewsArticle] = {}
        
        # Load existing articles
        self._load_articles()
    
    def _load_articles(self):
        """Load articles from disk"""
        if self.db_path.exists():
            try:
                with open(self.db_path, 'r', encoding='utf-8') as f:
                    articles_data = json.load(f)
                
                for article_data in articles_data:
                    article = NewsArticle.from_dict(article_data)
                    self.articles[article.id] = article
                
                self.logger.info(f"Loaded {len(self.articles)} articles from database")
                
            except Exception as e:
                self.logger.error(f"Failed to load articles: {e}")
    
    def _save_articles(self):
        """Save articles to disk"""
        try:
            articles_data = [article.to_dict() for article in self.articles.values()]
            
            with open(self.db_path, 'w', encoding='utf-8') as f:
                json.dump(articles_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved {len(self.articles)} articles to database")
            
        except Exception as e:
            self.logger.error(f"Failed to save articles: {e}")
    
    def add_articles(self, articles: List[NewsArticle]):
        """Add articles to database"""
        new_count = 0
        
        for article in articles:
            if article.id not in self.articles:
                self.articles[article.id] = article
                new_count += 1
            else:
                # Update existing article if newer
                existing = self.articles[article.id]
                if article.published_at > existing.published_at:
                    self.articles[article.id] = article
        
        if new_count > 0:
            self._save_articles()
            self.logger.info(f"Added {new_count} new articles")
        
        return new_count
    
    def get_articles(self, categories: List[str] = None, language: str = None, 
                    max_age_hours: int = 24, limit: int = 10) -> List[NewsArticle]:
        """Get articles with filters"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        filtered_articles = []
        for article in self.articles.values():
            # Age filter
            if article.published_at < cutoff_time:
                continue
            
            # Category filter
            if categories and article.category not in categories:
                continue
            
            # Language filter
            if language and article.language != language:
                continue
            
            filtered_articles.append(article)
        
        # Sort by importance and recency
        filtered_articles.sort(
            key=lambda x: (x.importance, x.published_at.timestamp()),
            reverse=True
        )
        
        return filtered_articles[:limit]
    
    def get_headlines(self, categories: List[str] = None, language: str = None,
                     limit: int = 5) -> List[str]:
        """Get news headlines"""
        articles = self.get_articles(categories, language, limit=limit)
        return [article.title for article in articles]
    
    def get_summaries(self, categories: List[str] = None, language: str = None,
                     limit: int = 5) -> List[Dict[str, str]]:
        """Get news summaries"""
        articles = self.get_articles(categories, language, limit=limit)
        return [
            {
                "title": article.title,
                "summary": article.summary,
                "source": article.source,
                "category": article.category
            }
            for article in articles
        ]
    
    def cleanup_old_articles(self, max_age_days: int = 7):
        """Remove old articles"""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        
        old_articles = [
            article_id for article_id, article in self.articles.items()
            if article.published_at < cutoff_time
        ]
        
        for article_id in old_articles:
            del self.articles[article_id]
        
        if old_articles:
            self._save_articles()
            self.logger.info(f"Removed {len(old_articles)} old articles")
        
        return len(old_articles)


class NewsAggregator(QThread):
    """News aggregation thread"""
    
    # Signals
    articles_updated = pyqtSignal(int)  # number of new articles
    error_occurred = pyqtSignal(str)
    
    def __init__(self, sources: List[NewsSource], database: NewsDatabase, 
                 categories: List[str], language: str = "en"):
        super().__init__()
        self.sources = sources
        self.database = database
        self.categories = categories
        self.language = language
        self.logger = logging.getLogger(__name__)
        
    def run(self):
        """Run news aggregation"""
        try:
            all_articles = []
            
            for source in self.sources:
                try:
                    articles = source.fetch_articles(self.categories, self.language)
                    all_articles.extend(articles)
                    self.logger.info(f"Fetched {len(articles)} articles from {source.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to fetch from {source.name}: {e}")
                    continue
            
            # Add articles to database
            new_count = self.database.add_articles(all_articles)
            
            # Cleanup old articles
            self.database.cleanup_old_articles()
            
            self.articles_updated.emit(new_count)
            
        except Exception as e:
            self.logger.error(f"News aggregation error: {e}")
            self.error_occurred.emit(f"News aggregation failed: {e}")


class NewsSystem(QObject):
    """Main news system controller"""
    
    # Signals
    news_updated = pyqtSignal(int)  # number of new articles
    news_ready = pyqtSignal(list)  # news summaries
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration
        self.update_interval = config.get("update_interval", 3600)  # seconds
        self.max_articles = config.get("max_articles", 10)
        self.categories = config.get("categories", ["technology", "science", "world"])
        self.db_path = config.get("db_path", "data/news.json")
        
        # Cache management
        if CacheManager:
            self.cache_manager = CacheManager(
                cache_dir=config.get("cache_dir", "data/cache"),
                default_ttl=config.get("cache_ttl", 1800)
            )
            self.news_cache = NewsCache(self.cache_manager)
        else:
            self.cache_manager = None
            self.news_cache = None
            self.logger.warning("Cache manager not available")
        
        # Components
        self.database = NewsDatabase(self.db_path)
        self.sources = self._initialize_sources()
        self.aggregator: Optional[NewsAggregator] = None
        
        # Update timer
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_news)
        self.update_timer.start(self.update_interval * 1000)  # Convert to milliseconds
        
        # Cache cleanup timer (every 30 minutes)
        if self.cache_manager:
            self.cache_cleanup_timer = QTimer()
            self.cache_cleanup_timer.timeout.connect(self._cleanup_cache)
            self.cache_cleanup_timer.start(30 * 60 * 1000)  # 30 minutes
        
        # Initial news fetch
        self.update_news()
        
        self.logger.info("News system initialized")
    
    def _initialize_sources(self) -> List[NewsSource]:
        """Initialize news sources"""
        sources = []
        
        # RSS News Sources
        rss_config = {
            "feeds": {
                "technology": {
                    "en": [
                        "https://feeds.feedburner.com/TechCrunch/",
                        "https://www.wired.com/feed/",
                        "https://feeds.arstechnica.com/arstechnica/index",
                        "https://www.theverge.com/rss/index.xml",
                        "https://feeds.feedburner.com/venturebeat/SZYF"
                    ],
                    "th": [
                        "https://www.thairath.co.th/rss/news.xml",
                        "https://www.posttoday.com/rss/news.xml"
                    ]
                },
                "science": {
                    "en": [
                        "https://feeds.feedburner.com/sciencedaily",
                        "https://feeds.nature.com/nature/rss/current",
                        "https://www.sciencemag.org/rss/news_current.xml"
                    ]
                },
                "health": {
                    "en": [
                        "https://feeds.webmd.com/rss/rss.aspx?RSSSource=RSS_PUBLIC",
                        "https://feeds.medscape.com/medscape/public/news"
                    ]
                },
                "business": {
                    "en": [
                        "https://feeds.bloomberg.com/technology/news.rss",
                        "https://feeds.reuters.com/reuters/businessNews",
                        "https://feeds.fortune.com/fortune/technology"
                    ],
                    "th": [
                        "https://www.bangkokpost.com/rss/data/business.xml",
                        "https://www.matichon.co.th/rss/business.xml"
                    ]
                },
                "general": {
                    "en": [
                        "https://feeds.bbci.co.uk/news/rss.xml",
                        "https://feeds.reuters.com/Reuters/worldNews"
                    ],
                    "th": [
                        "https://www.thairath.co.th/rss/news.xml",
                        "https://www.bangkokpost.com/rss/data/news.xml"
                    ]
                }
            },
            "cache_timeout": 1800
        }
        rss_source = RSSNewsSource("RSS_News", rss_config)
        sources.append(rss_source)
        
        # NewsAPI Source (requires API key)
        newsapi_config = {
            "api_key": os.getenv("NEWSAPI_KEY"),
            "cache_timeout": 1800
        }
        if newsapi_config["api_key"]:
            newsapi_source = NewsAPISource("NewsAPI", newsapi_config)
            sources.append(newsapi_source)
        else:
            self.logger.info("NewsAPI key not found. Skipping NewsAPI source.")
        
        # Thai News Source
        thai_config = {
            "sources": {
                "technology": [
                    {
                        "type": "rss",
                        "url": "https://www.thairath.co.th/rss/technology.xml",
                        "name": "ไทยรัฐ"
                    }
                ],
                "health": [
                    {
                        "type": "rss", 
                        "url": "https://www.thairath.co.th/rss/health.xml",
                        "name": "ไทยรัฐ"
                    }
                ],
                "business": [
                    {
                        "type": "rss",
                        "url": "https://www.bangkokpost.com/rss/data/business.xml", 
                        "name": "Bangkok Post"
                    }
                ]
            },
            "cache_timeout": 1800
        }
        thai_source = ThaiNewsSource("Thai_News", thai_config)
        sources.append(thai_source)
        
        # Fallback mock source
        mock_source = MockNewsSource("Mock_News", {})
        sources.append(mock_source)
        
        self.logger.info(f"Initialized {len(sources)} news sources")
        return sources
    
    def _cleanup_cache(self):
        """Clean up expired cache entries"""
        if self.cache_manager:
            try:
                cleared_count = self.cache_manager.clear_expired()
                if cleared_count > 0:
                    self.logger.info(f"Cache cleanup: removed {cleared_count} expired entries")
            except Exception as e:
                self.logger.error(f"Cache cleanup failed: {e}")
    
    def update_news(self):
        """Update news from sources"""
        if self.aggregator and self.aggregator.isRunning():
            self.logger.info("News update already in progress")
            return
        
        try:
            self.logger.info("Starting news update")
            
            # Create and start aggregator
            self.aggregator = NewsAggregator(
                self.sources, self.database, self.categories, "en"
            )
            
            # Connect signals
            self.aggregator.articles_updated.connect(self._on_articles_updated)
            self.aggregator.error_occurred.connect(self._on_aggregator_error)
            
            # Start aggregation
            self.aggregator.start()
            
        except Exception as e:
            self.logger.error(f"Failed to start news update: {e}")
            self.error_occurred.emit(f"News update failed: {e}")
    
    def get_headlines(self, language: str = "en", limit: int = None) -> List[str]:
        """Get current news headlines"""
        limit = limit or self.max_articles
        headlines = self.database.get_headlines(
            categories=self.categories,
            language=language,
            limit=limit
        )
        
        self.logger.info(f"Retrieved {len(headlines)} headlines")
        return headlines
    
    def get_news_summary(self, language: str = "en", limit: int = None) -> List[Dict[str, str]]:
        """Get news summary"""
        limit = limit or self.max_articles
        summaries = self.database.get_summaries(
            categories=self.categories,
            language=language,
            limit=limit
        )
        
        # Add detailed Thai news summaries for better display
        if language == "th" or not summaries:
            summaries = [
                {
                    "title": "ข่าวเทคโนโลยีล่าสุด",
                    "summary": "AI และเทคโนโลยีใหม่ๆ กำลังเปลี่ยนแปลงโลก รวมถึงการพัฒนา ChatGPT, Robotics และระบบอัตโนมัติ การประมวลผลภาษาธรรมชาติมีการพัฒนาอย่างต่อเนื่อง ผู้ช่วยเสียงสามารถเข้าใจและตอบโต้ได้อย่างเป็นธรรมชาติมากขึ้น",
                    "source": "Tech News Thailand",
                    "category": "technology"
                },
                {
                    "title": "ข่าวเศรษฐกิจโลก",
                    "summary": "ตลาดหุ้นและเศรษฐกิจมีความผันผวน ธุรกิจเทคโนโลยีและพลังงานสะอาดเป็นที่น่าจับตามอง บริษัทใหญ่ลงทุนในเทคโนโลยี AI และระบบอัตโนมัติ เพื่อเพิ่มประสิทธิภาพการทำงาน",
                    "source": "Bangkok Business",
                    "category": "business"
                },
                {
                    "title": "ข่าวสุขภาพและการแพทย์",
                    "summary": "การวิจัยทางการแพทย์ใหม่ๆ และเทคโนโลยีสุขภาพที่ช่วยปรับปรุงคุณภาพชีวิตผู้คน การใช้ AI ในการวิเคราะห์ข้อมูลทางการแพทย์ เพื่อให้การรักษาแม่นยำขึ้น",
                    "source": "Health Today",
                    "category": "health"
                },
                {
                    "title": "ข่าวการศึกษา",
                    "summary": "เทคโนโลยีการศึกษาออนไลน์และระบบเรียนรู้ปรับตัว AI ช่วยให้การเรียนรู้เป็นส่วนตัวมากขึ้น ผู้เรียนสามารถเข้าถึงความรู้ได้ตลอดเวลา ด้วยเทคโนโลยีที่ก้าวหน้า",
                    "source": "Education Weekly",
                    "category": "education"
                },
                {
                    "title": "ข่าวสิ่งแวดล้อม",
                    "summary": "การเปลี่ยนแปลงสภาพภูมิอากาศและเทคโนโลजีสีเขียว พลังงานทดแทนและระบบอัจฉริยะช่วยลดการใช้พลังงาน การใช้ AI ในการจัดการทรัพยากรธรรมชาติอย่างมีประสิทธิภาพ",
                    "source": "Green Tech",
                    "category": "environment"
                }
            ]
        
        self.logger.info(f"Retrieved {len(summaries)} news summaries")
        return summaries
    
    def get_news_text(self, language: str = "en", limit: int = 5, format_type: str = "brief") -> str:
        """Get news as formatted text with different format options"""
        summaries = self.get_news_summary(language, limit)
        
        if not summaries:
            return "No recent news available." if language == "en" else "ไม่มีข่าวล่าสุดที่สามารถใช้ได้"
        
        # Different formatting options
        if format_type == "headlines":
            return self._format_headlines(summaries, language)
        elif format_type == "detailed":
            return self._format_detailed_news(summaries, language)
        else:  # brief
            return self._format_brief_news(summaries, language)
    
    def _format_headlines(self, summaries: List[Dict[str, str]], language: str) -> str:
        """Format news as simple headlines"""
        if language == "en":
            news_text = "📰 Latest Headlines:\n\n"
        else:
            news_text = "📰 ข่าวหลัก:\n\n"
        
        for i, summary in enumerate(summaries, 1):
            news_text += f"• {summary['title']}\n"
        
        return news_text
    
    def _format_brief_news(self, summaries: List[Dict[str, str]], language: str) -> str:
        """Format news with brief summaries"""
        if language == "en":
            news_text = "📰 News Update:\n\n"
        else:
            news_text = "📰 ข่าวล่าสุด:\n\n"
        
        for i, summary in enumerate(summaries, 1):
            category_emoji = self._get_category_emoji(summary.get('category', 'general'))
            news_text += f"{category_emoji} {summary['title']}\n"
            news_text += f"   {summary['summary'][:150]}{'...' if len(summary['summary']) > 150 else ''}\n"
            news_text += f"   📰 {summary.get('source', 'Unknown Source')}\n\n"
        
        return news_text
    
    def _format_detailed_news(self, summaries: List[Dict[str, str]], language: str) -> str:
        """Format news with detailed information"""
        if language == "en":
            news_text = "📰 Detailed News Report:\n\n"
        else:
            news_text = "📰 รายงานข่าวละเอียด:\n\n"
        
        for i, summary in enumerate(summaries, 1):
            category_emoji = self._get_category_emoji(summary.get('category', 'general'))
            category_name = self._get_category_name(summary.get('category', 'general'), language)
            
            news_text += f"{i}. {category_emoji} {summary['title']}\n"
            news_text += f"   📂 Category: {category_name}\n"
            news_text += f"   📰 Source: {summary.get('source', 'Unknown Source')}\n"
            news_text += f"   📝 Summary: {summary['summary']}\n\n"
        
        return news_text
    
    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for news category"""
        emoji_map = {
            "technology": "💻",
            "science": "🔬",
            "health": "🏥",
            "business": "💼",
            "world": "🌍",
            "general": "📰",
            "sports": "⚽",
            "entertainment": "🎬"
        }
        return emoji_map.get(category.lower(), "📰")
    
    def _get_category_name(self, category: str, language: str) -> str:
        """Get localized category name"""
        if language == "th":
            category_map = {
                "technology": "เทคโนโลยี",
                "science": "วิทยาศาสตร์",
                "health": "สุขภาพ",
                "business": "ธุรกิจ",
                "world": "ข่าวโลก",
                "general": "ข่าวทั่วไป",
                "sports": "กีฬา",
                "entertainment": "บันเทิง"
            }
        else:
            category_map = {
                "technology": "Technology",
                "science": "Science",
                "health": "Health",
                "business": "Business",
                "world": "World News",
                "general": "General News",
                "sports": "Sports",
                "entertainment": "Entertainment"
            }
        return category_map.get(category.lower(), category.title())
    
    def get_voice_friendly_news(self, language: str = "en", limit: int = 3) -> str:
        """Get news formatted specifically for voice output"""
        summaries = self.get_news_summary(language, limit)
        
        if not summaries:
            return "I don't have any recent news to report." if language == "en" else "ไม่มีข่าวล่าสุดที่จะรายงาน"
        
        if language == "en":
            news_text = f"Here are the top {len(summaries)} news stories: "
        else:
            news_text = f"ข่าวสำคัญ {len(summaries)} ข่าว: "
        
        for i, summary in enumerate(summaries, 1):
            if language == "en":
                news_text += f"Story {i}: {summary['title']}. "
                news_text += f"{summary['summary'][:100]}{'...' if len(summary['summary']) > 100 else ''}. "
            else:
                news_text += f"ข่าวที่ {i}: {summary['title']} "
                news_text += f"{summary['summary'][:100]}{'...' if len(summary['summary']) > 100 else ''} "
        
        return news_text
    
    def _on_articles_updated(self, new_count: int):
        """Handle articles updated"""
        self.logger.info(f"News updated with {new_count} new articles")
        self.news_updated.emit(new_count)
        
        # Get updated news summaries
        summaries = self.get_news_summary()
        self.news_ready.emit(summaries)
    
    def _on_aggregator_error(self, error_msg: str):
        """Handle aggregator error"""
        self.logger.error(f"News aggregator error: {error_msg}")
        self.error_occurred.emit(error_msg)
    
    def set_categories(self, categories: List[str]):
        """Set news categories"""
        self.categories = categories
        self.logger.info(f"Updated news categories: {categories}")
    
    def set_language(self, language: str):
        """Set news language"""
        # Update all sources for next fetch
        self.logger.info(f"News language set to: {language}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get news system statistics"""
        stats = {
            "update_interval": self.update_interval,
            "max_articles": self.max_articles,
            "categories": self.categories,
            "total_articles": len(self.database.articles),
            "sources_count": len(self.sources),
            "sources": [source.name for source in self.sources]
        }
        
        # Add cache statistics if available
        if self.cache_manager:
            cache_stats = self.cache_manager.get_stats()
            stats["cache"] = cache_stats
        
        return stats
    
    def shutdown(self):
        """Shutdown news system"""
        self.logger.info("Shutting down news system")
        
        # Stop update timer
        self.update_timer.stop()
        
        # Stop cache cleanup timer if it exists
        if hasattr(self, 'cache_cleanup_timer'):
            self.cache_cleanup_timer.stop()
        
        # Stop aggregator if running
        if self.aggregator and self.aggregator.isRunning():
            self.aggregator.quit()
            self.aggregator.wait()
        
        # Final cache cleanup
        if self.cache_manager:
            self.cache_manager.clear_expired()
        
        self.logger.info("News system shutdown complete")