# JARVIS News System Integration Guide

## Overview
The JARVIS News System provides real-time news aggregation from multiple sources including RSS feeds, NewsAPI, and Thai news sources. It supports both English and Thai languages with intelligent caching to minimize API calls.

## Features
- **Multi-source news aggregation** (RSS, NewsAPI, Thai sources)
- **Bilingual support** (English and Thai)
- **Category filtering** (technology, science, health, business, world)
- **Intelligent caching** to avoid rate limits
- **Multiple output formats** (headlines, brief, detailed, voice-friendly)
- **Real-time updates** with configurable intervals

## Quick Setup

### 1. Install Dependencies
```bash
pip install feedparser beautifulsoup4 lxml python-dotenv
```

### 2. Configure API Keys (Optional)
Copy `.env.example` to `.env` and add your API keys:
```bash
cp .env.example .env
```

Edit `.env`:
```
NEWSAPI_KEY=your_newsapi_key_here
```

### 3. Basic Usage
```python
from src.features.news_system import NewsSystem

# Initialize with configuration
config = {
    "categories": ["technology", "science", "health"],
    "max_articles": 5,
    "update_interval": 3600,  # 1 hour
    "cache_ttl": 1800  # 30 minutes
}

news_system = NewsSystem(config)

# Get news in different formats
headlines = news_system.get_news_text("en", 5, "headlines")
brief_news = news_system.get_news_text("en", 5, "brief")
voice_news = news_system.get_voice_friendly_news("en", 3)

# Thai news
thai_news = news_system.get_news_text("th", 3, "brief")
```

## Voice Command Integration

### Supported Commands
- "หาข่าวให้หน่อย" (Thai)
- "Tell me the news" (English)
- "What's the latest news?"
- "Give me technology news"
- "ข่าวเทคโนโลยีล่าสุด" (Thai)

### Voice Integration Example
```python
def handle_news_command(text: str, language: str) -> str:
    # Detect news request
    news_keywords = {
        "en": ["news", "headlines", "latest", "update"],
        "th": ["ข่าว", "หาข่าว", "ข่าวล่าสุด"]
    }
    
    # Check if it's a news request
    if any(keyword in text.lower() for keyword in news_keywords.get(language, [])):
        # Determine category
        if "technology" in text.lower() or "เทคโนโลยี" in text:
            news_system.set_categories(["technology"])
        elif "health" in text.lower() or "สุขภาพ" in text:
            news_system.set_categories(["health"])
        
        # Get voice-friendly response
        return news_system.get_voice_friendly_news(language, 3)
    
    return None
```

## Configuration Options

### News System Configuration
```yaml
features:
  news:
    categories:
      - technology
      - science
      - health
      - business
      - world
    max_articles: 10
    update_interval: 3600  # seconds
    cache_ttl: 1800       # seconds
    cache_dir: data/cache/news
    db_path: data/news.json
```

### News Sources Configuration
The system automatically configures multiple news sources:

1. **RSS Sources**: TechCrunch, Wired, BBC, Reuters, etc.
2. **NewsAPI**: Requires API key for enhanced functionality
3. **Thai Sources**: Local Thai news feeds
4. **Mock Sources**: Fallback for testing

## Output Formats

### 1. Headlines Format
Simple bullet points of news titles:
```
📰 Latest Headlines:

• Revolutionary AI Breakthrough in Local Processing
• Quantum Computing Achieves New Stability Record
• Voice AI Systems Reach Human-Level Understanding
```

### 2. Brief Format
Titles with summaries and sources:
```
📰 News Update:

💻 Revolutionary AI Breakthrough in Local Processing
   Researchers achieve significant improvements in on-device AI processing...
   📰 TechCrunch

🔬 Quantum Computing Achieves New Stability Record
   Scientists maintain quantum coherence for record-breaking duration...
   📰 Science Daily
```

### 3. Voice-Friendly Format
Optimized for text-to-speech:
```
Here are the top 3 news stories: Story 1: Revolutionary AI Breakthrough in Local Processing. Researchers achieve significant improvements in on-device AI processing enabling faster and more private artificial intelligence applications...
```

## Cache Management

### Automatic Cache Cleanup
- Expired entries are automatically cleaned every 30 minutes
- Cache statistics are available via `get_stats()`
- Manual cleanup with `cache_manager.clear_expired()`

### Cache Performance
- First API call: ~2-5 seconds
- Cached responses: ~0.01-0.1 seconds
- Typical speedup: 20-50x faster

## Error Handling

### Network Issues
- Automatic fallback to cached content
- Graceful degradation to mock sources
- Retry logic for failed requests

### API Rate Limits
- Intelligent caching prevents excessive API calls
- Multiple source redundancy
- Staggered update intervals

## Testing

### Run Test Script
```bash
python test_news_system.py
```

### Test Different Languages
```python
# English news
news_en = news_system.get_voice_friendly_news("en", 3)

# Thai news
news_th = news_system.get_voice_friendly_news("th", 3)
```

## Performance Optimization

### Best Practices
1. **Use appropriate cache TTL**: 30 minutes for general news, 5 minutes for breaking news
2. **Limit article count**: 3-5 articles for voice output, 10+ for display
3. **Category filtering**: Request only needed categories
4. **Update intervals**: 1 hour for general use, 15 minutes for active monitoring

### Memory Usage
- Base memory: ~50-100 MB
- With cache: +20-50 MB
- Per article: ~1-5 KB

## Troubleshooting

### Common Issues
1. **No news returned**: Check internet connection and API keys
2. **Cache not working**: Verify cache directory permissions
3. **Thai text issues**: Ensure UTF-8 encoding
4. **API rate limits**: Increase cache TTL or reduce update frequency

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed logs including:
# - API calls and responses
# - Cache hits and misses
# - RSS feed parsing
# - Error details
```

## API Keys Setup

### NewsAPI (Free Tier)
1. Visit https://newsapi.org/register
2. Get your free API key (500 requests/day)
3. Add to `.env`: `NEWSAPI_KEY=your_key_here`

### Rate Limits
- NewsAPI Free: 500 requests/day
- RSS Feeds: Usually unlimited
- Thai Sources: Varies by source

## Integration with JARVIS Voice Assistant

### Main Application Integration
```python
# In your main JARVIS application
from src.features.news_system import NewsSystem

class JarvisAssistant:
    def __init__(self):
        # Initialize news system with config
        news_config = self.config.get("features", {}).get("news", {})
        self.news_system = NewsSystem(news_config)
    
    def process_voice_command(self, text: str, language: str) -> str:
        # Check if it's a news request
        if self.is_news_request(text, language):
            return self.news_system.get_voice_friendly_news(language, 3)
        
        # Handle other commands...
    
    def is_news_request(self, text: str, language: str) -> bool:
        news_keywords = {
            "en": ["news", "headlines", "latest", "update", "what's happening"],
            "th": ["ข่าว", "หาข่าว", "ข่าวล่าสุด", "อัพเดท"]
        }
        return any(keyword in text.lower() for keyword in news_keywords.get(language, []))
```

## Future Enhancements

### Planned Features
- [ ] Sentiment analysis for news articles
- [ ] Personalized news recommendations
- [ ] Breaking news alerts
- [ ] Social media integration
- [ ] News summarization with AI
- [ ] Audio news playback
- [ ] News search functionality

### Extension Points
- Custom news sources
- Additional languages
- News categorization AI
- Real-time notifications
- Web dashboard interface