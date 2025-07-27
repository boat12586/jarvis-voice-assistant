#!/usr/bin/env python3
"""
Test script for the improved JARVIS News System
Demonstrates the real news API functionality
"""

import os
import sys
import logging
import asyncio
from pathlib import Path

# Add the src directory to the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from features.news_system import NewsSystem
from system.config_manager import ConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def test_news_system():
    """Test the news system functionality"""
    print("🚀 Testing JARVIS News System")
    print("=" * 50)
    
    # Load configuration
    config_manager = ConfigManager()
    config = config_manager.load_config()
    
    # Get news configuration
    news_config = config.get("features", {}).get("news", {})
    
    # Override some settings for testing
    news_config.update({
        "update_interval": 60,  # 1 minute for testing
        "max_articles": 5,
        "categories": ["technology", "science", "health"],
        "cache_ttl": 300  # 5 minutes for testing
    })
    
    print(f"📰 News Configuration:")
    print(f"   Categories: {news_config.get('categories', [])}")
    print(f"   Max Articles: {news_config.get('max_articles', 10)}")
    print(f"   Update Interval: {news_config.get('update_interval', 3600)} seconds")
    print(f"   Cache TTL: {news_config.get('cache_ttl', 1800)} seconds")
    print()
    
    # Initialize news system
    print("🔧 Initializing News System...")
    news_system = NewsSystem(news_config)
    
    # Wait a moment for initial news fetch
    print("⏳ Waiting for initial news fetch...")
    import time
    time.sleep(5)
    
    # Test English news
    print("\n📰 English News:")
    print("-" * 30)
    
    # Test different formats
    print("1. Headlines Format:")
    headlines = news_system.get_news_text("en", 3, "headlines")
    print(headlines)
    
    print("\n2. Brief Format:")
    brief = news_system.get_news_text("en", 3, "brief")
    print(brief)
    
    print("\n3. Voice Friendly Format:")
    voice_news = news_system.get_voice_friendly_news("en", 3)
    print(voice_news)
    
    # Test Thai news
    print("\n📰 Thai News:")
    print("-" * 30)
    
    print("1. Headlines Format (Thai):")
    thai_headlines = news_system.get_news_text("th", 3, "headlines")
    print(thai_headlines)
    
    print("\n2. Brief Format (Thai):")
    thai_brief = news_system.get_news_text("th", 3, "brief")
    print(thai_brief)
    
    # Test statistics
    print("\n📊 News System Statistics:")
    print("-" * 30)
    stats = news_system.get_stats()
    for key, value in stats.items():
        if key == "cache" and isinstance(value, dict):
            print(f"   {key}:")
            for cache_key, cache_value in value.items():
                print(f"     {cache_key}: {cache_value}")
        else:
            print(f"   {key}: {value}")
    
    # Test specific categories
    print("\n🔬 Technology News:")
    print("-" * 30)
    news_system.set_categories(["technology"])
    time.sleep(2)  # Wait for update
    tech_news = news_system.get_news_text("en", 5, "detailed")
    print(tech_news)
    
    # Test voice commands simulation
    print("\n🎤 Voice Command Simulation:")
    print("-" * 30)
    
    # Simulate voice commands that the JARVIS system might receive
    voice_commands = [
        ("หาข่าวให้หน่อย", "th"),
        ("Tell me the news", "en"),
        ("What's happening in technology?", "en"),
        ("ข่าวเทคโนโลยีล่าสุด", "th")
    ]
    
    for command, lang in voice_commands:
        print(f"Command: '{command}' ({lang})")
        response = news_system.get_voice_friendly_news(lang, 2)
        print(f"Response: {response[:200]}...")
        print()
    
    # Cleanup
    print("🧹 Cleaning up...")
    news_system.shutdown()
    
    print("✅ News system test completed!")

def test_cache_performance():
    """Test cache performance"""
    print("\n⚡ Testing Cache Performance:")
    print("-" * 30)
    
    config = {
        "cache_ttl": 60,
        "categories": ["technology"],
        "max_articles": 3
    }
    
    news_system = NewsSystem(config)
    
    # First fetch (should hit APIs)
    import time
    start_time = time.time()
    news1 = news_system.get_news_text("en", 3)
    first_fetch_time = time.time() - start_time
    
    # Second fetch (should hit cache)
    start_time = time.time()
    news2 = news_system.get_news_text("en", 3)
    second_fetch_time = time.time() - start_time
    
    print(f"First fetch time: {first_fetch_time:.2f} seconds")
    print(f"Second fetch time: {second_fetch_time:.2f} seconds")
    print(f"Cache speedup: {first_fetch_time / second_fetch_time:.2f}x")
    
    # Check if cache is working
    if abs(len(news1) - len(news2)) < 100:  # Similar content length
        print("✅ Cache appears to be working correctly")
    else:
        print("⚠️ Cache might not be working as expected")
    
    news_system.shutdown()

if __name__ == "__main__":
    try:
        test_news_system()
        test_cache_performance()
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()