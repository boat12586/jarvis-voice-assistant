# JARVIS Emotional AI System Documentation

## Overview

The JARVIS Emotional AI System is a comprehensive emotional intelligence framework that enhances the voice assistant with advanced emotion detection, personality adaptation, and user preference learning capabilities. This system provides context-aware responses based on emotional analysis and maintains personalized user experiences across both Thai and English languages.

## Architecture

### Core Components

1. **Emotion Detection System** (`src/ai/emotion_detection.py`)
   - Text-based emotion recognition using transformers
   - Voice emotion analysis (optional, with librosa)
   - Multi-dimensional emotion mapping (valence, arousal, intensity)
   - Real-time emotional context tracking

2. **Personality System** (`src/ai/personality_system.py`)
   - Multiple personality profiles (Professional, Friendly, Casual)
   - Dynamic personality adaptation based on context
   - Thai cultural adaptation patterns
   - Learning and feedback mechanisms

3. **Emotional AI Engine** (`src/ai/emotional_ai_engine.py`)
   - Central coordinator for all emotional intelligence
   - Context-aware response generation
   - User preference integration
   - Multi-modal emotion processing

4. **User Preference System** (`src/features/user_preference_system.py`)
   - Individual user profiling and learning
   - Communication style preferences
   - Emotional pattern recognition
   - Long-term preference memory

5. **Sentiment Analysis System** (`src/ai/sentiment_analysis.py`)
   - Advanced sentiment analysis for Thai and English
   - Conversation-level sentiment tracking
   - Emotional pattern detection
   - Mood trend analysis

6. **Web Integration** (`src/web_emotional_integration.py`)
   - Seamless web interface integration
   - Real-time emotional feedback
   - Session-based emotional context
   - RESTful API endpoints

## Features

### Emotion Detection

- **Multi-language Support**: English and Thai emotion recognition
- **Transformer Models**: Uses state-of-the-art NLP models for accuracy
- **Voice Analysis**: Optional audio emotion detection using spectral features
- **Real-time Processing**: Low-latency emotion detection for interactive use

#### Supported Emotions
- Joy, Sadness, Anger, Fear, Surprise, Disgust
- Anxiety, Frustration, Excitement, Satisfaction
- Cultural-specific emotions for Thai context

### Personality Profiles

#### Professional Personality
- Formal tone and structured responses
- High directness and efficiency
- Minimal humor, maximum information density
- Stress-aware adaptations for urgent situations

#### Friendly Personality
- Warm and conversational tone
- High empathy and supportiveness
- Moderate humor and enthusiasm
- Emotional validation and comfort responses

#### Casual Personality
- Relaxed and informal communication
- Creative and spontaneous responses
- High humor and low formality
- Adaptable technical detail level

### Adaptive Response Generation

The system adapts responses based on:
- **User's emotional state** (stressed, happy, confused)
- **Conversation context** (formal, casual, urgent)
- **Personal preferences** (learned over time)
- **Cultural context** (Thai politeness patterns)

### User Learning

The system learns user preferences through:
- **Implicit signals**: Communication patterns, emotional responses
- **Explicit feedback**: Direct user preferences and ratings
- **Behavioral patterns**: Interaction timing, session length
- **Cultural adaptation**: Language-specific preferences

## Configuration

### Basic Configuration

```python
config = {
    "emotional_ai": {
        "auto_personality_adaptation": True,
        "emotion_memory_length": 20,
        "cache_max_size": 50
    },
    "emotion_detection": {
        "voice_analysis": False,  # Enable for voice emotion detection
        "max_history_length": 50,
        "text_emotion_model": "j-hartmann/emotion-english-distilroberta-base"
    },
    "personality_system": {
        "default_personality": "friendly",
        "enable_learning": True,
        "adaptation_rate": 0.1
    },
    "user_preferences": {
        "learning_rate": 0.1,
        "confidence_threshold": 0.7,
        "preferences_dir": "data/user_preferences"
    },
    "sentiment_analysis": {
        "max_history_length": 100,
        "enable_pattern_detection": True,
        "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest"
    }
}
```

### Advanced Configuration

#### Thai Language Support
```python
thai_config = {
    "thai_adaptations": {
        "politeness_particles": ["ครับ", "ค่ะ", "จ้า"],
        "formal_pronouns": ["ผม", "ดิฉัน", "คุณ"],
        "cultural_context": "thai_formal"
    }
}
```

#### Voice Emotion Detection
```python
voice_config = {
    "emotion_detection": {
        "voice_analysis": True,
        "audio_sample_rate": 16000,
        "voice_features": ["pitch", "energy", "spectral_centroid", "mfcc"]
    }
}
```

## API Usage

### Basic Emotional Processing

```python
from ai.emotional_ai_engine import EmotionalAIEngine

# Initialize
emotional_ai = EmotionalAIEngine(config)

# Process message with emotional intelligence
response = emotional_ai.process_with_emotional_intelligence(
    user_input="I'm feeling stressed about work",
    original_response="I can help with work questions",
    language="en",
    user_id="user123"
)

print(response.adapted_response)  # Enhanced with empathy
print(response.emotion_context)   # Detected emotional state
print(response.personality_used)  # Adapted personality
```

### Individual Component Usage

#### Emotion Detection
```python
from ai.emotion_detection import EmotionDetectionSystem

emotion_detector = EmotionDetectionSystem(config)
emotion = emotion_detector.detect_emotion_from_text("I'm so happy!", "en")
print(f"Emotion: {emotion.primary_emotion}, Confidence: {emotion.confidence}")
```

#### Personality Adaptation
```python
from ai.personality_system import PersonalitySystem

personality = PersonalitySystem(config)
personality.set_personality("professional")

adapted_response = personality.adapt_response_to_emotion(
    response="Here's the information",
    emotion_context={"primary_emotion": "anxiety", "stress_level": 0.8},
    language="en"
)
```

### Web Integration

```python
from web_emotional_integration import WebEmotionalIntegration

web_ai = WebEmotionalIntegration(config)

# Initialize session
session_result = web_ai.initialize_web_session("session123", {"language": "en"})

# Process web message
result = web_ai.process_web_message(
    session_id="session123",
    message="This is confusing",
    original_response="Let me explain",
    language="en"
)
```

## Web Interface

### Enhanced Web Application

The emotional AI system includes an enhanced web application (`jarvis_emotional_web_app.py`) with:

- **Real-time emotion display**: Visual indicators of detected emotions
- **Personality selector**: Users can choose or system auto-adapts personality
- **Emotional analytics**: Session-level emotional insights
- **Adaptive UI**: Interface adapts to user's emotional state

### API Endpoints

#### Core Endpoints
- `POST /api/message` - Process messages with emotional AI
- `GET /api/status` - System status and capabilities
- `POST /api/personality` - Change personality settings
- `GET /api/emotional-summary` - Session emotional analysis

#### WebSocket Events
- `message` - Send message for processing
- `response` - Receive enhanced response
- `emotion_detected` - Real-time emotion updates
- `personality_changed` - Personality adaptation notifications

## Installation

### Install Dependencies

```bash
# Install core requirements
pip install -r requirements.txt

# Install emotional AI dependencies
pip install -r requirements_emotional_ai.txt
```

### Download Models (Optional)

```python
# Download transformer models
from transformers import pipeline

# Emotion detection model
emotion_model = pipeline("text-classification", 
                        model="j-hartmann/emotion-english-distilroberta-base")

# Sentiment analysis model
sentiment_model = pipeline("sentiment-analysis",
                          model="cardiffnlp/twitter-roberta-base-sentiment-latest")
```

### Setup Thai Language Support

```bash
# Install Thai NLP tools
pip install pythainlp thai-segmenter

# Download Thai language data
python -c "import pythainlp; pythainlp.corpus.download('thai2fit_wv')"
```

## Testing

### Run Comprehensive Tests

```bash
# Run all emotional AI tests
python test_emotional_ai_system.py
```

### Test Individual Components

```python
# Test emotion detection
from ai.emotion_detection import EmotionDetectionSystem
detector = EmotionDetectionSystem({})
result = detector.detect_emotion_from_text("I'm excited!", "en")

# Test personality system
from ai.personality_system import PersonalitySystem
personality = PersonalitySystem({})
personality.set_personality("casual")
```

### Performance Benchmarks

Expected performance on standard hardware:
- **Emotion Detection**: ~50ms per request
- **Sentiment Analysis**: ~30ms per request
- **Full Emotional Processing**: ~200ms per request
- **Memory Usage**: ~500MB with transformer models

## Data Privacy

### User Data Handling

The emotional AI system follows privacy-first principles:

- **Local Processing**: All emotion detection runs locally
- **Encrypted Storage**: User preferences encrypted at rest
- **Data Minimization**: Only necessary emotional context stored
- **User Control**: Users can delete their emotional profiles
- **No External APIs**: No emotional data sent to external services

### Privacy Configuration

```python
privacy_config = {
    "user_preferences": {
        "enable_storage": True,          # Allow preference storage
        "encryption_enabled": True,      # Encrypt user data
        "auto_cleanup_days": 30,        # Auto-delete old data
        "anonymous_mode": False         # Use anonymous user IDs
    }
}
```

## Troubleshooting

### Common Issues

#### Model Download Failures
```bash
# Manually download models
python -c "
from transformers import pipeline
pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')
"
```

#### Thai Language Issues
```bash
# Reinstall Thai support
pip uninstall pythainlp
pip install pythainlp --upgrade
python -c "import pythainlp; pythainlp.corpus.download('thai2fit_wv')"
```

#### Memory Issues
```python
# Reduce model memory usage
config = {
    "emotion_detection": {
        "use_cpu": True,              # Force CPU usage
        "low_memory_mode": True,      # Reduce memory usage
        "cache_max_size": 10          # Smaller cache
    }
}
```

### Performance Optimization

#### GPU Acceleration
```python
# Enable GPU for faster processing
config = {
    "emotion_detection": {
        "device": 0,  # Use GPU 0
        "batch_processing": True
    }
}
```

#### Caching Strategy
```python
# Optimize caching for better performance
config = {
    "emotional_ai": {
        "cache_max_size": 100,        # Larger cache
        "cache_timeout": 300,         # 5 minutes
        "enable_response_cache": True
    }
}
```

## Contributing

### Development Setup

1. **Clone and Setup**
   ```bash
   git clone <repository>
   cd jarvis-voice-assistant
   pip install -r requirements_emotional_ai.txt
   ```

2. **Run Tests**
   ```bash
   python test_emotional_ai_system.py
   ```

3. **Code Style**
   - Follow PEP 8 guidelines
   - Use type hints for all functions
   - Add comprehensive docstrings
   - Include unit tests for new features

### Adding New Personalities

```python
# Example: Adding a "Scientific" personality
scientific_personality = PersonalityProfile(
    name="scientific",
    description="Analytical and fact-based assistant",
    traits={
        "formality": 0.8,
        "technical_detail": 0.9,
        "empathy": 0.4,
        "precision": 0.9
    },
    response_style={
        "tone": "analytical",
        "evidence_based": True,
        "uncertainty_acknowledgment": True
    }
)
```

### Adding New Emotions

```python
# Example: Adding custom emotions
custom_emotions = {
    "anticipation": {
        "keywords": ["excited", "looking forward", "can't wait"],
        "valence": 0.6,
        "arousal": 0.7
    },
    "nostalgia": {
        "keywords": ["remember", "miss", "old days"],
        "valence": 0.3,
        "arousal": 0.2
    }
}
```

## Roadmap

### Planned Features

1. **Advanced Voice Emotion Detection**
   - Real-time voice emotion analysis
   - Speaker emotion profiling
   - Multi-speaker emotion tracking

2. **Enhanced Cultural Adaptation**
   - Support for more languages
   - Cultural context understanding
   - Region-specific personality traits

3. **Advanced Analytics**
   - Emotional journey visualization
   - Conversation insights dashboard
   - Emotional health monitoring

4. **Integration Improvements**
   - REST API documentation
   - Plugin system for custom personalities
   - Third-party emotion model support

### Performance Goals

- **Latency**: <100ms for emotion detection
- **Accuracy**: >85% for emotion classification
- **Memory**: <200MB base memory usage
- **Languages**: Support for 10+ languages

## License

This emotional AI system is part of the JARVIS Voice Assistant project and follows the same licensing terms as the main project.

## Support

For issues, questions, or contributions related to the emotional AI system:

1. **Check Documentation**: Review this document and inline code documentation
2. **Run Tests**: Use the test suite to identify issues
3. **Check Logs**: Enable debug logging for detailed error information
4. **Performance Monitoring**: Use built-in performance metrics

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed emotional AI logging
config = {
    "emotional_ai": {
        "debug_mode": True,
        "log_emotional_context": True,
        "performance_monitoring": True
    }
}
```

---

## Example Usage Scenarios

### Customer Support Bot
```python
# Detect frustrated customer
emotion = emotion_detector.detect_emotion_from_text(
    "This is the third time I'm calling and nobody can help me!", "en"
)
# Result: emotion="anger", stress_level=0.9

# Adapt to professional, empathetic response
personality.set_personality("professional")
adapted_response = personality.adapt_response_to_emotion(
    "I can help you with that",
    {"primary_emotion": "anger", "stress_level": 0.9},
    "en"
)
# Result: "I understand your frustration and I'm here to help resolve this immediately."
```

### Educational Assistant
```python
# Detect confused student
emotion = emotion_detector.detect_emotion_from_text(
    "I don't understand this concept at all", "en"
)
# Result: emotion="confusion", confidence=0.8

# Adapt to patient, step-by-step explanation
response = personality.adapt_response_to_emotion(
    "Here's how it works...",
    {"primary_emotion": "confusion", "confidence": 0.8},
    "en"
)
# Result: Adds step-by-step breakdown and encouragement
```

### Thai Cultural Context
```python
# Detect Thai politeness level
thai_input = "กรุณาช่วยอธิบายให้ฟังหน่อยครับ"
emotion = emotion_detector.detect_emotion_from_text(thai_input, "th")

# Adapt with appropriate Thai politeness
adapted = personality.adapt_response_to_emotion(
    "อธิบายให้ฟัง",
    {"primary_emotion": "curiosity", "formality": "high"},
    "th"
)
# Result: "ขออธิบายให้ฟังนะครับ..." (with appropriate particles)
```

This comprehensive emotional AI system transforms JARVIS from a simple voice assistant into an emotionally intelligent companion that understands, adapts, and learns from human emotional expressions across different languages and cultural contexts.