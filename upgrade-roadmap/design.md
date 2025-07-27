# JARVIS Voice Assistant - Design Document

## Overview

The JARVIS Voice Assistant is a state-of-the-art, locally-run AI system inspired by J.A.R.V.I.S from Iron Man. Built with cutting-edge 2025 AI models, it features advanced reasoning capabilities through DeepSeek-R1, superior semantic understanding via mxbai-embed-large, and natural voice interaction powered by F5-TTS. The system operates entirely offline, ensuring complete privacy while delivering intelligent, contextual assistance through a cinematic glassmorphic interface.

## Current System Status (2025)

### ✅ Implemented Components
- **Core Architecture**: Complete modular structure with PyQt6 UI
- **Voice Recognition**: Faster-Whisper working with GPU/CPU compatibility  
- **Configuration System**: YAML-based settings with environment adaptation
- **Memory Management**: Dynamic allocation and performance optimization
- **Testing Framework**: Comprehensive test suites for all components
- **Logging System**: Advanced logging with rotation and performance monitoring

### 🔄 Upgrading Components
- **Primary LLM**: DeepSeek-R1 (deepseek-ai/deepseek-r1-distill-llama-8b)
- **Embeddings**: mxbai-embed-large (mixedbread-ai/mxbai-embed-large-v1)
- **RAG System**: Enhanced with new embedding dimensions (384→1024)
- **Context Processing**: Expanded to 8,192 tokens

## Architecture

The application follows a modular architecture with the following key components:

1. **UI Layer**: Handles the visual interface, including the main dashboard, buttons, and overlay system.
2. **Voice Processing Layer**: Manages speech recognition and text-to-speech functionality.
3. **AI Engine**: Processes user queries and generates responses using local LLMs and RAG.
4. **Feature Modules**: Implements specialized assistant capabilities like translation, news, and language learning.
5. **System Integration**: Coordinates between components and manages resources.

### High-Level Architecture Diagram

```mermaid
graph TD
    User[User] <--> UI[UI Layer]
    UI <--> VoiceProc[Voice Processing Layer]
    UI <--> OverlaySystem[Overlay System]
    VoiceProc <--> AIEngine[AI Engine]
    AIEngine <--> FeatureModules[Feature Modules]
    AIEngine <--> LocalRAG[Local RAG System]
    FeatureModules <--> ExternalTools[External Tool Integration]
    VoiceProc <--> AudioSystem[Audio Processing System]
    SystemIntegration[System Integration] <--> UI
    SystemIntegration <--> VoiceProc
    SystemIntegration <--> AIEngine
    SystemIntegration <--> FeatureModules
```

## Components and Interfaces

### UI Layer

#### Main Dashboard
- **Glassmorphic Window**: Frameless, translucent window with blur effects
- **Action Buttons**: Six main buttons with neon glow effects
- **Voice Visualization**: Audio waveform animation with reactive effects

#### Overlay System
- **Base Overlay Class**: Translucent, floating windows with animations
- **Specialized Overlays**: Text, news, translation, and image overlays
- **Overlay Manager**: Handles creation, positioning, and lifecycle

### Voice Processing Layer

#### Speech Recognition
- **Whisper Model**: Local speech-to-text processing
- **Language Detection**: Automatic identification of Thai or English
- **Audio Capture**: Low-latency microphone input system

#### Text-to-Speech
- **F5-TTS Pipeline**: Local TTS with J.A.R.V.I.S voice clone
- **Audio Effects**: Reverb and metallic tone processing
- **Playback System**: Low-latency audio output

#### Voice Interaction Controller
- **State Machine**: Manages listening, processing, and response states
- **Event System**: Coordinates voice input and output timing
- **Error Handling**: Manages recognition failures and fallbacks

### AI Engine (Updated 2025)

#### Advanced LLM System
- **Primary Model**: DeepSeek-R1 (deepseek-ai/deepseek-r1-distill-llama-8b)
  - 8B parameters with 8-bit quantization
  - Context length: 8,192 tokens  
  - Advanced reasoning and planning capabilities
  - Enhanced Thai language understanding
- **Fallback Model**: Microsoft DialoGPT-medium (for backup scenarios)
- **Model Management**: Dynamic loading/unloading based on memory constraints
- **Prompt Engineering**: Specialized templates for reasoning, conversation, and task execution

#### Enhanced RAG System  
- **Vector Database**: ChromaDB with persistent storage
- **Embedding Model**: mxbai-embed-large (1,024 dimensions)
  - State-of-the-art semantic understanding
  - Superior multilingual support (Thai/English)
  - 3x improvement in search accuracy
- **Document Processing**: Intelligent chunking with overlap optimization
- **Hybrid Search**: Combines semantic similarity with keyword matching
- **Knowledge Validation**: Fact-checking and source verification

#### Advanced Response Generation
- **Multi-Step Reasoning**: DeepSeek-R1 powered complex problem solving
- **Context Awareness**: Maintains conversation history and user preferences  
- **Personality Engine**: Consistent J.A.R.V.I.S character traits
- **Adaptive Responses**: Adjusts complexity based on user interaction patterns
- **Error Recovery**: Graceful handling of model failures with fallback strategies

### Feature Modules

#### News System
- **Local Database**: Periodically updated news articles
- **Summarization Pipeline**: Creates concise news briefs
- **Category Filter**: Organizes news by topic

#### Translation Module
- **Bilingual Engine**: Thai-English translation
- **Explanation Generator**: Provides context for translations
- **Language Switcher**: Handles code-switching in conversations

#### Language Learning
- **Lesson Generator**: Creates vocabulary and phrase exercises
- **Pronunciation Analyzer**: Provides feedback on speaking
- **Adaptive Difficulty**: Adjusts based on user performance

#### ComfyUI Integration
- **API Client**: Communicates with local ComfyUI instance
- **Workflow Manager**: Handles image generation pipelines
- **Prompt Enhancer**: Improves image generation prompts

### System Integration

#### Application Controller
- **Event Bus**: Central communication system
- **State Manager**: Tracks application mode and status
- **Error Handler**: Manages recovery from failures

#### Configuration System
- **Settings Storage**: Persists user preferences
- **Theme Manager**: Handles visual customization
- **Resource Configuration**: Manages system resource allocation

#### Resource Manager (Enhanced)
- **Smart Memory Management**: Intelligent allocation for 8B parameter models
- **Dynamic Model Loading**: Loads/unloads models based on usage patterns
- **Performance Monitoring**: Real-time tracking of GPU/RAM/CPU usage
- **Adaptive Quality**: Automatically adjusts model parameters under constraints
- **Thermal Management**: Monitors system temperature and throttles if needed
- **Startup Optimization**: Lazy loading and background initialization

## System Performance Metrics

### Resource Utilization Targets
```yaml
Memory Usage:
  - RAM: 6-10GB (with DeepSeek-R1 loaded)
  - VRAM: 4-6GB (depending on quantization)
  - Storage: 25GB+ (models and data)

Performance Targets:
  - Startup Time: <10 seconds
  - Voice Recognition: <2 seconds
  - AI Processing: 3-8 seconds
  - Total Response: <10 seconds end-to-end

System Requirements:
  - Minimum: RTX 2050, 16GB RAM
  - Recommended: RTX 3070+, 32GB RAM
```

### Quality Metrics
```yaml
Accuracy Targets:
  - Voice Recognition: >95%
  - Knowledge Retrieval: >85%
  - Response Relevance: >90%
  - System Uptime: >99%

User Experience:
  - Natural conversation flow
  - Consistent personality
  - Multilingual fluency
  - Contextual awareness
```

## Data Models

### User Query
```
{
  "text": string,           // Raw text from speech recognition
  "language": string,       // Detected language code
  "intent": string,         // Classified user intent
  "parameters": Object,     // Extracted parameters from query
  "timestamp": DateTime     // When the query was received
}
```

### Assistant Response
```
{
  "text": string,           // Text to be spoken
  "language": string,       // Response language code
  "visualContent": {        // Optional visual content
    "type": string,         // "text", "image", "news", etc.
    "content": Object       // Type-specific content
  },
  "sources": Array,         // Reference sources if applicable
  "timestamp": DateTime     // When the response was generated
}
```

### Overlay Content
```
{
  "type": string,           // Overlay type
  "title": string,          // Optional title
  "content": Object,        // Type-specific content
  "duration": number,       // Display duration in seconds
  "position": {             // Screen position
    "x": number,
    "y": number
  },
  "size": {                 // Dimensions
    "width": number,
    "height": number
  }
}
```

### Configuration
```
{
  "ui": {
    "theme": string,        // UI theme name
    "opacity": number,      // Window opacity
    "scale": number         // UI scaling factor
  },
  "voice": {
    "inputDevice": string,  // Microphone device ID
    "outputDevice": string, // Speaker device ID
    "volume": number,       // Output volume
    "effects": {            // Voice effect settings
      "reverb": number,
      "metallic": number
    }
  },
  "ai": {
    "model": string,        // LLM model name
    "temperature": number,  // Response randomness
    "contextLength": number // Context window size
  },
  "system": {
    "gpuMemoryLimit": number, // Max GPU memory usage
    "startWithSystem": boolean // Auto-start with OS
  }
}
```

## Error Handling

### Voice Processing Errors
- **Recognition Failures**: Prompt user to repeat or rephrase
- **TTS Failures**: Fall back to simpler voice model or text display
- **Audio Device Errors**: Provide visual feedback and configuration options

### AI Engine Errors
- **Model Loading Failures**: Graceful degradation to smaller models
- **Response Generation Timeouts**: Implement early stopping with partial results
- **Context Overflow**: Automatically summarize and truncate context

### System Errors
- **Resource Limitations**: Adjust model parameters or disable features
- **External Tool Failures**: Provide meaningful error messages and alternatives
- **Unexpected Crashes**: Log diagnostics and implement auto-recovery

## Testing Strategy

### Unit Testing
- **Voice Processing**: Test recognition accuracy and TTS quality
- **AI Components**: Validate response quality and consistency
- **UI Elements**: Verify rendering and interaction behavior

### Integration Testing
- **End-to-End Flows**: Test complete user interaction scenarios
- **Performance Testing**: Measure response times and resource usage
- **Multilingual Testing**: Verify Thai and English functionality

### User Experience Testing
- **Voice Recognition Accuracy**: Test with different accents and environments
- **Response Quality**: Evaluate helpfulness and naturalness
- **UI Responsiveness**: Measure perceived performance and smoothness

## Implementation Considerations

### Performance Optimization
- Use 8-bit quantization for LLM to reduce memory footprint
- Implement dynamic model loading/unloading based on usage
- Optimize overlay rendering with hardware acceleration

### Privacy and Security
- Ensure all processing happens locally without external API calls
- Implement secure storage for any sensitive configuration data
- Provide clear user controls for data management

### Accessibility
- Support keyboard shortcuts for all voice-activated features
- Implement high-contrast mode for visual elements
- Provide text alternatives for all voice interactions

### Extensibility
- Design plugin architecture for future feature modules
- Create standardized interfaces for component communication
- Document extension points for customization