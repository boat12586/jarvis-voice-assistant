# JARVIS Voice Assistant - Requirements Document

## Introduction

This document outlines the comprehensive requirements for a fully local, voice-driven AI assistant inspired by J.A.R.V.I.S from Iron Man. The system operates on local hardware (RTX 2050+ GPU), featuring a voice-first interface with floating visual overlays and advanced AI capabilities. The assistant uses state-of-the-art models including DeepSeek-R1 for reasoning and F5-TTS for voice synthesis, providing a cinematic and intelligent user experience.

## System Requirements (Updated 2025)

### Minimum Hardware Requirements
- **GPU**: NVIDIA RTX 2050 (4GB VRAM) or equivalent
- **RAM**: 16GB DDR4/DDR5 (increased for DeepSeek-R1)
- **Storage**: 25GB NVMe SSD (for models and data)
- **CPU**: 8-core processor (Intel i5/AMD Ryzen 5+)
- **Audio**: Microphone and speakers/headphones

### Recommended Hardware Requirements
- **GPU**: NVIDIA RTX 3070+ (8GB+ VRAM)
- **RAM**: 32GB DDR4/DDR5
- **Storage**: 50GB NVMe SSD
- **CPU**: 12-core processor (Intel i7/AMD Ryzen 7+)
- **Audio**: High-quality USB microphone and studio monitors

### Software Requirements
- **OS**: Windows 11, macOS 12+, or Ubuntu 20.04+
- **Python**: 3.9+ with CUDA support
- **NVIDIA Drivers**: Latest stable version
- **Docker**: Optional for containerized deployment

## Requirements

### Requirement 1: Voice Interface System

**User Story:** As a user, I want to interact with the assistant exclusively through voice commands and receive responses in a J.A.R.V.I.S-like voice, so that I can have a hands-free, immersive experience.

#### Acceptance Criteria

1. WHEN the user activates the voice input button THEN the system SHALL begin listening for voice commands.
2. WHEN the system receives voice input THEN it SHALL process the command without requiring text input.
3. WHEN the system responds to the user THEN it SHALL use the F5-TTS pipeline with a cloned J.A.R.V.I.S voice.
4. WHEN generating voice responses THEN the system SHALL apply mild reverb and metallic tone effects to match the J.A.R.V.I.S character.
5. WHEN the system is not explicitly triggered THEN it SHALL remain silent.

### Requirement 2: Multilingual Capabilities

**User Story:** As a bilingual user, I want the assistant to understand and respond in both Thai and English, so that I can communicate in my preferred language.

#### Acceptance Criteria

1. WHEN the user speaks in English THEN the system SHALL respond in English.
2. WHEN the user speaks in Thai THEN the system SHALL respond in Thai.
3. WHEN the user requests language learning mode THEN the system SHALL provide translations and explanations between Thai and English.
4. WHEN translating content THEN the system SHALL provide both the translation and explanation using TTS.

### Requirement 3: Visual Interface Components

**User Story:** As a user, I want a minimal visual interface with essential control buttons and floating information overlays, so that I can control the assistant and view information without a traditional chat interface.

#### Acceptance Criteria

1. WHEN the application launches THEN the system SHALL display the following buttons: Talk to AI, Show News or Article, Translate or Explain, Teach me a language, Ask a deep question, and Generate image with ComfyUI.
2. WHEN displaying information content THEN the system SHALL create popup overlays that appear for 10-20 seconds.
3. WHEN showing popup content THEN the system SHALL simultaneously read the content aloud using the TTS system.
4. WHEN the user requests visual information THEN the system SHALL display relevant information in an elegant, cinematic overlay.

### Requirement 4: Core Assistant Capabilities

**User Story:** As a user, I want the assistant to perform various helpful tasks like summarization, translation, explanation, and answering questions, so that it can serve as an intelligent companion.

#### Acceptance Criteria

1. WHEN the user asks a general knowledge question THEN the system SHALL provide an accurate, concise answer.
2. WHEN the user requests a summary of content THEN the system SHALL automatically summarize long inputs before reading them aloud.
3. WHEN the user asks for historical or current information THEN the system SHALL retrieve and present relevant data.
4. WHEN the user asks the system to explain a concept THEN the system SHALL provide clear, educational explanations.
5. WHEN responding to any query THEN the system SHALL maintain a confident, slightly robotic, J.A.R.V.I.S-like tone.

### Requirement 5: Integration with External Tools

**User Story:** As a user, I want the assistant to integrate with tools like ComfyUI, document readers, and information sources, so that it can provide rich, multimodal assistance.

#### Acceptance Criteria

1. WHEN the user requests an image generation THEN the system SHALL trigger the ComfyUI pipeline with the appropriate parameters.
2. WHEN the user says "Read this" THEN the system SHALL accept file or clipboard content and read it aloud using F5-TTS.
3. WHEN the user asks "What's the news" THEN the system SHALL fetch headlines, summarize them, display them in a popup, and read them aloud.
4. WHEN the user requests to open a visual THEN the system SHALL display an information overlay with relevant content.

### Requirement 6: Learning Mode

**User Story:** As a language learner, I want a dedicated learning mode for Thai-English practice, so that I can improve my language skills.

#### Acceptance Criteria

1. WHEN the user activates the "Teach me a language" function THEN the system SHALL enter tutor mode.
2. WHEN in tutor mode THEN the system SHALL provide vocabulary, phrases, and explanations in both Thai and English.
3. WHEN the user practices pronunciation THEN the system SHALL provide feedback on accuracy.
4. WHEN in learning mode THEN the system SHALL adapt difficulty based on user performance.

### Requirement 7: Local Processing and Privacy

**User Story:** As a privacy-conscious user, I want all processing to happen locally on my device, so that my data remains private and the system works without internet connectivity.

#### Acceptance Criteria

1. WHEN processing voice commands THEN the system SHALL perform speech recognition locally using Faster-Whisper.
2. WHEN generating voice responses THEN the system SHALL use the local F5-TTS pipeline with J.A.R.V.I.S voice clone.
3. WHEN retrieving information THEN the system SHALL prioritize local RAG with mxbai-embed-large embeddings.
4. WHEN generating images THEN the system SHALL use the local ComfyUI installation.
5. WHEN processing AI queries THEN the system SHALL use local DeepSeek-R1 model for reasoning.

### Requirement 8: Advanced AI Capabilities (New)

**User Story:** As a user, I want an intelligent assistant that can reason, remember conversations, and learn from interactions, so that it becomes more helpful over time.

#### Acceptance Criteria

1. WHEN asked complex questions THEN the system SHALL use DeepSeek-R1 for multi-step reasoning.
2. WHEN in conversation THEN the system SHALL maintain context and memory across turns.
3. WHEN processing knowledge THEN the system SHALL use mxbai-embed-large for accurate semantic search.
4. WHEN learning new information THEN the system SHALL update its knowledge base appropriately.
5. WHEN interacting THEN the system SHALL maintain consistent J.A.R.V.I.S personality traits.

## Model Specifications

### Core AI Models
- **Primary LLM**: deepseek-ai/deepseek-r1-distill-llama-8b (8B parameters)
  - Context length: 8,192 tokens
  - Quantization: 8-bit for memory efficiency
  - Capabilities: Advanced reasoning, planning, multilingual support

- **Embedding Model**: mixedbread-ai/mxbai-embed-large-v1
  - Dimensions: 1,024 (upgraded from 384)
  - Capabilities: State-of-the-art semantic understanding
  - Multilingual: Excellent Thai and English support

- **Speech Recognition**: Faster-Whisper
  - Models: base, small, medium (configurable)
  - Languages: Thai, English with automatic detection
  - Performance: GPU-accelerated, sub-2-second response

- **Text-to-Speech**: F5-TTS
  - Voice: Custom J.A.R.V.I.S clone
  - Effects: Reverb, metallic tone processing
  - Quality: High-fidelity, natural-sounding output

### Performance Targets
- **Total Response Time**: <10 seconds end-to-end
- **Memory Usage**: <8GB RAM, <6GB VRAM
- **Voice Recognition Accuracy**: >95%
- **Knowledge Retrieval Accuracy**: >85%
- **System Availability**: >99% uptime