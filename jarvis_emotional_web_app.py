#!/usr/bin/env python3
"""
JARVIS Voice Assistant Web Application with Emotional Intelligence
แอปพลิเคชันเว็บสำหรับผู้ช่วยเสียง JARVIS พร้อมปัญญาประดิษฐ์ทางอารมณ์
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any
from flask import Flask, render_template, render_template_string, request, jsonify, session
from flask_socketio import SocketIO, emit
import secrets

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# Initialize SocketIO for real-time communication
socketio = SocketIO(app, cors_allowed_origins="*")

# Global variables for JARVIS components
jarvis_components = {}

def initialize_jarvis():
    """Initialize JARVIS components with emotional AI"""
    try:
        from voice.command_parser import VoiceCommandParser
        # Try to import Thai language processor with fallback
        try:
            from features.thai_language_enhanced import ThaiLanguageProcessor
        except ImportError:
            try:
                import sys
                import os
                src_path = os.path.join(os.path.dirname(__file__), 'src')
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)
                from features.thai_language_enhanced import ThaiLanguageProcessor
            except ImportError:
                ThaiLanguageProcessor = None
                print("Warning: Thai language support not available")
        from features.conversation_memory import ConversationMemorySystem
        from features.news_system import NewsSystem
        from ai.ai_engine import AIEngine
        
        # Import emotional AI components
        from web_emotional_integration import WebEmotionalIntegration
        
        config = {
            "command_parser": {"enabled": True, "confidence_threshold": 0.6},
            "thai_language": {"enabled": True},
            "conversation_memory": {
                "max_turns_per_session": 50,
                "context_window_size": 10,
                "memory_dir": "data/web_conversation_memory"
            },
            "emotional_ai": {
                "auto_personality_adaptation": True,
                "emotion_memory_length": 20,
                "cache_max_size": 50
            },
            "emotion_detection": {
                "voice_analysis": False,  # Disabled for web initially
                "max_history_length": 50
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
            "enable_voice_emotion": False,
            "enable_auto_personality": True,
            "enable_user_learning": True,
            "session_timeout": 3600
        }
        
        jarvis_components['command_parser'] = VoiceCommandParser(config)
        if ThaiLanguageProcessor is not None:
            jarvis_components['thai_processor'] = ThaiLanguageProcessor(config)
        else:
            jarvis_components['thai_processor'] = None
        jarvis_components['conversation_memory'] = ConversationMemorySystem(config)
        jarvis_components['news_system'] = NewsSystem(config)
        jarvis_components['ai_engine'] = AIEngine(config)
        
        # Initialize emotional AI integration
        jarvis_components['emotional_integration'] = WebEmotionalIntegration(config)
        
        jarvis_components['initialized'] = True
        
        print("✅ JARVIS components with Emotional AI initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize JARVIS: {e}")
        jarvis_components['error'] = str(e)
        return False

def process_user_message_with_emotions(text: str, session_id: str):
    """Process user message with emotional intelligence"""
    try:
        if not jarvis_components.get('initialized'):
            return {
                "error": "JARVIS ยังไม่พร้อม กรุณารอสักครู่",
                "status": "not_ready"
            }
        
        # Get components
        command_parser = jarvis_components['command_parser']
        thai_processor = jarvis_components['thai_processor']
        conversation_memory = jarvis_components['conversation_memory']
        emotional_integration = jarvis_components['emotional_integration']
        
        # Detect language
        language = "th" if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else "en"
        
        # Process Thai if needed
        thai_context = {}
        if language == "th" and thai_processor:
            thai_result = thai_processor.process_thai_text(text)
            thai_context = thai_processor.enhance_for_ai_processing(text)
        
        # Parse command
        parsed = command_parser.parse_command(text, language)
        
        # Start session if not exists
        memory_session_id = None
        if f"session_{session_id}" not in jarvis_components:
            memory_session_id = conversation_memory.start_session(f"web_user_{session_id}", language)
            jarvis_components[f"session_{session_id}"] = memory_session_id
        else:
            memory_session_id = jarvis_components[f"session_{session_id}"]
        
        # Get conversation context
        context = conversation_memory.get_conversation_context(parsed.cleaned_text, max_turns=3)
        
        # Generate basic response first
        original_response = generate_jarvis_response(parsed, thai_context, context)
        
        # Enhance with emotional AI
        emotional_result = emotional_integration.process_web_message(
            session_id=session_id,
            message=text,
            original_response=original_response,
            language=language
        )
        
        # Use enhanced response if available
        final_response = original_response
        if emotional_result.get("status") == "success":
            final_response = emotional_result.get("enhanced_response", original_response)
        
        # Add to conversation memory
        turn_id = conversation_memory.add_conversation_turn(
            user_input=text,
            user_language=language,
            processed_input=parsed.cleaned_text,
            intent=parsed.intent,
            entities=parsed.entities,
            assistant_response=final_response,
            response_language=language,
            confidence=parsed.confidence
        )
        
        # Prepare response with emotional data
        response_data = {
            "response": final_response,
            "metadata": {
                "intent": parsed.intent,
                "confidence": round(parsed.confidence * 100),
                "entities": parsed.entities,
                "language": language,
                "thai_context": thai_context.get("cultural_notes", ""),
                "turn_id": turn_id,
                "context_turns": len(context)
            },
            "status": "success"
        }
        
        # Add emotional AI data if available
        if emotional_result.get("status") == "success":
            response_data["emotional_analysis"] = emotional_result.get("emotional_analysis", {})
            response_data["personality_info"] = emotional_result.get("personality_info", {})
            response_data["processing_info"] = emotional_result.get("processing_info", {})
            response_data["adaptations_applied"] = emotional_result.get("personality_info", {}).get("adaptations_applied", [])
        
        return response_data
        
    except Exception as e:
        return {
            "error": f"เกิดข้อผิดพลาด: {e}",
            "status": "error"
        }

def generate_jarvis_response(parsed, thai_context, context):
    """Generate appropriate JARVIS response using AI engine"""
    intent = parsed.intent
    text = parsed.cleaned_text
    language = parsed.language
    
    # Try to use AI engine first
    if 'ai_engine' in jarvis_components:
        try:
            ai_response = jarvis_components['ai_engine'].process_query(text, context=context)
            if ai_response and ai_response.strip():
                return ai_response
        except Exception as e:
            print(f"AI engine failed: {e}")
    
    # Handle specific intents with enhanced responses
    if intent == "news_request" or "ข่าว" in text or "news" in text.lower():
        return handle_news_request(text, language)
    elif intent == "question" or "คือ" in text or "ทำไม" in text or "อะไร" in text:
        return handle_knowledge_query(text, language)
    
    if intent == "greeting":
        if language == "th":
            return """สวัสดีครับ! ผม JARVIS ผู้ช่วยอัจฉริยะของคุณ 🤖

🌟 ความสามารถของผม:
• ตอบคำถามเกี่ยวกับเทคโนโลยีและ AI
• อธิบายแนวคิดซับซ้อนให้เข้าใจง่าย
• สนทนาภาษาไทยและภาษาอังกฤษ
• จำบริบทการสนทนาของเรา
• เข้าใจและตอบสนองต่ออารมณ์ของคุณ

มีอะไรให้ช่วยไหมครับ? 😊"""
        else:
            return """Hello! I'm JARVIS, your intelligent assistant with emotional intelligence 🤖

🌟 My capabilities:
• Answer questions about technology and AI
• Explain complex concepts in simple terms
• Converse in both Thai and English
• Remember our conversation context
• Understand and respond to your emotions

How can I help you today? 😊"""
            
    elif intent == "information_request":
        if "ปัญญาประดิษฐ์" in text or "artificial intelligence" in text.lower():
            if language == "th":
                return """ปัญญาประดิษฐ์ (Artificial Intelligence) คือเทคโนโลยีที่ทำให้คอมพิวเตอร์สามารถเรียนรู้ คิด และตัดสินใจได้เหมือนมนุษย์ครับ 🧠

📚 ความสามารถหลักของ AI:
🔹 Machine Learning - การเรียนรู้จากข้อมูล
🔹 Natural Language Processing - การประมวลผลภาษาธรรมชาติ
🔹 Computer Vision - การมองเห็นและเข้าใจภาพ
🔹 Emotional Intelligence - การเข้าใจและตอบสนองอารมณ์

🎯 การใช้งาน AI ในชีวิตประจำวัน:
• ผู้ช่วยเสียงอัจฉริยะ (เหมือนผม!)
• ระบบแนะนำสินค้า
• การแปลภาษาอัตโนมัติ
• การวิเคราะห์ทางการแพทย์

มีคำถามเพิ่มเติมเกี่ยวกับ AI หรือไม่ครับ? 🤔"""
            else:
                return """Artificial Intelligence (AI) is technology that enables computers to learn, think, and make decisions like humans 🧠

📚 Core AI Capabilities:
🔹 Machine Learning - Learning from data
🔹 Natural Language Processing - Understanding human language
🔹 Computer Vision - Seeing and understanding images
🔹 Emotional Intelligence - Understanding and responding to emotions

🎯 AI Applications in Daily Life:
• Intelligent voice assistants (like me!)
• Recommendation systems
• Automatic language translation
• Medical analysis

Do you have more questions about AI? 🤔"""
                
    # Default fallback response
    if language == "th":
        return f"ขอบคุณสำหรับคำถาม '{text}' ครับ ผมเข้าใจว่าคุณต้องการ {intent} ผมจะพยายามช่วยเหลือให้ดีที่สุดครับ! 🤖"
    else:
        return f"Thank you for your question '{text}'. I understand you're looking for {intent}. I'll do my best to help! 🤖"

def handle_news_request(text, language):
    """Handle news requests"""
    try:
        news_system = jarvis_components.get('news_system')
        if news_system:
            news_result = news_system.get_latest_news(category="general", count=3)
            if news_result.get("status") == "success":
                articles = news_result.get("articles", [])
                if articles:
                    if language == "th":
                        response = "📰 ข่าวล่าสุดสำหรับคุณ:\n\n"
                        for i, article in enumerate(articles[:3], 1):
                            response += f"{i}. {article.get('title', 'ไม่มีหัวข้อ')}\n"
                            response += f"   📅 {article.get('published_at', 'ไม่ระบุวันที่')}\n\n"
                    else:
                        response = "📰 Latest news for you:\n\n"
                        for i, article in enumerate(articles[:3], 1):
                            response += f"{i}. {article.get('title', 'No title')}\n"
                            response += f"   📅 {article.get('published_at', 'No date')}\n\n"
                    return response
        
        if language == "th":
            return "ขออภัยครับ ตอนนี้ไม่สามารถดึงข่าวสารได้ กรุณาลองใหม่ภายหลังครับ 📰"
        else:
            return "Sorry, I can't fetch news at the moment. Please try again later 📰"
            
    except Exception as e:
        if language == "th":
            return f"เกิดข้อผิดพลาดในการดึงข่าว: {e} 📰"
        else:
            return f"Error fetching news: {e} 📰"

def handle_knowledge_query(text, language):
    """Handle knowledge-based queries"""
    if language == "th":
        return f"นี่เป็นคำถามที่น่าสนใจเกี่ยวกับ '{text}' ครับ ผมกำลังพัฒนาความสามารถในการตอบคำถามให้ดีขึ้น ขณะนี้ผมสามารถช่วยเรื่องเทคโนโลยี AI และการเขียนโปรแกรมได้ดีครับ! 🧠"
    else:
        return f"That's an interesting question about '{text}'! I'm continuously improving my knowledge base. Currently, I'm best at helping with technology, AI, and programming topics! 🧠"

# Web routes
@app.route('/')
def index():
    """Main chat interface with emotional features"""
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS - Emotional AI Assistant</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.5.0/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            padding: 20px;
        }
        
        .chat-container {
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
            width: 100%;
            max-width: 900px;
            height: 90vh;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .header {
            background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
            color: white;
            padding: 20px;
            text-align: center;
            position: relative;
        }
        
        .header h1 {
            font-size: 24px;
            margin-bottom: 5px;
        }
        
        .header .subtitle {
            font-size: 14px;
            opacity: 0.9;
        }
        
        .emotional-status {
            position: absolute;
            top: 20px;
            right: 20px;
            display: flex;
            align-items: center;
            gap: 10px;
        }
        
        .emotion-indicator {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4CAF50;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.7; }
            100% { transform: scale(1); opacity: 1; }
        }
        
        .personality-selector {
            position: absolute;
            top: 20px;
            left: 20px;
        }
        
        .personality-select {
            background: rgba(255, 255, 255, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.3);
            border-radius: 20px;
            color: white;
            padding: 5px 15px;
            font-size: 12px;
        }
        
        .chat-messages {
            flex: 1;
            padding: 20px;
            overflow-y: auto;
            background: #f8f9fa;
        }
        
        .message {
            margin-bottom: 15px;
            display: flex;
            align-items: flex-start;
            gap: 10px;
        }
        
        .message.user {
            flex-direction: row-reverse;
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: bold;
            color: white;
        }
        
        .message.user .message-avatar {
            background: #007bff;
        }
        
        .message.assistant .message-avatar {
            background: #28a745;
        }
        
        .message-content {
            max-width: 70%;
            padding: 15px;
            border-radius: 15px;
            line-height: 1.5;
            position: relative;
        }
        
        .message.user .message-content {
            background: #007bff;
            color: white;
            border-bottom-right-radius: 5px;
        }
        
        .message.assistant .message-content {
            background: white;
            color: #333;
            border: 1px solid #e0e0e0;
            border-bottom-left-radius: 5px;
        }
        
        .emotional-tags {
            margin-top: 10px;
            display: flex;
            gap: 5px;
            flex-wrap: wrap;
        }
        
        .emotional-tag {
            background: rgba(0, 123, 255, 0.1);
            color: #007bff;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            border: 1px solid rgba(0, 123, 255, 0.3);
        }
        
        .adaptation-tag {
            background: rgba(40, 167, 69, 0.1);
            color: #28a745;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 11px;
            border: 1px solid rgba(40, 167, 69, 0.3);
        }
        
        .input-area {
            padding: 20px;
            background: white;
            border-top: 1px solid #e0e0e0;
        }
        
        .input-form {
            display: flex;
            gap: 10px;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            padding: 15px;
            border: 2px solid #e0e0e0;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .message-input:focus {
            border-color: #007bff;
        }
        
        .send-button {
            background: #007bff;
            color: white;
            border: none;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background-color 0.3s;
        }
        
        .send-button:hover {
            background: #0056b3;
        }
        
        .send-button:disabled {
            background: #ccc;
            cursor: not-allowed;
        }
        
        .typing-indicator {
            display: none;
            align-items: center;
            gap: 10px;
            margin-bottom: 15px;
            padding: 15px;
            background: rgba(40, 167, 69, 0.1);
            border-radius: 15px;
            border-left: 4px solid #28a745;
        }
        
        .typing-dots {
            display: flex;
            gap: 3px;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: #28a745;
            border-radius: 50%;
            animation: typing 1.4s infinite;
        }
        
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }
        
        @keyframes typing {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-10px); }
        }
        
        .status-area {
            padding: 10px 20px;
            background: #f8f9fa;
            border-top: 1px solid #e0e0e0;
            font-size: 12px;
            color: #666;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .connection-status {
            display: flex;
            align-items: center;
            gap: 5px;
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: #28a745;
        }
        
        .emotional-panel {
            background: rgba(255, 255, 255, 0.1);
            padding: 10px;
            border-radius: 10px;
            margin-top: 10px;
            font-size: 12px;
        }
        
        .emotion-bar {
            height: 4px;
            background: rgba(255, 255, 255, 0.3);
            border-radius: 2px;
            margin: 5px 0;
            overflow: hidden;
        }
        
        .emotion-fill {
            height: 100%;
            background: #4CAF50;
            transition: width 0.3s;
        }
    </style>
</head>
<body>
    <div class="chat-container">
        <div class="header">
            <div class="personality-selector">
                <select class="personality-select" id="personalitySelect">
                    <option value="friendly">😊 Friendly</option>
                    <option value="professional">👔 Professional</option>
                    <option value="casual">😎 Casual</option>
                </select>
            </div>
            
            <h1>JARVIS</h1>
            <div class="subtitle">Emotional AI Assistant • ผู้ช่วยปัญญาประดิษฐ์ทางอารมณ์</div>
            
            <div class="emotional-status">
                <div class="emotion-indicator"></div>
                <span id="currentEmotion">😊 Neutral</span>
            </div>
            
            <div class="emotional-panel">
                <div>Emotional Intelligence Active</div>
                <div class="emotion-bar">
                    <div class="emotion-fill" id="emotionBar" style="width: 60%"></div>
                </div>
                <div id="emotionalContext">Mood: Stable • Personality: Friendly</div>
            </div>
        </div>
        
        <div class="chat-messages" id="messages">
            <div class="message assistant">
                <div class="message-avatar">J</div>
                <div class="message-content">
                    Welcome! I'm JARVIS with emotional intelligence. I can understand your emotions and adapt my responses accordingly. How are you feeling today? 😊
                    <div class="emotional-tags">
                        <span class="emotional-tag">😊 Friendly</span>
                        <span class="adaptation-tag">Welcome Mode</span>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="typing-indicator" id="typingIndicator">
            <div>JARVIS is thinking</div>
            <div class="typing-dots">
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
                <div class="typing-dot"></div>
            </div>
        </div>
        
        <div class="input-area">
            <form class="input-form" id="messageForm">
                <input type="text" class="message-input" id="messageInput" 
                       placeholder="Type your message... พิมพ์ข้อความของคุณ..." 
                       autocomplete="off">
                <button type="submit" class="send-button" id="sendButton">
                    ➤
                </button>
            </form>
        </div>
        
        <div class="status-area">
            <div class="connection-status">
                <div class="status-dot" id="statusDot"></div>
                <span id="connectionStatus">Connected</span>
            </div>
            <div id="sessionInfo">Session active • Emotional AI enabled</div>
        </div>
    </div>

    <script>
        const socket = io();
        const messagesContainer = document.getElementById('messages');
        const messageInput = document.getElementById('messageInput');
        const sendButton = document.getElementById('sendButton');
        const typingIndicator = document.getElementById('typingIndicator');
        const currentEmotion = document.getElementById('currentEmotion');
        const emotionBar = document.getElementById('emotionBar');
        const emotionalContext = document.getElementById('emotionalContext');
        const personalitySelect = document.getElementById('personalitySelect');
        
        let isProcessing = false;
        let sessionId = 'session_' + Date.now() + '_' + Math.random().toString(36).substr(2, 9);
        
        // Emotion emoji mapping
        const emotionEmojis = {
            'joy': '😊', 'happiness': '😄', 'excitement': '🤩',
            'sadness': '😢', 'anger': '😠', 'fear': '😰',
            'surprise': '😲', 'disgust': '🤢', 'neutral': '😐',
            'anxiety': '😰', 'frustration': '😤', 'satisfaction': '😌'
        };
        
        // Initialize session
        socket.emit('initialize_session', {
            session_id: sessionId,
            user_context: {
                language: 'en',
                timezone: Intl.DateTimeFormat().resolvedOptions().timeZone
            }
        });
        
        // Handle session initialization
        socket.on('session_initialized', function(data) {
            console.log('Session initialized:', data);
            if (data.personality) {
                personalitySelect.value = data.personality;
            }
        });
        
        // Message form submission
        document.getElementById('messageForm').addEventListener('submit', function(e) {
            e.preventDefault();
            sendMessage();
        });
        
        // Personality selection
        personalitySelect.addEventListener('change', function() {
            const personality = this.value;
            socket.emit('set_personality', {
                session_id: sessionId,
                personality: personality
            });
        });
        
        function sendMessage() {
            const message = messageInput.value.trim();
            if (!message || isProcessing) return;
            
            isProcessing = true;
            sendButton.disabled = true;
            
            // Add user message to chat
            addMessage(message, 'user');
            
            // Show typing indicator
            typingIndicator.style.display = 'flex';
            
            // Send message to server
            socket.emit('message', {
                text: message,
                session_id: sessionId,
                timestamp: Date.now()
            });
            
            messageInput.value = '';
            scrollToBottom();
        }
        
        // Handle responses
        socket.on('response', function(data) {
            typingIndicator.style.display = 'none';
            isProcessing = false;
            sendButton.disabled = false;
            
            if (data.status === 'success') {
                // Add assistant message with emotional data
                addMessage(data.response, 'assistant', data);
                
                // Update emotional indicators
                updateEmotionalDisplay(data);
                
            } else {
                addMessage('ขออภัย เกิดข้อผิดพลาด: ' + (data.error || 'Unknown error'), 'assistant');
            }
            
            scrollToBottom();
        });
        
        // Handle personality changes
        socket.on('personality_changed', function(data) {
            personalitySelect.value = data.personality;
            updateEmotionalContext(data);
        });
        
        function addMessage(text, sender, emotionalData = null) {
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${sender}`;
            
            const avatar = document.createElement('div');
            avatar.className = 'message-avatar';
            avatar.textContent = sender === 'user' ? 'U' : 'J';
            
            const content = document.createElement('div');
            content.className = 'message-content';
            content.innerHTML = text.replace(/\\n/g, '<br>');
            
            // Add emotional tags if available
            if (emotionalData && sender === 'assistant') {
                const tagsDiv = document.createElement('div');
                tagsDiv.className = 'emotional-tags';
                
                // Emotion tag
                const emotionalAnalysis = emotionalData.emotional_analysis;
                if (emotionalAnalysis) {
                    const emotion = emotionalAnalysis.primary_emotion;
                    const emoji = emotionEmojis[emotion] || '😐';
                    const emotionTag = document.createElement('span');
                    emotionTag.className = 'emotional-tag';
                    emotionTag.textContent = `${emoji} ${emotion}`;
                    tagsDiv.appendChild(emotionTag);
                }
                
                // Adaptation tags
                const adaptations = emotionalData.adaptations_applied || [];
                adaptations.forEach(adaptation => {
                    const adaptTag = document.createElement('span');
                    adaptTag.className = 'adaptation-tag';
                    adaptTag.textContent = adaptation.replace(/_/g, ' ');
                    tagsDiv.appendChild(adaptTag);
                });
                
                if (tagsDiv.children.length > 0) {
                    content.appendChild(tagsDiv);
                }
            }
            
            messageDiv.appendChild(avatar);
            messageDiv.appendChild(content);
            messagesContainer.appendChild(messageDiv);
        }
        
        function updateEmotionalDisplay(data) {
            const emotionalAnalysis = data.emotional_analysis;
            if (!emotionalAnalysis) return;
            
            const emotion = emotionalAnalysis.primary_emotion;
            const confidence = emotionalAnalysis.confidence;
            const valence = emotionalAnalysis.valence;
            
            // Update emotion indicator
            const emoji = emotionEmojis[emotion] || '😐';
            currentEmotion.textContent = `${emoji} ${emotion}`;
            
            // Update emotion bar
            const barWidth = Math.max(20, confidence * 100);
            emotionBar.style.width = barWidth + '%';
            
            // Update color based on valence
            if (valence > 0.3) {
                emotionBar.style.background = '#4CAF50'; // Green for positive
            } else if (valence < -0.3) {
                emotionBar.style.background = '#f44336'; // Red for negative
            } else {
                emotionBar.style.background = '#FF9800'; // Orange for neutral
            }
            
            // Update context
            const personality = data.personality_info?.current_personality || 'friendly';
            emotionalContext.textContent = `Mood: ${emotion} • Personality: ${personality}`;
        }
        
        function updateEmotionalContext(data) {
            const personality = data.personality || 'friendly';
            emotionalContext.textContent = emotionalContext.textContent.replace(/Personality: \\w+/, `Personality: ${personality}`);
        }
        
        function scrollToBottom() {
            messagesContainer.scrollTop = messagesContainer.scrollHeight;
        }
        
        // Connection status
        socket.on('connect', function() {
            document.getElementById('statusDot').style.background = '#28a745';
            document.getElementById('connectionStatus').textContent = 'Connected';
        });
        
        socket.on('disconnect', function() {
            document.getElementById('statusDot').style.background = '#dc3545';
            document.getElementById('connectionStatus').textContent = 'Disconnected';
        });
    </script>
</body>
</html>
    """)

@app.route('/api/status')
def api_status():
    """API endpoint for system status"""
    try:
        if not jarvis_components.get('initialized'):
            return jsonify({
                "status": "initializing",
                "message": "JARVIS is starting up...",
                "components": {
                    "ai_engine": False,
                    "emotional_ai": False,
                    "user_preferences": False
                }
            })
        
        # Get emotional AI stats
        emotional_integration = jarvis_components.get('emotional_integration')
        stats = {}
        if emotional_integration:
            stats = emotional_integration.get_system_stats()
        
        return jsonify({
            "status": "ready",
            "message": "JARVIS with Emotional AI is ready",
            "components": {
                "ai_engine": True,
                "emotional_ai": True,
                "user_preferences": True,
                "thai_support": jarvis_components.get('thai_processor') is not None
            },
            "emotional_ai_stats": stats
        })
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": str(e)
        }), 500

@app.route('/api/message', methods=['POST'])
def api_message():
    """API endpoint for processing messages"""
    try:
        data = request.get_json()
        text = data.get('text', '')
        session_id = data.get('session_id', 'api_session')
        
        if not text:
            return jsonify({"error": "No text provided"}), 400
        
        result = process_user_message_with_emotions(text, session_id)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

@socketio.on('initialize_session')
def handle_initialize_session(data):
    """Initialize session with emotional AI"""
    try:
        session_id = data.get('session_id')
        user_context = data.get('user_context', {})
        
        if not session_id:
            emit('error', {'message': 'No session ID provided'})
            return
        
        emotional_integration = jarvis_components.get('emotional_integration')
        if emotional_integration:
            result = emotional_integration.initialize_web_session(session_id, user_context)
            emit('session_initialized', result)
        else:
            emit('error', {'message': 'Emotional AI not available'})
            
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('message')
def handle_message(data):
    """Handle incoming messages with emotional processing"""
    try:
        text = data.get('text', '')
        session_id = data.get('session_id', '')
        
        if not text:
            emit('error', {'message': 'No text provided'})
            return
        
        result = process_user_message_with_emotions(text, session_id)
        emit('response', result)
        
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('set_personality')
def handle_set_personality(data):
    """Handle personality changes"""
    try:
        session_id = data.get('session_id')
        personality = data.get('personality')
        
        emotional_integration = jarvis_components.get('emotional_integration')
        if emotional_integration:
            result = emotional_integration.set_session_personality(session_id, personality)
            emit('personality_changed', {'personality': personality, 'result': result})
        else:
            emit('error', {'message': 'Emotional AI not available'})
            
    except Exception as e:
        emit('error', {'message': str(e)})

@socketio.on('get_emotional_summary')
def handle_get_emotional_summary(data):
    """Get emotional summary for session"""
    try:
        session_id = data.get('session_id')
        
        emotional_integration = jarvis_components.get('emotional_integration')
        if emotional_integration:
            summary = emotional_integration.get_session_emotional_summary(session_id)
            emit('emotional_summary', summary)
        else:
            emit('error', {'message': 'Emotional AI not available'})
            
    except Exception as e:
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    print("🚀 Initializing JARVIS with Emotional Intelligence...")
    
    if initialize_jarvis():
        print("✅ Starting web server...")
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        print("❌ Failed to initialize JARVIS")
        if 'error' in jarvis_components:
            print(f"Error: {jarvis_components['error']}")