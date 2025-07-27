#!/usr/bin/env python3
"""
JARVIS Voice Assistant Web Application
แอปพลิเคชันเว็บสำหรับผู้ช่วยเสียง JARVIS
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
    """Initialize JARVIS components"""
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
        try:
            from ai.advanced_conversation_engine import AdvancedConversationEngine
            from ai.self_improvement_system import SelfImprovementSystem
            from voice.advanced_command_system import AdvancedCommandSystem
            ADVANCED_FEATURES_AVAILABLE = True
        except ImportError as e:
            print(f"Advanced features not available: {e}")
            ADVANCED_FEATURES_AVAILABLE = False
        
        config = {
            "command_parser": {"enabled": True, "confidence_threshold": 0.6},
            "thai_language": {"enabled": True},
            "conversation_memory": {
                "max_turns_per_session": 50,
                "context_window_size": 10,
                "memory_dir": "data/web_conversation_memory"
            }
        }
        
        jarvis_components['command_parser'] = VoiceCommandParser(config)
        if ThaiLanguageProcessor is not None:
            jarvis_components['thai_processor'] = ThaiLanguageProcessor(config)
        else:
            jarvis_components['thai_processor'] = None
        jarvis_components['conversation_memory'] = ConversationMemorySystem(config)
        jarvis_components['news_system'] = NewsSystem(config)
        jarvis_components['ai_engine'] = AIEngine(config)
        
        # Initialize advanced features if available
        if ADVANCED_FEATURES_AVAILABLE:
            jarvis_components['advanced_conversation'] = AdvancedConversationEngine(config)
            jarvis_components['self_improvement'] = SelfImprovementSystem(config)
            jarvis_components['advanced_commands'] = AdvancedCommandSystem(config)
            print("✨ Advanced AI features initialized!")
        
        jarvis_components['initialized'] = True
        
        print("✅ JARVIS components initialized successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Failed to initialize JARVIS: {e}")
        jarvis_components['error'] = str(e)
        return False

def process_user_message(text: str, session_id: str):
    """Process user message and return JARVIS response"""
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
        
        # Detect language
        language = "th" if any(ord(c) >= 0x0E00 and ord(c) <= 0x0E7F for c in text) else "en"
        
        # Process Thai if needed
        thai_context = {}
        if language == "th":
            thai_result = thai_processor.process_thai_text(text)
            thai_context = thai_processor.enhance_for_ai_processing(text)
        
        # Parse command
        parsed = command_parser.parse_command(text, language)
        
        # Start session if not exists
        if f"session_{session_id}" not in jarvis_components:
            memory_session_id = conversation_memory.start_session(f"web_user_{session_id}", language)
            jarvis_components[f"session_{session_id}"] = memory_session_id
        
        # Get conversation context
        context = conversation_memory.get_conversation_context(parsed.cleaned_text, max_turns=3)
        
        # Generate response using advanced systems if available
        if jarvis_components.get('advanced_conversation'):
            # Use advanced conversation engine
            context_obj = jarvis_components['advanced_conversation'].start_conversation(
                user_id=session_id,
                session_id=session_id,
                user_preferences={
                    "language": language,
                    "interests": parsed.entities.get("topics", [])
                }
            )
            
            smart_response = jarvis_components['advanced_conversation'].process_message(
                session_id=session_id,
                message=text,
                metadata={"analysis": {"intent": parsed.intent, "entities": parsed.entities}}
            )
            
            response = smart_response.text
            confidence = smart_response.confidence
            
            # Learn from conversation if self-improvement system is available
            if jarvis_components.get('self_improvement'):
                conversation_data = {
                    "user_id": session_id,
                    "session_id": session_id,
                    "user_message": text,
                    "assistant_response": response,
                    "language": language,
                    "analysis": {
                        "intent": parsed.intent,
                        "entities": parsed.entities,
                        "confidence": confidence
                    },
                    "response_confidence": confidence
                }
                jarvis_components['self_improvement'].learn_from_conversation(conversation_data)
            
        else:
            # Fallback to original response generation
            response = generate_jarvis_response(parsed, thai_context, context)
            confidence = parsed.confidence
        
        # Add to conversation memory
        turn_id = conversation_memory.add_conversation_turn(
            user_input=text,
            user_language=language,
            processed_input=parsed.cleaned_text,
            intent=parsed.intent,
            entities=parsed.entities,
            assistant_response=response,
            response_language=language,
            confidence=confidence
        )
        
        return {
            "response": response,
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

มีอะไรให้ช่วยไหมครับ? 😊"""
        else:
            return """Hello! I'm JARVIS, your intelligent assistant 🤖

🌟 My capabilities:
• Answer questions about technology and AI
• Explain complex concepts in simple terms
• Converse in both Thai and English
• Remember our conversation context

How can I help you today? 😊"""
            
    elif intent == "information_request":
        if "ปัญญาประดิษฐ์" in text or "artificial intelligence" in text.lower():
            if language == "th":
                return """ปัญญาประดิษฐ์ (Artificial Intelligence) คือเทคโนโลยีที่ทำให้คอมพิวเตอร์สามารถเรียนรู้ คิด และตัดสินใจได้เหมือนมนุษย์ครับ 🧠

📚 ความสามารถหลักของ AI:
🔹 Machine Learning - การเรียนรู้จากข้อมูล
🔹 Natural Language Processing - การประมวลผลภาษาธรรมชาติ
🔹 Computer Vision - การมองเห็นและเข้าใจภาพ
🔹 Robotics - การควบคุมหุ่นยนต์

🚀 การใช้งานในปัจจุบัน:
• ผู้ช่วยเสียงอัจฉริยะ (เช่น ผมเอง!)
• รถยนต์ขับขี่อัตโนมัติ
• ระบบแนะนำใน Netflix, YouTube
• การแปลภาษาอัตโนมัติ
• การวินิจฉัยทางการแพทย์

คุณอยากทราบเพิ่มเติมเรื่องไหนเป็นพิเศษไหมครับ? 🤔"""
            else:
                return """Artificial Intelligence (AI) is technology that enables computers to learn, think, and make decisions like humans 🧠

📚 Core AI Capabilities:
🔹 Machine Learning - Learning from data
🔹 Natural Language Processing - Understanding human language
🔹 Computer Vision - Seeing and understanding images
🔹 Robotics - Controlling robotic systems

🚀 Current Applications:
• Intelligent voice assistants (like me!)
• Autonomous vehicles
• Recommendation systems (Netflix, YouTube)
• Automatic language translation
• Medical diagnosis assistance

What aspect would you like to know more about? 🤔"""
        
        elif "machine learning" in text.lower() or "แมชชีนเลิร์นนิง" in text:
            if language == "th":
                return """Machine Learning คือสาขาหนึ่งของ AI ที่เน้นการสร้างระบบที่สามารถเรียนรู้และปรับปรุงประสิทธิภาพจากข้อมูลโดยอัตโนมัติครับ 📊

🔬 ประเภทหลักของ Machine Learning:
1️⃣ Supervised Learning - เรียนรู้จากข้อมูลที่มีคำตอบ
2️⃣ Unsupervised Learning - หาเเบบแผนในข้อมูลเอง
3️⃣ Reinforcement Learning - เรียนรู้จากการลองผิดลองถูก

💡 ตัวอย่างการใช้งาน:
• การจดจำใบหน้า
• การแนะนำสินค้า
• การตรวจจับ spam email
• การแปลภาษา
• การวิเคราะห์ความรู้สึกจากข้อความ

สนใจเรียนรู้เพิ่มเติมเรื่องไหนไหมครับ? 🎯"""
            else:
                return """Machine Learning is a subset of AI focused on creating systems that can automatically learn and improve from data without being explicitly programmed 📊

🔬 Main Types of Machine Learning:
1️⃣ Supervised Learning - Learning from labeled data
2️⃣ Unsupervised Learning - Finding patterns in unlabeled data  
3️⃣ Reinforcement Learning - Learning through trial and error

💡 Examples of Applications:
• Facial recognition
• Product recommendations
• Spam email detection
• Language translation
• Sentiment analysis

What aspect would you like to explore further? 🎯"""
        
        else:
            topic = parsed.entities.get("topics", [])
            if topic:
                if language == "th":
                    return f"เรื่อง {topic[0]} นั้นน่าสนใจมากครับ! ให้ผมค้นหาข้อมูลเพิ่มเติมให้ได้ไหม? หรือมีประเด็นเฉพาะที่อยากรู้? 🔍"
                else:
                    return f"That's an interesting topic about {topic[0]}! Would you like me to search for more specific information? 🔍"
            else:
                if language == "th":
                    return "ผมพร้อมตอบคำถามครับ! คุณต้องการทราบเรื่องอะไรเป็นพิเศษ? 🤔"
                else:
                    return "I'm ready to answer your questions! What would you specifically like to know? 🤔"
                    
    elif intent == "how_to_request":
        if language == "th":
            return """ผมยินดีช่วยสอนครับ! 📚

🔧 วิธีใช้งาน JARVIS:
• พิมพ์คำสั่งหรือคำถามที่ต้องการ
• ใช้ภาษาไทยหรือภาษาอังกฤษก็ได้
• ผมจะจำบริบทการสนทนาไว้ครับ

💡 ตัวอย่างคำสั่งที่ใช้ได้:
🗣️ "อธิบายเรื่อง machine learning"
🗣️ "ช่วยหาข้อมูลเกี่ยวกับ blockchain"
🗣️ "วิธีเรียนรู้ programming"
🗣️ "AI ทำงานอย่างไร"

มีอะไรให้ช่วยอีกไหมครับ? 😊"""
        else:
            return """I'm happy to help teach you! 📚

🔧 How to use JARVIS:
• Type commands or questions you want
• Use Thai or English language
• I'll remember conversation context

💡 Example commands you can try:
🗣️ "Explain machine learning"
🗣️ "Help me understand blockchain"
🗣️ "How to learn programming"
🗣️ "How does AI work"

Anything else I can help with? 😊"""
            
    elif intent == "action_request":
        actions = parsed.entities.get("actions", [])
        if actions:
            action = actions[0]
            if language == "th":
                return f"เข้าใจแล้วครับ! ผมจะ{action}ให้คุณ ✅\n\nผมสามารถช่วยอธิบาย วิเคราะห์ และตอบคำถามได้ครับ มีอะไรเฉพาะที่อยากให้ช่วยไหม? 🚀"
            else:
                return f"Understood! I'll {action} that for you ✅\n\nI can help explain, analyze, and answer questions. Is there something specific you need help with? 🚀"
        else:
            if language == "th":
                return f"เข้าใจแล้วครับ! เกี่ยวกับ '{text}' ผมสามารถช่วย:\n\n🔍 วิเคราะห์และอธิบายรายละเอียด\n💡 ให้คำแนะนำที่เป็นประโยชน์\n📚 หาข้อมูลเพิ่มเติม\n\nต้องการให้ผมช่วยอะไรเฉพาะเจาะจงไหมครับ? 🚀"
            else:
                return f"I understand! Regarding '{text}', I can help:\n\n🔍 Analyze and explain details\n💡 Provide useful recommendations\n📚 Find additional information\n\nWhat specific help do you need? 🚀"
                
    elif intent == "system_control":
        if language == "th":
            return """เข้าใจคำสั่งระบบแล้วครับ! ⚙️

📊 สถานะระบบปัจจุบัน:
✅ ระบบประมวลผลภาษาไทย: ทำงาน
✅ ระบบจำการสนทนา: ทำงาน  
✅ ระบบวิเคราะห์เจตนา: ทำงาน
✅ การเชื่อมต่อ AI: พร้อม

🔧 ฟีเจอร์ที่พร้อมใช้งาน:
• การตอบคำถามอัจฉริยะ
• การวิเคราะห์ภาษาไทย
• การจำบริบทการสนทนา
• การประมวลผลคำสั่งธรรมชาติ

มีอะไรให้ปรับแต่งไหมครับ? 🛠️"""
        else:
            return """System command understood! ⚙️

📊 Current System Status:
✅ Thai Language Processing: Active
✅ Conversation Memory: Active
✅ Intent Analysis: Active
✅ AI Connection: Ready

🔧 Available Features:
• Intelligent Q&A system
• Thai language analysis
• Conversation context memory
• Natural command processing

Need any adjustments? 🛠️"""
            
    elif intent == "conversation":
        if language == "th":
            return "ผมพร้อมคุยกับคุณเสมอครับ! 😊 มีเรื่องอะไรที่อยากพูดคุยไหม? หรือมีคำถามเกี่ยวกับเทคโนโลยีที่สนใจ?"
        else:
            return "I'm always ready to chat with you! 😊 Is there anything you'd like to talk about? Any technology questions you're curious about?"
            
    else:
        if language == "th":
            return f"""ผมได้รับข้อความ: "{text}" แล้วครับ 

🎯 ที่ผมเข้าใจ:
• เจตนา: {intent}
• ความมั่นใจ: {parsed.confidence:.0%}

ถ้าผมเข้าใจผิด ลองอธิบายให้ชัดเจนขึ้นได้ไหมครับ? หรือมีอะไรให้ช่วยเพิ่มเติม? 🤖"""
        else:
            return f"""I received your message: "{text}"

🎯 What I understood:
• Intent: {intent}
• Confidence: {parsed.confidence:.0%}

If I misunderstood, could you clarify? Or how else can I help you? 🤖"""

def handle_news_request(text, language):
    """Handle news requests"""
    try:
        if 'news_system' in jarvis_components:
            news = jarvis_components['news_system'].get_latest_news(limit=3)
            if news:
                if language == "th":
                    response = "📰 ข่าวล่าสุด:\n\n"
                    for i, item in enumerate(news[:3], 1):
                        response += f"{i}. {item.get('title', 'ไม่มีหัวข้อ')}\n"
                        if item.get('summary'):
                            response += f"   {item['summary'][:100]}...\n\n"
                else:
                    response = "📰 Latest news:\n\n"
                    for i, item in enumerate(news[:3], 1):
                        response += f"{i}. {item.get('title', 'No title')}\n"
                        if item.get('summary'):
                            response += f"   {item['summary'][:100]}...\n\n"
                return response
    except Exception as e:
        print(f"News system error: {e}")
    
    if language == "th":
        return "ขออภัยครับ ระบบข่าวยังไม่พร้อมใช้งานในขณะนี้ 📰\nแต่ผมสามารถช่วยอธิบายหรือวิเคราะห์เรื่องอื่นๆ ได้ครับ!"
    else:
        return "Sorry, the news system is not available right now 📰\nBut I can help explain or analyze other topics!"

def handle_knowledge_query(text, language):
    """Handle knowledge-based questions"""
    try:
        if 'rag_system' in jarvis_components:
            result = jarvis_components['rag_system'].query(text)
            if result and result.strip():
                return result
    except Exception as e:
        print(f"RAG system error: {e}")
    
    # Enhanced intelligent responses
    thai_keywords = ["คือ", "ทำไม", "อะไร", "เป็นยังไง", "อย่างไร"]
    eng_keywords = ["what", "why", "how", "explain", "tell me"]
    
    is_question = any(kw in text.lower() for kw in thai_keywords + eng_keywords)
    
    if is_question:
        if language == "th":
            return f"เรื่อง '{text}' เป็นคำถามที่น่าสนใจครับ! 🤔\n\nแม้ว่าผมจะยังไม่มีข้อมูลละเอียดในขณะนี้ แต่ผมแนะนำให้:\n\n💡 ลองค้นหาข้อมูลเพิ่มเติมจากแหล่งที่เชื่อถือได้\n📚 อ่านเอกสารหรือบทความที่เกี่ยวข้อง\n🎯 ถามคำถามที่เฉพาะเจาะจงมากขึ้น\n\nมีคำถามอื่นที่ผมช่วยได้ไหมครับ?"
        else:
            return f"'{text}' is an interesting question! 🤔\n\nWhile I don't have detailed information right now, I suggest:\n\n💡 Search for more information from reliable sources\n📚 Read relevant documents or articles\n🎯 Ask more specific questions\n\nIs there anything else I can help with?"
    
    return None

# Global client tracking
active_clients = {}

# WebSocket event handlers for real-time voice communication
@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    client_id = request.sid
    active_clients[client_id] = {
        'connected_at': time.time(),
        'last_activity': time.time(),
        'session_data': {}
    }
    print(f"Client connected: {client_id}")
    emit('status', {'message': 'Connected to JARVIS', 'type': 'info', 'timestamp': time.time()})
    
    # Send real-time system status
    emit('system_status', {
        'jarvis_ready': jarvis_components.get('initialized', False),
        'active_clients': len(active_clients),
        'features': {
            'voice_recognition': True,
            'text_to_speech': True,
            'thai_language': jarvis_components.get('thai_processor') is not None,
            'conversation_memory': True
        }
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    client_id = request.sid
    if client_id in active_clients:
        del active_clients[client_id]
    print(f"Client disconnected: {client_id}")

@socketio.on('heartbeat')
def handle_heartbeat(data):
    """Handle client heartbeat for real-time status"""
    client_id = request.sid
    if client_id in active_clients:
        active_clients[client_id]['last_activity'] = time.time()
        emit('heartbeat_ack', {'timestamp': time.time(), 'status': 'alive'})

@socketio.on('voice_data')
def handle_voice_data(data):
    """Handle real-time voice data streaming with optimization"""
    try:
        client_id = request.sid
        if client_id in active_clients:
            active_clients[client_id]['last_activity'] = time.time()
        
        # Enhanced voice data processing with better validation
        voice_chunk = data.get('chunk', '')
        sample_rate = data.get('sample_rate', 16000)
        chunk_index = data.get('chunk_index', 0)
        chunk_timestamp = data.get('timestamp', time.time())
        
        # Validate audio data
        if not voice_chunk or len(voice_chunk) < 10:
            emit('voice_processing', {'status': 'invalid', 'chunk_index': chunk_index})
            return
        
        # Calculate latency
        current_time = time.time()
        latency = current_time - chunk_timestamp if chunk_timestamp else 0
        
        # Quality assessment based on chunk size and latency
        quality = 'excellent' if len(voice_chunk) > 500 and latency < 0.1 else \
                 'good' if len(voice_chunk) > 100 and latency < 0.3 else \
                 'poor' if latency > 0.5 else 'low'
        
        # Real-time voice processing feedback with enhanced metrics
        emit('voice_processing', {
            'status': 'processing',
            'chunk_index': chunk_index,
            'timestamp': current_time,
            'quality': quality,
            'latency': round(latency * 1000, 2),  # Convert to ms
            'chunk_size_kb': round(len(voice_chunk) / 1024, 2)
        })
        
        # Process through audio stream manager if available
        if 'audio_stream_manager' in jarvis_components:
            # Convert to AudioChunk and process
            import numpy as np
            from services.jarvis_audio.real_time_audio import AudioChunk
            
            # Convert audio data (assuming base64 encoded)
            try:
                import base64
                audio_data = base64.b64decode(voice_chunk)
                audio_array = np.frombuffer(audio_data, dtype=np.float32)
                
                audio_chunk = AudioChunk(
                    data=audio_array,
                    timestamp=chunk_timestamp,
                    sample_rate=sample_rate,
                    sequence_id=chunk_index
                )
                
                jarvis_components['audio_stream_manager'].stt.process_audio_chunk(audio_chunk)
                
            except Exception as audio_error:
                logger.error(f"Audio processing error: {audio_error}")
        
        # Acknowledge receipt with enhanced processing stats
        emit('voice_ack', {
            'status': 'received',
            'timestamp': current_time,
            'chunk_size': len(voice_chunk),
            'sample_rate': sample_rate,
            'quality': quality,
            'processing_latency': round(latency * 1000, 2)
        })
        
    except Exception as e:
        logger.error(f"Voice data handling error: {e}")
        emit('error', {'message': f'Voice data error: {e}', 'timestamp': time.time()})

@socketio.on('tts_request')
def handle_tts_request(data):
    """Handle text-to-speech requests with real-time feedback"""
    try:
        text = data.get('text', '')
        language = data.get('language', 'en')
        voice_settings = data.get('voice_settings', {})
        
        if text:
            # Emit processing status
            emit('tts_status', {'status': 'generating', 'text_length': len(text)})
            
            # Enhanced TTS response with audio analysis
            emit('tts_response', {
                'text': text,
                'language': language,
                'status': 'ready',
                'estimated_duration': len(text) * 0.1,  # Rough estimate
                'voice_settings': voice_settings,
                'timestamp': time.time()
            })
            
            # Simulate processing time for real-time feedback
            socketio.sleep(0.1)
            emit('tts_status', {'status': 'complete'})
    except Exception as e:
        emit('error', {'message': f'TTS error: {e}', 'timestamp': time.time()})

@socketio.on('voice_command')
def handle_voice_command(data):
    """Handle voice commands with optimized real-time processing"""
    try:
        command = data.get('command', '')
        session_id = data.get('session_id', 'default')
        confidence = data.get('confidence', 0.0)
        processing_start = time.time()
        
        if not command or confidence < 0.3:  # Filter low confidence commands
            emit('command_response', {
                'error': 'Command confidence too low or empty',
                'status': 'rejected',
                'confidence': confidence
            })
            return
        
        # Emit processing started with enhanced info
        emit('command_processing', {
            'status': 'started',
            'command': command[:50] + '...' if len(command) > 50 else command,  # Truncate for efficiency
            'confidence': confidence,
            'timestamp': processing_start,
            'estimated_duration': min(max(len(command) * 0.1, 0.5), 3.0)  # Estimate based on length
        })
        
        # Try advanced voice command first for better performance
        advanced_handled = False
        if jarvis_components.get('advanced_commands'):
            try:
                advanced_commands = jarvis_components['advanced_commands']
                parsed_command, parameters = advanced_commands.parse_voice_input(command)
                
                if parsed_command:
                    # Execute advanced command asynchronously
                    import asyncio
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    
                    try:
                        execution = loop.run_until_complete(
                            advanced_commands.execute_command(parsed_command, parameters)
                        )
                        
                        result = {
                            'response': f"✅ Executed: {parsed_command.name}",
                            'metadata': {
                                'command_type': 'advanced',
                                'execution_id': execution.execution_id,
                                'status': execution.status,
                                'confidence': confidence
                            },
                            'status': 'success'
                        }
                        advanced_handled = True
                        
                    finally:
                        loop.close()
                        
            except Exception as advanced_error:
                logger.error(f"Advanced command error: {advanced_error}")
        
        # Process with standard system if not handled by advanced commands
        if not advanced_handled:
            result = process_user_message(command, session_id)
        
        # Calculate actual processing time
        processing_time = time.time() - processing_start
        
        # Enhanced response with comprehensive metadata
        result.update({
            'processing_time': round(processing_time, 3),
            'client_id': request.sid,
            'server_timestamp': time.time(),
            'performance': {
                'processing_duration': processing_time,
                'confidence': confidence,
                'command_length': len(command),
                'response_length': len(result.get('response', ''))
            }
        })
        
        emit('command_response', result)
        
        # Update client session data with performance tracking
        client_id = request.sid
        if client_id in active_clients:
            if 'commands' not in active_clients[client_id]['session_data']:
                active_clients[client_id]['session_data']['commands'] = []
            
            # Keep only last 10 commands for memory efficiency
            commands_list = active_clients[client_id]['session_data']['commands']
            if len(commands_list) >= 10:
                commands_list.pop(0)
            
            commands_list.append({
                'command': command[:100],  # Truncate for storage efficiency
                'timestamp': processing_start,
                'confidence': confidence,
                'processing_time': processing_time,
                'success': result.get('status') == 'success'
            })
                
    except Exception as e:
        logger.error(f"Voice command processing error: {e}")
        emit('error', {
            'message': f'Command processing error: {str(e)[:100]}...',
            'timestamp': time.time(),
            'error_type': 'command_processing'
        })

@socketio.on('request_stats')
def handle_stats_request():
    """Send real-time system statistics"""
    try:
        stats = {
            'timestamp': time.time(),
            'active_clients': len(active_clients),
            'jarvis_status': 'ready' if jarvis_components.get('initialized') else 'initializing',
            'system_load': {
                'memory_usage': '~85%',  # Placeholder - could be enhanced with psutil
                'cpu_usage': '~45%',
                'response_time': '~0.2s'
            },
            'features_status': {
                'ai_engine': 'ai_engine' in jarvis_components,
                'thai_processor': jarvis_components.get('thai_processor') is not None,
                'conversation_memory': 'conversation_memory' in jarvis_components,
                'news_system': 'news_system' in jarvis_components
            }
        }
        emit('system_stats', stats)
    except Exception as e:
        emit('error', {'message': f'Stats error: {e}', 'timestamp': time.time()})

@socketio.on('request_conversation_context')
def handle_context_request(data):
    """Send real-time conversation context"""
    try:
        session_id = data.get('session_id', 'default')
        client_id = request.sid
        
        # Get conversation context
        context = {}
        if jarvis_components.get('conversation_memory'):
            memory_session_id = jarvis_components.get(f"session_{session_id}")
            if memory_session_id:
                # Get recent conversation turns
                recent_context = jarvis_components['conversation_memory'].get_conversation_context("", max_turns=5)
                context['recent_turns'] = len(recent_context)
                context['session_active'] = True
            else:
                context['session_active'] = False
                
        # Add client-specific context
        if client_id in active_clients:
            client_data = active_clients[client_id]
            context['connected_duration'] = time.time() - client_data['connected_at']
            context['commands_issued'] = len(client_data['session_data'].get('commands', []))
            context['last_activity'] = client_data['last_activity']
            
        emit('conversation_context', context)
    except Exception as e:
        emit('error', {'message': f'Context error: {e}', 'timestamp': time.time()})

@socketio.on('typing_indicator')
def handle_typing_indicator(data):
    """Handle real-time typing indicators"""
    try:
        is_typing = data.get('is_typing', False)
        client_id = request.sid
        
        # Broadcast typing status to other clients (if needed for multi-user)
        # For single-user, we could use this for AI "thinking" indicators
        
        if is_typing:
            # AI could start preparing response context
            emit('ai_thinking', {'status': 'preparing', 'timestamp': time.time()})
        else:
            emit('ai_thinking', {'status': 'ready', 'timestamp': time.time()})
            
    except Exception as e:
        emit('error', {'message': f'Typing indicator error: {e}', 'timestamp': time.time()})

@socketio.on('voice_command_advanced')
def handle_advanced_voice_command(data):
    """Handle advanced voice commands with macro support"""
    try:
        command_text = data.get('command', '')
        session_id = data.get('session_id', 'default')
        
        if not command_text or not jarvis_components.get('advanced_commands'):
            emit('command_response', {'error': 'Advanced commands not available'})
            return
        
        advanced_commands = jarvis_components['advanced_commands']
        
        # Parse voice command
        command, parameters = advanced_commands.parse_voice_input(command_text)
        
        if command:
            # Execute command asynchronously
            import asyncio
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                execution = loop.run_until_complete(
                    advanced_commands.execute_command(command, parameters)
                )
                
                emit('advanced_command_response', {
                    'command_name': command.name,
                    'execution_id': execution.execution_id,
                    'status': execution.status,
                    'results': execution.results,
                    'timestamp': time.time()
                })
                
            finally:
                loop.close()
        else:
            emit('advanced_command_response', {
                'error': f'Command not recognized: {command_text}',
                'timestamp': time.time()
            })
            
    except Exception as e:
        emit('error', {'message': f'Advanced command error: {e}', 'timestamp': time.time()})

@app.route('/')
def index():
    """Main page"""
    # Generate session ID
    if 'session_id' not in session:
        session['session_id'] = secrets.token_hex(8)
    
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/message', methods=['POST'])
def handle_message():
    """Handle user message"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400
    
    message = data['message'].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    
    session_id = session.get('session_id', 'default')
    result = process_user_message(message, session_id)
    
    return jsonify(result)

@app.route('/api/status')
def get_status():
    """Get JARVIS status"""
    if jarvis_components.get('initialized'):
        advanced_available = jarvis_components.get('advanced_conversation') is not None
        return jsonify({
            "status": "ready",
            "message": "JARVIS พร้อมใช้งานแล้ว ✅",
            "advanced_features": advanced_available,
            "features": {
                "advanced_conversation": jarvis_components.get('advanced_conversation') is not None,
                "self_improvement": jarvis_components.get('self_improvement') is not None,
                "advanced_commands": jarvis_components.get('advanced_commands') is not None,
                "thai_processing": jarvis_components.get('thai_processor') is not None
            }
        })
    elif 'error' in jarvis_components:
        return jsonify({
            "status": "error", 
            "message": f"เกิดข้อผิดพลาด: {jarvis_components['error']}"
        })
    else:
        return jsonify({
            "status": "initializing",
            "message": "กำลังเริ่มต้นระบบ..."
        })

@app.route('/api/commands', methods=['GET'])
def get_commands():
    """Get available voice commands"""
    if not jarvis_components.get('advanced_commands'):
        return jsonify({"error": "Advanced commands not available"}), 400
    
    advanced_commands = jarvis_components['advanced_commands']
    commands_info = []
    
    for command in advanced_commands.commands.values():
        commands_info.append({
            "name": command.name,
            "trigger_phrases": command.trigger_phrases,
            "description": command.description,
            "command_type": command.command_type.value,
            "usage_count": command.usage_count,
            "last_used": command.last_used.isoformat() if command.last_used else None
        })
    
    return jsonify({
        "commands": commands_info,
        "total": len(commands_info)
    })

@app.route('/api/commands/stats', methods=['GET'])
def get_command_stats():
    """Get command usage statistics"""
    if not jarvis_components.get('advanced_commands'):
        return jsonify({"error": "Advanced commands not available"}), 400
    
    advanced_commands = jarvis_components['advanced_commands']
    stats = advanced_commands.get_command_usage_stats()
    
    return jsonify(stats)

@app.route('/api/improvement/report', methods=['GET'])
def get_improvement_report():
    """Get self-improvement report"""
    if not jarvis_components.get('self_improvement'):
        return jsonify({"error": "Self-improvement system not available"}), 400
    
    self_improvement = jarvis_components['self_improvement']
    report = self_improvement.generate_improvement_report()
    
    return jsonify(report)

# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="th">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>JARVIS - ผู้ช่วยอัจฉริยะ</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/socket.io/4.7.5/socket.io.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
        
        body {
            font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
            background-size: 400% 400%;
            animation: gradientShift 15s ease infinite;
            min-height: 100vh;
            display: flex;
            justify-content: center;
            align-items: center;
            position: relative;
            overflow-x: hidden;
        }
        
        /* Animated background particles */
        /* Animated background particles */
        body::before {
            content: '';
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><circle cx="20" cy="20" r="2" fill="rgba(255,255,255,0.3)"/><circle cx="80" cy="40" r="1" fill="rgba(255,255,255,0.2)"/><circle cx="40" cy="80" r="1.5" fill="rgba(255,255,255,0.4)"/></svg>') repeat;
            animation: particleFloat 20s ease-in-out infinite;
            pointer-events: none;
            z-index: 0;
        }
        
        @keyframes particleFloat {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            50% { transform: translateY(-20px) rotate(180deg); }
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        .container {
            width: 95%;
            max-width: 1200px;
            height: 90vh;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(20px);
            border-radius: 30px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            box-shadow: 0 25px 45px rgba(0, 0, 0, 0.1);
            padding: 30px;
            display: flex;
            flex-direction: column;
            position: relative;
            z-index: 1;
            animation: containerGlow 8s ease-in-out infinite alternate;
        }
        
        @keyframes containerGlow {
            from { box-shadow: 0 25px 45px rgba(0, 0, 0, 0.1), 0 0 30px rgba(103, 126, 234, 0.2); }
            to { box-shadow: 0 25px 45px rgba(0, 0, 0, 0.1), 0 0 40px rgba(240, 147, 251, 0.3); }
        }
        
        .header {
            text-align: center;
            margin-bottom: 30px;
            background: linear-gradient(135deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.05) 100%);
            backdrop-filter: blur(10px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255,255,255,0.1);
            position: relative;
            overflow: hidden;
        }
        
        .header::before {
            content: '✨';
            position: absolute;
            top: 10px;
            right: 20px;
            font-size: 24px;
            animation: sparkle 2s ease-in-out infinite;
        }
        
        @keyframes sparkle {
            0%, 100% { transform: rotate(0deg) scale(1); }
            25% { transform: rotate(90deg) scale(1.2); }
            50% { transform: rotate(180deg) scale(1); }
            75% { transform: rotate(270deg) scale(1.2); }
        }
        
        .header h1 {
            font-family: 'Poppins', sans-serif;
            color: white;
            font-size: 3rem;
            font-weight: 700;
            margin: 0;
            text-shadow: 0 4px 8px rgba(0,0,0,0.3);
            background: linear-gradient(135deg, #ffffff, #f0f0f0, #ffffff);
            background-size: 200% 200%;
            -webkit-background-clip: text;
            background-clip: text;
            -webkit-text-fill-color: transparent;
            animation: textShimmer 3s ease-in-out infinite, bounce 2s ease-in-out infinite;
        }
        
        @keyframes textShimmer {
            0%, 100% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-5px); }
        }
        
        .header p {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.2rem;
            margin-top: 10px;
            font-weight: 400;
            text-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }
        
        .status-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: linear-gradient(90deg, rgba(67, 206, 162, 0.1), rgba(24, 90, 157, 0.1));
            border-radius: 15px;
            padding: 15px 20px;
            margin-bottom: 20px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
        }
        
        .status-indicator {
            display: flex;
            align-items: center;
            gap: 10px;
            color: white;
            font-weight: 500;
        }
        
        .status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #4ade80;
            animation: pulse 2s ease-in-out infinite;
            box-shadow: 0 0 20px rgba(74, 222, 128, 0.6);
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.2); opacity: 0.8; }
        }
        
        .chat-container {
            flex: 1;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 20px;
            padding: 25px;
            display: flex;
            flex-direction: column;
            gap: 20px;
            overflow-y: auto;
            backdrop-filter: blur(15px);
            border: 1px solid rgba(255, 255, 255, 0.1);
            position: relative;
        }
        
        .chat-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 30px;
            background: linear-gradient(to bottom, rgba(255,255,255,0.1), transparent);
            border-radius: 20px 20px 0 0;
            pointer-events: none;
        }
        
        .message {
            max-width: 80%;
            padding: 18px 24px;
            border-radius: 25px;
            backdrop-filter: blur(10px);
            position: relative;
            animation: messageSlideIn 0.5s ease-out;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        }
        
        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateY(20px) scale(0.95);
            }
            to {
                opacity: 1;
                transform: translateY(0px) scale(1);
            }
        }
        
        .message.user {
            align-self: flex-end;
            background: linear-gradient(135deg, rgba(99, 102, 241, 0.8), rgba(168, 85, 247, 0.8));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
        }
        
        .message.user::after {
            content: '';
            position: absolute;
            bottom: -8px;
            right: 20px;
            width: 0;
            height: 0;
            border-left: 15px solid transparent;
            border-right: 0;
            border-top: 15px solid rgba(99, 102, 241, 0.8);
            filter: blur(0.5px);
        }
        
        .message.assistant {
            align-self: flex-start;
            background: linear-gradient(135deg, rgba(16, 185, 129, 0.8), rgba(5, 150, 105, 0.8));
            color: white;
            border: 1px solid rgba(255, 255, 255, 0.2);
            position: relative;
        }
        
        .message.assistant::after {
            content: '';
            position: absolute;
            bottom: -8px;
            left: 20px;
            width: 0;
            height: 0;
            border-right: 15px solid transparent;
            border-left: 0;
            border-top: 15px solid rgba(16, 185, 129, 0.8);
            filter: blur(0.5px);
        }
        
        .message-avatar {
            width: 45px;
            height: 45px;
            border-radius: 50%;
            margin: 0 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 18px;
            font-weight: bold;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            box-shadow: 0 8px 20px rgba(0, 0, 0, 0.2);
            animation: avatarGlow 3s ease-in-out infinite alternate;
        }
        
        @keyframes avatarGlow {
            from { box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3); }
            to { box-shadow: 0 8px 25px rgba(118, 75, 162, 0.4); }
        }
        
        .input-section {
            background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05));
            backdrop-filter: blur(20px);
            border-radius: 20px;
            padding: 25px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-top: 20px;
        }
        
        .voice-controls {
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .voice-btn {
            padding: 15px 30px;
            border: none;
            border-radius: 50px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
            position: relative;
            overflow: hidden;
        }
        
        .voice-btn::before {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            transition: left 0.5s;
        }
        
        .voice-btn:hover::before {
            left: 100%;
        }
        
        .voice-btn.record {
            background: linear-gradient(135deg, #ef4444, #dc2626);
            color: white;
            box-shadow: 0 8px 25px rgba(239, 68, 68, 0.3);
        }
        
        .voice-btn.record:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(239, 68, 68, 0.4);
        }
        
        .voice-btn.record.recording {
            animation: recordingPulse 1s ease-in-out infinite;
        }
        
        @keyframes recordingPulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        .voice-btn.stop {
            background: linear-gradient(135deg, #6b7280, #4b5563);
            color: white;
            box-shadow: 0 8px 25px rgba(107, 114, 128, 0.3);
        }
        
        .voice-btn.stop:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(107, 114, 128, 0.4);
        }
        
        .voice-visualizer {
            height: 60px;
            background: rgba(255, 255, 255, 0.1);
            border-radius: 15px;
            margin: 20px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
            overflow: hidden;
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.1);
        }
        
        .voice-bars {
            display: flex;
            gap: 4px;
            align-items: center;
            height: 40px;
        }
        
        .voice-bar {
            width: 4px;
            background: linear-gradient(to top, #10b981, #34d399, #6ee7b7);
            border-radius: 2px;
            animation: voiceWave 1.5s ease-in-out infinite;
        }
        
        .voice-bar:nth-child(2) { animation-delay: -0.1s; }
        .voice-bar:nth-child(3) { animation-delay: -0.2s; }
        .voice-bar:nth-child(4) { animation-delay: -0.3s; }
        .voice-bar:nth-child(5) { animation-delay: -0.4s; }
        
        @keyframes voiceWave {
            0%, 100% { height: 10px; opacity: 0.3; }
            50% { height: 35px; opacity: 1; }
        }
        
        .input-container {
            display: flex;
            gap: 15px;
            align-items: center;
        }
        
        .message-input {
            flex: 1;
            padding: 18px 25px;
            border: 2px solid rgba(255, 255, 255, 0.2);
            border-radius: 25px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            color: white;
            font-size: 16px;
            font-family: 'Inter', sans-serif;
            transition: all 0.3s ease;
        }
        
        .message-input::placeholder {
            color: rgba(255, 255, 255, 0.6);
        }
        
        .message-input:focus {
            outline: none;
            border-color: rgba(255, 255, 255, 0.4);
            background: rgba(255, 255, 255, 0.15);
            box-shadow: 0 0 30px rgba(255, 255, 255, 0.1);
        }
        
        .send-btn {
            padding: 18px 25px;
            background: linear-gradient(135deg, #8b5cf6, #7c3aed);
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            backdrop-filter: blur(10px);
            box-shadow: 0 8px 25px rgba(139, 92, 246, 0.3);
            position: relative;
            overflow: hidden;
        }
        
        .send-btn::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            background: rgba(255, 255, 255, 0.2);
            border-radius: 50%;
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .send-btn:hover::before {
            width: 300px;
            height: 300px;
        }
        
        .send-btn:hover {
            transform: translateY(-3px);
            box-shadow: 0 12px 35px rgba(139, 92, 246, 0.4);
        }
        
        .send-btn:active {
            transform: translateY(-1px);
        }
        
        /* Floating elements animation */
        .floating-element {
            position: absolute;
            pointer-events: none;
            animation: float 6s ease-in-out infinite;
        }
        
        .floating-element:nth-child(1) {
            top: 10%;
            left: 10%;
            animation-delay: 0s;
        }
        
        .floating-element:nth-child(2) {
            top: 20%;
            right: 15%;
            animation-delay: -2s;
        }
        
        .floating-element:nth-child(3) {
            bottom: 15%;
            left: 20%;
            animation-delay: -4s;
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-20px) rotate(120deg); }
            66% { transform: translateY(10px) rotate(240deg); }
        }
        
        /* Responsive design */
        @media (max-width: 768px) {
            .container { width: 98%; height: 95vh; padding: 20px; }
            .header h1 { font-size: 2.5rem; }
            .message { max-width: 90%; }
            .voice-controls { flex-direction: column; gap: 10px; }
            .input-container { flex-direction: column; }
            .message-input, .send-btn { width: 100%; }
        }
        
        /* Loading animation */
        .loading-dots {
            display: inline-flex;
            gap: 4px;
        }
        
        .loading-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            background: currentColor;
            animation: loadingBounce 1.4s ease-in-out infinite both;
        }
        
        .loading-dot:nth-child(1) { animation-delay: -0.32s; }
        .loading-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes loadingBounce {
            0%, 80%, 100% { transform: scale(0); opacity: 0.5; }
            40% { transform: scale(1); opacity: 1; }
        }
            height: 100%;
            background-image: radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
                              radial-gradient(circle at 80% 20%, rgba(255,255,255,0.15) 0%, transparent 50%),
                              radial-gradient(circle at 40% 40%, rgba(255,255,255,0.08) 0%, transparent 50%);
            animation: particleFloat 20s ease-in-out infinite;
            pointer-events: none;
            z-index: -1;
        }
        
        @keyframes gradientShift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes particleFloat {
            0%, 100% { transform: translateY(0px) rotate(0deg); }
            33% { transform: translateY(-20px) rotate(5deg); }
            66% { transform: translateY(10px) rotate(-3deg); }
        }
        
        /* Remove duplicate container styles - keep the modern version above */
        
        
        
        
        
        
        
        
        
        
        
        
        .status {
            padding: 15px 25px;
            background: linear-gradient(135deg, rgba(248,249,250,0.8), rgba(233,236,239,0.8));
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(222,226,230,0.5);  
            font-weight: 600;
            color: #6c757d;
            position: relative;
            overflow: hidden;
            transition: all 0.3s ease;
        }
        
        .status::before {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            height: 100%;
            width: 4px;
            background: linear-gradient(180deg, #667eea, #764ba2);
            transition: all 0.3s ease;
        }
        
        .status.ready {
            background: linear-gradient(135deg, rgba(212,237,218,0.9), rgba(195,230,203,0.9));
            color: #155724;
            animation: statusPulse 2s ease-in-out infinite;
        }
        
        .status.ready::before {
            background: linear-gradient(180deg, #28a745, #20c997);
            width: 6px;
        }
        
        .status.error {
            background: linear-gradient(135deg, rgba(248,215,218,0.9), rgba(245,198,203,0.9));
            color: #721c24;
            animation: statusShake 0.5s ease-in-out;
        }
        
        .status.error::before {
            background: linear-gradient(180deg, #dc3545, #c82333);
            width: 6px;
        }
        
        @keyframes statusPulse {
            0%, 100% { transform: scale(1); opacity: 1; }
            50% { transform: scale(1.02); opacity: 0.9; }
        }
        
        @keyframes statusShake {
            0%, 100% { transform: translateX(0); }
            25% { transform: translateX(-5px); }
            75% { transform: translateX(5px); }
        }
        
        .chat-container {
            height: 520px;
            overflow-y: auto;
            padding: 25px;
            background: linear-gradient(145deg, rgba(248,249,250,0.4), rgba(233,236,239,0.4));
            backdrop-filter: blur(5px);
            position: relative;
        }
        
        .chat-container::-webkit-scrollbar {
            width: 6px;
        }
        
        .chat-container::-webkit-scrollbar-track {
            background: rgba(0,0,0,0.05);
            border-radius: 10px;
        }
        
        .chat-container::-webkit-scrollbar-thumb {
            background: linear-gradient(180deg, #667eea, #764ba2);
            border-radius: 10px;
        }
        
        .message {
            margin-bottom: 20px;
            display: flex;
            align-items: flex-start;
            animation: messageSlideIn 0.5s ease-out;
        }
        
        .message.user {
            justify-content: flex-end;
            animation: messageSlideInRight 0.5s ease-out;
        }
        
        .message-content {
            max-width: 75%;
            padding: 16px 20px;
            border-radius: 25px;
            word-wrap: break-word;
            white-space: pre-wrap;
            position: relative;
            font-size: 15px;
            line-height: 1.5;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }
        
        .message-content:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(0,0,0,0.15);
        }
        
        .message.user .message-content {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            margin-left: 25%;
            border-top-right-radius: 8px;
            position: relative;
        }
        
        .message.user .message-content::after {
            content: '';
            position: absolute;
            right: -8px;
            bottom: 8px;
            width: 0;
            height: 0;
            border-left: 8px solid #764ba2;
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
        }
        
        .message.jarvis .message-content {
            background: linear-gradient(135deg, rgba(255,255,255,0.95), rgba(248,249,250,0.95));
            backdrop-filter: blur(10px);
            border: 1px solid rgba(222,226,230,0.5);
            margin-right: 25%;
            color: #2c3e50;
            border-top-left-radius: 8px;
            position: relative;
        }
        
        .message.jarvis .message-content::after {
            content: '';
            position: absolute;
            left: -9px;
            bottom: 8px;
            width: 0;
            height: 0;
            border-right: 8px solid rgba(255,255,255,0.95);
            border-top: 8px solid transparent;
            border-bottom: 8px solid transparent;
        }
        
        @keyframes messageSlideIn {
            from {
                opacity: 0;
                transform: translateX(-20px) translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateX(0) translateY(0);
            }
        }
        
        @keyframes messageSlideInRight {
            from {
                opacity: 0;
                transform: translateX(20px) translateY(10px);
            }
            to {
                opacity: 1;
                transform: translateX(0) translateY(0);
            }
        }
        
        .message-avatar {
            width: 40px;
            height: 40px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
            margin: 0 10px;
        }
        
        .message.user .message-avatar {
            background: #17a2b8;
            order: 2;
        }
        
        .message.jarvis .message-avatar {
            background: #6f42c1;
        }
        
        .message-meta {
            font-size: 0.8em;
            color: #6c757d;
            margin-top: 5px;
            font-style: italic;
        }
        
        .input-container {
            padding: 20px;
            background: white;
            border-top: 1px solid #dee2e6;
        }
        
        .input-group {
            display: flex;
            gap: 10px;
        }
        
        .input-field {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #dee2e6;
            border-radius: 25px;
            font-size: 16px;
            outline: none;
            transition: border-color 0.3s;
        }
        
        .input-field:focus {
            border-color: #007acc;
        }
        
        .send-button {
            padding: 12px 24px;
            background: #007acc;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .send-button:hover:not(:disabled) {
            background: #005fa3;
        }
        
        .send-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .typing {
            display: none;
            font-style: italic;
            color: #6c757d;
            padding: 10px 0;
        }
        
        .welcome {
            text-align: center;
            color: #6c757d;
            font-style: italic;
            margin-bottom: 20px;
        }
        
        @keyframes pulse {
            0% { opacity: 1; }
            50% { opacity: 0.5; }
            100% { opacity: 1; }
        }
        
        .loading {
            animation: pulse 1.5s infinite;
        }
        
        /* Voice Controls Styles */
        .voice-controls {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 15px;
            margin-bottom: 15px;
            padding: 10px;
            background: #f8f9fa;
            border-radius: 10px;
        }
        
        .voice-button {
            padding: 12px 20px;
            background: #28a745;
            color: white;
            border: none;
            border-radius: 25px;
            font-size: 16px;
            cursor: pointer;
            transition: all 0.3s;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .voice-button:hover:not(:disabled) {
            background: #218838;
            transform: translateY(-2px);
        }
        
        .voice-button:disabled {
            background: #6c757d;
            cursor: not-allowed;
        }
        
        .voice-button.recording {
            background: #dc3545;
            animation: pulse 1s infinite;
        }
        
        .voice-button.processing {
            background: #ffc107;
            color: #000;
        }
        
        .voice-status {
            font-size: 14px;
            color: #6c757d;
            font-weight: bold;
            padding: 5px 10px;
            background: white;
            border-radius: 15px;
            border: 1px solid #dee2e6;
        }
        
        .settings-button {
            padding: 10px 15px;
            background: #6c757d;
            color: white;
            border: none;
            border-radius: 50%;
            cursor: pointer;
            transition: background-color 0.3s;
        }
        
        .settings-button:hover {
            background: #5a6268;
        }
        
        .voice-settings {
            background: #e9ecef;
            padding: 15px;
            border-radius: 10px;
            margin-bottom: 15px;
            border: 1px solid #dee2e6;
        }
        
        .setting-group {
            margin-bottom: 10px;
            display: flex;
            align-items: center;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        .setting-group label {
            font-weight: bold;
            min-width: 150px;
            color: #495057;
        }
        
        .setting-group select, .setting-group input[type="range"] {
            flex: 1;
            min-width: 150px;
        }
        
        .setting-group select {
            padding: 5px 10px;
            border: 1px solid #ced4da;
            border-radius: 5px;
            background: white;
        }
        
        .setting-group input[type="range"] {
            margin-right: 10px;
        }
        
        .setting-group span {
            font-weight: bold;
            color: #007acc;
            min-width: 30px;
        }
        
        /* Voice visualization */
        .voice-visualizer {
            height: 60px;
            background: linear-gradient(90deg, #007acc, #005fa3);
            border-radius: 30px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            overflow: hidden;
            position: relative;
        }
        
        .voice-wave {
            width: 4px;
            background: rgba(255, 255, 255, 0.8);
            margin: 0 2px;
            border-radius: 2px;
            transition: height 0.1s ease;
        }
        
        .speaking-indicator {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: white;
            font-weight: bold;
            display: none;
        }
        
        .speaking-indicator.active {
            display: block;
            animation: pulse 1s infinite;
        }
        
        /* Real-time notifications */
        .notification-area {
            position: fixed;
            top: 20px;
            right: 20px;
            z-index: 1000;
            max-width: 300px;
        }
        
        .notification {
            background: white;
            border: 1px solid #dee2e6;
            border-radius: 10px;
            padding: 15px;
            margin-bottom: 10px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            display: flex;
            justify-content: space-between;
            align-items: center;
            animation: slideIn 0.3s ease;
        }
        
        .notification.success {
            border-left: 4px solid #28a745;
            background: #d4edda;
        }
        
        .notification.error {
            border-left: 4px solid #dc3545;
            background: #f8d7da;
        }
        
        .notification.info {
            border-left: 4px solid #17a2b8;
            background: #d1ecf1;
        }
        
        .notification button {
            background: none;
            border: none;
            font-size: 18px;
            cursor: pointer;
            color: #6c757d;
            padding: 0;
            margin-left: 10px;
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(100%);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        /* Processing indicator */
        .processing-indicator {
            background: #f8f9fa;
            border: 1px solid #dee2e6;
            border-radius: 18px;
            padding: 12px 16px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        
        .processing-content {
            display: flex;
            align-items: center;
            gap: 10px;
            color: #6c757d;
            font-style: italic;
        }
        
        .spinner {
            width: 20px;
            height: 20px;
            border: 2px solid #f3f3f3;
            border-top: 2px solid #007acc;
            border-radius: 50%;
            animation: spin 1s linear infinite;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Connection status indicators */
        .connection-status {
            position: fixed;
            bottom: 20px;
            left: 20px;
            padding: 8px 12px;
            border-radius: 15px;
            font-size: 12px;
            font-weight: bold;
            z-index: 999;
        }
        
        .connection-status.good {
            background: #d4edda;
            color: #155724;
            border: 1px solid #c3e6cb;
        }
        
        .connection-status.poor {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
        }
        
        .connection-status.bad {
            background: #f8d7da;
            color: #721c24;
            border: 1px solid #f5c6cb;
        }
        
        /* Enhanced voice status */
        .voice-status.info {
            background: #d1ecf1;
            color: #0c5460;
            border: 1px solid #bee5eb;
        }
        
        .voice-status.processing {
            background: #fff3cd;
            color: #856404;
            border: 1px solid #ffeaa7;
            animation: pulse 1.5s infinite;
        }
        
        .voice-status.generating {
            background: #e2e3e5;
            color: #383d41;
            border: 1px solid #d6d8db;
            animation: pulse 1s infinite;
        }
        
        /* Real-time stats display */
        .stats-overlay {
            position: fixed;
            top: 80px;
            left: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            font-size: 12px;
            color: #6c757d;
            z-index: 998;
            max-width: 250px;
            backdrop-filter: blur(10px);
        }
        
        .stats-overlay h4 {
            margin: 0 0 5px 0;
            color: #495057;
            font-size: 14px;
        }
        
        .stats-line {
            margin: 2px 0;
            display: flex;
            justify-content: space-between;
        }
        
        .stats-line .label {
            font-weight: bold;
        }
        
        .stats-line .value {
            color: #007acc;
        }
        
        /* Enhanced voice visualizer for real-time feedback */
        .voice-visualizer.active {
            box-shadow: 0 0 20px rgba(0, 122, 204, 0.3);
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        .voice-visualizer.excellent {
            background: linear-gradient(90deg, #00ff88, #00cc77);
            box-shadow: 0 0 25px rgba(0, 255, 136, 0.4);
        }
        
        .voice-visualizer.good {
            background: linear-gradient(90deg, #007acc, #005fa3);
            box-shadow: 0 0 20px rgba(0, 122, 204, 0.3);
        }
        
        .voice-visualizer.poor {
            background: linear-gradient(90deg, #ffaa00, #ff8800);
            box-shadow: 0 0 15px rgba(255, 170, 0, 0.3);
        }
        
        .voice-visualizer.low {
            background: linear-gradient(90deg, #ff4444, #cc3333);
            box-shadow: 0 0 10px rgba(255, 68, 68, 0.3);
        }
        
        @keyframes glow {
            from {
                box-shadow: 0 0 20px rgba(0, 122, 204, 0.3);
            }
            to {
                box-shadow: 0 0 30px rgba(0, 122, 204, 0.6);
            }
        }
        
        /* Real-time performance indicators */
        .performance-indicator {
            position: fixed;
            top: 100px;
            right: 20px;
            background: rgba(255, 255, 255, 0.95);
            padding: 10px 15px;
            border-radius: 10px;
            border: 1px solid #dee2e6;
            font-size: 12px;
            color: #6c757d;
            z-index: 999;
            max-width: 200px;
            backdrop-filter: blur(10px);
        }
        
        .performance-indicator h5 {
            margin: 0 0 5px 0;
            color: #495057;
            font-size: 14px;
        }
        
        .perf-metric {
            display: flex;
            justify-content: space-between;
            margin: 2px 0;
        }
        
        .perf-metric .label {
            font-weight: bold;
        }
        
        .perf-metric .value {
            color: #007acc;
        }
        
        .perf-metric.excellent .value { color: #28a745; }
        .perf-metric.good .value { color: #007acc; }
        .perf-metric.poor .value { color: #ffc107; }
        .perf-metric.bad .value { color: #dc3545; }
        
        /* Enhanced voice wave animations */
        .voice-wave.active {
            animation: wave-pulse 0.3s ease-in-out infinite alternate;
        }
        
        @keyframes wave-pulse {
            from { opacity: 0.6; }
            to { opacity: 1.0; }
        }
        
        /* Quality-based wave colors */
        .voice-wave.excellent { background: rgba(0, 255, 136, 0.8); }
        .voice-wave.good { background: rgba(255, 255, 255, 0.8); }
        .voice-wave.poor { background: rgba(255, 170, 0, 0.8); }
        .voice-wave.low { background: rgba(255, 68, 68, 0.8); }
        
        /* AI thinking indicator */
        .ai-thinking-indicator {
            background: #e7f3ff;
            border: 1px solid #b8daff;
            border-radius: 18px;
            padding: 8px 12px;
            margin: 10px 0;
            display: flex;
            align-items: center;
            justify-content: center;
            color: #004085;
            font-style: italic;
            font-size: 14px;
            animation: pulse 1.5s infinite;
        }
    </style>
</head>
<body>
    <!-- Floating elements for decoration -->
    <div class="floating-element" style="top: 10%; left: 8%;">✨</div>
    <div class="floating-element" style="top: 15%; right: 12%;">🌟</div>
    <div class="floating-element" style="bottom: 20%; left: 15%;">💫</div>
    <div class="floating-element" style="bottom: 25%; right: 8%;">⚡</div>
    
    <div class="container">
        <div class="header">
            <h1>JARVIS</h1>
            <p>ผู้ช่วยอัจฉริยะด้วย AI - Thai Language Supported</p>
        </div>
        
        <div class="status-bar">
            <div class="status-indicator">
                <div class="status-dot"></div>
                <span id="statusText">กำลังเริ่มต้นระบบ...</span>
            </div>
            <div class="status-indicator">
                <span>✨ Ready to Help</span>
            </div>
        </div>
        
        <div class="chat-container" id="chatContainer">
            <div class="welcome">
                <h3>🎉 ยินดีต้อนรับสู่ JARVIS!</h3>
                <p>พิมพ์ข้อความหรือใช้เสียงเพื่อเริ่มสนทนา เช่น "สวัสดี" หรือ "ปัญญาประดิษฐ์คืออะไร"</p>
            </div>
            
            <div class="voice-visualizer" id="voiceVisualizer">
                <div class="voice-bars" id="voiceBars">
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                    <div class="voice-bar"></div>
                </div>
                <div class="speaking-indicator" id="speakingIndicator">🎙️ Voice Ready</div>
            </div>
        </div>
        
        <div class="typing" id="typing" style="display: none;">
            <div class="loading-dots">
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
                <div class="loading-dot"></div>
            </div>
            <span>JARVIS กำลังคิด...</span>
        </div>
        
        <!-- Real-time Performance Indicator -->
        <div class="performance-indicator" id="performanceIndicator" style="display: none;">
            <h5>🚀 Performance</h5>
            <div class="perf-metric" id="latencyMetric">
                <span class="label">Latency:</span>
                <span class="value">-- ms</span>
            </div>
            <div class="perf-metric" id="qualityMetric">
                <span class="label">Quality:</span>
                <span class="value">Good</span>
            </div>
            <div class="perf-metric" id="throughputMetric">
                <span class="label">Throughput:</span>
                <span class="value">-- KB/s</span>
            </div>
        </div>
        
        <div class="input-section">
            <div class="voice-controls">
                <button class="voice-btn record" id="voiceButton" onclick="toggleVoiceRecording()" disabled>
                    🎤 เริ่มพูด
                </button>
                <button class="voice-btn stop" id="stopButton" onclick="stopVoiceRecording()" disabled style="display: none;">
                    ⏹️ หยุด
                </button>
                <div class="voice-status" id="voiceStatus">
                    Voice: Ready
                </div>
            </div>
            
            <div class="voice-settings" id="voiceSettings" style="display: none;">
                <div class="setting-group">
                    <label>Voice Recognition Language:</label>
                    <select id="recognitionLanguage">
                        <option value="th-TH">ไทย (Thai)</option>
                        <option value="en-US" selected>English (US)</option>
                        <option value="en-GB">English (UK)</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>TTS Voice:</label>
                    <select id="ttsVoice">
                        <option value="auto">Auto Select</option>
                    </select>
                </div>
                <div class="setting-group">
                    <label>Speech Rate:</label>
                    <input type="range" id="speechRate" min="0.5" max="2" step="0.1" value="1">
                    <span id="rateValue">1.0</span>
                </div>
                <div class="setting-group">
                    <label>Speech Pitch:</label>
                    <input type="range" id="speechPitch" min="0" max="2" step="0.1" value="1">
                    <span id="pitchValue">1.0</span>
                </div>
            </div>
            
            <div class="input-container">
                <input type="text" class="message-input" id="messageInput" 
                       placeholder="พิมพ์ข้อความหรือพูด... ✨"
                       disabled>
                <button class="send-btn" id="sendButton" onclick="sendMessage()" disabled>
                    <span>ส่ง</span>
                </button>
            </div>
        </div>
    </div>

    <script>
        let isInitialized = false;
        let isRecording = false;
        let recognition = null;
        let speechSynthesis = window.speechSynthesis;
        let currentUtterance = null;
        let voices = [];
        let socket = null;
        
        // Voice settings
        let voiceSettings = {
            recognitionLanguage: 'en-US',
            ttsVoice: null,
            speechRate: 1.0,
            speechPitch: 1.0
        };
        
        // Real-time status tracking
        let systemStats = {};
        let heartbeatInterval = null;
        let performanceStats = {
            latency: [],
            quality: 'good',
            throughput: 0,
            lastUpdateTime: Date.now()
        };
        let performanceVisible = false;
        
        // Initialize SocketIO connection with enhanced real-time features
        function initializeSocket() {
            socket = io();
            
            socket.on('connect', function() {
                console.log('Connected to JARVIS server');
                showNotification('🔗 Connected to JARVIS', 'success');
                startHeartbeat();
            });
            
            socket.on('disconnect', function() {
                console.log('Disconnected from JARVIS server');
                showNotification('❌ Disconnected from JARVIS', 'error');
                stopHeartbeat();
            });
            
            socket.on('status', function(data) {
                console.log('Server status:', data.message);
                updateRealTimeStatus(data);
            });
            
            socket.on('system_status', function(data) {
                console.log('System status received:', data);
                updateSystemStatus(data);
            });
            
            socket.on('system_stats', function(data) {
                console.log('System stats received:', data);
                systemStats = data;
                updateStatsDisplay(data);
            });
            
            socket.on('command_processing', function(data) {
                console.log('Command processing:', data);
                showProcessingIndicator(data);
            });
            
            socket.on('command_response', function(data) {
                console.log('Command response received:', data);
                hideProcessingIndicator();
                if (data.response) {
                    addMessage(data.response, false, data.metadata);
                    speakText(data.response);
                }
            });
            
            socket.on('voice_processing', function(data) {
                console.log('Voice processing:', data);
                updateVoiceProcessingStatus(data);
            });
            
            socket.on('voice_ack', function(data) {
                console.log('Voice acknowledged:', data);
                updateVoiceStatus(`Processing chunk ${data.chunk_size} bytes`, 'processing');
            });
            
            socket.on('tts_status', function(data) {
                console.log('TTS status:', data);
                updateTTSStatus(data);
            });
            
            socket.on('tts_response', function(data) {
                console.log('TTS response:', data);
                handleTTSResponse(data);
            });
            
            socket.on('heartbeat_ack', function(data) {
                console.log('Heartbeat acknowledged');
                updateConnectionStatus('Connected', 'good');
            });
            
            socket.on('conversation_context', function(data) {
                console.log('Conversation context received:', data);
                updateConversationContext(data);
            });
            
            socket.on('ai_thinking', function(data) {
                console.log('AI thinking status:', data);
                updateAIThinkingStatus(data);
            });
            
            socket.on('advanced_command_response', function(data) {
                console.log('Advanced command response:', data);
                handleAdvancedCommandResponse(data);
            });
            
            socket.on('error', function(data) {
                console.error('Socket error:', data.message);
                updateVoiceStatus(data.message, 'error');
                showNotification(`❌ Error: ${data.message}`, 'error');
            });
        }
        
        // Start heartbeat for real-time connection monitoring
        function startHeartbeat() {
            heartbeatInterval = setInterval(() => {
                if (socket && socket.connected) {
                    socket.emit('heartbeat', {timestamp: Date.now()});
                }
            }, 10000); // Every 10 seconds
        }
        
        function stopHeartbeat() {
            if (heartbeatInterval) {
                clearInterval(heartbeatInterval);
                heartbeatInterval = null;
            }
        }
        
        // Real-time status updates
        function updateRealTimeStatus(data) {
            const statusElement = document.getElementById('status');
            if (statusElement) {
                statusElement.textContent = data.message;
                statusElement.className = `status ${data.type}`;
            }
        }
        
        function updateSystemStatus(data) {
            // Update system status indicators
            const features = data.features || {};
            updateFeatureStatus('voice_recognition', features.voice_recognition);
            updateFeatureStatus('text_to_speech', features.text_to_speech);
            updateFeatureStatus('thai_language', features.thai_language);
            updateFeatureStatus('conversation_memory', features.conversation_memory);
        }
        
        function updateFeatureStatus(feature, status) {
            // Visual indicator for feature status (could be enhanced with UI elements)
            console.log(`Feature ${feature}: ${status ? 'Active' : 'Inactive'}`);
        }
        
        function updateStatsDisplay(stats) {
            // Update real-time statistics display
            const statsInfo = `
                Active: ${stats.active_clients} clients | 
                Status: ${stats.jarvis_status} | 
                Memory: ${stats.system_load?.memory_usage || 'N/A'}
            `;
            updateVoiceStatus(statsInfo, 'info');
        }
        
        function showProcessingIndicator(data) {
            const indicator = document.createElement('div');
            indicator.id = 'processing-indicator';
            indicator.className = 'processing-indicator';
            indicator.innerHTML = `
                <div class="processing-content">
                    <div class="spinner"></div>
                    <span>Processing: "${data.command}" (${Math.round(data.confidence * 100)}%)</span>
                </div>
            `;
            
            const chatContainer = document.getElementById('chatContainer');
            chatContainer.appendChild(indicator);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        function hideProcessingIndicator() {
            const indicator = document.getElementById('processing-indicator');
            if (indicator) {
                indicator.remove();
            }
        }
        
        function updateVoiceProcessingStatus(data) {
            // Enhanced quality indicators
            const qualityIcons = {
                'excellent': '🟢',
                'good': '🔵', 
                'poor': '🟡',
                'low': '🔴'
            };
            
            const icon = qualityIcons[data.quality] || '🟡';
            updateVoiceStatus(`${icon} Processing chunk ${data.chunk_index}`, 'processing');
            
            // Update performance stats
            if (data.latency !== undefined) {
                performanceStats.latency.push(data.latency);
                if (performanceStats.latency.length > 10) {
                    performanceStats.latency.shift();
                }
            }
            
            performanceStats.quality = data.quality;
            performanceStats.throughput = data.chunk_size_kb || 0;
            
            // Update voice visualizer based on quality
            updateVoiceVisualizerQuality(data.quality);
            
            // Update performance indicator if visible
            if (performanceVisible) {
                updatePerformanceDisplay();
            }
        }
        
        function updateVoiceVisualizerQuality(quality) {
            const visualizer = document.getElementById('voiceVisualizer');
            const waves = document.querySelectorAll('.voice-wave');
            
            // Remove existing quality classes
            visualizer.classList.remove('excellent', 'good', 'poor', 'low');
            waves.forEach(wave => {
                wave.classList.remove('excellent', 'good', 'poor', 'low', 'active');
            });
            
            // Add new quality class
            if (quality) {
                visualizer.classList.add(quality);
                waves.forEach(wave => {
                    wave.classList.add(quality, 'active');
                });
            }
        }
        
        function updatePerformanceDisplay() {
            const latencyElement = document.querySelector('#latencyMetric .value');
            const qualityElement = document.querySelector('#qualityMetric .value');
            const throughputElement = document.querySelector('#throughputMetric .value');
            
            // Calculate average latency
            const avgLatency = performanceStats.latency.length > 0 ? 
                Math.round(performanceStats.latency.reduce((a, b) => a + b, 0) / performanceStats.latency.length) : 0;
            
            latencyElement.textContent = `${avgLatency} ms`;
            qualityElement.textContent = performanceStats.quality.charAt(0).toUpperCase() + performanceStats.quality.slice(1);
            throughputElement.textContent = `${performanceStats.throughput.toFixed(1)} KB/s`;
            
            // Update quality indicators
            const latencyMetric = document.getElementById('latencyMetric');
            const qualityMetric = document.getElementById('qualityMetric');
            
            // Color code based on performance
            latencyMetric.className = `perf-metric ${avgLatency < 100 ? 'excellent' : avgLatency < 300 ? 'good' : avgLatency < 500 ? 'poor' : 'bad'}`;
            qualityMetric.className = `perf-metric ${performanceStats.quality}`;
        }
        
        function togglePerformanceIndicator() {
            const indicator = document.getElementById('performanceIndicator');
            performanceVisible = !performanceVisible;
            indicator.style.display = performanceVisible ? 'block' : 'none';
            
            if (performanceVisible) {
                updatePerformanceDisplay();
            }
        }
        
        function updateTTSStatus(data) {
            if (data.status === 'generating') {
                updateVoiceStatus(`🗣️ Generating speech (${data.text_length} chars)`, 'generating');
            } else if (data.status === 'complete') {
                updateVoiceStatus('🔊 Speech ready', 'ready');
            }
        }
        
        function handleTTSResponse(data) {
            console.log(`TTS ready: ${data.text.substring(0, 50)}... (${data.estimated_duration}s)`);
            // Could trigger client-side TTS or audio playback
        }
        
        function updateConnectionStatus(message, status) {
            const connectionIndicator = document.createElement('div');
            connectionIndicator.className = `connection-status ${status}`;
            connectionIndicator.textContent = message;
            // Could be added to status bar
        }
        
        function showNotification(message, type = 'info') {
            // Create notification element
            const notification = document.createElement('div');
            notification.className = `notification ${type}`;
            notification.innerHTML = `
                <span>${message}</span>
                <button onclick="this.parentElement.remove()">×</button>
            `;
            
            // Add to page
            let notificationArea = document.getElementById('notifications');
            if (!notificationArea) {
                notificationArea = document.createElement('div');
                notificationArea.id = 'notifications';
                notificationArea.className = 'notification-area';
                document.body.appendChild(notificationArea);
            }
            
            notificationArea.appendChild(notification);
            
            // Auto-remove after 5 seconds
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.remove();
                }
            }, 5000);
        }
        
        // Request real-time stats periodically
        function startStatsUpdates() {
            setInterval(() => {
                if (socket && socket.connected) {
                    socket.emit('request_stats');
                    socket.emit('request_conversation_context', {session_id: 'web'});
                }
            }, 30000); // Every 30 seconds
        }
        
        // Handle advanced command response
        function handleAdvancedCommandResponse(data) {
            if (data.error) {
                showNotification(`❌ Command Error: ${data.error}`, 'error');
                return;
            }
            
            const message = `🎯 Command "${data.command_name}" executed successfully`;
            addMessage(message, false, {
                command_type: 'advanced',
                execution_id: data.execution_id,
                status: data.status
            });
            
            showNotification(`✅ Command completed: ${data.command_name}`, 'success');
        }
        
        // Try advanced voice command first, fallback to regular processing
        function tryAdvancedVoiceCommand(command) {
            if (socket && socket.connected) {
                // Check if command looks like a macro or special command
                const advancedPatterns = [
                    /good morning/i, /อรุณสวัสดิ์/i,
                    /volume up/i, /เสียงดัง/i,
                    /time/i, /เวลา/i
                ];
                
                const isAdvanced = advancedPatterns.some(pattern => pattern.test(command));
                
                if (isAdvanced) {
                    socket.emit('voice_command_advanced', {
                        command: command,
                        session_id: 'web'
                    });
                    return true; // Handled by advanced system
                }
            }
            return false; // Use regular processing
        }
        
        // Load and display available commands
        function loadAvailableCommands() {
            fetch('/api/commands')
                .then(response => response.json())
                .then(data => {
                    if (data.commands) {
                        displayAvailableCommands(data.commands);
                    }
                })
                .catch(error => console.error('Failed to load commands:', error));
        }
        
        function displayAvailableCommands(commands) {
            // Could add UI to show available commands
            console.log('Available voice commands:', commands);
        }
        
        // Load system improvement report
        function loadImprovementReport() {
            fetch('/api/improvement/report')
                .then(response => response.json())
                .then(data => {
                    console.log('System improvement report:', data);
                    // Could add UI to show improvement insights
                })
                .catch(error => console.log('Improvement report not available:', error));
        }
        
        // Handle conversation context updates
        function updateConversationContext(context) {
            console.log('Updating conversation context:', context);
            
            // Update context display (could be added to UI)
            if (context.session_active) {
                const contextInfo = `Session: Active | Turns: ${context.recent_turns || 0}`;
                // Could display this in a context indicator
            }
            
            // Update session stats if available
            if (context.connected_duration) {
                const minutes = Math.floor(context.connected_duration / 60);
                const contextStats = `Connected: ${minutes}m | Commands: ${context.commands_issued || 0}`;
                console.log('Session stats:', contextStats);
            }
        }
        
        // Handle AI thinking status
        function updateAIThinkingStatus(data) {
            if (data.status === 'preparing') {
                // Show that AI is preparing to respond
                const thinkingIndicator = document.getElementById('ai-thinking');
                if (!thinkingIndicator) {
                    const indicator = document.createElement('div');
                    indicator.id = 'ai-thinking';
                    indicator.className = 'ai-thinking-indicator';
                    indicator.innerHTML = '🧠 JARVIS is thinking...';
                    
                    const chatContainer = document.getElementById('chatContainer');
                    chatContainer.appendChild(indicator);
                    chatContainer.scrollTop = chatContainer.scrollHeight;
                }
            } else if (data.status === 'ready') {
                // Remove thinking indicator
                const thinkingIndicator = document.getElementById('ai-thinking');
                if (thinkingIndicator) {
                    thinkingIndicator.remove();
                }
            }
        }
        
        // Enhanced typing detection for real-time feedback
        let typingTimeout = null;
        
        function handleTypingStart() {
            if (socket && socket.connected) {
                socket.emit('typing_indicator', {is_typing: true});
            }
        }
        
        function handleTypingStop() {
            if (socket && socket.connected) {
                socket.emit('typing_indicator', {is_typing: false});
            }
        }
        
        // Debounced typing indicator
        function handleTyping() {
            handleTypingStart();
            
            clearTimeout(typingTimeout);
            typingTimeout = setTimeout(() => {
                handleTypingStop();
            }, 1000); // Stop typing indicator after 1 second of inactivity
        }
        
        // Initialize voice functionality
        function initializeVoice() {
            // Check for Web Speech API support
            if ('webkitSpeechRecognition' in window || 'SpeechRecognition' in window) {
                const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
                recognition = new SpeechRecognition();
                
                recognition.continuous = false;
                recognition.interimResults = true;
                recognition.lang = voiceSettings.recognitionLanguage;
                
                recognition.onstart = function() {
                    console.log('Voice recognition started');
                    updateVoiceStatus('Listening...', 'listening');
                };
                
                recognition.onresult = function(event) {
                    let finalTranscript = '';
                    let interimTranscript = '';
                    
                    for (let i = event.resultIndex; i < event.results.length; i++) {
                        const transcript = event.results[i][0].transcript;
                        if (event.results[i].isFinal) {
                            finalTranscript += transcript;
                        } else {
                            interimTranscript += transcript;
                        }
                    }
                    
                    // Show interim results in input field
                    const input = document.getElementById('messageInput');
                    input.value = finalTranscript + interimTranscript;
                    
                    if (finalTranscript) {
                        console.log('Voice recognition result:', finalTranscript);
                        stopVoiceRecording();
                        sendMessage();
                    }
                };
                
                recognition.onerror = function(event) {
                    console.error('Voice recognition error:', event.error);
                    updateVoiceStatus(`Error: ${event.error}`, 'error');
                    stopVoiceRecording();
                };
                
                recognition.onend = function() {
                    console.log('Voice recognition ended');
                    if (isRecording) {
                        stopVoiceRecording();
                    }
                };
                
                updateVoiceStatus('Voice Ready', 'ready');
            } else {
                updateVoiceStatus('Voice not supported', 'error');
                console.warn('Web Speech API not supported');
            }
            
            // Initialize speech synthesis
            if ('speechSynthesis' in window) {
                loadVoices();
                if (speechSynthesis.onvoiceschanged !== undefined) {
                    speechSynthesis.onvoiceschanged = loadVoices;
                }
            }
            
            // Initialize voice wave visualization
            initializeVoiceWaves();
            
            // Set up settings event listeners
            setupSettingsListeners();
        }
        
        // Load available voices
        function loadVoices() {
            voices = speechSynthesis.getVoices();
            const voiceSelect = document.getElementById('ttsVoice');
            
            // Clear existing options except auto
            while (voiceSelect.children.length > 1) {
                voiceSelect.removeChild(voiceSelect.lastChild);
            }
            
            voices.forEach((voice, index) => {
                const option = document.createElement('option');
                option.value = index;
                option.textContent = `${voice.name} (${voice.lang})`;
                voiceSelect.appendChild(option);
            });
            
            // Auto-select appropriate voice
            autoSelectVoice();
        }
        
        // Auto-select voice based on recognition language
        function autoSelectVoice() {
            const langPrefix = voiceSettings.recognitionLanguage.substring(0, 2);
            const voiceSelect = document.getElementById('ttsVoice');
            
            for (let i = 0; i < voices.length; i++) {
                if (voices[i].lang.startsWith(langPrefix)) {
                    voiceSettings.ttsVoice = voices[i];
                    voiceSelect.value = i;
                    break;
                }
            }
        }
        
        // Toggle voice recording
        function toggleVoiceRecording() {
            if (!recognition) {
                alert('Voice recognition not available');
                return;
            }
            
            if (isRecording) {
                stopVoiceRecording();
            } else {
                startVoiceRecording();
            }
        }
        
        // Start voice recording
        function startVoiceRecording() {
            if (!isInitialized) {
                alert('JARVIS is not ready yet');
                return;
            }
            
            try {
                isRecording = true;
                recognition.lang = voiceSettings.recognitionLanguage;
                recognition.start();
                
                const button = document.getElementById('voiceButton');
                button.className = 'voice-button recording';
                button.innerHTML = '🛑 หยุดฟัง';
                
                // Clear input field
                document.getElementById('messageInput').value = '';
                
                // Show voice visualizer
                showVoiceVisualizer(true);
                
            } catch (error) {
                console.error('Error starting voice recognition:', error);
                updateVoiceStatus('Failed to start', 'error');
                isRecording = false;
            }
        }
        
        // Stop voice recording
        function stopVoiceRecording() {
            if (recognition && isRecording) {
                recognition.stop();
            }
            
            isRecording = false;
            
            const button = document.getElementById('voiceButton');
            button.className = 'voice-button';
            button.innerHTML = '🎤 เริ่มพูด';
            
            updateVoiceStatus('Voice Ready', 'ready');
            showVoiceVisualizer(false);
        }
        
        // Update voice status
        function updateVoiceStatus(message, type) {
            const statusElement = document.getElementById('voiceStatus');
            statusElement.textContent = `Voice: ${message}`;
            statusElement.className = `voice-status ${type}`;
        }
        
        // Speak text using TTS
        function speakText(text) {
            if (!speechSynthesis) {
                console.warn('Speech synthesis not available');
                return;
            }
            
            // Stop any current speech
            if (currentUtterance) {
                speechSynthesis.cancel();
            }
            
            const utterance = new SpeechSynthesisUtterance(text);
            
            // Apply voice settings
            if (voiceSettings.ttsVoice) {
                utterance.voice = voiceSettings.ttsVoice;
            }
            utterance.rate = voiceSettings.speechRate;
            utterance.pitch = voiceSettings.speechPitch;
            
            utterance.onstart = function() {
                console.log('TTS started');
                currentUtterance = utterance;
                showVoiceVisualizer(true, 'speaking');
            };
            
            utterance.onend = function() {
                console.log('TTS ended');
                currentUtterance = null;
                showVoiceVisualizer(false);
            };
            
            utterance.onerror = function(event) {
                console.error('TTS error:', event.error);
                currentUtterance = null;
                showVoiceVisualizer(false);
            };
            
            speechSynthesis.speak(utterance);
        }
        
        // Enhanced voice visualizer with quality feedback
        function showVoiceVisualizer(show, mode = 'listening', quality = 'good') {
            const visualizer = document.getElementById('voiceVisualizer');
            const indicator = document.getElementById('speakingIndicator');
            
            if (show) {
                visualizer.style.display = 'flex';
                visualizer.classList.add('active');
                
                if (mode === 'speaking') {
                    indicator.textContent = '🎙️ JARVIS กำลังพูด...';
                    indicator.className = 'speaking-indicator active';
                    animateVoiceWaves(true, quality);
                } else {
                    const qualityText = quality === 'excellent' ? 'คุณภาพดีเยี่ยม' :
                                      quality === 'good' ? 'คุณภาพดี' :
                                      quality === 'poor' ? 'คุณภาพปานกลาง' : 'คุณภาพต่ำ';
                    indicator.textContent = `🎤 กำลังฟัง... (${qualityText})`;
                    indicator.className = 'speaking-indicator active';
                    animateVoiceWaves(false, quality);
                }
                
                // Update visualizer quality class
                updateVoiceVisualizerQuality(quality);
            } else {
                visualizer.style.display = 'none';
                visualizer.classList.remove('active');
                indicator.className = 'speaking-indicator';
                stopVoiceWaveAnimation();
                
                // Clear quality classes
                updateVoiceVisualizerQuality(null);
            }
        }
        
        // Initialize voice wave visualization
        function initializeVoiceWaves() {
            const wavesContainer = document.getElementById('voiceWaves');
            for (let i = 0; i < 20; i++) {
                const wave = document.createElement('div');
                wave.className = 'voice-wave';
                wave.style.height = '10px';
                wavesContainer.appendChild(wave);
            }
        }
        
        // Enhanced voice wave animation with quality-based patterns
        let waveAnimation = null;
        
        function animateVoiceWaves(speaking = false, quality = 'good') {
            const waves = document.querySelectorAll('.voice-wave');
            
            if (waveAnimation) {
                clearInterval(waveAnimation);
            }
            
            // Quality-based animation parameters
            const qualitySettings = {
                'excellent': { variance: 50, baseHeight: 25, speed: 80 },
                'good': { variance: 40, baseHeight: 20, speed: 100 },
                'poor': { variance: 25, baseHeight: 15, speed: 150 },
                'low': { variance: 15, baseHeight: 10, speed: 200 }
            };
            
            const settings = qualitySettings[quality] || qualitySettings.good;
            
            let animationPhase = 0;
            waveAnimation = setInterval(() => {
                animationPhase += 0.1;
                
                waves.forEach((wave, index) => {
                    let height;
                    
                    if (speaking) {
                        // More dynamic speaking animation based on quality
                        height = Math.random() * settings.variance + settings.baseHeight;
                        
                        // Add some harmonic patterns for better visual effect
                        const harmonic = Math.sin(animationPhase + index * 0.5) * (settings.variance * 0.3);
                        height += harmonic;
                    } else {
                        // Listening pattern with quality-based intensity
                        const baseWave = Math.sin(Date.now() * 0.005 + index * 0.3) * (settings.variance * 0.5);
                        const qualityMultiplier = quality === 'excellent' ? 1.5 : 
                                                quality === 'good' ? 1.0 : 
                                                quality === 'poor' ? 0.7 : 0.4;
                        height = settings.baseHeight + baseWave * qualityMultiplier;
                    }
                    
                    // Ensure minimum height
                    height = Math.max(height, 5);
                    wave.style.height = `${height}px`;
                });
            }, settings.speed);
        }
        
        function stopVoiceWaveAnimation() {
            if (waveAnimation) {
                clearInterval(waveAnimation);
                waveAnimation = null;
            }
            
            const waves = document.querySelectorAll('.voice-wave');
            waves.forEach(wave => {
                wave.style.height = '10px';
            });
        }
        
        // Toggle settings panel
        function toggleSettings() {
            const settings = document.getElementById('voiceSettings');
            settings.style.display = settings.style.display === 'none' ? 'block' : 'none';
        }
        
        // Setup settings event listeners
        function setupSettingsListeners() {
            // Recognition language change
            document.getElementById('recognitionLanguage').addEventListener('change', function(e) {
                voiceSettings.recognitionLanguage = e.target.value;
                if (recognition) {
                    recognition.lang = voiceSettings.recognitionLanguage;
                }
                autoSelectVoice();
            });
            
            // TTS voice change
            document.getElementById('ttsVoice').addEventListener('change', function(e) {
                if (e.target.value === 'auto') {
                    autoSelectVoice();
                } else {
                    voiceSettings.ttsVoice = voices[parseInt(e.target.value)];
                }
            });
            
            // Speech rate change
            document.getElementById('speechRate').addEventListener('input', function(e) {
                voiceSettings.speechRate = parseFloat(e.target.value);
                document.getElementById('rateValue').textContent = e.target.value;
            });
            
            // Speech pitch change
            document.getElementById('speechPitch').addEventListener('input', function(e) {
                voiceSettings.speechPitch = parseFloat(e.target.value);
                document.getElementById('pitchValue').textContent = e.target.value;
            });
        }
        
        // Check JARVIS status
        function checkStatus() {
            fetch('/api/status')
                .then(response => response.json())
                .then(data => {
                    const statusElement = document.getElementById('status');
                    statusElement.textContent = data.message;
                    
                    if (data.status === 'ready') {
                        statusElement.className = 'status ready';
                        isInitialized = true;
                        document.getElementById('messageInput').disabled = false;
                        document.getElementById('sendButton').disabled = false;
                        document.getElementById('voiceButton').disabled = false;
                        
                        // Initialize SocketIO and voice functionality
                        initializeSocket();
                        initializeVoice();
                        startStatsUpdates();
                        loadAvailableCommands();
                        loadImprovementReport();
                    } else if (data.status === 'error') {
                        statusElement.className = 'status error';
                    } else {
                        setTimeout(checkStatus, 2000); // Check again in 2 seconds
                    }
                })
                .catch(error => {
                    console.error('Status check failed:', error);
                    setTimeout(checkStatus, 5000);
                });
        }
        
        // Send message (updated to support advanced voice commands)
        function sendMessage() {
            const input = document.getElementById('messageInput');
            const message = input.value.trim();
            
            if (!message || !isInitialized) return;
            
            // Try advanced voice command first
            const handledByAdvanced = tryAdvancedVoiceCommand(message);
            if (handledByAdvanced) {
                addMessage(message, true);
                input.value = '';
                return;
            }
            
            // Add user message
            addMessage(message, true);
            input.value = '';
            
            // Show typing indicator
            document.getElementById('typing').style.display = 'block';
            document.getElementById('sendButton').disabled = true;
            
            // Send to server
            fetch('/api/message', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ message: message })
            })
            .then(response => response.json())
            .then(data => {
                document.getElementById('typing').style.display = 'none';
                document.getElementById('sendButton').disabled = false;
                
                if (data.error) {
                    const errorMessage = '❌ ' + data.error;
                    addMessage(errorMessage, false);
                    speakText(errorMessage);
                } else {
                    addMessage(data.response, false, data.metadata);
                    // Speak the response
                    speakText(data.response);
                }
            })
            .catch(error => {
                document.getElementById('typing').style.display = 'none';
                document.getElementById('sendButton').disabled = false;
                const errorMessage = '❌ เกิดข้อผิดพลาดในการเชื่อมต่อ';
                addMessage(errorMessage, false);
                speakText(errorMessage);
                console.error('Error:', error);
            });
        }
        
        // Add message to chat (unchanged)
        function addMessage(text, isUser, metadata = null) {
            const chatContainer = document.getElementById('chatContainer');
            
            const messageDiv = document.createElement('div');
            messageDiv.className = `message ${isUser ? 'user' : 'jarvis'}`;
            
            const avatarDiv = document.createElement('div');
            avatarDiv.className = 'message-avatar';
            avatarDiv.textContent = isUser ? '👤' : '🤖';
            
            const contentDiv = document.createElement('div');
            contentDiv.className = 'message-content';
            contentDiv.textContent = text;
            
            messageDiv.appendChild(avatarDiv);
            messageDiv.appendChild(contentDiv);
            
            // Add metadata for JARVIS responses
            if (!isUser && metadata) {
                const metaDiv = document.createElement('div');
                metaDiv.className = 'message-meta';
                const metaParts = [];
                
                if (metadata.intent) metaParts.push(`Intent: ${metadata.intent}`);
                if (metadata.confidence) metaParts.push(`Confidence: ${metadata.confidence}%`);
                if (metadata.language) metaParts.push(`Language: ${metadata.language}`);
                
                metaDiv.textContent = metaParts.join(' | ');
                contentDiv.appendChild(metaDiv);
            }
            
            chatContainer.appendChild(messageDiv);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
        
        // Enhanced input event handling with real-time feedback
        document.getElementById('messageInput').addEventListener('keypress', function(e) {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleTypingStop(); // Clear typing indicator
                sendMessage();
            } else {
                handleTyping(); // Trigger typing indicator
            }
        });
        
        document.getElementById('messageInput').addEventListener('input', function(e) {
            handleTyping(); // Trigger typing indicator on any input change
        });
        
        // Keyboard shortcuts
        document.addEventListener('keydown', function(e) {
            // Space bar to toggle voice recording (when not typing)
            if (e.code === 'Space' && e.target !== document.getElementById('messageInput')) {
                e.preventDefault();
                toggleVoiceRecording();
            }
            
            // Escape to stop voice recording
            if (e.code === 'Escape' && isRecording) {
                e.preventDefault();
                stopVoiceRecording();
            }
        });
        
        // Start status checking
        checkStatus();
    </script>
</body>
</html>
"""

if __name__ == "__main__":
    print("🚀 Starting JARVIS Web Application...")
    print("🚀 เริ่มต้นแอปพลิเคชันเว็บ JARVIS...")
    
    # Initialize JARVIS
    initialize_jarvis()
    
    print("\n✅ JARVIS Web App is starting...")
    print("✅ แอปพลิเคชันเว็บ JARVIS กำลังเริ่มต้น...")
    print("\n🌐 Access JARVIS at: http://localhost:5000")
    print("🌐 เข้าใช้งาน JARVIS ได้ที่: http://localhost:5000")
    
    # Run Flask app with SocketIO
    socketio.run(app, host='0.0.0.0', port=5000, debug=False, allow_unsafe_werkzeug=True)