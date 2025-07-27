#!/usr/bin/env python3
"""
🤖 JARVIS Complete Voice System
ระบบเสียงครบครันที่รวม Real-time Conversation, Thai Processing, Memory และ Wake Word
"""

import logging
import sys
import time
import threading
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import JARVIS components
try:
    from voice.realtime_conversation import RealTimeVoiceConversation, VoiceMessage
    from voice.advanced_thai_processor import AdvancedThaiProcessor
    from voice.simple_wake_word import SimpleWakeWordDetector
    from features.advanced_conversation_memory import AdvancedConversationMemory
    from ai.llm_engine import LLMEngine
    from system.config_manager_v2 import ConfigurationManager
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    COMPONENTS_AVAILABLE = False
    logging.error(f"JARVIS components not available: {e}")


class JarvisVoiceSystem:
    """ระบบเสียงครบครันของ JARVIS"""
    
    def __init__(self, config_path: str = "config/default_config.yaml"):
        self.logger = logging.getLogger(__name__)
        
        # Load configuration  
        self.config_path = config_path
        self.config = {}
        
        # Use simple config loading first
        try:
            import yaml
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    self.config = yaml.safe_load(f) or {}
                self.logger.info("✅ Configuration loaded successfully")
            else:
                self.logger.warning(f"⚠️ Config file not found: {config_path}")
                self.config = {}
        except Exception as e:
            self.logger.error(f"❌ Configuration loading failed: {e}")
            self.config = {}
        
        # Initialize components
        self.conversation_system = None
        self.thai_processor = None
        self.wake_word_detector = None
        self.memory_system = None
        self.llm_engine = None
        
        # System state
        self.is_active = False
        self.current_mode = "idle"  # idle, listening, processing, responding
        self.conversation_active = False
        
        # Statistics
        self.stats = {
            "total_conversations": 0,
            "wake_words_detected": 0,
            "responses_generated": 0,
            "thai_interactions": 0,
            "english_interactions": 0,
            "start_time": time.time()
        }
        
        if COMPONENTS_AVAILABLE:
            self._initialize_components()
        else:
            self.logger.error("❌ Cannot initialize - components not available")
    
    def _initialize_components(self):
        """เริ่มต้นระบบย่อยทั้งหมด"""
        self.logger.info("🚀 Initializing JARVIS Voice System components...")
        
        try:
            # Initialize Thai processor
            self.thai_processor = AdvancedThaiProcessor(self.config.get('thai', {}))
            self.logger.info("✅ Thai processor ready")
            
            # Initialize conversation memory
            self.memory_system = AdvancedConversationMemory(
                config=self.config.get('memory', {}),
                data_dir="data/conversation_memory"
            )
            self.logger.info("✅ Memory system ready")
            
            # Initialize real-time conversation
            conversation_config = self.config.get('voice', {})
            conversation_config.update({
                'whisper_model': 'tiny',  # Fast for real-time
                'sample_rate': 16000,
                'silence_threshold': 0.01,
                'silence_duration': 2.0
            })
            
            self.conversation_system = RealTimeVoiceConversation(conversation_config)
            
            # Set up conversation callbacks
            self.conversation_system.on_speech_detected = self._on_speech_detected
            self.conversation_system.on_text_recognized = self._on_text_recognized
            self.conversation_system.on_response_generated = self._on_response_generated
            
            self.logger.info("✅ Real-time conversation ready")
            
            # Initialize wake word detector
            wake_word_config = {
                'whisper_model': 'tiny',
                'confidence_threshold': 0.7,
                'sample_rate': 16000
            }
            
            self.wake_word_detector = SimpleWakeWordDetector(wake_word_config)
            
            # Set up wake word callbacks
            self.wake_word_detector.on_wake_word = self._on_wake_word_detected
            self.wake_word_detector.on_listening_started = self._on_wake_listening_started
            self.wake_word_detector.on_listening_stopped = self._on_wake_listening_stopped
            
            self.logger.info("✅ Wake word detector ready")
            
            # Initialize AI with fallback support
            try:
                from ai.deepseek_integration import DeepSeekR1
                self.llm_engine = DeepSeekR1(config_path=str(Path(__file__).parent / "config" / "default_config.yaml"))
                model_info = self.llm_engine.get_model_info()
                self.logger.info(f"✅ AI system ready: {model_info['name']} ({model_info['status']})")
            except Exception as e:
                self.logger.warning(f"⚠️ AI engine not ready: {e}")
                # Create fallback AI directly
                try:
                    from ai.fallback_ai import FallbackAI
                    self.llm_engine = FallbackAI()
                    self.logger.info("✅ Fallback AI system ready")
                except Exception as e2:
                    self.logger.error(f"Fallback AI also failed: {e2}")
                    self.llm_engine = None
            
            self.logger.info("🎉 All JARVIS components initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Component initialization failed: {e}")
    
    def start_system(self) -> bool:
        """เริ่มระบบ JARVIS"""
        if not COMPONENTS_AVAILABLE:
            self.logger.error("❌ Cannot start - components not available")
            return False
        
        if self.is_active:
            self.logger.warning("⚠️ System already active")
            return True
        
        self.logger.info("🚀 Starting JARVIS Voice System...")
        
        try:
            # Start wake word detection
            if self.wake_word_detector:
                if not self.wake_word_detector.start_listening():
                    self.logger.error("❌ Failed to start wake word detection")
                    return False
            
            self.is_active = True
            self.current_mode = "listening"
            self.stats["start_time"] = time.time()
            
            self.logger.info("✅ JARVIS Voice System is now active!")
            self.logger.info("🎙️ Say 'Hey JARVIS' to start a conversation")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ System startup failed: {e}")
            return False
    
    def stop_system(self):
        """หยุดระบบ JARVIS"""
        if not self.is_active:
            return
        
        self.logger.info("🛑 Stopping JARVIS Voice System...")
        
        try:
            # Stop conversation if active
            if self.conversation_active and self.conversation_system:
                self.conversation_system.stop_conversation()
                self.conversation_active = False
            
            # Stop wake word detection
            if self.wake_word_detector:
                self.wake_word_detector.stop_listening()
            
            self.is_active = False
            self.current_mode = "idle"
            
            self.logger.info("✅ JARVIS Voice System stopped")
            
        except Exception as e:
            self.logger.error(f"❌ System shutdown error: {e}")
    
    def _on_wake_word_detected(self, text: str, confidence: float):
        """เมื่อตรวจพบคำปลุก"""
        self.logger.info(f"🚨 Wake word detected: '{text}' (confidence: {confidence:.2f})")
        self.stats["wake_words_detected"] += 1
        
        # Start conversation
        self._start_conversation()
    
    def _on_wake_listening_started(self):
        """เมื่อเริ่มฟังคำปลุก"""
        self.logger.debug("👂 Listening for wake words...")
    
    def _on_wake_listening_stopped(self):
        """เมื่อหยุดฟังคำปลุก"""
        self.logger.debug("🔇 Wake word listening stopped")
    
    def _start_conversation(self):
        """เริ่มการสนทนา"""
        if self.conversation_active:
            self.logger.warning("⚠️ Conversation already active")
            return
        
        self.logger.info("💬 Starting conversation...")
        self.current_mode = "processing"
        
        # Stop wake word detection temporarily
        if self.wake_word_detector:
            self.wake_word_detector.stop_listening()
        
        # Start conversation system
        if self.conversation_system:
            if self.conversation_system.start_conversation():
                self.conversation_active = True
                self.current_mode = "listening"
                self.stats["total_conversations"] += 1
                
                # Auto-timeout conversation after 30 seconds of silence
                threading.Timer(30.0, self._timeout_conversation).start()
                
                self.logger.info("✅ Conversation started! Speak now...")
            else:
                self.logger.error("❌ Failed to start conversation")
                self._restart_wake_detection()
    
    def _timeout_conversation(self):
        """หมดเวลาการสนทนา"""
        if self.conversation_active:
            self.logger.info("⏰ Conversation timeout")
            self._end_conversation()
    
    def _end_conversation(self):
        """จบการสนทนา"""
        if not self.conversation_active:
            return
        
        self.logger.info("👋 Ending conversation...")
        
        # Stop conversation system
        if self.conversation_system:
            self.conversation_system.stop_conversation()
        
        self.conversation_active = False
        self.current_mode = "listening"
        
        # Restart wake word detection
        self._restart_wake_detection()
    
    def _restart_wake_detection(self):
        """เริ่มฟังคำปลุกใหม่"""
        if self.wake_word_detector and self.is_active:
            time.sleep(1)  # Brief pause
            self.wake_word_detector.start_listening()
    
    def _on_speech_detected(self):
        """เมื่อตรวจพบการพูด"""
        self.logger.debug("🎤 Speech detected...")
        self.current_mode = "listening"
    
    def _on_text_recognized(self, message: VoiceMessage):
        """เมื่อรู้จำข้อความได้"""
        self.logger.info(f"📝 User said ({message.language}): {message.text}")
        self.current_mode = "processing"
        
        # Update language statistics
        if message.language == 'th':
            self.stats["thai_interactions"] += 1
        else:
            self.stats["english_interactions"] += 1
        
        # Process with Thai processor if Thai
        if message.language == 'th' and self.thai_processor:
            thai_result = self.thai_processor.process_text(message.text)
            context = self.thai_processor.generate_thai_response_context(
                message.text, thai_result.intent
            )
            self.logger.debug(f"🇹🇭 Thai context: {context}")
        
        # Generate AI response
        self._generate_response(message)
    
    def _generate_response(self, message: VoiceMessage):
        """สร้างการตอบสนอง"""
        try:
            # Get conversation context from memory
            if self.memory_system:
                context_turns = self.memory_system.get_relevant_context(
                    message.text, max_turns=3
                )
                context_text = " ".join([turn.user_input for turn in context_turns])
            else:
                context_text = ""
            
            # Generate response using AI engine
            if self.llm_engine:
                # Check if it's the new AI system or old format
                if hasattr(self.llm_engine, 'generate_response'):
                    # New AI system (DeepSeek or Fallback)
                    context_data = {
                        'conversation_history': [
                            {'user': turn.user_input, 'assistant': turn.ai_response}
                            for turn in (context_turns if self.memory_system else [])
                        ]
                    }
                    ai_response = self.llm_engine.generate_response(message.text, context_data)
                else:
                    # Old AI system format
                    prompt = f"Context: {context_text}\nUser: {message.text}\nJARVIS:"
                    response = self.llm_engine.process_query(
                        prompt, 
                        language=message.language
                    )
                    ai_response = response.get('response', '') if response else ''
                
                if ai_response:
                    
                    # Process Thai response if needed
                    if message.language == 'th' and self.thai_processor:
                        thai_context = {'recommended_particle': 'ครับ'}
                        ai_response = self.thai_processor.format_thai_response(
                            ai_response, thai_context
                        )
                    
                    # Add to conversation memory
                    if self.memory_system:
                        self.memory_system.add_conversation_turn(
                            user_input=message.text,
                            ai_response=ai_response,
                            user_language=message.language,
                            intent="response"
                        )
                    
                    # Add to conversation system
                    if self.conversation_system:
                        self.conversation_system.add_jarvis_response(
                            ai_response, message.language
                        )
                    
                    self.stats["responses_generated"] += 1
                    self.logger.info(f"🤖 JARVIS responds: {ai_response}")
                    
                else:
                    self.logger.warning("⚠️ No AI response generated")
            
            else:
                # Fallback response
                fallback_responses = {
                    'en': "I understand. How can I help you further?",
                    'th': "ผมเข้าใจแล้วครับ มีอะไรให้ช่วยอีกไหมครับ"
                }
                
                response = fallback_responses.get(message.language, fallback_responses['en'])
                
                if self.conversation_system:
                    self.conversation_system.add_jarvis_response(response, message.language)
                
                self.logger.info(f"🤖 JARVIS (fallback): {response}")
            
        except Exception as e:
            self.logger.error(f"❌ Response generation failed: {e}")
        
        finally:
            self.current_mode = "listening"
    
    def _on_response_generated(self, message: VoiceMessage):
        """เมื่อสร้างการตอบสนองเสร็จ"""
        self.logger.debug(f"✅ Response ready: {message.text}")
        self.current_mode = "responding"
    
    def get_system_status(self) -> Dict[str, Any]:
        """สถานะระบบ"""
        uptime = time.time() - self.stats.get("start_time", time.time())
        
        status = {
            "system_active": self.is_active,
            "current_mode": self.current_mode,
            "conversation_active": self.conversation_active,
            "components": {
                "conversation_system": self.conversation_system is not None,
                "thai_processor": self.thai_processor is not None,
                "wake_word_detector": self.wake_word_detector is not None,
                "memory_system": self.memory_system is not None,
                "llm_engine": self.llm_engine is not None
            },
            "statistics": self.stats.copy(),
            "uptime_seconds": uptime
        }
        
        return status
    
    def process_text_command(self, text: str, language: str = "en") -> str:
        """ประมวลผลคำสั่งข้อความ (สำหรับทดสอบ)"""
        if not self.is_active:
            return "System not active"
        
        # Create mock voice message
        message = VoiceMessage(
            message_id="test",
            text=text,
            language=language,
            timestamp=time.time(),
            speaker="user"
        )
        
        # Process normally
        self._on_text_recognized(message)
        
        return "Command processed"


def test_complete_voice_system():
    """ทดสอบระบบเสียงครบครัน"""
    print("🧪 Testing JARVIS Complete Voice System...")
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create system
    jarvis = JarvisVoiceSystem()
    
    # Check system status
    print(f"\n📊 Initial status: {jarvis.get_system_status()}")
    
    # Start system
    print(f"\n🚀 Starting system...")
    if jarvis.start_system():
        print("✅ System started successfully!")
        
        # Test text commands
        print(f"\n💬 Testing text commands...")
        test_commands = [
            ("Hello JARVIS", "en"),
            ("What time is it?", "en"),
            ("สวัสดีครับ", "th"),
            ("ช่วยบอกเวลาหน่อยครับ", "th")
        ]
        
        for text, lang in test_commands:
            print(f"   Testing: {text} ({lang})")
            jarvis.process_text_command(text, lang)
            time.sleep(1)  # Brief pause between commands
        
        # Show final status
        print(f"\n📊 Final status: {jarvis.get_system_status()}")
        
        # Stop system
        time.sleep(2)
        print(f"\n🛑 Stopping system...")
        jarvis.stop_system()
        
    else:
        print("❌ Failed to start system")
    
    print("\n✅ Complete voice system test finished!")


if __name__ == "__main__":
    test_complete_voice_system()