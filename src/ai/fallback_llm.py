"""
Fallback LLM Engine for Jarvis Voice Assistant
Provides intelligent responses without requiring external models
"""

import logging
import time
import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from PyQt6.QtCore import QObject, pyqtSignal


class FallbackLLMEngine(QObject):
    """Fallback LLM engine with intelligent pattern-based responses"""
    
    # Signals
    response_ready = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.is_ready = True  # Always ready
        
        # Response patterns and templates
        self.patterns = self._initialize_patterns()
        self.context_memory = []  # Simple conversation memory
        
        self.logger.info("Fallback LLM engine initialized")
    
    def _initialize_patterns(self) -> Dict[str, Any]:
        """Initialize response patterns"""
        return {
            # Greetings
            "greetings": {
                "patterns": [
                    r"^(สวัสดี|hello|hi|hey|good morning|good afternoon|good evening)",
                    r"(สวัสดี|หวัดดี)",
                    r"(hello|hi|hey)"
                ],
                "responses": {
                    "th": [
                        "สวัสดีครับ! ผม J.A.R.V.I.S พร้อมช่วยเหลือคุณ มีอะไรให้ช่วยไหมครับ?",
                        "ยินดีต้อนรับครับ! วันนี้มีอะไรให้ผมช่วยบ้างไหมครับ?",
                        "สวัสดีครับ! ผมพร้อมให้บริการแล้ว"
                    ],
                    "en": [
                        "Hello! I'm J.A.R.V.I.S, ready to assist you. How may I help you today?",
                        "Greetings! What can I do for you today?",
                        "Hello! I'm at your service. What do you need?"
                    ]
                }
            },
            
            # Questions about time
            "time_queries": {
                "patterns": [
                    r"(เวลา|time|กี่โมง|what time)",
                    r"(วันนี้|today|date|วันที่)"
                ],
                "responses": {
                    "th": [
                        "ตอนนี้เวลา {time} น. วัน{weekday}ที่ {date} ครับ",
                        "เวลาปัจจุบัน {time} น. ครับ"
                    ],
                    "en": [
                        "The current time is {time}. Today is {weekday}, {date}",
                        "It's currently {time}"
                    ]
                }
            },
            
            # Questions about weather
            "weather_queries": {
                "patterns": [
                    r"(อากาศ|weather|ฝน|แดด|หนาว|ร้อน)",
                    r"(temperature|rain|sunny|cloudy)"
                ],
                "responses": {
                    "th": [
                        "ขออภัยครับ ผมไม่สามารถเข้าถึงข้อมูลสภาพอากาศปัจจุบันได้ กรุณาตรวจสอบจากแหล่งข้อมูลอื่นครับ",
                        "สำหรับข้อมูลสภาพอากาศที่แม่นยำ แนะนำให้ตรวจสอบจากแอปพยากรณ์อากาศครับ"
                    ],
                    "en": [
                        "I apologize, but I don't have access to current weather data. Please check a weather app for accurate information.",
                        "For precise weather information, I recommend checking a weather service or app."
                    ]
                }
            },
            
            # Technology questions
            "tech_queries": {
                "patterns": [
                    r"(ai|ปัญญาประดิษฐ์|artificial intelligence|machine learning|robot|หุ่นยนต์)",
                    r"(technology|เทคโนโลยี|computer|คอมพิวเตอร์)"
                ],
                "responses": {
                    "th": [
                        "ปัญญาประดิษฐ์เป็นเทคโนโลยีที่ให้เครื่องจักรเรียนรู้และทำงานเหมือนมนุษย์ได้ ปัจจุบันใช้ในหลายด้าน เช่น การแพทย์ การศึกษา และการธุรกิจครับ",
                        "เทคโนโลยี AI กำลังพัฒนาอย่างรวดเร็ว ช่วยให้เราทำงานได้อย่างมีประสิทธิภาพมากขึ้นครับ"
                    ],
                    "en": [
                        "Artificial Intelligence enables machines to learn and perform tasks that typically require human intelligence. It's widely used in healthcare, education, and business.",
                        "AI technology is rapidly advancing, helping us work more efficiently and solve complex problems."
                    ]
                }
            },
            
            # Help requests
            "help_requests": {
                "patterns": [
                    r"(ช่วย|help|assist|support|คำแนะนำ)",
                    r"(ทำอย่างไร|how to|how can)"
                ],
                "responses": {
                    "th": [
                        "ผมพร้อมช่วยเหลือครับ! คุณสามารถถามผมเกี่ยวกับ:\n- ข้อมูลทั่วไป\n- เวลาและวันที่\n- เทคโนโลยี\n- การแปลภาษา\n- และอื่นๆ อีกมากมาย",
                        "ยินดีช่วยเหลือครับ! มีอะไรเฉพาะเจาะจงที่อยากทราบไหมครับ?"
                    ],
                    "en": [
                        "I'm here to help! You can ask me about:\n- General information\n- Time and date\n- Technology\n- Language translation\n- And much more",
                        "Happy to assist! What specific information do you need?"
                    ]
                }
            },
            
            # Thanks
            "thanks": {
                "patterns": [
                    r"(ขอบคุณ|thank you|thanks|ขอบใจ)",
                    r"(เยี่ยม|great|good|ดี|excellent)"
                ],
                "responses": {
                    "th": [
                        "ด้วยความยินดีครับ! มีอะไรอื่นให้ช่วยอีกไหมครับ?",
                        "ไม่เป็นไรครับ ยินดีที่ได้ช่วยเหลือ",
                        "ขอบคุณครับ! พร้อมช่วยเหลือเสมอ"
                    ],
                    "en": [
                        "You're welcome! Is there anything else I can help you with?",
                        "My pleasure! Happy to assist anytime.",
                        "Glad I could help! Let me know if you need anything else."
                    ]
                }
            },
            
            # Questions about Jarvis/self
            "self_queries": {
                "patterns": [
                    r"(คุณคือใคร|who are you|what are you|jarvis)",
                    r"(ทำอะไรได้|capabilities|features|ฟีเจอร์)"
                ],
                "responses": {
                    "th": [
                        "ผมคือ J.A.R.V.I.S ผู้ช่วยอัจฉริยะส่วนตัวของคุณ พัฒนาขึ้นเพื่อช่วยงานต่างๆ ด้วยการประมวลผลภายในเครื่องเพื่อความเป็นส่วนตัว",
                        "ผม J.A.R.V.I.S สามารถช่วยตอบคำถาม แปลภาษา สร้างภาพ และอื่นๆ อีกมากมายครับ"
                    ],
                    "en": [
                        "I'm J.A.R.V.I.S, your personal AI assistant. I'm designed to help with various tasks using local processing for privacy.",
                        "I'm J.A.R.V.I.S. I can answer questions, translate languages, generate images, and much more."
                    ]
                }
            }
        }
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is Thai or English"""
        thai_chars = "กขคงจฉชซณดตถทธนบปผฝพฟภมยรลวศษสหฬอฮะัาำิีึืุูเแโใไ"
        thai_count = sum(1 for char in text if char in thai_chars)
        return "th" if thai_count > 0 else "en"
    
    def _get_time_vars(self) -> Dict[str, str]:
        """Get time-related variables for templates"""
        now = datetime.now()
        thai_weekdays = ["จันทร์", "อังคาร", "พุธ", "พฤหัสบดี", "ศุกร์", "เสาร์", "อาทิตย์"]
        english_weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        return {
            "time": now.strftime("%H:%M"),
            "date": now.strftime("%d/%m/%Y"),
            "weekday": thai_weekdays[now.weekday()],
            "weekday_en": english_weekdays[now.weekday()]
        }
    
    def _match_pattern(self, text: str) -> tuple:
        """Match text against patterns and return category and language"""
        text_lower = text.lower()
        language = self._detect_language(text)
        
        for category, data in self.patterns.items():
            for pattern in data["patterns"]:
                if re.search(pattern, text_lower, re.IGNORECASE):
                    return category, language
        
        return "unknown", language
    
    def _generate_response(self, text: str, language: str, category: str) -> str:
        """Generate appropriate response"""
        try:
            if category == "unknown":
                if language == "th":
                    return f"ผมเข้าใจว่าคุณพูดเรื่อง '{text}' ครับ ผมกำลังพัฒนาการเข้าใจที่ดีขึ้น มีอะไรเฉพาะเจาะจงที่ผมช่วยได้ไหมครับ?"
                else:
                    return f"I understand you're asking about '{text}'. I'm continuously learning to provide better responses. Is there something specific I can help you with?"
            
            pattern_data = self.patterns[category]
            responses = pattern_data["responses"][language]
            
            # Choose response (could be randomized)
            import random
            response = random.choice(responses)
            
            # Fill in time variables if needed
            if "{time}" in response or "{date}" in response or "{weekday}" in response:
                time_vars = self._get_time_vars()
                response = response.format(**time_vars)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Response generation error: {e}")
            if language == "th":
                return "ขออภัยครับ เกิดข้อผิดพลาดในการประมวลผล กรุณาลองใหม่อีกครั้งครับ"
            else:
                return "I apologize, there was an error processing your request. Please try again."
    
    def generate_response(self, prompt: str, language: str = None, **kwargs) -> Dict[str, Any]:
        """Generate response for given prompt"""
        try:
            start_time = time.time()
            
            if not prompt or not prompt.strip():
                error_msg = "Empty prompt provided"
                self.error_occurred.emit(error_msg)
                return {"error": error_msg}
            
            # Auto-detect language if not provided
            if not language:
                language = self._detect_language(prompt)
            
            # Store in conversation memory
            self.context_memory.append({
                "timestamp": datetime.now().isoformat(),
                "input": prompt,
                "language": language
            })
            
            # Keep only last 5 exchanges
            if len(self.context_memory) > 5:
                self.context_memory = self.context_memory[-5:]
            
            # Match pattern and generate response
            category, detected_lang = self._match_pattern(prompt)
            response_text = self._generate_response(prompt, detected_lang, category)
            
            processing_time = time.time() - start_time
            
            # Create response object
            response = {
                "text": response_text,
                "language": detected_lang,
                "confidence": 0.9 if category != "unknown" else 0.6,
                "processing_time": processing_time,
                "token_count": len(response_text.split()),
                "model_info": {
                    "model_name": "Fallback Pattern Engine",
                    "version": "1.0",
                    "category": category
                }
            }
            
            self.logger.info(f"Generated fallback response in {processing_time:.3f}s (category: {category})")
            
            # Add to memory
            self.context_memory[-1]["response"] = response_text
            self.context_memory[-1]["category"] = category
            
            # Emit success signal
            self.response_ready.emit(response)
            return response
            
        except Exception as e:
            error_msg = f"Fallback LLM error: {e}"
            self.logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return {"error": error_msg}
    
    def is_engine_ready(self) -> bool:
        """Check if engine is ready"""
        return self.is_ready
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            "model_name": "Fallback Pattern Engine",
            "version": "1.0",
            "ready": self.is_ready,
            "patterns_loaded": len(self.patterns),
            "conversation_memory": len(self.context_memory),
            "supported_languages": ["th", "en"]
        }