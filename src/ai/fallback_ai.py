#!/usr/bin/env python3
"""
🤖 Fallback AI System for JARVIS
When DeepSeek-R1 is not available, use lighter models
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime
import random

logger = logging.getLogger(__name__)

class FallbackAI:
    """Lightweight AI for when main models aren't available"""
    
    def __init__(self):
        self.name = "JARVIS Fallback AI"
        self.responses = {
            "greeting": [
                "Hello! I'm JARVIS operating in fallback mode. How may I assist you?",
                "Greetings! JARVIS fallback system is ready to help.",
                "สวัสดีครับ! ระบบสำรองของ JARVIS พร้อมให้บริการ"
            ],
            "thai_greeting": [
                "สวัสดีครับ! ผม JARVIS กำลังทำงานในโหมดสำรอง มีอะไรให้ช่วยไหมครับ?",
                "สวัสดีครับ! ระบบสำรองของ JARVIS พร้อมแล้วครับ",
                "ยินดีต้อนรับครับ! ผม JARVIS ในโหมดประหยัดพลังงาน"
            ],
            "status": [
                "JARVIS fallback system is operational. Main AI is currently downloading.",
                "Operating in emergency mode. All basic functions available.",
                "ระบบสำรองทำงานปกติ โมเดล AI หลักกำลังดาวน์โหลด"
            ],
            "time": [
                f"The current time is {datetime.now().strftime('%H:%M:%S')}",
                f"It's {datetime.now().strftime('%I:%M %p')} right now",
                f"ขณะนี้เวลา {datetime.now().strftime('%H:%M น.')}"
            ],
            "capabilities": [
                "I can respond to basic commands, tell time, and maintain conversation while the main AI downloads.",
                "Currently available: basic conversation, time queries, system status, and voice responses.",
                "ตอนนี้ใช้งานได้: การสนทนาพื้นฐาน, ดูเวลา, ตรวจสอบระบบ, และตอบกลับด้วยเสียง"
            ],
            "help": [
                "Available commands: hello, time, status, help, capabilities. Say 'Hey JARVIS' to activate.",
                "You can ask about time, system status, or just chat while waiting for main AI.",
                "คำสั่งที่ใช้ได้: สวัสดี, เวลา, สถานะ, ช่วยเหลือ, ความสามารถ"
            ],
            "unknown": [
                "I'm operating in fallback mode. Please try a simpler question or wait for the main AI to be ready.",
                "Sorry, that's beyond my current capabilities. The main AI will be available soon.",
                "ขออภัยครับ ขณะนี้ยังใช้งานในโหมดสำรอง กรุณาลองใหม่เมื่อ AI หลักพร้อมแล้ว"
            ]
        }
        
        self.is_thai_pattern = [
            'สวัสดี', 'ครับ', 'ค่ะ', 'ขอบคุณ', 'เวลา', 'อะไร', 'ไง', 'ไหม',
            'เป็นยังไง', 'ทำไม', 'ที่ไหน', 'เมื่อไหร่', 'ใคร', 'อย่างไร'
        ]
        
    def is_thai_text(self, text: str) -> bool:
        """Check if text contains Thai characters"""
        return any(thai_word in text.lower() for thai_word in self.is_thai_pattern)
    
    def generate_response(self, text: str, context: Dict[str, Any] = None) -> str:
        """Generate appropriate fallback response"""
        text_lower = text.lower()
        
        try:
            # Thai language detection
            is_thai = self.is_thai_text(text)
            
            # Greeting responses
            if any(word in text_lower for word in ['hello', 'hi', 'hey', 'สวัสดี']):
                return random.choice(self.responses["thai_greeting" if is_thai else "greeting"])
            
            # Time queries
            elif any(word in text_lower for word in ['time', 'clock', 'เวลา', 'กี่โมง']):
                return random.choice(self.responses["time"])
            
            # Status queries
            elif any(word in text_lower for word in ['status', 'health', 'system', 'สถานะ', 'ระบบ']):
                return random.choice(self.responses["status"])
            
            # Capabilities
            elif any(word in text_lower for word in ['can you', 'what can', 'abilities', 'ความสามารถ', 'ทำอะไรได้']):
                return random.choice(self.responses["capabilities"])
            
            # Help
            elif any(word in text_lower for word in ['help', 'commands', 'ช่วย', 'คำสั่ง']):
                return random.choice(self.responses["help"])
            
            # Default unknown response
            else:
                return random.choice(self.responses["unknown"])
                
        except Exception as e:
            logger.error(f"Fallback AI error: {e}")
            return "JARVIS fallback system encountered an error. Please try again."
    
    def is_available(self) -> bool:
        """Always available as fallback"""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Return model information"""
        return {
            "name": self.name,
            "type": "fallback",
            "status": "ready",
            "capabilities": ["basic_conversation", "time_queries", "system_status"],
            "languages": ["en", "th"]
        }

def test_fallback_ai():
    """Test fallback AI system"""
    print("🧪 Testing Fallback AI...")
    
    ai = FallbackAI()
    
    test_inputs = [
        "Hello JARVIS",
        "What time is it?",
        "System status",
        "What can you do?",
        "สวัสดี JARVIS",
        "เวลาเท่าไหร่แล้ว",
        "ความสามารถของคุณ",
        "Random unknown query"
    ]
    
    for text in test_inputs:
        response = ai.generate_response(text)
        print(f"🔹 Input: {text}")
        print(f"🤖 Response: {response}")
        print()
    
    print("✅ Fallback AI test completed")

if __name__ == "__main__":
    test_fallback_ai()