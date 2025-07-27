#\!/usr/bin/env python3
"""
🎤 Advanced Voice Command System for JARVIS
ระบบรู้จำคำสั่งเสียงขั้นสูงภาษาไทย-อังกฤษ
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import time
import threading


class CommandType(Enum):
    """ประเภทคำสั่ง"""
    SYSTEM = "system"           # คำสั่งระบบ
    SEARCH = "search"           # ค้นหาข้อมูล
    CONTROL = "control"         # ควบคุมอุปกรณ์
    INFORMATION = "information" # ขอข้อมูล
    CONVERSATION = "conversation" # สนทนา
    ACTION = "action"           # ทำงาน
    QUESTION = "question"       # ถามคำถาม


class CommandPriority(Enum):
    """ลำดับความสำคัญคำสั่ง"""
    CRITICAL = "critical"   # ฉุกเฉิน
    HIGH = "high"          # สูง
    MEDIUM = "medium"      # ปานกลาง
    LOW = "low"            # ต่ำ


@dataclass
class VoiceCommand:
    """คำสั่งเสียง"""
    command_id: str
    patterns: List[str]           # รูปแบบการพูด
    command_type: CommandType
    priority: CommandPriority
    action: str                   # ฟังก์ชันที่จะเรียก
    parameters: Dict[str, Any]    # พารามิเตอร์
    description: str
    examples: List[str]
    languages: List[str] = None   # ['en', 'th']
    enabled: bool = True
    confidence_threshold: float = 0.7


@dataclass
class CommandMatch:
    """ผลการจับคู่คำสั่ง"""
    command: VoiceCommand
    confidence: float
    matched_text: str
    extracted_params: Dict[str, Any]
    processing_time: float


class AdvancedCommandSystem:
    """ระบบคำสั่งเสียงขั้นสูง"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Command storage
        self.commands: Dict[str, VoiceCommand] = {}
        self.command_handlers: Dict[str, Callable] = {}
        
        # Processing settings
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.max_processing_time = self.config.get('max_processing_time', 5.0)
        self.fuzzy_matching = self.config.get('fuzzy_matching', True)
        
        # Statistics
        self.stats = {
            'total_commands': 0,
            'successful_matches': 0,
            'failed_matches': 0,
            'average_confidence': 0.0,
            'commands_by_type': {}
        }
        
        # Initialize built-in commands
        self._initialize_builtin_commands()
        
        self.logger.info("🎤 Advanced Command System initialized")
    
    def _initialize_builtin_commands(self):
        """เริ่มต้นคำสั่งในตัว"""
        
        # System commands
        self.register_command(VoiceCommand(
            command_id="system_shutdown",
            patterns=[
                r"(shutdown|turn off|power off|stop) (jarvis|system)",
                r"(ปิด|หยุด|ดับ) (จาร์วิส|ระบบ)",
                r"goodbye jarvis",
                r"ลาก่อน จาร์วิส"
            ],
            command_type=CommandType.SYSTEM,
            priority=CommandPriority.HIGH,
            action="shutdown_system",
            parameters={},
            description="Shutdown JARVIS system",
            examples=["Shutdown JARVIS", "ปิดระบบจาร์วิส"],
            languages=["en", "th"]
        ))
        
        self.register_command(VoiceCommand(
            command_id="get_time",
            patterns=[
                r"what.*(time|clock)",
                r"(tell me|show me|what's) the time",
                r"(เวลา|นาฬิกา).*(อะไร|เท่าไหร่|กี่โมง)",
                r"(บอก|แสดง).*(เวลา|นาฬิกา)",
                r"ตอนนี้กี่โมง"
            ],
            command_type=CommandType.INFORMATION,
            priority=CommandPriority.LOW,
            action="get_current_time",
            parameters={},
            description="Get current time",
            examples=["What time is it?", "ตอนนี้กี่โมงแล้ว"],
            languages=["en", "th"]
        ))
        
        # Date command
        self.register_command(VoiceCommand(
            command_id="get_date",
            patterns=[
                r"what.*(date|day)",
                r"(tell me|show me|what's) (the date|today)",
                r"(วันที่|วัน).*(อะไร|เท่าไหร่|ไหน)",
                r"(บอก|แสดง).*(วันที่|วัน)",
                r"วันนี้วันที่เท่าไหร่"
            ],
            command_type=CommandType.INFORMATION,
            priority=CommandPriority.LOW,
            action="get_current_date",
            parameters={},
            description="Get current date",
            examples=["What date is it?", "วันนี้วันที่เท่าไหร่"],
            languages=["en", "th"]
        ))
        
        # Greeting command
        self.register_command(VoiceCommand(
            command_id="greeting",
            patterns=[
                r"(hello|hi|hey|good morning|good afternoon|good evening)",
                r"how are you",
                r"(สวัสดี|หวัดดี|ดีครับ|ดีค่ะ)",
                r"(สบายดี|เป็นไง|เป็นอย่างไร)"
            ],
            command_type=CommandType.CONVERSATION,
            priority=CommandPriority.MEDIUM,
            action="handle_greeting",
            parameters={},
            description="Handle greetings",
            examples=["Hello", "สวัสดี", "How are you?", "สบายดีไหม"],
            languages=["en", "th"]
        ))
        
        # Information about JARVIS
        self.register_command(VoiceCommand(
            command_id="self_info",
            patterns=[
                r"(who are you|what are you|tell me about yourself)",
                r"what.*(name|called)",
                r"(คุณคือใคร|เป็นใคร|ชื่ออะไร)",
                r"(บอกเกี่ยวกับ|เล่าเกี่ยวกับ).*(ตัวเอง|คุณ)"
            ],
            command_type=CommandType.INFORMATION,
            priority=CommandPriority.MEDIUM,
            action="provide_self_info",
            parameters={},
            description="Provide information about JARVIS",
            examples=["Who are you?", "คุณคือใคร", "What's your name?", "ชื่ออะไร"],
            languages=["en", "th"]
        ))
        
        # Help command
        self.register_command(VoiceCommand(
            command_id="help",
            patterns=[
                r"(help|what can you do|commands)",
                r"(ช่วย|ช่วยเหลือ|ทำอะไรได้|คำสั่ง)"
            ],
            command_type=CommandType.SYSTEM,
            priority=CommandPriority.MEDIUM,
            action="show_help",
            parameters={},
            description="Show available commands",
            examples=["Help", "ช่วยเหลือ", "What can you do?", "ทำอะไรได้บ้าง"],
            languages=["en", "th"]
        ))
        
        # Status command
        self.register_command(VoiceCommand(
            command_id="status",
            patterns=[
                r"(status|how are you doing|system status)",
                r"(สถานะ|เป็นยังไงบ้าง|สถานะระบบ)"
            ],
            command_type=CommandType.SYSTEM,
            priority=CommandPriority.MEDIUM,
            action="show_status",
            parameters={},
            description="Show system status",
            examples=["Status", "สถานะระบบ", "How are you doing?", "เป็นยังไงบ้าง"],
            languages=["en", "th"]
        ))
        
        self.logger.info(f"✅ Initialized {len(self.commands)} built-in commands")
    
    def register_command(self, command: VoiceCommand):
        """ลงทะเบียนคำสั่งใหม่"""
        self.commands[command.command_id] = command
        self.logger.debug(f"📝 Registered command: {command.command_id}")
    
    def process_voice_input(self, text: str, language: str = 'en') -> Optional[CommandMatch]:
        """ประมวลผลเสียงที่เข้ามา"""
        start_time = time.time()
        self.logger.info(f"🎤 Processing: '{text}' ({language})")
        
        # Clean input text
        text_clean = self._clean_input_text(text)
        
        best_match = None
        best_confidence = 0.0
        
        # Try to match each command
        for command_id, command in self.commands.items():
            if not command.enabled:
                continue
            
            # Check if command supports the language
            if command.languages and language not in command.languages:
                continue
            
            confidence = self._calculate_match_confidence(text_clean, command, language)
            
            if confidence > command.confidence_threshold and confidence > best_confidence:
                best_confidence = confidence
                best_match = CommandMatch(
                    command=command,
                    confidence=confidence,
                    matched_text=text,
                    extracted_params=self._extract_parameters(text_clean, command),
                    processing_time=time.time() - start_time
                )
        
        # Update statistics
        self.stats['total_commands'] += 1
        if best_match:
            self.stats['successful_matches'] += 1
            cmd_type = best_match.command.command_type.value
            self.stats['commands_by_type'][cmd_type] = self.stats['commands_by_type'].get(cmd_type, 0) + 1
        else:
            self.stats['failed_matches'] += 1
        
        return best_match
    
    def _clean_input_text(self, text: str) -> str:
        """ทำความสะอาดข้อความ"""
        # Remove wake words
        wake_words = ['hey jarvis', 'hi jarvis', 'jarvis', 'เฮ้ จาร์วิส', 'สวัสดี จาร์วิส', 'จาร์วิส']
        
        text_lower = text.lower()
        for wake_word in wake_words:
            text_lower = text_lower.replace(wake_word, '').strip()
        
        # Remove extra spaces
        text_lower = re.sub(r'\s+', ' ', text_lower).strip()
        return text_lower
    
    def _calculate_match_confidence(self, text: str, command: VoiceCommand, language: str) -> float:
        """คำนวณความเชื่อมั่นการจับคู่"""
        total_confidence = 0.0
        pattern_matches = 0
        
        for pattern in command.patterns:
            # Use regex matching
            if re.search(pattern, text, re.IGNORECASE):
                pattern_matches += 1
                total_confidence += 0.8
        
        # Keyword matching
        keywords = self._extract_keywords_from_patterns(command.patterns)
        text_words = text.lower().split()
        keyword_matches = sum(1 for keyword in keywords if keyword.lower() in text_words)
        
        if keyword_matches > 0:
            total_confidence += (keyword_matches / len(keywords)) * 0.4
        
        # Normalize confidence
        if pattern_matches > 0:
            return min(total_confidence / max(1, pattern_matches), 1.0)
        elif keyword_matches > 0:
            return min(total_confidence, 1.0)
        
        return 0.0
    
    def _extract_keywords_from_patterns(self, patterns: List[str]) -> List[str]:
        """สกัดคีย์เวิร์ดจากรูปแบบ"""
        keywords = []
        for pattern in patterns:
            # Simple keyword extraction (remove regex special chars)
            clean_pattern = re.sub(r'[^\w\sก-๙]', ' ', pattern)
            words = clean_pattern.split()
            keywords.extend([word for word in words if len(word) > 2])
        return list(set(keywords))
    
    def _extract_parameters(self, text: str, command: VoiceCommand) -> Dict[str, Any]:
        """สกัดพารามิเตอร์จากข้อความ"""
        # Basic parameter extraction - can be enhanced later
        return {
            'original_text': text,
            'word_count': len(text.split()),
            'detected_numbers': re.findall(r'\d+', text),
            'detected_times': re.findall(r'\d{1,2}:\d{2}', text)
        }
    
    def execute_command(self, command_match: CommandMatch) -> Dict[str, Any]:
        """ประมวลผลคำสั่ง"""
        action = command_match.command.action
        
        if action == "get_current_time":
            import datetime
            now = datetime.datetime.now()
            
            # Detect language for appropriate response
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                response = f"ตอนนี้เวลา {now.strftime('%H:%M')} น. ครับ"
            else:
                response = f"The current time is {now.strftime('%I:%M %p')}"
            
            return {
                'success': True,
                'result': {
                    'response': response,
                    'time': now.isoformat()
                }
            }
        
        elif action == "get_current_date":
            import datetime
            now = datetime.datetime.now()
            
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                weekdays_th = ['จันทร์', 'อังคาร', 'พุธ', 'พฤหัสบดี', 'ศุกร์', 'เสาร์', 'อาทิตย์']
                weekday_th = weekdays_th[now.weekday()]
                response = f"วันนี้วัน{weekday_th} ที่ {now.strftime('%d/%m/%Y')} ครับ"
            else:
                response = f"Today is {now.strftime('%A, %B %d, %Y')}"
            
            return {
                'success': True,
                'result': {
                    'response': response,
                    'date': now.date().isoformat()
                }
            }
        
        elif action == "handle_greeting":
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                responses = [
                    "สวัสดีครับ! ผมคือ JARVIS พร้อมรับใช้แล้วครับ",
                    "ดีครับ! ผมสบายดีมาก มีอะไรให้ช่วยไหมครับ",
                    "หวัดดีครับ! JARVIS พร้อมให้บริการครับ"
                ]
            else:
                responses = [
                    "Hello! I'm JARVIS, your AI assistant. How may I help you?",
                    "Hi there! I'm doing great. What can I do for you today?",
                    "Good day! JARVIS at your service."
                ]
            
            import random
            response = random.choice(responses)
            
            return {
                'success': True,
                'result': {
                    'response': response
                }
            }
        
        elif action == "provide_self_info":
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                response = """ผมคือ JARVIS (Just A Rather Very Intelligent System) 
เป็น AI Assistant ที่ออกแบบมาเพื่อช่วยเหลือคุณครับ 
ผมสามารถตอบคำถาม ให้ข้อมูล และสนทนาได้ทั้งภาษาไทยและอังกฤษ 
มีอะไรให้ช่วยไหมครับ?"""
            else:
                response = """I'm JARVIS - Just A Rather Very Intelligent System.
I'm an AI Assistant designed to help you with various tasks.
I can answer questions, provide information, and converse in both Thai and English.
How may I assist you today?"""
            
            return {
                'success': True,
                'result': {
                    'response': response
                }
            }
        
        elif action == "show_help":
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                response = """คำสั่งที่ใช้ได้:
🎙️ ทักทาย: "สวัสดี", "สบายดีไหม"
⏰ เวลา: "เวลาเท่าไหร่", "กี่โมงแล้ว"
📅 วันที่: "วันนี้วันที่เท่าไหร่", "วันอะไร"
ℹ️ ข้อมูล: "คุณคือใคร", "ชื่ออะไร"
⚙️ ระบบ: "สถานะระบบ", "ช่วยเหลือ"
🔚 ปิดระบบ: "ปิดจาร์วิส", "ลาก่อน"""
            else:
                response = """Available Commands:
🎙️ Greeting: "Hello", "How are you"
⏰ Time: "What time is it"
📅 Date: "What date is it", "What day is it"
ℹ️ Information: "Who are you", "What's your name"
⚙️ System: "Status", "Help"
🔚 Shutdown: "Shutdown JARVIS", "Goodbye"""
            
            return {
                'success': True,
                'result': {
                    'response': response
                }
            }
        
        elif action == "show_status":
            stats = self.get_statistics()
            
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                response = f"""สถานะระบบ JARVIS:
✅ พร้อมใช้งาน
📊 คำสั่งทั้งหมด: {stats['total_commands']}
✅ สำเร็จ: {stats['successful_matches']}
❌ ล้มเหลว: {stats['failed_matches']}
🎯 อัตราความสำเร็จ: {(stats['successful_matches']/max(1,stats['total_commands'])*100):.1f}%
📝 คำสั่งที่มี: {len(self.commands)} คำสั่ง"""
            else:
                response = f"""JARVIS System Status:
✅ Online and Ready
📊 Total Commands: {stats['total_commands']}
✅ Successful: {stats['successful_matches']}
❌ Failed: {stats['failed_matches']}
🎯 Success Rate: {(stats['successful_matches']/max(1,stats['total_commands'])*100):.1f}%
📝 Available Commands: {len(self.commands)}"""
            
            return {
                'success': True,
                'result': {
                    'response': response,
                    'statistics': stats
                }
            }
        
        elif action == "shutdown_system":
            if any(thai_char in command_match.matched_text for thai_char in 'กข้เาไ'):
                response = "ลาก่อนครับ! กำลังปิดระบบ..."
            else:
                response = "Goodbye! Shutting down system..."
            
            return {
                'success': True,
                'result': {
                    'response': response,
                    'action': 'shutdown'
                }
            }
        
        return {'success': False, 'error': f'Command handler not implemented: {action}'}
    
    def get_statistics(self) -> Dict[str, Any]:
        """ดึงสถิติการใช้งาน"""
        return self.stats.copy()


def test_advanced_command_system():
    """ทดสอบระบบคำสั่งเสียงขั้นสูง"""
    print("🧪 Testing Advanced Voice Command System...")
    
    cmd_system = AdvancedCommandSystem()
    
    test_inputs = [
        ("What time is it?", "en"),
        ("ตอนนี้กี่โมงแล้ว", "th"),
        ("What date is it?", "en"),
        ("วันนี้วันที่เท่าไหร่", "th"),
        ("Hello JARVIS", "en"),
        ("สวัสดี", "th"),
        ("Who are you?", "en"),
        ("คุณคือใคร", "th"),
        ("Help", "en"),
        ("ช่วยเหลือ", "th"),
        ("Status", "en"),
        ("สถานะระบบ", "th"),
        ("Unknown command", "en")
    ]
    
    for text, language in test_inputs:
        print(f"\n🎤 Input: '{text}' ({language})")
        
        match = cmd_system.process_voice_input(text, language)
        
        if match:
            print(f"   ✅ Matched: {match.command.command_id}")
            result = cmd_system.execute_command(match)
            if result.get('success'):
                response = result.get('result', {}).get('response', 'No response')
                print(f"   💬 Response: {response}")
        else:
            print(f"   ❌ No command matched")
    
    print("\n✅ Advanced Command System test completed!")


if __name__ == "__main__":
    test_advanced_command_system()
