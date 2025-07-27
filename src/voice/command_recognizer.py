#!/usr/bin/env python3
"""
🎤 Simple Voice Command Recognizer for JARVIS
ระบบรู้จำคำสั่งเสียงแบบง่าย
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import time


class CommandType(Enum):
    """ประเภทคำสั่ง"""
    SYSTEM = "system"
    INFORMATION = "information"
    CONTROL = "control"
    CONVERSATION = "conversation"


@dataclass
class Command:
    """คำสั่งเสียง"""
    id: str
    patterns: List[str]
    action: str
    description: str
    examples: List[str]
    command_type: CommandType = CommandType.INFORMATION


@dataclass
class CommandResult:
    """ผลการประมวลผลคำสั่ง"""
    command: Command
    confidence: float
    params: Dict[str, Any]
    response: str


class SimpleCommandRecognizer:
    """ระบบรู้จำคำสั่งแบบง่าย"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.commands = {}
        self._init_commands()
    
    def _init_commands(self):
        """เริ่มต้นคำสั่งพื้นฐาน"""
        commands = [
            Command(
                id="get_time",
                patterns=[
                    r"what.*time",
                    r"เวลา.*อะไร",
                    r"กี่โมง",
                    r"current time"
                ],
                action="get_current_time",
                description="Get current time",
                examples=["What time is it?", "กี่โมงแล้ว"],
                command_type=CommandType.INFORMATION
            ),
            Command(
                id="get_date",
                patterns=[
                    r"what.*date",
                    r"วันที่.*อะไร",
                    r"today.*date"
                ],
                action="get_current_date",
                description="Get current date",
                examples=["What date is it?", "วันนี้วันที่เท่าไหร่"],
                command_type=CommandType.INFORMATION
            ),
            Command(
                id="greeting",
                patterns=[
                    r"hello.*jarvis",
                    r"สวัสดี.*จาร์วิส",
                    r"hi jarvis",
                    r"good morning"
                ],
                action="respond_greeting",
                description="Respond to greetings",
                examples=["Hello JARVIS", "สวัสดี จาร์วิส"],
                command_type=CommandType.CONVERSATION
            ),
            Command(
                id="status",
                patterns=[
                    r"system.*status",
                    r"how.*you",
                    r"สถานะ.*ระบบ",
                    r"เป็นไง"
                ],
                action="get_status",
                description="Get system status",
                examples=["System status", "เป็นไงบ้าง"],
                command_type=CommandType.SYSTEM
            ),
            Command(
                id="help",
                patterns=[
                    r"help",
                    r"ช่วย",
                    r"what.*can.*do",
                    r"commands"
                ],
                action="show_help",
                description="Show help",
                examples=["Help", "ช่วยเหลือ"],
                command_type=CommandType.INFORMATION
            )
        ]
        
        for cmd in commands:
            self.commands[cmd.id] = cmd
        
        self.logger.info(f"✅ Initialized {len(self.commands)} commands")
    
    def recognize_command(self, text: str) -> Optional[CommandResult]:
        """รู้จำคำสั่ง"""
        text = text.lower().strip()
        
        best_match = None
        best_confidence = 0.0
        
        for cmd in self.commands.values():
            for pattern in cmd.patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    # Simple confidence calculation
                    confidence = len(pattern) / len(text) * 0.8 + 0.2
                    confidence = min(1.0, confidence)
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_match = cmd
        
        if best_match and best_confidence > 0.5:
            response = self._execute_command(best_match.action)
            
            return CommandResult(
                command=best_match,
                confidence=best_confidence,
                params={},
                response=response
            )
        
        return None
    
    def _execute_command(self, action: str) -> str:
        """ประมวลผลคำสั่ง"""
        if action == "get_current_time":
            import datetime
            now = datetime.datetime.now()
            return f"The current time is {now.strftime('%I:%M %p')}"
        
        elif action == "get_current_date":
            import datetime
            now = datetime.datetime.now()
            return f"Today is {now.strftime('%A, %B %d, %Y')}"
        
        elif action == "respond_greeting":
            import random
            greetings = [
                "Hello! How can I help you?",
                "สวัสดีครับ มีอะไรให้ช่วยไหมครับ",
                "Good day! What can I do for you?",
                "ยินดีต้อนรับครับ"
            ]
            return random.choice(greetings)
        
        elif action == "get_status":
            return "All systems operational. JARVIS is ready to assist."
        
        elif action == "show_help":
            commands = [
                "Ask for time: 'What time is it?'",
                "Ask for date: 'What date is it?'", 
                "Greet me: 'Hello JARVIS'",
                "System status: 'How are you?'",
                "ถามเวลา: 'กี่โมงแล้ว'",
                "ทักทาย: 'สวัสดี จาร์วิส'"
            ]
            return "Available commands:\n" + "\n".join(commands)
        
        else:
            return "Command recognized but not implemented yet."
    
    def get_available_commands(self) -> List[Dict[str, Any]]:
        """ดึงรายการคำสั่ง"""
        return [
            {
                'id': cmd.id,
                'description': cmd.description,
                'examples': cmd.examples,
                'type': cmd.command_type.value
            }
            for cmd in self.commands.values()
        ]


def test_command_recognizer():
    """ทดสอบระบบรู้จำคำสั่ง"""
    print("🧪 Testing Simple Command Recognizer...")
    
    recognizer = SimpleCommandRecognizer()
    
    test_commands = [
        "What time is it?",
        "กี่โมงแล้ว",
        "Hello JARVIS",
        "สวัสดี จาร์วิส", 
        "What date is today?",
        "System status",
        "Help me",
        "This is not a command"
    ]
    
    for text in test_commands:
        print(f"\n🎤 Testing: '{text}'")
        result = recognizer.recognize_command(text)
        
        if result:
            print(f"   ✅ Command: {result.command.id}")
            print(f"   🎯 Confidence: {result.confidence:.2f}")
            print(f"   💬 Response: {result.response}")
        else:
            print(f"   ❌ No command recognized")
    
    # Show available commands
    print(f"\n📝 Available Commands:")
    commands = recognizer.get_available_commands()
    for cmd in commands:
        print(f"   - {cmd['id']}: {cmd['description']}")
    
    print("\n✅ Command recognizer test completed!")
    return recognizer


if __name__ == "__main__":
    test_command_recognizer()