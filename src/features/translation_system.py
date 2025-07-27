"""
Translation System for Jarvis Voice Assistant
Handles text translation, language detection, and multilingual support
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import requests
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer


@dataclass
class TranslationResult:
    """Translation result structure"""
    original_text: str
    translated_text: str
    source_language: str
    target_language: str
    confidence: float
    method: str
    timestamp: datetime
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TranslationResult':
        """Create from dictionary"""
        return cls(**data)


class LanguageDetector:
    """Language detection utility"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Common language patterns
        self.language_patterns = {
            'th': ['ก', 'ข', 'ค', 'ง', 'จ', 'ฉ', 'ช', 'ซ', 'ฌ', 'ญ', 'ฎ', 'ฏ', 'ฐ', 'ฑ', 'ฒ', 'ณ', 
                   'ด', 'ต', 'ถ', 'ท', 'ธ', 'น', 'บ', 'ป', 'ผ', 'ฝ', 'พ', 'ฟ', 'ภ', 'ม', 'ย', 'ร', 
                   'ล', 'ว', 'ศ', 'ษ', 'ส', 'ห', 'ฬ', 'อ', 'ฮ', 'ะ', 'ั', 'า', 'ำ', 'ิ', 'ี', 'ึ', 'ื', 
                   'ุ', 'ู', 'เ', 'แ', 'โ', 'ใ', 'ไ', '่', '้', '๊', '๋', '์', 'ํ', '๎'],
            'en': ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
                   'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'],
            'zh': ['一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '的', '是', '在', '了', '不', '和', '有', '大', '这', '主', '我', '们', '他', '她', '它'],
            'ja': ['あ', 'い', 'う', 'え', 'お', 'か', 'き', 'く', 'け', 'こ', 'が', 'ぎ', 'ぐ', 'げ', 'ご', 'さ', 'し', 'す', 'せ', 'そ', 'ざ', 'じ', 'ず', 'ぜ', 'ぞ', 'た', 'ち', 'つ', 'て', 'と', 'だ', 'ぢ', 'づ', 'で', 'ど', 'な', 'に', 'ぬ', 'ね', 'の', 'は', 'ひ', 'ふ', 'へ', 'ほ', 'ば', 'び', 'ぶ', 'べ', 'ぼ', 'ぱ', 'ぴ', 'ぷ', 'ぺ', 'ぽ', 'ま', 'み', 'む', 'め', 'も', 'や', 'ゆ', 'よ', 'ら', 'り', 'る', 'れ', 'ろ', 'わ', 'ゐ', 'ゑ', 'を', 'ん', 'ア', 'イ', 'ウ', 'エ', 'オ', 'カ', 'キ', 'ク', 'ケ', 'コ', 'ガ', 'ギ', 'グ', 'ゲ', 'ゴ', 'サ', 'シ', 'ス', 'セ', 'ソ', 'ザ', 'ジ', 'ズ', 'ゼ', 'ゾ', 'タ', 'チ', 'ツ', 'テ', 'ト', 'ダ', 'ヂ', 'ヅ', 'デ', 'ド', 'ナ', 'ニ', 'ヌ', 'ネ', 'ノ', 'ハ', 'ヒ', 'フ', 'ヘ', 'ホ', 'バ', 'ビ', 'ブ', 'ベ', 'ボ', 'パ', 'ピ', 'プ', 'ペ', 'ポ', 'マ', 'ミ', 'ム', 'メ', 'モ', 'ヤ', 'ユ', 'ヨ', 'ラ', 'リ', 'ル', 'レ', 'ロ', 'ワ', 'ヰ', 'ヱ', 'ヲ', 'ン'],
            'ko': ['가', '나', '다', '라', '마', '바', '사', '아', '자', '차', '카', '타', '파', '하', '의', '이', '그', '것', '들', '수', '있', '없', '하', '되', '같', '또', '만', '더', '크', '작', '좋', '나쁘', '새', '옛', '높', '낮', '많', '적', '빠', '느린']
        }
    
    def detect_language(self, text: str) -> Tuple[str, float]:
        """Detect language with confidence score"""
        if not text or not text.strip():
            return "unknown", 0.0
        
        text_lower = text.lower()
        language_scores = {}
        
        # Count characters for each language
        for lang, patterns in self.language_patterns.items():
            score = 0
            for char in text_lower:
                if char in patterns:
                    score += 1
            
            if len(text) > 0:
                language_scores[lang] = score / len(text)
        
        # Get highest scoring language
        if language_scores:
            best_lang = max(language_scores, key=language_scores.get)
            confidence = language_scores[best_lang]
            
            # Minimum confidence threshold
            if confidence > 0.1:
                return best_lang, confidence
        
        # Default to English if no strong match
        return "en", 0.3


class LocalTranslator:
    """Local translation implementation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.language_detector = LanguageDetector()
        
        # Translation dictionaries for common phrases
        self.translation_dict = {
            'th_to_en': {
                'สวัสดี': 'Hello',
                'ขอบคุณ': 'Thank you',
                'ขอโทษ': 'Sorry',
                'ดีครับ': 'Good',
                'ดีค่ะ': 'Good',
                'ไม่เป็นไร': 'No problem',
                'ไม่ใช่': 'No',
                'ใช่': 'Yes',
                'ช่วยเหลือ': 'Help',
                'ข้อมูล': 'Information',
                'ข่าวสาร': 'News',
                'แปลภาษา': 'Translate',
                'เรียนรู้': 'Learning',
                'คำถาม': 'Question',
                'รูปภาพ': 'Image',
                'เสียง': 'Voice',
                'พูด': 'Speak',
                'ฟัง': 'Listen',
                'อ่าน': 'Read',
                'เขียน': 'Write',
                'คิด': 'Think',
                'เข้าใจ': 'Understand',
                'ไม่เข้าใจ': 'Don\'t understand',
                'เทคโนโลยี': 'Technology',
                'ปัญญาประดิษฐ์': 'Artificial Intelligence',
                'คอมพิวเตอร์': 'Computer',
                'โปรแกรม': 'Program',
                'ระบบ': 'System',
                'ข้อมูล': 'Data',
                'วิเคราะห์': 'Analysis',
                'ประมวลผล': 'Processing',
                'ผลลัพธ์': 'Result',
                'ความสำเร็จ': 'Success',
                'ข้อผิดพลาด': 'Error',
                'ปัญหา': 'Problem',
                'แก้ไข': 'Fix',
                'ปรับปรุง': 'Improve',
                'พัฒนา': 'Develop',
                'สร้าง': 'Create',
                'ทำ': 'Do',
                'ใช้': 'Use',
                'ทดสอบ': 'Test',
                'ตรวจสอบ': 'Check',
                'ค้นหา': 'Search',
                'พบ': 'Find',
                'ได้': 'Can/Get',
                'เป็น': 'Is/Are',
                'มี': 'Have',
                'ไป': 'Go',
                'มา': 'Come',
                'เอา': 'Take',
                'ให้': 'Give',
                'รับ': 'Receive',
                'ส่ง': 'Send',
                'บอก': 'Tell',
                'พูด': 'Say',
                'ถาม': 'Ask',
                'ตอบ': 'Answer',
                'รู้': 'Know',
                'จำ': 'Remember',
                'ลืม': 'Forget',
                'เห็น': 'See',
                'ดู': 'Look',
                'ฟัง': 'Listen',
                'ได้ยิน': 'Hear',
                'กิน': 'Eat',
                'ดื่ม': 'Drink',
                'นอน': 'Sleep',
                'ตื่น': 'Wake up',
                'ทำงาน': 'Work',
                'เรียน': 'Study',
                'เล่น': 'Play',
                'วิ่ง': 'Run',
                'เดิน': 'Walk',
                'นั่ง': 'Sit',
                'ยืน': 'Stand',
                'เวลา': 'Time',
                'วัน': 'Day',
                'คืน': 'Night',
                'เช้า': 'Morning',
                'บ่าย': 'Afternoon',
                'เย็น': 'Evening',
                'วันนี้': 'Today',
                'เมื่อวาน': 'Yesterday',
                'พรุ่งนี้': 'Tomorrow',
                'สัปดาห์': 'Week',
                'เดือน': 'Month',
                'ปี': 'Year',
                'ตอนนี้': 'Now',
                'ก่อน': 'Before',
                'หลัง': 'After',
                'เร็ว': 'Fast',
                'ช้า': 'Slow',
                'ใหม่': 'New',
                'เก่า': 'Old',
                'ใหญ่': 'Big',
                'เล็ก': 'Small',
                'สูง': 'High',
                'ต่ำ': 'Low',
                'ดี': 'Good',
                'เลว': 'Bad',
                'ง่าย': 'Easy',
                'ยาก': 'Difficult',
                'สำคัญ': 'Important',
                'ไม่สำคัญ': 'Not important',
                'จริง': 'True',
                'เท็จ': 'False',
                'แน่นอน': 'Sure',
                'ไม่แน่ใจ': 'Not sure',
                'อาจจะ': 'Maybe',
                'ชัดเจน': 'Clear',
                'ไม่ชัดเจน': 'Not clear',
                'เสร็จ': 'Finished',
                'ไม่เสร็จ': 'Not finished',
                'เริ่ม': 'Start',
                'จบ': 'End',
                'ต่อ': 'Continue',
                'หยุด': 'Stop',
                'รอ': 'Wait',
                'เร่ง': 'Hurry',
                'ช่วย': 'Help',
                'ขอ': 'Request',
                'ได้โปรด': 'Please',
                'ขอบคุณมาก': 'Thank you very much',
                'ยินดี': 'Welcome',
                'ขอโทษมาก': 'Very sorry',
                'ไม่เป็นไรค่ะ': 'It\'s okay',
                'ไม่เป็นไรครับ': 'It\'s okay',
                'ด้วย': 'With',
                'โดย': 'By',
                'จาก': 'From',
                'ถึง': 'To',
                'ใน': 'In',
                'นอก': 'Out',
                'บน': 'On',
                'ล่าง': 'Below',
                'ข้าง': 'Side',
                'หน้า': 'Front',
                'หลัง': 'Back',
                'ซ้าย': 'Left',
                'ขวา': 'Right',
                'กลาง': 'Middle',
                'ที่': 'At/That',
                'นี่': 'This',
                'นั่น': 'That',
                'อัน': 'One (classifier)',
                'คน': 'Person',
                'สิ่ง': 'Thing',
                'เรื่อง': 'Matter',
                'ทาง': 'Way',
                'วิธี': 'Method',
                'แบบ': 'Style',
                'ประเภท': 'Type',
                'ชนิด': 'Kind',
                'รูปแบบ': 'Format',
                'ตัวอย่าง': 'Example',
                'เหตุผล': 'Reason',
                'ผล': 'Result',
                'เหตุ': 'Cause',
                'ผลก': 'Effect',
                'ความ': 'Quality/State',
                'การ': 'Action',
                'บาง': 'Some',
                'ทั้ง': 'All',
                'หลาย': 'Many',
                'น้อย': 'Few',
                'เพียง': 'Only',
                'แค่': 'Just',
                'เท่านั้น': 'Only',
                'และ': 'And',
                'หรือ': 'Or',
                'แต่': 'But',
                'อย่างไรก็ตาม': 'However',
                'ดังนั้น': 'Therefore',
                'เพราะ': 'Because',
                'ถ้า': 'If',
                'เมื่อ': 'When',
                'ตั้งแต่': 'Since',
                'จนกว่า': 'Until',
                'ในขณะที่': 'While',
                'ก่อนที่': 'Before',
                'หลังจาก': 'After',
                'เพื่อ': 'To/For',
                'เพื่อให้': 'In order to',
                'ตาม': 'According to',
                'อย่าง': 'Like',
                'เช่น': 'Such as',
                'รวมทั้ง': 'Including',
                'ยกเว้น': 'Except',
                'นอกจาก': 'Besides',
                'เกี่ยวกับ': 'About',
                'เรื่อง': 'About',
                'ด้วยเหตุนี้': 'For this reason',
                'ด้วยเหตุนั้น': 'For that reason',
                'อย่างนั้น': 'Like that',
                'อย่างนี้': 'Like this',
                'ดังนั้น': 'So',
                'ดังนี้': 'As follows',
                'ดังกล่าว': 'As mentioned',
                'ดังที่': 'As',
                'ตามที่': 'As',
                'ซึ่ง': 'Which',
                'ที่': 'That/Which',
                'อะไร': 'What',
                'ใคร': 'Who',
                'ที่ไหน': 'Where',
                'เมื่อไร': 'When',
                'ทำไม': 'Why',
                'อย่างไร': 'How',
                'เท่าไร': 'How much',
                'กี่': 'How many',
                'กี่โมง': 'What time',
                'เวลาไหน': 'What time',
                'ไหน': 'Which',
                'ไหม': 'Question particle',
                'หรือไม่': 'Or not',
                'หรือเปล่า': 'Or not',
                'มั้ย': 'Question particle',
                'ครับ': 'Polite particle (male)',
                'ค่ะ': 'Polite particle (female)',
                'คะ': 'Polite particle (female)',
                'จ้า': 'Polite particle',
                'นะ': 'Particle',
                'ล่ะ': 'Particle',
                'เลย': 'Particle',
                'เลยค่ะ': 'Particle',
                'เลยครับ': 'Particle',
                'ค่ะ': 'Yes (female)',
                'ครับ': 'Yes (male)',
                'ใช่ค่ะ': 'Yes (female)',
                'ใช่ครับ': 'Yes (male)',
                'ไม่ค่ะ': 'No (female)',
                'ไม่ครับ': 'No (male)',
                'ไม่ใช่ค่ะ': 'No (female)',
                'ไม่ใช่ครับ': 'No (male)'
            },
            'en_to_th': {
                'hello': 'สวัสดี',
                'hi': 'สวัสดี',
                'thank you': 'ขอบคุณ',
                'thanks': 'ขอบคุณ',
                'sorry': 'ขอโทษ',
                'excuse me': 'ขอโทษ',
                'good': 'ดี',
                'bad': 'เลว',
                'yes': 'ใช่',
                'no': 'ไม่ใช่',
                'help': 'ช่วยเหลือ',
                'information': 'ข้อมูล',
                'news': 'ข่าวสาร',
                'translate': 'แปลภาษา',
                'learning': 'เรียนรู้',
                'question': 'คำถาม',
                'image': 'รูปภาพ',
                'voice': 'เสียง',
                'speak': 'พูด',
                'listen': 'ฟัง',
                'read': 'อ่าน',
                'write': 'เขียน',
                'think': 'คิด',
                'understand': 'เข้าใจ',
                'technology': 'เทคโนโลยี',
                'artificial intelligence': 'ปัญญาประดิษฐ์',
                'computer': 'คอมพิวเตอร์',
                'program': 'โปรแกรม',
                'system': 'ระบบ',
                'data': 'ข้อมูล',
                'analysis': 'วิเคราะห์',
                'processing': 'ประมวลผล',
                'result': 'ผลลัพธ์',
                'success': 'ความสำเร็จ',
                'error': 'ข้อผิดพลาด',
                'problem': 'ปัญหา',
                'fix': 'แก้ไข',
                'improve': 'ปรับปรุง',
                'develop': 'พัฒนา',
                'create': 'สร้าง',
                'do': 'ทำ',
                'use': 'ใช้',
                'test': 'ทดสอบ',
                'check': 'ตรวจสอบ',
                'search': 'ค้นหา',
                'find': 'พบ',
                'can': 'ได้',
                'get': 'ได้',
                'is': 'เป็น',
                'are': 'เป็น',
                'have': 'มี',
                'go': 'ไป',
                'come': 'มา',
                'take': 'เอา',
                'give': 'ให้',
                'receive': 'รับ',
                'send': 'ส่ง',
                'tell': 'บอก',
                'say': 'พูด',
                'ask': 'ถาม',
                'answer': 'ตอบ',
                'know': 'รู้',
                'remember': 'จำ',
                'forget': 'ลืม',
                'see': 'เห็น',
                'look': 'ดู',
                'hear': 'ได้ยิน',
                'eat': 'กิน',
                'drink': 'ดื่ม',
                'sleep': 'นอน',
                'wake up': 'ตื่น',
                'work': 'ทำงาน',
                'study': 'เรียน',
                'play': 'เล่น',
                'run': 'วิ่ง',
                'walk': 'เดิน',
                'sit': 'นั่ง',
                'stand': 'ยืน',
                'time': 'เวลา',
                'day': 'วัน',
                'night': 'คืน',
                'morning': 'เช้า',
                'afternoon': 'บ่าย',
                'evening': 'เย็น',
                'today': 'วันนี้',
                'yesterday': 'เมื่อวาน',
                'tomorrow': 'พรุ่งนี้',
                'week': 'สัปดาห์',
                'month': 'เดือน',
                'year': 'ปี',
                'now': 'ตอนนี้',
                'before': 'ก่อน',
                'after': 'หลัง',
                'fast': 'เร็ว',
                'slow': 'ช้า',
                'new': 'ใหม่',
                'old': 'เก่า',
                'big': 'ใหญ่',
                'small': 'เล็ก',
                'high': 'สูง',
                'low': 'ต่ำ',
                'easy': 'ง่าย',
                'difficult': 'ยาก',
                'important': 'สำคัญ',
                'true': 'จริง',
                'false': 'เท็จ',
                'sure': 'แน่นอน',
                'maybe': 'อาจจะ',
                'clear': 'ชัดเจน',
                'finished': 'เสร็จ',
                'start': 'เริ่ม',
                'end': 'จบ',
                'continue': 'ต่อ',
                'stop': 'หยุด',
                'wait': 'รอ',
                'hurry': 'เร่ง',
                'please': 'ได้โปรด',
                'welcome': 'ยินดี',
                'with': 'ด้วย',
                'by': 'โดย',
                'from': 'จาก',
                'to': 'ถึง',
                'in': 'ใน',
                'out': 'นอก',
                'on': 'บน',
                'below': 'ล่าง',
                'side': 'ข้าง',
                'front': 'หน้า',
                'back': 'หลัง',
                'left': 'ซ้าย',
                'right': 'ขวา',
                'middle': 'กลาง',
                'this': 'นี่',
                'that': 'นั่น',
                'person': 'คน',
                'thing': 'สิ่ง',
                'matter': 'เรื่อง',
                'way': 'ทาง',
                'method': 'วิธี',
                'style': 'แบบ',
                'type': 'ประเภท',
                'kind': 'ชนิด',
                'format': 'รูปแบบ',
                'example': 'ตัวอย่าง',
                'reason': 'เหตุผล',
                'cause': 'เหตุ',
                'effect': 'ผล',
                'some': 'บาง',
                'all': 'ทั้ง',
                'many': 'หลาย',
                'few': 'น้อย',
                'only': 'เพียง',
                'just': 'แค่',
                'and': 'และ',
                'or': 'หรือ',
                'but': 'แต่',
                'however': 'อย่างไรก็ตาม',
                'therefore': 'ดังนั้น',
                'because': 'เพราะ',
                'if': 'ถ้า',
                'when': 'เมื่อ',
                'since': 'ตั้งแต่',
                'until': 'จนกว่า',
                'while': 'ในขณะที่',
                'for': 'เพื่อ',
                'to': 'เพื่อให้',
                'according to': 'ตาม',
                'like': 'อย่าง',
                'such as': 'เช่น',
                'including': 'รวมทั้ง',
                'except': 'ยกเว้น',
                'besides': 'นอกจาก',
                'about': 'เกี่ยวกับ',
                'so': 'ดังนั้น',
                'as follows': 'ดังนี้',
                'as mentioned': 'ดังกล่าว',
                'as': 'ดังที่',
                'which': 'ซึ่ง',
                'what': 'อะไร',
                'who': 'ใคร',
                'where': 'ที่ไหน',
                'when': 'เมื่อไร',
                'why': 'ทำไม',
                'how': 'อย่างไร',
                'how much': 'เท่าไร',
                'how many': 'กี่',
                'what time': 'กี่โมง',
                'which': 'ไหน'
            }
        }
    
    def translate(self, text: str, source_lang: str, target_lang: str) -> TranslationResult:
        """Translate text using local dictionaries"""
        original_text = text
        
        # Detect source language if not specified
        if source_lang == "auto":
            source_lang, _ = self.language_detector.detect_language(text)
        
        # Create translation key
        translation_key = f"{source_lang}_to_{target_lang}"
        
        # Check if we have a translation dictionary
        if translation_key in self.translation_dict:
            translation_map = self.translation_dict[translation_key]
            
            # Try exact match first
            text_lower = text.lower().strip()
            if text_lower in translation_map:
                translated_text = translation_map[text_lower]
                confidence = 0.95
            else:
                # Try word-by-word translation
                words = text.split()
                translated_words = []
                found_count = 0
                
                for word in words:
                    word_lower = word.lower().strip('.,!?;:')
                    if word_lower in translation_map:
                        translated_words.append(translation_map[word_lower])
                        found_count += 1
                    else:
                        translated_words.append(word)  # Keep original if not found
                
                translated_text = ' '.join(translated_words)
                confidence = found_count / len(words) if words else 0.0
                
                # If no words found, use simple replacement
                if confidence == 0:
                    translated_text = f"[Translation: {text}]"
                    confidence = 0.1
        else:
            # No translation dictionary available
            translated_text = f"[Translation from {source_lang} to {target_lang}: {text}]"
            confidence = 0.1
        
        return TranslationResult(
            original_text=original_text,
            translated_text=translated_text,
            source_language=source_lang,
            target_language=target_lang,
            confidence=confidence,
            method="local_dictionary",
            timestamp=datetime.now()
        )


class TranslationSystem(QObject):
    """Main translation system controller"""
    
    # Signals
    translation_ready = pyqtSignal(dict)  # translation result
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration
        self.supported_languages = config.get("supported_languages", ["en", "th", "zh", "ja", "ko"])
        self.default_source_lang = config.get("default_source_lang", "auto")
        self.default_target_lang = config.get("default_target_lang", "th")
        
        # Components
        self.local_translator = LocalTranslator()
        self.language_detector = LanguageDetector()
        
        # Translation history
        self.translation_history = []
        self.max_history = config.get("max_history", 100)
        
        # Initialize
        self._initialize()
        
        self.logger.info("Translation system initialized")
    
    def _initialize(self):
        """Initialize translation system"""
        try:
            # Load translation history if exists
            self._load_history()
            
            self.logger.info("Translation system ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize translation system: {e}")
            self.error_occurred.emit(f"Translation system initialization failed: {e}")
    
    def translate_text(self, text: str, source_lang: str = None, target_lang: str = None) -> Dict[str, Any]:
        """Translate text and return result"""
        try:
            if not text or not text.strip():
                raise ValueError("Empty text provided")
            
            # Use defaults if not specified
            source_lang = source_lang or self.default_source_lang
            target_lang = target_lang or self.default_target_lang
            
            # Detect source language if auto
            if source_lang == "auto":
                detected_lang, confidence = self.language_detector.detect_language(text)
                if confidence > 0.5:
                    source_lang = detected_lang
                else:
                    source_lang = "en"  # Default fallback
            
            # Perform translation
            result = self.local_translator.translate(text, source_lang, target_lang)
            
            # Add to history
            self.translation_history.append(result)
            if len(self.translation_history) > self.max_history:
                self.translation_history.pop(0)
            
            # Save history
            self._save_history()
            
            # Convert to dict for UI
            result_dict = result.to_dict()
            
            self.logger.info(f"Translation completed: '{text}' -> '{result.translated_text}'")
            
            # Emit signal
            self.translation_ready.emit(result_dict)
            
            return result_dict
            
        except Exception as e:
            self.logger.error(f"Translation failed: {e}")
            self.error_occurred.emit(f"Translation failed: {e}")
            return None
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages"""
        language_names = {
            "en": "English",
            "th": "ไทย (Thai)",
            "zh": "中文 (Chinese)",
            "ja": "日本語 (Japanese)",
            "ko": "한국어 (Korean)",
            "auto": "Auto Detect"
        }
        
        return [
            {"code": lang, "name": language_names.get(lang, lang)}
            for lang in self.supported_languages
        ]
    
    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect text language"""
        try:
            language, confidence = self.language_detector.detect_language(text)
            
            language_names = {
                "en": "English",
                "th": "Thai",
                "zh": "Chinese",
                "ja": "Japanese",
                "ko": "Korean",
                "unknown": "Unknown"
            }
            
            return {
                "language_code": language,
                "language_name": language_names.get(language, language),
                "confidence": confidence
            }
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {
                "language_code": "unknown",
                "language_name": "Unknown",
                "confidence": 0.0
            }
    
    def get_translation_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent translation history"""
        try:
            recent_history = self.translation_history[-limit:] if limit > 0 else self.translation_history
            return [result.to_dict() for result in reversed(recent_history)]
            
        except Exception as e:
            self.logger.error(f"Failed to get translation history: {e}")
            return []
    
    def clear_history(self):
        """Clear translation history"""
        self.translation_history.clear()
        self._save_history()
        self.logger.info("Translation history cleared")
    
    def _load_history(self):
        """Load translation history from file"""
        try:
            history_file = Path(__file__).parent.parent.parent / "data" / "translation_history.json"
            
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                self.translation_history = [
                    TranslationResult.from_dict(item) for item in history_data
                ]
                
                self.logger.info(f"Loaded {len(self.translation_history)} translation history items")
                
        except Exception as e:
            self.logger.error(f"Failed to load translation history: {e}")
    
    def _save_history(self):
        """Save translation history to file"""
        try:
            history_file = Path(__file__).parent.parent.parent / "data" / "translation_history.json"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history_data = [result.to_dict() for result in self.translation_history]
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save translation history: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get translation system statistics"""
        return {
            "supported_languages": len(self.supported_languages),
            "translation_history_count": len(self.translation_history),
            "default_source_lang": self.default_source_lang,
            "default_target_lang": self.default_target_lang
        }
    
    def set_default_languages(self, source_lang: str, target_lang: str):
        """Set default languages"""
        if source_lang in self.supported_languages + ["auto"]:
            self.default_source_lang = source_lang
        if target_lang in self.supported_languages:
            self.default_target_lang = target_lang
        
        self.logger.info(f"Default languages updated: {source_lang} -> {target_lang}")
    
    def shutdown(self):
        """Shutdown translation system"""
        self.logger.info("Shutting down translation system")
        self._save_history()
        self.logger.info("Translation system shutdown complete")