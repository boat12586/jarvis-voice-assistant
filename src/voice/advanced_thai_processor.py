#!/usr/bin/env python3
"""
🇹🇭 Advanced Thai Language Processor for JARVIS
ระบบประมวลผลภาษาไทยขั้นสูงที่ใช้ DeepSeek-R1 และ mxbai-embed-large
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import time

# Thai language processing with fallbacks
try:
    import pythainlp
    from pythainlp import word_tokenize, sent_tokenize, pos_tag
    from pythainlp.normalize import normalize
    from pythainlp.transliterate import romanize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    logging.warning("PyThaiNLP not available - using basic Thai processing")


@dataclass
class ThaiProcessingResult:
    """ผลการประมวลผลภาษาไทย"""
    original: str
    normalized: str
    tokens: List[str]
    language_detected: str
    confidence: float
    pos_tags: Optional[List[Tuple[str, str]]] = None
    romanized: Optional[str] = None
    cultural_context: Optional[str] = None
    intent: Optional[str] = None
    entities: Optional[List[Dict[str, Any]]] = None


class AdvancedThaiProcessor:
    """ระบบประมวลผลภาษาไทยขั้นสูง"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # Thai language patterns
        self.thai_patterns = {
            'polite_particles': ['ครับ', 'ค่ะ', 'คะ', 'จ้ะ', 'นะ', 'หรอ'],
            'question_words': ['อะไร', 'ไหน', 'เมื่อไหร่', 'ทำไม', 'อย่างไร', 'ใคร', 'กี่'],
            'request_patterns': ['ช่วย', 'กรุณา', 'ได้ไหม', 'หน่อย', 'ขอ'],
            'greeting_patterns': ['สวัสดี', 'ดีครับ', 'ดีค่ะ', 'หวัดดี'],
            'farewell_patterns': ['ลาก่อน', 'บาย', 'แล้วพบกันใหม่', 'โชคดี']
        }
        
        # Cultural context mapping
        self.cultural_contexts = {
            'formal': ['เรียน', 'ท่าน', 'กราบ', 'สมเด็จ', 'ฝ่าบาท'],
            'casual': ['เพื่อน', 'พี่', 'น้อง', 'เฮ้ย', 'ว่าไง'],
            'business': ['บริษัท', 'ประชุม', 'โครงการ', 'งบประมาณ', 'ผลงาน'],
            'technical': ['โปรแกรม', 'คอมพิวเตอร์', 'เทคโนโลยี', 'ซอฟต์แวร์', 'ฮาร์ดแวร์']
        }
        
        # Intent patterns
        self.intent_patterns = {
            'question': ['อะไร', 'ไหน', 'เมื่อไหร่', 'ทำไม', 'อย่างไร', '?'],
            'request': ['ช่วย', 'กรุณา', 'ขอ', 'ได้ไหม'],
            'command': ['ทำ', 'เปิด', 'ปิด', 'ส่ง', 'แสดง', 'หา'],
            'greeting': ['สวัสดี', 'ดี', 'หวัดดี'],
            'information': ['บอก', 'อธิบาย', 'แนะนำ', 'ข้อมูล']
        }
        
        self.logger.info("🇹🇭 Advanced Thai Processor initialized")
    
    def process_text(self, text: str) -> ThaiProcessingResult:
        """ประมวลผลข้อความภาษาไทย"""
        start_time = time.time()
        
        # Detect language
        language, confidence = self._detect_language(text)
        
        # Normalize text
        normalized = self._normalize_thai_text(text)
        
        # Tokenize
        tokens = self._tokenize_thai(normalized)
        
        # POS tagging (if available)
        pos_tags = self._pos_tag_thai(tokens) if PYTHAINLP_AVAILABLE else None
        
        # Romanization (if available)
        romanized = self._romanize_thai(normalized) if PYTHAINLP_AVAILABLE else None
        
        # Cultural context analysis
        cultural_context = self._analyze_cultural_context(text)
        
        # Intent recognition
        intent = self._recognize_intent(text)
        
        # Named entity recognition (basic)
        entities = self._extract_entities(text)
        
        processing_time = time.time() - start_time
        
        self.logger.debug(f"🇹🇭 Processed Thai text in {processing_time:.3f}s")
        
        return ThaiProcessingResult(
            original=text,
            normalized=normalized,
            tokens=tokens,
            language_detected=language,
            confidence=confidence,
            pos_tags=pos_tags,
            romanized=romanized,
            cultural_context=cultural_context,
            intent=intent,
            entities=entities
        )
    
    def _detect_language(self, text: str) -> Tuple[str, float]:
        """ตรวจจับภาษา"""
        # Count Thai characters
        thai_chars = len(re.findall(r'[ก-๙]', text))
        total_chars = len(re.findall(r'[a-zA-Zก-๙]', text))
        
        if total_chars == 0:
            return 'unknown', 0.0
        
        thai_ratio = thai_chars / total_chars
        
        if thai_ratio > 0.6:
            return 'th', thai_ratio
        elif thai_ratio > 0.3:
            return 'mixed', thai_ratio
        else:
            return 'en', 1.0 - thai_ratio
    
    def _normalize_thai_text(self, text: str) -> str:
        """ปรับปรุงข้อความภาษาไทย"""
        if PYTHAINLP_AVAILABLE:
            return normalize(text)
        else:
            # Basic normalization
            text = re.sub(r'\s+', ' ', text)  # Multiple spaces
            text = text.strip()
            return text
    
    def _tokenize_thai(self, text: str) -> List[str]:
        """แบ่งคำภาษาไทย"""
        if PYTHAINLP_AVAILABLE:
            return word_tokenize(text, engine='longest')
        else:
            # Basic tokenization - split by spaces and punctuation
            return re.findall(r'\S+', text)
    
    def _pos_tag_thai(self, tokens: List[str]) -> List[Tuple[str, str]]:
        """ระบุชนิดของคำ"""
        if PYTHAINLP_AVAILABLE:
            return pos_tag(tokens, engine='perceptron')
        else:
            return [(token, 'UNKNOWN') for token in tokens]
    
    def _romanize_thai(self, text: str) -> str:
        """แปลงภาษาไทยเป็นอักษรโรมัน"""
        if PYTHAINLP_AVAILABLE:
            return romanize(text, engine='thai2rom')
        else:
            return text
    
    def _analyze_cultural_context(self, text: str) -> str:
        """วิเคราะห์บริบททางวัฒนธรรม"""
        text_lower = text.lower()
        
        for context_type, keywords in self.cultural_contexts.items():
            for keyword in keywords:
                if keyword in text_lower:
                    return context_type
        
        # Check politeness level
        polite_count = sum(1 for particle in self.thai_patterns['polite_particles'] 
                          if particle in text_lower)
        
        if polite_count > 0:
            return 'polite'
        
        return 'neutral'
    
    def _recognize_intent(self, text: str) -> str:
        """ระบุความตั้งใจ"""
        text_lower = text.lower()
        
        # Check for question patterns
        for pattern in self.thai_patterns['question_words']:
            if pattern in text_lower:
                return 'question'
        
        if '?' in text:
            return 'question'
        
        # Check for request patterns
        for pattern in self.thai_patterns['request_patterns']:
            if pattern in text_lower:
                return 'request'
        
        # Check for greeting patterns
        for pattern in self.thai_patterns['greeting_patterns']:
            if pattern in text_lower:
                return 'greeting'
        
        # Check for specific intents
        for intent_type, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    return intent_type
        
        return 'statement'
    
    def _extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """สกัดหัวข้อสำคัญ"""
        entities = []
        
        # Time patterns
        time_patterns = [
            (r'(\d{1,2}:\d{2})', 'TIME'),
            (r'(วันนี้|เมื่อวาน|พรุ่งนี้)', 'DATE'),
            (r'(จันทร์|อังคาร|พุธ|พฤหัสบดี|ศุกร์|เสาร์|อาทิตย์)', 'DAY'),
            (r'(\d+\s*(บาท|ดอลลาร์|ยูโร))', 'MONEY'),
            (r'(\d+\s*(เปอร์เซ็นต์|%))', 'PERCENTAGE')
        ]
        
        for pattern, entity_type in time_patterns:
            matches = re.finditer(pattern, text)
            for match in matches:
                entities.append({
                    'text': match.group(1),
                    'type': entity_type,
                    'start': match.start(),
                    'end': match.end()
                })
        
        return entities
    
    def generate_thai_response_context(self, user_input: str, intent: str) -> Dict[str, Any]:
        """สร้างบริบทสำหรับการตอบกลับภาษาไทย"""
        result = self.process_text(user_input)
        
        context = {
            'user_language': result.language_detected,
            'cultural_level': result.cultural_context,
            'intent': result.intent,
            'politeness_level': 'formal' if result.cultural_context == 'polite' else 'casual',
            'response_style': 'helpful_thai',
            'should_use_particles': True,
            'recommended_particle': 'ครับ' if 'ครับ' in user_input else 'ค่ะ' if 'ค่ะ' in user_input else 'ครับ'
        }
        
        # Add conversation suggestions
        if result.intent == 'greeting':
            context['suggested_responses'] = [
                f"สวัสดี{context['recommended_particle']} มีอะไรให้ผมช่วยไหม{context['recommended_particle']}",
                f"ยินดีต้อนรับ{context['recommended_particle']} วันนี้เป็นอย่างไรบ้าง{context['recommended_particle']}"
            ]
        elif result.intent == 'question':
            context['suggested_responses'] = [
                f"ให้ผมหาข้อมูลให้{context['recommended_particle']}",
                f"ผมจะตอบคำถามนี้ให้{context['recommended_particle']}"
            ]
        
        return context
    
    def format_thai_response(self, response: str, context: Dict[str, Any]) -> str:
        """จัดรูปแบบการตอบกลับภาษาไทย"""
        if not response:
            return response
        
        # Add politeness particles if needed
        if context.get('should_use_particles', True):
            particle = context.get('recommended_particle', 'ครับ')
            
            # Don't add if already has particle
            if not any(p in response for p in self.thai_patterns['polite_particles']):
                if not response.endswith(('.', '!', '?')):
                    response += particle
                else:
                    response = response[:-1] + particle + response[-1]
        
        return response


def test_thai_processor():
    """ทดสอบระบบประมวลผลภาษาไทย"""
    print("🧪 Testing Advanced Thai Processor...")
    
    processor = AdvancedThaiProcessor()
    
    test_cases = [
        "สวัสดีครับ วันนี้เป็นอย่างไรบ้างครับ",
        "ช่วยหาข้อมูลเกี่ยวกับ AI ให้หน่อยได้ไหมครับ",
        "เวลาตอนนี้กี่โมงแล้ว?",
        "ขอบคุณมากครับ",
        "Hello, how are you today?"
    ]
    
    for test_text in test_cases:
        print(f"\n📝 Testing: {test_text}")
        result = processor.process_text(test_text)
        
        print(f"   🌐 Language: {result.language_detected} ({result.confidence:.2f})")
        print(f"   🔤 Tokens: {result.tokens[:5]}...")  # Show first 5 tokens
        print(f"   🎭 Cultural: {result.cultural_context}")
        print(f"   🎯 Intent: {result.intent}")
        
        if result.romanized and result.language_detected == 'th':
            print(f"   🔤 Romanized: {result.romanized}")
    
    # Test response context generation
    print(f"\n🤖 Testing response context...")
    context = processor.generate_thai_response_context("สวัสดีครับ", "greeting")
    print(f"   Context: {context}")
    
    # Test response formatting
    response = processor.format_thai_response("สวัสดี มีอะไรให้ช่วยไหม", context)
    print(f"   Formatted: {response}")
    
    print("\n✅ Advanced Thai Processor test completed!")


if __name__ == "__main__":
    test_thai_processor()