"""
Enhanced Thai Language Processing for JARVIS Voice Assistant
Leverages DeepSeek-R1 and mxbai-embed-large for superior Thai language understanding
"""

import logging
import re
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass
import json
from pathlib import Path

# Thai language processing libraries (with fallbacks)
try:
    import pythainlp
    from pythainlp import word_tokenize, sent_tokenize, pos_tag
    from pythainlp.normalize import normalize
    from pythainlp.transliterate import romanize
    from pythainlp.util import normalize as util_normalize
    PYTHAINLP_AVAILABLE = True
except ImportError:
    PYTHAINLP_AVAILABLE = False
    word_tokenize = None
    sent_tokenize = None
    pos_tag = None
    normalize = None
    romanize = None

try:
    from thai_segmenter import tokenize as thai_segment
    THAI_SEGMENTER_AVAILABLE = True
except ImportError:
    THAI_SEGMENTER_AVAILABLE = False
    thai_segment = None

@dataclass
class ThaiLanguageResult:
    """Result structure for Thai language processing"""
    original_text: str
    processed_text: str
    language: str
    confidence: float
    features: Dict[str, Any]
    translations: Optional[Dict[str, str]] = None
    cultural_context: Optional[str] = None

class ThaiLanguageProcessor(QObject):
    """Enhanced Thai language processing with cultural awareness"""
    
    # Signals
    text_processed = pyqtSignal(dict)
    translation_completed = pyqtSignal(dict)
    cultural_context_detected = pyqtSignal(str)
    error_occurred = pyqtSignal(str)
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("thai_language", {})
        
        # Initialize Thai language capabilities
        self.pythainlp_available = PYTHAINLP_AVAILABLE
        self.thai_segmenter_available = THAI_SEGMENTER_AVAILABLE
        
        if self.pythainlp_available:
            self.logger.info("PyThaiNLP is available - full Thai language processing enabled")
        elif self.thai_segmenter_available:
            self.logger.info("Thai-segmenter is available - basic word segmentation enabled")
        else:
            self.logger.warning("No Thai NLP libraries available - using basic processing only")
        
        # Thai language patterns and rules
        self.thai_patterns = {
            # Tone marks
            "tone_marks": r"[่-๋]",
            # Vowels
            "vowels": r"[ะ-ฺุ-ูเ-๎]",
            # Consonants
            "consonants": r"[ก-ฮ]",
            # Numbers
            "thai_numbers": r"[๐-๙]",
            # Punctuation
            "thai_punctuation": r"[ฯๆ]"
        }
        
        # Cultural context patterns
        self.cultural_patterns = {
            "polite_particles": [
                "ครับ", "ค่ะ", "คะ", "จ้ะ", "นะ", "นะครับ", "นะคะ"
            ],
            "formal_pronouns": [
                "ผม", "ดิฉัน", "กระผม", "ข้าพเจ้า", "คุณ", "ท่าน"
            ],
            "royal_language": [
                "สมเด็จ", "พระ", "ฝ่า", "เจ้า", "ทรง", "โปรด"
            ],
            "religious_terms": [
                "พระ", "สงฆ์", "บุญ", "กรรม", "นิพพาน", "ธรรม"
            ]
        }
        
        # Common Thai expressions and meanings
        self.thai_expressions = {
            "สวัสดี": {
                "english": "hello/goodbye",
                "context": "universal greeting",
                "formality": "neutral"
            },
            "ขอบคุณ": {
                "english": "thank you",
                "context": "gratitude expression",
                "formality": "neutral"
            },
            "ไม่เป็นไร": {
                "english": "it's okay/no problem",
                "context": "dismissing concern",
                "formality": "casual"
            },
            "กรุณา": {
                "english": "please",
                "context": "polite request",
                "formality": "formal"
            },
            "ขอโทษ": {
                "english": "excuse me/sorry",
                "context": "apology or attention",
                "formality": "neutral"
            }
        }
        
        # Load additional Thai resources
        self._load_thai_resources()
        
    def _load_thai_resources(self):
        """Load additional Thai language resources"""
        try:
            # Load Thai dictionary if available
            resources_path = Path(__file__).parent.parent.parent / "data" / "thai_resources"
            resources_path.mkdir(exist_ok=True)
            
            # Create basic Thai-English dictionary
            self.thai_dictionary = {
                # AI and Technology terms
                "ปัญญาประดิษฐ์": "artificial intelligence",
                "เทคโนโลยี": "technology", 
                "คอมพิวเตอร์": "computer",
                "อินเทอร์เน็ต": "internet",
                "โปรแกรม": "program",
                "ซอฟต์แวร์": "software",
                "ฮาร์ดแวร์": "hardware",
                "ข้อมูล": "data",
                "ระบบ": "system",
                "เครือข่าย": "network",
                
                # JARVIS specific terms
                "จาร์วิส": "JARVIS",
                "ผู้ช่วย": "assistant",
                "เสียง": "voice",
                "คำสั่ง": "command",
                "คำถาม": "question",
                "คำตอบ": "answer",
                "การสนทนา": "conversation",
                "ความช่วยเหลือ": "help",
                
                # Common actions
                "ช่วย": "help",
                "บอก": "tell",
                "อธิบาย": "explain",
                "แปล": "translate",
                "ค้นหา": "search",
                "เปิด": "open",
                "ปิด": "close",
                "เริ่ม": "start",
                "หยุด": "stop",
                "ทำ": "do/make",
                
                # Time and date
                "วันนี้": "today",
                "เมื่อวาน": "yesterday", 
                "พรุ่งนี้": "tomorrow",
                "ตอนนี้": "now",
                "เวลา": "time",
                "วันที่": "date",
                "สัปดาห์": "week",
                "เดือน": "month",
                "ปี": "year"
            }
            
            self.logger.info(f"Loaded {len(self.thai_dictionary)} Thai dictionary entries")
            
        except Exception as e:
            self.logger.error(f"Failed to load Thai resources: {e}")
    
    def process_thai_text(self, text: str) -> ThaiLanguageResult:
        """Process Thai text with enhanced understanding"""
        try:
            # Detect language and analyze text
            language_info = self._detect_language_detailed(text)
            
            # Process based on language
            if language_info["is_thai"]:
                return self._process_thai_content(text, language_info)
            else:
                return self._process_mixed_content(text, language_info)
                
        except Exception as e:
            self.logger.error(f"Thai text processing failed: {e}")
            return ThaiLanguageResult(
                original_text=text,
                processed_text=text,
                language="unknown",
                confidence=0.0,
                features={"error": str(e)}
            )
    
    def _detect_language_detailed(self, text: str) -> Dict[str, Any]:
        """Detailed language detection with Thai script analysis"""
        try:
            # Count character types
            thai_chars = len(re.findall(r'[ก-๙]', text))
            english_chars = len(re.findall(r'[A-Za-z]', text))
            total_chars = len(re.sub(r'\s+', '', text))
            
            if total_chars == 0:
                return {"is_thai": False, "is_mixed": False, "confidence": 0.0}
            
            thai_ratio = thai_chars / total_chars
            english_ratio = english_chars / total_chars
            
            # Classify language
            is_thai = thai_ratio > 0.5
            is_mixed = thai_ratio > 0.1 and english_ratio > 0.1
            confidence = max(thai_ratio, english_ratio)
            
            return {
                "is_thai": is_thai,
                "is_mixed": is_mixed,
                "thai_ratio": thai_ratio,
                "english_ratio": english_ratio,
                "confidence": confidence,
                "total_chars": total_chars,
                "thai_chars": thai_chars,
                "english_chars": english_chars
            }
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return {"is_thai": False, "is_mixed": False, "confidence": 0.0}
    
    def _process_thai_content(self, text: str, lang_info: Dict[str, Any]) -> ThaiLanguageResult:
        """Process primarily Thai content"""
        try:
            # Clean and normalize text
            processed_text = self._normalize_thai_text(text)
            
            # Extract features
            features = self._extract_thai_features(processed_text)
            
            # Detect cultural context
            cultural_context = self._detect_cultural_context(processed_text)
            
            # Generate translations if needed
            translations = self._generate_translations(processed_text)
            
            result = ThaiLanguageResult(
                original_text=text,
                processed_text=processed_text,
                language="thai",
                confidence=lang_info["confidence"],
                features=features,
                translations=translations,
                cultural_context=cultural_context
            )
            
            self.text_processed.emit(result.__dict__)
            return result
            
        except Exception as e:
            self.logger.error(f"Thai content processing failed: {e}")
            raise
    
    def _process_mixed_content(self, text: str, lang_info: Dict[str, Any]) -> ThaiLanguageResult:
        """Process mixed Thai-English content"""
        try:
            # Separate Thai and English parts
            parts = self._separate_language_parts(text)
            
            # Process each part
            processed_parts = []
            for part in parts:
                if part["language"] == "thai":
                    part["processed"] = self._normalize_thai_text(part["text"])
                    part["features"] = self._extract_thai_features(part["processed"])
                else:
                    part["processed"] = part["text"].strip()
                    part["features"] = {}
                
                processed_parts.append(part)
            
            # Combine results
            processed_text = " ".join([part["processed"] for part in processed_parts])
            
            # Aggregate features
            features = {
                "is_mixed": True,
                "parts": processed_parts,
                "thai_ratio": lang_info["thai_ratio"],
                "english_ratio": lang_info["english_ratio"]
            }
            
            result = ThaiLanguageResult(
                original_text=text,
                processed_text=processed_text,
                language="mixed",
                confidence=lang_info["confidence"],
                features=features
            )
            
            self.text_processed.emit(result.__dict__)
            return result
            
        except Exception as e:
            self.logger.error(f"Mixed content processing failed: {e}")
            raise
    
    def _normalize_thai_text(self, text: str) -> str:
        """Normalize Thai text with advanced NLP if available"""
        try:
            # Use PyThaiNLP normalization if available
            if self.pythainlp_available and normalize:
                text = normalize(text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text.strip())
            
            # Normalize Thai numerals to Arabic numerals
            thai_to_arabic = {
                '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
                '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
            }
            
            for thai_num, arabic_num in thai_to_arabic.items():
                text = text.replace(thai_num, arabic_num)
            
            # Remove duplicate tone marks
            text = re.sub(r'([่-๋])\1+', r'\1', text)
            
            return text
            
        except Exception as e:
            self.logger.error(f"Thai text normalization failed: {e}")
            return text
    
    def _extract_thai_features(self, text: str) -> Dict[str, Any]:
        """Extract linguistic features from Thai text"""
        try:
            features = {}
            
            # Character analysis
            features["has_tone_marks"] = bool(re.search(self.thai_patterns["tone_marks"], text))
            features["vowel_count"] = len(re.findall(self.thai_patterns["vowels"], text))
            features["consonant_count"] = len(re.findall(self.thai_patterns["consonants"], text))
            
            # Politeness analysis
            polite_particles = [p for p in self.cultural_patterns["polite_particles"] if p in text]
            features["politeness_level"] = len(polite_particles)
            features["polite_particles"] = polite_particles
            
            # Formality analysis
            formal_pronouns = [p for p in self.cultural_patterns["formal_pronouns"] if p in text]
            features["formality_level"] = len(formal_pronouns)
            features["formal_pronouns"] = formal_pronouns
            
            # Known expressions
            known_expressions = []
            for expr, info in self.thai_expressions.items():
                if expr in text:
                    known_expressions.append({
                        "thai": expr,
                        "english": info["english"],
                        "context": info["context"]
                    })
            features["known_expressions"] = known_expressions
            
            # Dictionary matches
            dictionary_matches = []
            for thai_word, english_word in self.thai_dictionary.items():
                if thai_word in text:
                    dictionary_matches.append({
                        "thai": thai_word,
                        "english": english_word
                    })
            features["dictionary_matches"] = dictionary_matches
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def _detect_cultural_context(self, text: str) -> Optional[str]:
        """Detect cultural context in Thai text"""
        try:
            contexts = []
            
            # Check for religious context
            religious_terms = [term for term in self.cultural_patterns["religious_terms"] if term in text]
            if religious_terms:
                contexts.append("religious/buddhist")
            
            # Check for royal language
            royal_terms = [term for term in self.cultural_patterns["royal_language"] if term in text]
            if royal_terms:
                contexts.append("royal/formal")
            
            # Check for politeness level
            polite_particles = [p for p in self.cultural_patterns["polite_particles"] if p in text]
            if len(polite_particles) >= 2:
                contexts.append("highly_polite")
            elif polite_particles:
                contexts.append("polite")
            
            # Check for formality
            formal_pronouns = [p for p in self.cultural_patterns["formal_pronouns"] if p in text]
            if formal_pronouns:
                contexts.append("formal")
            
            if contexts:
                return ", ".join(contexts)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Cultural context detection failed: {e}")
            return None
    
    def _generate_translations(self, text: str) -> Dict[str, str]:
        """Generate translations for Thai text"""
        try:
            translations = {}
            
            # Direct dictionary lookup
            words = text.split()
            translated_words = []
            
            for word in words:
                # Remove punctuation for lookup
                clean_word = re.sub(r'[^\w]', '', word)
                
                if clean_word in self.thai_dictionary:
                    translated_words.append(self.thai_dictionary[clean_word])
                else:
                    translated_words.append(f"({clean_word})")
            
            translations["word_by_word"] = " ".join(translated_words)
            
            # Phrase translations
            for thai_phrase, info in self.thai_expressions.items():
                if thai_phrase in text:
                    translations[thai_phrase] = info["english"]
            
            return translations
            
        except Exception as e:
            self.logger.error(f"Translation generation failed: {e}")
            return {}
    
    def _separate_language_parts(self, text: str) -> List[Dict[str, Any]]:
        """Separate mixed language text into parts"""
        try:
            parts = []
            current_part = ""
            current_lang = None
            
            for char in text:
                # Determine character language
                if re.match(r'[ก-๙]', char):
                    char_lang = "thai"
                elif re.match(r'[A-Za-z0-9]', char):
                    char_lang = "english"
                else:
                    char_lang = "neutral"
                
                # Handle language switches
                if char_lang != "neutral":
                    if current_lang is None:
                        current_lang = char_lang
                    elif current_lang != char_lang:
                        # Language switch detected
                        if current_part.strip():
                            parts.append({
                                "text": current_part.strip(),
                                "language": current_lang
                            })
                        current_part = ""
                        current_lang = char_lang
                
                current_part += char
            
            # Add final part
            if current_part.strip():
                parts.append({
                    "text": current_part.strip(),
                    "language": current_lang or "unknown"
                })
            
            return parts
            
        except Exception as e:
            self.logger.error(f"Language separation failed: {e}")
            return [{"text": text, "language": "unknown"}]
    
    def enhance_for_ai_processing(self, text: str) -> Dict[str, Any]:
        """Enhance Thai text for better AI model processing"""
        try:
            # Process the text
            result = self.process_thai_text(text)
            
            # Create enhanced prompt for AI models
            enhanced_context = {
                "original_text": text,
                "processed_text": result.processed_text,
                "language": result.language,
                "confidence": result.confidence
            }
            
            # Add translation context
            if result.translations:
                enhanced_context["translations"] = result.translations
                
                # Create bilingual context
                english_context = []
                for thai_word, english_word in result.translations.items():
                    if thai_word != "word_by_word":
                        english_context.append(f"{thai_word} means {english_word}")
                
                if english_context:
                    enhanced_context["translation_context"] = "; ".join(english_context)
            
            # Add cultural context
            if result.cultural_context:
                enhanced_context["cultural_notes"] = f"Cultural context: {result.cultural_context}"
            
            # Add formality and politeness hints
            if result.features:
                if result.features.get("politeness_level", 0) > 0:
                    enhanced_context["tone_hint"] = "The user is speaking politely"
                
                if result.features.get("formality_level", 0) > 0:
                    enhanced_context["formality_hint"] = "The user is speaking formally"
            
            # Create AI-ready prompt enhancement
            ai_enhancement = self._create_ai_prompt_enhancement(enhanced_context)
            enhanced_context["ai_prompt_enhancement"] = ai_enhancement
            
            return enhanced_context
            
        except Exception as e:
            self.logger.error(f"AI processing enhancement failed: {e}")
            return {"original_text": text, "error": str(e)}
    
    def _create_ai_prompt_enhancement(self, context: Dict[str, Any]) -> str:
        """Create prompt enhancement for AI models"""
        try:
            enhancements = []
            
            # Language context
            if context["language"] == "thai":
                enhancements.append("The user is communicating in Thai language.")
            elif context["language"] == "mixed":
                enhancements.append("The user is using both Thai and English languages.")
            
            # Translation context
            if "translation_context" in context:
                enhancements.append(f"Key translations: {context['translation_context']}")
            
            # Cultural context
            if "cultural_notes" in context:
                enhancements.append(context["cultural_notes"])
            
            # Tone and formality
            if "tone_hint" in context:
                enhancements.append(context["tone_hint"])
            
            if "formality_hint" in context:
                enhancements.append(context["formality_hint"])
            
            if enhancements:
                return "Context: " + " ".join(enhancements)
            else:
                return ""
                
        except Exception as e:
            self.logger.error(f"AI prompt enhancement creation failed: {e}")
            return ""
    
    def get_thai_language_stats(self) -> Dict[str, Any]:
        """Get Thai language processing statistics"""
        return {
            "dictionary_size": len(self.thai_dictionary),
            "expressions_count": len(self.thai_expressions),
            "cultural_patterns": {
                "polite_particles": len(self.cultural_patterns["polite_particles"]),
                "formal_pronouns": len(self.cultural_patterns["formal_pronouns"]),
                "royal_language": len(self.cultural_patterns["royal_language"]),
                "religious_terms": len(self.cultural_patterns["religious_terms"])
            },
            "supported_features": [
                "language_detection",
                "text_normalization", 
                "cultural_context_detection",
                "politeness_analysis",
                "formality_analysis",
                "dictionary_translation",
                "ai_prompt_enhancement"
            ]
        }