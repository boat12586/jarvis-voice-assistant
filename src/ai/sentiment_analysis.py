"""
Sentiment Analysis System for JARVIS Voice Assistant
Advanced sentiment analysis with support for Thai and English
"""

import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from PyQt6.QtCore import QObject, pyqtSignal
from dataclasses import dataclass
import json
import time
import re

@dataclass
class SentimentResult:
    """Sentiment analysis result"""
    sentiment: str  # positive, negative, neutral
    confidence: float
    polarity: float  # -1 to 1
    subjectivity: float  # 0 to 1
    intensity: float  # 0 to 1
    emotional_indicators: List[str]
    language: str
    timestamp: float

@dataclass
class ConversationSentiment:
    """Sentiment analysis for entire conversation"""
    overall_sentiment: str
    sentiment_trend: str  # improving, declining, stable
    sentiment_distribution: Dict[str, float]
    emotional_journey: List[SentimentResult]
    conversation_mood: str
    engagement_level: float

class SentimentAnalysisSystem(QObject):
    """Advanced sentiment analysis with multilingual support"""
    
    # Signals
    sentiment_analyzed = pyqtSignal(dict)  # SentimentResult as dict
    conversation_sentiment_updated = pyqtSignal(dict)  # ConversationSentiment as dict
    sentiment_shift_detected = pyqtSignal(str, str, float)  # old_sentiment, new_sentiment, intensity
    emotional_pattern_detected = pyqtSignal(str, dict)  # pattern_type, pattern_data
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config.get("sentiment_analysis", {})
        
        # Initialize sentiment models
        self.text_sentiment_model = None
        self.transformer_model = None
        
        # Language support
        self.supported_languages = ["en", "th"]
        
        # Sentiment tracking
        self.conversation_history: List[SentimentResult] = []
        self.max_history_length = self.config.get("max_history_length", 100)
        
        # Sentiment lexicons
        self.sentiment_lexicons = {
            "en": {},
            "th": {}
        }
        
        # Pattern detection
        self.sentiment_patterns = []
        self.pattern_detection_enabled = self.config.get("enable_pattern_detection", True)
        
        # Initialize components
        self._initialize_sentiment_models()
        self._load_sentiment_lexicons()
        
    def _initialize_sentiment_models(self):
        """Initialize sentiment analysis models"""
        try:
            # Try to initialize transformer-based sentiment analysis
            from transformers import pipeline
            
            # English sentiment model
            try:
                model_name = self.config.get("sentiment_model", "cardiffnlp/twitter-roberta-base-sentiment-latest")
                self.transformer_model = pipeline(
                    "sentiment-analysis",
                    model=model_name,
                    tokenizer=model_name,
                    device=-1  # CPU
                )
                self.logger.info(f"Transformer sentiment model initialized: {model_name}")
            except Exception as e:
                self.logger.warning(f"Failed to initialize transformer model: {e}")
            
            # Fallback to TextBlob if transformers fail
            if not self.transformer_model:
                try:
                    from textblob import TextBlob
                    self.textblob = TextBlob
                    self.logger.info("TextBlob sentiment analysis initialized as fallback")
                except ImportError:
                    self.logger.warning("TextBlob not available")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize sentiment models: {e}")
            # Initialize rule-based fallback
            self._initialize_rule_based_sentiment()
    
    def _initialize_rule_based_sentiment(self):
        """Initialize rule-based sentiment analysis as fallback"""
        try:
            self.rule_based_active = True
            self.logger.info("Rule-based sentiment analysis initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize rule-based sentiment: {e}")
    
    def _load_sentiment_lexicons(self):
        """Load sentiment lexicons for different languages"""
        try:
            # English sentiment words
            self.sentiment_lexicons["en"] = {
                "positive": [
                    "happy", "joy", "love", "excellent", "wonderful", "amazing", "great", "good", 
                    "fantastic", "awesome", "brilliant", "perfect", "beautiful", "pleased", 
                    "satisfied", "delighted", "excited", "thrilled", "grateful", "thankful",
                    "optimistic", "confident", "proud", "successful", "accomplished", "blessed"
                ],
                "negative": [
                    "sad", "angry", "hate", "terrible", "awful", "horrible", "bad", "worst",
                    "disgusting", "annoying", "frustrated", "disappointed", "upset", "worried",
                    "anxious", "depressed", "miserable", "furious", "outraged", "devastated",
                    "stressed", "overwhelmed", "confused", "lost", "hopeless", "helpless"
                ],
                "intensifiers": [
                    "very", "extremely", "incredibly", "absolutely", "completely", "totally",
                    "really", "quite", "rather", "so", "too", "super", "ultra", "mega"
                ],
                "negators": [
                    "not", "no", "never", "none", "nothing", "nobody", "nowhere", "neither",
                    "nor", "hardly", "barely", "scarcely", "seldom", "rarely"
                ]
            }
            
            # Thai sentiment words
            self.sentiment_lexicons["th"] = {
                "positive": [
                    "ดี", "ดีมาก", "เยี่ยม", "วิเศษ", "สุดยอด", "ยอดเยี่ยม", "ดีเลิศ", "ชอบ",
                    "รัก", "มีความสุข", "ดีใจ", "ยินดี", "ปลื้ม", "ตื่นเต้น", "กระตือรือร้น",
                    "ภูมิใจ", "พอใจ", "ขอบคุณ", "ชื่นชม", "ประทับใจ", "หลงใหล", "ชื่นชอบ",
                    "สนุก", "สนุกสนาน", "เพลิดเพลิน", "สบายใจ", "อุ่นใจ", "คล่องใจ"
                ],
                "negative": [
                    "แย่", "แย่มาก", "เลว", "ไม่ดี", "น่าเกลียด", "ไม่ชอบ", "เกลียด", "โกรธ",
                    "เศร้า", "เศร้าโศก", "หดหู่", "เซ็ง", "เครียด", "กังวล", "วิตกกังวล",
                    "ผิดหวัง", "น่าผิดหวัง", "เบื่อ", "เหนื่อย", "ปวดหัว", "ท้อ", "ท้อแท้",
                    "สิ้นหวัง", "หมดหวัง", "เจ็บปวด", "ทุกข์", "ยาก", "ลำบาก", "ลำบากใจ"
                ],
                "intensifiers": [
                    "มาก", "มากๆ", "เป็นอย่างมาก", "อย่างยิ่ง", "สุดๆ", "เกินไป", "ยิ่งนัก",
                    "อย่างมาก", "เต็มที่", "สุด", "แสน", "ยอด", "จัด", "หนัก", "แรง"
                ],
                "negators": [
                    "ไม่", "ไม่ได้", "ไม่มี", "ไม่เคย", "ไม่ค่อย", "แทบไม่", "หาไม่", "ไม่ใช่",
                    "มิได้", "มิใช่", "ไม่ต้อง", "ไม่จำเป็น", "ไม่สามารถ"
                ]
            }
            
            self.logger.info("Sentiment lexicons loaded for English and Thai")
            
        except Exception as e:
            self.logger.error(f"Failed to load sentiment lexicons: {e}")
    
    def analyze_sentiment(self, text: str, language: str = "en") -> SentimentResult:
        """Analyze sentiment of given text"""
        try:
            start_time = time.time()
            
            # Clean and preprocess text
            cleaned_text = self._preprocess_text(text, language)
            
            # Try transformer model first (for English)
            if self.transformer_model and language == "en":
                result = self._analyze_with_transformer(cleaned_text)
            else:
                # Use rule-based analysis
                result = self._analyze_with_rules(cleaned_text, language)
            
            # Add emotional indicators
            emotional_indicators = self._extract_emotional_indicators(text, language)
            result.emotional_indicators = emotional_indicators
            
            # Add to conversation history
            self.conversation_history.append(result)
            if len(self.conversation_history) > self.max_history_length:
                self.conversation_history = self.conversation_history[-self.max_history_length:]
            
            # Detect sentiment shifts
            self._detect_sentiment_shifts()
            
            # Detect emotional patterns
            if self.pattern_detection_enabled:
                self._detect_emotional_patterns()
            
            # Emit signal
            self.sentiment_analyzed.emit(self._sentiment_result_to_dict(result))
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to analyze sentiment: {e}")
            return self._create_neutral_sentiment_result(text, language)
    
    def _preprocess_text(self, text: str, language: str) -> str:
        """Preprocess text for sentiment analysis"""
        try:
            # Remove extra whitespace
            cleaned = re.sub(r'\s+', ' ', text.strip())
            
            # Remove URLs
            cleaned = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', cleaned)
            
            # Remove email addresses
            cleaned = re.sub(r'\S+@\S+', '', cleaned)
            
            # Handle repetitive characters (like "sooooo good" -> "so good")
            cleaned = re.sub(r'(.)\1{2,}', r'\1\1', cleaned)
            
            return cleaned
            
        except Exception:
            return text
    
    def _analyze_with_transformer(self, text: str) -> SentimentResult:
        """Analyze sentiment using transformer model"""
        try:
            results = self.transformer_model(text)
            
            if results and len(results) > 0:
                result = results[0]
                label = result['label'].lower()
                confidence = result['score']
                
                # Map labels to sentiment
                if 'positive' in label or 'pos' in label:
                    sentiment = "positive"
                    polarity = confidence
                elif 'negative' in label or 'neg' in label:
                    sentiment = "negative"
                    polarity = -confidence
                else:
                    sentiment = "neutral"
                    polarity = 0.0
                
                return SentimentResult(
                    sentiment=sentiment,
                    confidence=confidence,
                    polarity=polarity,
                    subjectivity=0.5,  # Default for transformer models
                    intensity=confidence,
                    emotional_indicators=[],
                    language="en",
                    timestamp=time.time()
                )
            
        except Exception as e:
            self.logger.error(f"Transformer sentiment analysis failed: {e}")
        
        return self._analyze_with_rules(text, "en")
    
    def _analyze_with_rules(self, text: str, language: str) -> SentimentResult:
        """Analyze sentiment using rule-based approach"""
        try:
            text_lower = text.lower()
            
            # Get lexicon for language
            lexicon = self.sentiment_lexicons.get(language, self.sentiment_lexicons["en"])
            
            positive_words = lexicon.get("positive", [])
            negative_words = lexicon.get("negative", [])
            intensifiers = lexicon.get("intensifiers", [])
            negators = lexicon.get("negators", [])
            
            # Count sentiment words
            positive_score = 0
            negative_score = 0
            
            words = text_lower.split()
            
            for i, word in enumerate(words):
                # Check for negation
                negated = False
                if i > 0 and words[i-1] in negators:
                    negated = True
                elif i > 1 and words[i-2] in negators:
                    negated = True
                
                # Check for intensification
                intensified = False
                intensifier_multiplier = 1.0
                if i > 0 and words[i-1] in intensifiers:
                    intensified = True
                    intensifier_multiplier = 1.5
                
                # Score words
                if word in positive_words:
                    score = 1.0 * intensifier_multiplier
                    if negated:
                        negative_score += score
                    else:
                        positive_score += score
                        
                elif word in negative_words:
                    score = 1.0 * intensifier_multiplier
                    if negated:
                        positive_score += score
                    else:
                        negative_score += score
            
            # Calculate final sentiment
            total_score = positive_score + negative_score
            if total_score == 0:
                sentiment = "neutral"
                polarity = 0.0
                confidence = 0.5
            else:
                polarity = (positive_score - negative_score) / total_score
                
                if polarity > 0.1:
                    sentiment = "positive"
                    confidence = min(positive_score / (positive_score + negative_score), 1.0)
                elif polarity < -0.1:
                    sentiment = "negative"
                    confidence = min(negative_score / (positive_score + negative_score), 1.0)
                else:
                    sentiment = "neutral"
                    confidence = 0.6
            
            # Calculate subjectivity (rough estimation)
            total_words = len(words)
            sentiment_words = positive_score + negative_score
            subjectivity = min(sentiment_words / max(total_words, 1), 1.0)
            
            return SentimentResult(
                sentiment=sentiment,
                confidence=confidence,
                polarity=polarity,
                subjectivity=subjectivity,
                intensity=abs(polarity),
                emotional_indicators=[],
                language=language,
                timestamp=time.time()
            )
            
        except Exception as e:
            self.logger.error(f"Rule-based sentiment analysis failed: {e}")
            return self._create_neutral_sentiment_result(text, language)
    
    def _extract_emotional_indicators(self, text: str, language: str) -> List[str]:
        """Extract emotional indicators from text"""
        try:
            indicators = []
            text_lower = text.lower()
            
            # Emotional expressions
            if language == "en":
                emotional_patterns = {
                    "excitement": ["!", "wow", "amazing", "incredible", "fantastic"],
                    "frustration": ["ugh", "argh", "damn", "annoying", "irritating"],
                    "questioning": ["?", "why", "how", "what", "confused"],
                    "emphasis": ["very", "really", "so", "totally", "absolutely"],
                    "uncertainty": ["maybe", "perhaps", "might", "could", "unsure"]
                }
            else:  # Thai
                emotional_patterns = {
                    "excitement": ["!", "ว้าว", "เยี่ยม", "สุดยอด", "วิเศษ"],
                    "frustration": ["เฮ้อ", "อ่า", "น่าร้าย", "รำคาญ", "หงุดหงิด"],
                    "questioning": ["?", "ทำไม", "อย่างไร", "อะไร", "สับสน"],
                    "emphasis": ["มาก", "จริงๆ", "สุดๆ", "เป็นอย่างมาก", "อย่างยิ่ง"],
                    "uncertainty": ["อาจจะ", "บางที", "น่าจะ", "คงจะ", "ไม่แน่ใจ"]
                }
            
            for emotion_type, patterns in emotional_patterns.items():
                for pattern in patterns:
                    if pattern in text_lower:
                        indicators.append(emotion_type)
                        break
            
            # Punctuation-based indicators
            if text.count("!") > 1:
                indicators.append("high_excitement")
            if text.count("?") > 1:
                indicators.append("confusion")
            if "..." in text:
                indicators.append("hesitation")
            
            # Capital letters (for emphasis)
            if len(re.findall(r'[A-Z]{2,}', text)) > 0:
                indicators.append("emphasis")
            
            return list(set(indicators))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to extract emotional indicators: {e}")
            return []
    
    def _detect_sentiment_shifts(self):
        """Detect significant sentiment shifts in conversation"""
        try:
            if len(self.conversation_history) < 2:
                return
            
            current = self.conversation_history[-1]
            previous = self.conversation_history[-2]
            
            # Check for sentiment change
            if current.sentiment != previous.sentiment:
                # Calculate shift intensity
                polarity_change = abs(current.polarity - previous.polarity)
                
                if polarity_change > 0.5:  # Significant shift
                    self.sentiment_shift_detected.emit(
                        previous.sentiment, 
                        current.sentiment, 
                        polarity_change
                    )
                    
                    self.logger.info(f"Sentiment shift detected: {previous.sentiment} -> {current.sentiment}")
            
        except Exception as e:
            self.logger.error(f"Failed to detect sentiment shifts: {e}")
    
    def _detect_emotional_patterns(self):
        """Detect emotional patterns in conversation history"""
        try:
            if len(self.conversation_history) < 5:
                return
            
            recent_sentiments = [s.sentiment for s in self.conversation_history[-5:]]
            recent_polarities = [s.polarity for s in self.conversation_history[-5:]]
            
            # Pattern: Consistently negative
            if recent_sentiments.count("negative") >= 4:
                pattern_data = {
                    "type": "consistently_negative",
                    "confidence": 0.8,
                    "duration": len(recent_sentiments)
                }
                self.emotional_pattern_detected.emit("negative_spiral", pattern_data)
            
            # Pattern: Emotional roller coaster
            sentiment_changes = sum(1 for i in range(1, len(recent_sentiments)) 
                                  if recent_sentiments[i] != recent_sentiments[i-1])
            if sentiment_changes >= 3:
                pattern_data = {
                    "type": "emotional_volatility",
                    "confidence": 0.7,
                    "changes": sentiment_changes
                }
                self.emotional_pattern_detected.emit("volatility", pattern_data)
            
            # Pattern: Improving mood
            if len(recent_polarities) >= 3:
                trend = np.polyfit(range(len(recent_polarities)), recent_polarities, 1)[0]
                if trend > 0.2:
                    pattern_data = {
                        "type": "improving_mood",
                        "confidence": 0.6,
                        "trend_strength": trend
                    }
                    self.emotional_pattern_detected.emit("mood_improvement", pattern_data)
            
        except Exception as e:
            self.logger.error(f"Failed to detect emotional patterns: {e}")
    
    def analyze_conversation_sentiment(self) -> ConversationSentiment:
        """Analyze sentiment of entire conversation"""
        try:
            if not self.conversation_history:
                return ConversationSentiment(
                    overall_sentiment="neutral",
                    sentiment_trend="stable",
                    sentiment_distribution={"neutral": 1.0},
                    emotional_journey=[],
                    conversation_mood="neutral",
                    engagement_level=0.5
                )
            
            # Calculate overall sentiment
            sentiments = [s.sentiment for s in self.conversation_history]
            sentiment_counts = {
                "positive": sentiments.count("positive"),
                "negative": sentiments.count("negative"),
                "neutral": sentiments.count("neutral")
            }
            
            total_count = len(sentiments)
            sentiment_distribution = {
                k: v / total_count for k, v in sentiment_counts.items()
            }
            
            # Determine overall sentiment
            if sentiment_distribution["positive"] > 0.5:
                overall_sentiment = "positive"
            elif sentiment_distribution["negative"] > 0.5:
                overall_sentiment = "negative"
            else:
                overall_sentiment = "neutral"
            
            # Calculate sentiment trend
            sentiment_trend = self._calculate_sentiment_trend()
            
            # Determine conversation mood
            conversation_mood = self._determine_conversation_mood(sentiment_distribution)
            
            # Calculate engagement level
            engagement_level = self._calculate_engagement_level()
            
            conversation_sentiment = ConversationSentiment(
                overall_sentiment=overall_sentiment,
                sentiment_trend=sentiment_trend,
                sentiment_distribution=sentiment_distribution,
                emotional_journey=self.conversation_history[-10:],  # Last 10 for journey
                conversation_mood=conversation_mood,
                engagement_level=engagement_level
            )
            
            # Emit signal
            self.conversation_sentiment_updated.emit(
                self._conversation_sentiment_to_dict(conversation_sentiment)
            )
            
            return conversation_sentiment
            
        except Exception as e:
            self.logger.error(f"Failed to analyze conversation sentiment: {e}")
            return ConversationSentiment(
                overall_sentiment="neutral",
                sentiment_trend="stable",
                sentiment_distribution={"neutral": 1.0},
                emotional_journey=[],
                conversation_mood="neutral",
                engagement_level=0.5
            )
    
    def _calculate_sentiment_trend(self) -> str:
        """Calculate sentiment trend over conversation"""
        try:
            if len(self.conversation_history) < 3:
                return "stable"
            
            # Look at recent polarities
            recent_polarities = [s.polarity for s in self.conversation_history[-5:]]
            
            if len(recent_polarities) < 2:
                return "stable"
            
            # Calculate trend using linear regression
            x = np.arange(len(recent_polarities))
            slope = np.polyfit(x, recent_polarities, 1)[0]
            
            if slope > 0.1:
                return "improving"
            elif slope < -0.1:
                return "declining"
            else:
                return "stable"
                
        except Exception:
            return "stable"
    
    def _determine_conversation_mood(self, sentiment_distribution: Dict[str, float]) -> str:
        """Determine overall conversation mood"""
        try:
            positive_ratio = sentiment_distribution.get("positive", 0)
            negative_ratio = sentiment_distribution.get("negative", 0)
            
            if positive_ratio > 0.6:
                return "upbeat"
            elif negative_ratio > 0.6:
                return "somber"
            elif positive_ratio > 0.4 and negative_ratio < 0.2:
                return "optimistic"
            elif negative_ratio > 0.4 and positive_ratio < 0.2:
                return "pessimistic"
            else:
                return "balanced"
                
        except Exception:
            return "neutral"
    
    def _calculate_engagement_level(self) -> float:
        """Calculate engagement level based on sentiment activity"""
        try:
            if not self.conversation_history:
                return 0.5
            
            # Factors for engagement:
            # 1. Variety of sentiments
            sentiments = [s.sentiment for s in self.conversation_history]
            sentiment_variety = len(set(sentiments)) / 3  # Max 3 sentiment types
            
            # 2. Intensity of emotions
            intensities = [s.intensity for s in self.conversation_history]
            avg_intensity = np.mean(intensities)
            
            # 3. Presence of emotional indicators
            total_indicators = sum(len(s.emotional_indicators) for s in self.conversation_history)
            indicator_factor = min(total_indicators / len(self.conversation_history), 1.0)
            
            # Combine factors
            engagement = (sentiment_variety * 0.3 + avg_intensity * 0.4 + indicator_factor * 0.3)
            
            return min(max(engagement, 0.0), 1.0)
            
        except Exception:
            return 0.5
    
    def _create_neutral_sentiment_result(self, text: str, language: str) -> SentimentResult:
        """Create neutral sentiment result as fallback"""
        return SentimentResult(
            sentiment="neutral",
            confidence=0.5,
            polarity=0.0,
            subjectivity=0.3,
            intensity=0.3,
            emotional_indicators=[],
            language=language,
            timestamp=time.time()
        )
    
    def _sentiment_result_to_dict(self, result: SentimentResult) -> Dict[str, Any]:
        """Convert SentimentResult to dictionary"""
        return {
            "sentiment": result.sentiment,
            "confidence": result.confidence,
            "polarity": result.polarity,
            "subjectivity": result.subjectivity,
            "intensity": result.intensity,
            "emotional_indicators": result.emotional_indicators,
            "language": result.language,
            "timestamp": result.timestamp
        }
    
    def _conversation_sentiment_to_dict(self, conv_sentiment: ConversationSentiment) -> Dict[str, Any]:
        """Convert ConversationSentiment to dictionary"""
        return {
            "overall_sentiment": conv_sentiment.overall_sentiment,
            "sentiment_trend": conv_sentiment.sentiment_trend,
            "sentiment_distribution": conv_sentiment.sentiment_distribution,
            "emotional_journey": [self._sentiment_result_to_dict(s) for s in conv_sentiment.emotional_journey],
            "conversation_mood": conv_sentiment.conversation_mood,
            "engagement_level": conv_sentiment.engagement_level
        }
    
    def get_sentiment_summary(self) -> Dict[str, Any]:
        """Get summary of sentiment analysis system"""
        try:
            conversation_sentiment = self.analyze_conversation_sentiment()
            
            return {
                "system_status": "active",
                "supported_languages": self.supported_languages,
                "conversation_history_length": len(self.conversation_history),
                "current_conversation": self._conversation_sentiment_to_dict(conversation_sentiment),
                "models_available": {
                    "transformer": self.transformer_model is not None,
                    "rule_based": True,
                    "textblob": hasattr(self, 'textblob')
                },
                "pattern_detection_enabled": self.pattern_detection_enabled
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get sentiment summary: {e}")
            return {"system_status": "error", "error": str(e)}
    
    def reset_conversation_history(self):
        """Reset conversation sentiment history"""
        self.conversation_history = []
        self.sentiment_patterns = []
        self.logger.info("Conversation sentiment history reset")
    
    def get_recent_sentiment_trend(self, window_size: int = 5) -> Dict[str, Any]:
        """Get recent sentiment trend"""
        try:
            if len(self.conversation_history) < window_size:
                return {"trend": "insufficient_data", "confidence": 0.0}
            
            recent_results = self.conversation_history[-window_size:]
            polarities = [r.polarity for r in recent_results]
            
            # Calculate trend
            x = np.arange(len(polarities))
            slope, intercept = np.polyfit(x, polarities, 1)
            
            if slope > 0.1:
                trend = "improving"
                confidence = min(abs(slope) * 2, 1.0)
            elif slope < -0.1:
                trend = "declining"
                confidence = min(abs(slope) * 2, 1.0)
            else:
                trend = "stable"
                confidence = 1.0 - abs(slope)
            
            return {
                "trend": trend,
                "confidence": confidence,
                "slope": slope,
                "window_size": window_size,
                "data_points": len(recent_results)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get recent sentiment trend: {e}")
            return {"trend": "error", "confidence": 0.0, "error": str(e)}