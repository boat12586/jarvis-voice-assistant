"""
Advanced OCR System for JARVIS Voice Assistant
Supports Thai and English text recognition with multiple engines
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import time
import re
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import json

# OCR imports
import easyocr
import pytesseract
from pytesseract import Output

# Thai processing
import pythainlp
from pythainlp import word_tokenize, sent_tokenize
from pythainlp.transliterate import romanize


class AdvancedOCRSystem:
    """Advanced OCR system with multiple engines and language support"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # OCR engines
        self.easyocr_reader = None
        self.easyocr_thai_reader = None
        
        # Configuration
        self.supported_languages = ['en', 'th']
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        
        # Performance tracking
        self.ocr_stats = {
            'easyocr': {'usage_count': 0, 'total_time': 0},
            'tesseract': {'usage_count': 0, 'total_time': 0}
        }
        
        # Text processing patterns
        self.thai_patterns = self._init_thai_patterns()
        self.english_patterns = self._init_english_patterns()
        
        self._initialize_ocr_engines()
    
    def _initialize_ocr_engines(self):
        """Initialize all OCR engines"""
        try:
            # Initialize EasyOCR for multiple languages
            self.logger.info("Initializing EasyOCR...")
            self.easyocr_reader = easyocr.Reader(
                ['en', 'th'], 
                gpu=torch.cuda.is_available() if 'torch' in globals() else False
            )
            
            # Initialize specialized Thai reader
            self.easyocr_thai_reader = easyocr.Reader(['th'], 
                gpu=torch.cuda.is_available() if 'torch' in globals() else False
            )
            
            # Configure Tesseract
            tesseract_path = self.config.get('tesseract_path', '/usr/bin/tesseract')
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Test Tesseract installation
            try:
                version = pytesseract.get_tesseract_version()
                self.logger.info(f"Tesseract version: {version}")
            except Exception as e:
                self.logger.warning(f"Tesseract test failed: {e}")
            
            self.logger.info("OCR engines initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR engines: {e}")
    
    def _init_thai_patterns(self) -> Dict[str, Any]:
        """Initialize Thai text processing patterns"""
        return {
            'thai_chars': re.compile(r'[ก-๛]+'),
            'thai_numbers': re.compile(r'[๐-๙]+'),
            'thai_punctuation': re.compile(r'[ๆ฿่-๎็-๏]+'),
            'mixed_thai_english': re.compile(r'[ก-๛a-zA-Z0-9\s]+')
        }
    
    def _init_english_patterns(self) -> Dict[str, Any]:
        """Initialize English text processing patterns"""
        return {
            'english_words': re.compile(r'\b[A-Za-z]+\b'),
            'numbers': re.compile(r'\b\d+\b'),
            'email': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'url': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'phone': re.compile(r'\b\d{3}-\d{3}-\d{4}\b|\b\d{10}\b')
        }
    
    def extract_text(self, image_input: Union[str, Image.Image, np.ndarray], 
                    language: str = "auto", 
                    engine: str = "auto",
                    preprocessing: bool = True) -> Dict[str, Any]:
        """Extract text from image using specified engine and language"""
        
        start_time = time.time()
        
        try:
            # Preprocess image if needed
            if preprocessing:
                processed_image = self.preprocess_image(image_input)
            else:
                processed_image = self._prepare_image(image_input)
            
            # Detect language if auto
            if language == "auto":
                language = self.detect_language(processed_image)
            
            # Choose engine if auto
            if engine == "auto":
                engine = self._choose_optimal_engine(language)
            
            # Extract text using chosen engine
            if engine == "easyocr":
                result = self._extract_with_easyocr(processed_image, language)
            elif engine == "tesseract":
                result = self._extract_with_tesseract(processed_image, language)
            elif engine == "hybrid":
                result = self._extract_with_hybrid(processed_image, language)
            else:
                raise ValueError(f"Unknown engine: {engine}")
            
            # Post-process results
            result = self._post_process_results(result, language)
            
            # Add metadata
            result.update({
                'processing_time': time.time() - start_time,
                'engine_used': engine,
                'language_detected': language,
                'preprocessing_applied': preprocessing
            })
            
            # Update stats
            if engine in self.ocr_stats:
                self.ocr_stats[engine]['usage_count'] += 1
                self.ocr_stats[engine]['total_time'] += result['processing_time']
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in text extraction: {e}")
            return self._create_error_result(str(e))
    
    def preprocess_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Advanced image preprocessing for better OCR results"""
        try:
            # Convert to numpy array
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image: {image_input}")
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image):
                image = np.array(image_input)
            else:
                image = image_input.copy()
            
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image
            
            # Apply preprocessing steps
            processed = self._apply_preprocessing_pipeline(gray)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Error in image preprocessing: {e}")
            return self._prepare_image(image_input)
    
    def _apply_preprocessing_pipeline(self, gray_image: np.ndarray) -> np.ndarray:
        """Apply comprehensive preprocessing pipeline"""
        
        # 1. Noise reduction
        denoised = cv2.bilateralFilter(gray_image, 9, 75, 75)
        
        # 2. Contrast enhancement using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. Morphological operations to clean up text
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        processed = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, kernel)
        
        # 4. Adaptive thresholding for better text separation
        binary = cv2.adaptiveThreshold(
            processed, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # 5. Additional cleanup for small noise
        kernel_cleanup = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_cleanup)
        
        return cleaned
    
    def _extract_with_easyocr(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """Extract text using EasyOCR"""
        try:
            # Choose appropriate reader
            if language == "th" and self.easyocr_thai_reader:
                reader = self.easyocr_thai_reader
            else:
                reader = self.easyocr_reader
            
            if reader is None:
                raise ValueError("EasyOCR reader not available")
            
            # Extract text
            results = reader.readtext(image)
            
            # Process results
            text_blocks = []
            full_text = []
            total_confidence = 0
            
            for (bbox, text, confidence) in results:
                if confidence >= self.confidence_threshold:
                    text_blocks.append({
                        'text': text,
                        'confidence': float(confidence),
                        'bbox': [list(map(float, point)) for point in bbox],
                        'area': self._calculate_bbox_area(bbox)
                    })
                    full_text.append(text)
                    total_confidence += confidence
            
            avg_confidence = total_confidence / len(text_blocks) if text_blocks else 0
            
            return {
                'text_blocks': text_blocks,
                'full_text': ' '.join(full_text),
                'word_count': len(full_text),
                'avg_confidence': avg_confidence,
                'raw_results': results
            }
            
        except Exception as e:
            self.logger.error(f"EasyOCR extraction failed: {e}")
            return self._create_error_result(str(e))
    
    def _extract_with_tesseract(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """Extract text using Tesseract"""
        try:
            # Map language codes
            lang_map = {'en': 'eng', 'th': 'tha'}
            tesseract_lang = lang_map.get(language, 'eng+tha')
            
            # Get detailed data
            data = pytesseract.image_to_data(
                image, 
                lang=tesseract_lang, 
                output_type=Output.DICT
            )
            
            # Extract text with confidence
            text_blocks = []
            full_text = []
            
            for i in range(len(data['text'])):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                if text and confidence >= (self.confidence_threshold * 100):
                    bbox = [
                        [data['left'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i]],
                        [data['left'][i] + data['width'][i], data['top'][i] + data['height'][i]],
                        [data['left'][i], data['top'][i] + data['height'][i]]
                    ]
                    
                    text_blocks.append({
                        'text': text,
                        'confidence': confidence / 100.0,
                        'bbox': bbox,
                        'area': data['width'][i] * data['height'][i]
                    })
                    full_text.append(text)
            
            # Also get simple text extraction
            simple_text = pytesseract.image_to_string(image, lang=tesseract_lang)
            
            avg_confidence = sum(block['confidence'] for block in text_blocks) / len(text_blocks) if text_blocks else 0
            
            return {
                'text_blocks': text_blocks,
                'full_text': ' '.join(full_text) if full_text else simple_text.strip(),
                'word_count': len(full_text),
                'avg_confidence': avg_confidence,
                'simple_text': simple_text.strip()
            }
            
        except Exception as e:
            self.logger.error(f"Tesseract extraction failed: {e}")
            return self._create_error_result(str(e))
    
    def _extract_with_hybrid(self, image: np.ndarray, language: str) -> Dict[str, Any]:
        """Extract text using hybrid approach (EasyOCR + Tesseract)"""
        try:
            # Get results from both engines
            easyocr_result = self._extract_with_easyocr(image, language)
            tesseract_result = self._extract_with_tesseract(image, language)
            
            # Combine and deduplicate results
            combined_blocks = []
            seen_texts = set()
            
            # Prioritize EasyOCR results (generally better for Thai)
            for block in easyocr_result.get('text_blocks', []):
                text = block['text'].strip()
                if text and text not in seen_texts:
                    combined_blocks.append(block)
                    seen_texts.add(text)
            
            # Add unique Tesseract results
            for block in tesseract_result.get('text_blocks', []):
                text = block['text'].strip()
                if text and text not in seen_texts:
                    block['source'] = 'tesseract'
                    combined_blocks.append(block)
                    seen_texts.add(text)
            
            # Sort by confidence
            combined_blocks.sort(key=lambda x: x['confidence'], reverse=True)
            
            full_text = ' '.join(block['text'] for block in combined_blocks)
            avg_confidence = sum(block['confidence'] for block in combined_blocks) / len(combined_blocks) if combined_blocks else 0
            
            return {
                'text_blocks': combined_blocks,
                'full_text': full_text,
                'word_count': len(combined_blocks),
                'avg_confidence': avg_confidence,
                'easyocr_result': easyocr_result,
                'tesseract_result': tesseract_result
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid extraction failed: {e}")
            return self._create_error_result(str(e))
    
    def detect_language(self, image: np.ndarray) -> str:
        """Detect primary language in image"""
        try:
            # Quick extraction to detect language
            if self.easyocr_reader:
                results = self.easyocr_reader.readtext(image)
                
                thai_chars = 0
                english_chars = 0
                
                for (bbox, text, confidence) in results:
                    if confidence >= 0.3:  # Lower threshold for language detection
                        for char in text:
                            if 'ก' <= char <= '๛':
                                thai_chars += 1
                            elif char.isalpha() and char.isascii():
                                english_chars += 1
                
                # Determine primary language
                if thai_chars > english_chars:
                    return "th"
                else:
                    return "en"
            
            return "en"  # Default to English
            
        except Exception as e:
            self.logger.error(f"Language detection failed: {e}")
            return "en"
    
    def _choose_optimal_engine(self, language: str) -> str:
        """Choose optimal OCR engine based on language"""
        if language == "th":
            return "easyocr"  # EasyOCR is generally better for Thai
        elif language == "en":
            return "hybrid"   # Hybrid approach for English
        else:
            return "hybrid"   # Default to hybrid
    
    def _post_process_results(self, result: Dict[str, Any], language: str) -> Dict[str, Any]:
        """Post-process OCR results"""
        try:
            if 'full_text' in result and result['full_text']:
                text = result['full_text']
                
                # Language-specific post-processing
                if language == "th":
                    processed_text = self._post_process_thai(text)
                else:
                    processed_text = self._post_process_english(text)
                
                result['processed_text'] = processed_text
                result['text_analysis'] = self._analyze_text(processed_text, language)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Post-processing failed: {e}")
            return result
    
    def _post_process_thai(self, text: str) -> str:
        """Post-process Thai text"""
        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Fix common OCR errors in Thai
            text = text.replace('๐', '0')  # Fix Thai number to Arabic
            text = text.replace('๑', '1')
            text = text.replace('๒', '2')
            text = text.replace('๓', '3')
            text = text.replace('๔', '4')
            text = text.replace('๕', '5')
            text = text.replace('๖', '6')
            text = text.replace('๗', '7')
            text = text.replace('๘', '8')
            text = text.replace('๙', '9')
            
            return text
            
        except Exception as e:
            self.logger.error(f"Thai post-processing failed: {e}")
            return text
    
    def _post_process_english(self, text: str) -> str:
        """Post-process English text"""
        try:
            # Remove excessive whitespace
            text = ' '.join(text.split())
            
            # Fix common OCR errors
            text = text.replace('|', 'I')  # Common OCR mistake
            text = text.replace('0', 'O')  # Sometimes numbers are confused with letters
            
            # Remove lone single characters that are likely OCR errors
            words = text.split()
            filtered_words = []
            for word in words:
                if len(word) > 1 or word.isalnum():
                    filtered_words.append(word)
            
            text = ' '.join(filtered_words)
            
            return text
            
        except Exception as e:
            self.logger.error(f"English post-processing failed: {e}")
            return text
    
    def _analyze_text(self, text: str, language: str) -> Dict[str, Any]:
        """Analyze extracted text"""
        analysis = {
            'character_count': len(text),
            'word_count': len(text.split()),
            'line_count': len(text.split('\n')),
            'has_numbers': bool(re.search(r'\d', text)),
            'has_punctuation': bool(re.search(r'[.,!?;:]', text))
        }
        
        if language == "th":
            analysis.update({
                'thai_chars': len(self.thai_patterns['thai_chars'].findall(text)),
                'has_thai_numbers': bool(self.thai_patterns['thai_numbers'].search(text)),
                'mixed_script': bool(re.search(r'[a-zA-Z]', text) and re.search(r'[ก-๛]', text))
            })
            
            # Thai-specific analysis using pythainlp
            try:
                analysis['thai_words'] = len(word_tokenize(text, engine='newmm'))
                analysis['thai_sentences'] = len(sent_tokenize(text))
            except Exception as e:
                self.logger.warning(f"Thai analysis failed: {e}")
        
        else:
            analysis.update({
                'english_words': len(self.english_patterns['english_words'].findall(text)),
                'emails_found': len(self.english_patterns['email'].findall(text)),
                'urls_found': len(self.english_patterns['url'].findall(text)),
                'phones_found': len(self.english_patterns['phone'].findall(text))
            })
        
        return analysis
    
    def _prepare_image(self, image_input: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Prepare image for OCR processing"""
        try:
            if isinstance(image_input, str):
                image = cv2.imread(image_input)
                if image is None:
                    raise ValueError(f"Could not load image: {image_input}")
                return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif isinstance(image_input, Image.Image):
                return np.array(image_input)
            else:
                return image_input
                
        except Exception as e:
            self.logger.error(f"Error preparing image: {e}")
            raise
    
    def _calculate_bbox_area(self, bbox: List[List[float]]) -> float:
        """Calculate area of bounding box"""
        try:
            # Assuming bbox is [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            return width * height
            
        except Exception:
            return 0.0
    
    def _create_error_result(self, error_msg: str) -> Dict[str, Any]:
        """Create error result structure"""
        return {
            'text_blocks': [],
            'full_text': '',
            'word_count': 0,
            'avg_confidence': 0.0,
            'error': error_msg,
            'success': False
        }
    
    def batch_ocr(self, image_paths: List[str], **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images with OCR"""
        results = []
        
        for image_path in image_paths:
            try:
                result = self.extract_text(image_path, **kwargs)
                result['image_path'] = image_path
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                error_result = self._create_error_result(str(e))
                error_result['image_path'] = image_path
                results.append(error_result)
        
        return results
    
    def get_ocr_stats(self) -> Dict[str, Any]:
        """Get OCR system statistics"""
        stats = {
            'engines': self.ocr_stats.copy(),
            'supported_languages': self.supported_languages,
            'confidence_threshold': self.confidence_threshold
        }
        
        # Calculate average processing times
        for engine, data in stats['engines'].items():
            if data['usage_count'] > 0:
                data['avg_time'] = data['total_time'] / data['usage_count']
            else:
                data['avg_time'] = 0
        
        return stats
    
    def save_results(self, results: Dict[str, Any], output_path: str):
        """Save OCR results to file"""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")


class ThaiOCRSpecialist:
    """Specialized Thai OCR with enhanced processing"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Thai-specific patterns
        self.thai_digit_map = {
            '๐': '0', '๑': '1', '๒': '2', '๓': '3', '๔': '4',
            '๕': '5', '๖': '6', '๗': '7', '๘': '8', '๙': '9'
        }
        
        # Common Thai OCR corrections
        self.thai_corrections = {
            'ะ': 'ะ',  # Ensure proper Sara A
            'ิ': 'ิ',   # Ensure proper Sara I
            'ี': 'ี',   # Ensure proper Sara II
            'ึ': 'ึ',   # Ensure proper Sara UE
            'ื': 'ื'    # Ensure proper Sara UEE
        }
    
    def enhance_thai_text(self, text: str) -> str:
        """Enhance Thai text with corrections and normalization"""
        try:
            # Convert Thai digits to Arabic
            for thai_digit, arabic_digit in self.thai_digit_map.items():
                text = text.replace(thai_digit, arabic_digit)
            
            # Apply common corrections
            for wrong, correct in self.thai_corrections.items():
                text = text.replace(wrong, correct)
            
            # Normalize whitespace around Thai text
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
            
            return text
            
        except Exception as e:
            self.logger.error(f"Thai text enhancement failed: {e}")
            return text
    
    def segment_thai_words(self, text: str) -> List[str]:
        """Segment Thai text into words"""
        try:
            return word_tokenize(text, engine='newmm')
        except Exception as e:
            self.logger.error(f"Thai word segmentation failed: {e}")
            return text.split()
    
    def romanize_thai(self, text: str) -> str:
        """Convert Thai text to Roman script"""
        try:
            return romanize(text)
        except Exception as e:
            self.logger.error(f"Thai romanization failed: {e}")
            return text