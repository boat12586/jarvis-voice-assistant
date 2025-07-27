"""
Multimodal AI Engine for JARVIS Voice Assistant
Handles Vision, OCR, and Multimodal Fusion
"""

import logging
import torch
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtCore import QObject, pyqtSignal, QThread, QTimer
import json
import time
from datetime import datetime

# Vision and ML imports
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    AutoProcessor, AutoModel
)
import easyocr
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import imageio
import av


class MultimodalEngine(QObject):
    """Advanced multimodal AI engine for vision, OCR, and fusion"""
    
    # Signals
    image_processed = pyqtSignal(dict)  # results
    video_processed = pyqtSignal(dict)  # results
    ocr_completed = pyqtSignal(dict)    # results
    visual_qa_ready = pyqtSignal(dict)  # results
    fusion_complete = pyqtSignal(dict)  # multimodal results
    error_occurred = pyqtSignal(str)
    processing_progress = pyqtSignal(str, int)  # status, percentage
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.logger.info(f"Multimodal engine using device: {self.device}")
        
        # Model components
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        self.blip_qa_model = None
        self.blip_qa_processor = None
        
        # OCR components
        self.easyocr_reader = None
        self.thai_ocr_reader = None
        
        # State tracking
        self.is_ready = False
        self.models_loaded = False
        self.processing_queue = []
        
        # Thread pool for async processing
        self.thread_pool = ThreadPoolExecutor(max_workers=4)
        
        # Initialize components
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all vision and multimodal models"""
        try:
            self.processing_progress.emit("Initializing CLIP model...", 10)
            
            # Initialize CLIP for image-text similarity
            clip_model_name = self.config.get("clip_model", "openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            self.logger.info("CLIP model loaded successfully")
            
            self.processing_progress.emit("Initializing BLIP models...", 30)
            
            # Initialize BLIP for image captioning and VQA
            blip_model_name = self.config.get("blip_model", "Salesforce/blip-image-captioning-base")
            self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(self.device)
            
            # BLIP for Question Answering
            blip_qa_model_name = self.config.get("blip_qa_model", "Salesforce/blip-vqa-base")
            self.blip_qa_processor = BlipProcessor.from_pretrained(blip_qa_model_name)
            self.blip_qa_model = BlipForQuestionAnswering.from_pretrained(blip_qa_model_name).to(self.device)
            
            self.logger.info("BLIP models loaded successfully")
            
            self.processing_progress.emit("Initializing OCR systems...", 50)
            
            # Initialize OCR readers
            self._initialize_ocr()
            
            self.processing_progress.emit("Models initialization complete", 100)
            
            self.models_loaded = True
            self.is_ready = True
            self.logger.info("Multimodal engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize multimodal models: {e}")
            self.error_occurred.emit(f"Model initialization failed: {e}")
    
    def _initialize_ocr(self):
        """Initialize OCR systems for multiple languages"""
        try:
            # EasyOCR for multiple languages including Thai
            languages = ['en', 'th']  # English and Thai
            self.easyocr_reader = easyocr.Reader(languages, gpu=torch.cuda.is_available())
            
            # Configure Tesseract for Thai
            pytesseract.pytesseract.tesseract_cmd = self.config.get(
                "tesseract_path", 
                "/usr/bin/tesseract"
            )
            
            self.logger.info("OCR systems initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize OCR: {e}")
            # Continue without OCR if it fails
    
    def process_image(self, image_path: str, tasks: List[str] = None) -> Dict[str, Any]:
        """Process image with multiple AI tasks"""
        if not self.is_ready:
            raise ValueError("Multimodal engine not ready")
        
        if tasks is None:
            tasks = ["caption", "ocr", "objects", "scene"]
        
        try:
            # Load and preprocess image
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            results = {
                "image_path": image_path,
                "timestamp": datetime.now().isoformat(),
                "tasks_completed": [],
                "processing_time": 0
            }
            
            start_time = time.time()
            
            # Image captioning
            if "caption" in tasks:
                caption = self._generate_caption(image)
                results["caption"] = caption
                results["tasks_completed"].append("caption")
            
            # OCR text extraction
            if "ocr" in tasks:
                ocr_results = self._extract_text(image_path)
                results["text_content"] = ocr_results
                results["tasks_completed"].append("ocr")
            
            # Object detection and classification
            if "objects" in tasks:
                objects = self._detect_objects(image)
                results["objects"] = objects
                results["tasks_completed"].append("objects")
            
            # Scene understanding
            if "scene" in tasks:
                scene = self._analyze_scene(image)
                results["scene_analysis"] = scene
                results["tasks_completed"].append("scene")
            
            # Image embeddings for similarity search
            if "embeddings" in tasks:
                embeddings = self._get_image_embeddings(image)
                results["embeddings"] = embeddings.tolist()
                results["tasks_completed"].append("embeddings")
            
            results["processing_time"] = time.time() - start_time
            
            self.image_processed.emit(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing image {image_path}: {e}")
            self.error_occurred.emit(f"Image processing failed: {e}")
            return {}
    
    def process_video(self, video_path: str, sample_rate: int = 1) -> Dict[str, Any]:
        """Process video with frame analysis"""
        if not self.is_ready:
            raise ValueError("Multimodal engine not ready")
        
        try:
            results = {
                "video_path": video_path,
                "timestamp": datetime.now().isoformat(),
                "frames_analyzed": 0,
                "scenes": [],
                "overall_summary": "",
                "processing_time": 0
            }
            
            start_time = time.time()
            
            # Extract key frames
            frames = self._extract_video_frames(video_path, sample_rate)
            
            scene_descriptions = []
            frame_count = 0
            
            for i, frame in enumerate(frames):
                try:
                    # Generate caption for each frame
                    caption = self._generate_caption(frame)
                    
                    # Detect objects in frame
                    objects = self._detect_objects(frame)
                    
                    scene_data = {
                        "frame_number": i * sample_rate,
                        "timestamp": i * sample_rate / 30.0,  # Assuming 30 FPS
                        "description": caption,
                        "objects": objects
                    }
                    
                    results["scenes"].append(scene_data)
                    scene_descriptions.append(caption)
                    frame_count += 1
                    
                    # Update progress
                    progress = int((i + 1) / len(frames) * 100)
                    self.processing_progress.emit(f"Analyzing frame {i+1}/{len(frames)}", progress)
                    
                except Exception as e:
                    self.logger.warning(f"Error processing frame {i}: {e}")
                    continue
            
            # Generate overall video summary
            if scene_descriptions:
                results["overall_summary"] = self._summarize_video_content(scene_descriptions)
            
            results["frames_analyzed"] = frame_count
            results["processing_time"] = time.time() - start_time
            
            self.video_processed.emit(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing video {video_path}: {e}")
            self.error_occurred.emit(f"Video processing failed: {e}")
            return {}
    
    def visual_question_answering(self, image_path: str, question: str) -> Dict[str, Any]:
        """Answer questions about images"""
        if not self.is_ready or not self.blip_qa_model:
            raise ValueError("Visual QA not available")
        
        try:
            image = self._load_image(image_path)
            if image is None:
                raise ValueError(f"Could not load image: {image_path}")
            
            # Process with BLIP VQA
            inputs = self.blip_qa_processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.blip_qa_model.generate(**inputs, max_length=50)
                answer = self.blip_qa_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Also get image caption for context
            caption = self._generate_caption(image)
            
            results = {
                "image_path": image_path,
                "question": question,
                "answer": answer,
                "context": caption,
                "timestamp": datetime.now().isoformat()
            }
            
            self.visual_qa_ready.emit(results)
            return results
            
        except Exception as e:
            self.logger.error(f"Error in visual QA: {e}")
            self.error_occurred.emit(f"Visual QA failed: {e}")
            return {}
    
    def multimodal_fusion(self, text: str, image_path: str = None, audio_context: str = None) -> Dict[str, Any]:
        """Fuse multiple modalities for comprehensive understanding"""
        try:
            fusion_results = {
                "timestamp": datetime.now().isoformat(),
                "modalities": [],
                "text_input": text,
                "analysis": {},
                "confidence_scores": {},
                "fusion_summary": ""
            }
            
            # Text analysis
            text_features = self._analyze_text_context(text)
            fusion_results["analysis"]["text"] = text_features
            fusion_results["modalities"].append("text")
            
            # Image analysis if provided
            if image_path and Path(image_path).exists():
                image_results = self.process_image(image_path, ["caption", "objects", "embeddings"])
                fusion_results["analysis"]["vision"] = image_results
                fusion_results["modalities"].append("vision")
                
                # Calculate text-image similarity
                similarity = self._calculate_text_image_similarity(text, image_path)
                fusion_results["confidence_scores"]["text_image_similarity"] = similarity
            
            # Audio context if provided
            if audio_context:
                fusion_results["analysis"]["audio_context"] = audio_context
                fusion_results["modalities"].append("audio")
            
            # Generate fusion summary
            fusion_summary = self._generate_multimodal_summary(fusion_results)
            fusion_results["fusion_summary"] = fusion_summary
            
            self.fusion_complete.emit(fusion_results)
            return fusion_results
            
        except Exception as e:
            self.logger.error(f"Error in multimodal fusion: {e}")
            self.error_occurred.emit(f"Multimodal fusion failed: {e}")
            return {}
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """Load and preprocess image"""
        try:
            if isinstance(image_path, str):
                image = Image.open(image_path).convert("RGB")
            elif isinstance(image_path, np.ndarray):
                image = Image.fromarray(image_path)
            else:
                image = image_path
            
            # Resize if too large
            max_size = 1024
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error loading image: {e}")
            return None
    
    def _generate_caption(self, image: Image.Image) -> str:
        """Generate image caption using BLIP"""
        try:
            inputs = self.blip_processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return "Unable to generate caption"
    
    def _extract_text(self, image_path: str) -> Dict[str, Any]:
        """Extract text using multiple OCR methods"""
        results = {
            "easyocr": [],
            "tesseract": "",
            "combined_text": "",
            "languages_detected": []
        }
        
        try:
            # EasyOCR (better for Thai)
            if self.easyocr_reader:
                easyocr_results = self.easyocr_reader.readtext(image_path)
                for (bbox, text, confidence) in easyocr_results:
                    if confidence > 0.5:  # Filter low confidence
                        results["easyocr"].append({
                            "text": text,
                            "confidence": confidence,
                            "bbox": bbox
                        })
                
                # Extract just the text
                easyocr_text = " ".join([item["text"] for item in results["easyocr"]])
            
            # Tesseract OCR
            try:
                tesseract_text = pytesseract.image_to_string(
                    Image.open(image_path),
                    lang='eng+tha'  # English and Thai
                )
                results["tesseract"] = tesseract_text.strip()
            except Exception as e:
                self.logger.warning(f"Tesseract OCR failed: {e}")
                results["tesseract"] = ""
            
            # Combine results
            combined_parts = []
            if easyocr_text:
                combined_parts.append(easyocr_text)
            if results["tesseract"]:
                combined_parts.append(results["tesseract"])
            
            results["combined_text"] = " ".join(combined_parts)
            
            # Detect languages
            if any("ก" <= char <= "๛" for char in results["combined_text"]):
                results["languages_detected"].append("thai")
            if any(char.isascii() and char.isalpha() for char in results["combined_text"]):
                results["languages_detected"].append("english")
            
        except Exception as e:
            self.logger.error(f"Error in text extraction: {e}")
        
        return results
    
    def _detect_objects(self, image: Image.Image) -> List[Dict[str, Any]]:
        """Detect objects using CLIP with predefined categories"""
        try:
            # Common object categories
            categories = [
                "person", "car", "building", "tree", "animal", "food", "furniture",
                "computer", "phone", "book", "bottle", "cup", "chair", "table",
                "window", "door", "sign", "text", "logo", "sky", "road", "grass"
            ]
            
            # Create text descriptions
            text_queries = [f"a photo of a {category}" for category in categories]
            
            inputs = self.clip_processor(
                text=text_queries, 
                images=image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], k=5)
            
            objects = []
            for prob, idx in zip(top_probs, top_indices):
                if prob > 0.1:  # Confidence threshold
                    objects.append({
                        "category": categories[idx],
                        "confidence": float(prob)
                    })
            
            return objects
            
        except Exception as e:
            self.logger.error(f"Error in object detection: {e}")
            return []
    
    def _analyze_scene(self, image: Image.Image) -> Dict[str, Any]:
        """Analyze scene context and environment"""
        try:
            # Scene categories
            scene_categories = [
                "indoor scene", "outdoor scene", "natural landscape", "urban environment",
                "office", "home", "restaurant", "street", "park", "beach", "mountain",
                "kitchen", "bedroom", "living room", "bathroom", "classroom"
            ]
            
            text_queries = [f"a photo of {category}" for category in scene_categories]
            
            inputs = self.clip_processor(
                text=text_queries,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top scene predictions
            top_probs, top_indices = torch.topk(probs[0], k=3)
            
            scene_analysis = {
                "primary_scene": scene_categories[top_indices[0]],
                "confidence": float(top_probs[0]),
                "alternative_scenes": []
            }
            
            for i in range(1, len(top_probs)):
                if top_probs[i] > 0.1:
                    scene_analysis["alternative_scenes"].append({
                        "scene": scene_categories[top_indices[i]],
                        "confidence": float(top_probs[i])
                    })
            
            return scene_analysis
            
        except Exception as e:
            self.logger.error(f"Error in scene analysis: {e}")
            return {}
    
    def _get_image_embeddings(self, image: Image.Image) -> np.ndarray:
        """Get CLIP image embeddings for similarity search"""
        try:
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            self.logger.error(f"Error getting image embeddings: {e}")
            return np.array([])
    
    def _extract_video_frames(self, video_path: str, sample_rate: int = 30) -> List[Image.Image]:
        """Extract frames from video"""
        frames = []
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            frame_count = 0
            for frame in container.decode(video_stream):
                if frame_count % sample_rate == 0:
                    # Convert to PIL Image
                    img = frame.to_image()
                    frames.append(img)
                
                frame_count += 1
                
                # Limit number of frames
                if len(frames) >= 20:  # Max 20 frames
                    break
            
            container.close()
            
        except Exception as e:
            self.logger.error(f"Error extracting video frames: {e}")
        
        return frames
    
    def _summarize_video_content(self, scene_descriptions: List[str]) -> str:
        """Generate overall video summary from scene descriptions"""
        try:
            # Simple approach: combine and deduplicate common themes
            all_text = " ".join(scene_descriptions)
            
            # Count common words/phrases
            words = all_text.lower().split()
            word_counts = {}
            for word in words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get most common themes
            common_themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            summary_parts = []
            if common_themes:
                theme_words = [theme[0] for theme in common_themes[:5]]
                summary_parts.append(f"Main themes: {', '.join(theme_words)}")
            
            if len(scene_descriptions) > 1:
                summary_parts.append(f"Video contains {len(scene_descriptions)} distinct scenes")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error summarizing video: {e}")
            return "Unable to generate video summary"
    
    def _analyze_text_context(self, text: str) -> Dict[str, Any]:
        """Analyze text for multimodal context"""
        return {
            "length": len(text),
            "word_count": len(text.split()),
            "has_questions": "?" in text,
            "has_visual_keywords": any(word in text.lower() for word in 
                ["see", "look", "show", "image", "picture", "video", "visual"]),
            "language": "thai" if any("ก" <= char <= "๛" for char in text) else "english"
        }
    
    def _calculate_text_image_similarity(self, text: str, image_path: str) -> float:
        """Calculate similarity between text and image using CLIP"""
        try:
            image = self._load_image(image_path)
            if image is None:
                return 0.0
            
            inputs = self.clip_processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = torch.softmax(logits_per_image, dim=1)[0, 0]
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating text-image similarity: {e}")
            return 0.0
    
    def _generate_multimodal_summary(self, fusion_results: Dict[str, Any]) -> str:
        """Generate comprehensive multimodal summary"""
        try:
            summary_parts = []
            
            # Text analysis
            if "text" in fusion_results["analysis"]:
                text_info = fusion_results["analysis"]["text"]
                summary_parts.append(f"Text input contains {text_info['word_count']} words")
                if text_info["has_visual_keywords"]:
                    summary_parts.append("Request involves visual content")
            
            # Vision analysis
            if "vision" in fusion_results["analysis"]:
                vision_info = fusion_results["analysis"]["vision"]
                if "caption" in vision_info:
                    summary_parts.append(f"Image shows: {vision_info['caption']}")
                if "objects" in vision_info and vision_info["objects"]:
                    objects = [obj["category"] for obj in vision_info["objects"][:3]]
                    summary_parts.append(f"Key objects: {', '.join(objects)}")
            
            # Similarity assessment
            if "text_image_similarity" in fusion_results["confidence_scores"]:
                similarity = fusion_results["confidence_scores"]["text_image_similarity"]
                if similarity > 0.7:
                    summary_parts.append("High relevance between text and image")
                elif similarity > 0.4:
                    summary_parts.append("Moderate relevance between text and image")
                else:
                    summary_parts.append("Low relevance between text and image")
            
            return ". ".join(summary_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating multimodal summary: {e}")
            return "Unable to generate summary"
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get engine status and capabilities"""
        return {
            "is_ready": self.is_ready,
            "models_loaded": self.models_loaded,
            "device": str(self.device),
            "capabilities": {
                "image_captioning": self.blip_model is not None,
                "visual_qa": self.blip_qa_model is not None,
                "ocr": self.easyocr_reader is not None,
                "object_detection": self.clip_model is not None,
                "video_analysis": True,
                "multimodal_fusion": True
            },
            "supported_formats": {
                "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
                "videos": [".mp4", ".avi", ".mov", ".mkv"]
            }
        }
    
    def shutdown(self):
        """Shutdown multimodal engine"""
        self.logger.info("Shutting down multimodal engine")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True)
        
        self.is_ready = False
        self.models_loaded = False
        
        self.logger.info("Multimodal engine shutdown complete")