"""
Computer Vision Models for JARVIS Voice Assistant
Specialized handlers for CLIP, BLIP, and other vision models
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import time
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# Model imports
from transformers import (
    CLIPProcessor, CLIPModel,
    BlipProcessor, BlipForConditionalGeneration,
    BlipForQuestionAnswering,
    AutoProcessor, AutoModel,
    OwlViTProcessor, OwlViTForObjectDetection
)
from ultralytics import YOLO
import timm


class VisionModelManager:
    """Manages different computer vision models"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model instances
        self.models = {}
        self.processors = {}
        
        # Performance tracking
        self.model_stats = {}
        
        self.logger.info(f"Vision models using device: {self.device}")
    
    def load_clip_model(self, model_name: str = "openai/clip-vit-base-patch32") -> bool:
        """Load CLIP model for image-text understanding"""
        try:
            self.logger.info(f"Loading CLIP model: {model_name}")
            
            processor = CLIPProcessor.from_pretrained(model_name)
            model = CLIPModel.from_pretrained(model_name).to(self.device)
            
            # Test the model
            test_image = Image.new('RGB', (224, 224), color='white')
            test_inputs = processor(
                text=["a test image"], 
                images=test_image, 
                return_tensors="pt", 
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**test_inputs)
                self.logger.info("CLIP model test successful")
            
            self.models['clip'] = model
            self.processors['clip'] = processor
            self.model_stats['clip'] = {'loaded_at': time.time(), 'usage_count': 0}
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load CLIP model: {e}")
            return False
    
    def load_blip_models(self, 
                        caption_model: str = "Salesforce/blip-image-captioning-base",
                        qa_model: str = "Salesforce/blip-vqa-base") -> bool:
        """Load BLIP models for captioning and VQA"""
        try:
            # Load captioning model
            self.logger.info(f"Loading BLIP captioning model: {caption_model}")
            
            caption_processor = BlipProcessor.from_pretrained(caption_model)
            caption_model_instance = BlipForConditionalGeneration.from_pretrained(caption_model).to(self.device)
            
            # Load VQA model
            self.logger.info(f"Loading BLIP VQA model: {qa_model}")
            
            qa_processor = BlipProcessor.from_pretrained(qa_model)
            qa_model_instance = BlipForQuestionAnswering.from_pretrained(qa_model).to(self.device)
            
            # Test models
            test_image = Image.new('RGB', (224, 224), color='blue')
            
            # Test captioning
            caption_inputs = caption_processor(test_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                caption_outputs = caption_model_instance.generate(**caption_inputs, max_length=20)
                test_caption = caption_processor.decode(caption_outputs[0], skip_special_tokens=True)
                self.logger.info(f"BLIP caption test: {test_caption}")
            
            # Test VQA
            qa_inputs = qa_processor(test_image, "What color is this?", return_tensors="pt").to(self.device)
            with torch.no_grad():
                qa_outputs = qa_model_instance.generate(**qa_inputs, max_length=20)
                test_answer = qa_processor.decode(qa_outputs[0], skip_special_tokens=True)
                self.logger.info(f"BLIP VQA test: {test_answer}")
            
            self.models['blip_caption'] = caption_model_instance
            self.processors['blip_caption'] = caption_processor
            self.models['blip_qa'] = qa_model_instance
            self.processors['blip_qa'] = qa_processor
            
            self.model_stats['blip_caption'] = {'loaded_at': time.time(), 'usage_count': 0}
            self.model_stats['blip_qa'] = {'loaded_at': time.time(), 'usage_count': 0}
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load BLIP models: {e}")
            return False
    
    def load_yolo_model(self, model_size: str = "yolov8n") -> bool:
        """Load YOLO model for object detection"""
        try:
            self.logger.info(f"Loading YOLO model: {model_size}")
            
            model = YOLO(f"{model_size}.pt")
            
            # Test the model
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            results = model(test_image, verbose=False)
            self.logger.info("YOLO model test successful")
            
            self.models['yolo'] = model
            self.model_stats['yolo'] = {'loaded_at': time.time(), 'usage_count': 0}
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load YOLO model: {e}")
            return False
    
    def generate_image_caption(self, image: Union[str, Image.Image, np.ndarray], 
                             max_length: int = 50, num_beams: int = 5) -> str:
        """Generate caption for image using BLIP"""
        if 'blip_caption' not in self.models:
            raise ValueError("BLIP caption model not loaded")
        
        try:
            # Prepare image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            model = self.models['blip_caption']
            processor = self.processors['blip_caption']
            
            inputs = processor(image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length, num_beams=num_beams)
                caption = processor.decode(outputs[0], skip_special_tokens=True)
            
            self.model_stats['blip_caption']['usage_count'] += 1
            return caption
            
        except Exception as e:
            self.logger.error(f"Error generating caption: {e}")
            return "Unable to generate caption"
    
    def answer_visual_question(self, image: Union[str, Image.Image, np.ndarray], 
                              question: str, max_length: int = 50) -> str:
        """Answer question about image using BLIP VQA"""
        if 'blip_qa' not in self.models:
            raise ValueError("BLIP VQA model not loaded")
        
        try:
            # Prepare image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            model = self.models['blip_qa']
            processor = self.processors['blip_qa']
            
            inputs = processor(image, question, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length)
                answer = processor.decode(outputs[0], skip_special_tokens=True)
            
            self.model_stats['blip_qa']['usage_count'] += 1
            return answer
            
        except Exception as e:
            self.logger.error(f"Error answering visual question: {e}")
            return "Unable to answer question"
    
    def calculate_image_text_similarity(self, image: Union[str, Image.Image, np.ndarray], 
                                       text: str) -> float:
        """Calculate similarity between image and text using CLIP"""
        if 'clip' not in self.models:
            raise ValueError("CLIP model not loaded")
        
        try:
            # Prepare image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            model = self.models['clip']
            processor = self.processors['clip']
            
            inputs = processor(
                text=[text],
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                similarity = torch.softmax(logits_per_image, dim=1)[0, 0]
            
            self.model_stats['clip']['usage_count'] += 1
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def classify_image_with_categories(self, image: Union[str, Image.Image, np.ndarray], 
                                     categories: List[str], top_k: int = 3) -> List[Dict[str, Any]]:
        """Classify image against given categories using CLIP"""
        if 'clip' not in self.models:
            raise ValueError("CLIP model not loaded")
        
        try:
            # Prepare image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            model = self.models['clip']
            processor = self.processors['clip']
            
            # Create text descriptions
            text_queries = [f"a photo of {category}" for category in categories]
            
            inputs = processor(
                text=text_queries,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Get top predictions
            top_probs, top_indices = torch.topk(probs[0], k=min(top_k, len(categories)))
            
            results = []
            for prob, idx in zip(top_probs, top_indices):
                results.append({
                    "category": categories[idx],
                    "confidence": float(prob)
                })
            
            self.model_stats['clip']['usage_count'] += 1
            return results
            
        except Exception as e:
            self.logger.error(f"Error in image classification: {e}")
            return []
    
    def detect_objects_yolo(self, image: Union[str, Image.Image, np.ndarray], 
                           confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Detect objects using YOLO"""
        if 'yolo' not in self.models:
            raise ValueError("YOLO model not loaded")
        
        try:
            # Prepare image
            if isinstance(image, str):
                image_array = cv2.imread(image)
                image_array = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
            elif isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            model = self.models['yolo']
            results = model(image_array, verbose=False)
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        confidence = float(box.conf[0])
                        if confidence >= confidence_threshold:
                            class_id = int(box.cls[0])
                            class_name = model.names[class_id]
                            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            
                            detections.append({
                                "class": class_name,
                                "confidence": confidence,
                                "bbox": bbox,
                                "center": [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2],
                                "area": (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                            })
            
            self.model_stats['yolo']['usage_count'] += 1
            return detections
            
        except Exception as e:
            self.logger.error(f"Error in YOLO object detection: {e}")
            return []
    
    def get_image_embeddings(self, image: Union[str, Image.Image, np.ndarray]) -> np.ndarray:
        """Get CLIP image embeddings for similarity search"""
        if 'clip' not in self.models:
            raise ValueError("CLIP model not loaded")
        
        try:
            # Prepare image
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            elif isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            
            model = self.models['clip']
            processor = self.processors['clip']
            
            inputs = processor(images=image, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            self.model_stats['clip']['usage_count'] += 1
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            self.logger.error(f"Error getting image embeddings: {e}")
            return np.array([])
    
    def enhance_image_quality(self, image: Union[str, Image.Image], 
                            enhancement_type: str = "auto") -> Image.Image:
        """Enhance image quality for better model performance"""
        try:
            if isinstance(image, str):
                image = Image.open(image).convert("RGB")
            
            if enhancement_type == "auto":
                # Auto-enhance based on image properties
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(1.1)  # Slight color boost
                
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.1)  # Slight contrast boost
                
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.05)  # Slight sharpening
                
            elif enhancement_type == "contrast":
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(1.2)
                
            elif enhancement_type == "brightness":
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(1.1)
                
            elif enhancement_type == "sharpness":
                enhancer = ImageEnhance.Sharpness(image)
                image = enhancer.enhance(1.2)
            
            return image
            
        except Exception as e:
            self.logger.error(f"Error enhancing image: {e}")
            return image if isinstance(image, Image.Image) else Image.new('RGB', (224, 224))
    
    def batch_process_images(self, image_paths: List[str], 
                           operation: str, **kwargs) -> List[Dict[str, Any]]:
        """Process multiple images in batch"""
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                if operation == "caption":
                    result = self.generate_image_caption(image_path, **kwargs)
                elif operation == "classify":
                    result = self.classify_image_with_categories(image_path, **kwargs)
                elif operation == "detect":
                    result = self.detect_objects_yolo(image_path, **kwargs)
                elif operation == "embeddings":
                    result = self.get_image_embeddings(image_path)
                else:
                    result = f"Unknown operation: {operation}"
                
                results.append({
                    "image_path": image_path,
                    "result": result,
                    "success": True
                })
                
            except Exception as e:
                self.logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    "image_path": image_path,
                    "result": None,
                    "success": False,
                    "error": str(e)
                })
        
        return results
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about loaded models"""
        info = {
            "device": str(self.device),
            "loaded_models": list(self.models.keys()),
            "model_stats": self.model_stats,
            "gpu_available": torch.cuda.is_available()
        }
        
        if torch.cuda.is_available():
            info["gpu_memory"] = {
                "allocated": torch.cuda.memory_allocated(self.device),
                "cached": torch.cuda.memory_reserved(self.device)
            }
        
        return info
    
    def unload_model(self, model_name: str) -> bool:
        """Unload specific model to free memory"""
        try:
            if model_name in self.models:
                del self.models[model_name]
                del self.processors[model_name]
                del self.model_stats[model_name]
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                self.logger.info(f"Model {model_name} unloaded successfully")
                return True
            else:
                self.logger.warning(f"Model {model_name} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error unloading model {model_name}: {e}")
            return False
    
    def clear_all_models(self):
        """Clear all loaded models"""
        for model_name in list(self.models.keys()):
            self.unload_model(model_name)
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        self.logger.info("All vision models cleared")


class ImagePreprocessor:
    """Advanced image preprocessing for optimal model performance"""
    
    @staticmethod
    def resize_and_pad(image: Image.Image, target_size: Tuple[int, int], 
                      fill_color: Tuple[int, int, int] = (255, 255, 255)) -> Image.Image:
        """Resize image while maintaining aspect ratio with padding"""
        original_size = image.size
        target_width, target_height = target_size
        
        # Calculate resize ratio
        ratio = min(target_width / original_size[0], target_height / original_size[1])
        new_size = (int(original_size[0] * ratio), int(original_size[1] * ratio))
        
        # Resize image
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create new image with target size and fill color
        new_image = Image.new("RGB", target_size, fill_color)
        
        # Paste resized image in center
        paste_x = (target_width - new_size[0]) // 2
        paste_y = (target_height - new_size[1]) // 2
        new_image.paste(resized, (paste_x, paste_y))
        
        return new_image
    
    @staticmethod
    def normalize_lighting(image: Image.Image) -> Image.Image:
        """Normalize image lighting conditions"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        img_array = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def remove_noise(image: Image.Image) -> Image.Image:
        """Remove noise from image"""
        # Convert to numpy array
        img_array = np.array(image)
        
        # Apply bilateral filter
        denoised = cv2.bilateralFilter(img_array, 9, 75, 75)
        
        return Image.fromarray(denoised)