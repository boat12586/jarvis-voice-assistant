"""
Image Generation System for Jarvis Voice Assistant
Handles image generation requests and manages generated images
"""

import os
import json
import logging
import time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import hashlib
from PyQt6.QtCore import QObject, pyqtSignal, QThread


@dataclass
class ImageRequest:
    """Image generation request structure"""
    request_id: str
    prompt: str
    style: str
    dimensions: tuple
    quality: str
    timestamp: datetime
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "request_id": self.request_id,
            "prompt": self.prompt,
            "style": self.style,
            "dimensions": self.dimensions,
            "quality": self.quality,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class GeneratedImage:
    """Generated image structure"""
    image_id: str
    request: ImageRequest
    file_path: str
    thumbnail_path: str
    generation_time: float
    success: bool
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "image_id": self.image_id,
            "request": self.request.to_dict(),
            "file_path": self.file_path,
            "thumbnail_path": self.thumbnail_path,
            "generation_time": self.generation_time,
            "success": self.success,
            "error_message": self.error_message
        }


class MockImageGenerator:
    """Mock image generator for demonstration"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Style templates
        self.style_templates = {
            "realistic": {
                "colors": [(64, 128, 192), (192, 128, 64), (128, 192, 64)],
                "patterns": "gradient"
            },
            "artistic": {
                "colors": [(255, 100, 100), (100, 255, 100), (100, 100, 255)],
                "patterns": "abstract"
            },
            "cartoon": {
                "colors": [(255, 200, 100), (100, 255, 200), (200, 100, 255)],
                "patterns": "simple"
            },
            "cyberpunk": {
                "colors": [(255, 0, 255), (0, 255, 255), (255, 255, 0)],
                "patterns": "neon"
            },
            "nature": {
                "colors": [(34, 139, 34), (139, 69, 19), (70, 130, 180)],
                "patterns": "organic"
            }
        }
        
        # Prompt keywords for visual elements
        self.prompt_keywords = {
            "landscape": ["mountain", "forest", "ocean", "sunset", "field"],
            "portrait": ["person", "face", "character", "human", "people"],
            "abstract": ["pattern", "geometric", "design", "art", "color"],
            "technology": ["robot", "ai", "computer", "future", "digital"],
            "fantasy": ["dragon", "magic", "castle", "wizard", "mythical"]
        }
    
    def generate_image(self, request: ImageRequest) -> GeneratedImage:
        """Generate a mock image based on the request"""
        start_time = time.time()
        
        try:
            # Create output directory
            output_dir = Path(__file__).parent.parent.parent / "data" / "generated_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate image filename
            image_filename = f"{request.request_id}.png"
            image_path = output_dir / image_filename
            
            # Generate thumbnail filename
            thumbnail_filename = f"{request.request_id}_thumb.png"
            thumbnail_path = output_dir / thumbnail_filename
            
            # Create mock image
            width, height = request.dimensions
            image = Image.new('RGB', (width, height), color='white')
            draw = ImageDraw.Draw(image)
            
            # Get style configuration
            style_config = self.style_templates.get(request.style, self.style_templates["realistic"])
            colors = style_config["colors"]
            
            # Analyze prompt for visual elements
            visual_elements = self._analyze_prompt(request.prompt)
            
            # Generate background
            self._generate_background(draw, width, height, colors, style_config["patterns"])
            
            # Add visual elements based on prompt
            self._add_visual_elements(draw, width, height, visual_elements, colors)
            
            # Add text overlay with prompt
            self._add_text_overlay(draw, width, height, request.prompt[:50])
            
            # Save main image
            image.save(image_path, "PNG")
            
            # Generate thumbnail
            thumbnail = image.copy()
            thumbnail.thumbnail((200, 200), Image.Resampling.LANCZOS)
            thumbnail.save(thumbnail_path, "PNG")
            
            generation_time = time.time() - start_time
            
            self.logger.info(f"Generated mock image: {image_filename}")
            
            return GeneratedImage(
                image_id=request.request_id,
                request=request,
                file_path=str(image_path),
                thumbnail_path=str(thumbnail_path),
                generation_time=generation_time,
                success=True
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate image: {e}")
            
            return GeneratedImage(
                image_id=request.request_id,
                request=request,
                file_path="",
                thumbnail_path="",
                generation_time=time.time() - start_time,
                success=False,
                error_message=str(e)
            )
    
    def _analyze_prompt(self, prompt: str) -> List[str]:
        """Analyze prompt to identify visual elements"""
        visual_elements = []
        prompt_lower = prompt.lower()
        
        for category, keywords in self.prompt_keywords.items():
            for keyword in keywords:
                if keyword in prompt_lower:
                    visual_elements.append(category)
                    break
        
        if not visual_elements:
            visual_elements = ["abstract"]
        
        return visual_elements
    
    def _generate_background(self, draw: ImageDraw.Draw, width: int, height: int, 
                            colors: List[tuple], pattern: str):
        """Generate background pattern"""
        if pattern == "gradient":
            # Simple gradient effect
            for i in range(height):
                ratio = i / height
                color = self._blend_colors(colors[0], colors[1], ratio)
                draw.line([(0, i), (width, i)], fill=color)
        
        elif pattern == "abstract":
            # Abstract shapes
            for _ in range(10):
                x1, y1 = width // 4, height // 4
                x2, y2 = 3 * width // 4, 3 * height // 4
                color = colors[_ % len(colors)]
                draw.ellipse([x1 + _ * 20, y1 + _ * 15, x2 - _ * 20, y2 - _ * 15], 
                           fill=color, outline=None)
        
        elif pattern == "simple":
            # Simple colored shapes
            sections = len(colors)
            section_width = width // sections
            
            for i, color in enumerate(colors):
                x1 = i * section_width
                x2 = (i + 1) * section_width
                draw.rectangle([x1, 0, x2, height], fill=color)
        
        elif pattern == "neon":
            # Neon-like effects
            draw.rectangle([0, 0, width, height], fill=(0, 0, 0))  # Black background
            
            # Neon lines
            for i in range(0, width, 50):
                color = colors[i % len(colors)]
                draw.line([(i, 0), (i, height)], fill=color, width=3)
            
            for i in range(0, height, 50):
                color = colors[i % len(colors)]
                draw.line([(0, i), (width, i)], fill=color, width=3)
        
        elif pattern == "organic":
            # Organic patterns
            draw.rectangle([0, 0, width, height], fill=colors[0])
            
            # Organic shapes
            for _ in range(5):
                x = width // 6 + _ * width // 8
                y = height // 6 + _ * height // 8
                radius = 50 + _ * 20
                color = colors[_ % len(colors)]
                draw.ellipse([x - radius, y - radius, x + radius, y + radius], 
                           fill=color, outline=None)
        
        else:
            # Default solid color
            draw.rectangle([0, 0, width, height], fill=colors[0])
    
    def _add_visual_elements(self, draw: ImageDraw.Draw, width: int, height: int, 
                           visual_elements: List[str], colors: List[tuple]):
        """Add visual elements based on prompt analysis"""
        for element in visual_elements:
            if element == "landscape":
                # Simple landscape elements
                # Mountains
                points = [(0, height * 0.7), (width * 0.3, height * 0.4), 
                         (width * 0.7, height * 0.5), (width, height * 0.6), 
                         (width, height), (0, height)]
                draw.polygon(points, fill=colors[1])
                
                # Sun
                sun_center = (width * 0.8, height * 0.2)
                sun_radius = 30
                draw.ellipse([sun_center[0] - sun_radius, sun_center[1] - sun_radius,
                            sun_center[0] + sun_radius, sun_center[1] + sun_radius], 
                           fill=(255, 255, 0))
            
            elif element == "portrait":
                # Simple portrait elements
                # Face outline
                face_center = (width // 2, height // 2)
                face_width, face_height = 100, 120
                draw.ellipse([face_center[0] - face_width//2, face_center[1] - face_height//2,
                            face_center[0] + face_width//2, face_center[1] + face_height//2], 
                           fill=colors[2], outline=colors[0], width=3)
                
                # Eyes
                eye_y = face_center[1] - 20
                draw.ellipse([face_center[0] - 25, eye_y - 5, face_center[0] - 15, eye_y + 5], 
                           fill=(0, 0, 0))
                draw.ellipse([face_center[0] + 15, eye_y - 5, face_center[0] + 25, eye_y + 5], 
                           fill=(0, 0, 0))
                
                # Mouth
                mouth_y = face_center[1] + 20
                draw.arc([face_center[0] - 20, mouth_y - 10, face_center[0] + 20, mouth_y + 10], 
                        start=0, end=180, fill=(0, 0, 0), width=2)
            
            elif element == "technology":
                # Technology elements
                # Circuit-like patterns
                for i in range(5):
                    x = width // 6 + i * width // 6
                    y = height // 6 + i * height // 6
                    
                    # Squares
                    draw.rectangle([x - 10, y - 10, x + 10, y + 10], 
                                 fill=colors[i % len(colors)], outline=(0, 0, 0), width=2)
                    
                    # Connecting lines
                    if i < 4:
                        next_x = width // 6 + (i + 1) * width // 6
                        next_y = height // 6 + (i + 1) * height // 6
                        draw.line([(x + 10, y), (next_x - 10, next_y)], 
                                fill=(0, 0, 0), width=2)
            
            elif element == "fantasy":
                # Fantasy elements
                # Castle-like structure
                castle_x = width // 2
                castle_y = height * 0.8
                castle_width = 80
                castle_height = 100
                
                # Castle base
                draw.rectangle([castle_x - castle_width//2, castle_y - castle_height,
                              castle_x + castle_width//2, castle_y], 
                             fill=colors[1], outline=(0, 0, 0), width=2)
                
                # Castle towers
                tower_width = 20
                tower_height = 40
                for i in [-1, 1]:
                    tower_x = castle_x + i * castle_width//3
                    draw.rectangle([tower_x - tower_width//2, castle_y - castle_height - tower_height,
                                  tower_x + tower_width//2, castle_y - castle_height], 
                                 fill=colors[2], outline=(0, 0, 0), width=2)
                
                # Stars
                for _ in range(5):
                    star_x = width // 6 + _ * width // 6
                    star_y = height // 6
                    self._draw_star(draw, star_x, star_y, 8, (255, 255, 0))
    
    def _draw_star(self, draw: ImageDraw.Draw, x: int, y: int, size: int, color: tuple):
        """Draw a simple star"""
        points = []
        for i in range(10):
            angle = i * 36 * 3.14159 / 180
            radius = size if i % 2 == 0 else size // 2
            px = x + radius * (1 if i % 4 < 2 else -1)
            py = y + radius * (1 if i % 4 == 0 or i % 4 == 3 else -1)
            points.append((px, py))
        
        draw.polygon(points, fill=color, outline=(0, 0, 0))
    
    def _add_text_overlay(self, draw: ImageDraw.Draw, width: int, height: int, text: str):
        """Add text overlay to the image"""
        try:
            # Try to use a system font
            font = ImageFont.truetype("arial.ttf", 20)
        except:
            # Fallback to default font
            font = ImageFont.load_default()
        
        # Calculate text size and position
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        text_x = (width - text_width) // 2
        text_y = height - text_height - 20
        
        # Draw text background
        padding = 10
        draw.rectangle([text_x - padding, text_y - padding, 
                       text_x + text_width + padding, text_y + text_height + padding], 
                      fill=(255, 255, 255, 128))
        
        # Draw text
        draw.text((text_x, text_y), text, fill=(0, 0, 0), font=font)
    
    def _blend_colors(self, color1: tuple, color2: tuple, ratio: float) -> tuple:
        """Blend two colors with given ratio"""
        r1, g1, b1 = color1
        r2, g2, b2 = color2
        
        r = int(r1 * (1 - ratio) + r2 * ratio)
        g = int(g1 * (1 - ratio) + g2 * ratio)
        b = int(b1 * (1 - ratio) + b2 * ratio)
        
        return (r, g, b)


class ImageGenerationThread(QThread):
    """Thread for image generation"""
    
    # Signals
    generation_complete = pyqtSignal(dict)  # generated image info
    generation_error = pyqtSignal(str)
    
    def __init__(self, request: ImageRequest, generator: MockImageGenerator):
        super().__init__()
        self.request = request
        self.generator = generator
        self.logger = logging.getLogger(__name__)
    
    def run(self):
        """Run image generation in thread"""
        try:
            # Generate image
            result = self.generator.generate_image(self.request)
            
            # Emit result
            self.generation_complete.emit(result.to_dict())
            
        except Exception as e:
            self.logger.error(f"Image generation thread error: {e}")
            self.generation_error.emit(str(e))


class ImageGenerationSystem(QObject):
    """Main image generation system controller"""
    
    # Signals
    generation_started = pyqtSignal(str)  # request_id
    generation_complete = pyqtSignal(dict)  # generated image info
    generation_error = pyqtSignal(str)
    error_occurred = pyqtSignal(str)  # Add missing signal for compatibility
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Configuration
        self.supported_styles = config.get("supported_styles", 
                                          ["realistic", "artistic", "cartoon", "cyberpunk", "nature"])
        self.default_dimensions = config.get("default_dimensions", (512, 512))
        self.max_image_size = config.get("max_image_size", (1024, 1024))
        self.quality_levels = config.get("quality_levels", ["low", "medium", "high"])
        
        # Components
        self.image_generator = MockImageGenerator()
        
        # Image storage
        self.generated_images: Dict[str, GeneratedImage] = {}
        self.generation_history: List[ImageRequest] = []
        
        # Active generations
        self.active_generations: Dict[str, ImageGenerationThread] = {}
        
        # Initialize
        self._initialize()
        
        # Connect internal signals
        self.generation_error.connect(self.error_occurred.emit)
        
        self.logger.info("Image generation system initialized")
    
    def _initialize(self):
        """Initialize image generation system"""
        try:
            # Load generation history
            self._load_generation_history()
            
            # Create output directory
            output_dir = Path(__file__).parent.parent.parent / "data" / "generated_images"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.logger.info("Image generation system ready")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize image generation system: {e}")
    
    def generate_image(self, prompt: str, style: str = "realistic", 
                      dimensions: tuple = None, quality: str = "medium") -> str:
        """Generate an image based on prompt and parameters"""
        try:
            # Validate parameters
            if not prompt or not prompt.strip():
                raise ValueError("Empty prompt provided")
            
            if style not in self.supported_styles:
                style = "realistic"
            
            if dimensions is None:
                dimensions = self.default_dimensions
            
            # Validate dimensions
            if (dimensions[0] > self.max_image_size[0] or 
                dimensions[1] > self.max_image_size[1]):
                dimensions = self.max_image_size
            
            if quality not in self.quality_levels:
                quality = "medium"
            
            # Generate request ID
            request_id = hashlib.md5(
                f"{prompt}_{style}_{dimensions}_{quality}_{time.time()}".encode()
            ).hexdigest()
            
            # Create request
            request = ImageRequest(
                request_id=request_id,
                prompt=prompt,
                style=style,
                dimensions=dimensions,
                quality=quality,
                timestamp=datetime.now()
            )
            
            # Add to history
            self.generation_history.append(request)
            
            # Start generation thread
            generation_thread = ImageGenerationThread(request, self.image_generator)
            generation_thread.generation_complete.connect(self._on_generation_complete)
            generation_thread.generation_error.connect(self._on_generation_error)
            
            self.active_generations[request_id] = generation_thread
            generation_thread.start()
            
            # Emit started signal
            self.generation_started.emit(request_id)
            
            self.logger.info(f"Started image generation: {prompt[:50]}...")
            
            return request_id
            
        except Exception as e:
            self.logger.error(f"Failed to start image generation: {e}")
            self.generation_error.emit(f"Failed to start image generation: {e}")
            return None
    
    def _on_generation_complete(self, result_dict: Dict[str, Any]):
        """Handle generation completion"""
        try:
            # Reconstruct result
            request_data = result_dict["request"]
            request = ImageRequest(
                request_id=request_data["request_id"],
                prompt=request_data["prompt"],
                style=request_data["style"],
                dimensions=tuple(request_data["dimensions"]),
                quality=request_data["quality"],
                timestamp=datetime.fromisoformat(request_data["timestamp"])
            )
            
            result = GeneratedImage(
                image_id=result_dict["image_id"],
                request=request,
                file_path=result_dict["file_path"],
                thumbnail_path=result_dict["thumbnail_path"],
                generation_time=result_dict["generation_time"],
                success=result_dict["success"],
                error_message=result_dict.get("error_message")
            )
            
            # Store result
            self.generated_images[result.image_id] = result
            
            # Clean up thread
            if result.image_id in self.active_generations:
                del self.active_generations[result.image_id]
            
            # Save history
            self._save_generation_history()
            
            self.logger.info(f"Image generation completed: {result.image_id}")
            
            # Emit signal
            self.generation_complete.emit(result_dict)
            
        except Exception as e:
            self.logger.error(f"Error handling generation completion: {e}")
            self.generation_error.emit(f"Error handling generation completion: {e}")
    
    def _on_generation_error(self, error_msg: str):
        """Handle generation error"""
        self.logger.error(f"Image generation error: {error_msg}")
        self.generation_error.emit(error_msg)
    
    def get_generated_images(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get list of generated images"""
        try:
            # Get most recent images
            recent_images = list(self.generated_images.values())
            recent_images.sort(key=lambda x: x.request.timestamp, reverse=True)
            
            if limit > 0:
                recent_images = recent_images[:limit]
            
            return [image.to_dict() for image in recent_images]
            
        except Exception as e:
            self.logger.error(f"Failed to get generated images: {e}")
            return []
    
    def get_image_info(self, image_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific image"""
        try:
            if image_id in self.generated_images:
                return self.generated_images[image_id].to_dict()
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get image info: {e}")
            return None
    
    def get_supported_styles(self) -> List[str]:
        """Get list of supported styles"""
        return self.supported_styles.copy()
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """Get generation statistics"""
        try:
            total_generated = len(self.generated_images)
            successful_generations = len([img for img in self.generated_images.values() if img.success])
            
            # Average generation time
            generation_times = [img.generation_time for img in self.generated_images.values() if img.success]
            avg_generation_time = sum(generation_times) / len(generation_times) if generation_times else 0
            
            # Style distribution
            style_counts = {}
            for image in self.generated_images.values():
                style = image.request.style
                style_counts[style] = style_counts.get(style, 0) + 1
            
            return {
                "total_generated": total_generated,
                "successful_generations": successful_generations,
                "failed_generations": total_generated - successful_generations,
                "success_rate": (successful_generations / total_generated * 100) if total_generated > 0 else 0,
                "average_generation_time": avg_generation_time,
                "style_distribution": style_counts,
                "active_generations": len(self.active_generations),
                "supported_styles": self.supported_styles,
                "max_image_size": self.max_image_size
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get generation stats: {e}")
            return {}
    
    def _load_generation_history(self):
        """Load generation history from file"""
        try:
            history_file = Path(__file__).parent.parent.parent / "data" / "image_generation_history.json"
            
            if history_file.exists():
                with open(history_file, 'r', encoding='utf-8') as f:
                    history_data = json.load(f)
                
                # Load requests
                for request_data in history_data.get("requests", []):
                    request = ImageRequest(
                        request_id=request_data["request_id"],
                        prompt=request_data["prompt"],
                        style=request_data["style"],
                        dimensions=tuple(request_data["dimensions"]),
                        quality=request_data["quality"],
                        timestamp=datetime.fromisoformat(request_data["timestamp"])
                    )
                    self.generation_history.append(request)
                
                # Load generated images
                for image_data in history_data.get("images", []):
                    request_data = image_data["request"]
                    request = ImageRequest(
                        request_id=request_data["request_id"],
                        prompt=request_data["prompt"],
                        style=request_data["style"],
                        dimensions=tuple(request_data["dimensions"]),
                        quality=request_data["quality"],
                        timestamp=datetime.fromisoformat(request_data["timestamp"])
                    )
                    
                    image = GeneratedImage(
                        image_id=image_data["image_id"],
                        request=request,
                        file_path=image_data["file_path"],
                        thumbnail_path=image_data["thumbnail_path"],
                        generation_time=image_data["generation_time"],
                        success=image_data["success"],
                        error_message=image_data.get("error_message")
                    )
                    
                    self.generated_images[image.image_id] = image
                
                self.logger.info(f"Loaded {len(self.generation_history)} generation requests and {len(self.generated_images)} generated images")
                
        except Exception as e:
            self.logger.error(f"Failed to load generation history: {e}")
    
    def _save_generation_history(self):
        """Save generation history to file"""
        try:
            history_file = Path(__file__).parent.parent.parent / "data" / "image_generation_history.json"
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            history_data = {
                "requests": [request.to_dict() for request in self.generation_history],
                "images": [image.to_dict() for image in self.generated_images.values()]
            }
            
            with open(history_file, 'w', encoding='utf-8') as f:
                json.dump(history_data, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            self.logger.error(f"Failed to save generation history: {e}")
    
    def shutdown(self):
        """Shutdown image generation system"""
        self.logger.info("Shutting down image generation system")
        
        # Stop all active generations
        for generation_thread in self.active_generations.values():
            if generation_thread.isRunning():
                generation_thread.quit()
                generation_thread.wait()
        
        # Save history
        self._save_generation_history()
        
        self.logger.info("Image generation system shutdown complete")