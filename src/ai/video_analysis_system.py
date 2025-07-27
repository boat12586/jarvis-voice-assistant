"""
Video Analysis System for JARVIS Voice Assistant
Advanced video processing with scene understanding and temporal analysis
"""

import logging
import cv2
import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path
import time
import json
from datetime import datetime, timedelta
import threading
import queue
from PIL import Image
import imageio
import av

# ML and Vision imports
import torch
from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity


class VideoAnalysisSystem:
    """Advanced video analysis with scene detection and content understanding"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Model components
        self.clip_model = None
        self.clip_processor = None
        self.blip_model = None
        self.blip_processor = None
        
        # Processing parameters
        self.default_fps = config.get('target_fps', 1.0)  # Frames per second for analysis
        self.max_frames = config.get('max_frames', 100)
        self.scene_threshold = config.get('scene_threshold', 0.3)
        
        # Analysis state
        self.current_analysis = None
        self.processing_queue = queue.Queue()
        
        # Performance tracking
        self.analysis_stats = {
            'videos_processed': 0,
            'total_frames_analyzed': 0,
            'avg_processing_time': 0,
            'scene_detection_accuracy': 0
        }
        
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize video analysis models"""
        try:
            # Load CLIP for scene understanding
            clip_model_name = self.config.get('clip_model', 'openai/clip-vit-base-patch32')
            self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)
            self.clip_model = CLIPModel.from_pretrained(clip_model_name).to(self.device)
            
            # Load BLIP for frame captioning
            blip_model_name = self.config.get('blip_model', 'Salesforce/blip-image-captioning-base')
            self.blip_processor = BlipProcessor.from_pretrained(blip_model_name)
            self.blip_model = BlipForConditionalGeneration.from_pretrained(blip_model_name).to(self.device)
            
            self.logger.info("Video analysis models loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize video analysis models: {e}")
            raise
    
    def analyze_video(self, video_path: str, 
                     analysis_type: str = "comprehensive",
                     sample_rate: Optional[float] = None) -> Dict[str, Any]:
        """Analyze video with comprehensive scene understanding"""
        
        start_time = time.time()
        
        try:
            # Validate video file
            if not Path(video_path).exists():
                raise FileNotFoundError(f"Video file not found: {video_path}")
            
            # Get video metadata
            video_info = self._get_video_metadata(video_path)
            
            # Determine sampling rate
            if sample_rate is None:
                sample_rate = self._calculate_optimal_sample_rate(video_info)
            
            # Extract frames for analysis
            frames, timestamps = self._extract_frames(video_path, sample_rate)
            
            if not frames:
                raise ValueError("No frames could be extracted from video")
            
            # Perform analysis based on type
            if analysis_type == "comprehensive":
                analysis_results = self._comprehensive_analysis(frames, timestamps, video_info)
            elif analysis_type == "scene_detection":
                analysis_results = self._scene_detection_analysis(frames, timestamps)
            elif analysis_type == "content_summary":
                analysis_results = self._content_summary_analysis(frames, timestamps)
            elif analysis_type == "temporal_analysis":
                analysis_results = self._temporal_analysis(frames, timestamps)
            else:
                raise ValueError(f"Unknown analysis type: {analysis_type}")
            
            # Compile final results
            results = {
                'video_path': video_path,
                'analysis_type': analysis_type,
                'video_metadata': video_info,
                'processing_time': time.time() - start_time,
                'frames_analyzed': len(frames),
                'sample_rate_used': sample_rate,
                'timestamp': datetime.now().isoformat(),
                **analysis_results
            }
            
            # Update statistics
            self._update_analysis_stats(results)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Video analysis failed for {video_path}: {e}")
            return self._create_error_result(video_path, str(e))
    
    def _get_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata"""
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            metadata = {
                'duration': float(container.duration / av.time_base),
                'fps': float(video_stream.average_rate),
                'width': video_stream.width,
                'height': video_stream.height,
                'total_frames': video_stream.frames,
                'codec': video_stream.codec.name,
                'format': container.format.name
            }
            
            container.close()
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to get video metadata: {e}")
            # Fallback to OpenCV
            try:
                cap = cv2.VideoCapture(video_path)
                metadata = {
                    'duration': cap.get(cv2.CAP_PROP_FRAME_COUNT) / cap.get(cv2.CAP_PROP_FPS),
                    'fps': cap.get(cv2.CAP_PROP_FPS),
                    'width': int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    'height': int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    'total_frames': int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    'codec': 'unknown',
                    'format': 'unknown'
                }
                cap.release()
                return metadata
            except:
                return {'duration': 0, 'fps': 30, 'width': 0, 'height': 0, 'total_frames': 0}
    
    def _calculate_optimal_sample_rate(self, video_info: Dict[str, Any]) -> float:
        """Calculate optimal sampling rate for analysis"""
        duration = video_info.get('duration', 0)
        fps = video_info.get('fps', 30)
        
        # Adaptive sampling based on video length
        if duration <= 30:  # Short video
            return min(fps / 5, 2.0)  # Every 5th frame or 2 FPS max
        elif duration <= 300:  # Medium video (5 minutes)
            return min(fps / 10, 1.0)  # Every 10th frame or 1 FPS max
        else:  # Long video
            return min(fps / 30, 0.5)  # Every 30th frame or 0.5 FPS max
    
    def _extract_frames(self, video_path: str, sample_rate: float) -> Tuple[List[Image.Image], List[float]]:
        """Extract frames from video at specified sample rate"""
        frames = []
        timestamps = []
        
        try:
            container = av.open(video_path)
            video_stream = container.streams.video[0]
            
            fps = float(video_stream.average_rate)
            frame_interval = int(fps / sample_rate) if sample_rate > 0 else 30
            
            frame_count = 0
            extracted_count = 0
            
            for frame in container.decode(video_stream):
                if frame_count % frame_interval == 0 and extracted_count < self.max_frames:
                    # Convert frame to PIL Image
                    img = frame.to_image()
                    frames.append(img)
                    
                    # Calculate timestamp
                    timestamp = frame_count / fps
                    timestamps.append(timestamp)
                    
                    extracted_count += 1
                
                frame_count += 1
            
            container.close()
            
            self.logger.info(f"Extracted {len(frames)} frames from {video_path}")
            return frames, timestamps
            
        except Exception as e:
            self.logger.error(f"Frame extraction failed: {e}")
            # Fallback to OpenCV
            return self._extract_frames_opencv(video_path, sample_rate)
    
    def _extract_frames_opencv(self, video_path: str, sample_rate: float) -> Tuple[List[Image.Image], List[float]]:
        """Fallback frame extraction using OpenCV"""
        frames = []
        timestamps = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(fps / sample_rate) if sample_rate > 0 else 30
            
            frame_count = 0
            extracted_count = 0
            
            while cap.isOpened() and extracted_count < self.max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    # Convert BGR to RGB and then to PIL
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(frame_rgb)
                    frames.append(img)
                    
                    timestamp = frame_count / fps
                    timestamps.append(timestamp)
                    extracted_count += 1
                
                frame_count += 1
            
            cap.release()
            return frames, timestamps
            
        except Exception as e:
            self.logger.error(f"OpenCV frame extraction failed: {e}")
            return [], []
    
    def _comprehensive_analysis(self, frames: List[Image.Image], 
                              timestamps: List[float], 
                              video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Perform comprehensive video analysis"""
        
        results = {}
        
        # Scene detection
        scenes = self._detect_scenes(frames, timestamps)
        results['scenes'] = scenes
        
        # Content analysis for each scene
        scene_analysis = []
        for scene in scenes:
            start_idx = scene['start_frame_idx']
            end_idx = scene['end_frame_idx']
            scene_frames = frames[start_idx:end_idx + 1]
            
            # Analyze scene content
            scene_content = self._analyze_scene_content(scene_frames)
            scene_content.update({
                'start_time': scene['start_time'],
                'end_time': scene['end_time'],
                'duration': scene['duration']
            })
            scene_analysis.append(scene_content)
        
        results['scene_analysis'] = scene_analysis
        
        # Overall video summary
        video_summary = self._generate_video_summary(scene_analysis)
        results['video_summary'] = video_summary
        
        # Temporal patterns
        temporal_analysis = self._analyze_temporal_patterns(frames, timestamps)
        results['temporal_analysis'] = temporal_analysis
        
        # Key moments identification
        key_moments = self._identify_key_moments(frames, timestamps, scenes)
        results['key_moments'] = key_moments
        
        return results
    
    def _detect_scenes(self, frames: List[Image.Image], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Detect scene boundaries using visual similarity"""
        
        if len(frames) < 2:
            return [{'start_frame_idx': 0, 'end_frame_idx': len(frames) - 1, 
                    'start_time': timestamps[0], 'end_time': timestamps[-1],
                    'duration': timestamps[-1] - timestamps[0]}]
        
        try:
            # Get embeddings for all frames
            embeddings = []
            for frame in frames:
                embedding = self._get_frame_embedding(frame)
                embeddings.append(embedding)
            
            embeddings = np.array(embeddings)
            
            # Calculate frame-to-frame similarities
            similarities = []
            for i in range(len(embeddings) - 1):
                sim = cosine_similarity([embeddings[i]], [embeddings[i + 1]])[0, 0]
                similarities.append(sim)
            
            # Find scene boundaries (low similarity points)
            scene_boundaries = [0]  # Start with first frame
            
            for i, sim in enumerate(similarities):
                if sim < self.scene_threshold:
                    scene_boundaries.append(i + 1)
            
            scene_boundaries.append(len(frames) - 1)  # End with last frame
            
            # Create scene objects
            scenes = []
            for i in range(len(scene_boundaries) - 1):
                start_idx = scene_boundaries[i]
                end_idx = scene_boundaries[i + 1] - 1 if i < len(scene_boundaries) - 2 else scene_boundaries[i + 1]
                
                scene = {
                    'scene_id': i,
                    'start_frame_idx': start_idx,
                    'end_frame_idx': end_idx,
                    'start_time': timestamps[start_idx],
                    'end_time': timestamps[end_idx],
                    'duration': timestamps[end_idx] - timestamps[start_idx],
                    'frame_count': end_idx - start_idx + 1
                }
                scenes.append(scene)
            
            return scenes
            
        except Exception as e:
            self.logger.error(f"Scene detection failed: {e}")
            # Return single scene
            return [{'scene_id': 0, 'start_frame_idx': 0, 'end_frame_idx': len(frames) - 1,
                    'start_time': timestamps[0], 'end_time': timestamps[-1],
                    'duration': timestamps[-1] - timestamps[0], 'frame_count': len(frames)}]
    
    def _get_frame_embedding(self, frame: Image.Image) -> np.ndarray:
        """Get CLIP embedding for frame"""
        try:
            inputs = self.clip_processor(images=frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            self.logger.error(f"Frame embedding failed: {e}")
            return np.zeros(512)  # Default embedding size
    
    def _analyze_scene_content(self, scene_frames: List[Image.Image]) -> Dict[str, Any]:
        """Analyze content of a scene"""
        
        # Take representative frames (first, middle, last)
        representative_frames = []
        if len(scene_frames) == 1:
            representative_frames = scene_frames
        elif len(scene_frames) == 2:
            representative_frames = scene_frames
        else:
            indices = [0, len(scene_frames) // 2, len(scene_frames) - 1]
            representative_frames = [scene_frames[i] for i in indices]
        
        # Generate captions for representative frames
        captions = []
        for frame in representative_frames:
            caption = self._generate_frame_caption(frame)
            captions.append(caption)
        
        # Analyze scene characteristics
        scene_characteristics = self._analyze_scene_characteristics(representative_frames)
        
        # Generate scene description
        scene_description = self._generate_scene_description(captions, scene_characteristics)
        
        return {
            'description': scene_description,
            'captions': captions,
            'characteristics': scene_characteristics,
            'representative_frame_count': len(representative_frames)
        }
    
    def _generate_frame_caption(self, frame: Image.Image) -> str:
        """Generate caption for a single frame"""
        try:
            inputs = self.blip_processor(frame, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.blip_model.generate(**inputs, max_length=50, num_beams=5)
                caption = self.blip_processor.decode(outputs[0], skip_special_tokens=True)
            
            return caption
            
        except Exception as e:
            self.logger.error(f"Frame caption generation failed: {e}")
            return "Unable to generate caption"
    
    def _analyze_scene_characteristics(self, frames: List[Image.Image]) -> Dict[str, Any]:
        """Analyze visual characteristics of scene"""
        
        characteristics = {
            'lighting': 'unknown',
            'setting': 'unknown',
            'activity_level': 'unknown',
            'object_categories': [],
            'dominant_colors': []
        }
        
        try:
            # Analyze lighting and setting using CLIP
            lighting_categories = ['bright lighting', 'dim lighting', 'natural lighting', 'artificial lighting']
            setting_categories = ['indoor scene', 'outdoor scene', 'urban environment', 'natural environment']
            activity_categories = ['static scene', 'dynamic scene', 'people moving', 'vehicles moving']
            
            # Use first frame for analysis
            frame = frames[0]
            
            # Lighting analysis
            lighting_scores = self._classify_with_clip(frame, lighting_categories)
            characteristics['lighting'] = max(lighting_scores.items(), key=lambda x: x[1])[0]
            
            # Setting analysis
            setting_scores = self._classify_with_clip(frame, setting_categories)
            characteristics['setting'] = max(setting_scores.items(), key=lambda x: x[1])[0]
            
            # Activity analysis
            activity_scores = self._classify_with_clip(frame, activity_categories)
            characteristics['activity_level'] = max(activity_scores.items(), key=lambda x: x[1])[0]
            
            # Object detection
            object_categories = ['person', 'vehicle', 'building', 'nature', 'furniture', 'animal']
            object_scores = self._classify_with_clip(frame, object_categories)
            characteristics['object_categories'] = [cat for cat, score in object_scores.items() if score > 0.3]
            
            # Color analysis
            characteristics['dominant_colors'] = self._analyze_dominant_colors(frame)
            
        except Exception as e:
            self.logger.error(f"Scene characteristics analysis failed: {e}")
        
        return characteristics
    
    def _classify_with_clip(self, image: Image.Image, categories: List[str]) -> Dict[str, float]:
        """Classify image against categories using CLIP"""
        try:
            text_queries = [f"a photo of {category}" for category in categories]
            
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
            
            scores = {}
            for i, category in enumerate(categories):
                scores[category] = float(probs[0, i])
            
            return scores
            
        except Exception as e:
            self.logger.error(f"CLIP classification failed: {e}")
            return {cat: 0.0 for cat in categories}
    
    def _analyze_dominant_colors(self, image: Image.Image) -> List[str]:
        """Analyze dominant colors in image"""
        try:
            # Convert to numpy array
            img_array = np.array(image.resize((100, 100)))  # Resize for speed
            pixels = img_array.reshape(-1, 3)
            
            # Use KMeans to find dominant colors
            kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
            kmeans.fit(pixels)
            
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert to color names
            color_names = []
            for color in colors:
                name = self._rgb_to_color_name(color)
                if name not in color_names:
                    color_names.append(name)
            
            return color_names[:3]  # Return top 3 colors
            
        except Exception as e:
            self.logger.error(f"Color analysis failed: {e}")
            return ['unknown']
    
    def _rgb_to_color_name(self, rgb: np.ndarray) -> str:
        """Convert RGB to color name"""
        r, g, b = rgb
        
        if r > 200 and g > 200 and b > 200:
            return "white"
        elif r < 50 and g < 50 and b < 50:
            return "black"
        elif r > 150 and g < 100 and b < 100:
            return "red"
        elif r < 100 and g > 150 and b < 100:
            return "green"
        elif r < 100 and g < 100 and b > 150:
            return "blue"
        elif r > 150 and g > 150 and b < 100:
            return "yellow"
        elif r > 150 and g < 100 and b > 150:
            return "purple"
        elif r > 150 and g > 100 and b < 100:
            return "orange"
        else:
            return "mixed"
    
    def _generate_scene_description(self, captions: List[str], 
                                  characteristics: Dict[str, Any]) -> str:
        """Generate comprehensive scene description"""
        
        description_parts = []
        
        # Main content from captions
        if captions:
            main_caption = captions[0] if len(captions) == 1 else f"Scene showing {', '.join(captions[:2])}"
            description_parts.append(main_caption)
        
        # Add characteristics
        if characteristics['setting'] != 'unknown':
            description_parts.append(f"in {characteristics['setting']}")
        
        if characteristics['lighting'] != 'unknown':
            description_parts.append(f"with {characteristics['lighting']}")
        
        if characteristics['object_categories']:
            objects = ', '.join(characteristics['object_categories'][:3])
            description_parts.append(f"featuring {objects}")
        
        return ". ".join(description_parts)
    
    def _generate_video_summary(self, scene_analysis: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate overall video summary"""
        
        summary = {
            'total_scenes': len(scene_analysis),
            'main_themes': [],
            'overall_description': '',
            'key_elements': [],
            'duration_breakdown': {}
        }
        
        try:
            # Collect all scene descriptions
            descriptions = [scene['description'] for scene in scene_analysis]
            
            # Extract common themes
            all_words = ' '.join(descriptions).lower().split()
            word_counts = {}
            for word in all_words:
                if len(word) > 3:  # Skip short words
                    word_counts[word] = word_counts.get(word, 0) + 1
            
            # Get most common themes
            common_themes = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            summary['main_themes'] = [theme[0] for theme in common_themes]
            
            # Generate overall description
            if len(scene_analysis) == 1:
                summary['overall_description'] = f"Single scene video: {descriptions[0]}"
            else:
                summary['overall_description'] = f"Multi-scene video with {len(scene_analysis)} distinct segments"
            
            # Analyze duration breakdown
            total_duration = sum(scene.get('duration', 0) for scene in scene_analysis)
            if total_duration > 0:
                for i, scene in enumerate(scene_analysis):
                    percentage = (scene.get('duration', 0) / total_duration) * 100
                    summary['duration_breakdown'][f'scene_{i}'] = f"{percentage:.1f}%"
            
            # Extract key elements
            all_characteristics = []
            for scene in scene_analysis:
                chars = scene.get('characteristics', {})
                all_characteristics.extend(chars.get('object_categories', []))
            
            # Count occurrences
            element_counts = {}
            for element in all_characteristics:
                element_counts[element] = element_counts.get(element, 0) + 1
            
            summary['key_elements'] = list(element_counts.keys())[:5]
            
        except Exception as e:
            self.logger.error(f"Video summary generation failed: {e}")
            summary['overall_description'] = "Unable to generate summary"
        
        return summary
    
    def _analyze_temporal_patterns(self, frames: List[Image.Image], 
                                 timestamps: List[float]) -> Dict[str, Any]:
        """Analyze temporal patterns in video"""
        
        patterns = {
            'motion_analysis': {},
            'content_stability': 0.0,
            'scene_transitions': 0,
            'visual_complexity_trend': []
        }
        
        try:
            if len(frames) < 2:
                return patterns
            
            # Analyze motion by comparing consecutive frames
            motion_scores = []
            for i in range(len(frames) - 1):
                # Simple motion detection using frame difference
                frame1 = np.array(frames[i].resize((64, 64)))
                frame2 = np.array(frames[i + 1].resize((64, 64)))
                
                diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
                motion_scores.append(diff)
            
            patterns['motion_analysis'] = {
                'avg_motion': float(np.mean(motion_scores)),
                'max_motion': float(np.max(motion_scores)),
                'motion_variance': float(np.var(motion_scores))
            }
            
            # Content stability (inverse of motion)
            patterns['content_stability'] = max(0, 1 - (np.mean(motion_scores) / 255))
            
            # Count significant scene transitions
            motion_threshold = np.mean(motion_scores) + np.std(motion_scores)
            patterns['scene_transitions'] = int(np.sum(np.array(motion_scores) > motion_threshold))
            
            # Visual complexity trend (simplified)
            complexity_scores = []
            for frame in frames[::max(1, len(frames)//10)]:  # Sample frames
                # Simple complexity measure based on edge density
                gray = np.array(frame.convert('L'))
                edges = cv2.Canny(gray, 50, 150)
                complexity = np.sum(edges > 0) / edges.size
                complexity_scores.append(complexity)
            
            patterns['visual_complexity_trend'] = complexity_scores
            
        except Exception as e:
            self.logger.error(f"Temporal analysis failed: {e}")
        
        return patterns
    
    def _identify_key_moments(self, frames: List[Image.Image], 
                            timestamps: List[float], 
                            scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify key moments in video"""
        
        key_moments = []
        
        try:
            # Scene transitions as key moments
            for i, scene in enumerate(scenes):
                if i > 0:  # Skip first scene
                    key_moments.append({
                        'type': 'scene_transition',
                        'timestamp': scene['start_time'],
                        'description': f'Transition to scene {i + 1}',
                        'confidence': 0.8
                    })
            
            # High motion moments
            if len(frames) > 1:
                motion_scores = []
                for i in range(len(frames) - 1):
                    frame1 = np.array(frames[i].resize((64, 64)))
                    frame2 = np.array(frames[i + 1].resize((64, 64)))
                    diff = np.mean(np.abs(frame1.astype(float) - frame2.astype(float)))
                    motion_scores.append(diff)
                
                # Find peaks in motion
                motion_threshold = np.mean(motion_scores) + 1.5 * np.std(motion_scores)
                for i, score in enumerate(motion_scores):
                    if score > motion_threshold and i < len(timestamps) - 1:
                        key_moments.append({
                            'type': 'high_motion',
                            'timestamp': timestamps[i],
                            'description': 'High motion detected',
                            'confidence': min(0.9, score / (motion_threshold * 2))
                        })
            
            # Sort by timestamp
            key_moments.sort(key=lambda x: x['timestamp'])
            
            # Limit to top moments
            return key_moments[:10]
            
        except Exception as e:
            self.logger.error(f"Key moment identification failed: {e}")
            return []
    
    def _scene_detection_analysis(self, frames: List[Image.Image], 
                                timestamps: List[float]) -> Dict[str, Any]:
        """Focused scene detection analysis"""
        scenes = self._detect_scenes(frames, timestamps)
        return {'scenes': scenes, 'scene_count': len(scenes)}
    
    def _content_summary_analysis(self, frames: List[Image.Image], 
                                timestamps: List[float]) -> Dict[str, Any]:
        """Content summary focused analysis"""
        # Sample key frames for summary
        sample_indices = np.linspace(0, len(frames) - 1, min(5, len(frames))).astype(int)
        sample_frames = [frames[i] for i in sample_indices]
        
        captions = []
        for frame in sample_frames:
            caption = self._generate_frame_caption(frame)
            captions.append(caption)
        
        return {
            'sample_captions': captions,
            'content_summary': '. '.join(captions[:3])
        }
    
    def _temporal_analysis(self, frames: List[Image.Image], 
                         timestamps: List[float]) -> Dict[str, Any]:
        """Temporal analysis focused processing"""
        return self._analyze_temporal_patterns(frames, timestamps)
    
    def _update_analysis_stats(self, results: Dict[str, Any]):
        """Update analysis statistics"""
        self.analysis_stats['videos_processed'] += 1
        self.analysis_stats['total_frames_analyzed'] += results.get('frames_analyzed', 0)
        
        # Update average processing time
        current_avg = self.analysis_stats['avg_processing_time']
        count = self.analysis_stats['videos_processed']
        new_time = results.get('processing_time', 0)
        
        self.analysis_stats['avg_processing_time'] = (current_avg * (count - 1) + new_time) / count
    
    def _create_error_result(self, video_path: str, error_msg: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'video_path': video_path,
            'success': False,
            'error': error_msg,
            'analysis_type': 'error',
            'processing_time': 0,
            'frames_analyzed': 0
        }
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics"""
        return self.analysis_stats.copy()
    
    def clear_cache(self):
        """Clear analysis cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def shutdown(self):
        """Shutdown video analysis system"""
        self.logger.info("Shutting down video analysis system")
        self.clear_cache()
        self.logger.info("Video analysis system shutdown complete")