"""
Real-time Camera Processing System for JARVIS Voice Assistant
Live camera feed analysis with multimodal AI capabilities
"""

import logging
import cv2
import numpy as np
import threading
import queue
import time
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime, timedelta
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum

from PIL import Image
import torch

# Import multimodal components
from ..ai.multimodal_fusion_system import (
    MultimodalFusionSystem, 
    ModalityInput, 
    ModalityType
)
from ..ai.vision_models import VisionModelManager
from ..ai.ocr_system import AdvancedOCRSystem


class CameraState(Enum):
    """Camera system states"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    ERROR = "error"


@dataclass
class CameraFrame:
    """Container for camera frame data"""
    frame: np.ndarray
    timestamp: datetime
    frame_id: int
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class AnalysisResult:
    """Result from frame analysis"""
    frame_id: int
    timestamp: datetime
    analysis_type: str
    results: Dict[str, Any]
    confidence: float
    processing_time: float


class RealtimeCameraSystem:
    """Real-time camera processing with multimodal AI"""
    
    def __init__(self, config: Dict[str, Any]):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
        # Camera configuration
        self.camera_id = config.get('camera_id', 0)
        self.fps = config.get('fps', 30)
        self.resolution = config.get('resolution', (640, 480))
        self.analysis_fps = config.get('analysis_fps', 2)  # Analysis frames per second
        
        # Camera and threading
        self.cap = None
        self.camera_thread = None
        self.analysis_thread = None
        self.state = CameraState.STOPPED
        self._stop_event = threading.Event()
        
        # Frame processing
        self.frame_queue = queue.Queue(maxsize=10)
        self.analysis_queue = queue.Queue(maxsize=5)
        self.result_queue = queue.Queue(maxsize=20)
        
        # Analysis components
        self.fusion_system = None
        self.vision_manager = None
        self.ocr_system = None
        
        # Frame tracking
        self.frame_counter = 0
        self.last_analysis_time = 0
        self.analysis_interval = 1.0 / self.analysis_fps
        
        # Results and callbacks
        self.result_callbacks = []
        self.analysis_history = []
        self.max_history = config.get('max_history', 100)
        
        # Performance tracking
        self.performance_stats = {
            'frames_captured': 0,
            'frames_analyzed': 0,
            'avg_capture_fps': 0,
            'avg_analysis_time': 0,
            'last_reset': datetime.now()
        }
        
        # Initialize components
        self._initialize_multimodal_components()
    
    def _initialize_multimodal_components(self):
        """Initialize multimodal AI components"""
        try:
            # Initialize vision manager for lightweight operations
            vision_config = self.config.get('vision', {})
            self.vision_manager = VisionModelManager(vision_config)
            
            # Load CLIP for real-time analysis (lighter than BLIP)
            if not self.vision_manager.load_clip_model():
                self.logger.warning("Failed to load CLIP model for real-time analysis")
            
            # Initialize OCR system
            ocr_config = self.config.get('ocr', {})
            self.ocr_system = AdvancedOCRSystem(ocr_config)
            
            # Initialize fusion system for complex queries
            fusion_config = self.config.get('fusion', {})
            self.fusion_system = MultimodalFusionSystem(fusion_config)
            
            self.logger.info("Real-time camera multimodal components initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize multimodal components: {e}")
    
    def start_camera(self) -> bool:
        """Start camera capture"""
        try:
            if self.state == CameraState.RUNNING:
                self.logger.warning("Camera already running")
                return True
            
            self.state = CameraState.STARTING
            self.logger.info(f"Starting camera {self.camera_id}")
            
            # Initialize camera
            self.cap = cv2.VideoCapture(self.camera_id)
            
            if not self.cap.isOpened():
                raise RuntimeError(f"Failed to open camera {self.camera_id}")
            
            # Configure camera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)
            
            # Clear stop event
            self._stop_event.clear()
            
            # Start threads
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
            
            self.camera_thread.start()
            self.analysis_thread.start()
            
            self.state = CameraState.RUNNING
            self.logger.info("Camera started successfully")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            self.state = CameraState.ERROR
            return False
    
    def stop_camera(self):
        """Stop camera capture"""
        self.logger.info("Stopping camera")
        
        # Signal threads to stop
        self._stop_event.set()
        self.state = CameraState.STOPPED
        
        # Wait for threads to finish
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=2)
        
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join(timeout=2)
        
        # Release camera
        if self.cap:
            self.cap.release()
            self.cap = None
        
        # Clear queues
        self._clear_queues()
        
        self.logger.info("Camera stopped")
    
    def pause_camera(self):
        """Pause camera processing"""
        if self.state == CameraState.RUNNING:
            self.state = CameraState.PAUSED
            self.logger.info("Camera paused")
    
    def resume_camera(self):
        """Resume camera processing"""
        if self.state == CameraState.PAUSED:
            self.state = CameraState.RUNNING
            self.logger.info("Camera resumed")
    
    def _camera_loop(self):
        """Main camera capture loop"""
        last_fps_time = time.time()
        fps_counter = 0
        
        while not self._stop_event.is_set():
            try:
                if self.state != CameraState.RUNNING:
                    time.sleep(0.1)
                    continue
                
                # Capture frame
                ret, frame = self.cap.read()
                
                if not ret:
                    self.logger.warning("Failed to capture frame")
                    continue
                
                # Create frame object
                camera_frame = CameraFrame(
                    frame=frame,
                    timestamp=datetime.now(),
                    frame_id=self.frame_counter,
                    metadata={'resolution': frame.shape[:2]}
                )
                
                # Add to queue (non-blocking)
                try:
                    self.frame_queue.put_nowait(camera_frame)
                except queue.Full:
                    # Remove oldest frame if queue is full
                    try:
                        self.frame_queue.get_nowait()
                        self.frame_queue.put_nowait(camera_frame)
                    except queue.Empty:
                        pass
                
                self.frame_counter += 1
                self.performance_stats['frames_captured'] += 1
                fps_counter += 1
                
                # Calculate FPS
                current_time = time.time()
                if current_time - last_fps_time >= 1.0:
                    self.performance_stats['avg_capture_fps'] = fps_counter
                    fps_counter = 0
                    last_fps_time = current_time
                
                # Control frame rate
                time.sleep(1.0 / self.fps)
                
            except Exception as e:
                self.logger.error(f"Error in camera loop: {e}")
                time.sleep(0.1)
    
    def _analysis_loop(self):
        """Analysis processing loop"""
        while not self._stop_event.is_set():
            try:
                if self.state != CameraState.RUNNING:
                    time.sleep(0.1)
                    continue
                
                # Check if it's time for analysis
                current_time = time.time()
                if current_time - self.last_analysis_time < self.analysis_interval:
                    time.sleep(0.05)
                    continue
                
                # Get latest frame
                try:
                    frame = self.frame_queue.get_nowait()
                    self.last_analysis_time = current_time
                except queue.Empty:
                    time.sleep(0.05)
                    continue
                
                # Perform analysis
                self._analyze_frame(frame)
                
            except Exception as e:
                self.logger.error(f"Error in analysis loop: {e}")
                time.sleep(0.1)
    
    def _analyze_frame(self, camera_frame: CameraFrame):
        """Analyze a single frame"""
        start_time = time.time()
        
        try:
            # Convert frame to PIL Image
            frame_rgb = cv2.cvtColor(camera_frame.frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Perform basic analysis
            analysis_results = {}
            
            # Object detection using CLIP
            if self.vision_manager and 'clip' in self.vision_manager.models:
                object_categories = ['person', 'car', 'animal', 'food', 'device', 'furniture']
                objects = self.vision_manager.classify_image_with_categories(
                    pil_image, object_categories, top_k=3
                )
                analysis_results['objects'] = objects
            
            # OCR detection (lightweight check)
            if self.ocr_system:
                try:
                    # Quick OCR check - only if high confidence of text
                    ocr_result = self.ocr_system.extract_text(
                        frame_rgb, 
                        preprocessing=False,
                        engine="tesseract"  # Faster for real-time
                    )
                    if ocr_result.get('avg_confidence', 0) > 0.7:
                        analysis_results['text'] = ocr_result
                except Exception as e:
                    self.logger.debug(f"OCR analysis failed: {e}")
            
            # Scene analysis
            scene_categories = ['indoor', 'outdoor', 'office', 'home', 'street']
            if self.vision_manager and 'clip' in self.vision_manager.models:
                scene_analysis = self.vision_manager.classify_image_with_categories(
                    pil_image, scene_categories, top_k=2
                )
                analysis_results['scene'] = scene_analysis
            
            # Create analysis result
            processing_time = time.time() - start_time
            result = AnalysisResult(
                frame_id=camera_frame.frame_id,
                timestamp=camera_frame.timestamp,
                analysis_type='realtime_basic',
                results=analysis_results,
                confidence=self._calculate_overall_confidence(analysis_results),
                processing_time=processing_time
            )
            
            # Store result
            self._store_analysis_result(result)
            
            # Update performance stats
            self.performance_stats['frames_analyzed'] += 1
            current_avg = self.performance_stats['avg_analysis_time']
            total_analyzed = self.performance_stats['frames_analyzed']
            self.performance_stats['avg_analysis_time'] = (
                (current_avg * (total_analyzed - 1) + processing_time) / total_analyzed
            )
            
            # Notify callbacks
            self._notify_result_callbacks(result)
            
        except Exception as e:
            self.logger.error(f"Frame analysis failed: {e}")
    
    def _calculate_overall_confidence(self, analysis_results: Dict[str, Any]) -> float:
        """Calculate overall confidence from analysis results"""
        confidences = []
        
        if 'objects' in analysis_results:
            obj_confidences = [obj['confidence'] for obj in analysis_results['objects']]
            if obj_confidences:
                confidences.append(max(obj_confidences))
        
        if 'text' in analysis_results:
            text_conf = analysis_results['text'].get('avg_confidence', 0)
            confidences.append(text_conf)
        
        if 'scene' in analysis_results:
            scene_confidences = [scene['confidence'] for scene in analysis_results['scene']]
            if scene_confidences:
                confidences.append(max(scene_confidences))
        
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def _store_analysis_result(self, result: AnalysisResult):
        """Store analysis result"""
        self.analysis_history.append(result)
        
        # Limit history size
        if len(self.analysis_history) > self.max_history:
            self.analysis_history = self.analysis_history[-self.max_history:]
        
        # Add to result queue
        try:
            self.result_queue.put_nowait(result)
        except queue.Full:
            # Remove oldest result
            try:
                self.result_queue.get_nowait()
                self.result_queue.put_nowait(result)
            except queue.Empty:
                pass
    
    def _notify_result_callbacks(self, result: AnalysisResult):
        """Notify registered callbacks of new results"""
        for callback in self.result_callbacks:
            try:
                callback(result)
            except Exception as e:
                self.logger.error(f"Callback notification failed: {e}")
    
    def _clear_queues(self):
        """Clear all processing queues"""
        queues = [self.frame_queue, self.analysis_queue, self.result_queue]
        
        for q in queues:
            while not q.empty():
                try:
                    q.get_nowait()
                except queue.Empty:
                    break
    
    def add_result_callback(self, callback: Callable[[AnalysisResult], None]):
        """Add callback for analysis results"""
        self.result_callbacks.append(callback)
    
    def remove_result_callback(self, callback: Callable[[AnalysisResult], None]):
        """Remove result callback"""
        if callback in self.result_callbacks:
            self.result_callbacks.remove(callback)
    
    def get_latest_frame(self) -> Optional[CameraFrame]:
        """Get the latest captured frame"""
        try:
            # Get the most recent frame without blocking
            latest_frame = None
            while not self.frame_queue.empty():
                try:
                    latest_frame = self.frame_queue.get_nowait()
                except queue.Empty:
                    break
            return latest_frame
        except Exception as e:
            self.logger.error(f"Error getting latest frame: {e}")
            return None
    
    def get_latest_results(self, count: int = 5) -> List[AnalysisResult]:
        """Get latest analysis results"""
        return self.analysis_history[-count:] if self.analysis_history else []
    
    def query_current_view(self, query: str) -> Dict[str, Any]:
        """Query the current camera view with natural language"""
        try:
            # Get current frame
            latest_frame = self.get_latest_frame()
            
            if not latest_frame:
                return {
                    'success': False,
                    'error': 'No current frame available'
                }
            
            # Convert frame to temporary file for analysis
            frame_rgb = cv2.cvtColor(latest_frame.frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Create temporary file
            temp_path = Path(f"/tmp/camera_frame_{latest_frame.frame_id}.jpg")
            pil_image.save(temp_path)
            
            try:
                # Use fusion system for complex analysis
                modality_inputs = [
                    ModalityInput(
                        modality=ModalityType.TEXT,
                        data=query,
                        confidence=1.0
                    ),
                    ModalityInput(
                        modality=ModalityType.VISION,
                        data=str(temp_path),
                        confidence=0.9
                    )
                ]
                
                fusion_result = self.fusion_system.fuse_multimodal_input(modality_inputs)
                
                return {
                    'success': True,
                    'response': fusion_result.fused_response,
                    'confidence': fusion_result.confidence,
                    'frame_id': latest_frame.frame_id,
                    'timestamp': latest_frame.timestamp.isoformat(),
                    'processing_time': fusion_result.processing_time
                }
                
            finally:
                # Clean up temporary file
                if temp_path.exists():
                    temp_path.unlink()
            
        except Exception as e:
            self.logger.error(f"Current view query failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def take_snapshot(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """Take a snapshot of current view"""
        try:
            latest_frame = self.get_latest_frame()
            
            if not latest_frame:
                return {
                    'success': False,
                    'error': 'No current frame available'
                }
            
            # Generate save path if not provided
            if not save_path:
                timestamp = latest_frame.timestamp.strftime("%Y%m%d_%H%M%S")
                save_path = f"camera_snapshot_{timestamp}_{latest_frame.frame_id}.jpg"
            
            # Save frame
            cv2.imwrite(save_path, latest_frame.frame)
            
            # Perform analysis on snapshot
            frame_rgb = cv2.cvtColor(latest_frame.frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            
            # Basic analysis
            analysis = {}
            
            if self.vision_manager and 'blip_caption' in self.vision_manager.models:
                caption = self.vision_manager.generate_image_caption(pil_image)
                analysis['caption'] = caption
            
            if self.ocr_system:
                ocr_result = self.ocr_system.extract_text(frame_rgb)
                if ocr_result.get('full_text'):
                    analysis['text_content'] = ocr_result['full_text']
            
            return {
                'success': True,
                'save_path': save_path,
                'frame_id': latest_frame.frame_id,
                'timestamp': latest_frame.timestamp.isoformat(),
                'analysis': analysis
            }
            
        except Exception as e:
            self.logger.error(f"Snapshot failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def start_motion_detection(self, sensitivity: float = 0.3) -> bool:
        """Start motion detection"""
        try:
            self.motion_detection = True
            self.motion_sensitivity = sensitivity
            self.previous_frame = None
            
            self.logger.info(f"Motion detection started with sensitivity {sensitivity}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start motion detection: {e}")
            return False
    
    def stop_motion_detection(self):
        """Stop motion detection"""
        self.motion_detection = False
        self.previous_frame = None
        self.logger.info("Motion detection stopped")
    
    def detect_motion(self, current_frame: np.ndarray) -> Dict[str, Any]:
        """Detect motion between frames"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (21, 21), 0)
            
            # Initialize previous frame
            if self.previous_frame is None:
                self.previous_frame = gray
                return {'motion_detected': False, 'motion_area': 0}
            
            # Calculate difference
            frame_delta = cv2.absdiff(self.previous_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_area = 0
            for contour in contours:
                if cv2.contourArea(contour) > 500:  # Minimum area threshold
                    motion_area += cv2.contourArea(contour)
            
            # Calculate motion percentage
            total_area = current_frame.shape[0] * current_frame.shape[1]
            motion_percentage = motion_area / total_area
            
            motion_detected = motion_percentage > self.motion_sensitivity
            
            # Update previous frame
            self.previous_frame = gray
            
            return {
                'motion_detected': motion_detected,
                'motion_area': motion_area,
                'motion_percentage': motion_percentage,
                'contour_count': len(contours)
            }
            
        except Exception as e:
            self.logger.error(f"Motion detection failed: {e}")
            return {'motion_detected': False, 'motion_area': 0}
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        stats = self.performance_stats.copy()
        stats['state'] = self.state.value
        stats['frame_queue_size'] = self.frame_queue.qsize()
        stats['result_queue_size'] = self.result_queue.qsize()
        stats['analysis_history_size'] = len(self.analysis_history)
        
        # Calculate uptime
        uptime = datetime.now() - stats['last_reset']
        stats['uptime_seconds'] = uptime.total_seconds()
        
        return stats
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'frames_captured': 0,
            'frames_analyzed': 0,
            'avg_capture_fps': 0,
            'avg_analysis_time': 0,
            'last_reset': datetime.now()
        }
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        info = {
            'camera_id': self.camera_id,
            'resolution': self.resolution,
            'fps': self.fps,
            'analysis_fps': self.analysis_fps,
            'state': self.state.value,
            'multimodal_components': {
                'fusion_system': self.fusion_system is not None,
                'vision_manager': self.vision_manager is not None,
                'ocr_system': self.ocr_system is not None
            }
        }
        
        if self.cap:
            info['camera_properties'] = {
                'actual_width': self.cap.get(cv2.CAP_PROP_FRAME_WIDTH),
                'actual_height': self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT),
                'actual_fps': self.cap.get(cv2.CAP_PROP_FPS)
            }
        
        return info
    
    def is_running(self) -> bool:
        """Check if camera is running"""
        return self.state == CameraState.RUNNING
    
    def shutdown(self):
        """Shutdown camera system"""
        self.logger.info("Shutting down real-time camera system")
        
        # Stop camera
        self.stop_camera()
        
        # Clear callbacks
        self.result_callbacks.clear()
        
        # Clear history
        self.analysis_history.clear()
        
        # Shutdown multimodal components
        if self.fusion_system:
            self.fusion_system.shutdown()
        
        if self.vision_manager:
            self.vision_manager.clear_all_models()
        
        self.logger.info("Real-time camera system shutdown complete")


class CameraWebInterface:
    """Web interface for camera system integration"""
    
    def __init__(self, camera_system: RealtimeCameraSystem):
        self.camera_system = camera_system
        self.logger = logging.getLogger(__name__)
    
    def get_camera_routes(self, app):
        """Add camera routes to Flask app"""
        
        @app.route('/api/camera/start', methods=['POST'])
        def start_camera():
            success = self.camera_system.start_camera()
            return {'success': success, 'state': self.camera_system.state.value}
        
        @app.route('/api/camera/stop', methods=['POST'])
        def stop_camera():
            self.camera_system.stop_camera()
            return {'success': True, 'state': self.camera_system.state.value}
        
        @app.route('/api/camera/status', methods=['GET'])
        def camera_status():
            return {
                'state': self.camera_system.state.value,
                'is_running': self.camera_system.is_running(),
                'performance': self.camera_system.get_performance_stats(),
                'system_info': self.camera_system.get_system_info()
            }
        
        @app.route('/api/camera/query', methods=['POST'])
        def camera_query():
            data = request.get_json()
            query = data.get('query', '')
            result = self.camera_system.query_current_view(query)
            return result
        
        @app.route('/api/camera/snapshot', methods=['POST'])
        def take_snapshot():
            data = request.get_json()
            save_path = data.get('save_path')
            result = self.camera_system.take_snapshot(save_path)
            return result
        
        @app.route('/api/camera/results', methods=['GET'])
        def get_results():
            count = request.args.get('count', 5, type=int)
            results = self.camera_system.get_latest_results(count)
            
            # Convert to serializable format
            serialized_results = []
            for result in results:
                serialized_results.append({
                    'frame_id': result.frame_id,
                    'timestamp': result.timestamp.isoformat(),
                    'analysis_type': result.analysis_type,
                    'results': result.results,
                    'confidence': result.confidence,
                    'processing_time': result.processing_time
                })
            
            return {'results': serialized_results}