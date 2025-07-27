#!/usr/bin/env python3
"""
⚡ GPU Accelerator for JARVIS
ตัวเร่งความเร็ว GPU สำหรับ JARVIS
"""

import logging
import time
import threading
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("⚠️ PyTorch not available - GPU acceleration disabled")

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False


class AccelerationType(Enum):
    """ประเภทการเร่งความเร็ว"""
    CPU_ONLY = "cpu_only"
    CUDA = "cuda"
    OPENCL = "opencl"
    AUTO = "auto"


@dataclass
class GPUInfo:
    """ข้อมูล GPU"""
    name: str
    memory_total: int
    memory_free: int
    compute_capability: Tuple[int, int]
    is_available: bool
    device_id: int


class GPUAccelerator:
    """ตัวเร่งความเร็ว GPU สำหรับ JARVIS"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.logger = logging.getLogger(__name__)
        self.config = config or {}
        
        # GPU configuration
        self.acceleration_type = AccelerationType(
            self.config.get('acceleration_type', 'auto')
        )
        self.preferred_device = self.config.get('preferred_device', 'auto')
        self.memory_limit = self.config.get('memory_limit', 0.8)  # 80% ของ GPU memory
        
        # State
        self.device = None
        self.gpu_info = None
        self.is_initialized = False
        self.performance_stats = {}
        
        # Optimization settings
        self.enable_mixed_precision = self.config.get('mixed_precision', True)
        self.enable_memory_optimization = self.config.get('memory_optimization', True)
        self.batch_size_optimization = self.config.get('batch_optimization', True)
        
        self._initialize()
    
    def _initialize(self):
        """เริ่มต้นตัวเร่งความเร็ว GPU"""
        try:
            self.logger.info("🚀 Initializing GPU Accelerator...")
            
            # Detect and select best device
            self.device = self._select_best_device()
            
            if self.device and self.device.type != 'cpu':
                self.gpu_info = self._get_gpu_info()
                self._optimize_gpu_settings()
                self.logger.info(f"✅ GPU Accelerator initialized: {self.device}")
            else:
                self.logger.info("💻 Using CPU-only acceleration")
            
            self.is_initialized = True
            
        except Exception as e:
            self.logger.error(f"❌ GPU Accelerator initialization failed: {e}")
            self.device = torch.device('cpu') if TORCH_AVAILABLE else None
    
    def _select_best_device(self) -> Optional[torch.device]:
        """เลือกอุปกรณ์ที่ดีที่สุด"""
        if not TORCH_AVAILABLE:
            return None
        
        if self.acceleration_type == AccelerationType.CPU_ONLY:
            return torch.device('cpu')
        
        # Auto-detect best device
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            self.logger.info(f"🎮 Found {device_count} CUDA device(s)")
            
            # Select device with most memory
            best_device = 0
            max_memory = 0
            
            for i in range(device_count):
                props = torch.cuda.get_device_properties(i)
                memory = props.total_memory
                
                self.logger.info(f"   GPU {i}: {props.name} ({memory / 1024**3:.1f} GB)")
                
                if memory > max_memory:
                    max_memory = memory
                    best_device = i
            
            return torch.device(f'cuda:{best_device}')
        
        return torch.device('cpu')
    
    def _get_gpu_info(self) -> Optional[GPUInfo]:
        """ได้รับข้อมูล GPU"""
        if not self.device or self.device.type == 'cpu':
            return None
        
        try:
            device_id = self.device.index or 0
            props = torch.cuda.get_device_properties(device_id)
            
            memory_total = torch.cuda.get_device_properties(device_id).total_memory
            memory_free = memory_total - torch.cuda.memory_allocated(device_id)
            
            return GPUInfo(
                name=props.name,
                memory_total=memory_total,
                memory_free=memory_free,
                compute_capability=(props.major, props.minor),
                is_available=True,
                device_id=device_id
            )
        except Exception as e:
            self.logger.error(f"❌ Failed to get GPU info: {e}")
            return None
    
    def _optimize_gpu_settings(self):
        """ปรับแต่งการตั้งค่า GPU"""
        if not self.device or self.device.type == 'cpu':
            return
        
        try:
            # Enable optimizations
            if hasattr(torch.backends, 'cudnn'):
                torch.backends.cudnn.benchmark = True
                torch.backends.cudnn.deterministic = False
                self.logger.info("✅ CuDNN optimizations enabled")
            
            # Set memory fraction
            if self.memory_limit < 1.0:
                torch.cuda.set_per_process_memory_fraction(
                    self.memory_limit, self.device.index
                )
                self.logger.info(f"🧠 GPU memory limited to {self.memory_limit*100:.0f}%")
            
            # Clear cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU cache cleared")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to optimize GPU settings: {e}")
    
    def accelerate_audio_processing(self, audio_data: np.ndarray) -> np.ndarray:
        """เร่งความเร็วการประมวลผลเสียง"""
        if not self.is_initialized or not TORCH_AVAILABLE:
            return audio_data
        
        start_time = time.time()
        
        try:
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio_data).float()
            
            if self.device.type != 'cpu':
                audio_tensor = audio_tensor.to(self.device)
            
            # Apply optimizations
            with torch.no_grad():
                # Normalize audio
                audio_tensor = F.normalize(audio_tensor, dim=-1)
                
                # Apply windowing for better FFT performance
                if len(audio_tensor.shape) == 1:
                    window = torch.hann_window(
                        audio_tensor.size(0), 
                        device=audio_tensor.device
                    )
                    audio_tensor = audio_tensor * window
                
                # Batch processing if needed
                if self.batch_size_optimization and len(audio_tensor) > 1024:
                    audio_tensor = self._batch_process_audio(audio_tensor)
            
            # Convert back to numpy
            if self.device.type != 'cpu':
                result = audio_tensor.cpu().numpy()
            else:
                result = audio_tensor.numpy()
            
            processing_time = time.time() - start_time
            self._update_performance_stats('audio_processing', processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Audio acceleration failed: {e}")
            return audio_data
    
    def _batch_process_audio(self, audio_tensor: torch.Tensor) -> torch.Tensor:
        """ประมวลผลเสียงแบบ batch"""
        batch_size = 1024
        results = []
        
        for i in range(0, len(audio_tensor), batch_size):
            batch = audio_tensor[i:i + batch_size]
            
            # Apply processing to batch
            processed_batch = self._process_audio_batch(batch)
            results.append(processed_batch)
        
        return torch.cat(results, dim=0)
    
    def _process_audio_batch(self, batch: torch.Tensor) -> torch.Tensor:
        """ประมวลผล batch เสียง"""
        # Apply noise reduction
        batch = self._gpu_noise_reduction(batch)
        
        # Apply enhancement
        batch = self._gpu_audio_enhancement(batch)
        
        return batch
    
    def _gpu_noise_reduction(self, audio: torch.Tensor) -> torch.Tensor:
        """ลดเสียงรบกวนด้วย GPU"""
        # Simple spectral subtraction
        if len(audio) < 256:
            return audio
        
        # Compute short-time FFT
        window_size = min(256, len(audio) // 4)
        hop_length = window_size // 2
        
        # Apply noise gate
        threshold = torch.std(audio) * 0.1
        mask = torch.abs(audio) > threshold
        
        return audio * mask.float()
    
    def _gpu_audio_enhancement(self, audio: torch.Tensor) -> torch.Tensor:
        """ปรับปรุงเสียงด้วย GPU"""
        # Apply dynamic range compression
        audio = torch.tanh(audio * 2.0) * 0.8
        
        # Apply gentle high-pass filter
        if len(audio) > 3:
            # Simple difference filter
            enhanced = audio.clone()
            enhanced[1:] = audio[1:] - 0.1 * audio[:-1]
            return enhanced
        
        return audio
    
    def accelerate_ai_inference(self, input_data: np.ndarray) -> np.ndarray:
        """เร่งความเร็วการอนุมาน AI"""
        if not self.is_initialized or not TORCH_AVAILABLE:
            return input_data
        
        start_time = time.time()
        
        try:
            # Convert to tensor
            input_tensor = torch.from_numpy(input_data).float()
            
            if self.device.type != 'cpu':
                input_tensor = input_tensor.to(self.device)
            
            # Apply mixed precision if available
            if self.enable_mixed_precision and hasattr(torch, 'autocast'):
                with torch.autocast(device_type=self.device.type):
                    result_tensor = self._process_ai_inference(input_tensor)
            else:
                result_tensor = self._process_ai_inference(input_tensor)
            
            # Convert back
            if self.device.type != 'cpu':
                result = result_tensor.cpu().numpy()
            else:
                result = result_tensor.numpy()
            
            processing_time = time.time() - start_time
            self._update_performance_stats('ai_inference', processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ AI inference acceleration failed: {e}")
            return input_data
    
    def _process_ai_inference(self, input_tensor: torch.Tensor) -> torch.Tensor:
        """ประมวลผลการอนุมาน AI"""
        with torch.no_grad():
            # Simple processing pipeline
            processed = input_tensor
            
            # Normalize
            processed = F.normalize(processed, dim=-1)
            
            # Apply transformations
            if len(processed.shape) > 1:
                processed = F.softmax(processed, dim=-1)
            
            return processed
    
    def accelerate_visualization(self, data: np.ndarray) -> np.ndarray:
        """เร่งความเร็วการแสดงผล"""
        if not self.is_initialized:
            return data
        
        start_time = time.time()
        
        try:
            # Use CuPy if available for faster array operations
            if CUPY_AVAILABLE and self.device and self.device.type != 'cpu':
                # Convert to CuPy array
                gpu_data = cp.asarray(data)
                
                # Apply optimizations
                gpu_data = self._optimize_visualization_data(gpu_data)
                
                # Convert back
                result = cp.asnumpy(gpu_data)
            else:
                # CPU fallback with NumPy optimizations
                result = self._cpu_optimize_visualization(data)
            
            processing_time = time.time() - start_time
            self._update_performance_stats('visualization', processing_time)
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Visualization acceleration failed: {e}")
            return data
    
    def _optimize_visualization_data(self, data):
        """ปรับแต่งข้อมูลการแสดงผล"""
        if CUPY_AVAILABLE:
            # Apply smoothing
            if len(data) > 3:
                data = cp.convolve(data, cp.array([0.25, 0.5, 0.25]), mode='same')
            
            # Normalize for better visualization
            data_min = cp.min(data)
            data_max = cp.max(data)
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        
        return data
    
    def _cpu_optimize_visualization(self, data: np.ndarray) -> np.ndarray:
        """ปรับแต่งการแสดงผลด้วย CPU"""
        # Apply smoothing
        if len(data) > 3:
            kernel = np.array([0.25, 0.5, 0.25])
            data = np.convolve(data, kernel, mode='same')
        
        # Normalize
        data_min = np.min(data)
        data_max = np.max(data)
        if data_max > data_min:
            data = (data - data_min) / (data_max - data_min)
        
        return data
    
    def _update_performance_stats(self, operation: str, processing_time: float):
        """อัปเดตสถิติประสิทธิภาพ"""
        if operation not in self.performance_stats:
            self.performance_stats[operation] = {
                'count': 0,
                'total_time': 0.0,
                'avg_time': 0.0,
                'min_time': float('inf'),
                'max_time': 0.0
            }
        
        stats = self.performance_stats[operation]
        stats['count'] += 1
        stats['total_time'] += processing_time
        stats['avg_time'] = stats['total_time'] / stats['count']
        stats['min_time'] = min(stats['min_time'], processing_time)
        stats['max_time'] = max(stats['max_time'], processing_time)
    
    def get_performance_report(self) -> Dict[str, Any]:
        """รายงานประสิทธิภาพ"""
        report = {
            'device': str(self.device) if self.device else 'None',
            'gpu_info': self.gpu_info.__dict__ if self.gpu_info else None,
            'is_initialized': self.is_initialized,
            'acceleration_enabled': self.device and self.device.type != 'cpu',
            'performance_stats': self.performance_stats.copy()
        }
        
        if self.device and self.device.type != 'cpu':
            try:
                # Add current GPU memory usage
                memory_allocated = torch.cuda.memory_allocated(self.device.index)
                memory_cached = torch.cuda.memory_reserved(self.device.index)
                
                report['memory_usage'] = {
                    'allocated_mb': memory_allocated / 1024**2,
                    'cached_mb': memory_cached / 1024**2,
                    'utilization': memory_allocated / self.gpu_info.memory_total if self.gpu_info else 0
                }
            except Exception:
                pass
        
        return report
    
    def optimize_batch_size(self, operation_type: str, data_size: int) -> int:
        """ปรับแต่งขนาด batch ให้เหมาะสม"""
        if not self.batch_size_optimization or not self.gpu_info:
            return min(64, data_size)  # Default batch size
        
        # Calculate optimal batch size based on GPU memory
        available_memory = self.gpu_info.memory_free * 0.8  # Use 80% of free memory
        
        # Estimate memory per sample (rough approximation)
        memory_per_sample = {
            'audio_processing': 1024,     # 1KB per audio sample
            'ai_inference': 4096,         # 4KB per inference
            'visualization': 512          # 512B per visualization point
        }
        
        sample_memory = memory_per_sample.get(operation_type, 1024)
        optimal_batch = int(available_memory / sample_memory)
        
        # Clamp to reasonable range
        optimal_batch = max(1, min(optimal_batch, data_size, 1024))
        
        return optimal_batch
    
    def cleanup(self):
        """ทำความสะอาดทรัพยากร"""
        try:
            if self.device and self.device.type != 'cpu':
                torch.cuda.empty_cache()
                self.logger.info("🧹 GPU resources cleaned up")
        except Exception as e:
            self.logger.error(f"❌ Cleanup failed: {e}")


def test_gpu_accelerator():
    """ทดสอบ GPU Accelerator"""
    print("🧪 Testing GPU Accelerator")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    
    # Create accelerator
    accelerator = GPUAccelerator()
    
    # Test audio processing
    audio_data = np.random.randn(1024).astype(np.float32)
    processed_audio = accelerator.accelerate_audio_processing(audio_data)
    print(f"✅ Audio processing: {audio_data.shape} -> {processed_audio.shape}")
    
    # Test AI inference
    ai_data = np.random.randn(64, 128).astype(np.float32)
    inference_result = accelerator.accelerate_ai_inference(ai_data)
    print(f"✅ AI inference: {ai_data.shape} -> {inference_result.shape}")
    
    # Test visualization
    viz_data = np.random.randn(256).astype(np.float32)
    viz_result = accelerator.accelerate_visualization(viz_data)
    print(f"✅ Visualization: {viz_data.shape} -> {viz_result.shape}")
    
    # Performance report
    report = accelerator.get_performance_report()
    print(f"\n📊 Performance Report:")
    print(f"   Device: {report['device']}")
    print(f"   GPU Acceleration: {'✅' if report['acceleration_enabled'] else '❌'}")
    
    if report['gpu_info']:
        gpu = report['gpu_info']
        print(f"   GPU: {gpu['name']}")
        print(f"   Memory: {gpu['memory_total'] / 1024**3:.1f} GB")
    
    for op, stats in report['performance_stats'].items():
        print(f"   {op}: {stats['avg_time']*1000:.2f}ms avg ({stats['count']} operations)")
    
    # Cleanup
    accelerator.cleanup()
    
    return accelerator


if __name__ == "__main__":
    test_gpu_accelerator()