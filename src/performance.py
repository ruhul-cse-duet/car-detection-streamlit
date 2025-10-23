"""
Performance optimization utilities for Car Object Detection System
"""

import gc
import psutil
import time
import logging
from typing import Optional, Tuple
import numpy as np
import cv2
from PIL import Image

class PerformanceMonitor:
    """Monitor and optimize system performance"""
    
    def __init__(self):
        self.start_time = None
        self.memory_usage = []
        self.processing_times = []
        
    def start_monitoring(self):
        """Start performance monitoring"""
        self.start_time = time.time()
        self.memory_usage = []
        self.processing_times = []
        
    def log_memory_usage(self):
        """Log current memory usage"""
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
        
        if memory_percent > 90:
            logging.warning(f"High memory usage: {memory_percent:.1f}%")
            self.cleanup_memory()
            
    def log_processing_time(self, operation: str, duration: float):
        """Log processing time for an operation"""
        self.processing_times.append((operation, duration))
        logging.info(f"{operation} took {duration:.3f}s")
        
    def cleanup_memory(self):
        """Force garbage collection to free memory"""
        gc.collect()
        logging.info("Memory cleanup performed")
        
    def get_performance_summary(self) -> dict:
        """Get performance summary"""
        if not self.processing_times:
            return {}
            
        total_time = time.time() - self.start_time if self.start_time else 0
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        max_memory = np.max(self.memory_usage) if self.memory_usage else 0
        
        return {
            "total_time": total_time,
            "avg_memory_usage": avg_memory,
            "max_memory_usage": max_memory,
            "operations": self.processing_times
        }

class VideoProcessor:
    """Optimized video processing utilities"""
    
    @staticmethod
    def resize_frame_if_needed(frame: np.ndarray, max_size: Tuple[int, int] = (1280, 720)) -> np.ndarray:
        """Resize frame if it's too large for processing"""
        h, w = frame.shape[:2]
        max_w, max_h = max_size
        
        if w > max_w or h > max_h:
            # Calculate new dimensions maintaining aspect ratio
            scale = min(max_w / w, max_h / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            logging.info(f"Resized frame from {w}x{h} to {new_w}x{new_h}")
            
        return frame
    
    @staticmethod
    def optimize_frame_quality(frame: np.ndarray, quality: int = 80) -> np.ndarray:
        """Optimize frame quality for faster processing"""
        # Convert to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Already in correct format
            pass
        elif len(frame.shape) == 3 and frame.shape[2] == 4:
            # Remove alpha channel
            frame = frame[:, :, :3]
        else:
            # Convert grayscale to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
        return frame
    
    @staticmethod
    def batch_process_frames(frames: list, batch_size: int = 4) -> list:
        """Process frames in batches for better memory management"""
        processed_frames = []
        
        for i in range(0, len(frames), batch_size):
            batch = frames[i:i + batch_size]
            
            # Process batch
            for frame in batch:
                # Apply optimizations
                frame = VideoProcessor.resize_frame_if_needed(frame)
                frame = VideoProcessor.optimize_frame_quality(frame)
                processed_frames.append(frame)
            
            # Cleanup after each batch
            gc.collect()
            
        return processed_frames

class ModelOptimizer:
    """Optimize model inference performance"""
    
    @staticmethod
    def warmup_model(model, num_warmup: int = 3):
        """Warm up the model with dummy inputs"""
        logging.info("Warming up model...")
        
        dummy_input = np.zeros((640, 640, 3), dtype=np.uint8)
        
        for i in range(num_warmup):
            try:
                _ = model.predict(dummy_input, verbose=False)
            except Exception as e:
                logging.warning(f"Warmup iteration {i} failed: {e}")
                
        logging.info("Model warmup completed")
    
    @staticmethod
    def optimize_inference_settings():
        """Get optimized inference settings"""
        return {
            "conf": 0.25,
            "iou": 0.6,
            "max_det": 300,
            "agnostic_nms": False,
            "verbose": False,
            "save": False,
            "save_conf": False,
            "save_crop": False,
            "save_txt": False,
            "save_json": False
        }

class MemoryManager:
    """Manage memory usage during processing"""
    
    @staticmethod
    def get_memory_info() -> dict:
        """Get current memory information"""
        memory = psutil.virtual_memory()
        return {
            "total": memory.total,
            "available": memory.available,
            "used": memory.used,
            "percent": memory.percent,
            "free": memory.free
        }
    
    @staticmethod
    def is_memory_available(required_mb: int = 100) -> bool:
        """Check if enough memory is available"""
        memory = psutil.virtual_memory()
        available_mb = memory.available / (1024 * 1024)
        return available_mb > required_mb
    
    @staticmethod
    def cleanup_if_needed():
        """Clean up memory if usage is high"""
        memory = psutil.virtual_memory()
        if memory.percent > 85:
            gc.collect()
            logging.info("Memory cleanup performed due to high usage")
            return True
        return False

def optimize_detection_pipeline():
    """Optimize the entire detection pipeline"""
    optimizations = {
        "enable_gpu": True,
        "batch_processing": True,
        "memory_management": True,
        "frame_optimization": True,
        "model_warmup": True
    }
    
    return optimizations

# Export all classes and functions
__all__ = [
    "PerformanceMonitor",
    "VideoProcessor", 
    "ModelOptimizer",
    "MemoryManager",
    "optimize_detection_pipeline"
]
