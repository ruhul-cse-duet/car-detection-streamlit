"""
Configuration file for Car Object Detection System
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / "models"
TEST_IMG_DIR = PROJECT_ROOT / "test_img"
VIDEOS_DIR = PROJECT_ROOT / "Videos"

# Model configuration
MODEL_CONFIG = {
    "model_path": MODELS_DIR / "yolov11_trained.pt",
    "model_name": "YOLOv11s",
    "input_size": 640,
    "confidence_threshold": 0.25,
    "device": "auto",  # "auto", "cpu", "cuda"
}

# Detection configuration
DETECTION_CONFIG = {
    "max_detections": 300,
    "iou_threshold": 0.6,
    "agnostic_nms": False,
    "save_conf": False,
    "save_crop": False,
    "save_txt": False,
    "save_json": False,
}

# Tracking configuration
TRACKING_CONFIG = {
    "iou_threshold": 0.35,
    "max_age": 15,
    "min_hits": 1,
    "tracker_type": "iou_based",
}

# Video processing configuration
VIDEO_CONFIG = {
    "max_frames_in_memory": 1000,
    "frame_skip": 1,
    "output_fps": 30,
    "video_codec": "mp4v",
    "jpeg_quality": 80,
}

# UI configuration
UI_CONFIG = {
    "page_title": "Car Detection System",
    "layout": "wide",
    "theme": "auto",  # "auto", "light", "dark"
    "show_sidebar": False,
    "max_upload_size": 200,  # MB
}

# Performance configuration
PERFORMANCE_CONFIG = {
    "enable_gpu": True,
    "batch_size": 1,
    "num_workers": 4,
    "pin_memory": True,
    "cache_models": True,
}

# Logging configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file": None,  # Set to a file path to enable file logging
}

# Validation
def validate_config():
    """Validate configuration settings"""
    errors = []
    
    # Check if model file exists
    if not MODEL_CONFIG["model_path"].exists():
        errors.append(f"Model file not found: {MODEL_CONFIG['model_path']}")
    
    # Check if test images directory exists
    if not TEST_IMG_DIR.exists():
        errors.append(f"Test images directory not found: {TEST_IMG_DIR}")
    
    # Validate confidence threshold
    if not 0.0 <= MODEL_CONFIG["confidence_threshold"] <= 1.0:
        errors.append("Confidence threshold must be between 0.0 and 1.0")
    
    # Validate IoU threshold
    if not 0.0 <= TRACKING_CONFIG["iou_threshold"] <= 1.0:
        errors.append("IoU threshold must be between 0.0 and 1.0")
    
    # Validate max age
    if TRACKING_CONFIG["max_age"] < 1:
        errors.append("Max age must be at least 1")
    
    return errors

# Environment variables
def get_env_config():
    """Get configuration from environment variables"""
    return {
        "device": os.getenv("DEVICE", MODEL_CONFIG["device"]),
        "confidence_threshold": float(os.getenv("CONFIDENCE_THRESHOLD", MODEL_CONFIG["confidence_threshold"])),
        "iou_threshold": float(os.getenv("IOU_THRESHOLD", TRACKING_CONFIG["iou_threshold"])),
        "max_frames": int(os.getenv("MAX_FRAMES", VIDEO_CONFIG["max_frames_in_memory"])),
        "log_level": os.getenv("LOG_LEVEL", LOGGING_CONFIG["level"]),
    }

# Export all configurations
__all__ = [
    "MODEL_CONFIG",
    "DETECTION_CONFIG", 
    "TRACKING_CONFIG",
    "VIDEO_CONFIG",
    "UI_CONFIG",
    "PERFORMANCE_CONFIG",
    "LOGGING_CONFIG",
    "validate_config",
    "get_env_config",
]
