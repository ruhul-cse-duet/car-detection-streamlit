# src/custom_resnet.py
import os
import subprocess
import sys
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)
import warnings

warnings.filterwarnings("ignore")

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_cached_models = {
    "yolo": None,
}

def _get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]

def _ensure_cv2_headless(auto_install: bool = True):
    try:
        import cv2  # noqa: F401
        return
    except Exception as e:
        if not auto_install:
            raise ImportError("cv2 import failed. Install `opencv-python-headless`. Original: " + str(e))
        try:
            print("Installing opencv-python-headless ...", file=sys.stderr)
            subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python-headless==4.12.0.88"])
            import cv2  # noqa: F401
            return
        except Exception as e2:
            raise ImportError("Auto-install of opencv-python-headless failed. Original: " + str(e2))



def _load_yolo_model(weights_relative="models/yolov11_trained.pt"):
    """
    Loads and caches a YOLO model (ultralytics).
    Expects weights at project_root/models/yolov11_trained.pt by default.
    """
    if _cached_models["yolo"] is not None:
        return _cached_models["yolo"]

    _ensure_cv2_headless(auto_install=True)
    try:
        from ultralytics import YOLO  # lazy import
    except Exception as e:
        raise RuntimeError("ultralytics import failed. Install `ultralytics`.") from e

    weights_path = _get_project_root() / weights_relative
    
    # Enhanced model validation
    if not weights_path.exists():
        # Try alternative model paths
        alternative_paths = [
            _get_project_root() / "models" / "yolov11_trained.pt",
            _get_project_root() / "yolov11_trained.pt",
            Path("models/yolov11_trained.pt"),
            Path("yolov11_trained.pt")
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                weights_path = alt_path
                break
        else:
            raise FileNotFoundError(
                f"YOLO weights not found. Searched paths:\n" +
                "\n".join([f"• {p}" for p in [weights_path] + alternative_paths]) +
                f"\n\nPlease ensure the model file exists at one of these locations."
            )
    
    # Validate model file
    try:
        file_size = weights_path.stat().st_size
        if file_size < 1024:  # Less than 1KB
            raise ValueError(f"Model file appears to be corrupted (size: {file_size} bytes)")
    except Exception as e:
        raise ValueError(f"Model file validation failed: {e}")

    try:
        model = YOLO(str(weights_path))
        
        # Test model loading
        try:
            # Create a dummy image to test model
            dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
            _ = model.predict(dummy_img, verbose=False)
        except Exception as e:
            logging.warning(f"Model test prediction failed: {e}")
        
        # try to set device explicitly
        try:
            model.to(device)
        except Exception as e:
            logging.warning(f"Failed to move model to device {device}: {e}")
            
        _cached_models["yolo"] = model
        logging.info(f"Successfully loaded YOLO model from {weights_path}")
        return model
        
    except Exception as e:
        raise RuntimeError(f"Failed to load YOLO model from {weights_path}: {e}") from e



def _safe_extract_box_and_score(b):
    """
    Accept various ultralytics box representations and return (xyxy, score, cls_id)
    xyxy => numpy array length 4 (x1,y1,x2,y2) as ints or None on failure
    """
    # xyxy
    xyxy = None
    try:
        arr = b.xyxy.cpu().numpy()
        # Could be shape (N,4) or (4,)
        if arr.ndim == 2 and arr.shape[0] >= 1:
            xyxy = arr[0][:4]
        elif arr.ndim == 1 and arr.size >= 4:
            xyxy = arr[:4]
    except Exception:
        try:
            arr = np.array(b.xyxy)
            if arr.ndim == 2 and arr.shape[0] >= 1:
                xyxy = arr[0][:4]
            elif arr.ndim == 1 and arr.size >= 4:
                xyxy = arr[:4]
        except Exception:
            xyxy = None

    if xyxy is None:
        return None, 0.0, -1

    # score
    score = 0.0
    try:
        score = float(b.conf.cpu().numpy())
    except Exception:
        try:
            score = float(b.conf)
        except Exception:
            score = 0.0

    # class id
    cls_id = -1
    try:
        cls_id = int(b.cls.cpu().numpy())
    except Exception:
        try:
            cls_id = int(b.cls)
        except Exception:
            cls_id = -1

    return xyxy, float(score), int(cls_id)

def run_detection_and_classification(pil_image: Image.Image, conf_threshold: float = 0.25, show_global_badge: bool = True):
    """
    Run YOLO detection on a PIL image and return:
        annotated_np: np.array(H,W,3) RGB annotated image (with numbered boxes + global count badge)
        detections: list of dicts: {'bbox':[x1,y1,x2,y2], 'score':float, 'class_id':int, 'label': str}
    """
    try:
        model = _load_yolo_model()
    except Exception as e:
        logging.error(f"Failed to load model: {e}")
        raise RuntimeError(f"Model loading failed: {e}") from e
    
    try:
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")

        img_np = np.array(pil_image)
        
        # Validate image
        if img_np.size == 0:
            raise ValueError("Empty image provided")
        
        if len(img_np.shape) != 3 or img_np.shape[2] != 3:
            raise ValueError(f"Invalid image shape: {img_np.shape}. Expected (H, W, 3)")
        
        results = model.predict(source=img_np, conf=conf_threshold, verbose=False)
        
    except Exception as e:
        logging.error(f"Detection failed: {e}")
        raise RuntimeError(f"Detection failed: {e}") from e

    annotated = pil_image.copy()
    draw = ImageDraw.Draw(annotated)

    # get class name map, fall back gracefully
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        id2name = names
    elif isinstance(names, list):
        id2name = {i: n for i, n in enumerate(names)}
    else:
        id2name = {}

    detections = []
    car_like_labels = {"car"}  # tweak if your model uses e.g. "automobile"

    # --- Collect detections ---
    boxes = []
    if len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
        for b in results[0].boxes:
            xyxy, score, cls_id = _safe_extract_box_and_score(b)
            if xyxy is None:
                continue
            x1, y1, x2, y2 = [int(max(0, int(round(v)))) for v in xyxy[:4]]
            label = id2name.get(int(cls_id), str(int(cls_id)))
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "score": float(score),
                "class_id": int(cls_id),
                "label": label,
            })
            boxes.append((x1, y1, x2, y2, float(score), label))

    # --- Draw numbered boxes + per-box label ---
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_big = ImageFont.truetype("DejaVuSans.ttf", 24)
    except Exception:
        font = ImageFont.load_default()
        font_big = ImageFont.load_default()

    car_idx = 0
    for (x1, y1, x2, y2, score, label) in boxes:
        is_car = (label.lower() in car_like_labels) or (len(car_like_labels) == 0)
        # Only increment index for car class
        if is_car:
            car_idx += 1
            tag = f"Car #{car_idx}"
        else:
            tag = f"{label}"

        # Draw rectangle (green)
        draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=3)

        # Build text "Car #i  (0.92)"
        text = f"{tag}  ({score:.2f})"
        # Text size
        try:
            tx, ty, bx, by = draw.textbbox((0, 0), text, font=font)
            tw, th = bx - tx, by - ty
        except Exception:
            tw, th = font.getsize(text)

        # Text background above the box
        text_bg = [x1, max(0, y1 - th - 8), x1 + tw + 10, y1]
        draw.rectangle(text_bg, fill=(0, 128, 0))
        draw.text((x1 + 5, max(0, y1 - th - 5)), text, fill=(255, 255, 255), font=font)

    # --- Global counter badge (top-left) ---
    if show_global_badge:
        car_count = sum(1 for d in detections if d.get("label", "").lower() in car_like_labels)
        badge_text = f"Cars: {car_count}"
        try:
            tx, ty, bx, by = draw.textbbox((0, 0), badge_text, font=font_big)
            tw, th = bx - tx, by - ty
        except Exception:
            tw, th = font_big.getsize(badge_text)

        pad = 8
        badge_bg = [10, 10, 10 + tw + 2 * pad, 10 + th + 2 * pad]
        draw.rectangle(badge_bg, fill=(0, 0, 0))
        draw.text((10 + pad, 10 + pad), badge_text, fill=(255, 255, 255), font=font_big)

    try:
        annotated_np = np.array(annotated)
        return annotated_np, detections
    except Exception as e:
        logging.error(f"Failed to convert annotated image to numpy array: {e}")
        # Return original image if annotation fails
        return np.array(pil_image), detections

