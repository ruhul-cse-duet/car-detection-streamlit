# app.py
import streamlit as st
import time
import logging
import tempfile
import os
from pathlib import Path
import warnings
import io
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import traceback

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# Import configuration
try:
    from config import (
        MODEL_CONFIG, DETECTION_CONFIG, TRACKING_CONFIG, 
        VIDEO_CONFIG, UI_CONFIG, validate_config, get_env_config
    )
    # Validate configuration
    config_errors = validate_config()
    if config_errors:
        st.error("Configuration errors found:")
        for error in config_errors:
            st.error(f"• {error}")
        st.stop()
except ImportError as e:
    st.error(f"Failed to import configuration: {e}")
    st.stop()

# Import detection functions
try:
    from src.custom_resnet import run_detection_and_classification
    from src.tracker import CarTracker
except ImportError as e:
    st.error(f"Failed to import detection modules: {e}")
    st.stop()

# Get device configuration
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"
if MODEL_CONFIG["device"] != "auto":
    DEVICE = MODEL_CONFIG["device"]

st.set_page_config(
    page_title=UI_CONFIG["page_title"], 
    layout=UI_CONFIG["layout"],
    initial_sidebar_state="collapsed" if not UI_CONFIG["show_sidebar"] else "expanded"
)

# --- UI state defaults
if "page" not in st.session_state:
    st.session_state["page"] = "home"
if "uploader_key" not in st.session_state:
    st.session_state["uploader_key"] = 0

page = st.session_state["page"]
MAX_FRAMES_IN_MEMORY = VIDEO_CONFIG["max_frames_in_memory"]

# Add custom CSS
def load_css():
    css_file = Path("assets/style.css")
    if css_file.exists():
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()


# ---------- Helper: process video to frames (unused here, kept for reference) ----------
def process_video_to_frames_bytes(video_path: str, max_frames: int = MAX_FRAMES_IN_MEMORY):
    frames_bytes = []
    processed = 0
    abort_reason = None
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError("Failed to open video file for processing.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    try:
        pbar = st.progress(0.0)
        with st.spinner("Running detection on frames..."):
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil = Image.fromarray(rgb)

                try:
                    # Try with show_global_badge=False if your function supports it
                    try:
                        annotated_np, detections = run_detection_and_classification(
                            pil, conf_threshold=0.25, show_global_badge=False
                        )
                    except TypeError:
                        annotated_np, detections = run_detection_and_classification(
                            pil, conf_threshold=0.25
                        )
                except Exception as e:
                    logging.error(f"Detection error on frame {processed}: {e}")
                    annotated_np = rgb

                bgr_out = cv2.cvtColor(annotated_np, cv2.COLOR_RGB2BGR)
                success, encoded = cv2.imencode(".jpg", bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                frame_bytes = encoded.tobytes() if success else io.BytesIO()
                frames_bytes.append(frame_bytes)
                processed += 1

                if processed >= max_frames:
                    abort_reason = "too_many_frames"
                    break
                if total_frames > 0:
                    pbar.progress(min(processed / total_frames, 1.0))
    finally:
        cap.release()
        pbar.empty()

    return frames_bytes, fps, w, h, processed, total_frames, abort_reason


# ---------- HOME PAGE ----------
if page == "home":
    st.markdown('<h1 class="title">🚗 Car Detection System</h1>', unsafe_allow_html=True)
    
    # System status
    with st.expander("🔧 System Status", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Device", DEVICE, help="Current processing device")
        with col2:
            st.metric("Model", MODEL_CONFIG["model_name"], help="Detection model")
        with col3:
            st.metric("Confidence", f"{MODEL_CONFIG['confidence_threshold']:.2f}", help="Detection threshold")
    
    # Main content
    colA, colB = st.columns([1.2, 1])
    with colA:
        st.markdown("""
        <div class="hero">
            <p class="hero-t">
                Upload a car image or video to detect and track vehicles using our advanced YOLOv11-based AI model. 
                The system provides real-time object detection with unique ID tracking for video analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            st.session_state["page"] = "analysis"
            st.session_state["uploader_key"] += 1
            st.rerun()
            
        # Features
        st.markdown("### ✨ Features")
        features = [
            "🎯 AI-Powered Car Detection",
            "📹 Real-time Video Tracking", 
            "🔢 Unique ID Assignment",
            "📊 Confidence Scoring",
            "⚡ GPU Acceleration"
        ]
        for feature in features:
            st.markdown(f"• {feature}")
    
    with colB:
        sample_image_path = "test_img/vid_5_27620.jpg"
        if os.path.exists(sample_image_path):
            st.image(sample_image_path, caption="Example Car Detection", width=500)
        else:
            st.info("Sample image not found.")
            
        # Quick stats
        st.markdown("### 📈 Model Performance")
        st.metric("mAP50", "99.4%", help="Mean Average Precision at IoU 0.5")
        st.metric("mAP50-95", "70.5%", help="Mean Average Precision at IoU 0.5-0.95")

    st.markdown("""
                <div class="footer">For education use/Developed By Ruhul Amin</div>
            """, unsafe_allow_html=True)
# ---------- ANALYSIS PAGE ----------
elif page == "analysis":
    st.markdown('<h2 class="title">🔍 Car Detection Analysis</h2>', unsafe_allow_html=True)

    # Settings sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        confidence_threshold = st.slider(
            "Confidence Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=MODEL_CONFIG["confidence_threshold"],
            step=0.05,
            help="Minimum confidence for detections"
        )
        
        iou_threshold = st.slider(
            "IoU Threshold", 
            min_value=0.1, 
            max_value=1.0, 
            value=TRACKING_CONFIG["iou_threshold"],
            step=0.05,
            help="IoU threshold for tracking"
        )
        
        max_age = st.slider(
            "Max Track Age", 
            min_value=5, 
            max_value=50, 
            value=TRACKING_CONFIG["max_age"],
            step=1,
            help="Maximum frames to keep lost tracks"
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### 📁 Upload Media")
        uploaded_img = st.file_uploader(
            "Upload Image", 
            type=["jpg", "jpeg", "png"], 
            key=f"img_{st.session_state['uploader_key']}",
            help="Upload an image to detect cars"
        )
        uploaded_vid = st.file_uploader(
            "Upload Video", 
            type=["mp4", "avi", "mov", "mkv"], 
            key=f"vid_{st.session_state['uploader_key']}",
            help="Upload a video to detect and track cars"
        )
        
        # System info
        st.markdown("### 💻 System Info")
        st.info(f"**Device:** {DEVICE}")
        st.info(f"**Model:** {MODEL_CONFIG['model_name']}")

        # --- Buttons ---
        st.markdown("### 🎮 Controls")
        back_col, clear_col = st.columns(2)
        with back_col:
            if st.button("⬅️ Back Home", width=300):
                st.session_state["page"] = "home"
                st.session_state["uploader_key"] += 1
                st.rerun()
        with clear_col:
            if st.button("🧹 Clear All", width=300):
                st.session_state["uploader_key"] += 1
                st.rerun()

    with col2:
        if uploaded_img is None and uploaded_vid is None:
            st.info("Upload an image or video to start detection.")


        # ----------------- Image Detection -------------------------
        if uploaded_img is not None:
            try:
                img = Image.open(uploaded_img).convert("RGB")
                st.image(img, caption="Input Image", width=400)
                
                # Image info
                st.markdown("### 📊 Image Information")
                col_info1, col_info2 = st.columns(2)
                with col_info1:
                    st.metric("Width", f"{img.width}px")
                with col_info2:
                    st.metric("Height", f"{img.height}px")
                
                if st.button("🔍 Run Image Detection", type="primary", use_container_width=True):
                    with st.spinner("Running detection on image..."):
                        try:
                            # Update progress
                            progress_bar = st.progress(0)
                            status_text = st.empty()
                            
                            status_text.text("Loading model...")
                            progress_bar.progress(20)
                            
                            # Try with optional flag if available
                            try:
                                annotated_np, detections = run_detection_and_classification(
                                    img, 
                                    conf_threshold=confidence_threshold, 
                                    show_global_badge=True
                                )
                            except TypeError:
                                annotated_np, detections = run_detection_and_classification(
                                    img, 
                                    conf_threshold=confidence_threshold
                                )
                            
                            status_text.text("Processing results...")
                            progress_bar.progress(80)
                            
                            annotated_img = Image.fromarray(annotated_np)
                            st.image(annotated_img, caption=f"Detections ({len(detections)})", width=600)
                            
                            # Results summary
                            num_cars = len(detections)
                            st.success(f"🚗 Number of cars detected: {num_cars}")
                            
                            # Detailed results
                            if detections:
                                st.markdown("### 📋 Detection Details")
                                for i, detection in enumerate(detections, 1):
                                    with st.expander(f"Car #{i} - Confidence: {detection['score']:.3f}"):
                                        st.write(f"**Bounding Box:** {detection['bbox']}")
                                        st.write(f"**Class:** {detection['label']}")
                                        st.write(f"**Confidence:** {detection['score']:.3f}")
                            
                            progress_bar.progress(100)
                            status_text.text("Detection completed!")
                            
                        except Exception as e:
                            st.error(f"Image detection failed: {e}")
                            st.error(f"Error details: {traceback.format_exc()}")
                            
            except Exception as e:
                st.error(f"Failed to load image: {e}")
                st.error("Please try uploading a different image.")


        # ---------------------------- Video Detection + Tracker (UNIQUE COUNT) -------------------------------------
        if uploaded_vid is not None:
            st.video(uploaded_vid)
            
            # Video info
            st.markdown("### 📊 Video Information")
            col_vid1, col_vid2, col_vid3 = st.columns(3)
            with col_vid1:
                st.metric("File Size", f"{uploaded_vid.size / (1024*1024):.1f} MB")
            with col_vid2:
                st.metric("File Type", uploaded_vid.type)
            with col_vid3:
                st.metric("File Name", uploaded_vid.name)
            
            if st.button("🎬 Run Video Detection", type="primary", use_container_width=True):
                # Save uploaded video temporarily
                try:
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_vid.name)[1])
                    tfile.write(uploaded_vid.read())
                    tfile.flush()
                    tfile.close()

                    # Process video into frames with detection + tracking
                    frames_bytes = []
                    detections_per_frame = []
                    abort_reason = None
                    fps, w, h, processed, total_frames = 25.0, 640, 480, 0, 0

                    # Initialize tracker with user settings
                    tracker = CarTracker(iou_threshold=iou_threshold, max_age=max_age)
                    seen_ids = set()
                    
                    # Progress tracking
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    metrics_col1, metrics_col2, metrics_col3 = st.columns(3)

                    try:
                        cap = cv2.VideoCapture(str(tfile.name))
                        if not cap.isOpened():
                            st.error("Failed to open video.")
                        else:
                            fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
                            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
                            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
                            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                            
                            status_text.text("Initializing video processing...")
                            progress_bar.progress(0.1)

                            with st.spinner("Processing video frames..."):
                                frame_count = 0
                                while True:
                                    ret, frame = cap.read()
                                    if not ret:
                                        break

                                    # BGR -> RGB
                                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    pil = Image.fromarray(rgb)

                                    # Run detection (no global badge in video loop)
                                    try:
                                        try:
                                            annotated_np, detections = run_detection_and_classification(
                                                pil, conf_threshold=confidence_threshold, show_global_badge=False
                                            )
                                        except TypeError:
                                            annotated_np, detections = run_detection_and_classification(
                                                pil, conf_threshold=confidence_threshold
                                            )
                                    except Exception as e:
                                        logging.warning(f"Detection failed on frame {frame_count}: {e}")
                                        annotated_np = rgb
                                        detections = []

                                    detections_per_frame.append(detections)

                                    # model is car-only -> all bboxes are cars
                                    car_boxes = [d["bbox"] for d in detections]

                                    # Update tracker, get tracks with IDs
                                    tracks = tracker.update(car_boxes)

                                    # Update unique set
                                    for t in tracks:
                                        seen_ids.add(t["id"])

                                    # --- Draw IDs and "Unique Cars" badge on the frame ---
                                    pil_annot = Image.fromarray(annotated_np)
                                    draw = ImageDraw.Draw(pil_annot)
                                    try:
                                        font_id = ImageFont.truetype("DejaVuSans.ttf", 20)
                                        font_badge = ImageFont.truetype("DejaVuSans.ttf", 24)
                                    except Exception:
                                        font_id = ImageFont.load_default()
                                        font_badge = ImageFont.load_default()

                                    # Per-box ID tag
                                    for t in tracks:
                                        x1, y1, x2, y2 = t["bbox"]
                                        label = f"ID {t['id']}"
                                        try:
                                            tx1, ty1, tx2, ty2 = draw.textbbox((0, 0), label, font=font_id)
                                            tw, th = tx2 - tx1, ty2 - ty1
                                        except Exception:
                                            tw, th = font_id.getsize(label)
                                        tag_bg = [x1, max(0, y1 - th - 8), x1 + tw + 10, y1]
                                        draw.rectangle(tag_bg, fill=(0, 0, 0))
                                        draw.text((x1 + 5, max(0, y1 - th - 5)), label, fill=(255, 255, 255), font=font_id)

                                    # Unique badge
                                    badge_text = f"Unique Cars: {len(seen_ids)}"
                                    try:
                                        bx1, by1, bx2, by2 = draw.textbbox((0, 0), badge_text, font=font_badge)
                                        bw, bh = bx2 - bx1, by2 - by1
                                    except Exception:
                                        bw, bh = font_badge.getsize(badge_text)
                                    pad = 8
                                    draw.rectangle([10, 10, 10 + bw + 2 * pad, 10 + bh + 2 * pad], fill=(0, 0, 0))
                                    draw.text((10 + pad, 10 + pad), badge_text, fill=(255, 255, 255), font=font_badge)

                                    # Convert to JPEG bytes for inline playback
                                    bgr_out = cv2.cvtColor(np.array(pil_annot), cv2.COLOR_RGB2BGR)
                                    success, encoded = cv2.imencode(".jpg", bgr_out, [int(cv2.IMWRITE_JPEG_QUALITY), VIDEO_CONFIG["jpeg_quality"]])
                                    frame_bytes = encoded.tobytes() if success else None
                                    if frame_bytes:
                                        frames_bytes.append(frame_bytes)

                                    frame_count += 1
                                    processed += 1
                                    
                                    # Update progress
                                    if total_frames > 0:
                                        progress = min(processed / total_frames, 1.0)
                                        progress_bar.progress(progress)
                                        status_text.text(f"Processing frame {processed}/{total_frames} - Unique cars: {len(seen_ids)}")

                                    # Safety: avoid memory overflow
                                    if processed >= MAX_FRAMES_IN_MEMORY:
                                        abort_reason = "too_many_frames"
                                        st.warning(f"Video too long - processing first {MAX_FRAMES_IN_MEMORY} frames only")
                                        break

                        cap.release()
                        progress_bar.progress(1.0)
                        status_text.text("Video processing completed!")

                    except Exception as e:
                        st.error(f"Video processing failed: {e}")
                        st.error(f"Error details: {traceback.format_exc()}")
                        abort_reason = "error"
                    finally:
                        try:
                            os.remove(tfile.name)
                        except Exception:
                            pass

                    # ---- Playback and final outputs ----
                    if abort_reason == "too_many_frames":
                        st.warning("Video too long for memory playback. Creating output video file...")
                        tmp_out = Path(tempfile.gettempdir()) / f"annotated_{int(time.time())}.mp4"
                        writer = cv2.VideoWriter(str(tmp_out), cv2.VideoWriter_fourcc(*VIDEO_CONFIG["video_codec"]), fps, (w, h))
                        for fb in frames_bytes:
                            arr = np.frombuffer(fb, np.uint8)
                            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                            if frame is not None:
                                writer.write(frame)
                        writer.release()
                        
                        st.success("✅ Annotated video created!")
                        with open(tmp_out, "rb") as f:
                            st.video(f.read())
                            
                    elif abort_reason is None and frames_bytes:
                        st.success(f"Processed {processed} frames — Playing now...")
                        placeholder = st.empty()
                        delay = 1.0 / min(fps, VIDEO_CONFIG["output_fps"])
                        for fb in frames_bytes:
                            placeholder.image(fb, width="stretch")
                            time.sleep(delay)

                        # Summary metrics
                        st.success(f"✅ Unique cars in this video: {len(seen_ids)}")
                        
                        # Final statistics
                        st.markdown("### 📊 Video Analysis Summary")
                        col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
                        with col_sum1:
                            st.metric("Total Frames", processed)
                        with col_sum2:
                            st.metric("Unique Cars", len(seen_ids))
                        with col_sum3:
                            st.metric("Average FPS", f"{fps:.1f}")
                        with col_sum4:
                            st.metric("Video Duration", f"{processed/fps:.1f}s")

                    else:
                        st.error("Video processing failed or no frames processed.")
                        
                except Exception as e:
                    st.error(f"Failed to process video: {e}")
                    st.error("Please try uploading a different video file.")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div class="footer">
        <p>🚗 Car Object Detection System | Powered by YOLOv11 & Streamlit</p>
        <p><em>Note: Inline playback simulates video via frame updates.</em></p>
    </div>
    """, unsafe_allow_html=True)