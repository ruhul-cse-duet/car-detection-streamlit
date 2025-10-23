# 🚗 Car Object Detection System

A powerful AI-powered car detection system built with YOLOv11 and Streamlit. Upload images or videos to detect and track cars with real-time object tracking capabilities.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)
![YOLO](https://img.shields.io/badge/YOLO-v11-green.svg)

## 🚀 Features

- **AI-Powered Detection**: Uses a fine-tuned YOLOv11s model for car detection
- **Real-time Tracking**: Advanced object tracking with unique ID assignment
- **Video Processing**: Process videos with frame-by-frame detection and tracking
- **Modern UI**: Beautiful, responsive interface with dark/light mode support
- **Docker Support**: Easy deployment with Docker
- **Performance Optimized**: GPU acceleration with automatic CPU fallback
- **Mobile Responsive**: Works on desktop, tablet, and mobile devices

## 🎯 Detection Capabilities

- **Object Detection**: Detect cars in images and videos
- **Object Tracking**: Track cars across video frames with unique IDs
- **Confidence Scoring**: Display detection confidence scores
- **Batch Processing**: Handle multiple images and long videos
- **Real-time Analysis**: Live video processing with tracking

## 🏗️ Architecture

- **Detection Model**: YOLOv11s fine-tuned for car detection
- **Tracking System**: Custom IoU-based tracker for object persistence
- **Web Interface**: Streamlit-based responsive UI
- **Backend**: Python with PyTorch and OpenCV

## 📁 Project Structure

```
Car Object Detection Project/
├── app.py                          # Main Streamlit application
├── src/
│   ├── custom_resnet.py           # YOLO detection inference
│   └── tracker.py                  # Object tracking system
├── models/
│   ├── yolov11_trained.pt         # Trained YOLO model weights
│   └── m.py                       # Model utilities
├── test_img/                      # Sample test images
├── Videos/                        # Sample test videos
├── assets/
│   └── style.css                  # Custom styling
├── requirements.txt               # Python dependencies
├── Dockerfile                     # Docker configuration
└── README.md                      # This file
```

## 🚀 Quick Start

### Local Setup
```bash
# Clone the repository
git clone <repository-url>
cd "Car Object Detection Project"

# Create virtual environment
python -m venv .venv
. .venv/Scripts/activate  # Windows
# source .venv/bin/activate  # Linux/macOS

# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```
Open the app at `http://localhost:8501`.

### Docker Deployment
```bash
# Build the Docker image
docker build -t car-detection-app .

# Run the container
docker run --rm -p 8501:8501 -v $(pwd)/models:/app/models car-detection-app
```

## 🎮 Usage

1. **Upload Media**: Upload images or videos using the file uploader
2. **Image Detection**: Click "Run Image Detection" to detect cars in images
3. **Video Processing**: Click "Run Video Detection" to process videos with tracking
4. **View Results**: See detected cars with bounding boxes, confidence scores, and unique IDs

## ⚙️ Configuration

The system uses the following default parameters:
- **Confidence Threshold**: 0.25
- **IoU Threshold**: 0.35 (for tracking)
- **Max Track Age**: 15 frames
- **Model**: YOLOv11s fine-tuned for car detection

## 🔧 Model Training

The model was trained on a car detection dataset with the following specifications:
- **Architecture**: YOLOv11s
- **Classes**: 1 (car)
- **Input Size**: 640x640
- **Epochs**: 50
- **Performance**: mAP50: 99.4%, mAP50-95: 70.5%

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Car detection datasets used for training
- PyTorch and Streamlit communities
- Ultralytics YOLO team
- Open source contributors

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/ruhul-cse-duet/car-detection-streamlit/issues) page
2. Create a new issue with detailed information
3. Contact the maintainers

## 🔮 Future Enhancements

- [ ] Support for multiple vehicle types (trucks, motorcycles, etc.)
- [ ] Real-time webcam detection
- [ ] Advanced analytics and reporting
- [ ] Batch processing capabilities
- [ ] API endpoints for integration
- [ ] Mobile app development
- [ ] Multi-language support
- [ ] Cloud deployment options

---
## Author
[Md Ruhul Amin](https://www.linkedin.com/in/ruhul-duet-cse/);  
Email: ruhul.cse.duet@gmail.com

**Disclaimer**: This application is for educational and research purposes only. It should not be used for critical safety applications without proper validation.

**Made with ❤️ for the Computer Vision AI community**

