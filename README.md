# 🚦 Smart Traffic Surveillance System

An intelligent traffic monitoring system that detects traffic light violations using computer vision and machine learning. The system automatically identifies vehicles that run red lights, reads license plates, and generates violation reports.

## ✨ Features

- 🎥 **Real-time Video Processing**: Analyze traffic footage frame by frame
- 🚦 **Traffic Light Detection**: Automatically detect red, yellow, and green lights
- 🚗 **Vehicle Tracking**: Track cars, trucks, buses, and motorcycles
- 📸 **License Plate Recognition**: Read and extract vehicle license plates
- ⚠️ **Violation Detection**: Identify red light violations with stop line crossing
- 📊 **Live Violation Display**: Real-time grid view of detected violations
- 📁 **Export Capabilities**: Download CSV reports and processed videos

## 🛠️ Technology Stack

- **FastAPI**: High-performance web framework for the backend API
- **HTML/CSS/JavaScript**: Custom frontend with real-time WebSocket communication
- **OpenCV**: Computer vision and image processing
- **YOLO**: Object detection (vehicles and license plates)
- **EasyOCR**: Optical character recognition for license plates
- **NumPy**: Numerical operations and array handling
- **Uvicorn**: ASGI server for FastAPI

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for better performance)

## 🚀 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/smart-surveillance-system.git
   cd smart-surveillance-system
   ```

2. **Create virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Download required models** (if not included in repo)

   ```bash
   # These will be downloaded automatically on first run
   # Or manually place them in the project directory:
   # - yolov8s.pt (YOLOv8 small model)
   # - plate_detector.pt (License plate detection model)
   ```

5. **Run the application**

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

6. **Open in browser**
   Navigate to `http://localhost:8000`

## 🎯 How to Use

1. **Start the server**

   ```bash
   uvicorn app:app --reload --host 0.0.0.0 --port 8000
   ```

2. **Open the web interface**
   - Navigate to `http://localhost:8000`
   - The web interface will load automatically

3. **Upload a video**
   - Click "Browse files" or drag & drop your traffic video
   - Supported formats: MP4, AVI, MOV, MKV
   - Wait for the video to upload and process

4. **Draw the stop line**
   - On the first frame, click and drag to draw a stop line
   - This line represents where vehicles should stop at red lights
   - Use the undo/redo/delete buttons if needed

5. **Configure settings**
   - Adjust detection confidence threshold (0.2-0.8)
   - Lower values detect more but may include false positives

6. **Start analysis**
   - Click "▶️ Start Analysis" to begin processing
   - Watch live violations appear in the grid below
   - Monitor the progress bar and frame counter

7. **Review results**
   - View violation statistics and detailed reports
   - Download CSV file with violation data
   - Download processed video with violation overlays
   - Browse violation snapshots in grid view

## 📊 Output Files

- **`output_final.mp4`**: Processed video with violation overlays
- **`violations.csv`**: Detailed violation report with timestamps
- **`violations_snapshots/`**: Individual violation images

## 🎨 UI Features

- **Modern Web Interface**: Custom HTML/CSS frontend with dark theme
- **Real-time WebSocket**: Live updates during video processing
- **Live Violation Grid**: Real-time display of up to 5 violations
- **Violation Snapshots**: 3-column grid of all detected violations
- **Info Panels**: On-screen violation details with plate zoom
- **Responsive Design**: Works on desktop and tablet devices
- **Interactive Canvas**: Draw stop lines with undo/redo functionality

## 🔧 Configuration

### Detection Parameters

- **Confidence Threshold**: Minimum confidence for object detection
- **Panel Duration**: How long violation panels stay on screen (90 frames)
- **Grid Layout**: Configurable grid columns for violations

### Customization

You can modify detection parameters in the code:

- Vehicle classes to detect
- Traffic light color thresholds
- Plate recognition settings
- Panel styling and positioning

## 🐛 Troubleshooting

### Common Issues

1. **"No GPU detected"**: The system works on CPU but will be slower
2. **"Model not found":** Models download automatically on first run
3. **"WebSocket connection failed":** Refresh the page and ensure port 8000 is available
4. **"Video upload fails":** Check file format and size (max 100MB recommended)
5. **"Canvas not responsive":** Ensure JavaScript is enabled in browser

### Performance Tips

- Use GPU for faster processing
- Lower confidence threshold for more detections
- Reduce video resolution for faster analysis
- Close other applications to free up memory

## 📁 Project Structure

```
smart-surveillance-system/
├── app.py                 # FastAPI backend application
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── .gitignore            # Git ignore file
├── yolov8s.pt            # YOLO model (auto-download)
├── plate_detector.pt     # License plate model (auto-download)
├── static/               # Static assets (CSS, JS, images)
├── templates/            # HTML templates
│   └── index.html        # Main frontend interface
├── violations_snapshots/ # Generated violation images
└── output_final.mp4      # Processed video output
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) for YOLO models
- [EasyOCR](https://github.com/JaidedAI/EasyOCR) for license plate recognition
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [OpenCV](https://opencv.org/) for computer vision
- [Uvicorn](https://www.uvicorn.org/) for the ASGI server

## 📞 Support

If you encounter any issues or have questions:

- Open an issue on GitHub
- Check the troubleshooting section above
- Review the code comments for additional details

---

**⚠️ Disclaimer**: This system is for educational and demonstration purposes. For real-world traffic enforcement, ensure compliance with local regulations and privacy laws.
