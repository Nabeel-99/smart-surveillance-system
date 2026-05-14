# 🚦 Smart Traffic Surveillance System

> An intelligent traffic monitoring system that detects red-light violations using computer vision and machine learning. The system identifies violating vehicles, reads their license plates, and generates detailed violation reports — automatically.

---

## 📽️ Demo Video

https://github.com/YOUR_USERNAME/YOUR_REPO/assets/YOUR_ASSET_ID/demo_video.mp4

> **To publish:** Go to your GitHub repo → click any Issue or the README edit box → drag and drop `demo_video.mp4` into the text area → GitHub will generate a permanent URL. Paste that URL here and delete this tip.

---

## 🎬 Sample Output Video

https://github.com/YOUR_USERNAME/YOUR_REPO/assets/YOUR_ASSET_ID/output_video.mp4

> **To publish:** Same as above — drag `output_video.mp4` into a GitHub Issue/PR text box, copy the generated URL, paste it here.

---

## ✨ Features

- 🎥 **Real-time Video Processing** — Analyze pre-recorded traffic footage frame by frame
- 🚦 **Traffic Light Detection** — Automatically detect red, yellow, and green lights via HSV color analysis
- 🚗 **Vehicle Tracking** — Track cars, trucks, buses, and motorcycles using YOLO + ByteTrack
- 📸 **License Plate Recognition** — Multi-variant EasyOCR preprocessing pipeline
- ⚠️ **Violation Detection** — Flags vehicles that cross the stop line while the light is red
- 🖥️ **Live Preview Mode** — Connect an IP camera (DroidCam/similar) for real-time monitoring
- 📊 **Live Violation Grid** — Real-time display of detected violations in the browser
- 📁 **Export Capabilities** — Download CSV reports and a highlights clip video
- 🔴 **Virtual Traffic Light (VTL)** — Manual override to simulate a red/amber/green signal when the camera cannot see the physical light

---

## 🛠️ Technology Stack

| Layer             | Technology                       |
| ----------------- | -------------------------------- |
| Backend API       | FastAPI + Uvicorn (ASGI)         |
| Real-time Comms   | WebSockets                       |
| Computer Vision   | OpenCV                           |
| Object Detection  | YOLOv8 (ultralytics)             |
| Vehicle Tracking  | ByteTrack (built into YOLO)      |
| License Plate OCR | EasyOCR (multi-variant pipeline) |
| Frontend          | Vanilla HTML / CSS / JavaScript  |
| Numerical Ops     | NumPy                            |

---

## 📋 Prerequisites

- Python **3.8** or higher
- CUDA-compatible GPU — **strongly recommended** (CPU works but is significantly slower)
- ~4 GB disk space for models on first run

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/smart-surveillance-system.git
cd smart-surveillance-system
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Make sure your CUDA version matches the `torch` wheel. Visit https://pytorch.org/get-started/locally/ if you need a specific build.

### 4. Model weights

The two YOLO models download automatically on first run:

| File                | Purpose                                                            |
| ------------------- | ------------------------------------------------------------------ |
| `yolov8s.pt`        | Vehicle + traffic-light detection (auto-downloaded by ultralytics) |
| `plate_detector.pt` | License plate bounding-box detection — **you must supply this**    |

Place `plate_detector.pt` in the project root before starting. If you have a custom model trained on your region's plates, drop it in the same location and rename it accordingly (or update the path in `app.py → get_models()`).

### 5. Run the server

```bash
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### 6. Open in browser

```
http://localhost:8000
```

---

## 🎯 How to Use

### Step-by-step

1. **Upload a video** — click "Browse files" or drag-and-drop. Supported: MP4, AVI, MOV, MKV (max ~100 MB recommended).
2. **Draw the stop line** — on the first frame preview, click and drag to mark where vehicles must stop. Use the undo/redo/delete buttons to adjust.
3. **Configure confidence** — the slider controls YOLO detection confidence (0.2–0.8). Lower values catch more vehicles but increase false positives.
4. **Virtual Traffic Light (optional)** — if the physical traffic light is not visible in your footage, enable the VTL toggle and set the colour manually to simulate signal state.
5. **Start analysis** — click ▶️ **Start Analysis**. Live violation cards will appear as the system processes each frame.
6. **Review results** — once complete, download the CSV report and/or the highlights clip from the results panel.

### Live camera mode (IP camera)

1. Install [DroidCam](https://www.dev47apps.com/) (or any IP camera app) on your phone.
2. In the web UI, enter the phone's local IP and click **Connect**.
3. Draw the stop line on the live preview, then click **Start Analysis**.

---

## 📁 Project Structure

```
smart-surveillance-system/
├── app.py                     # FastAPI backend — all routes, WebSocket handlers, CV logic
├── requirements.txt           # Python dependencies
├── README.md                  # This file
├── .gitignore
├── yolov8s.pt                 # Auto-downloaded on first run
├── plate_detector.pt          # Supply manually (see Installation)
├── templates/
│   └── index.html             # Single-page frontend
├── static/                    # Static assets (CSS, JS, images)
├── sessions/                  # Auto-created per analysis run
│   └── YYYY-MM-DD_HH-MM-SS/
│       ├── snapshots/         # Per-vehicle violation images
│       └── highlights.mp4     # Violation clip (±3 s around each event)
└── violations.csv             # Last-run violation report (written by frontend download)
```

### Key sections inside `app.py`

| Section               | What it does                                                                                   |
| --------------------- | ---------------------------------------------------------------------------------------------- |
| `get_models()`        | Lazy-loads YOLO + EasyOCR once; shared across all requests                                     |
| `get_light_color()`   | HSV-based traffic light colour classifier (top/mid/bottom thirds)                              |
| `draw_vtl_on_frame()` | Renders the virtual traffic light overlay onto frames                                          |
| `crosses_line()`      | Cross-product geometry test — checks if a vehicle bounding box straddles the stop line         |
| `read_plate()`        | Multi-variant OCR: scales, sharpens, thresholds, tries beam-search EasyOCR on 8 image variants |
| `PlateOCRQueue`       | Background `ThreadPoolExecutor` so OCR never blocks the main async loop                        |
| `ViolationClipWriter` | Ring-buffer that saves ±3 s of footage around each violation event                             |
| `/ws/analyze`         | Main WebSocket — frame loop, YOLO tracking, violation logic, OCR polling                       |
| `/ws/live-preview`    | Separate WebSocket for IP-camera live preview (no analysis)                                    |

---

## 📊 Output Files

| File                                     | Description                                                         |
| ---------------------------------------- | ------------------------------------------------------------------- |
| `sessions/<timestamp>/highlights.mp4`    | Short clip of all violation events (±3 s each)                      |
| `sessions/<timestamp>/snapshots/`        | Individual JPEG snapshots per violating vehicle                     |
| `violations.csv` (downloaded in browser) | Timestamped violation log: frame, time, car ID, plate, vehicle type |

---

## ⚙️ Configuration Reference

All tunable constants are at the top of the `analyze()` WebSocket handler in `app.py`:

| Constant            | Default         | Effect                                                   |
| ------------------- | --------------- | -------------------------------------------------------- |
| `YOLO_SKIP`         | `3`             | Run YOLO every N frames (higher = faster, less accurate) |
| `STALE_FRAMES`      | `YOLO_SKIP + 1` | Frames before a missing vehicle box is cleared           |
| `SEND`              | `3`             | Send a frame to the browser every N frames               |
| `LEMA`              | `0.25`          | EMA smoothing factor for traffic-light bounding box      |
| `LWIN`              | `6`             | Rolling window size for majority-vote light colour       |
| `OCR_TIMEOUT`       | `8.0 s`         | Force "UNREAD" if OCR thread hangs longer than this      |
| Panel duration      | `90 frames`     | How long the violation info panel stays on screen        |
| ViolationClipWriter | `±3 s`          | Pre/post buffer around each violation event              |

---

## ⚠️ Known Limitations

These are **documented bugs and design gaps** — not undocumented surprises. Future contributors should be aware of all of them.

### 1. Stop line is drawn manually

The stop line must be drawn by the user on the first frame before analysis begins. The system has **no automatic stop-line detection**. In real-world deployments, a properly-trained semantic segmentation or lane-detection model should locate the stop line automatically — similar to how ADAS/self-driving systems identify road markings.

### 2. Duplicate snapshots for stationary vehicles

A violation is logged the first time a tracked vehicle crosses the stop line (`mem["logged"] = True` prevents re-logging for the same car ID). However, if YOLO loses track of the vehicle and reassigns it a new ID, the system will treat it as a new vehicle and log another violation. This is a tracker drift issue — the `logged` flag is ID-bound, not position-bound.

### 3. OCR runs live during processing (not post-processing)

Plate reading happens in a background thread the moment a violation is detected, using a single frame crop. This means OCR sometimes receives a blurry, occluded, or motion-blurred plate crop, producing "UNREAD" results. A more accurate approach would be to **collect all violation snapshots first, then run OCR in bulk at the end of the video** — choosing the sharpest crop across multiple frames for each vehicle ID.

### 4. Traffic light colour relies on position heuristics

`get_light_color()` discards any traffic light whose top edge is below 45% of the frame height (`y1 > frame_h * 0.45`). This heuristic works for overhead lights but fails for side-mounted, near-camera, or unusual setups.

### 5. No persistent session management

Sessions are stored as timestamped folders. There is no cleanup mechanism — disk usage grows unbounded over time. Add a retention policy (e.g. keep last N sessions, or delete sessions older than X days).

---

## 🔮 Suggested Improvements for Future Contributors

These are direct improvements the original team identified. They are listed in approximate order of impact.

### High priority

**A. Automatic stop-line detection**
Replace manual drawing with a computer-vision pipeline that detects the stop line from the video. Approaches include:

- Hough line transform on the first frame (fast, works for straight markings)
- A fine-tuned segmentation model (e.g. YOLOv8-seg trained on road markings)
- Perspective-transform + lane-detection as used in self-driving datasets (BDD100K, CULane)

**B. Post-processing OCR for higher plate accuracy**
Instead of reading plates live during analysis, collect all violation snapshots into `sessions/<ts>/snapshots/`. After the video finishes processing, iterate over all snapshots, run EasyOCR on every frame where that car ID appears, and pick the result with the highest confidence score. This decouples OCR quality from processing speed and allows re-reading without reprocessing the video.

**C. Deduplicate violations by spatial proximity, not just tracker ID**
Add a check: if a new violation bounding box overlaps significantly (IoU > 0.5) with an already-logged violation at a similar timestamp, skip it. This handles tracker drift without relying solely on YOLO's ID persistence.

### Medium priority

**D. Automatic traffic-light localisation**
Train or fine-tune a dedicated traffic-light detector (YOLO or SSD) that outputs not just the box but also the signal state. Remove the HSV colour heuristic in favour of a classifier head trained on labelled traffic-light images from your target geography.

**E. Session management and cleanup**
Add a background task or startup hook that deletes sessions older than a configurable threshold (e.g. 7 days). Expose a `/sessions` API endpoint to list, download, and delete past sessions from the UI.

**F. GPU memory management**
Currently `get_models()` loads all three models (YOLO vehicle, YOLO plate, EasyOCR) into memory on first request and never unloads them. Add model unloading or quantisation (INT8 via TensorRT) if running on a GPU with limited VRAM.

### Lower priority / Nice-to-have

**G. Replace EasyOCR with a faster OCR engine**
EasyOCR is accurate but slow. For production throughput, consider [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) (faster GPU inference) or a region-specific ALPR model such as [OpenALPR](https://github.com/openalpr/openalpr) or [fast-alpr](https://github.com/Muhammad-Zaka/fast-alpr).

**H. Front-end session history panel**
Add a sidebar that lists all previous sessions so users can review past analyses without needing filesystem access.

**I. Docker packaging**
Wrap the app in a `Dockerfile` with CUDA base image, model pre-download step, and a `docker-compose.yml` for easy deployment.

---

## 🐛 Troubleshooting

| Symptom                          | Likely cause                     | Fix                                                                               |
| -------------------------------- | -------------------------------- | --------------------------------------------------------------------------------- |
| "No GPU detected"                | CUDA not available               | System works on CPU — just slower. Install CUDA drivers and matching torch wheel. |
| "Model not found"                | `plate_detector.pt` missing      | Place the model file in the project root (see Installation §4).                   |
| "WebSocket connection failed"    | Port conflict or page cache      | Refresh the page; confirm port 8000 is free.                                      |
| "Video upload fails"             | File too large or wrong format   | Keep videos under 100 MB; use MP4/H.264.                                          |
| All plates show "UNREAD"         | Poor crop quality or small plate | Lower detection confidence; use higher-resolution input video.                    |
| Violations logged multiple times | Tracker ID reassignment          | Known limitation — see Limitations §2.                                            |
| High CPU/slow processing         | No GPU / low `YOLO_SKIP`         | Increase `YOLO_SKIP` to 5–6; use GPU if available.                                |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/auto-stopline`
3. Commit with a clear message: `git commit -m 'feat: automatic stop-line detection via Hough transform'`
4. Push and open a Pull Request against `main`

Please document any new configuration constants in the **Configuration Reference** table above, and update the **Limitations** section if your PR resolves one.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://ultralytics.com/) — YOLOv8 and ByteTrack
- [JaidedAI / EasyOCR](https://github.com/JaidedAI/EasyOCR) — license plate OCR
- [FastAPI](https://fastapi.tiangolo.com/) — async web framework
- [OpenCV](https://opencv.org/) — computer vision primitives
- [Uvicorn](https://www.uvicorn.org/) — ASGI server

---

> **⚠️ Disclaimer:** This system is for educational and research purposes. For real-world traffic enforcement, ensure full compliance with local privacy laws, data-protection regulations, and law-enforcement guidelines before deployment.
