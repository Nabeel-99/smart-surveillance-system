import os, cv2, tempfile, asyncio, base64, subprocess, threading
from datetime import datetime
from collections import Counter, deque
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
import uvicorn
import numpy as np
from ultralytics import YOLO
import easyocr

app = FastAPI()
os.makedirs("sessions", exist_ok=True)
os.makedirs("static", exist_ok=True)

_model = _plate_model = _reader = None


def get_models():
    global _model, _plate_model, _reader
    if _model is None:
        _model       = YOLO("yolov8s.pt")
        _plate_model = YOLO("plate_detector.pt")
        _reader      = easyocr.Reader(['en'])
    return _model, _plate_model, _reader


def make_session_dir():
    ts      = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    session = os.path.join("sessions", ts)
    os.makedirs(os.path.join(session, "snapshots"), exist_ok=True)
    os.makedirs(os.path.join(session, "clips"),     exist_ok=True)
    return session


# ── TRAFFIC LIGHT COLOR DETECTION ─────────────────────────────────────────────
def get_light_color(frame, box, frame_h):
    x1, y1, x2, y2 = map(int, box)
    if y1 > frame_h * 0.45:
        return "UNKNOWN"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN"

    h     = crop.shape[0]
    third = max(1, h // 3)

    def dominant(c):
        if c.size == 0:
            return "UNKNOWN", 0
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        r   = cv2.inRange(hsv, (0,70,50),   (10,255,255)) + \
              cv2.inRange(hsv, (160,70,50),  (180,255,255))
        g   = cv2.inRange(hsv, (40,40,40),  (95,255,255))
        a   = cv2.inRange(hsv, (15,40,40),  (35,255,255))
        scores = {"RED": int(r.sum()), "GREEN": int(g.sum()), "AMBER": int(a.sum())}
        best   = max(scores, key=scores.get)
        return (best, scores[best]) if scores[best] > 300 else ("UNKNOWN", 0)

    tc, ts = dominant(crop[:third])
    mc, ms = dominant(crop[third:2*third])
    bc, bs = dominant(crop[2*third:])

    if tc == "RED"   and ts > bs:  return "RED"
    if bc == "GREEN" and bs > ts:  return "GREEN"
    if mc == "AMBER" and ms > 300: return "AMBER"
    return "UNKNOWN"


# ── VIRTUAL TRAFFIC LIGHT OVERLAY ─────────────────────────────────────────────
def draw_vtl_on_frame(frame, color, scale=1.0):
    if not color:
        return

    x, y    = 16, 16
    box_w   = int(52 * scale)
    box_h   = int(116 * scale)
    bulb_r  = int(13 * scale)
    gap     = int(32 * scale)
    housing = (20, 28, 36)
    border  = (58, 80, 96)
    cyan    = (255, 212, 0)

    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), housing, -1)
    cv2.rectangle(frame, (x, y), (x + box_w, y + box_h), border, 2)
    cv2.line(frame, (x+2, y+2), (x + box_w - 2, y+2), (10,15,20), 1)

    bulbs = [
        ("RED",   (0, 0, 255),   (0, 0, 60)),
        ("AMBER", (0, 165, 255), (0, 35, 70)),
        ("GREEN", (0, 200, 0),   (0, 50, 10)),
    ]

    for i, (name, on_bgr, off_bgr) in enumerate(bulbs):
        cx    = x + box_w // 2
        cy    = y + int(24 * scale) + i * gap
        is_on = (color == name)
        fill  = on_bgr if is_on else off_bgr
        cv2.circle(frame, (cx, cy), bulb_r, fill, -1)
        cv2.circle(frame, (cx, cy), bulb_r, border, 1)
        if is_on:
            overlay = frame.copy()
            cv2.circle(overlay, (cx, cy), bulb_r + 5, on_bgr, 2)
            cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    font_scale = max(0.3, 0.38 * scale)
    cv2.putText(frame, "VTL",
                (x + int(box_w * 0.18), y + box_h - int(6 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, cyan, 1, cv2.LINE_AA)

    color_bgr = {"RED":(0,0,255),"AMBER":(0,165,255),"GREEN":(0,200,0)}.get(color, border)
    cv2.putText(frame, color,
                (x + box_w + int(6 * scale), y + int(28 * scale)),
                cv2.FONT_HERSHEY_SIMPLEX, max(0.4, 0.5 * scale),
                color_bgr, 1, cv2.LINE_AA)


# ── STOP LINE CROSSING ────────────────────────────────────────────────────────
def crosses_line(p1, p2, x1, y1, x2, y2):
    def sign(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    mx    = (x1 + x2) // 2
    s_bl  = sign(p1, p2, (x1, y2))
    s_br  = sign(p1, p2, (x2, y2))
    s_tl  = sign(p1, p2, (x1, y1))
    s_tr  = sign(p1, p2, (x2, y1))
    s_top = sign(p1, p2, (mx, y1))
    s_bot = sign(p1, p2, (mx, y2))

    if (s_bl > 0)  != (s_br > 0):  return True
    if (s_top > 0) != (s_bot > 0): return True
    if (s_tl > 0)  != (s_bl > 0):  return True
    if (s_tr > 0)  != (s_br > 0):  return True
    return False


# ── PLATE OCR ─────────────────────────────────────────────────────────────────
# ── PLATE OCR (reads from file path, not from crop directly) ──────────────────
def read_plate_from_file(plate_path: str, reader) -> str:
    """
    Read plate text from a saved image file.
    Returns plate string or None.
    """
    img = cv2.imread(plate_path)
    if img is None or img.size == 0:
        return None

    h, w = img.shape[:2]
    if w < 20 or h < 8:
        return None

    scale = max(4.0, 200 / max(w, 1))
    big   = cv2.resize(img, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)

    # Gamma brighten if dark (nighttime)
    mean_val = float(np.mean(gray))
    if mean_val < 100:
        gamma = 0.45
        lut   = np.array([min(255, int(((i / 255.0) ** gamma) * 255))
                          for i in range(256)], dtype=np.uint8)
        gray  = cv2.LUT(gray, lut)

    clahe     = cv2.createCLAHE(clipLimit=3.5, tileGridSize=(4, 4))
    enhanced  = clahe.apply(gray)
    kernel    = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
    sharpened = cv2.filter2D(enhanced, -1, kernel)

    _, otsu     = cv2.threshold(enhanced, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv    = cv2.bitwise_not(otsu)
    adaptive    = cv2.adaptiveThreshold(enhanced, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY, 15, 4)

    # Extra: try reading each horizontal half separately (catches two-line plates)
    half_h = big.shape[0] // 2
    top_half    = cv2.cvtColor(big[:half_h], cv2.COLOR_BGR2GRAY) if len(big.shape)==3 \
                  else big[:half_h]
    bottom_half = cv2.cvtColor(big[half_h:], cv2.COLOR_BGR2GRAY) if len(big.shape)==3 \
                  else big[half_h:]

    variants = [sharpened, enhanced, otsu, otsu_inv, adaptive, big, gray,
                top_half, bottom_half]

    best_text = ""
    best_conf = 0.0

    for img_v in variants:
        try:
            results = reader.readtext(
                img_v,
                allowlist='ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
                decoder='beamsearch',
                beamWidth=10,
                detail=1,
                paragraph=False,
            )
        except Exception:
            continue

        if not results:
            continue

        results.sort(key=lambda r: r[0][0][1])  # top-to-bottom

        combined   = ""
        total_conf = 0.0
        for bbox, text, conf in results:
            clean = ''.join(c for c in text.upper() if c.isalnum())
            if len(clean) >= 2:
                combined   += clean
                total_conf += conf

        avg_conf = total_conf / max(len(results), 1)

        if len(combined) >= 3 and avg_conf > best_conf:
            best_conf = avg_conf
            best_text = combined

        if best_conf >= 0.75 and len(best_text) >= 4:
            break

    return best_text if len(best_text) >= 3 else None


# ── BACKGROUND OCR DISPATCHER ─────────────────────────────────────────────────
class PlateOCRQueue:
    """
    Background thread pool for plate OCR.
    When a plate image is saved, call submit().
    Results land in self.results dict keyed by car_id.
    """
    def __init__(self, reader, max_workers=2):
        from concurrent.futures import ThreadPoolExecutor
        self._pool    = ThreadPoolExecutor(max_workers=max_workers)
        self._reader  = reader
        self.results  = {}   # car_id -> plate_text or "UNREAD"

    def submit(self, car_id: int, plate_path: str):
        """Kick off background OCR for this car. Result goes into self.results."""
        self.results[car_id] = "READING..."
        self._pool.submit(self._run, car_id, plate_path)

    def _run(self, car_id: int, plate_path: str):
        try:
            text = read_plate_from_file(plate_path, self._reader)
            self.results[car_id] = text if text else "UNREAD"
        except Exception:
            self.results[car_id] = "UNREAD"

    def get(self, car_id: int) -> str:
        return self.results.get(car_id, "READING...")

    def shutdown(self):
        self._pool.shutdown(wait=False)
        
def reencode_to_h264(src_path):
    """Re-encode clip to H.264 so browsers can play it inline."""
    tmp = src_path + ".h264tmp.mp4"
    try:
        r = subprocess.run(
            ['ffmpeg', '-y', '-i', src_path,
             '-vcodec', 'libx264', '-crf', '23', '-preset', 'fast',
             '-pix_fmt', 'yuv420p', tmp],
            capture_output=True, timeout=120
        )
        if r.returncode == 0 and os.path.exists(tmp):
            os.replace(tmp, src_path)
    except FileNotFoundError:
        pass   # ffmpeg not installed — mp4v stays, may not play in browser
    except Exception:
        try:
            if os.path.exists(tmp): os.remove(tmp)
        except Exception:
            pass
        
# ── VIOLATION INFO PANEL (on-frame overlay) ───────────────────────────────────
def draw_info_panel(disp, plate_text, label, plate_crop, frame_w):
    pw, ph = 320, 100
    px, py = frame_w - pw - 10, 10
    ov = disp.copy()
    cv2.rectangle(ov, (px,py), (px+pw,py+ph), (20,20,20), -1)
    cv2.addWeighted(ov, 0.7, disp, 0.3, 0, disp)
    cv2.rectangle(disp, (px,py), (px+pw,py+ph), (0,0,255), 2)
    if plate_crop is not None and plate_crop.size > 0:
        ch, cw = plate_crop.shape[:2]
        if cw > 0 and ch > 0:
            disp[py+5:py+50, px+5:px+145] = cv2.resize(plate_crop, (140,45))
    cv2.putText(disp, f"PLATE: {plate_text}", (px+5,py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    cv2.putText(disp, label.upper(), (px+5,py+82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(disp, "VIOLATION", (px+170,py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)


# ── ROLLING RAW-FRAME BUFFER (for per-violation clips) ───────────────────────
class ViolationClipBuffer:
    """
    Buffers annotated (disp) frames.
    trigger() flushes the pre-buffer to a new per-violation clip,
    then push() keeps writing annotated frames until POST_FRAMES are done.
    Re-encodes to H.264 in background when done.
    """
    PRE_FRAMES  = 90
    POST_FRAMES = 90

    def __init__(self):
        self.buf     = deque(maxlen=self.PRE_FRAMES)
        self._active = {}   # car_id -> {writer, remaining, path}

    def push(self, frame):
        self.buf.append(frame.copy())
        for cid in list(self._active.keys()):
            rec = self._active[cid]
            rec['writer'].write(frame)
            rec['remaining'] -= 1
            if rec['remaining'] <= 0:
                rec['writer'].release()
                path = rec['path']
                del self._active[cid]
                threading.Thread(target=reencode_to_h264,
                                 args=(path,), daemon=True).start()

    def trigger(self, car_id, path, fps, size):
        """Start a new per-violation clip using buffered + upcoming annotated frames."""
        w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for f in self.buf:
            w.write(f)
        self._active[car_id] = {'writer': w,
                                 'remaining': self.POST_FRAMES,
                                 'path': path}

    def release_all(self):
        for rec in list(self._active.values()):
            try:
                rec['writer'].release()
                threading.Thread(target=reencode_to_h264,
                                 args=(rec['path'],), daemon=True).start()
            except Exception:
                pass
        self._active.clear()
# ── HIGHLIGHTS CLIP WRITER ────────────────────────────────────────────────────
class ViolationClipWriter:
    PRE_FRAMES  = 90
    POST_FRAMES = 90

    def __init__(self, path, fps, size):
        self.writer      = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        self.pre_buf     = deque(maxlen=self.PRE_FRAMES)
        self.post_remain = 0
        self.active      = False

    def push(self, frame):
        if self.active:
            self.writer.write(frame)
            self.post_remain -= 1
            if self.post_remain <= 0:
                self.active = False
        else:
            self.pre_buf.append(frame.copy())

    def trigger(self):
        if not self.active:
            for f in self.pre_buf:
                self.writer.write(f)
            self.pre_buf.clear()
            self.active = True
        self.post_remain = self.POST_FRAMES

    def release(self):
        self.writer.release()


# ── FRAME ENCODING ────────────────────────────────────────────────────────────
def f2b(frame, q=60):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()

async def safe_send(ws, payload):
    try: await ws.send_json(payload)
    except Exception: pass


# ── ROUTES ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f: return f.read()


@app.get("/violation-clip/{session}/{filename}")
async def violation_clip(session: str, filename: str):
    path = os.path.join("sessions", session, "clips", filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="video/mp4")


@app.get("/violation-snap/{session}/{filename}")
async def violation_snap(session: str, filename: str):
    path = os.path.join("sessions", session, "snapshots", filename)
    if not os.path.exists(path):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="image/jpeg")


# ── LIVE PREVIEW ──────────────────────────────────────────────────────────────
@app.websocket("/ws/live-preview")
async def live_preview(ws: WebSocket):
    await ws.accept()
    cap = None
    try:
        data = await ws.receive_json()
        ip   = data.get("ip","").strip()
        url  = f"http://{ip}:4747/video"
        print(f"[PREVIEW] connecting {url}")

        cap = cv2.VideoCapture(url)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        await asyncio.sleep(0.5)

        ret, frame = cap.read()
        if not ret or frame is None:
            await safe_send(ws, {"type":"error","msg":f"Cannot open {url}"})
            return

        h, w = frame.shape[:2]
        print(f"[PREVIEW] OK {w}x{h}")

        app.state.live_cap     = cap
        app.state.video_path   = url
        app.state.frame_shape  = (h, w)
        app.state.fps          = 30
        app.state.total_frames = float('inf')
        app.state.is_live      = True

        await safe_send(ws, {"type":"init","width":w,"height":h,"frame":f2b(frame)})

        frame_n = 0
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                await asyncio.sleep(0.05)
                continue
            frame_n += 1
            if frame_n % 2 == 0:
                await safe_send(ws, {"type":"frame","frame":f2b(frame, 45)})
            await asyncio.sleep(0.05)

    except WebSocketDisconnect:
        pass
    finally:
        if not getattr(app.state, 'analyzing', False):
            if cap: cap.release()
            app.state.live_cap = None
        print("[PREVIEW] WS closed")


# ── FIRST FRAME ───────────────────────────────────────────────────────────────
@app.post("/first-frame")
async def first_frame(video: UploadFile = File(...)):
    data  = await video.read()
    tmp   = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(data); tmp.flush()
    cap   = cv2.VideoCapture(tmp.name)
    ret, frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    h, w  = frame.shape[:2]
    cap.release()
    app.state.video_path   = tmp.name
    app.state.frame_shape  = (h, w)
    app.state.total_frames = total
    app.state.fps          = fps
    app.state.is_live      = False
    app.state.live_cap     = None
    return {"frame":f2b(frame),"width":w,"height":h,"total_frames":total}


# ── MAIN ANALYSIS WEBSOCKET ───────────────────────────────────────────────────
@app.websocket("/ws/analyze")
async def analyze(ws: WebSocket):
    await ws.accept()
    own_cap     = None
    clip_writer = None
    violations  = []
    app.state.analyzing = True

    session_dir = make_session_dir()
    snap_dir    = os.path.join(session_dir, "snapshots")
    clips_dir   = os.path.join(session_dir, "clips")
    session_ts  = os.path.basename(session_dir)
    app.state.last_session = session_dir
    print(f"[ANALYZE] session: {session_dir}")

    try:
        data = await ws.receive_json()
        lp1  = tuple(data["line_p1"])
        lp2  = tuple(data["line_p2"])

        vtl_enabled = data.get("vtl_enabled", False)
        vtl_color   = data.get("vtl_color", None)

        is_live      = getattr(app.state, 'is_live', False)
        h, w         = app.state.frame_shape
        total_frames = app.state.total_frames
        fps          = app.state.fps

        print(f"[ANALYZE] mode={'LIVE' if is_live else 'FILE'} | "
              f"vtl_enabled={vtl_enabled} vtl_color={vtl_color}")

        model, plate_model, reader = get_models()
        ocr_queue = PlateOCRQueue(reader, max_workers=2)

        if is_live:
            cap = getattr(app.state, 'live_cap', None)
            if cap is None:
                await safe_send(ws, {"type":"error",
                                     "msg":"Camera not connected — click Connect first"})
                return
        else:
            own_cap     = cv2.VideoCapture(app.state.video_path)
            cap         = own_cap
            out_video   = os.path.join(session_dir, "highlights.mp4")
            clip_writer = ViolationClipWriter(out_video, fps, (w, h))

        vtl_scale = max(0.7, w / 900)
        vcb       = ViolationClipBuffer()

        car_memory    = {}
        logged_plates = set()
        last_disp     = {}
        active_panels = {}
        frame_count   = 0
        lbs           = None
        lch           = []
        LEMA, LWIN    = 0.25, 6
        SEND          = 3
        YOLO_SKIP     = 3

        while True:
            try:
                msg_in = await asyncio.wait_for(ws.receive_json(), timeout=0.001)
                if isinstance(msg_in, dict) and msg_in.get("type") == "vtl_update":
                    vtl_enabled = msg_in.get("enabled", False)
                    vtl_color   = msg_in.get("color", None)
                    print(f"[ANALYZE] VTL → enabled={vtl_enabled} color={vtl_color}")
            except (asyncio.TimeoutError, Exception):
                pass

            ret, frame = cap.read()
            if not ret or frame is None:
                if is_live:
                    await asyncio.sleep(0.05)
                    continue
                else:
                    break

            frame_count += 1
            disp = frame.copy()

            # Rolling buffer for per-violation clips
            # vcb.push(disp)

            # Stop line
            cv2.line(disp, lp1, lp2, (0,0,255), 2)
            mx = (lp1[0]+lp2[0])//2 - 60
            my = (lp1[1]+lp2[1])//2 - 10
            cv2.putText(disp, "STOP LINE", (mx,my),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            if vtl_enabled and vtl_color:
                effective_light = vtl_color
            else:
                effective_light = Counter(lch).most_common(1)[0][0] if lch else "UNKNOWN"

            # ── Skip-frame path ───────────────────────────────────────────
            if frame_count % YOLO_SKIP != 0:
                for cid, info in last_disp.items():
                    bx1,by1,bx2,by2 = info["box"]
                    cv2.rectangle(disp, (bx1,by1), (bx2,by2), info["vc"], 2)
                    cv2.putText(disp, info["sl"], (bx1,by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["vc"], 2)
                    if info.get("plate") and info.get("pbox_rel"):
                        r1,r2,r3,r4 = info["pbox_rel"]
                        cb1,cb2,_,_ = info["box"]
                        cv2.rectangle(disp, (cb1+r1,cb2+r2), (cb1+r3,cb2+r4), (0,255,255), 2)
                        cv2.putText(disp, info["plate"], (cb1+r1,cb2+r2-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)
                    if lbs and not (vtl_enabled and vtl_color):
                        lx1,ly1,lx2,ly2 = [int(v) for v in lbs]
                        cur = Counter(lch).most_common(1)[0][0] if lch else "UNKNOWN"
                        lc  = {"RED":(0,0,255),"AMBER":(0,165,255),
                               "GREEN":(0,200,0)}.get(cur,(128,128,128))
                        cv2.rectangle(disp, (lx1,ly1), (lx2,ly2), lc, 2)
                        cv2.putText(disp, f"Light:{cur}", (lx1,ly1-10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, lc, 2)

                for cid, panel in list(active_panels.items()):
                    if frame_count <= panel["until_frame"]:
                        draw_info_panel(disp, panel["plate"], panel["label"],
                                        panel["plate_crop"], w)
                    else:
                        del active_panels[cid]

                if vtl_enabled and vtl_color:
                    draw_vtl_on_frame(disp, vtl_color, vtl_scale)
                vcb.push(disp)
                if clip_writer: clip_writer.push(disp)
                for v in violations:
                    cid = v["car_id"]
                    if v["plate"] == "READING...":
                        result = ocr_queue.get(cid)
                        if result != "READING...":
                            v["plate"] = result
                            if result not in ("UNREAD", "READING..."):
                                logged_plates.add(result)
                            await safe_send(ws, {
                                "type":    "plate_update",
                                "car_id":  cid,
                                "plate":   result,
                            })
                if frame_count % SEND == 0:
                    pct = 0 if is_live else round(frame_count/total_frames*100, 1)
                    await safe_send(ws, {
                        "type":        "frame",
                        "frame":       f2b(disp, 55),
                        "progress":    pct,
                        "frame_count": frame_count,
                        "total":       total_frames if not is_live else "LIVE",
                        "light_color": effective_light,
                    })
                    await asyncio.sleep(0)
                continue

            # ── YOLO inference frame ──────────────────────────────────────
            results = model.track(frame, persist=True, conf=0.35, verbose=False)[0]
            if results.boxes.id is None:
                if vtl_enabled and vtl_color:
                    draw_vtl_on_frame(disp, vtl_color, vtl_scale)
                if clip_writer: clip_writer.push(disp)
                continue

            nl, nc, bc = None, "UNKNOWN", 0
            if not (vtl_enabled and vtl_color):
                for box, cls, conf in zip(results.boxes.xyxy,
                                          results.boxes.cls,
                                          results.boxes.conf):
                    if model.names[int(cls)] == "traffic light" and float(conf) > bc:
                        c = get_light_color(frame, box, h)
                        if c != "UNKNOWN":
                            bc = float(conf); nl = box; nc = c

                if nc != "UNKNOWN": lch.append(nc)
                if len(lch) > LWIN: lch.pop(0)

                if nl is not None:
                    raw = [float(int(x)) for x in nl]
                    lbs = raw if lbs is None else \
                          [LEMA*r+(1-LEMA)*s for r,s in zip(raw,lbs)]

            if vtl_enabled and vtl_color:
                effective_light = vtl_color
            else:
                effective_light = Counter(lch).most_common(1)[0][0] if lch else "UNKNOWN"
                if lbs:
                    lx1,ly1,lx2,ly2 = [int(v) for v in lbs]
                    lc_bgr = {"RED":(0,0,255),"AMBER":(0,165,255),
                              "GREEN":(0,200,0)}.get(effective_light,(128,128,128))
                    cv2.rectangle(disp, (lx1,ly1), (lx2,ly2), lc_bgr, 2)
                    cv2.putText(disp, f"Light:{effective_light}", (lx1,ly1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, lc_bgr, 2)

            # ── Vehicle loop ──────────────────────────────────────────────
            last_disp = {}
            for box, cls, tid in zip(results.boxes.xyxy,
                                     results.boxes.cls,
                                     results.boxes.id):
                label  = model.names[int(cls)]
                if label not in ["car","truck","bus","motorcycle"] or tid is None:
                    continue
                car_id = int(tid)
                x1, y1, x2, y2 = map(int, box)

                if car_id not in car_memory:
                    car_memory[car_id] = {
                        "label":       label,
                        "plates":      [],
                        "pbox_rel":    None,
                        "logged":      False,
                        "is_violator": False,
                    }
                mem = car_memory[car_id]

                if effective_light == "RED" and crosses_line(lp1, lp2, x1, y1, x2, y2):
                    mem["is_violator"] = True

                pcr = None
                if mem["is_violator"] and not mem["logged"]:
                    cc = frame[y1:y2, x1:x2]
                    pr = plate_model.predict(cc, conf=0.25, verbose=False)[0]
                    # Save plate crop to disk — OCR runs from file, not live crop
                    if len(pr.boxes.xyxy) > 0:
                        px1,py1,px2,py2 = map(int, pr.boxes.xyxy[0])
                        pcr = cc[py1:py2, px1:px2]
                        mem["pbox_rel"] = (px1, py1, px2, py2)

                bp = Counter(mem["plates"]).most_common(1)[0][0] \
                     if mem["plates"] else None

                if mem["is_violator"]:
                    vc, sl = (0,0,255), "VIOLATION"
                elif effective_light == "AMBER":
                    vc, sl = (0,165,255), "CAUTION"
                elif effective_light == "GREEN":
                    vc, sl = (0,200,0), "OK"
                else:
                    vc, sl = (128,128,128), label.upper()

                cv2.rectangle(disp, (x1,y1), (x2,y2), vc, 2)
                cv2.putText(disp, sl, (x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, vc, 2)
                if bp and mem["pbox_rel"]:
                    r1,r2,r3,r4 = mem["pbox_rel"]
                    cv2.rectangle(disp, (x1+r1,y1+r2), (x1+r3,y1+r4), (0,255,255), 2)
                    cv2.putText(disp, bp, (x1+r1,y1+r2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

                last_disp[car_id] = {
                    "box":      (x1,y1,x2,y2),
                    "vc":       vc,
                    "sl":       sl,
                    "plate":    bp,
                    "pbox_rel": mem["pbox_rel"],
                }

                # ── Log violation ─────────────────────────────────────────
                if mem["is_violator"] and not mem["logged"]:
                    car_crop = frame[y1:y2, x1:x2]

                    snap_name = f"car{car_id}_f{frame_count}.jpg"
                    snap_path = os.path.join(snap_dir, snap_name)
                    cv2.imwrite(snap_path, car_crop)

                    plate_snap_name = None
                    if pcr is not None and pcr.size > 0:
                        plate_snap_name = f"car{car_id}_f{frame_count}_plate.jpg"
                        plate_snap_path = os.path.join(snap_dir, plate_snap_name)
                        cv2.imwrite(plate_snap_path, pcr)
                        ocr_queue.submit(car_id, plate_snap_path)
                    else:
                        ocr_queue.results[car_id] = "UNREAD"
                        # No plate crop — notify frontend immediately so it doesn't hang on READING...
                        await safe_send(ws, {
                            "type":   "plate_update",
                            "car_id": car_id,
                            "plate":  "UNREAD",
                        })

                    clip_name = f"car{car_id}_f{frame_count}.mp4"
                    clip_url  = None
                    if not is_live:
                        clip_path = os.path.join(clips_dir, clip_name)
                        vcb.trigger(car_id, clip_path, fps, (w, h))
                        clip_url = f"/violation-clip/{session_ts}/{clip_name}"

                    ts_str = datetime.now().strftime("%H:%M:%S")
                    ps     = "READING..."
                    viol   = {
                        "frame":    frame_count,
                        "time":     ts_str,
                        "car_id":   car_id,
                        "plate":    ps,
                        "label":    label,
                        "snap":     f2b(car_crop, 85),
                        "snap_url": f"/violation-snap/{session_ts}/{snap_name}",
                        "clip_url": clip_url,
                        "session":  session_ts,
                    }
                    violations.append(viol)
                    active_panels[car_id] = {
                        "plate":       ps,
                        "label":       label,
                        "plate_crop":  pcr,
                        "until_frame": frame_count + 90,
                    }
                    mem["logged"] = True
                    if clip_writer: clip_writer.trigger()
                    print(f"[ANALYZE] VIOLATION car={car_id} frame={frame_count} → OCR queued")
                    await safe_send(ws, {"type":"violation","data":viol})
                    
            for cid, panel in list(active_panels.items()):
                if frame_count <= panel["until_frame"]:
                    draw_info_panel(disp, panel["plate"], panel["label"],
                                    panel["plate_crop"], w)
                else:
                    del active_panels[cid]

            if vtl_enabled and vtl_color:
                draw_vtl_on_frame(disp, vtl_color, vtl_scale)
            vcb.push(disp)
            if clip_writer: clip_writer.push(disp)
            for v in violations:
                cid = v["car_id"]
                if v["plate"] == "READING...":
                    result = ocr_queue.get(cid)
                    if result != "READING...":
                        v["plate"] = result
                        if result not in ("UNREAD", "READING..."):
                            logged_plates.add(result)
                        await safe_send(ws, {
                            "type":    "plate_update",
                            "car_id":  cid,
                            "plate":   result,
                        })
            if frame_count % SEND == 0:
                pct = 0 if is_live else round(frame_count/total_frames*100, 1)
                await safe_send(ws, {
                    "type":        "frame",
                    "frame":       f2b(disp, 55),
                    "progress":    pct,
                    "frame_count": frame_count,
                    "total":       total_frames if not is_live else "LIVE",
                    "light_color": effective_light,
                })
                await asyncio.sleep(0)

    except WebSocketDisconnect:
        pass
    finally:
        app.state.analyzing = False
        if own_cap:     own_cap.release()
        if clip_writer: clip_writer.release()
        try: vcb.release_all()
        except Exception: pass
        ocr_queue.shutdown()

    await safe_send(ws, {"type":"done","total_violations":len(violations)})


# ── DOWNLOAD HIGHLIGHTS VIDEO ─────────────────────────────────────────────────
@app.get("/download/video")
async def download_video():
    session = getattr(app.state, "last_session", None)
    if session:
        path = os.path.join(session, "highlights.mp4")
        ts   = os.path.basename(session)
        if os.path.exists(path):
            return FileResponse(path, media_type="video/mp4",
                                filename=f"violations_{ts}.mp4")
    return FileResponse("output_final.mp4", media_type="video/mp4",
                        filename="violations_highlight.mp4")


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)