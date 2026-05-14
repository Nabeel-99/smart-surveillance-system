import os, cv2, tempfile, asyncio, base64, threading
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


# ── PLATE OCR (accurate multi-variant, reads from numpy crop) ─────────────────
def read_plate(crop, reader):
    """
    High-accuracy plate OCR on a numpy crop.
    Uses multiple image preprocessing variants + beam search,
    same approach as the proven read_plate_from_file but without
    the disk roundtrip for speed.
    """
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if w < 20 or h < 8:
        return None

    # Scale up aggressively for small plates
    scale = max(4.0, 200 / max(w, 1))
    big   = cv2.resize(crop, None, fx=scale, fy=scale,
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

    # Top and bottom halves for two-line plates
    half_h      = big.shape[0] // 2
    top_half    = gray[:half_h]
    bottom_half = gray[half_h:]

    variants = [sharpened, enhanced, otsu, otsu_inv, adaptive, gray,
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

        # Sort top-to-bottom so multi-line plates read correctly
        results.sort(key=lambda r: r[0][0][1])

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
    Runs read_plate() in a background thread so it never stalls the main loop.
    Results land in self.results keyed by car_id.
    """
    def __init__(self, reader, max_workers=2):
        from concurrent.futures import ThreadPoolExecutor
        self._pool   = ThreadPoolExecutor(max_workers=max_workers)
        self._reader = reader
        self.results = {}  # car_id -> plate text | "UNREAD"

    def submit(self, car_id: int, crop):
        """Submit a numpy crop for background OCR."""
        self.results[car_id] = "READING..."
        self._pool.submit(self._run, car_id, crop.copy())

    def _run(self, car_id: int, crop):
        try:
            text = read_plate(crop, self._reader)
            self.results[car_id] = text if text else "UNREAD"
        except Exception:
            self.results[car_id] = "UNREAD"

    def get(self, car_id: int) -> str:
        return self.results.get(car_id, "READING...")

    def shutdown(self):
        self._pool.shutdown(wait=False)


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
            try:
                disp[py+5:py+50, px+5:px+145] = cv2.resize(plate_crop, (140,45))
            except Exception:
                pass
    # Show the plate image in top-left of panel if crop provided
    display_plate = plate_text if plate_text not in (None, "READING...") else "READING..."
    if plate_text == "UNREAD":
        display_plate = "UNREAD"
    cv2.putText(disp, f"PLATE: {display_plate}", (px+5,py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    cv2.putText(disp, label.upper(), (px+5,py+82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(disp, "VIOLATION", (px+170,py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)


# ── HIGHLIGHTS CLIP WRITER ────────────────────────────────────────────────────
class ViolationClipWriter:
    SECONDS_PRE  = 3
    SECONDS_POST = 3

    def __init__(self, path, fps, size):
        pre_frames       = max(1, round(fps * self.SECONDS_PRE))
        post_frames      = max(1, round(fps * self.SECONDS_POST))
        self.post_frames = post_frames
        self.writer      = cv2.VideoWriter(
            path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        self.pre_buf     = deque(maxlen=pre_frames)
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
        self.post_remain = self.post_frames

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
        car_memory    = {}
        logged_plates = set()
        last_disp     = {}   # car_id -> {box, vc, sl, plate, pbox_rel, last_seen}
        active_panels = {}
        resolved_plates = {}  # car_id -> latest resolved plate text
        frame_count   = 0
        lbs           = None
        lch           = []
        LEMA, LWIN    = 0.25, 6
        SEND          = 3
        YOLO_SKIP     = 3
        # A car must be absent this many frames before its box is removed.
        # Set to YOLO_SKIP+1 so skip frames never flicker, but gone cars clear fast.
        STALE_FRAMES    = YOLO_SKIP + 1
        last_good_frame = None  # holds last valid frame for live drop recovery

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
                    # Network drop - reuse last good frame so display does not flicker
                    if last_good_frame is not None:
                        frame = last_good_frame.copy()
                    else:
                        await asyncio.sleep(0.03)
                        continue
                else:
                    break
            elif is_live:
                last_good_frame = frame.copy()

            frame_count += 1
            disp = frame.copy()

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

            # ── Poll OCR results every frame (fast, no blocking) ──────────
            OCR_TIMEOUT = 8.0  # seconds — force UNREAD if thread hangs
            now_ts = datetime.now().timestamp()
            for v in violations:
                cid = v["car_id"]
                if v["plate"] == "READING...":
                    result = ocr_queue.get(cid)
                    # Timed out — thread probably crashed silently on a bad crop
                    if result == "READING..." and (now_ts - v.get("submitted_at", now_ts)) > OCR_TIMEOUT:
                        result = "UNREAD"
                        ocr_queue.results[cid] = "UNREAD"
                        print(f"[OCR] timeout car={cid} — forcing UNREAD")
                    if result != "READING...":
                        v["plate"] = result
                        resolved_plates[cid] = result
                        if cid in active_panels:
                            active_panels[cid]["plate"] = result
                        if result not in ("UNREAD", "READING..."):
                            logged_plates.add(result)
                        await safe_send(ws, {
                            "type":    "plate_update",
                            "car_id":  cid,
                            "plate":   result,
                        })

            # ── Skip-frame path ───────────────────────────────────────────
            if frame_count % YOLO_SKIP != 0:
                # Only draw cars seen within STALE_FRAMES — clears ghost boxes
                for cid, info in list(last_disp.items()):
                    if frame_count - info["last_seen"] > STALE_FRAMES:
                        del last_disp[cid]
                        continue
                    bx1,by1,bx2,by2 = info["box"]
                    display_plate = resolved_plates.get(cid, info.get("plate"))
                    cv2.rectangle(disp, (bx1,by1), (bx2,by2), info["vc"], 2)
                    cv2.putText(disp, info["sl"], (bx1,by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["vc"], 2)
                    if display_plate and info.get("pbox_rel"):
                        r1,r2,r3,r4 = info["pbox_rel"]
                        cv2.rectangle(disp, (bx1+r1,by1+r2), (bx1+r3,by1+r4), (0,255,255), 2)
                        cv2.putText(disp, display_plate, (bx1+r1,by1+r2-8),
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
                if clip_writer: clip_writer.push(disp)
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
                # FIX: still draw last known boxes so no flicker on frames with no detections
                for cid, info in last_disp.items():
                    bx1,by1,bx2,by2 = info["box"]
                    cv2.rectangle(disp, (bx1,by1), (bx2,by2), info["vc"], 2)
                    cv2.putText(disp, info["sl"], (bx1,by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["vc"], 2)
                if vtl_enabled and vtl_color:
                    draw_vtl_on_frame(disp, vtl_color, vtl_scale)
                if clip_writer: clip_writer.push(disp)
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
            # Track which IDs YOLO saw this frame
            current_frame_ids = set()

            for box, cls, tid in zip(results.boxes.xyxy,
                                     results.boxes.cls,
                                     results.boxes.id):
                label  = model.names[int(cls)]
                if label not in ["car","truck","bus","motorcycle"] or tid is None:
                    continue
                car_id = int(tid)
                x1, y1, x2, y2 = map(int, box)
                current_frame_ids.add(car_id)

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
                    if len(pr.boxes.xyxy) > 0:
                        px1,py1,px2,py2 = map(int, pr.boxes.xyxy[0])
                        pcr = cc[py1:py2, px1:px2]
                        mem["pbox_rel"] = (px1, py1, px2, py2)

                bp = Counter(mem["plates"]).most_common(1)[0][0] \
                     if mem["plates"] else None
                # Use resolved plate if we have one
                display_plate = resolved_plates.get(car_id, bp)

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
                if display_plate and mem["pbox_rel"]:
                    r1,r2,r3,r4 = mem["pbox_rel"]
                    cv2.rectangle(disp, (x1+r1,y1+r2), (x1+r3,y1+r4), (0,255,255), 2)
                    cv2.putText(disp, display_plate, (x1+r1,y1+r2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

                last_disp[car_id] = {
                    "box":       (x1,y1,x2,y2),
                    "vc":        vc,
                    "sl":        sl,
                    "plate":     display_plate,
                    "pbox_rel":  mem["pbox_rel"],
                    "last_seen": frame_count,
                }

                # ── Log violation ─────────────────────────────────────────
                if mem["is_violator"] and not mem["logged"]:
                    car_crop = frame[y1:y2, x1:x2]

                    snap_name = f"car{car_id}_f{frame_count}.jpg"
                    snap_path = os.path.join(snap_dir, snap_name)
                    cv2.imwrite(snap_path, car_crop)

                    if pcr is not None and pcr.size > 0:
                        ocr_queue.submit(car_id, pcr)
                    else:
                        ocr_queue.results[car_id] = "UNREAD"
                        resolved_plates[car_id] = "UNREAD"
                        await safe_send(ws, {
                            "type":   "plate_update",
                            "car_id": car_id,
                            "plate":  "UNREAD",
                        })

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

            # Draw boxes for cars YOLO missed this frame but seen recently
            for cid, info in list(last_disp.items()):
                if cid in current_frame_ids:
                    continue
                if frame_count - info["last_seen"] > STALE_FRAMES:
                    del last_disp[cid]
                    continue
                bx1,by1,bx2,by2 = info["box"]
                dp = resolved_plates.get(cid, info.get("plate"))
                cv2.rectangle(disp, (bx1,by1), (bx2,by2), info["vc"], 2)
                cv2.putText(disp, info["sl"], (bx1,by1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, info["vc"], 2)
                if dp and info.get("pbox_rel"):
                    r1,r2,r3,r4 = info["pbox_rel"]
                    cv2.rectangle(disp, (bx1+r1,by1+r2), (bx1+r3,by1+r4), (0,255,255), 2)
                    cv2.putText(disp, dp, (bx1+r1,by1+r2-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0,255,255), 2)

            for cid, panel in list(active_panels.items()):
                if frame_count <= panel["until_frame"]:
                    draw_info_panel(disp, panel["plate"], panel["label"],
                                    panel["plate_crop"], w)
                else:
                    del active_panels[cid]

            if vtl_enabled and vtl_color:
                draw_vtl_on_frame(disp, vtl_color, vtl_scale)
            if clip_writer: clip_writer.push(disp)

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
    return JSONResponse({"error": "No video available"}, status_code=404)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)