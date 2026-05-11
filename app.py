import os, cv2, tempfile, asyncio, base64, subprocess
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


# ── FFMPEG RE-ENCODE: mp4v → H.264 (browser-compatible) ──────────────────────
def remux_to_h264(src_path: str) -> str:
    """
    Re-encode a mp4v .mp4 to H.264 so browsers can play it natively.
    Returns path to the new file (src replaced in-place).
    ffmpeg must be installed (apt install ffmpeg).
    Falls back to returning src_path if ffmpeg is unavailable.
    """
    dst_path = src_path.replace(".mp4", "_h264.mp4")
    try:
        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-i", src_path,
                "-vcodec", "libx264",
                "-crf", "23",          # quality: 18=great, 28=smaller
                "-preset", "fast",
                "-pix_fmt", "yuv420p", # required for broad browser compat
                "-movflags", "+faststart",  # enables streaming / instant play
                dst_path,
            ],
            capture_output=True,
            timeout=120,
        )
        if result.returncode == 0 and os.path.exists(dst_path):
            os.remove(src_path)        # delete the mp4v original
            os.rename(dst_path, src_path)  # keep original filename
            return src_path
        else:
            # ffmpeg failed – log and serve the mp4v file anyway
            print(f"[FFMPEG] re-encode failed: {result.stderr.decode()[-400:]}")
            if os.path.exists(dst_path):
                os.remove(dst_path)
            return src_path
    except FileNotFoundError:
        print("[FFMPEG] not found – clips will be mp4v (may not play in browser)")
        return src_path
    except subprocess.TimeoutExpired:
        print("[FFMPEG] timeout during re-encode")
        return src_path


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
def read_plate(crop, reader):
    if crop is None or crop.size == 0:
        return None
    h, w = crop.shape[:2]
    if w < 20 or h < 8:
        return None

    scale = max(4.0, 200 / max(w, 1))
    big   = cv2.resize(crop, None, fx=scale, fy=scale,
                       interpolation=cv2.INTER_CUBIC)

    gray     = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    all_results = []
    for img in [enhanced, big]:
        for _, text, conf in reader.readtext(img):
            clean = ''.join(c for c in text.upper() if c.isalnum())
            if len(clean) >= 2:
                all_results.append((clean, conf))

    if not all_results:
        return None

    all_results.sort(key=lambda x: x[1], reverse=True)
    return all_results[0][0]


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
    PRE_FRAMES  = 90   # 3 s at 30 fps
    POST_FRAMES = 90   # 3 s after

    def __init__(self):
        self.buf = deque(maxlen=self.PRE_FRAMES)

    def push(self, frame):
        self.buf.append(frame.copy())

    def write_clip(self, path, fps, size, cap, post_frames=None):
        """
        Write buffered pre-frames + post_frames read live from cap.
        After writing, re-encodes to H.264 for browser compatibility.
        Caller must save/restore cap position after this call.
        """
        if post_frames is None:
            post_frames = self.POST_FRAMES

        # Write raw mp4v first
        tmp_path = path.replace(".mp4", "_raw.mp4")
        writer = cv2.VideoWriter(
            tmp_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
        for f in self.buf:
            writer.write(f)
        written = 0
        while written < post_frames:
            ret, frm = cap.read()
            if not ret or frm is None:
                break
            writer.write(frm)
            written += 1
        writer.release()

        # Re-encode to H.264 so browser <video> can play it
        os.rename(tmp_path, path)
        remux_to_h264(path)


# ── HIGHLIGHTS CLIP WRITER ────────────────────────────────────────────────────
class ViolationClipWriter:
    PRE_FRAMES  = 90
    POST_FRAMES = 90

    def __init__(self, path, fps, size):
        self.path        = path
        self.fps         = fps
        self.size        = size
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
        # Re-encode the highlights reel too
        remux_to_h264(self.path)


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
        SEND          = 5

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
            vcb.push(disp)

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
            if frame_count % 3 != 0:
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
                    for pb in pr.boxes.xyxy:
                        px1,py1,px2,py2 = map(int, pb)
                        pc = cc[py1:py2, px1:px2]
                        t  = read_plate(pc, reader)
                        if t:
                            mem["plates"].append(t)
                            mem["pbox_rel"] = (px1, py1, px2, py2)
                            pcr = pc
                    if pcr is None and len(pr.boxes.xyxy) > 0:
                        px1,py1,px2,py2 = map(int, pr.boxes.xyxy[0])
                        pcr = cc[py1:py2, px1:px2]

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
                    ps       = bp or "UNREAD"
                    car_crop = frame[y1:y2, x1:x2]

                    if ps == "UNREAD" or ps not in logged_plates:

                        # Save full car snapshot
                        snap_name = f"car{car_id}_f{frame_count}.jpg"
                        snap_path = os.path.join(snap_dir, snap_name)
                        cv2.imwrite(snap_path, car_crop)

                        # Re-run OCR on saved snapshot if still UNREAD
                        if ps == "UNREAD":
                            retry_img = cv2.imread(snap_path)
                            if retry_img is not None:
                                pr2 = plate_model.predict(
                                    retry_img, conf=0.20, verbose=False)[0]
                                for pb in pr2.boxes.xyxy:
                                    px1, py1, px2, py2 = map(int, pb)
                                    px1 = max(0, px1); py1 = max(0, py1)
                                    px2 = min(retry_img.shape[1], px2)
                                    py2 = min(retry_img.shape[0], py2)
                                    pc2 = retry_img[py1:py2, px1:px2]
                                    t   = read_plate(pc2, reader)
                                    if t:
                                        ps = t
                                        mem["plates"].append(t)
                                        mem["pbox_rel"] = (px1, py1, px2, py2)
                                        pcr = pc2
                                        break

                        # Write per-violation clip (file mode only)
                        # NOTE: write_clip advances cap by POST_FRAMES; we
                        # save & restore position so the main loop isn't skipped.
                        clip_name = f"car{car_id}_f{frame_count}.mp4"
                        clip_url  = None
                        if not is_live and own_cap is not None:
                            clip_path = os.path.join(clips_dir, clip_name)
                            saved_pos = int(own_cap.get(cv2.CAP_PROP_POS_FRAMES))
                            vcb.write_clip(clip_path, fps, (w, h), own_cap)
                            own_cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)
                            clip_url = f"/violation-clip/{session_ts}/{clip_name}"

                        ts_str = datetime.now().strftime("%H:%M:%S")
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
                        if ps != "UNREAD": logged_plates.add(ps)
                        mem["logged"] = True
                        if clip_writer: clip_writer.trigger()
                        print(f"[ANALYZE] VIOLATION plate={ps} frame={frame_count}")
                        await safe_send(ws, {"type":"violation","data":viol})

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