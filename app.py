import os, cv2, tempfile, json, asyncio, base64
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import numpy as np
from ultralytics import YOLO
import easyocr

app = FastAPI()

os.makedirs("violations_snapshots", exist_ok=True)
os.makedirs("static", exist_ok=True)

# ── model loading ──────────────────────────────────────────────────────────────
_model = None
_plate_model = None
_reader = None

def get_models():
    global _model, _plate_model, _reader
    if _model is None:
        _model       = YOLO("yolov8s.pt")
        _plate_model = YOLO("plate_detector.pt")
        _reader      = easyocr.Reader(['en'])
    return _model, _plate_model, _reader

# ── helpers ────────────────────────────────────────────────────────────────────
def get_light_color(frame, box, frame_h):
    x1,y1,x2,y2 = map(int, box)
    if y1 > frame_h * 0.45: return "UNKNOWN"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0: return "UNKNOWN"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    r = cv2.inRange(hsv,(0,70,50),(10,255,255)) + cv2.inRange(hsv,(160,70,50),(180,255,255))
    g = cv2.inRange(hsv,(40,40,40),(95,255,255))
    a = cv2.inRange(hsv,(15,40,40),(35,255,255))
    scores = {"RED":int(r.sum()),"GREEN":int(g.sum()),"AMBER":int(a.sum())}
    best = max(scores, key=scores.get)
    return best if scores[best] > 300 else "UNKNOWN"

def read_plate(crop, reader):
    if crop.size == 0: return None
    crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    for _, text, conf in reader.readtext(crop):
        clean = ''.join(c for c in text.upper() if c.isalnum())
        if len(clean) >= 4 and conf > 0.3:
            return clean
    return None

def crosses_line(p1, p2, x1, y1, x2, y2):
    def sign(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    d1 = sign(p1, p2, (x1, y2))
    d2 = sign(p1, p2, (x2, y2))
    return (d1 > 0) != (d2 > 0)

def draw_info_panel(disp, plate_text, label, plate_crop, frame_w):
    panel_w, panel_h = 320, 100
    margin = 10
    px = frame_w - panel_w - margin
    py = margin
    overlay = disp.copy()
    cv2.rectangle(overlay, (px, py), (px+panel_w, py+panel_h), (20,20,20), -1)
    cv2.addWeighted(overlay, 0.7, disp, 0.3, 0, disp)
    cv2.rectangle(disp, (px, py), (px+panel_w, py+panel_h), (0,0,255), 2)
    if plate_crop is not None and plate_crop.size > 0:
        ph, pw = plate_crop.shape[:2]
        if pw > 0 and ph > 0:
            zoom_w, zoom_h = 140, 45
            zoomed = cv2.resize(plate_crop, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
            disp[py+5:py+5+zoom_h, px+5:px+5+zoom_w] = zoomed
    cv2.putText(disp, f"PLATE: {plate_text}", (px+5, py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,255,255), 2)
    cv2.putText(disp, label.upper(), (px+5, py+82),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
    cv2.putText(disp, "VIOLATION", (px+170, py+60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0,0,255), 2)

def frame_to_b64(frame):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    return base64.b64encode(buf).decode()

def img_to_b64(img_bgr):
    _, buf = cv2.imencode('.jpg', img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return base64.b64encode(buf).decode()

# ── routes ─────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f:
        return f.read()

@app.post("/connect-camera")
async def connect_camera():
    cap = cv2.VideoCapture("http://192.168.1.195:4747/video")  # EpocCam appears as camera 0 or 1
    ret, frame = cap.read()
    if not ret:
        cap = cv2.VideoCapture(1)  # try index 1 if 0 doesn't work
        ret, frame = cap.read()
    h, w = frame.shape[:2]
    b64 = frame_to_b64(frame)
    app.state.video_path = "LIVE"
    app.state.frame_shape = (h, w)
    app.state.fps = 30
    app.state.total_frames = float('inf')
    app.state.camera_index = 0
    cap.release()
    return {"frame": b64, "width": w, "height": h, "total_frames": 999999}

@app.post("/first-frame")
async def first_frame(video: UploadFile = File(...)):
    data = await video.read()
    tmp  = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tmp.write(data); tmp.flush()
    cap = cv2.VideoCapture(tmp.name)
    ret, frame = cap.read()
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps   = cap.get(cv2.CAP_PROP_FPS)
    h, w  = frame.shape[:2]
    cap.release()
    b64 = frame_to_b64(frame)
    app.state.video_path   = tmp.name
    app.state.frame_shape  = (h, w)
    app.state.total_frames = total
    app.state.fps          = fps
    return {"frame": b64, "width": w, "height": h, "total_frames": total}

@app.websocket("/ws/analyze")
async def analyze(ws: WebSocket):
    await ws.accept()
    data = await ws.receive_json()
    lp1 = tuple(data["line_p1"])
    lp2 = tuple(data["line_p2"])

    video_path   = app.state.video_path
    h, w         = app.state.frame_shape
    total_frames = app.state.total_frames
    fps          = app.state.fps

    model, plate_model, reader = get_models()

    if video_path == "LIVE":
        cap = cv2.VideoCapture(app.state.camera_index)
    else:
        cap = cv2.VideoCapture(video_path)
    out = cv2.VideoWriter("output_final.mp4",
                          cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    car_memory    = {}
    logged_plates = set()
    last_disp     = {}
    active_panels = {}
    PANEL_DURATION = 90
    frame_count   = 0
    violations    = []

    # ── FIX 1: traffic light stabilisation ────────────────────────────────────
    light_box_smooth = None          # EMA-smoothed [x1,y1,x2,y2] as floats
    light_color_hist = []            # rolling window of recent colour calls
    LIGHT_EMA        = 0.25          # smoothing factor for box coords
    LIGHT_WIN        = 12            # window size for majority-vote colour

    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frame_count += 1

            disp = frame.copy()
            cv2.line(disp, lp1, lp2, (0, 0, 255), 2)
            mid_x = (lp1[0] + lp2[0]) // 2 - 60
            mid_y = (lp1[1] + lp2[1]) // 2 - 10
            cv2.putText(disp, "STOP LINE", (mid_x, mid_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

            # every-3-frame skip kept exactly as original for performance
            if frame_count % 3 != 0:
                for cid, info in last_disp.items():
                    bx1,by1,bx2,by2 = info["box"]
                    cv2.rectangle(disp,(bx1,by1),(bx2,by2),info["vc"],2)
                    cv2.putText(disp,info["sl"],(bx1,by1-10),
                                cv2.FONT_HERSHEY_SIMPLEX,0.7,info["vc"],2)
                    # FIX 2: plate box redrawn from CURRENT car position each frame
                    if info.get("plate") and info.get("pbox_rel"):
                        rpx1,rpy1,rpx2,rpy2 = info["pbox_rel"]
                        cbx1,cby1,_,_ = info["box"]
                        apx1 = cbx1 + rpx1
                        apy1 = cby1 + rpy1
                        apx2 = cbx1 + rpx2
                        apy2 = cby1 + rpy2
                        cv2.rectangle(disp,(apx1,apy1),(apx2,apy2),(0,255,255),2)
                        cv2.putText(disp,info["plate"],(apx1,apy1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
                    # draw stable traffic light box on skip frames too
                    if light_box_smooth is not None:
                        lx1,ly1,lx2,ly2 = [int(v) for v in light_box_smooth]
                        lc_skip = {"RED":(0,0,255),"AMBER":(0,165,255),
                                   "GREEN":(0,200,0)}.get(
                                       Counter(light_color_hist).most_common(1)[0][0]
                                       if light_color_hist else "UNKNOWN",
                                       (128,128,128))
                        cv2.rectangle(disp,(lx1,ly1),(lx2,ly2),lc_skip,2)
                        cv2.putText(disp,
                                    f"Light:{Counter(light_color_hist).most_common(1)[0][0] if light_color_hist else 'UNKNOWN'}",
                                    (lx1,ly1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,lc_skip,2)
                for cid, panel in list(active_panels.items()):
                    if frame_count <= panel["until_frame"]:
                        draw_info_panel(disp, panel["plate"],
                                        panel["label"], panel["plate_crop"], w)
                    else:
                        del active_panels[cid]
                out.write(disp)
                continue

            # ── YOLO on every 3rd frame (unchanged) ───────────────────────────
            results = model.track(frame, persist=True, conf=0.35, verbose=False)[0]
            if results.boxes.id is None:
                out.write(disp)
                continue

            # ── FIX 1: traffic light — EMA box + majority-vote colour ─────────
            new_lbox   = None
            new_lcolor = "UNKNOWN"
            best_conf  = 0

            for box, cls, conf in zip(results.boxes.xyxy,
                                      results.boxes.cls,
                                      results.boxes.conf):
                if model.names[int(cls)] == "traffic light" and float(conf) > best_conf:
                    col_c = get_light_color(frame, box, h)
                    if col_c != "UNKNOWN":
                        best_conf  = float(conf)
                        new_lbox   = box
                        new_lcolor = col_c

            # update colour history only on actual detections
            if new_lcolor != "UNKNOWN":
                light_color_hist.append(new_lcolor)
            if len(light_color_hist) > LIGHT_WIN:
                light_color_hist.pop(0)

            # stable colour via majority vote
            light_color = (Counter(light_color_hist).most_common(1)[0][0]
                           if light_color_hist else "UNKNOWN")

            # EMA smoothing on box coords — only updates when detected
            if new_lbox is not None:
                raw = [float(v) for v in [int(x) for x in new_lbox]]
                if light_box_smooth is None:
                    light_box_smooth = raw
                else:
                    light_box_smooth = [
                        LIGHT_EMA * r + (1 - LIGHT_EMA) * s
                        for r, s in zip(raw, light_box_smooth)
                    ]

            # draw traffic light box using smoothed coords
            # (persists last known position even when YOLO misses it)
            lc_bgr      = (128, 128, 128)
            lbox_coords = None
            if light_box_smooth is not None:
                lx1,ly1,lx2,ly2 = [int(v) for v in light_box_smooth]
                lc_bgr = {"RED":(0,0,255),"AMBER":(0,165,255),
                           "GREEN":(0,200,0)}.get(light_color,(128,128,128))
                lbox_coords = (lx1,ly1,lx2,ly2)
                cv2.rectangle(disp,(lx1,ly1),(lx2,ly2),lc_bgr,2)
                cv2.putText(disp,f"Light:{light_color}",(lx1,ly1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,lc_bgr,2)

            last_disp = {}

            for box, cls, tid in zip(results.boxes.xyxy,
                                     results.boxes.cls,
                                     results.boxes.id):
                label = model.names[int(cls)]
                if label not in ["car","truck","bus","motorcycle"]: continue
                if tid is None: continue

                car_id       = int(tid)
                x1,y1,x2,y2 = map(int, box)

                if car_id not in car_memory:
                    car_memory[car_id] = {
                        "label":       label,
                        "plates":      [],
                        "pbox_rel":    None,   # FIX 2: relative coords inside car crop
                        "logged":      False,
                        "is_violator": False,
                    }
                mem = car_memory[car_id]

                crossed = crosses_line(lp1, lp2, x1, y1, x2, y2)
                is_viol = light_color == "RED" and crossed
                if is_viol:
                    mem["is_violator"] = True

                plate_crop_raw = None
                if is_viol and not mem["logged"]:
                    car_crop  = frame[y1:y2, x1:x2]
                    plate_res = plate_model.predict(
                        car_crop, conf=0.25, verbose=False)[0]
                    for pbox in plate_res.boxes.xyxy:
                        px1,py1,px2,py2 = map(int, pbox)
                        pc   = car_crop[py1:py2, px1:px2]
                        text = read_plate(pc, reader)
                        if text:
                            mem["plates"].append(text)
                            # FIX 2: store RELATIVE coords inside the car crop
                            mem["pbox_rel"] = (px1, py1, px2, py2)
                            plate_crop_raw  = pc
                    if plate_crop_raw is None and len(plate_res.boxes.xyxy) > 0:
                        px1,py1,px2,py2 = map(int, plate_res.boxes.xyxy[0])
                        plate_crop_raw  = car_crop[py1:py2, px1:px2]

                best_plate = (Counter(mem["plates"]).most_common(1)[0][0]
                              if mem["plates"] else None)

                if mem["is_violator"]:
                    vc, sl = (0,0,255), "VIOLATION"
                elif light_color == "AMBER":
                    vc, sl = (0,165,255), "CAUTION"
                elif light_color == "GREEN":
                    vc, sl = (0,200,0), "OK"
                else:
                    vc, sl = (128,128,128), label.upper()

                cv2.rectangle(disp,(x1,y1),(x2,y2),vc,2)
                cv2.putText(disp,sl,(x1,y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,0.7,vc,2)

                # FIX 2: recompute absolute plate coords from CURRENT car position
                if best_plate and mem["pbox_rel"]:
                    rpx1,rpy1,rpx2,rpy2 = mem["pbox_rel"]
                    apx1 = x1 + rpx1
                    apy1 = y1 + rpy1
                    apx2 = x1 + rpx2
                    apy2 = y1 + rpy2
                    cv2.rectangle(disp,(apx1,apy1),(apx2,apy2),(0,255,255),2)
                    cv2.putText(disp,best_plate,(apx1,apy1-8),
                                cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)

                last_disp[car_id] = {
                    "box":      (x1,y1,x2,y2),
                    "vc":       vc,
                    "sl":       sl,
                    "plate":    best_plate,
                    "pbox_rel": mem["pbox_rel"],   # FIX 2: store relative, not absolute
                    "light_box": lbox_coords,
                    "lc":       lc_bgr,
                }

                if is_viol and not mem["logged"]:
                    plate_str = best_plate or "UNREAD"
                    if plate_str == "UNREAD" or plate_str not in logged_plates:
                        snap_path = f"violations_snapshots/car{car_id}_f{frame_count}.jpg"
                        cv2.imwrite(snap_path, frame[y1:y2, x1:x2])
                        ts = datetime.now().strftime("%H:%M:%S")

                        snap_b64 = img_to_b64(frame[y1:y2, x1:x2])
                        viol = {
                            "frame":  frame_count,
                            "time":   ts,
                            "car_id": car_id,
                            "plate":  plate_str,
                            "label":  label,
                            "snap":   snap_b64,
                        }
                        violations.append(viol)

                        active_panels[car_id] = {
                            "plate":       plate_str,
                            "label":       label,
                            "plate_crop":  plate_crop_raw,
                            "until_frame": frame_count + PANEL_DURATION,
                        }
                        if plate_str != "UNREAD":
                            logged_plates.add(plate_str)
                        mem["logged"] = True

                        await ws.send_json({"type": "violation", "data": viol})

            for cid, panel in list(active_panels.items()):
                if frame_count <= panel["until_frame"]:
                    draw_info_panel(disp, panel["plate"], panel["label"],
                                    panel["plate_crop"], w)
                else:
                    del active_panels[cid]

            out.write(disp)

            if frame_count % 15 == 0:
                pct = 0 if total_frames == float('inf') else round(frame_count / total_frames * 100, 1)
                await ws.send_json({
                    "type":        "frame",
                    "frame":       frame_to_b64(disp),
                    "progress":    pct,
                    "frame_count": frame_count,
                    "total":       total_frames,
                })
                await asyncio.sleep(0)

    except WebSocketDisconnect:
        pass
    finally:
        cap.release()
        out.release()

    await ws.send_json({"type": "done", "total_violations": len(violations)})

@app.get("/download/video")
async def download_video():
    return FileResponse("output_final.mp4", media_type="video/mp4",
                        filename="surveillance_output.mp4")

@app.get("/download/csv")
async def download_csv():
    import csv, io
    return JSONResponse({"message": "Use the violations data from the session"})

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)