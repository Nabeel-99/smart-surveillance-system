import os, cv2, tempfile, asyncio, base64
from datetime import datetime
from collections import Counter
from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse
import uvicorn
import numpy as np
from ultralytics import YOLO
import easyocr

app = FastAPI()
os.makedirs("violations_snapshots", exist_ok=True)
os.makedirs("static", exist_ok=True)

_model = _plate_model = _reader = None

def get_models():
    global _model, _plate_model, _reader
    if _model is None:
        _model       = YOLO("yolov8s.pt")
        _plate_model = YOLO("plate_detector.pt")
        _reader      = easyocr.Reader(['en'])
    return _model, _plate_model, _reader

# def get_light_color(frame, box, frame_h):
#     x1,y1,x2,y2 = map(int, box)
#     if y1 > frame_h * 0.45: return "UNKNOWN"
#     crop = frame[y1:y2, x1:x2]
#     if crop.size == 0: return "UNKNOWN"
#     hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
#     r = cv2.inRange(hsv,(0,70,50),(10,255,255)) + cv2.inRange(hsv,(160,70,50),(180,255,255))
#     g = cv2.inRange(hsv,(40,40,40),(95,255,255))
#     a = cv2.inRange(hsv,(15,40,40),(35,255,255))
#     scores = {"RED":int(r.sum()),"GREEN":int(g.sum()),"AMBER":int(a.sum())}
#     best = max(scores, key=scores.get)
#     return best if scores[best] > 300 else "UNKNOWN"
def get_light_color(frame, box, frame_h):
    x1, y1, x2, y2 = map(int, box)
    if y1 > frame_h * 0.45:
        return "UNKNOWN"
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return "UNKNOWN"

    h = crop.shape[0]
    third = max(1, h // 3)

    def dominant(c):
        if c.size == 0:
            return "UNKNOWN", 0
        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
        r = cv2.inRange(hsv, (0,70,50), (10,255,255)) + \
            cv2.inRange(hsv, (160,70,50), (180,255,255))
        g = cv2.inRange(hsv, (40,40,40), (95,255,255))
        a = cv2.inRange(hsv, (15,40,40), (35,255,255))
        scores = {"RED": int(r.sum()), "GREEN": int(g.sum()), "AMBER": int(a.sum())}
        best = max(scores, key=scores.get)
        return (best, scores[best]) if scores[best] > 300 else ("UNKNOWN", 0)

    tc, ts = dominant(crop[:third])        # red lives here
    mc, ms = dominant(crop[third:2*third]) # amber lives here
    bc, bs = dominant(crop[2*third:])      # green lives here

    if tc == "RED"   and ts > bs: return "RED"
    if bc == "GREEN" and bs > ts: return "GREEN"
    if mc == "AMBER" and ms > 300: return "AMBER"  # ← the missing line
    return "UNKNOWN"

def read_plate(crop, reader):
    if crop.size == 0: return None
    h, w = crop.shape[:2]
    if w < 80 or h < 20: return None
    crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    for _, text, conf in reader.readtext(crop):
        clean = ''.join(c for c in text.upper() if c.isalnum())
        if len(clean) >= 4 and conf > 0.3: return clean
    return None

def crosses_line(p1, p2, x1, y1, x2, y2):
    def sign(o, a, b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    return (sign(p1,p2,(x1,y2)) > 0) != (sign(p1,p2,(x2,y2)) > 0)

def draw_info_panel(disp, plate_text, label, plate_crop, frame_w):
    pw, ph = 320, 100
    px, py = frame_w - pw - 10, 10
    ov = disp.copy()
    cv2.rectangle(ov,(px,py),(px+pw,py+ph),(20,20,20),-1)
    cv2.addWeighted(ov,0.7,disp,0.3,0,disp)
    cv2.rectangle(disp,(px,py),(px+pw,py+ph),(0,0,255),2)
    if plate_crop is not None and plate_crop.size > 0:
        ch,cw = plate_crop.shape[:2]
        if cw>0 and ch>0:
            disp[py+5:py+50, px+5:px+145] = cv2.resize(plate_crop,(140,45))
    cv2.putText(disp,f"PLATE: {plate_text}",(px+5,py+60),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,255,255),2)
    cv2.putText(disp,label.upper(),(px+5,py+82),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),1)
    cv2.putText(disp,"VIOLATION",(px+170,py+60),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,255),2)

def f2b(frame, q=60):
    _, buf = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, q])
    return base64.b64encode(buf).decode()

async def safe_send(ws, payload):
    try: await ws.send_json(payload)
    except Exception: pass


@app.get("/", response_class=HTMLResponse)
async def index():
    with open("templates/index.html") as f: return f.read()


# ── LIVE PREVIEW ───────────────────────────────────────────────────────────────
# Opens ONE cap, stores it on app.state.live_cap.
# Analyze will READ from this same cap — no second DroidCam connection ever.
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

        # Store cap so analyze can share it
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
        # Only release if analyze is NOT running
        if not getattr(app.state, 'analyzing', False):
            if cap: cap.release()
            app.state.live_cap = None
        print("[PREVIEW] WS closed")


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
    app.state.video_path   = tmp.name
    app.state.frame_shape  = (h, w)
    app.state.total_frames = total
    app.state.fps          = fps
    app.state.is_live      = False
    app.state.live_cap     = None
    return {"frame":f2b(frame),"width":w,"height":h,"total_frames":total}


@app.websocket("/ws/analyze")
async def analyze(ws: WebSocket):
    await ws.accept()
    own_cap  = None   # cap we opened ourselves (file mode)
    out      = None
    violations = []
    app.state.analyzing = True

    try:
        data = await ws.receive_json()
        lp1 = tuple(data["line_p1"])
        lp2 = tuple(data["line_p2"])

        is_live      = getattr(app.state, 'is_live', False)
        h, w         = app.state.frame_shape
        total_frames = app.state.total_frames
        fps          = app.state.fps

        print(f"[ANALYZE] mode={'LIVE' if is_live else 'FILE'}")
        model, plate_model, reader = get_models()

        if is_live:
            # Reuse the preview cap — DroidCam only allows one connection
            cap = getattr(app.state, 'live_cap', None)
            if cap is None:
                await safe_send(ws, {"type":"error","msg":"Camera not connected — click Connect first"})
                return
            print("[ANALYZE] sharing live cap from preview")
        else:
            own_cap = cv2.VideoCapture(app.state.video_path)
            cap = own_cap
            out = cv2.VideoWriter("output_final.mp4", cv2.VideoWriter_fourcc(*'mp4v'), fps, (w,h))

        car_memory    = {}
        logged_plates = set()
        last_disp     = {}
        active_panels = {}
        frame_count   = 0
        lbs           = None   # light_box_smooth
        lch           = []     # light_color_hist
        LEMA, LWIN    = 0.25, 6
        SEND          = 5

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                if is_live: await asyncio.sleep(0.05); continue
                else: break

            frame_count += 1
            disp = frame.copy()
            cv2.line(disp, lp1, lp2, (0,0,255), 2)
            mx = (lp1[0]+lp2[0])//2 - 60
            my = (lp1[1]+lp2[1])//2 - 10
            cv2.putText(disp,"STOP LINE",(mx,my),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,0,255),2)

            if frame_count % 3 != 0:
                for cid, info in last_disp.items():
                    bx1,by1,bx2,by2 = info["box"]
                    cv2.rectangle(disp,(bx1,by1),(bx2,by2),info["vc"],2)
                    cv2.putText(disp,info["sl"],(bx1,by1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,info["vc"],2)
                    if info.get("plate") and info.get("pbox_rel"):
                        r1,r2,r3,r4 = info["pbox_rel"]; cb1,cb2,_,_ = info["box"]
                        cv2.rectangle(disp,(cb1+r1,cb2+r2),(cb1+r3,cb2+r4),(0,255,255),2)
                        cv2.putText(disp,info["plate"],(cb1+r1,cb2+r2-8),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)
                    if lbs:
                        lx1,ly1,lx2,ly2=[int(v) for v in lbs]
                        cur = Counter(lch).most_common(1)[0][0] if lch else "UNKNOWN"
                        lc={"RED":(0,0,255),"AMBER":(0,165,255),"GREEN":(0,200,0)}.get(cur,(128,128,128))
                        cv2.rectangle(disp,(lx1,ly1),(lx2,ly2),lc,2)
                        cv2.putText(disp,f"Light:{cur}",(lx1,ly1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,lc,2)
                for cid, panel in list(active_panels.items()):
                    if frame_count <= panel["until_frame"]: draw_info_panel(disp,panel["plate"],panel["label"],panel["plate_crop"],w)
                    else: del active_panels[cid]
                if out: out.write(disp)
                if frame_count % SEND == 0:
                    pct = 0 if is_live else round(frame_count/total_frames*100,1)
                    await safe_send(ws,{"type":"frame","frame":f2b(disp,55),"progress":pct,
                                        "frame_count":frame_count,"total":total_frames if not is_live else "LIVE"})
                    await asyncio.sleep(0)
                continue

            results = model.track(frame, persist=True, conf=0.35, verbose=False)[0]
            if results.boxes.id is None:
                if out: out.write(disp)
                continue

            nl, nc, bc = None, "UNKNOWN", 0
            for box, cls, conf in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.conf):
                if model.names[int(cls)] == "traffic light" and float(conf) > bc:
                    c = get_light_color(frame, box, h)
                    if c != "UNKNOWN": bc=float(conf); nl=box; nc=c

            if nc != "UNKNOWN": lch.append(nc)
            if len(lch) > LWIN: lch.pop(0)
            light_color = Counter(lch).most_common(1)[0][0] if lch else "UNKNOWN"

            if nl is not None:
                raw = [float(int(x)) for x in nl]
                lbs = raw if lbs is None else [LEMA*r+(1-LEMA)*s for r,s in zip(raw,lbs)]

            lc_bgr, lcoords = (128,128,128), None
            if lbs:
                lx1,ly1,lx2,ly2=[int(v) for v in lbs]
                lc_bgr={"RED":(0,0,255),"AMBER":(0,165,255),"GREEN":(0,200,0)}.get(light_color,(128,128,128))
                lcoords=(lx1,ly1,lx2,ly2)
                cv2.rectangle(disp,(lx1,ly1),(lx2,ly2),lc_bgr,2)
                cv2.putText(disp,f"Light:{light_color}",(lx1,ly1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,lc_bgr,2)

            last_disp = {}
            for box, cls, tid in zip(results.boxes.xyxy, results.boxes.cls, results.boxes.id):
                label = model.names[int(cls)]
                if label not in ["car","truck","bus","motorcycle"] or tid is None: continue
                car_id = int(tid); x1,y1,x2,y2 = map(int, box)
                if car_id not in car_memory:
                    car_memory[car_id]={"label":label,"plates":[],"pbox_rel":None,"logged":False,"is_violator":False}
                mem = car_memory[car_id]
                if light_color=="RED" and crosses_line(lp1,lp2,x1,y1,x2,y2): mem["is_violator"]=True

                pcr = None
                if mem["is_violator"] and not mem["logged"]:
                    cc = frame[y1:y2,x1:x2]
                    pr = plate_model.predict(cc, conf=0.25, verbose=False)[0]
                    for pb in pr.boxes.xyxy:
                        px1,py1,px2,py2=map(int,pb); pc=cc[py1:py2,px1:px2]
                        t=read_plate(pc,reader)
                        if t: mem["plates"].append(t); mem["pbox_rel"]=(px1,py1,px2,py2); pcr=pc
                    if pcr is None and len(pr.boxes.xyxy)>0:
                        px1,py1,px2,py2=map(int,pr.boxes.xyxy[0]); pcr=cc[py1:py2,px1:px2]

                bp = Counter(mem["plates"]).most_common(1)[0][0] if mem["plates"] else None
                if mem["is_violator"]: vc,sl=(0,0,255),"VIOLATION"
                elif light_color=="AMBER": vc,sl=(0,165,255),"CAUTION"
                elif light_color=="GREEN": vc,sl=(0,200,0),"OK"
                else: vc,sl=(128,128,128),label.upper()

                cv2.rectangle(disp,(x1,y1),(x2,y2),vc,2)
                cv2.putText(disp,sl,(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.7,vc,2)
                if bp and mem["pbox_rel"]:
                    r1,r2,r3,r4=mem["pbox_rel"]
                    cv2.rectangle(disp,(x1+r1,y1+r2),(x1+r3,y1+r4),(0,255,255),2)
                    cv2.putText(disp,bp,(x1+r1,y1+r2-8),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0,255,255),2)

                last_disp[car_id]={"box":(x1,y1,x2,y2),"vc":vc,"sl":sl,"plate":bp,
                                    "pbox_rel":mem["pbox_rel"],"light_box":lcoords,"lc":lc_bgr}

                if mem["is_violator"] and not mem["logged"]:
                    ps = bp or "UNREAD"
                    if ps=="UNREAD" or ps not in logged_plates:
                        snap=f"violations_snapshots/car{car_id}_f{frame_count}.jpg"
                        cv2.imwrite(snap, frame[y1:y2,x1:x2])
                        ts=datetime.now().strftime("%H:%M:%S")
                        viol={"frame":frame_count,"time":ts,"car_id":car_id,
                              "plate":ps,"label":label,"snap":f2b(frame[y1:y2,x1:x2],85)}
                        violations.append(viol)
                        active_panels[car_id]={"plate":ps,"label":label,"plate_crop":pcr,"until_frame":frame_count+90}
                        if ps!="UNREAD": logged_plates.add(ps)
                        mem["logged"]=True
                        print(f"[ANALYZE] VIOLATION plate={ps} frame={frame_count}")
                        await safe_send(ws,{"type":"violation","data":viol})

            for cid, panel in list(active_panels.items()):
                if frame_count<=panel["until_frame"]: draw_info_panel(disp,panel["plate"],panel["label"],panel["plate_crop"],w)
                else: del active_panels[cid]
            if out: out.write(disp)
            if frame_count % SEND == 0:
                pct = 0 if is_live else round(frame_count/total_frames*100,1)
                await safe_send(ws,{"type":"frame","frame":f2b(disp,55),"progress":pct,
                                    "frame_count":frame_count,"total":total_frames if not is_live else "LIVE"})
                await asyncio.sleep(0)

    except WebSocketDisconnect:
        pass
    finally:
        app.state.analyzing = False
        if own_cap: own_cap.release()
        if out: out.release()

    await safe_send(ws,{"type":"done","total_violations":len(violations)})


@app.get("/download/video")
async def download_video():
    return FileResponse("output_final.mp4", media_type="video/mp4", filename="surveillance_output.mp4")

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)