"""
main.py  —  LIVE Multimodal Emotion Recognition with EDR
═══════════════════════════════════════════════════════════
Authors : Md Adil     (22SCSE1012699)
          Rohit Kumar Ram (22SCSE1012599)
College : Galgotias University, Greater Noida

HOW TO RUN:
  python main.py                    ← webcam 0, mock speech
  python main.py --model models/ser_lstm.h5   ← real speech
  python main.py --camera 1         ← different webcam
  python main.py --tau 0.15         ← change EDR threshold

LIVE CONTROLS (press inside the window):
  Q  → Quit
  S  → Save screenshot
  T  → τ + 0.05  (makes system more likely to pick dominant)
  R  → τ - 0.05  (makes system output Uncertain more often)
  P  → Pause / Resume
═══════════════════════════════════════════════════════════
"""

import os, sys, time, threading, argparse
import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

try:
    import cv2
except ImportError:
    print("ERROR: pip install opencv-python")
    sys.exit(1)

from edr_framework import EDRFramework
from ser_module    import SERModule, SAMPLE_RATE, DURATION, MIC_OK

try:
    from deepface import DeepFace
    DEEPFACE_OK = True
except ImportError:
    DEEPFACE_OK = False
    print("[FER] deepface not installed → pip install deepface")
    print("[FER] Using Haar cascade fallback.\n")

try:
    import sounddevice as sd
except (ImportError, OSError):
    pass

# ── Emotion → BGR colour ─────────────────────────────────────────────
EMO_COLOUR = {
    "Happy":     (80,  210,  80),
    "Angry":     (50,   50, 220),
    "Sad":       (220, 120,  50),
    "Fear":      (50,  200, 220),
    "Disgust":   (120,  50, 200),
    "Surprise":  (220, 200,  50),
    "Neutral":   (160, 160, 160),
    "Uncertain": (200,  80, 220),
    "...":       (100, 100, 100),
}

# ── Shared state (between main thread and mic thread) ────────────────
class State:
    def __init__(self):
        self.lock        = threading.Lock()
        self.Es          = "..."
        self.Cs          = 0.0
        self.recording   = False
        self.edr_result  = None
        self.paused      = False

ST = State()


# ════════════════════════════════════════════════════════════════════
#  FACIAL EMOTION — DeepFace CNN
# ════════════════════════════════════════════════════════════════════
def get_face_emotion(frame: np.ndarray) -> tuple[str, float, list]:
    """
    Returns (emotion, confidence, face_boxes)
    face_boxes: list of (x,y,w,h)
    """
    if not DEEPFACE_OK:
        return _haar_fallback(frame)

    try:
        results = DeepFace.analyze(
            img_path      = frame,
            actions       = ["emotion"],
            enforce_detection = False,
            silent        = True,
        )
        if isinstance(results, list):
            results = results[0]

        emotions = results["emotion"]             # {label: 0-100}
        dominant = max(emotions, key=emotions.get)
        conf     = round(emotions[dominant] / 100.0, 4)
        label    = dominant.capitalize()

        # Try to get face region
        region = results.get("region", {})
        x = region.get("x", 0)
        y = region.get("y", 0)
        w = region.get("w", 0)
        h = region.get("h", 0)
        boxes = [(x, y, w, h)] if w > 0 else []

        if conf < 0.25:
            return "Neutral", conf, boxes
        return label, conf, boxes

    except Exception:
        return "Neutral", 0.15, []


def _haar_fallback(frame: np.ndarray) -> tuple[str, float, list]:
    """Haar-cascade face detector + random emotion (no DeepFace)."""
    import random
    gray    = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces   = cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    boxes   = [tuple(f) for f in faces] if len(faces) else []
    emo     = random.choice(["Happy","Neutral","Sad","Angry","Fear"])
    conf    = round(random.uniform(0.45, 0.90), 3)
    return emo, conf, boxes


# ════════════════════════════════════════════════════════════════════
#  MIC THREAD — runs every DURATION seconds
# ════════════════════════════════════════════════════════════════════
def mic_worker(ser: SERModule):
    while True:
        if ST.paused:
            time.sleep(0.5)
            continue

        with ST.lock:
            ST.recording = True

        if MIC_OK and ser.model is not None:
            try:
                rec = sd.rec(int(DURATION * SAMPLE_RATE),
                             samplerate=SAMPLE_RATE,
                             channels=1, dtype="float32")
                sd.wait()
                Es, Cs = ser.predict_array(rec.flatten())
            except Exception:
                Es, Cs = ser.predict_mock()
        else:
            Es, Cs = ser.predict_mock()
            time.sleep(DURATION)

        with ST.lock:
            ST.Es        = Es
            ST.Cs        = Cs
            ST.recording = False


# ════════════════════════════════════════════════════════════════════
#  HUD DRAWING
# ════════════════════════════════════════════════════════════════════
def draw_hud(frame, Ef, Cf, Es, Cs, result, tau, recording, paused):
    h, w = frame.shape[:2]

    # ── Dark top panel ───────────────────────────────────────────────
    panel_h = 230
    overlay  = frame.copy()
    cv2.rectangle(overlay, (0,0), (w, panel_h), (18,18,28), -1)
    cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)

    def put(text, x, y, scale=0.65, col=(220,220,220), thick=1):
        cv2.putText(frame, text, (x,y),
                    cv2.FONT_HERSHEY_DUPLEX, scale, col, thick, cv2.LINE_AA)

    def bar(x, y, w_bar, val, col):
        cv2.rectangle(frame, (x,y), (x+w_bar, y+12), (50,50,50), -1)
        cv2.rectangle(frame, (x,y), (x+int(w_bar*val), y+12), col, -1)

    # Title
    put("EDR Multimodal Emotion Recognition", 12, 28,
        0.72, (170,170,255), 2)
    put("Galgotias Univ | Md Adil & Rohit Kumar Ram",
        12, 48, 0.38, (110,110,160))

    # ── FACIAL ───────────────────────────────────────────────────────
    fc = EMO_COLOUR.get(Ef, (180,180,180))
    put(f"FACE  : {Ef}", 12, 82, 0.70, fc, 2)
    bar(130, 90, 220, min(Cf,1.0), fc)
    put(f"Cf={Cf:.2f}", 360, 100, 0.50, fc)

    # ── SPEECH ───────────────────────────────────────────────────────
    sc = EMO_COLOUR.get(Es, (180,180,180))
    mic_tag = " [REC...]" if recording else ""
    put(f"SPEECH: {Es}{mic_tag}", 12, 125, 0.70, sc, 2)
    bar(130, 133, 220, min(Cs,1.0), sc)
    put(f"Cs={Cs:.2f}", 360, 143, 0.50, sc)

    # ── EDR Result ───────────────────────────────────────────────────
    if result:
        dec = result["decision"]
        emo = result["final_emotion"]
        delta = result["delta"]

        if dec == "AGREEMENT":
            dc = (70, 210, 70);  label = "AGREEMENT"
        elif "DOMINANT" in dec:
            dc = (70, 150, 255); label = f"DOMINANT ({result['dominant_src']})"
        else:
            dc = (200, 70, 220); label = "UNCERTAIN"

        cv2.line(frame, (12,162), (w-12,162), (50,50,70), 1)
        put(f"EDR : {label}", 12, 185, 0.65, dc, 2)
        put(f"FINAL EMOTION → {emo}     Δ={delta:.3f}   τ={tau:.2f}",
            12, 210, 0.58, dc, 1)

    # ── PAUSED banner ────────────────────────────────────────────────
    if paused:
        put("[ PAUSED ]", w//2 - 80, h//2,
            1.2, (0,120,255), 3)

    # ── Bottom bar ───────────────────────────────────────────────────
    cv2.rectangle(frame, (0, h-28), (w,h), (18,18,28), -1)
    put("Q=Quit  S=Screenshot  T=τ+  R=τ-  P=Pause",
        10, h-10, 0.40, (110,110,160))

    return frame


def draw_face_boxes(frame, boxes, Ef):
    """Draw bounding boxes around detected faces."""
    col = EMO_COLOUR.get(Ef, (180,180,180))
    for (x,y,w,h) in boxes:
        cv2.rectangle(frame, (x,y), (x+w, y+h), col, 2)
        cv2.putText(frame, Ef, (x, y-8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return frame


# ════════════════════════════════════════════════════════════════════
#  MAIN LOOP
# ════════════════════════════════════════════════════════════════════
def run(args):
    print("\n" + "═"*55)
    print("  EDR Live System — Galgotias University")
    print(f"  Camera={args.camera}   τ={args.tau}")
    print("  Press Q to quit\n" + "═"*55)

    # Init
    edr = EDRFramework(tau=args.tau)
    ser = SERModule(model_path=args.model)

    # Open webcam
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"ERROR: Cannot open camera {args.camera}")
        sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Mic thread
    t = threading.Thread(target=mic_worker, args=(ser,), daemon=True)
    t.start()

    # Screenshot dir
    ss_dir = os.path.join(os.path.dirname(__file__), "screenshots")
    os.makedirs(ss_dir, exist_ok=True)

    # FER state cache (run every FER_EVERY frames for speed)
    FER_EVERY = 4
    frame_n   = 0
    Ef_cache, Cf_cache, boxes_cache = "...", 0.0, []

    print("[LIVE] Running. Press Q in the window to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[LIVE] Camera read failed.")
            break

        if ST.paused:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"): break
            if key == ord("p"): ST.paused = False
            frame_disp = frame.copy()
            draw_hud(frame_disp, Ef_cache, Cf_cache,
                     ST.Es, ST.Cs, ST.edr_result,
                     edr.tau, ST.recording, True)
            cv2.imshow("EDR — Live Emotion Detection", frame_disp)
            continue

        frame_n += 1

        # FER
        if frame_n % FER_EVERY == 0:
            Ef_cache, Cf_cache, boxes_cache = get_face_emotion(frame)

        Ef, Cf = Ef_cache, Cf_cache

        # Read speech state
        with ST.lock:
            Es = ST.Es
            Cs = ST.Cs
            recording = ST.recording

        # EDR
        if Ef not in ("...",) and Es not in ("...",):
            result = edr.resolve(Ef, Cf, Es, Cs)
            with ST.lock:
                ST.edr_result = result
        else:
            with ST.lock:
                result = ST.edr_result

        # Draw
        frame = draw_face_boxes(frame, boxes_cache, Ef)
        frame = draw_hud(frame, Ef, Cf, Es, Cs,
                         result, edr.tau, recording, False)

        cv2.imshow("EDR — Live Emotion Detection", frame)

        # Keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            p = os.path.join(ss_dir, f"capture_{int(time.time())}.png")
            cv2.imwrite(p, frame)
            print(f"[LIVE] Screenshot → {p}")
        elif key == ord("t"):
            edr.tau = min(0.90, round(edr.tau+0.05, 2))
            print(f"[LIVE] τ → {edr.tau}")
        elif key == ord("r"):
            edr.tau = max(0.05, round(edr.tau-0.05, 2))
            print(f"[LIVE] τ → {edr.tau}")
        elif key == ord("p"):
            ST.paused = True
            print("[LIVE] Paused.")

    cap.release()
    cv2.destroyAllWindows()
    print("[LIVE] Stopped.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", type=int,   default=0,
                    help="Webcam index (default 0)")
    ap.add_argument("--model",  type=str,   default=None,
                    help="Path to trained SER model .h5")
    ap.add_argument("--tau",    type=float, default=0.20,
                    help="EDR threshold (default 0.20)")
    run(ap.parse_args())
