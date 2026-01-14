from flask import Flask, Response, request, jsonify, render_template
import cv2
import torch
import numpy as np
import threading
import time
from model import UNetSmall

# ======================
# CONFIG
# ======================
RTSP_URL = "rtsp://username:password@ip:port/stream"
IMG_SIZE = 256
THRESHOLD = 0.4
AREA_THRESHOLD = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# LOAD MODEL
# ======================
model = UNetSmall().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

app = Flask(__name__)

# ======================
# RTSP FRAME BUFFER
# ======================
latest_frame = None
lock = threading.Lock()

def rtsp_reader():
    global latest_frame
    cap = cv2.VideoCapture(RTSP_URL)

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.2)
            cap.release()
            cap = cv2.VideoCapture(RTSP_URL)
            continue

        with lock:
            latest_frame = frame.copy()

# Start RTSP thread
threading.Thread(target=rtsp_reader, daemon=True).start()

# ======================
# ROUTES
# ======================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_frame")
def video_frame():
    with lock:
        if latest_frame is None:
            return "", 204
        frame = latest_frame.copy()

    _, buffer = cv2.imencode(".jpg", frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")

@app.route("/inspect", methods=["POST"])
def inspect():
    with lock:
        if latest_frame is None:
            return jsonify({"error": "No frame"}), 500
        img = latest_frame.copy()

    h, w = img.shape[:2]

    # Preprocess
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_norm = img_resized.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img_norm).permute(2,0,1).unsqueeze(0).to(DEVICE)

    # Inference
    with torch.no_grad():
        pred = torch.sigmoid(model(tensor))[0,0].cpu().numpy()

    mask = (pred > THRESHOLD).astype(np.uint8)
    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    defect_area = int(mask.sum())
    status = "FAIL" if defect_area > AREA_THRESHOLD else "PASS"

    overlay = img.copy()
    color = (0,0,255) if status == "FAIL" else (0,255,0)
    overlay[mask > 0] = color
    final = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    _, buffer = cv2.imencode(".jpg", final)

    return Response(
        buffer.tobytes(),
        mimetype="image/jpeg",
        headers={
            "X-Result": status,
            "X-Defect-Area": str(defect_area)
        }
    )

if __name__ == "__main__":
    app.run(debug=False)
