from flask import Flask, Response, request, jsonify, render_template
import cv2
import torch
import numpy as np
from model import UNetSmall

app = Flask(__name__)

# -------- MODEL --------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = UNetSmall().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

cap = cv2.VideoCapture("/home/priyanshu/Downloads/WhatsApp Video 2026-01-14 at 2.16.44 PM.mp4")

IMG_SIZE = 256
THRESHOLD = 0.4
AREA_THRESHOLD = 300

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_frame")
def video_frame():
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, frame = cap.read()

    _, buffer = cv2.imencode(".jpg", frame)
    return Response(buffer.tobytes(), mimetype="image/jpeg")

@app.route("/inspect", methods=["POST"])
def inspect():
    file = request.files["frame"]
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
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

    # Overlay
    overlay = img.copy()
    color = (0,0,255) if status == "FAIL" else (0,255,0)
    overlay[mask > 0] = color
    final = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

    # Encode overlay image
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
