from flask import Flask, request, render_template
import cv2
import torch
import numpy as np
import base64
from model import UNetSmall

app = Flask(__name__)

# ----------------------
# CONFIG
# ----------------------
IMG_SIZE = 256
THRESHOLD = 0.4
AREA_THRESHOLD = 300

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# LOAD MODEL
# ----------------------
model = UNetSmall().to(DEVICE)
model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))
model.eval()

# ----------------------
# ROUTES
# ----------------------
@app.route("/", methods=["GET", "POST"])
def upload_image():
    result = None
    overlay_b64 = None

    if request.method == "POST":
        file = request.files["image"]
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
        result = "FAIL" if defect_area > AREA_THRESHOLD else "PASS"

        # Overlay
        overlay = img.copy()
        color = (0,0,255) if result == "FAIL" else (0,255,0)
        overlay[mask > 0] = color
        final = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)

        # Encode image to base64
        _, buffer = cv2.imencode(".jpg", final)
        overlay_b64 = base64.b64encode(buffer).decode("utf-8")

    return render_template(
        "upload.html",
        result=result,
        overlay_b64=overlay_b64
    )

if __name__ == "__main__":
    app.run(debug=False)
