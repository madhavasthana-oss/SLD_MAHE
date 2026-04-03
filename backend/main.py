# ─────────────────────────────────────────────
# main.py — FastAPI inference server
# ─────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import torch

from YOLO import HandDetector
from SLD import SLD

# ─────────────────────────────────────────────
# INIT APP
# ─────────────────────────────────────────────

app = FastAPI()

# allow frontend (api.html) to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# LOAD MODELS (runs once at startup)
# ─────────────────────────────────────────────

print("<<<< Loading models... >>>>")

# YOLO detector
detector = HandDetector(
    model_path="yolov8n.pt",   # replace with your trained weights if needed
    target_classes=["hand"]
)

# ResNet classifier
device = "cuda" if torch.cuda.is_available() else "cpu"

model = SLD(num_classes=26)
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.to(device)
model.eval()

print(f"✅ Models loaded on {device}")


# ─────────────────────────────────────────────
# PREPROCESS FUNCTION
# ─────────────────────────────────────────────

def preprocess(img):
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.transpose(img, (2, 0, 1))  # HWC → CHW
    tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    return tensor.to(device)


# ─────────────────────────────────────────────
# HEALTH CHECK (optional but useful)
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "ASL backend running"}


# ─────────────────────────────────────────────
# MAIN PREDICTION ENDPOINT
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ── Read image ───────────────────────
        contents = await file.read()
        np_arr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"letter": None, "confidence": 0, "bbox": None}

        # ── YOLO detection ───────────────────
        detections = detector.detect(frame)

        if not detections:
            return {"letter": None, "confidence": 0, "bbox": None}

        # take largest hand
        det = detections[0]
        roi = det.roi
        x1, y1, x2, y2 = det.bbox

        # ── Classification ───────────────────
        input_tensor = preprocess(roi)

        with torch.no_grad():
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        # map index → letter (A-Z)
        label = chr(ord('A') + pred.item())

        # ── Response ─────────────────────────
        return {
            "letter": label,
            "confidence": float(conf.item()),
            "bbox": [x1, y1, x2 - x1, y2 - y1]
        }

    except Exception as e:
        print("Error:", str(e))
        return {"letter": None, "confidence": 0, "bbox": None}