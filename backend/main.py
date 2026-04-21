# ─────────────────────────────────────────────
# main.py — FastAPI inference server
# ─────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import torch
from torchvision import transforms

from MEDIAPIPE import HandDetector   
from SLD import SLD

# ─────────────────────────────────────────────
# INIT APP
# ─────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

DETECTION_CONF = 0.50   # MediaPipe hand detection confidence
PRESENCE_CONF  = 0.50   # MediaPipe hand presence confidence
CLF_THRESHOLD  = 0.50   # Classifier softmax confidence gate
PADDING        = 0.20   # Fractional bbox padding before classifier crop

# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────

print("<<<< Loading models... >>>>")

detector = HandDetector(
    model_path="hand_landmarker.task",
    min_hand_detection_confidence=DETECTION_CONF,
    min_hand_presence_confidence=PRESENCE_CONF,
    num_hands=1,
    roi_padding=0,      # padding is applied below in apply_padding()
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SLD(num_classes=29)
checkpoint = torch.load("BEST_MODEL.pth", map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

print(f"✅ Models loaded on {device}")

# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────

_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(img: np.ndarray) -> torch.Tensor:
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tensor  = _preprocess(img_rgb).unsqueeze(0)
    return tensor.to(device)

# ─────────────────────────────────────────────
# UTILS
# ─────────────────────────────────────────────

def apply_padding(x1, y1, x2, y2, frame_shape, pad=PADDING):
    h, w = frame_shape[:2]
    bw   = x2 - x1
    bh   = y2 - y1
    x1   = max(0, int(x1 - bw * pad))
    y1   = max(0, int(y1 - bh * pad))
    x2   = min(w, int(x2 + bw * pad))
    y2   = min(h, int(y2 + bh * pad))
    return x1, y1, x2, y2

def select_best_detection(detections):
    """Pick highest-confidence detection above threshold. Ties broken by area."""
    if not detections:
        return None

    filtered = [d for d in detections if d.confidence >= DETECTION_CONF]
    if not filtered:
        return None

    filtered.sort(key=lambda d: (d.confidence, d.area), reverse=True)
    return filtered[0]

# ASL class labels — 26 letters + del / nothing / space
_CLASSES = [chr(ord('A') + i) for i in range(26)] + ['del', 'nothing', 'space']

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "ASL backend running (MediaPipe)"}


@app.get("/classes")
def list_classes():
    """Diagnostic — shows detector info and ASL label list."""
    return {
        "detector": detector.probe_classes(),
        "asl_classes": _CLASSES,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # ─── Decode image ───
        contents = await file.read()

        if not contents:
            return {"letter": None, "confidence": 0.0, "bbox": None, "error": "empty file"}

        np_arr = np.frombuffer(contents, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"letter": None, "confidence": 0.0, "bbox": None, "error": "decode failed"}

        # ─── Detect hand ───
        detections = detector.detect(frame)

        det = select_best_detection(detections)
        if det is None:
            return {"letter": None, "confidence": 0.0, "bbox": None, "error": "no hand detected"}

        x1, y1, x2, y2 = det.bbox

        # ─── Pad bbox before crop ───
        x1, y1, x2, y2 = apply_padding(x1, y1, x2, y2, frame.shape)

        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            return {"letter": None, "confidence": 0.0, "bbox": None, "error": "empty roi after padding"}

        # ─── Classify ───
        input_tensor = preprocess(roi)

        with torch.no_grad():
            output     = model(input_tensor)
            probs      = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        clf_conf = float(conf.item())

        if clf_conf < CLF_THRESHOLD:
            return {
                "letter": None,
                "confidence": clf_conf,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "error": "low classifier confidence",
            }

        label = _CLASSES[pred.item()]

        return {
            "letter":     label,
            "confidence": clf_conf,
            "bbox":       [x1, y1, x2 - x1, y2 - y1],
            "error":      None,
        }

    except Exception as e:
        print("🔥 ERROR:", str(e))
        return {"letter": None, "confidence": 0.0, "bbox": None, "error": str(e)}