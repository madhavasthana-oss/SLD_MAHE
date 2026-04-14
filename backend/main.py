# ─────────────────────────────────────────────
# main.py — FastAPI inference server
# ─────────────────────────────────────────────

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import numpy as np
import cv2
import torch
from torchvision import transforms

from YOLO import HandDetector
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
# LOAD MODELS (runs once at startup)
# ─────────────────────────────────────────────

print("<<<< Loading models... >>>>")

detector = HandDetector(
    model_path="hand_yolov8n.pt",
    target_classes=["hand"]
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SLD(num_classes=29)
model.load_state_dict(
    torch.load("BEST_MODEL.pth", map_location=device, weights_only=False)['model_state_dict']
)
model.to(device)
model.eval()

print(f"✅ Models loaded on {device}")

# ─────────────────────────────────────────────
# PREPROCESS FUNCTION
# ─────────────────────────────────────────────

# ✅ Must match eval_transform used during training
_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],   # ImageNet stats — same as training
        std=[0.229, 0.224, 0.225]
    )
])

def preprocess(img: np.ndarray) -> torch.Tensor:
    """Convert BGR numpy frame → normalised float tensor on device."""
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)   # YOLO gives BGR
    tensor  = _preprocess(img_rgb).unsqueeze(0)       # (1, 3, 224, 224)
    return tensor.to(device)

# Class index → label
# ASL alphabet dataset folder names sorted alphabetically:
# A–Z (indices 0–25), then del (26), nothing (27), space (28)
_CLASSES = [chr(ord('A') + i) for i in range(26)] + ['del', 'nothing', 'space']

# ─────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────

@app.get("/")
def health():
    return {"status": "ok", "message": "ASL backend running"}

# ─────────────────────────────────────────────
# PREDICTION ENDPOINT
# ─────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read image
        contents = await file.read()
        np_arr   = np.frombuffer(contents, np.uint8)
        frame    = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if frame is None:
            return {"letter": None, "confidence": 0.0, "bbox": None}

        # YOLO detection
        detections = detector.detect(frame)

        if not detections:
            return {"letter": None, "confidence": 0.0, "bbox": None}

        # Largest hand first (already sorted in HandDetector.detect)
        det          = detections[0]
        roi          = det.roi
        x1, y1, x2, y2 = det.bbox

        # Classification
        input_tensor = preprocess(roi)

        with torch.no_grad():
            output = model(input_tensor)
            probs  = torch.softmax(output, dim=1)
            conf, pred = torch.max(probs, dim=1)

        # Map index → label
        label = _CLASSES[pred.item()]

        return {
            "letter":     label,
            "confidence": float(conf.item()),
            "bbox":       [x1, y1, x2 - x1, y2 - y1]
        }

    except Exception as e:
        print("Error:", str(e))
        return {"letter": None, "confidence": 0.0, "bbox": None}