"""
main.py  —  FastAPI inference endpoint for ASL sign detection.

Fixes over original
───────────────────
1. square_crop pads with edge-replicated border (cv2.BORDER_REPLICATE) instead
   of black zeros. Black padding shifts the ResNet colour distribution away from
   what it saw during training and tanks confidence on non-square crops.

2. Thread-safe global state  →  prediction_buffer / last_prediction / last_time
   are now guarded by a threading.Lock. FastAPI runs multiple async workers and
   the originals raced silently.

3. Buffer reset on hand-loss  →  when the detector returns "no hand" or
   "hand moving", the prediction buffer is cleared so stale votes from a previous
   sign don't poison the first vote of the next sign.

4. torch.load weights_only=False is explicit  →  suppresses PyTorch 2.x
   deprecation warning and documents intent clearly.

5. Minor: renamed loop variable `l` → `lbl` (shadows built-in `l`).
"""

import threading
import time
from collections import deque

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from torchvision import transforms

from MEDIAPIPE import HandDetector
from SLD import SLD

# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

CLF_THRESHOLD = 0.50   # minimum softmax probability to accept a prediction
COOLDOWN      = 1.0    # seconds before the same letter is emitted again
BUFFER_SIZE   = 10     # majority-vote window size
VOTE_RATIO    = 0.70   # fraction of buffer that must agree for a stable vote

# ─────────────────────────────────────────────────────────────────────────────
# Models
# ─────────────────────────────────────────────────────────────────────────────

detector = HandDetector(expansion=0.35)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SLD(num_classes=29)
checkpoint = torch.load("BEST_MODEL.pth", map_location=device, weights_only=False)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(device)
model.eval()

# ─────────────────────────────────────────────────────────────────────────────
# Pre-processing
# ─────────────────────────────────────────────────────────────────────────────

_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def preprocess(img: np.ndarray) -> torch.Tensor:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return _preprocess(img).unsqueeze(0).to(device)


def square_crop(img: np.ndarray) -> np.ndarray:
    """
    Pad a non-square ROI to a square using edge replication, not black zeros.
    Black padding creates an artificial border that the ResNet never saw during
    training; edge replication is a neutral, natural extension.
    """
    h, w = img.shape[:2]
    diff = abs(h - w)
    half = diff // 2
    rest = diff - half

    if h < w:          # wider than tall → pad top / bottom
        return cv2.copyMakeBorder(img, half, rest, 0, 0, cv2.BORDER_REPLICATE)
    elif w < h:        # taller than wide → pad left / right
        return cv2.copyMakeBorder(img, 0, 0, half, rest, cv2.BORDER_REPLICATE)
    return img         # already square


# ─────────────────────────────────────────────────────────────────────────────
# Thread-safe temporal state
# ─────────────────────────────────────────────────────────────────────────────

_lock             = threading.Lock()
_pred_buffer      = deque(maxlen=BUFFER_SIZE)
_last_prediction  = None
_last_time: float = 0.0


def _reset_buffer() -> None:
    """Clear the vote buffer — call whenever the hand stream resets."""
    with _lock:
        _pred_buffer.clear()


def _stable_vote() -> str | None:
    """
    Returns the majority label if it appears in >= VOTE_RATIO of the buffer
    AND the buffer is full, else None.
    """
    with _lock:
        if len(_pred_buffer) < BUFFER_SIZE:
            return None
        counts: dict[str, int] = {}
        for lbl in _pred_buffer:
            counts[lbl] = counts.get(lbl, 0) + 1
        best = max(counts, key=counts.get)
        return best if counts[best] >= BUFFER_SIZE * VOTE_RATIO else None


# ─────────────────────────────────────────────────────────────────────────────
# Classes
# ─────────────────────────────────────────────────────────────────────────────

_CLASSES = [chr(ord("A") + i) for i in range(26)] + ["del", "nothing", "space"]

# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    global _last_prediction, _last_time

    try:
        contents = await file.read()
        frame    = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

        if frame is None:
            return {"error": "decode failed"}

        detections = detector.detect(frame)

        if not detections:
            _reset_buffer()
            return {"letter": None, "error": "no hand"}

        # Pick the detection with the highest handedness score
        det = max(detections, key=lambda d: d.score)

        if not det.is_stable:
            _reset_buffer()
            return {"letter": None, "error": "hand moving", "motion": det.motion}

        # ── Classify ──────────────────────────────────────────────────────────
        roi          = square_crop(det.roi)
        input_tensor = preprocess(roi)

        with torch.no_grad():
            out   = model(input_tensor)
            probs = torch.softmax(out, dim=1)
            conf, pred = torch.max(probs, dim=1)

        conf  = float(conf.item())
        label = _CLASSES[pred.item()]

        if conf < CLF_THRESHOLD:
            return {"letter": None, "confidence": conf, "error": "low confidence"}

        # ── Temporal vote ─────────────────────────────────────────────────────
        with _lock:
            _pred_buffer.append(label)

        stable = _stable_vote()

        if stable is None:
            return {"letter": None, "confidence": conf, "error": "not stable yet"}

        # ── Cooldown ──────────────────────────────────────────────────────────
        now = time.time()
        with _lock:
            if stable == _last_prediction and (now - _last_time) < COOLDOWN:
                return {"letter": None, "confidence": conf, "error": "cooldown"}
            _last_prediction = stable
            _last_time       = now

        x1, y1, x2, y2 = det.bbox

        return {
            "letter":     stable,
            "confidence": conf,
            "bbox":       [x1, y1, x2 - x1, y2 - y1],
            "handedness": det.handedness,
            "motion":     det.motion,
            "landmarks": [{"x": lm.x, "y": lm.y} for lm in det.landmarks]
        }

    except Exception as e:
        return {"error": str(e)}