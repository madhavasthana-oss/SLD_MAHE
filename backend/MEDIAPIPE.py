import os
import time
import logging
import urllib.request
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
import mediapipe as mp
import mediapipe.tasks as tasks

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("HandDetector")

# ── MediaPipe Tasks shortcuts ─────────────────────────────────────────────────
_vision       = tasks.vision
HandLandmarker        = _vision.HandLandmarker
HandLandmarkerOptions = _vision.HandLandmarkerOptions
RunningMode           = _vision.RunningMode
BaseOptions           = tasks.BaseOptions

# ── Model download ────────────────────────────────────────────────────────────
_MODEL_FILENAME = "hand_landmarker.task"
_MODEL_URL      = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

def _ensure_model(path: str) -> None:
    if os.path.exists(path):
        return
    logger.info("Downloading %s …", path)
    try:
        urllib.request.urlretrieve(_MODEL_URL, path)
        logger.info("Download complete → %s (%.1f MB)",
                    path, os.path.getsize(path) / 1e6)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download {path}.\n"
            f"Download manually from:\n  {_MODEL_URL}\n"
            f"and place it in your working directory.\n"
            f"Error: {exc}"
        ) from exc


# ── Detection result ──────────────────────────────────────────────────────────
@dataclass
class HandDetection:
    """
    Single detected hand — same interface as the old YOLO-based detector
    so main.py needs no structural changes.
    """
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2) pixels
    confidence: float                   # hand presence confidence from MediaPipe
    class_id: int                       # always 0 — MediaPipe is hand-only
    class_name: str                     # 'Left' or 'Right'
    roi: Optional[np.ndarray] = field(default=None, repr=False)

    # Landmarks — bonus data your SLD doesn't need but you may use later
    landmarks: Optional[list] = field(default=None, repr=False)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)


# ── Core detector ─────────────────────────────────────────────────────────────
class HandDetector:
    """
    Stateless single-frame hand detector using MediaPipe Tasks (v0.10+).

    Replaces YOLO.py with the same public interface:
        detector.detect(frame)          → list[HandDetection]
        detector.draw(frame, dets)      → annotated np.ndarray
        detector.process_frame(frame)   → (list[HandDetection], np.ndarray)
        detector.probe_classes()        → dict  (diagnostic)

    Notes
    -----
    - Uses RunningMode.IMAGE — correct for stateless per-request inference.
    - MediaPipe returns NormalizedLandmarks (0–1). We convert to pixel bbox here.
    - No stale-frame fallback — each call is fully independent.
    - roi_padding mirrors the old interface but padding is also applied in
      main.py (apply_padding). Keep roi_padding=0 here to avoid double-padding.
    """

    _BOX_COLOUR    = (0, 220, 120)
    _LABEL_COLOUR  = (255, 255, 255)
    _FPS_COLOUR    = (0, 200, 255)
    _BOX_THICKNESS = 2
    _FONT          = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        model_path: str = _MODEL_FILENAME,
        min_hand_detection_confidence: float = 0.50,
        min_hand_presence_confidence:  float = 0.50,
        min_tracking_confidence:       float = 0.50,
        num_hands: int = 1,
        roi_padding: int = 0,
    ):
        _ensure_model(model_path)

        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=RunningMode.IMAGE,     # stateless — correct for API
            num_hands=num_hands,
            min_hand_detection_confidence=min_hand_detection_confidence,
            min_hand_presence_confidence=min_hand_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._landmarker = HandLandmarker.create_from_options(options)
        self.roi_padding = roi_padding

        # FPS tracking (meaningful only in webcam mode)
        self._fps: float = 0.0
        self._prev_tick: float = 0.0

        logger.info(
            "HandDetector (MediaPipe) ready | "
            "det_conf=%.2f  presence_conf=%.2f  num_hands=%d",
            min_hand_detection_confidence,
            min_hand_presence_confidence,
            num_hands,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        """
        Detect hands in a single BGR frame (OpenCV format).
        Returns HandDetection objects sorted by area descending.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received empty frame — skipping.")
            return []

        h, w = frame.shape[:2]

        # MediaPipe expects RGB
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        result = self._landmarker.detect(mp_img)

        detections: list[HandDetection] = []

        if not result.hand_landmarks:
            logger.debug("No hands detected in frame (shape=%s)", frame.shape)
            return []

        for idx, landmark_list in enumerate(result.hand_landmarks):
            # ── Derive pixel bbox from landmark extents ────────────────────
            xs = [lm.x for lm in landmark_list]
            ys = [lm.y for lm in landmark_list]

            x1 = max(0, int(min(xs) * w))
            y1 = max(0, int(min(ys) * h))
            x2 = min(w, int(max(xs) * w))
            y2 = min(h, int(max(ys) * h))

            # ── Hand presence confidence ───────────────────────────────────
            # result.handedness[idx] is a list of Category objects
            confidence  = 0.0
            class_name  = "hand"
            if result.handedness and idx < len(result.handedness):
                cat        = result.handedness[idx][0]
                confidence = float(cat.score)
                class_name = cat.display_name or cat.category_name  # 'Left' / 'Right'

            roi = self._crop_roi(frame, x1, y1, x2, y2)

            detections.append(HandDetection(
                bbox=(x1, y1, x2, y2),
                confidence=confidence,
                class_id=0,
                class_name=class_name,
                roi=roi,
                landmarks=landmark_list,
            ))

        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def draw(
        self,
        frame: np.ndarray,
        detections: list[HandDetection],
        show_fps: bool = True,
        show_confidence: bool = True,
        draw_landmarks: bool = True,
    ) -> np.ndarray:
        """Draw bboxes, landmarks, and labels onto a copy of the frame."""
        canvas = frame.copy()
        h, w   = canvas.shape[:2]

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2),
                          self._BOX_COLOUR, self._BOX_THICKNESS)

            # Label
            label = det.class_name
            if show_confidence:
                label += f"  {det.confidence:.0%}"

            (tw, th), baseline = cv2.getTextSize(label, self._FONT, 0.55, 1)
            cv2.rectangle(canvas,
                          (x1, y1 - th - baseline - 6),
                          (x1 + tw + 6, y1),
                          self._BOX_COLOUR, cv2.FILLED)
            cv2.putText(canvas, label,
                        (x1 + 3, y1 - baseline - 3),
                        self._FONT, 0.55, self._LABEL_COLOUR, 1, cv2.LINE_AA)

            cv2.circle(canvas, det.center, 4, self._BOX_COLOUR, cv2.FILLED)

            # Landmarks (21 points)
            if draw_landmarks and det.landmarks:
                for lm in det.landmarks:
                    px, py = int(lm.x * w), int(lm.y * h)
                    cv2.circle(canvas, (px, py), 3, (255, 255, 0), cv2.FILLED)

        if show_fps:
            cv2.putText(canvas, f"FPS: {self._fps:.1f}", (10, 30),
                        self._FONT, 0.9, self._FPS_COLOUR, 2, cv2.LINE_AA)

        return canvas

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> tuple[list[HandDetection], np.ndarray]:
        """Convenience wrapper: detect + optionally draw + update FPS."""
        self._tick()
        detections = self.detect(frame)
        annotated  = self.draw(frame, detections) if draw else frame
        return detections, annotated

    def probe_classes(self) -> dict[int, str]:
        """Diagnostic — mirrors YOLO interface. MediaPipe is always hand-only."""
        return {0: "hand (Left/Right via handedness)"}

    @property
    def fps(self) -> float:
        return self._fps

    def __del__(self):
        # Clean up the landmarker context
        try:
            self._landmarker.close()
        except Exception:
            pass

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.perf_counter()
        if self._prev_tick:
            elapsed   = now - self._prev_tick
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / elapsed)
        self._prev_tick = now

    def _crop_roi(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        px1  = max(0, x1 - self.roi_padding)
        py1  = max(0, y1 - self.roi_padding)
        px2  = min(w, x2 + self.roi_padding)
        py2  = min(h, y2 + self.roi_padding)
        return frame[py1:py2, px1:px2].copy()


# ── Stand-alone webcam demo ───────────────────────────────────────────────────
def run_webcam(
    model_path: str = _MODEL_FILENAME,
    camera_index: int = 0,
    min_detection_confidence: float = 0.50,
    window_name: str = "Sign Language — MediaPipe Hand Detection",
) -> None:
    """
    Live webcam loop. Q / ESC to quit. S to snapshot.
    """
    detector = HandDetector(
        model_path=model_path,
        min_hand_detection_confidence=min_detection_confidence,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", camera_index)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)

    logger.info("Webcam opened. Q/ESC = quit, S = snapshot.")
    snapshot_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame — retrying…")
                continue

            detections, canvas = detector.process_frame(frame)

            for i, d in enumerate(detections):
                logger.debug(
                    "Hand %d | class=%s  conf=%.2f  bbox=%s  center=%s",
                    i, d.class_name, d.confidence, d.bbox, d.center,
                )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                path = f"snapshot_{snapshot_idx:04d}.jpg"
                cv2.imwrite(path, canvas)
                logger.info("Snapshot saved → %s", path)
                snapshot_idx += 1

    finally:
        cap.release()
        cv2.destroyAllWindows()
        logger.info("Camera released. Exiting.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Real-time MediaPipe hand detector for sign language."
    )
    parser.add_argument("--model",  default=_MODEL_FILENAME)
    parser.add_argument("--camera", type=int,   default=0)
    parser.add_argument("--conf",   type=float, default=0.50)
    args = parser.parse_args()

    run_webcam(
        model_path=args.model,
        camera_index=args.camera,
        min_detection_confidence=args.conf,
    )