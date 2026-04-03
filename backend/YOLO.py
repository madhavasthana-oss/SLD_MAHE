import time
import logging
from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("HandDetector")


# ── Detection result container ────────────────────────────────────────────────
@dataclass
class HandDetection:
    """Single detected hand with all relevant metadata."""
    bbox: tuple[int, int, int, int]     # (x1, y1, x2, y2) in pixels
    confidence: float
    class_id: int
    class_name: str
    roi: Optional[np.ndarray] = field(default=None, repr=False)  # cropped hand image

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
    Real-time hand detector built on YOLOv8.

    Parameters
    ----------
    model_path      : Path to .pt weights. Defaults to pretrained YOLOv8n.
                      For best sign-language accuracy, provide a model
                      fine-tuned on hand / gesture data.
    confidence      : Minimum detection confidence (0–1). Default 0.5.
    iou_threshold   : NMS IoU threshold. Default 0.45.
    device          : 'cpu', 'cuda', 'mps', or '' (auto). Default '' (auto).
    target_classes  : List of class names to keep (e.g. ['hand']).
                      None = keep all classes the model outputs.
    roi_padding     : Extra pixels added around each bbox when cropping ROI.
    """

    # Visual config (BGR colours)
    _BOX_COLOUR    = (0, 220, 120)
    _LABEL_COLOUR  = (255, 255, 255)
    _FPS_COLOUR    = (0, 200, 255)
    _BOX_THICKNESS = 2
    _FONT          = cv2.FONT_HERSHEY_SIMPLEX

    def __init__(
        self,
        model_path: str = "yolov8n.pt",
        confidence: float = 0.50,
        iou_threshold: float = 0.45,
        device: str = "",
        target_classes: Optional[list[str]] = None,
        roi_padding: int = 10,
    ):
        logger.info("Loading YOLO model: %s", model_path)
        self.model = YOLO(model_path)

        self.confidence    = confidence
        self.iou_threshold = iou_threshold
        self.device        = device
        self.target_classes = set(target_classes) if target_classes else None
        self.roi_padding   = roi_padding

        # FPS tracking
        self._fps: float = 0.0
        self._prev_tick: float = 0.0

        logger.info(
            "HandDetector ready | conf=%.2f  iou=%.2f  device=%s  classes=%s",
            confidence, iou_threshold,
            device or "auto",
            list(target_classes) if target_classes else "all",
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> list[HandDetection]:
        """
        Run inference on a single BGR frame.

        Returns
        -------
        List[HandDetection] — sorted largest-area first.
        """
        if frame is None or frame.size == 0:
            logger.warning("detect() received an empty frame — skipping.")
            return []

        results = self.model(
            frame,
            conf=self.confidence,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections: list[HandDetection] = []
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                cls_id   = int(box.cls[0])
                cls_name = result.names.get(cls_id, str(cls_id))

                # Filter to target classes if specified
                if self.target_classes and cls_name not in self.target_classes:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])

                roi = self._crop_roi(frame, x1, y1, x2, y2)

                detections.append(HandDetection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                    roi=roi,
                ))

        # Largest hand first — useful for single-hand gesture focus
        detections.sort(key=lambda d: d.area, reverse=True)
        return detections

    def draw(
        self,
        frame: np.ndarray,
        detections: list[HandDetection],
        show_fps: bool = True,
        show_confidence: bool = True,
    ) -> np.ndarray:
        """
        Draw bounding boxes, labels, and FPS onto *a copy* of the frame.
        Does NOT mutate the original.
        """
        canvas = frame.copy()

        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Bounding box
            cv2.rectangle(canvas, (x1, y1), (x2, y2),
                          self._BOX_COLOUR, self._BOX_THICKNESS)

            # Label background + text
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

            # Centre dot
            cv2.circle(canvas, det.center, 4, self._BOX_COLOUR, cv2.FILLED)

        if show_fps:
            fps_text = f"FPS: {self._fps:.1f}"
            cv2.putText(canvas, fps_text, (10, 30),
                        self._FONT, 0.9, self._FPS_COLOUR, 2, cv2.LINE_AA)

        return canvas

    def process_frame(
        self,
        frame: np.ndarray,
        draw: bool = True,
    ) -> tuple[list[HandDetection], np.ndarray]:
        """
        Convenience wrapper: detect + optionally draw in one call.
        Also updates the internal FPS counter.

        Returns
        -------
        (detections, annotated_frame)
        """
        self._tick()
        detections = self.detect(frame)
        annotated  = self.draw(frame, detections) if draw else frame
        return detections, annotated

    @property
    def fps(self) -> float:
        return self._fps

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _tick(self) -> None:
        now = time.perf_counter()
        if self._prev_tick:
            elapsed   = now - self._prev_tick
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / elapsed)   # smoothed EMA
        self._prev_tick = now

    def _crop_roi(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int,
    ) -> np.ndarray:
        """Return a padded crop of the detected region, clamped to frame bounds."""
        h, w = frame.shape[:2]
        px1 = max(0, x1 - self.roi_padding)
        py1 = max(0, y1 - self.roi_padding)
        px2 = min(w, x2 + self.roi_padding)
        py2 = min(h, y2 + self.roi_padding)
        return frame[py1:py2, px1:px2].copy()


# ── Stand-alone real-time demo ─────────────────────────────────────────────────
def run_webcam(
    model_path: str = "yolov8n.pt",
    camera_index: int = 0,
    confidence: float = 0.50,
    target_classes: Optional[list[str]] = None,
    window_name: str = "Sign Language — Hand Detection",
) -> None:
    """
    Launch a live webcam loop.
    Press  Q  or  ESC  to quit.
    Press  S  to save a snapshot.
    """
    detector = HandDetector(
        model_path=model_path,
        confidence=confidence,
        target_classes=target_classes,
    )

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        logger.error("Cannot open camera index %d", camera_index)
        return

    # Prefer higher resolution if supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_BUFFERSIZE,   1)   # reduce latency

    logger.info("Webcam opened. Press Q / ESC to quit, S to snapshot.")
    snapshot_idx = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                logger.warning("Failed to grab frame — retrying…")
                continue

            detections, canvas = detector.process_frame(frame)

            # Console log (optional — comment out for silent mode)
            if detections:
                for i, d in enumerate(detections):
                    logger.debug(
                        "Hand %d | class=%s  conf=%.2f  bbox=%s  center=%s",
                        i, d.class_name, d.confidence, d.bbox, d.center,
                    )

            cv2.imshow(window_name, canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):          # Q or ESC
                break
            elif key == ord("s"):              # snapshot
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
        description="Real-time YOLO hand detector for sign language."
    )
    parser.add_argument("--model",      default="yolov8n.pt",
                        help="Path to YOLO .pt weights")
    parser.add_argument("--camera",     type=int, default=0,
                        help="Camera device index (default: 0)")
    parser.add_argument("--conf",       type=float, default=0.50,
                        help="Detection confidence threshold (default: 0.50)")
    parser.add_argument("--classes",    nargs="*", default=None,
                        help="Class names to keep, e.g. --classes hand")
    args = parser.parse_args()

    run_webcam(
        model_path=args.model,
        camera_index=args.camera,
        confidence=args.conf,
        target_classes=args.classes,
    )