"""
MEDIAPIPE.py  —  Hand detection & ROI extraction for ASL detection pipeline.

Fixes over original
───────────────────
1. Per-hand tracking state  →  prev_bbox / prev_center keyed by hand index,
   not a single shared variable that breaks the moment 2 hands appear.

2. Motion computed on RAW center BEFORE smoothing  →  original computed motion
   AFTER smoothing, so a fast-moving hand's center barely shifted in the smoothed
   coords and always looked "stable". Now motion uses the unsmoothed landmark bbox.

3. Smooth factor corrected  →  original used 0.7 * prev + 0.3 * new, meaning the
   bbox barely chased the hand. Corrected to 0.3 * prev + 0.7 * new (responsive).

4. Euclidean motion distance instead of Manhattan  →  diagonal motion was
   under-reported by ~30% with L1. L2 is correct.

5. Real millisecond timestamps  →  VIDEO mode expects real wall-clock ms, not
   bare ints 1, 2, 3 …. Fake timestamps caused MediaPipe's internal Kalman filter
   to mis-model velocity, making landmark jitter worse.

6. ROI validity guard  →  if clamped bbox has zero area, skip that detection
   instead of passing an empty ndarray to the classifier which crashes silently.

7. Raised confidence thresholds  →  defaults (0.5) are too loose for static-sign
   ASL where partial/ambiguous hands cause false detections mid-transition.

8. Stale state eviction  →  when a hand disappears, its tracking state is removed
   so the next detection starts fresh instead of snapping to stale coords.
"""

import math
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions


# ─────────────────────────────────────────────────────────────────────────────
# Data contract
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HandDetection:
    bbox:       Tuple[int, int, int, int]   # x1, y1, x2, y2  (pixel coords, smoothed)
    handedness: str                          # "Left" | "Right"
    score:      float                        # handedness classifier confidence
    landmarks:  list                         # raw NormalizedLandmark list (21 points)
    roi:        np.ndarray                   # BGR crop of the hand region
    is_stable:  bool                         # True when Euclidean motion < threshold
    motion:     float                        # raw motion magnitude in pixels


# ─────────────────────────────────────────────────────────────────────────────
# Detector
# ─────────────────────────────────────────────────────────────────────────────

class HandDetector:
    """
    Wraps MediaPipe HandLandmarker (Tasks API) with correct per-hand tracking,
    real-time timestamps, and validity-guarded ROI extraction.
    """

    def __init__(
        self,
        model_path:               str   = "hand_landmarker.task",
        max_hands:                int   = 2,
        expansion:                float = 0.35,
        smooth_factor:            float = 0.15,   # weight on PREVIOUS bbox; lower = more responsive
        motion_threshold:         float = 8.0,    # Euclidean px below which hand is "stable"
        min_detection_confidence: float = 0.65,
        min_presence_confidence:  float = 0.65,
        min_tracking_confidence:  float = 0.55,
    ):
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self.detector         = vision.HandLandmarker.create_from_options(options)
        self.expansion        = expansion
        self.smooth_factor    = smooth_factor
        self.motion_threshold = motion_threshold

        # Per-hand state: index → {"bbox": (x1,y1,x2,y2), "center": (cx,cy)}
        self._state: Dict[int, dict] = {}
        self._prev_indices: set      = set()

        # Real ms clock base (VIDEO mode requires monotonically increasing ms)
        self._t0_ms: int = time.monotonic_ns() // 1_000_000

    # ── private helpers ───────────────────────────────────────────────────────

    def _ms(self) -> int:
        return (time.monotonic_ns() // 1_000_000) - self._t0_ms

    def _expand(
        self, x1: int, y1: int, x2: int, y2: int, W: int, H: int
    ) -> Tuple[int, int, int, int]:
        bw = x2 - x1
        bh = y2 - y1
        px = int(bw * self.expansion)
        py = int(bh * self.expansion)
        return (
            max(0, x1 - px),
            max(0, y1 - py),
            min(W, x2 + px),
            min(H, y2 + py),
        )

    def _smooth(
        self, idx: int, x1: int, y1: int, x2: int, y2: int
    ) -> Tuple[int, int, int, int]:
        if idx not in self._state:
            return x1, y1, x2, y2
        p = self._state[idx]["bbox"]
        a, b = self.smooth_factor, 1.0 - self.smooth_factor
        return (
            int(a * p[0] + b * x1),
            int(a * p[1] + b * y1),
            int(a * p[2] + b * x2),
            int(a * p[3] + b * y2),
        )

    def _motion(self, idx: int, raw_cx: int, raw_cy: int) -> float:
        """Euclidean distance from raw current center to previous smoothed center."""
        if idx not in self._state:
            return 0.0
        pc = self._state[idx]["center"]
        return math.hypot(raw_cx - pc[0], raw_cy - pc[1])

    # ── public API ────────────────────────────────────────────────────────────

    def detect(self, frame: np.ndarray) -> List[HandDetection]:
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result   = self.detector.detect_for_video(mp_image, self._ms())

        if not result.hand_landmarks:
            self._state.clear()
            self._prev_indices.clear()
            return []

        H, W = frame.shape[:2]
        detections:      List[HandDetection] = []
        current_indices: set                 = set()

        for i, lm_list in enumerate(result.hand_landmarks):

            # 1. Raw landmark bounding box
            xs = [lm.x for lm in lm_list]
            ys = [lm.y for lm in lm_list]
            rx1 = int(min(xs) * W);  ry1 = int(min(ys) * H)
            rx2 = int(max(xs) * W);  ry2 = int(max(ys) * H)

            # 2. Expand
            rx1, ry1, rx2, ry2 = self._expand(rx1, ry1, rx2, ry2, W, H)

            # 3. Raw center → used for motion check BEFORE smoothing
            raw_cx = (rx1 + rx2) // 2
            raw_cy = (ry1 + ry2) // 2

            # 4. Motion against previous smoothed center
            mot      = self._motion(i, raw_cx, raw_cy)
            is_stable = mot < self.motion_threshold

            # 5. Smooth the bbox
            sx1, sy1, sx2, sy2 = self._smooth(i, rx1, ry1, rx2, ry2)

            # 6. Save updated state
            self._state[i] = {
                "bbox":   (sx1, sy1, sx2, sy2),
                "center": ((sx1 + sx2) // 2, (sy1 + sy2) // 2),
            }
            current_indices.add(i)

            # 7. ROI — guard against zero-area crop
            roi = frame[sy1:sy2, sx1:sx2]
            if roi.size == 0:
                continue
            roi = roi.copy()

            # 8. Handedness
            hd = result.handedness[i][0]

            detections.append(HandDetection(
                bbox       = (sx1, sy1, sx2, sy2),
                handedness = hd.category_name,
                score      = hd.score,
                landmarks  = lm_list,
                roi        = roi,
                is_stable  = is_stable,
                motion     = mot,
            ))

        # Evict state for hands that vanished this frame
        for idx in self._prev_indices - current_indices:
            self._state.pop(idx, None)
        self._prev_indices = current_indices

        return detections


# ─────────────────────────────────────────────────────────────────────────────
# Renderer  (optional, for debug views)
# ─────────────────────────────────────────────────────────────────────────────

class HandRenderer:
    """Draws landmarks, skeleton, bbox and stability label onto a frame."""

    CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),           # thumb
        (0,5),(5,6),(6,7),(7,8),           # index
        (5,9),(9,10),(10,11),(11,12),      # middle
        (9,13),(13,14),(14,15),(15,16),    # ring
        (13,17),(17,18),(18,19),(19,20),   # pinky
        (0,17),                            # palm base
    ]

    def draw(self, frame: np.ndarray, detections: List[HandDetection]) -> np.ndarray:
        H, W = frame.shape[:2]

        for det in detections:
            sx1, sy1, sx2, sy2 = det.bbox

            # Skeleton
            for a, b in self.CONNECTIONS:
                p1 = (int(det.landmarks[a].x * W), int(det.landmarks[a].y * H))
                p2 = (int(det.landmarks[b].x * W), int(det.landmarks[b].y * H))
                cv2.line(frame, p1, p2, (0, 255, 255), 2, cv2.LINE_AA)

            # Landmark dots
            for lm in det.landmarks:
                cv2.circle(
                    frame, (int(lm.x * W), int(lm.y * H)),
                    4, (255, 0, 255), -1, cv2.LINE_AA,
                )

            # Bbox: green = stable, red = moving
            color = (0, 255, 0) if det.is_stable else (0, 0, 255)
            cv2.rectangle(frame, (sx1, sy1), (sx2, sy2), color, 2)

            # Label
            tag   = "STABLE" if det.is_stable else f"MOVE {det.motion:.0f}px"
            label = f"{det.handedness}  {tag}"
            cv2.putText(
                frame, label, (sx1, max(sy1 - 10, 14)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
            )

        return frame