"""Webcam capture and MediaPipe hand detection (the I/O side of the vision package).

Reused from the original drawing app: DirectShow camera opening on Windows,
automatic hand-model download, mirrored selfie view and skeleton drawing.
New: a background worker thread so camera + inference never block rendering.
"""
from __future__ import annotations

import os
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from typing import List, Optional

import cv2
import numpy as np

from utils.config import CameraConfig
from vision.landmark_processor import HAND_CONNECTIONS, HandLandmarks

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/1/hand_landmarker.task")


class CameraError(RuntimeError):
    """The webcam could not be opened or read."""


class ModelError(RuntimeError):
    """The hand-landmark model is missing and could not be downloaded."""


def ensure_model(path: str = MODEL_PATH, url: str = MODEL_URL) -> str:
    """Return the model path, downloading the ~8 MB file on first run."""
    if os.path.exists(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print("Downloading hand tracking model (one-time, ~8 MB)...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as exc:  # network errors vary widely
        if os.path.exists(path):
            os.remove(path)
        raise ModelError(
            f"Could not download the hand model ({exc}). Download it manually from\n"
            f"  {url}\nand save it as\n  {path}") from exc
    return path


def open_camera(cfg: CameraConfig) -> cv2.VideoCapture:
    """Open the webcam; DirectShow is faster and more reliable on Windows."""
    backend = cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY
    cap = cv2.VideoCapture(cfg.index, backend)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(cfg.index)
    if not cap.isOpened():
        cap.release()
        raise CameraError("Could not open the webcam. Close other apps using it, check "
                          "Windows camera privacy settings, or try --camera-index 1.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.height)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


class HandTracker:
    """MediaPipe Tasks HandLandmarker wrapper returning :class:`HandLandmarks`."""

    def __init__(self, model_path: Optional[str] = None, max_hands: int = 2) -> None:
        from mediapipe.tasks.python import BaseOptions, vision  # deferred: heavy import

        path = ensure_model(model_path or MODEL_PATH)
        with open(path, "rb") as f:  # buffer avoids Windows non-ASCII path problems
            model_bytes = f.read()
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_buffer=model_bytes),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def detect(self, frame_bgr: np.ndarray) -> List[HandLandmarks]:
        import mediapipe as mp

        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts = max(int(time.monotonic() * 1000), self._last_ts + 1)  # must strictly increase
        self._last_ts = ts
        result = self._landmarker.detect_for_video(image, ts)

        hands: List[HandLandmarks] = []
        for i, lms in enumerate(result.hand_landmarks or []):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
            label = "Unknown"
            if result.handedness and i < len(result.handedness) and result.handedness[i]:
                label = result.handedness[i][0].category_name
            hands.append(HandLandmarks(pts, label, w / h))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def draw_hand_skeleton(frame_bgr: np.ndarray, hand: HandLandmarks) -> None:
    """Draw the 21-point skeleton in place."""
    h, w = frame_bgr.shape[:2]
    px = [(int(x * w), int(y * h)) for x, y in hand.points[:, :2]]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame_bgr, px[a], px[b], (0, 200, 0), 2)
    for p in px:
        cv2.circle(frame_bgr, p, 4, (0, 0, 255), cv2.FILLED)


@dataclass
class VisionFrame:
    frame_id: int
    timestamp: float                 # time.monotonic() when the frame was processed
    hands: List[HandLandmarks]
    preview_rgb: np.ndarray          # small annotated RGB thumbnail for the UI


class VisionWorker:
    """Runs capture + detection on a daemon thread; the UI polls :meth:`latest`."""

    def __init__(self, cfg: CameraConfig) -> None:
        self._cfg = cfg
        self._cap: Optional[cv2.VideoCapture] = None
        self._tracker: Optional[HandTracker] = None
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[VisionFrame] = None
        self.error: Optional[str] = None

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> bool:
        """Open camera + model and start the thread. Returns False and sets ``error`` on failure."""
        try:
            self._tracker = HandTracker()
            self._cap = open_camera(self._cfg)
        except (CameraError, ModelError) as exc:
            self.error = str(exc)
            self._release()
            return False
        except Exception as exc:
            self.error = f"Vision start-up failed: {exc}"
            self._release()
            return False
        self._thread = threading.Thread(target=self._loop, name="aircad-vision", daemon=True)
        self._thread.start()
        return True

    def latest(self) -> Optional[VisionFrame]:
        with self._lock:
            return self._latest

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        self._release()

    def _release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._tracker is not None:
            try:
                self._tracker.close()
            except Exception:
                pass
            self._tracker = None

    def _loop(self) -> None:
        frame_id = 0
        try:
            while not self._stop.is_set():
                ok, frame = self._cap.read()
                if not ok:
                    self.error = "Lost connection to the webcam."
                    break
                if self._cfg.mirror:
                    frame = cv2.flip(frame, 1)
                hands = self._tracker.detect(frame)
                for hand in hands:
                    draw_hand_skeleton(frame, hand)
                scale = self._cfg.preview_width / frame.shape[1]
                thumb = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                preview = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                frame_id += 1
                with self._lock:
                    self._latest = VisionFrame(frame_id, time.monotonic(), hands, preview)
        except Exception as exc:
            self.error = f"Vision thread stopped: {exc}"
