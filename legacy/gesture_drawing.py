"""The original Hand Gesture Drawing App, now running on AirCAD's shared vision modules.

Index finger only -> draw. Any other finger -> pen up.   C clear | S save | Q quit
Run with:  python main.py --legacy-draw
"""
from __future__ import annotations

import os
from datetime import datetime

import cv2
import numpy as np

from utils.config import CameraConfig
from vision.hand_tracker import BASE_DIR, CameraError, HandTracker, ModelError, draw_hand_skeleton, open_camera
from vision.landmark_processor import extended_fingers

BRUSH_COLOR = (255, 0, 255)  # BGR
BRUSH_THICKNESS = 6
SMOOTHING = 0.5
WINDOW = "Hand Gesture Drawing App"


def _save(canvas: np.ndarray) -> str:
    folder = os.path.join(BASE_DIR, "drawings")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"drawing_{datetime.now():%Y%m%d_%H%M%S}.png")
    out = np.full_like(canvas, 255)
    mask = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY) > 0
    out[mask] = canvas[mask]
    cv2.imwrite(path, out)
    return path


def _overlay(frame: np.ndarray, canvas: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    return cv2.add(cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(mask)),
                   cv2.bitwise_and(canvas, canvas, mask=mask))


def main(cam_cfg: CameraConfig | None = None) -> None:
    cam_cfg = cam_cfg or CameraConfig(width=1280, height=720)
    try:
        tracker = HandTracker(max_hands=1)
        cap = open_camera(cam_cfg)
    except (CameraError, ModelError) as exc:
        print(f"ERROR: {exc}")
        return

    ok, frame = cap.read()
    if not ok:
        print("ERROR: Could not read from the webcam.")
        cap.release()
        tracker.close()
        return
    h, w = frame.shape[:2]
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    prev = smooth = None
    note, note_frames = "", 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)
            hands = tracker.detect(frame)
            mode = "NO HAND"
            if hands:
                hand = hands[0]
                draw_hand_skeleton(frame, hand)
                thumb, index, middle, ring, pinky = extended_fingers(hand)
                tip = hand.points[8]
                x, y = int(tip[0] * w), int(tip[1] * h)
                smooth = (x, y) if smooth is None else (int(SMOOTHING * smooth[0] + (1 - SMOOTHING) * x),
                                                         int(SMOOTHING * smooth[1] + (1 - SMOOTHING) * y))
                if index and not (thumb or middle or ring or pinky):
                    mode = "DRAWING"
                    if prev is not None:
                        cv2.line(canvas, prev, smooth, BRUSH_COLOR, BRUSH_THICKNESS)
                    prev = smooth
                    cv2.circle(frame, smooth, 10, BRUSH_COLOR, cv2.FILLED)
                else:
                    mode, prev = "PEN UP", None
            else:
                prev = smooth = None
            out = _overlay(frame, canvas)
            cv2.putText(out, f"Mode: {mode}", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(out, "C: Clear | S: Save | Q: Quit", (15, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            if note_frames > 0:
                cv2.putText(out, note, (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                note_frames -= 1
            cv2.imshow(WINDOW, out)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            if key in (ord("c"), ord("C")):
                canvas[:] = 0
                prev = None
                note, note_frames = "Canvas cleared", 45
            if key in (ord("s"), ord("S")):
                print(f"Drawing saved to {_save(canvas)}")
                note, note_frames = "Drawing saved", 90
            if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
