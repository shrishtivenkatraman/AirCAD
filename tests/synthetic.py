"""Synthetic hand landmarks so gesture logic can be tested without a webcam."""
from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np

from vision.landmark_processor import HandLandmarks, pinch_point

# Open-hand template in normalized image coords (wrist at the bottom, fingers up).
_BASE = {0: (0.50, 0.80), 1: (0.44, 0.76), 2: (0.40, 0.70), 3: (0.37, 0.65),
         5: (0.44, 0.60), 9: (0.50, 0.58), 13: (0.56, 0.60), 17: (0.62, 0.64)}
_FINGER_MCP = {"index": 5, "middle": 9, "ring": 13, "pinky": 17}
_EXT = (-0.08, -0.14, -0.20)    # pip, dip, tip offsets (y) when extended
_CURL = (-0.07, -0.02, 0.03)    # ... when curled toward the palm


def make_hand(thumb=False, index=False, middle=False, ring=False, pinky=False,
              pinch_ratio: Optional[float] = None, aspect: float = 4 / 3,
              roll_deg: float = 0.0, offset: Tuple[float, float] = (0.0, 0.0),
              label: str = "Right") -> HandLandmarks:
    """Build a hand with the given fingers extended.

    ``pinch_ratio`` places the thumb tip that many hand-sizes from the index tip.
    ``roll_deg`` rotates the hand counter-clockwise (on screen) about the wrist.
    """
    pts = np.zeros((21, 3))
    for i, (x, y) in _BASE.items():
        pts[i, :2] = (x, y)
    flags = {"index": index, "middle": middle, "ring": ring, "pinky": pinky}
    for name, mcp in _FINGER_MCP.items():
        x, y = _BASE[mcp]
        for k, dy in enumerate(_EXT if flags[name] else _CURL):
            pts[mcp + 1 + k, :2] = (x, y + dy)
    pts[4, :2] = (0.34, 0.61) if thumb else (0.50, 0.70)  # curled thumb rests across the palm
    if pinch_ratio is not None:
        hand_size = 0.22  # wrist -> middle MCP in the template (x unchanged)
        pts[4, :2] = pts[8, :2] + (pinch_ratio * hand_size / aspect, 0.0)
    if roll_deg:
        c, s = math.cos(math.radians(roll_deg)), math.sin(math.radians(roll_deg))
        wrist = pts[0, :2].copy()
        for i in range(21):
            dx, dy = (pts[i, 0] - wrist[0]) * aspect, pts[i, 1] - wrist[1]
            pts[i, 0] = wrist[0] + (c * dx + s * dy) / aspect
            pts[i, 1] = wrist[1] + (-s * dx + c * dy)
    pts[:, 0] += offset[0]
    pts[:, 1] += offset[1]
    return HandLandmarks(pts, label, aspect)


def make_hand_at(target_xy: Tuple[float, float], **kwargs) -> HandLandmarks:
    """Like :func:`make_hand` but shifted so the pinch point sits at ``target_xy``."""
    h = make_hand(**kwargs)
    p = pinch_point(h)
    return make_hand(offset=(target_xy[0] - p[0], target_xy[1] - p[1]), **kwargs)
