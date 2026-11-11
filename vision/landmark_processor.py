"""Pure-NumPy geometry on MediaPipe's 21 hand landmarks.

Nothing here imports OpenCV, MediaPipe or Ursina, so every function can be
unit-tested with synthetic landmarks.

Conventions
-----------
* ``points`` is a (21, 3) array of MediaPipe *normalized* coordinates:
  x, y in [0, 1] (origin top-left, y grows downward), z is relative depth.
* Normalized x and y are fractions of different lengths (width vs height), so
  all distances are computed in *metric* space where x is multiplied by the
  image aspect ratio (width / height). This keeps gestures rotation- and
  aspect-ratio independent.
* Distances that decide gestures are divided by the hand size (wrist -> middle
  finger MCP), which makes thresholds independent of how far the hand is from
  the camera.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

NUM_LANDMARKS = 21
WRIST = 0
THUMB_IP, THUMB_TIP = 3, 4
INDEX_MCP, INDEX_PIP, INDEX_TIP = 5, 6, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_TIP = 9, 10, 12
RING_MCP, RING_PIP, RING_TIP = 13, 14, 16
PINKY_MCP, PINKY_PIP, PINKY_TIP = 17, 18, 20
FINGER_NAMES = ("thumb", "index", "middle", "ring", "pinky")

HAND_CONNECTIONS: Tuple[Tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(eq=False)
class HandLandmarks:
    """One detected hand: 21 normalized landmarks plus image aspect ratio."""

    points: np.ndarray
    label: str = "Unknown"
    aspect: float = 4.0 / 3.0

    def __post_init__(self) -> None:
        pts = np.asarray(self.points, dtype=float)
        if pts.shape != (NUM_LANDMARKS, 3):
            raise ValueError(f"expected ({NUM_LANDMARKS}, 3) landmarks, got {pts.shape}")
        self.points = pts


def metric_xy(hand: HandLandmarks) -> np.ndarray:
    """(21, 2) landmark positions with x stretched by the aspect ratio."""
    xy = hand.points[:, :2].copy()
    xy[:, 0] *= hand.aspect
    return xy


def _dist(xy: np.ndarray, a: int, b: int) -> float:
    return float(np.linalg.norm(xy[a] - xy[b]))


def hand_scale(hand: HandLandmarks) -> float:
    """Hand size: wrist -> middle-finger MCP distance (never zero)."""
    return max(_dist(metric_xy(hand), WRIST, MIDDLE_MCP), 1e-6)


def pinch_ratio(hand: HandLandmarks) -> float:
    """Thumb-tip to index-tip distance divided by hand size."""
    xy = metric_xy(hand)
    return _dist(xy, THUMB_TIP, INDEX_TIP) / max(_dist(xy, WRIST, MIDDLE_MCP), 1e-6)


def pinch_point(hand: HandLandmarks) -> np.ndarray:
    """Normalized (x, y) midpoint between thumb tip and index tip."""
    return 0.5 * (hand.points[THUMB_TIP, :2] + hand.points[INDEX_TIP, :2])


def extended_fingers(hand: HandLandmarks, extend_ratio: float = 1.05) -> Tuple[bool, bool, bool, bool, bool]:
    """Return (thumb, index, middle, ring, pinky) extension flags.

    A finger is extended when its tip is farther from the wrist than its PIP
    joint (times ``extend_ratio``). Comparing distances instead of y values
    keeps the test valid when the hand is rotated. The thumb is extended when
    its tip is farther from the pinky MCP than the thumb IP joint is.
    """
    xy = metric_xy(hand)
    thumb = _dist(xy, THUMB_TIP, PINKY_MCP) > extend_ratio * _dist(xy, THUMB_IP, PINKY_MCP)
    flags = [
        _dist(xy, WRIST, tip) > extend_ratio * _dist(xy, WRIST, pip)
        for tip, pip in ((INDEX_TIP, INDEX_PIP), (MIDDLE_TIP, MIDDLE_PIP),
                         (RING_TIP, RING_PIP), (PINKY_TIP, PINKY_PIP))
    ]
    return (bool(thumb), *map(bool, flags))  # type: ignore[return-value]


def wrap_angle_deg(angle: float) -> float:
    """Wrap an angle to [-180, 180)."""
    return (angle + 180.0) % 360.0 - 180.0


def hand_roll_deg(hand: HandLandmarks) -> float:
    """In-plane hand roll in degrees: 0 = fingers up, positive = counter-clockwise on screen."""
    xy = metric_xy(hand)
    v = xy[MIDDLE_MCP] - xy[WRIST]
    return wrap_angle_deg(math.degrees(math.atan2(-v[1], v[0])) - 90.0)


class HandAssigner:
    """Keeps hand identities ("A"/"B") stable across frames.

    MediaPipe's handedness label can flicker, so hands are matched to the slot
    whose last wrist position is closest instead. A hand that disappears and
    returns near its old position re-acquires the same slot.
    """

    SLOTS = ("A", "B")

    def __init__(self) -> None:
        self._last: Dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self._last.clear()

    def assign(self, hands: Sequence[HandLandmarks]) -> Dict[str, HandLandmarks]:
        hands = list(hands)[:2]
        result: Dict[str, HandLandmarks] = {}
        pairs: List[Tuple[float, str, int]] = []
        for slot, pos in self._last.items():
            for i, h in enumerate(hands):
                pairs.append((float(np.linalg.norm(h.points[WRIST, :2] - pos)), slot, i))
        used_hands = set()
        for _, slot, i in sorted(pairs):
            if slot in result or i in used_hands:
                continue
            result[slot] = hands[i]
            used_hands.add(i)
        for i, h in enumerate(hands):
            if i in used_hands:
                continue
            free = next(s for s in self.SLOTS if s not in result)
            result[free] = h
        for slot, h in result.items():
            self._last[slot] = h.points[WRIST, :2].copy()
        return result
