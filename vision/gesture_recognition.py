"""Static pose classification plus a debounced, hysteresis-based recognizer."""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from utils.config import GestureConfig
from vision.landmark_processor import HandLandmarks, extended_fingers, pinch_ratio


class GestureType(Enum):
    NONE = "none"                # no hand
    NEUTRAL = "neutral"          # hand visible, no command gesture
    PINCH = "pinch"              # thumb + index touching
    OPEN_PALM = "open palm"      # index..pinky extended
    THREE_FINGER = "three fingers"  # index, middle, ring extended; pinky curled


@dataclass(frozen=True)
class GestureReading:
    gesture: GestureType
    present: bool        # False while the hand is missing (grace period or lost)
    pinch_ratio: float


def classify_static(hand: HandLandmarks, cfg: GestureConfig, was_pinching: bool = False) -> GestureType:
    """Classify one frame. Categories are mutually exclusive by priority:
    pinch > open palm > three fingers > neutral. The thumb is ignored for the
    palm / three-finger poses because it is the least reliable landmark chain.
    """
    ratio = pinch_ratio(hand)
    if ratio < (cfg.pinch_off if was_pinching else cfg.pinch_on):
        return GestureType.PINCH
    _, index, middle, ring, pinky = extended_fingers(hand, cfg.extend_ratio)
    if index and middle and ring and pinky:
        return GestureType.OPEN_PALM
    if index and middle and ring:
        return GestureType.THREE_FINGER
    return GestureType.NEUTRAL


class GestureRecognizer:
    """Per-hand recognizer: hysteresis on the pinch, N-frame debounce on all changes,
    and a grace period so a one-frame detection dropout does not cancel a grab.
    """

    def __init__(self, cfg: GestureConfig) -> None:
        self._cfg = cfg
        self._confirmed = GestureType.NONE
        self._candidate = GestureType.NONE
        self._count = 0
        self._last_seen: Optional[float] = None
        self._last_ratio = math.inf

    @property
    def confirmed(self) -> GestureType:
        return self._confirmed

    def reset(self) -> None:
        self._confirmed = GestureType.NONE
        self._candidate = GestureType.NONE
        self._count = 0
        self._last_seen = None
        self._last_ratio = math.inf

    def update(self, hand: Optional[HandLandmarks], now: float) -> GestureReading:
        if hand is None:
            if self._last_seen is not None and now - self._last_seen <= self._cfg.lost_grace_s:
                return GestureReading(self._confirmed, False, self._last_ratio)
            self.reset()
            return GestureReading(GestureType.NONE, False, math.inf)

        self._last_seen = now
        self._last_ratio = pinch_ratio(hand)
        raw = classify_static(hand, self._cfg, was_pinching=self._confirmed is GestureType.PINCH)

        if raw is self._confirmed:
            self._candidate, self._count = GestureType.NONE, 0
        else:
            if raw is self._candidate:
                self._count += 1
            else:
                self._candidate, self._count = raw, 1
            if self._count >= self._cfg.confirm_frames:
                self._confirmed = raw
                self._candidate, self._count = GestureType.NONE, 0
        return GestureReading(self._confirmed, True, self._last_ratio)
