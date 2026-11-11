"""Math that turns normalized hand measurements into 3D transforms.

Pipeline
--------
1. ``normalize_to_plane``: image coords -> virtual interaction plane in [-1, 1]^2
   (the central part of the camera frame, y pointing up).
2. Relative motion on that plane (current - value at grab time) drives
   translation / rotation, so the absolute hand position never matters.
3. Two-hand distance ratio drives scale.

A standard webcam gives no metric depth, so translation happens in a plane
parallel to the screen. ``depth_offset`` offers an optional, noisy hand-size cue.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from utils.config import MappingConfig


@dataclass(frozen=True)
class CameraBasis:
    """World-space unit vectors of the viewing camera."""

    right: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    up: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))
    forward: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))


def two_hand_distance(p1: Sequence[float], p2: Sequence[float], aspect: float = 1.0) -> float:
    """Distance between two 2D points with x stretched by ``aspect``."""
    return float(np.hypot((p1[0] - p2[0]) * aspect, p1[1] - p2[1]))


class CoordinateMapper:
    """Maps hand measurements to workspace values using a shared :class:`MappingConfig`."""

    def __init__(self, cfg: MappingConfig) -> None:
        self.cfg = cfg

    def normalize_to_plane(self, xy: Sequence[float]) -> np.ndarray:
        """Normalized image (x right, y down) -> plane (x right, y up) clamped to [-1, 1]."""
        m = self.cfg.active_margin
        span = max(1.0 - 2.0 * m, 1e-6)
        nx = (float(xy[0]) - m) / span
        ny = (float(xy[1]) - m) / span
        return np.array([np.clip(nx * 2.0 - 1.0, -1.0, 1.0),
                         np.clip(1.0 - ny * 2.0, -1.0, 1.0)])

    @staticmethod
    def plane_to_ui(plane_xy: Sequence[float], ui_aspect: float) -> np.ndarray:
        """Plane [-1, 1]^2 -> UI coordinates (height spans -0.5..0.5, width +-0.5*aspect)."""
        return np.array([plane_xy[0] * 0.5 * ui_aspect, plane_xy[1] * 0.5])

    def translation(self, start_position: Sequence[float], plane_delta: Sequence[float],
                    basis: CameraBasis, depth: float = 0.0) -> np.ndarray:
        """New world position: start + sensitivity * (dx*right + dy*up) - depth*forward."""
        s = self.cfg.translate_sensitivity
        return (np.asarray(start_position, dtype=float)
                + s * (plane_delta[0] * basis.right + plane_delta[1] * basis.up)
                - depth * basis.forward)

    def rotation(self, start_rotation: Sequence[float], plane_delta: Sequence[float],
                 roll_delta_deg: float) -> np.ndarray:
        """New Euler angles (deg). Drag up -> pitch, drag right -> yaw, hand roll -> roll."""
        k = self.cfg.rotate_sensitivity_deg
        rx, ry, rz = (float(v) for v in start_rotation)
        return np.array([rx + plane_delta[1] * k,
                         ry - plane_delta[0] * k,
                         rz + roll_delta_deg * self.cfg.roll_gain])

    def scale(self, start_scale: float, start_distance: float, current_distance: float) -> float:
        """Uniform scale = start * (distance ratio) ** gain, clamped to [min_scale, max_scale]."""
        ratio = current_distance / max(start_distance, 1e-6)
        value = start_scale * ratio ** self.cfg.scale_gain
        return float(np.clip(value, self.cfg.min_scale, self.cfg.max_scale))

    def depth_offset(self, start_hand_size: float, current_hand_size: float) -> float:
        """Experimental: hand growing in the image (closer to the camera) -> object toward viewer."""
        if self.cfg.depth_gain == 0.0:
            return 0.0
        return self.cfg.depth_gain * (current_hand_size / max(start_hand_size, 1e-6) - 1.0)
