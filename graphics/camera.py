"""Perspective workspace camera: orbit, zoom, basis vectors and world->UI projection."""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
from ursina import Vec3, camera, window

from controls.gesture_mapping import CameraBasis


class WorkspaceCamera:
    """Orbit camera looking at ``target``; yaw 0 means the camera sits on -Z looking toward +Z."""

    def __init__(self, distance: float = 13.0, yaw: float = -20.0, pitch: float = 25.0,
                 target: Sequence[float] = (0.0, 0.8, 0.0), fov: float = 60.0) -> None:
        self.distance, self.yaw, self.pitch = distance, yaw, pitch
        self.target = np.asarray(target, dtype=float)
        camera.fov = fov
        self.apply()

    # ------------------------------------------------------------------ pose
    def position(self) -> np.ndarray:
        yaw, pitch = math.radians(self.yaw), math.radians(self.pitch)
        offset = self.distance * np.array([math.cos(pitch) * math.sin(yaw),
                                           math.sin(pitch),
                                           -math.cos(pitch) * math.cos(yaw)])
        return self.target + offset

    def apply(self) -> None:
        camera.position = Vec3(*self.position())
        camera.look_at(Vec3(*self.target))

    def orbit(self, d_yaw: float, d_pitch: float) -> None:
        self.yaw += d_yaw
        self.pitch = float(np.clip(self.pitch + d_pitch, -85.0, 85.0))
        self.apply()

    def zoom(self, factor: float) -> None:
        self.distance = float(np.clip(self.distance * factor, 4.0, 40.0))
        self.apply()

    # ------------------------------------------------------------------ math
    def basis(self) -> CameraBasis:
        forward = self.target - self.position()
        forward /= np.linalg.norm(forward)
        right = np.cross([0.0, 1.0, 0.0], forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)
        return CameraBasis(right=right, up=up, forward=forward)

    def project(self, world_xyz: Sequence[float]) -> Optional[Tuple[float, float]]:
        """World point -> UI coordinates (height -0.5..0.5, width +-0.5*aspect), or None if behind the camera.

        Uses the renderer's own lens so picking matches exactly what is drawn.
        """
        b = self.basis()
        v = np.asarray(world_xyz, dtype=float) - self.position()
        cam_space = (float(v @ b.right), float(v @ b.up), float(v @ b.forward))
        if cam_space[2] <= 0.01:
            return None
        from panda3d.core import Point2, Point3
        lens = getattr(camera, "lens", None)
        if lens is None:
            import builtins
            lens = builtins.base.camLens  # type: ignore[attr-defined]
        out = Point2()
        if not lens.project(Point3(*cam_space), out):
            return None
        return (out.x * 0.5 * window.aspect_ratio, out.y * 0.5)
