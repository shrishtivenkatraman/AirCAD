"""AirCAD application: wires vision -> gestures -> commands -> 3D scene -> UI."""
from __future__ import annotations

import time
from typing import Dict, Optional

import numpy as np
from ursina import Entity, Ursina, application, camera, color, held_keys, window

from controls.interaction_manager import InteractionManager
from controls.object_controller import ObjectController
from graphics.camera import WorkspaceCamera
from graphics.scene import Scene
from graphics.ui import AirCADUI
from utils.config import AppConfig
from vision.gesture_recognition import GestureType
from vision.hand_tracker import VisionWorker
from vision.landmark_processor import HandAssigner

STALE_AFTER_S = 0.3   # no new vision frame for this long -> treat as "no hands"


class AirCADApp(Entity):
    """Single driver entity: Ursina calls ``update`` every render frame and ``input`` on key presses."""

    def __init__(self, config: AppConfig, worker: Optional[VisionWorker]) -> None:
        super().__init__()
        self.config = config
        self.worker = worker
        self.workspace = Scene()
        self.cam = WorkspaceCamera()
        self.controller = ObjectController(self.workspace, self.cam)
        self.manager = InteractionManager(config)
        self.assigner = HandAssigner()
        self.controller.actions = {
            "cube": lambda: self.controller.create("cube"),
            "sphere": lambda: self.controller.create("sphere"),
            "cylinder": lambda: self.controller.create("cylinder"),
            "delete": self.controller.delete_selected,
            "reset": self._reset,
        }
        self.hud = AirCADUI(config, self.controller.actions, camera_available=worker is not None)
        self._cursor_dots: Dict[str, Entity] = {}
        self._debug_dots: Dict[int, Entity] = {}
        self._debug = False
        self._last_frame_id = -1
        self._last_frame_time = time.monotonic()
        self._last_preview = 0.0
        self._last_tick = time.perf_counter()

    # ---------------------------------------------------------------- actions
    def _reset(self) -> None:
        self.controller.reset_scene()
        self.manager.reset()

    # ------------------------------------------------------------------ frame
    def update(self) -> None:
        now = time.monotonic()
        tick = time.perf_counter()
        dt, self._last_tick = tick - self._last_tick, tick

        self._poll_vision(now)
        self._camera_keys(dt)

        error = ""
        if self.worker is not None and self.worker.error:
            error = self.worker.error
        elif self.worker is None:
            error = "Camera disabled: use keys 1/2/3, Delete, R. Arrow keys orbit, scroll zooms."
        self.hud.set_status(self.manager.mode.value, self.manager.state.value,
                            self.manager.describe_gestures(), self.workspace.selected_description(), error)
        self._draw_cursors()
        if self._debug:
            self._draw_debug_dots()

    def _poll_vision(self, now: float) -> None:
        if self.worker is None:
            return
        frame = self.worker.latest()
        if frame is not None and frame.frame_id != self._last_frame_id:
            self._last_frame_id = frame.frame_id
            self._last_frame_time = now
            self._run_gestures(self.assigner.assign(frame.hands), now)
            if now - self._last_preview >= 1.0 / self.config.camera.preview_fps:
                self._last_preview = now
                self.hud.set_preview(frame.preview_rgb)
        elif now - self._last_frame_time > STALE_AFTER_S:
            self._run_gestures({}, now)     # vision stalled: behave as if hands left the view

    def _run_gestures(self, hands, now: float) -> None:
        snap = self.controller.snapshot(self.hud.ui_targets(), window.aspect_ratio)
        self.controller.apply(self.manager.update(hands, now, snap))

    def _camera_keys(self, dt: float) -> None:
        yaw = (held_keys["right arrow"] - held_keys["left arrow"]) * 60.0 * dt
        pitch = (held_keys["up arrow"] - held_keys["down arrow"]) * 40.0 * dt
        if yaw or pitch:
            self.cam.orbit(yaw, pitch)

    def input(self, key: str) -> None:
        if key in ("1", "2", "3"):
            self.controller.create(("cube", "sphere", "cylinder")[int(key) - 1])
        elif key in ("delete", "backspace"):
            self.controller.delete_selected()
        elif key == "r":
            self._reset()
        elif key == "scroll up":
            self.cam.zoom(0.92)
        elif key == "scroll down":
            self.cam.zoom(1.08)
        elif key == "f1":
            self._debug = not self._debug
            if not self._debug:
                for dot in self._debug_dots.values():
                    dot.enabled = False
        elif key == "escape":
            application.quit()

    # ------------------------------------------------------------ overlays
    def _draw_cursors(self) -> None:
        for slot in ("A", "B"):
            dot = self._cursor_dots.get(slot)
            if dot is None:
                dot = Entity(parent=camera.ui, model="circle", scale=0.025, z=-1)
                self._cursor_dots[slot] = dot
            cursor = self.manager.cursors.get(slot)
            dot.enabled = cursor is not None
            if cursor is not None:
                dot.position = (float(cursor[0]), float(cursor[1]), -1)
                dot.color = color.lime if self.manager.gestures[slot] is GestureType.PINCH else color.white

    def _draw_debug_dots(self) -> None:
        """F1: red dots at each object's projected screen position (verifies picking alignment)."""
        snap = self.controller.snapshot([], window.aspect_ratio)
        for o in snap.objects:
            dot = self._debug_dots.get(o.object_id)
            if dot is None:
                dot = Entity(parent=camera.ui, model="circle", color=color.red, scale=0.012, z=-2)
                self._debug_dots[o.object_id] = dot
            dot.enabled = True
            dot.position = (o.screen_xy[0], o.screen_xy[1], -2)


def run(config: AppConfig, worker: Optional[VisionWorker]) -> None:
    app = Ursina(title=config.window_title, borderless=False)
    window.size = (1280, 720)
    window.color = color.rgb(28, 30, 36)
    if hasattr(window, "exit_button"):
        window.exit_button.visible = False
    AirCADApp(config, worker)
    try:
        app.run()
    finally:
        if worker is not None:
            worker.stop()
