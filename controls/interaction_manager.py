"""Gesture state machine: hands in, high-level commands out.

The manager is deliberately free of any rendering code. It receives per-frame
hand landmarks plus a :class:`SceneSnapshot` (plain data describing the scene)
and returns :class:`Command` objects that an ``ObjectController`` applies.

States
------
IDLE      nothing is held
GRABBING  one hand pinches a selected object -> move or rotate (per mode)
SCALING   both hands pinch while an object is selected -> uniform scale

Discrete gestures (only evaluated in IDLE, edge-triggered, with cooldowns):
three-finger -> toggle MOVE/ROTATE mode; open palm -> deselect.
Pinching on an on-screen button presses it, which makes create/delete/reset
usable without a mouse.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from controls.gesture_mapping import CameraBasis, CoordinateMapper, two_hand_distance
from utils.config import AppConfig
from utils.smoothing import ExponentialSmoother
from vision.gesture_recognition import GestureRecognizer, GestureType
from vision.landmark_processor import (HandLandmarks, hand_roll_deg, hand_scale,
                                       pinch_point, wrap_angle_deg)

SLOTS = ("A", "B")
Vec3 = Tuple[float, float, float]
Vec2 = Tuple[float, float]


class Mode(Enum):
    MOVE = "Move"
    ROTATE = "Rotate"


class State(Enum):
    IDLE = "Idle"
    GRABBING = "Grabbing"
    SCALING = "Scaling"


class CommandKind(Enum):
    SELECT = "select"
    DESELECT = "deselect"
    TRANSFORM = "transform"
    MODE_CHANGED = "mode_changed"
    PRESS_BUTTON = "press_button"


@dataclass(frozen=True)
class Command:
    kind: CommandKind
    object_id: Optional[int] = None
    position: Optional[Vec3] = None
    rotation: Optional[Vec3] = None
    scale: Optional[float] = None
    mode: Optional[Mode] = None
    name: Optional[str] = None


@dataclass(frozen=True)
class ObjectState:
    object_id: int
    position: Vec3
    rotation: Vec3
    scale: float
    screen_xy: Vec2          # position in UI coordinates
    screen_radius: float     # approximate on-screen radius in UI units


@dataclass(frozen=True)
class UiTarget:
    name: str
    center: Vec2
    half_size: Vec2


@dataclass
class SceneSnapshot:
    objects: Sequence[ObjectState]
    ui_targets: Sequence[UiTarget] = ()
    basis: CameraBasis = CameraBasis()
    selected_id: Optional[int] = None
    ui_aspect: float = 16.0 / 9.0


def pick_ui_target(cursor: Sequence[float], targets: Sequence[UiTarget]) -> Optional[str]:
    """Name of the on-screen button under ``cursor`` (None if no button is hit)."""
    for t in targets:
        if abs(cursor[0] - t.center[0]) <= t.half_size[0] and abs(cursor[1] - t.center[1]) <= t.half_size[1]:
            return t.name
    return None


def pick_object(cursor: Sequence[float], objects: Sequence[ObjectState], min_radius: float) -> Optional[int]:
    """Id of the object whose on-screen disc contains ``cursor`` (nearest centre wins)."""
    best: Optional[int] = None
    best_d = math.inf
    for o in objects:
        d = math.hypot(cursor[0] - o.screen_xy[0], cursor[1] - o.screen_xy[1])
        if d <= max(o.screen_radius, min_radius) and d < best_d:
            best, best_d = o.object_id, d
    return best


@dataclass
class _Grab:
    slot: str
    object_id: int
    start_plane: np.ndarray
    start_position: Vec3
    start_rotation: Vec3
    start_roll: float
    start_hand_size: float


@dataclass
class _Scale:
    object_id: int
    start_distance: float
    start_scale: float


class InteractionManager:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mapper = CoordinateMapper(config.mapping)
        self.mode = Mode.MOVE
        self.state = State.IDLE
        self.gestures: Dict[str, GestureType] = {s: GestureType.NONE for s in SLOTS}
        self.cursors: Dict[str, np.ndarray] = {}  # UI coords of currently visible hands

        self._rec = {s: GestureRecognizer(config.gesture) for s in SLOTS}
        self._smooth = {s: ExponentialSmoother(config.mapping.smoothing_alpha) for s in SLOTS}
        self._plane: Dict[str, np.ndarray] = {}
        self._roll: Dict[str, float] = {}
        self._hsize: Dict[str, float] = {}
        self._aspect: Dict[str, float] = {}
        self._present = {s: False for s in SLOTS}
        self._was_present = {s: False for s in SLOTS}
        self._prev = {s: GestureType.NONE for s in SLOTS}
        self._consumed = {s: False for s in SLOTS}
        self._last_mode_switch = -math.inf
        self._last_deselect = -math.inf
        self._last_click = -math.inf
        self._grab: Optional[_Grab] = None
        self._scale: Optional[_Scale] = None

    # ------------------------------------------------------------------ public
    def reset(self) -> None:
        """Drop any held object (e.g. after the scene was reset)."""
        self.state = State.IDLE
        self._grab = None
        self._scale = None

    def describe_gestures(self) -> str:
        parts = [f"{s}: {self.gestures[s].value}" for s in SLOTS if self._present[s] or self.gestures[s] is not GestureType.NONE]
        return " | ".join(parts) if parts else "no hand"

    def update(self, hands: Dict[str, HandLandmarks], now: float, snap: SceneSnapshot) -> List[Command]:
        cmds: List[Command] = []
        self._ingest(hands, now, snap)
        objs = {o.object_id: o for o in snap.objects}
        pinching = [s for s in SLOTS if self.gestures[s] is GestureType.PINCH]
        for s in SLOTS:
            if s not in pinching:
                self._consumed[s] = False

        # A held object that no longer exists (deleted / scene reset) ends the interaction.
        if self._grab and self._grab.object_id not in objs:
            self.reset()
        if self._scale and self._scale.object_id not in objs:
            self.reset()

        # A hand that re-appeared jumped in image space: restart contexts to avoid a snap.
        if self.state is not State.IDLE and any(self._present[s] and not self._was_present[s] for s in SLOTS):
            self._restart_context(objs)

        self._step(pinching, objs, snap, now, cmds)
        self._emit_transform(objs, snap, cmds)

        for s in SLOTS:
            self._prev[s] = self.gestures[s]
            self._was_present[s] = self._present[s]
        return cmds

    # ----------------------------------------------------------------- internals
    def _ingest(self, hands: Dict[str, HandLandmarks], now: float, snap: SceneSnapshot) -> None:
        self.cursors = {}
        for s in SLOTS:
            hand = hands.get(s)
            reading = self._rec[s].update(hand, now)
            self.gestures[s] = reading.gesture
            self._present[s] = hand is not None
            if hand is None:
                continue
            smoother = self._smooth[s]
            smoother.alpha = self.config.mapping.smoothing_alpha
            if not self._was_present[s]:
                smoother.reset()
            self._plane[s] = smoother.update(self.mapper.normalize_to_plane(pinch_point(hand)))
            self._roll[s] = hand_roll_deg(hand)
            self._hsize[s] = hand_scale(hand)
            self._aspect[s] = hand.aspect
            self.cursors[s] = self.mapper.plane_to_ui(self._plane[s], snap.ui_aspect)

    def _rising(self, slot: str, gesture: GestureType) -> bool:
        return self.gestures[slot] is gesture and self._prev[slot] is not gesture

    def _step(self, pinching: List[str], objs: Dict[int, ObjectState], snap: SceneSnapshot,
              now: float, cmds: List[Command]) -> None:
        if self.state is State.GRABBING and self._grab:
            g = self._grab
            if g.slot not in pinching:
                self.reset()
                if pinching:  # the other hand is still pinching: keep holding with it
                    self._begin_grab(pinching[0], objs[g.object_id])
            elif len(pinching) == 2:
                self._begin_scale(objs[g.object_id])
            return
        if self.state is State.SCALING and self._scale:
            oid = self._scale.object_id
            if len(pinching) < 2:
                self.reset()
                if len(pinching) == 1:
                    self._begin_grab(pinching[0], objs[oid])
            return

        # ---- IDLE ----
        selected = snap.selected_id if snap.selected_id in objs else None
        if len(pinching) == 2 and selected is not None:
            self._begin_scale(objs[selected])
            return
        if pinching:
            for s in pinching:
                if self._consumed[s] or s not in self._plane:
                    continue
                self._consumed[s] = True  # one pick per pinch
                cursor = self.mapper.plane_to_ui(self._plane[s], snap.ui_aspect)
                button = pick_ui_target(cursor, snap.ui_targets)
                if button is not None:
                    if now - self._last_click >= self.config.gesture.click_cooldown_s:
                        self._last_click = now
                        cmds.append(Command(CommandKind.PRESS_BUTTON, name=button))
                    return
                oid = pick_object(cursor, snap.objects, self.config.mapping.pick_radius)
                if oid is not None:
                    cmds.append(Command(CommandKind.SELECT, object_id=oid))
                    self._begin_grab(s, objs[oid])
                return
            return

        cfg = self.config.gesture
        if any(self._rising(s, GestureType.THREE_FINGER) for s in SLOTS):
            if now - self._last_mode_switch >= cfg.mode_switch_cooldown_s:
                self._last_mode_switch = now
                self.mode = Mode.ROTATE if self.mode is Mode.MOVE else Mode.MOVE
                cmds.append(Command(CommandKind.MODE_CHANGED, mode=self.mode))
        elif selected is not None and any(self._rising(s, GestureType.OPEN_PALM) for s in SLOTS):
            if now - self._last_deselect >= cfg.deselect_cooldown_s:
                self._last_deselect = now
                cmds.append(Command(CommandKind.DESELECT))

    def _begin_grab(self, slot: str, obj: ObjectState) -> None:
        self.state = State.GRABBING
        self._scale = None
        self._grab = _Grab(slot, obj.object_id, self._plane[slot].copy(), obj.position,
                           obj.rotation, self._roll[slot], self._hsize[slot])

    def _begin_scale(self, obj: ObjectState) -> None:
        self.state = State.SCALING
        self._grab = None
        self._scale = _Scale(obj.object_id, self._hand_distance(), obj.scale)

    def _hand_distance(self) -> float:
        aspect = self._aspect.get("A", self._aspect.get("B", 4.0 / 3.0))
        return two_hand_distance(self._plane["A"], self._plane["B"], aspect)

    def _restart_context(self, objs: Dict[int, ObjectState]) -> None:
        if self.state is State.GRABBING and self._grab and self._grab.slot in self._plane:
            self._begin_grab(self._grab.slot, objs[self._grab.object_id])
        elif self.state is State.SCALING and self._scale and all(s in self._plane for s in SLOTS):
            self._begin_scale(objs[self._scale.object_id])

    def _emit_transform(self, objs: Dict[int, ObjectState], snap: SceneSnapshot, cmds: List[Command]) -> None:
        if self.state is State.GRABBING and self._grab:
            g = self._grab
            aspect = self._aspect.get(g.slot, 4.0 / 3.0)
            delta = (self._plane[g.slot] - g.start_plane) * np.array([aspect, 1.0])  # isotropic hand travel
            if self.mode is Mode.MOVE:
                depth = self.mapper.depth_offset(g.start_hand_size, self._hsize[g.slot])
                pos = self.mapper.translation(g.start_position, delta, snap.basis, depth)
                cmds.append(Command(CommandKind.TRANSFORM, object_id=g.object_id,
                                    position=tuple(float(v) for v in pos)))
            else:
                roll = wrap_angle_deg(self._roll[g.slot] - g.start_roll)
                rot = self.mapper.rotation(g.start_rotation, delta, roll)
                cmds.append(Command(CommandKind.TRANSFORM, object_id=g.object_id,
                                    rotation=tuple(float(v) for v in rot)))
        elif self.state is State.SCALING and self._scale:
            sc = self._scale
            value = self.mapper.scale(sc.start_scale, sc.start_distance, self._hand_distance())
            cmds.append(Command(CommandKind.TRANSFORM, object_id=sc.object_id, scale=value))
