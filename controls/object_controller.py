"""Bridges the pure interaction logic and the 3D scene.

Duck-typed on purpose (no Ursina import) so it can be unit-tested with fakes.

``scene`` must provide: ``object_states()`` -> iterable of
``(id, position, rotation, scale)``, ``selected_id``, ``select(id)``,
``deselect()``, ``set_transform(id, position, rotation, scale)``,
``spawn(kind)``, ``delete(id)``, ``reset()``.

``camera`` must provide: ``basis()`` -> CameraBasis and
``project(world_xyz)`` -> (x, y) in UI coordinates or None.
"""
from __future__ import annotations

from typing import Callable, Dict, Iterable, List, Sequence

import numpy as np

from controls.interaction_manager import (Command, CommandKind, ObjectState,
                                          SceneSnapshot, UiTarget)


class ObjectController:
    def __init__(self, scene, camera) -> None:
        self.scene = scene
        self.camera = camera
        self.actions: Dict[str, Callable[[], None]] = {}

    # ---------------------------------------------------------------- snapshot
    def snapshot(self, ui_targets: Sequence[UiTarget], ui_aspect: float) -> SceneSnapshot:
        basis = self.camera.basis()
        states: List[ObjectState] = []
        for oid, pos, rot, scale in self.scene.object_states():
            center = self.camera.project(pos)
            if center is None:  # behind the camera: not pickable
                continue
            edge = self.camera.project(tuple(np.asarray(pos, dtype=float) + basis.right * 0.5 * scale))
            radius = float(np.hypot(edge[0] - center[0], edge[1] - center[1])) if edge else 0.0
            states.append(ObjectState(oid, tuple(pos), tuple(rot), float(scale), center, radius))
        return SceneSnapshot(states, tuple(ui_targets), basis, self.scene.selected_id, ui_aspect)

    # ------------------------------------------------------------------- apply
    def apply(self, commands: Iterable[Command]) -> None:
        for c in commands:
            if c.kind is CommandKind.SELECT and c.object_id is not None:
                self.scene.select(c.object_id)
            elif c.kind is CommandKind.DESELECT:
                self.scene.deselect()
            elif c.kind is CommandKind.TRANSFORM and c.object_id is not None:
                self.scene.set_transform(c.object_id, c.position, c.rotation, c.scale)
            elif c.kind is CommandKind.PRESS_BUTTON and c.name in self.actions:
                self.actions[c.name]()
            # MODE_CHANGED is informational: the mode lives in the InteractionManager.

    # ------------------------------------------------------- user-level actions
    def create(self, kind: str) -> None:
        obj_id = self.scene.spawn(kind)
        self.scene.select(obj_id)

    def delete_selected(self) -> None:
        if self.scene.selected_id is not None:
            self.scene.delete(self.scene.selected_id)

    def reset_scene(self) -> None:
        self.scene.reset()
