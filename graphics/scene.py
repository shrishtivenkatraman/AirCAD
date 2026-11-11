"""Scene manager: grid, axes, lights and the set of SceneObjects."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence, Tuple

from ursina import AmbientLight, DirectionalLight, Entity, Text, Vec3, color, destroy

from graphics.objects import KINDS, SceneObject

DEFAULT_LAYOUT = (("cube", (-2.5, 0.5, 0.0)), ("sphere", (0.0, 0.5, 0.0)), ("cylinder", (2.5, 0.5, 0.0)))


class Scene:
    def __init__(self) -> None:
        self.objects: Dict[int, SceneObject] = {}
        self.selected_id: Optional[int] = None
        self._next_id = 1
        self._build_environment()
        self.reset()

    # ------------------------------------------------------------ environment
    def _build_environment(self) -> None:
        size = 20
        try:
            from ursina import Grid
            self.grid = Entity(model=Grid(size, size), rotation_x=90, scale=size,
                               color=color.rgba(255, 255, 255, 70))
        except Exception:  # fallback: build the grid from thin boxes
            self.grid = Entity()
            for i in range(-size // 2, size // 2 + 1):
                Entity(parent=self.grid, model="cube", position=(i, 0, 0), scale=(0.02, 0.01, size),
                       color=color.rgba(255, 255, 255, 70))
                Entity(parent=self.grid, model="cube", position=(0, 0, i), scale=(size, 0.01, 0.02),
                       color=color.rgba(255, 255, 255, 70))
        length = 6.0
        self.axes = [
            Entity(model="cube", position=(length / 2, 0.02, 0), scale=(length, 0.05, 0.05), color=color.red),
            Entity(model="cube", position=(0, length / 2, 0), scale=(0.05, length, 0.05), color=color.green),
            Entity(model="cube", position=(0, 0.02, length / 2), scale=(0.05, 0.05, length), color=color.blue),
        ]
        for label, pos, col in (("X", (length + 0.4, 0, 0), color.red),
                                ("Y", (0, length + 0.4, 0), color.green),
                                ("Z", (0, 0, length + 0.4), color.blue)):
            Text(label, position=pos, scale=12, billboard=True, color=col, origin=(0, 0))
        self.sun = DirectionalLight(shadows=False)
        self.sun.look_at(Vec3(1, -1.5, 1))
        self.ambient = AmbientLight(color=color.rgba(110, 110, 110, 0.1))

    # --------------------------------------------------------------- objects
    def spawn(self, kind: str, position: Optional[Sequence[float]] = None) -> int:
        if kind not in KINDS:
            raise ValueError(kind)
        if position is None:
            position = (random.uniform(-3.0, 3.0), 0.5, random.uniform(-2.0, 2.0))
        obj = SceneObject(self._next_id, kind, tuple(position))
        self.objects[obj.object_id] = obj
        self._next_id += 1
        return obj.object_id

    def select(self, object_id: int) -> None:
        if object_id not in self.objects:
            return
        self.deselect()
        self.selected_id = object_id
        self.objects[object_id].set_selected(True)

    def deselect(self) -> None:
        if self.selected_id in self.objects:
            self.objects[self.selected_id].set_selected(False)
        self.selected_id = None

    def delete(self, object_id: int) -> None:
        obj = self.objects.pop(object_id, None)
        if obj is None:
            return
        if self.selected_id == object_id:
            self.selected_id = None
        destroy(obj)

    def reset(self) -> None:
        for oid in list(self.objects):
            self.delete(oid)
        self.selected_id = None
        for kind, pos in DEFAULT_LAYOUT:
            self.spawn(kind, pos)

    def set_transform(self, object_id: int, position: Optional[Sequence[float]] = None,
                      rotation: Optional[Sequence[float]] = None, scale: Optional[float] = None) -> None:
        obj = self.objects.get(object_id)
        if obj is None:
            return
        if position is not None:
            obj.position = Vec3(*position)
        if rotation is not None:
            obj.rotation = Vec3(*rotation)
        if scale is not None:
            obj.set_uniform_scale(scale)

    def object_states(self) -> List[Tuple[int, Tuple[float, float, float], Tuple[float, float, float], float]]:
        return [(o.object_id, (o.x, o.y, o.z), o.euler, o.uniform_scale) for o in self.objects.values()]

    def selected_description(self) -> str:
        obj = self.objects.get(self.selected_id) if self.selected_id is not None else None
        if obj is None:
            return "Selected: none"
        return (f"Selected: {obj.kind} #{obj.object_id}\n"
                f"pos ({obj.x:.1f}, {obj.y:.1f}, {obj.z:.1f})\n"
                f"rot ({obj.rotation_x:.0f}, {obj.rotation_y:.0f}, {obj.rotation_z:.0f})\n"
                f"scale {obj.uniform_scale:.2f}")
