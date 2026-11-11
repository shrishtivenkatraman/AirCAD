"""Selectable 3D entities."""
from __future__ import annotations

from typing import Tuple

from ursina import Entity, Mesh, Vec3, color

from graphics.mesh_data import cylinder_mesh_data

try:  # lighting shader; fall back to flat colours if unavailable in this Ursina version
    from ursina.shaders import lit_with_shadows_shader as LIT_SHADER
except Exception:  # pragma: no cover
    LIT_SHADER = None

KINDS = ("cube", "sphere", "cylinder")
KIND_COLORS = {"cube": color.azure, "sphere": color.orange, "cylinder": color.lime}


def _model_for(kind: str):
    if kind == "cylinder":
        verts, tris, normals = cylinder_mesh_data(32)
        return Mesh(vertices=verts, triangles=tris, normals=normals)
    return kind  # built-in 'cube' / 'sphere' (both unit-sized and centred)


class SceneObject(Entity):
    """A real 3D entity with uniform scale, Euler rotation and a selection outline."""

    def __init__(self, object_id: int, kind: str, position: Tuple[float, float, float]) -> None:
        if kind not in KINDS:
            raise ValueError(f"unknown object kind {kind!r}")
        super().__init__(model=_model_for(kind), color=KIND_COLORS[kind],
                         position=Vec3(*position), double_sided=True, shader=LIT_SHADER)
        self.object_id = object_id
        self.kind = kind
        self.base_color = KIND_COLORS[kind]
        self._outline = Entity(parent=self, model="wireframe_cube", color=color.yellow,
                               scale=1.04, enabled=False)

    @property
    def euler(self) -> Tuple[float, float, float]:
        return (self.rotation_x, self.rotation_y, self.rotation_z)

    @property
    def uniform_scale(self) -> float:
        return float(self.scale_x)

    def set_uniform_scale(self, value: float) -> None:
        self.scale = Vec3(value, value, value)

    def set_selected(self, selected: bool) -> None:
        self.color = self.base_color.tint(0.25) if selected else self.base_color
        self._outline.enabled = selected
