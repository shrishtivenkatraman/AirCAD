"""Pure-Python mesh generators (no Ursina import, so they are unit-testable)."""
from __future__ import annotations

import math
from typing import List, Tuple

Vec3 = Tuple[float, float, float]


def cylinder_mesh_data(segments: int = 32, radius: float = 0.5, height: float = 1.0
                       ) -> Tuple[List[Vec3], List[Tuple[int, int, int]], List[Vec3]]:
    """Return ``(vertices, triangles, normals)`` for a capped cylinder centred on the origin.

    The mesh is non-indexed (3 unique vertices per triangle) so caps and sides keep
    their own normals. Side normals are smooth around the circumference.
    """
    if segments < 3:
        raise ValueError("a cylinder needs at least 3 segments")
    h = height / 2.0
    verts: List[Vec3] = []
    norms: List[Vec3] = []

    def tri(points, normals):
        verts.extend(points)
        norms.extend(normals)

    for i in range(segments):
        a0 = 2 * math.pi * i / segments
        a1 = 2 * math.pi * (i + 1) / segments
        c0, s0, c1, s1 = math.cos(a0), math.sin(a0), math.cos(a1), math.sin(a1)
        b0, b1 = (radius * c0, -h, radius * s0), (radius * c1, -h, radius * s1)
        t0, t1 = (radius * c0, h, radius * s0), (radius * c1, h, radius * s1)
        n0, n1 = (c0, 0.0, s0), (c1, 0.0, s1)
        tri([b0, t0, t1], [n0, n0, n1])            # side quad = 2 triangles
        tri([b0, t1, b1], [n0, n1, n1])
        tri([(0.0, h, 0.0), t1, t0], [(0.0, 1.0, 0.0)] * 3)      # top cap
        tri([(0.0, -h, 0.0), b0, b1], [(0.0, -1.0, 0.0)] * 3)    # bottom cap

    triangles = [(i, i + 1, i + 2) for i in range(0, len(verts), 3)]
    return verts, triangles, norms
