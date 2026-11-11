import math
import unittest

from graphics.mesh_data import cylinder_mesh_data


class CylinderMeshTests(unittest.TestCase):
    def test_counts_and_consistency(self):
        verts, tris, norms = cylinder_mesh_data(16)
        self.assertEqual(len(verts), len(norms))
        self.assertEqual(len(verts), 12 * 16)
        self.assertEqual(len(tris), len(verts) // 3)
        self.assertTrue(all(max(t) < len(verts) for t in tris))

    def test_bounds_and_unit_normals(self):
        verts, _, norms = cylinder_mesh_data(24, radius=0.5, height=1.0)
        for x, y, z in verts:
            self.assertLessEqual(math.hypot(x, z), 0.5 + 1e-9)
            self.assertLessEqual(abs(y), 0.5 + 1e-9)
        for n in norms:
            self.assertAlmostEqual(math.sqrt(sum(c * c for c in n)), 1.0, places=9)

    def test_rejects_too_few_segments(self):
        with self.assertRaises(ValueError):
            cylinder_mesh_data(2)


if __name__ == "__main__":
    unittest.main()
