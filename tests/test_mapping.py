import unittest

import numpy as np

from controls.gesture_mapping import CameraBasis, CoordinateMapper, two_hand_distance
from utils.config import MappingConfig


class MappingTests(unittest.TestCase):
    def setUp(self):
        self.cfg = MappingConfig(active_margin=0.1, translate_sensitivity=10.0,
                                 rotate_sensitivity_deg=100.0, min_scale=0.5, max_scale=3.0)
        self.m = CoordinateMapper(self.cfg)

    # ---- normalization
    def test_center_maps_to_origin(self):
        np.testing.assert_allclose(self.m.normalize_to_plane((0.5, 0.5)), [0, 0], atol=1e-9)

    def test_active_region_edges_map_to_unit_square(self):
        np.testing.assert_allclose(self.m.normalize_to_plane((0.1, 0.1)), [-1, 1], atol=1e-9)
        np.testing.assert_allclose(self.m.normalize_to_plane((0.9, 0.9)), [1, -1], atol=1e-9)

    def test_outside_active_region_is_clamped(self):
        np.testing.assert_allclose(self.m.normalize_to_plane((0.0, 1.0)), [-1, -1])
        np.testing.assert_allclose(self.m.normalize_to_plane((2.0, -1.0)), [1, 1])

    def test_y_axis_points_up(self):
        self.assertGreater(self.m.normalize_to_plane((0.5, 0.3))[1], 0)

    def test_plane_to_ui(self):
        np.testing.assert_allclose(CoordinateMapper.plane_to_ui((1, 1), 2.0), [1.0, 0.5])
        np.testing.assert_allclose(CoordinateMapper.plane_to_ui((-1, -1), 2.0), [-1.0, -0.5])

    # ---- translation
    def test_translation_follows_camera_axes(self):
        basis = CameraBasis()
        out = self.m.translation((1, 2, 3), (0.1, -0.2), basis)
        np.testing.assert_allclose(out, [2.0, 0.0, 3.0])

    def test_translation_uses_rotated_basis(self):
        basis = CameraBasis(right=np.array([0.0, 0.0, 1.0]), up=np.array([0.0, 1.0, 0.0]),
                            forward=np.array([-1.0, 0.0, 0.0]))
        out = self.m.translation((0, 0, 0), (0.1, 0.0), basis)
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0])

    def test_depth_disabled_by_default_and_enabled_by_gain(self):
        self.assertEqual(self.m.depth_offset(0.2, 0.3), 0.0)
        self.cfg.depth_gain = 4.0
        self.assertAlmostEqual(self.m.depth_offset(0.2, 0.3), 2.0)
        out = self.m.translation((0, 0, 0), (0, 0), CameraBasis(), depth=2.0)
        np.testing.assert_allclose(out, [0, 0, -2.0])

    # ---- rotation
    def test_rotation_axes_and_roll(self):
        out = self.m.rotation((10, 20, 30), (0.1, 0.2), roll_delta_deg=15)
        np.testing.assert_allclose(out, [10 + 20, 20 - 10, 45])

    def test_zero_motion_keeps_rotation(self):
        np.testing.assert_allclose(self.m.rotation((1, 2, 3), (0, 0), 0), [1, 2, 3])

    # ---- scale
    def test_double_distance_doubles_scale(self):
        self.assertAlmostEqual(self.m.scale(1.0, 0.4, 0.8), 2.0)

    def test_half_distance_halves_scale(self):
        self.assertAlmostEqual(self.m.scale(2.0, 0.8, 0.4), 1.0)

    def test_scale_is_clamped(self):
        self.assertAlmostEqual(self.m.scale(1.0, 0.1, 10.0), 3.0)
        self.assertAlmostEqual(self.m.scale(1.0, 10.0, 0.1), 0.5)

    def test_scale_gain_exponent(self):
        self.cfg.scale_gain = 2.0
        self.assertAlmostEqual(self.m.scale(1.0, 1.0, 1.5), 2.25)

    def test_scale_handles_zero_start_distance(self):
        self.assertAlmostEqual(self.m.scale(1.0, 0.0, 1.0), 3.0)  # clamped, no ZeroDivisionError

    def test_two_hand_distance_respects_aspect(self):
        self.assertAlmostEqual(two_hand_distance((0, 0), (3, 4)), 5.0)
        self.assertAlmostEqual(two_hand_distance((0, 0), (1, 0), aspect=2.0), 2.0)


if __name__ == "__main__":
    unittest.main()
