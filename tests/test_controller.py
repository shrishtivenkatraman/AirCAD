import unittest

import numpy as np

from controls.gesture_mapping import CameraBasis
from controls.interaction_manager import Command, CommandKind, UiTarget
from controls.object_controller import ObjectController


class FakeScene:
    def __init__(self):
        self.objs = {1: [(0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0], 2: [(1.0, 0.0, 0.0), (0.0, 0.0, 0.0), 2.0]}
        self.selected_id = None
        self.spawned = []
        self.deleted = []
        self.was_reset = False

    def object_states(self):
        return [(i, *v) for i, v in self.objs.items()]

    def select(self, i): self.selected_id = i
    def deselect(self): self.selected_id = None

    def set_transform(self, i, position, rotation, scale):
        o = self.objs[i]
        o[0] = position or o[0]
        o[1] = rotation or o[1]
        o[2] = scale if scale is not None else o[2]

    def spawn(self, kind):
        self.spawned.append(kind)
        return 99

    def delete(self, i): self.deleted.append(i)
    def reset(self): self.was_reset = True


class FakeCamera:
    def basis(self): return CameraBasis()

    def project(self, p):
        return None if p[2] < -5 else (p[0] * 0.1, p[1] * 0.1)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.scene, self.cam = FakeScene(), FakeCamera()
        self.ctl = ObjectController(self.scene, self.cam)

    def test_snapshot_projects_objects_and_radius_scales_with_size(self):
        snap = self.ctl.snapshot([UiTarget("x", (0, 0), (0.1, 0.1))], 16 / 9)
        by_id = {o.object_id: o for o in snap.objects}
        self.assertEqual(by_id[2].screen_xy, (0.1, 0.0))
        self.assertAlmostEqual(by_id[1].screen_radius, 0.05)
        self.assertAlmostEqual(by_id[2].screen_radius, 0.1)
        self.assertEqual(len(snap.ui_targets), 1)

    def test_objects_behind_camera_are_skipped(self):
        self.scene.objs[1][0] = (0.0, 0.0, -10.0)
        ids = [o.object_id for o in self.ctl.snapshot([], 1.0).objects]
        self.assertEqual(ids, [2])

    def test_apply_commands(self):
        hit = []
        self.ctl.actions["cube"] = lambda: hit.append("cube")
        self.ctl.apply([
            Command(CommandKind.SELECT, object_id=2),
            Command(CommandKind.TRANSFORM, object_id=2, position=(5.0, 1.0, 0.0)),
            Command(CommandKind.TRANSFORM, object_id=2, scale=1.5),
            Command(CommandKind.PRESS_BUTTON, name="cube"),
            Command(CommandKind.PRESS_BUTTON, name="unknown"),
        ])
        self.assertEqual(self.scene.selected_id, 2)
        self.assertEqual(self.scene.objs[2][0], (5.0, 1.0, 0.0))
        self.assertEqual(self.scene.objs[2][2], 1.5)
        self.assertEqual(hit, ["cube"])
        self.ctl.apply([Command(CommandKind.DESELECT)])
        self.assertIsNone(self.scene.selected_id)

    def test_create_delete_reset(self):
        self.ctl.create("sphere")
        self.assertEqual(self.scene.spawned, ["sphere"])
        self.assertEqual(self.scene.selected_id, 99)
        self.ctl.delete_selected()
        self.assertEqual(self.scene.deleted, [99])
        self.ctl.reset_scene()
        self.assertTrue(self.scene.was_reset)


if __name__ == "__main__":
    unittest.main()
