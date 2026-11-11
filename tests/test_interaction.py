"""Scripted end-to-end scenarios for the gesture state machine (no webcam, no renderer)."""
import unittest

from controls.interaction_manager import (CommandKind, InteractionManager, Mode, ObjectState,
                                          SceneSnapshot, State, UiTarget, pick_object, pick_ui_target)
from tests.synthetic import make_hand_at
from utils.config import AppConfig

DT = 1 / 30
OPEN = dict(thumb=True, index=True, middle=True, ring=True, pinky=True)
PINCH = dict(pinch_ratio=0.1)
NEUTRAL = {**OPEN, "index": False}   # visible hand, no command gesture


class World:
    """Minimal stand-in for the scene: one object at the screen centre."""

    def __init__(self):
        self.pos, self.rot, self.scale, self.selected = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), 1.0, None
        self.exists = True
        self.targets = ()

    def snapshot(self):
        objs = [ObjectState(1, self.pos, self.rot, self.scale, (0.0, 0.0), 0.06)] if self.exists else []
        return SceneSnapshot(objs, self.targets, selected_id=self.selected, ui_aspect=16 / 9)

    def apply(self, cmds):
        for c in cmds:
            if c.kind is CommandKind.SELECT:
                self.selected = c.object_id
            elif c.kind is CommandKind.DESELECT:
                self.selected = None
            elif c.kind is CommandKind.TRANSFORM:
                self.pos = c.position or self.pos
                self.rot = c.rotation or self.rot
                self.scale = c.scale if c.scale is not None else self.scale


class Rig:
    def __init__(self):
        self.mgr = InteractionManager(AppConfig())
        self.world = World()
        self.t = 0.0
        self.log = []

    def step(self, hands, n=1):
        for _ in range(n):
            cmds = self.mgr.update(hands, self.t, self.world.snapshot())
            self.world.apply(cmds)
            self.log.extend(cmds)
            self.t += DT

    def hand(self, xy, **kw):
        return make_hand_at(xy, **kw)


class PickingTests(unittest.TestCase):
    def test_pick_object_nearest_within_radius(self):
        objs = [ObjectState(1, (0, 0, 0), (0, 0, 0), 1, (0.0, 0.0), 0.05),
                ObjectState(2, (0, 0, 0), (0, 0, 0), 1, (0.3, 0.0), 0.05)]
        self.assertEqual(pick_object((0.29, 0.0), objs, 0.02), 2)
        self.assertIsNone(pick_object((0.15, 0.0), objs, 0.02))

    def test_pick_ui_target(self):
        t = [UiTarget("delete", (0.5, 0.2), (0.1, 0.03))]
        self.assertEqual(pick_ui_target((0.55, 0.21), t), "delete")
        self.assertIsNone(pick_ui_target((0.0, 0.0), t))


class InteractionScenarioTests(unittest.TestCase):
    def test_pinch_on_object_selects_and_moves_it_right(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        self.assertIs(r.mgr.state, State.GRABBING)
        self.assertEqual(r.world.selected, 1)
        x0 = r.world.pos[0]
        r.step({"A": r.hand((0.7, 0.5), **PINCH)}, 40)
        self.assertGreater(r.world.pos[0], x0 + 0.5)       # moved right (identity camera basis)
        self.assertAlmostEqual(r.world.pos[1], 0.0, delta=0.05)

    def test_release_returns_to_idle_and_keeps_selection(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.step({"A": r.hand((0.5, 0.5), **NEUTRAL)}, 6)  # neutral pose
        self.assertIs(r.mgr.state, State.IDLE)
        self.assertEqual(r.world.selected, 1)

    def test_pinch_in_empty_space_selects_nothing(self):
        r = Rig()
        r.step({"A": r.hand((0.85, 0.2), **PINCH)}, 8)
        self.assertIs(r.mgr.state, State.IDLE)
        self.assertIsNone(r.world.selected)

    def test_two_hand_scale_follows_distance_ratio(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)             # select object
        r.step({"A": r.hand((0.5, 0.5), **NEUTRAL)}, 6)  # release
        r.step({"A": r.hand((0.3, 0.8), **NEUTRAL), "B": r.hand((0.7, 0.8), **NEUTRAL)}, 15)  # hands arrive
        a, b = r.hand((0.3, 0.8), **PINCH), r.hand((0.7, 0.8), **PINCH)
        r.step({"A": a, "B": b}, 6)
        self.assertIs(r.mgr.state, State.SCALING)
        a, b = r.hand((0.2, 0.8), **PINCH), r.hand((0.8, 0.8), **PINCH)
        r.step({"A": a, "B": b}, 60)
        self.assertAlmostEqual(r.world.scale, 1.5, delta=0.05)

    def test_dropping_one_hand_during_scale_continues_as_grab_without_jump(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.step({"A": r.hand((0.5, 0.5), **NEUTRAL)}, 6)
        r.step({"A": r.hand((0.3, 0.8), **NEUTRAL), "B": r.hand((0.7, 0.8), **NEUTRAL)}, 15)
        r.step({"A": r.hand((0.3, 0.8), **PINCH), "B": r.hand((0.7, 0.8), **PINCH)}, 6)
        pos = r.world.pos
        r.step({"A": r.hand((0.3, 0.8), **PINCH),
                "B": r.hand((0.7, 0.8), **NEUTRAL)}, 8)
        self.assertIs(r.mgr.state, State.GRABBING)
        for i in range(3):
            self.assertAlmostEqual(r.world.pos[i], pos[i], delta=0.05)

    def test_three_finger_toggles_mode_once_and_respects_cooldown(self):
        r = Rig()
        three = dict(index=True, middle=True, ring=True)
        r.step({"A": r.hand((0.5, 0.7), **three)}, 10)            # held
        self.assertIs(r.mgr.mode, Mode.ROTATE)
        r.step({"A": r.hand((0.5, 0.7), **three)}, 40)            # still held: no re-trigger
        self.assertIs(r.mgr.mode, Mode.ROTATE)
        changes = [c for c in r.log if c.kind is CommandKind.MODE_CHANGED]
        self.assertEqual(len(changes), 1)

    def test_rotate_mode_changes_rotation_not_position(self):
        r = Rig()
        r.mgr.mode = Mode.ROTATE
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.step({"A": r.hand((0.5, 0.3), **PINCH)}, 40)            # drag up -> pitch
        self.assertGreater(r.world.rot[0], 20.0)
        self.assertEqual(r.world.pos, (0.0, 0.0, 0.0))

    def test_open_palm_deselects_when_idle(self):
        r = Rig()
        r.world.selected = 1
        r.step({"A": r.hand((0.5, 0.7), **OPEN)}, 8)
        self.assertIsNone(r.world.selected)

    def test_open_palm_releases_grab_without_deselecting(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.step({"A": r.hand((0.5, 0.5), **OPEN)}, 6)
        self.assertIs(r.mgr.state, State.IDLE)
        self.assertEqual(r.world.selected, 1)

    def test_brief_hand_loss_keeps_grab(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.step({}, 5)                                             # ~0.17 s < grace
        self.assertIs(r.mgr.state, State.GRABBING)

    def test_long_hand_loss_releases(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.step({}, 20)                                            # ~0.67 s > grace
        self.assertIs(r.mgr.state, State.IDLE)

    def test_hand_reappearing_elsewhere_does_not_teleport_object(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        pos = r.world.pos
        r.step({}, 4)
        r.step({"A": r.hand((0.8, 0.8), **PINCH)}, 1)             # reappears far away
        for i in range(3):
            self.assertAlmostEqual(r.world.pos[i], pos[i], delta=0.05)

    def test_pinching_a_button_presses_it_and_grabs_nothing(self):
        r = Rig()
        r.world.targets = (UiTarget("delete", (0.5, 0.2), (0.1, 0.03)),)
        # plane (x,y) -> ui: x*0.5*aspect, y*0.5 ; pick normalized point that lands on the button
        # ui (0.5, 0.2) -> plane (0.5/(0.5*16/9), 0.4) = (0.5625, 0.4) -> normalized
        m = 0.12
        nx = (0.5625 + 1) / 2 * (1 - 2 * m) + m
        ny = (1 - 0.4) / 2 * (1 - 2 * m) + m
        r.step({"A": r.hand((nx, ny), **PINCH)}, 40)
        presses = [c for c in r.log if c.kind is CommandKind.PRESS_BUTTON]
        self.assertEqual([c.name for c in presses], ["delete"])   # once, despite a long pinch
        self.assertIs(r.mgr.state, State.IDLE)

    def test_deleted_object_ends_grab(self):
        r = Rig()
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 6)
        r.world.exists = False
        r.step({"A": r.hand((0.5, 0.5), **PINCH)}, 2)
        self.assertIs(r.mgr.state, State.IDLE)


if __name__ == "__main__":
    unittest.main()
