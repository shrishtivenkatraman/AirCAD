import math
import unittest

from tests.synthetic import make_hand
from utils.config import GestureConfig
from vision.gesture_recognition import GestureRecognizer, GestureType, classify_static
from vision.landmark_processor import (HandAssigner, extended_fingers, hand_roll_deg,
                                       hand_scale, pinch_ratio, wrap_angle_deg)

CFG = GestureConfig()
DT = 1 / 30


def feed(rec, hand, frames, t0=0.0):
    out = None
    for i in range(frames):
        out = rec.update(hand, t0 + i * DT)
    return out


class StaticClassificationTests(unittest.TestCase):
    def test_pinch_detected(self):
        self.assertIs(classify_static(make_hand(pinch_ratio=0.1), CFG), GestureType.PINCH)

    def test_pinch_ratio_is_scale_invariant(self):
        # same pose, different aspect ratio -> same ratio
        self.assertAlmostEqual(pinch_ratio(make_hand(pinch_ratio=0.2, aspect=4 / 3)),
                               pinch_ratio(make_hand(pinch_ratio=0.2, aspect=16 / 9)), places=6)

    def test_open_hand_is_not_pinch(self):
        self.assertGreater(pinch_ratio(make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True)), CFG.pinch_off)

    def test_open_palm_detected_with_or_without_thumb(self):
        for thumb in (True, False):
            hand = make_hand(thumb=thumb, index=True, middle=True, ring=True, pinky=True)
            self.assertIs(classify_static(hand, CFG), GestureType.OPEN_PALM)

    def test_three_finger_detected(self):
        self.assertIs(classify_static(make_hand(index=True, middle=True, ring=True), CFG), GestureType.THREE_FINGER)

    def test_fist_and_pointing_are_neutral(self):
        self.assertIs(classify_static(make_hand(), CFG), GestureType.NEUTRAL)
        self.assertIs(classify_static(make_hand(index=True), CFG), GestureType.NEUTRAL)

    def test_gestures_are_mutually_exclusive_pinch_wins(self):
        hand = make_hand(index=True, middle=True, ring=True, pinky=True, pinch_ratio=0.05)
        self.assertIs(classify_static(hand, CFG), GestureType.PINCH)

    def test_pinch_hysteresis(self):
        between = make_hand(pinch_ratio=0.38)  # between pinch_on (0.30) and pinch_off (0.45)
        self.assertIsNot(classify_static(between, CFG, was_pinching=False), GestureType.PINCH)
        self.assertIs(classify_static(between, CFG, was_pinching=True), GestureType.PINCH)

    def test_finger_extension_is_rotation_invariant(self):
        for roll in (0, 45, 90, 180, -60):
            hand = make_hand(index=True, middle=True, roll_deg=roll)
            self.assertEqual(extended_fingers(hand)[1:], (True, True, False, False), roll)

    def test_hand_scale_positive(self):
        self.assertAlmostEqual(hand_scale(make_hand()), 0.22, places=6)

    def test_roll_angle(self):
        self.assertAlmostEqual(hand_roll_deg(make_hand()), 0.0, places=6)
        for roll in (-70, -20, 30, 90):
            self.assertAlmostEqual(hand_roll_deg(make_hand(roll_deg=roll)), roll, places=4)

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle_deg(190), -170)
        self.assertAlmostEqual(wrap_angle_deg(-190), 170)
        self.assertAlmostEqual(wrap_angle_deg(45), 45)


class RecognizerStateTests(unittest.TestCase):
    def test_debounce_requires_consecutive_frames(self):
        rec = GestureRecognizer(CFG)
        pinch = make_hand(pinch_ratio=0.1)
        self.assertIs(feed(rec, pinch, CFG.confirm_frames - 1).gesture, GestureType.NONE)
        self.assertIs(rec.update(pinch, 1.0).gesture, GestureType.PINCH)

    def test_single_noisy_frame_does_not_change_state(self):
        rec = GestureRecognizer(CFG)
        feed(rec, make_hand(pinch_ratio=0.1), 6)
        self.assertIs(rec.update(make_hand(), 1.0).gesture, GestureType.PINCH)  # one open frame
        self.assertIs(rec.update(make_hand(pinch_ratio=0.1), 1.1).gesture, GestureType.PINCH)

    def test_pinch_release_transition(self):
        rec = GestureRecognizer(CFG)
        feed(rec, make_hand(pinch_ratio=0.1), 5)
        self.assertIs(rec.confirmed, GestureType.PINCH)
        feed(rec, make_hand(thumb=True, index=True, middle=True, ring=True, pinky=True), 5, t0=1.0)
        self.assertIs(rec.confirmed, GestureType.OPEN_PALM)

    def test_hand_loss_within_grace_keeps_gesture(self):
        rec = GestureRecognizer(CFG)
        feed(rec, make_hand(pinch_ratio=0.1), 5, t0=0.0)
        last = 4 * DT
        r = rec.update(None, last + 0.2)
        self.assertIs(r.gesture, GestureType.PINCH)
        self.assertFalse(r.present)

    def test_hand_loss_beyond_grace_resets(self):
        rec = GestureRecognizer(CFG)
        feed(rec, make_hand(pinch_ratio=0.1), 5, t0=0.0)
        r = rec.update(None, 4 * DT + CFG.lost_grace_s + 0.1)
        self.assertIs(r.gesture, GestureType.NONE)
        self.assertIs(rec.confirmed, GestureType.NONE)

    def test_never_seen_hand_is_none(self):
        self.assertIs(GestureRecognizer(CFG).update(None, 0.0).gesture, GestureType.NONE)


class HandAssignerTests(unittest.TestCase):
    def test_identity_is_stable_when_hands_swap_order(self):
        a = HandAssigner()
        left = make_hand(offset=(-0.25, 0))
        right = make_hand(offset=(0.25, 0))
        first = a.assign([left, right])
        second = a.assign([right, left])  # detector order flipped
        self.assertIs(first["A"], left)
        self.assertIs(second["A"], left)
        self.assertIs(second["B"], right)

    def test_single_hand_keeps_its_slot_after_second_hand_leaves(self):
        a = HandAssigner()
        left = make_hand(offset=(-0.25, 0))
        right = make_hand(offset=(0.25, 0))
        a.assign([left, right])
        only_right = make_hand(offset=(0.26, 0))
        self.assertIn("B", a.assign([only_right]))

    def test_empty_input(self):
        self.assertEqual(HandAssigner().assign([]), {})


if __name__ == "__main__":
    unittest.main()
