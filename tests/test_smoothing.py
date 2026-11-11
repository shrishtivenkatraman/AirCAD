import unittest

import numpy as np

from utils.smoothing import ExponentialSmoother, lerp


class SmoothingTests(unittest.TestCase):
    def test_first_sample_passes_through(self):
        s = ExponentialSmoother(0.3)
        np.testing.assert_allclose(s.update([2.0, -1.0]), [2.0, -1.0])

    def test_ema_formula(self):
        s = ExponentialSmoother(0.5)
        s.update([0.0])
        np.testing.assert_allclose(s.update([10.0]), [5.0])
        np.testing.assert_allclose(s.update([10.0]), [7.5])

    def test_converges_to_constant_input(self):
        s = ExponentialSmoother(0.35)
        s.update([0.0, 0.0])
        for _ in range(60):
            out = s.update([1.0, 2.0])
        np.testing.assert_allclose(out, [1.0, 2.0], atol=1e-6)

    def test_alpha_one_is_identity(self):
        s = ExponentialSmoother(1.0)
        s.update([0.0])
        np.testing.assert_allclose(s.update([4.0]), [4.0])

    def test_reduces_jitter(self):
        rng = np.random.default_rng(0)
        noisy = 1.0 + rng.normal(0, 0.05, 200)
        s = ExponentialSmoother(0.2)
        out = np.array([s.update([v])[0] for v in noisy])
        self.assertLess(out[20:].std(), noisy[20:].std())

    def test_reset_and_validation(self):
        s = ExponentialSmoother(0.4)
        s.update([1.0])
        s.reset()
        self.assertIsNone(s.value)
        np.testing.assert_allclose(s.update([9.0]), [9.0])
        with self.assertRaises(ValueError):
            ExponentialSmoother(0.0)
        with self.assertRaises(ValueError):
            s.alpha = 1.5

    def test_lerp(self):
        np.testing.assert_allclose(lerp([0, 0], [10, 20], 0.25), [2.5, 5.0])


if __name__ == "__main__":
    unittest.main()
