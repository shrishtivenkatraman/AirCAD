"""Small smoothing helpers used to tame landmark jitter."""
from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np

ArrayLike = Union[float, Sequence[float], np.ndarray]


def lerp(a: ArrayLike, b: ArrayLike, t: float) -> np.ndarray:
    """Linear interpolation between ``a`` and ``b`` (t=0 -> a, t=1 -> b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return a + (b - a) * t


class ExponentialSmoother:
    """Exponential moving average: ``s = alpha * x + (1 - alpha) * s``.

    ``alpha`` close to 1 follows the input tightly (responsive, jittery);
    ``alpha`` close to 0 is very smooth but laggy. The first sample passes
    through unchanged so there is no ramp-up from zero.
    """

    def __init__(self, alpha: float = 0.35) -> None:
        self._alpha = 0.35
        self.alpha = alpha
        self._state: Optional[np.ndarray] = None

    @property
    def alpha(self) -> float:
        return self._alpha

    @alpha.setter
    def alpha(self, value: float) -> None:
        if not 0.0 < value <= 1.0:
            raise ValueError("alpha must be in (0, 1]")
        self._alpha = float(value)

    @property
    def value(self) -> Optional[np.ndarray]:
        return None if self._state is None else self._state.copy()

    def update(self, sample: ArrayLike) -> np.ndarray:
        x = np.asarray(sample, dtype=float)
        if self._state is None or self._state.shape != x.shape:
            self._state = x.copy()
        else:
            self._state = self._alpha * x + (1.0 - self._alpha) * self._state
        return self._state.copy()

    def reset(self) -> None:
        self._state = None
