from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


@dataclass
class EventStream:
    """Container for event streams.

    Attributes
    ----------
    t, x, y, p: np.ndarray
        Event timestamps and coordinates.
    meta: dict, optional
        Auxiliary metadata preserved from disk.
    """

    t: np.ndarray
    x: np.ndarray
    y: np.ndarray
    p: np.ndarray
    meta: Optional[Dict] = field(default=None)

    def validate(self, *, check_monotonic: bool = False) -> List[str]:
        """Validate array shapes and dtypes.

        Parameters
        ----------
        check_monotonic : bool, optional
            If True, report a warning when timestamps are not monotonic.

        Returns
        -------
        list of str
            Validation warnings (does not raise for monotonicity).
        """

        warnings: List[str] = []

        if not (self.t.shape == self.x.shape == self.y.shape == self.p.shape):
            raise ValueError("t, x, y, p must share the same shape")

        # dtype sanity checks (align with existing util expectations)
        if self.t.dtype.kind != "f":
            raise ValueError("t must be a float array")
        if self.x.dtype.kind not in {"i", "u"}:
            raise ValueError("x must be an integer array")
        if self.y.dtype.kind not in {"i", "u"}:
            raise ValueError("y must be an integer array")
        if self.p.dtype.kind not in {"i", "u"}:
            raise ValueError("p must be an integer array")

        if not np.all(np.isin(self.p, [-1, 1])):
            raise ValueError("p must be in {-1, +1}")

        if check_monotonic and self.t.size:
            if np.any(np.diff(self.t) < 0):
                warnings.append("timestamps are not sorted (monotonicity not enforced)")

        return warnings
