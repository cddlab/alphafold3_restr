from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from .chiral_data import length


@dataclass
class BondData:
    """Flat-bottomed bond length restraint."""

    aid0: int
    aid1: int
    r0: float   # reference bond length (Angstrom)
    slack: float = 0.0
    w: float = 0.05
    half: bool = False  # one-sided restraint (only penalizes compression)

    def is_valid(self) -> bool:
        if self.aid0 >= 0 and self.aid1 >= 0:
            return True
        if self.w > 0.0:
            return True
        return False

    def reset_indices(self) -> None:
        self.aid0 = -1
        self.aid1 = -1

    def setup(self, ind: int, aid: int) -> None:
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    def calc(self, crds: np.ndarray) -> float:
        v1 = crds[self.aid0] - crds[self.aid1]
        n1l = length(v1)
        r2 = self.r0 + self.slack
        r1 = self.r0 - self.slack
        if n1l > r2:
            delta = n1l - r2
        elif n1l < r1:
            delta = n1l - r1
        else:
            return 0.0
        if self.half and delta < 0:
            return 0.0
        return self.w * delta * delta

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        v1 = crds[self.aid0] - crds[self.aid1]
        n1l = length(v1)
        r2 = self.r0 + self.slack
        r1 = self.r0 - self.slack
        if n1l > r2:
            delta = r2 / n1l
        elif n1l < r1:
            delta = r1 / n1l
        else:
            return
        if self.half and 1.0 < delta:
            return
        con = 2.0 * self.w * (1.0 - delta)
        if not self.half:
            grad[self.aid0] += v1 * con
        grad[self.aid1] -= v1 * con

    def print(self, crds: np.ndarray) -> None:
        v1 = crds[self.aid0] - crds[self.aid1]
        n1l = length(v1)
        print(
            f"B {self.aid0}-{self.aid1}:"
            f" cur {n1l:.2f} ref {self.r0:.2f} dif {n1l - self.r0:.2f}"
        )

    def calc_sd(self, crds: np.ndarray) -> float:
        v1 = crds[self.aid0] - crds[self.aid1]
        n1l = length(v1)
        return (n1l - self.r0) ** 2
