from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from rdkit import Chem
import math


def length(v: np.ndarray, eps: float = 1e-6) -> float:
    """Calculate the length of a vector."""
    return math.sqrt(max(eps, v[0] * v[0] + v[1] * v[1] + v[2] * v[2]))


def unit_vec(v: np.ndarray, eps: float = 1e-6) -> tuple[np.ndarray, float]:
    """Calculate the unit vector."""
    vl = length(v, eps=eps)
    return v / vl, vl


def calc_chiral_vol(crds: np.ndarray, iatm: int, aj: list[int]) -> float:
    """Calculate the chiral volume (scalar triple product)."""
    vc = crds[iatm]
    v1 = crds[aj[0]] - vc
    v2 = crds[aj[1]] - vc
    v3 = crds[aj[2]] - vc
    return np.dot(v1, np.cross(v2, v3))


@dataclass
class ChiralData:
    """Class for chiral volume restraint data."""

    aid0: int  # chiral center
    aid1: int
    aid2: int
    aid3: int
    chiral_vol: float  # reference signed volume
    w: float = 0.1
    slack: float = 0.05
    fmax: float = -100.0

    def setup(self, ind: int, aid: int) -> None:
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        elif aid == 2:
            self.aid2 = ind
        elif aid == 3:
            self.aid3 = ind
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    def is_valid(self) -> bool:
        if self.aid0 >= 0 and self.aid1 >= 0 and self.aid2 >= 0 and self.aid3 >= 0:
            return True
        if self.w > 0.0:
            return True
        return False

    def reset_indices(self) -> None:
        self.aid0 = -1
        self.aid1 = -1
        self.aid2 = -1
        self.aid3 = -1

    def calc(self, crds: np.ndarray) -> float:
        vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])
        thr = self.chiral_vol - self.slack if self.chiral_vol > 0 else self.chiral_vol + self.slack
        delta = vol - thr
        return delta * delta * self.w

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> bool:
        a0 = crds[self.aid0]
        a1 = crds[self.aid1]
        a2 = crds[self.aid2]
        a3 = crds[self.aid3]
        v1 = a1 - a0
        v2 = a2 - a0
        v3 = a3 - a0

        thr = self.chiral_vol - self.slack if self.chiral_vol > 0 else self.chiral_vol + self.slack
        vol = np.dot(v1, np.cross(v2, v3))
        delta = vol - thr
        dE = 2.0 * delta * self.w

        f1 = np.cross(v2, v3) * dE
        f2 = np.cross(v3, v1) * dE
        f3 = np.cross(v1, v2) * dE
        fc = -f1 - f2 - f3

        n1, n1l = unit_vec(f1)
        n2, n2l = unit_vec(f2)
        n3, n3l = unit_vec(f3)
        nc, ncl = unit_vec(fc)

        if self.fmax > 0:
            n1l = min(n1l, self.fmax)
            n2l = min(n2l, self.fmax)
            n3l = min(n3l, self.fmax)
            ncl = min(ncl, self.fmax)
            f1 = n1 * n1l
            f2 = n2 * n2l
            f3 = n3 * n3l
            fc = nc * ncl

        grad[self.aid0] += fc
        grad[self.aid1] += f1
        grad[self.aid2] += f2
        grad[self.aid3] += f3
        return True

    def print(self, crds: np.ndarray) -> None:
        vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])
        print(
            f"C {self.aid0}-{self.aid1}-{self.aid2}-{self.aid3}:"
            f" cur {vol:.2f} ref {self.chiral_vol:.2f} dif {vol - self.chiral_vol:.2f}"
        )

    def calc_sd(self, crds: np.ndarray) -> float:
        vol = calc_chiral_vol(crds, self.aid0, [self.aid1, self.aid2, self.aid3])
        return (vol - self.chiral_vol) ** 2

    @staticmethod
    def get_nei_atoms(iatm: int, mol: Chem.Mol) -> list[int]:
        atom = mol.GetAtomWithIdx(iatm)
        return [b.GetOtherAtom(atom).GetIdx() for b in atom.GetBonds()]
