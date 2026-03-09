from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from rdkit import Chem
from .chiral_data import unit_vec


_angl_patt = Chem.MolFromSmarts("*~*~*")


def get_angle_idxs(mol: Chem.Mol, base_id: int = 0) -> np.ndarray:
    """Get all 1-2-3 angle triples in the molecule."""
    ids = mol.GetSubstructMatches(_angl_patt)
    return np.asarray(ids) + base_id


@dataclass
class AngleData:
    """Flat-bottomed bond angle restraint."""

    aid0: int
    aid1: int  # vertex atom
    aid2: int
    th0: float  # reference angle (radians)
    slack: float = math.radians(5.0)
    w: float = 0.05

    def is_valid(self) -> bool:
        if self.aid0 >= 0 and self.aid1 >= 0 and self.aid2 >= 0:
            return True
        if self.w > 0.0:
            return True
        return False

    def reset_indices(self) -> None:
        self.aid0 = -1
        self.aid1 = -1
        self.aid2 = -1

    def setup(self, ind: int, aid: int) -> None:
        if aid == 0:
            self.aid0 = ind
        elif aid == 1:
            self.aid1 = ind
        elif aid == 2:
            self.aid2 = ind
        else:
            raise ValueError(f"Invalid data {ind=} {aid=}")

    @staticmethod
    def calc_angle(ai: int, aj: int, ak: int, conf) -> float:
        crds = conf.GetPositions()
        eps = 1e-6
        theta, _, _, _, _, _ = AngleData._calc_angle_impl(ai, aj, ak, crds, eps)
        return theta

    @staticmethod
    def _calc_angle_impl(ai, aj, ak, crds, eps=1e-6):
        rij = crds[ai] - crds[aj]
        rkj = crds[ak] - crds[aj]
        eij, Rij = unit_vec(rij, eps)
        ekj, Rkj = unit_vec(rkj, eps)
        costh = min(1.0, max(-1.0, eij.dot(ekj)))
        theta = math.acos(costh)
        return theta, costh, eij, Rij, ekj, Rkj

    def calc(self, crds: np.ndarray) -> float:
        eps = 1e-6
        theta, _, _, _, _, _ = self._calc_angle_impl(self.aid0, self.aid1, self.aid2, crds, eps)
        th2 = self.th0 + self.slack
        th1 = self.th0 - self.slack
        if theta > th2:
            delta = theta - th2
        elif theta < th1:
            delta = theta - th1
        else:
            return 0.0
        return self.w * delta * delta

    def grad(self, crds: np.ndarray, grad: np.ndarray) -> None:
        eps = 1e-6
        theta, costh, eij, Rij, ekj, Rkj = self._calc_angle_impl(
            self.aid0, self.aid1, self.aid2, crds, eps
        )
        th2 = self.th0 + self.slack
        th1 = self.th0 - self.slack
        if theta > th2:
            delta = theta - th2
        elif theta < th1:
            delta = theta - th1
        else:
            return
        df = 2.0 * self.w * delta
        sinth = math.sqrt(max(0.0, 1.0 - costh * costh))
        Dij = df / (max(eps, sinth) * Rij)
        Dkj = df / (max(eps, sinth) * Rkj)
        vec_dij = Dij * (costh * eij - ekj)
        vec_dkj = Dkj * (costh * ekj - eij)
        grad[self.aid0] += vec_dij
        grad[self.aid1] -= vec_dij
        grad[self.aid2] += vec_dkj
        grad[self.aid1] -= vec_dkj

    def print(self, crds: np.ndarray) -> None:
        theta, _, _, _, _, _ = self._calc_angle_impl(self.aid0, self.aid1, self.aid2, crds)
        print(
            f"A {self.aid0}-{self.aid1}-{self.aid2}:"
            f" cur {math.degrees(theta):.2f}"
            f" ref {math.degrees(self.th0):.2f}"
            f" dif {math.degrees(theta - self.th0):.2f}"
        )

    def calc_sd(self, crds: np.ndarray) -> float:
        theta, _, _, _, _, _ = self._calc_angle_impl(self.aid0, self.aid1, self.aid2, crds)
        return math.degrees(theta - self.th0) ** 2
