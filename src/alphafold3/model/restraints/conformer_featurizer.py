# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Build conformer restraints (bond/angle/chiral) from a ligand RDKit mol.

Given an RDKit molecule and the mapping from its atom indices to flat
position indices in the AF3 batch layout, this module generates the
bond/angle/chiral restraint arrays needed by CombinedRestraints.

AF3 atom layout:
  positions[token_idx, within_token_idx, :] for shape (num_tokens, max_per_token, 3)
  For ligands: each atom is its own token with within_token_idx = 0.
  flat_idx = token_idx * max_atoms_per_token + within_token_idx

Usage:
  featurizer = ConformerFeaturizer(mol, conf, flat_indices, config)
  data = featurizer.build()
  # data['bond'], data['angle'], data['chiral'] are dict of numpy arrays
"""
from __future__ import annotations

import itertools
import math

import numpy as np

try:
  from rdkit import Chem
  from rdkit.Chem import AllChem
  _HAS_RDKIT = True
except ImportError:
  _HAS_RDKIT = False


# VDW radii in Angstroms, keyed by atomic number.
# Values from Bondi (1964) / Alvarez (2013).
_VDW_RADII: dict[int, float] = {
    1: 1.20,   # H
    6: 1.70,   # C
    7: 1.55,   # N
    8: 1.52,   # O
    9: 1.47,   # F
    14: 2.10,  # Si
    15: 1.80,  # P
    16: 1.80,  # S
    17: 1.75,  # Cl
    35: 1.85,  # Br
    53: 1.98,  # I
}
_DEFAULT_VDW_RADIUS = 1.70  # fallback for unlisted elements


def _get_vdw_radius(atomic_num: int) -> float:
  return _VDW_RADII.get(atomic_num, _DEFAULT_VDW_RADIUS)


_ANGLE_PATTERN = None


def _get_angle_pattern():
  global _ANGLE_PATTERN
  if _ANGLE_PATTERN is None:
    _ANGLE_PATTERN = Chem.MolFromSmarts('*~*~*')
  return _ANGLE_PATTERN


def _get_angle_triples(mol: 'Chem.Mol') -> list[tuple[int, int, int]]:
  """Return all (ai, aj_vertex, ak) angle triples in mol."""
  patt = _get_angle_pattern()
  matches = mol.GetSubstructMatches(patt)
  return [(int(m[0]), int(m[1]), int(m[2])) for m in matches]


def _calc_bond_length(crds: np.ndarray, ai: int, aj: int) -> float:
  return float(np.linalg.norm(crds[aj] - crds[ai]))


def _calc_angle_rad(crds: np.ndarray, ai: int, aj: int, ak: int) -> float:
  rij = crds[ai] - crds[aj]
  rkj = crds[ak] - crds[aj]
  n_ij = np.linalg.norm(rij)
  n_kj = np.linalg.norm(rkj)
  if n_ij < 1e-8 or n_kj < 1e-8:
    return 0.0
  cos_th = np.clip(np.dot(rij, rkj) / (n_ij * n_kj), -1.0, 1.0)
  return float(math.acos(cos_th))


def _calc_chiral_vol(crds: np.ndarray, ai: int, aj: list[int]) -> float:
  vc = crds[ai]
  v1 = crds[aj[0]] - vc
  v2 = crds[aj[1]] - vc
  v3 = crds[aj[2]] - vc
  return float(np.dot(v1, np.cross(v2, v3)))


class ConformerFeaturizer:
  """Build bond/angle/chiral/vdw restraint arrays from an RDKit ligand molecule.

  Args:
    mol: RDKit mol (with Hs removed, used for topology).
    conf_crds: (n_atoms, 3) reference conformer coordinates (from ref_pos batch).
    flat_indices: (n_atoms,) mapping from mol atom index → flat batch position index.
    bond_config: dict with keys 'slack', 'weight' for bond restraints.
    angle_config: dict with keys 'slack', 'weight' for angle restraints.
    chiral_config: dict with keys 'slack', 'weight', 'f_max' for chiral restraints.
    vdw_config: dict with keys 'weight', 'scale', 'dmax' for VDW repulsion.
      scale: multiplier on sum of VDW radii (default 0.75).
      dmax: max reference distance to consider a pair (default 5.0 Å).
    verbose: print debug info.
  """

  def __init__(
      self,
      mol: 'Chem.Mol',
      conf_crds: np.ndarray,
      flat_indices: np.ndarray,
      bond_config: dict | None = None,
      angle_config: dict | None = None,
      chiral_config: dict | None = None,
      vdw_config: dict | None = None,
      verbose: bool = False,
  ):
    self.mol = mol
    self.conf_crds = conf_crds  # (n_atoms, 3)
    self.flat_indices = flat_indices  # (n_atoms,) global flat indices
    self.bond_config = bond_config or {}
    self.angle_config = angle_config or {}
    self.chiral_config = chiral_config or {}
    self.vdw_config = vdw_config or {}
    self.verbose = verbose

    # Lists of restraint dicts
    self._bonds: list[dict] = []
    self._angles: list[dict] = []
    self._chirals: list[dict] = []
    self._vdws: list[dict] = []

  # ------------------------------------------------------------------
  # Geometry extraction
  # ------------------------------------------------------------------

  def _add_bond(self, rdkit_ai: int, rdkit_aj: int) -> None:
    r0 = _calc_bond_length(self.conf_crds, rdkit_ai, rdkit_aj)
    self._bonds.append(dict(
        flat_i=int(self.flat_indices[rdkit_ai]),
        flat_j=int(self.flat_indices[rdkit_aj]),
        r0=r0,
        slack=float(self.bond_config.get('slack', 0.0)),
        weight=float(self.bond_config.get('weight', 0.05)),
    ))

  def _add_angle(self, rdkit_ai: int, rdkit_aj: int, rdkit_ak: int) -> None:
    th0 = _calc_angle_rad(self.conf_crds, rdkit_ai, rdkit_aj, rdkit_ak)
    self._angles.append(dict(
        flat_i=int(self.flat_indices[rdkit_ai]),
        flat_j=int(self.flat_indices[rdkit_aj]),
        flat_k=int(self.flat_indices[rdkit_ak]),
        th0=th0,
        slack=float(self.angle_config.get('slack', math.radians(5.0))),
        weight=float(self.angle_config.get('weight', 0.05)),
    ))

  def _add_chiral(
      self, rdkit_center: int, rdkit_neighbors: list[int], invert: bool = False
  ) -> None:
    vol = _calc_chiral_vol(self.conf_crds, rdkit_center, rdkit_neighbors)
    if invert:
      vol = -vol
    self._chirals.append(dict(
        flat_center=int(self.flat_indices[rdkit_center]),
        flat_n1=int(self.flat_indices[rdkit_neighbors[0]]),
        flat_n2=int(self.flat_indices[rdkit_neighbors[1]]),
        flat_n3=int(self.flat_indices[rdkit_neighbors[2]]),
        vol0=vol,
        slack=float(self.chiral_config.get('slack', 0.05)),
        weight=float(self.chiral_config.get('weight', 0.05)),
    ))
    if self.verbose:
      print(f'chiral center={rdkit_center} neighbors={rdkit_neighbors} vol={vol:.3f}')

  def _extract_bonds(self) -> None:
    for bond in self.mol.GetBonds():
      self._add_bond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

  def _extract_angles(self) -> None:
    for ai, aj, ak in _get_angle_triples(self.mol):
      self._add_angle(ai, aj, ak)

  def _extract_chirals(self) -> None:
    for atom in self.mol.GetAtoms():
      if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
        continue
      iatm = atom.GetIdx()
      nei_inds = [b.GetOtherAtom(atom).GetIdx() for b in atom.GetBonds()]
      # All combinations of 3 neighbors (Protenix approach)
      for cand in itertools.combinations(nei_inds, 3):
        self._add_chiral(iatm, list(cand))

  def _extract_vdw(self) -> None:
    """Add VDW repulsion restraints for non-bonded atom pairs.

    Skips 1,2 (bonded) and 1,3 (two-bond) pairs via topological distance.
    Only includes pairs whose reference distance < dmax.
    """
    from rdkit.Chem import rdmolops  # pylint: disable=g-import-not-at-top

    n_atoms = self.mol.GetNumAtoms()
    if n_atoms < 2:
      return

    scale = float(self.vdw_config.get('scale', 0.75))
    dmax = float(self.vdw_config.get('dmax', 5.0))
    weight = float(self.vdw_config.get('weight', 0.05))

    # Topological distance matrix: entry [i,j] = shortest bond path length.
    # 1,2 pairs have distance 1; 1,3 pairs have distance 2.
    topo_dist = rdmolops.GetDistanceMatrix(self.mol)

    for i in range(n_atoms):
      ri = _get_vdw_radius(self.mol.GetAtomWithIdx(i).GetAtomicNum())
      for j in range(i + 1, n_atoms):
        # Skip bonded (1,2) and angle (1,3) pairs.
        if topo_dist[i, j] <= 2:
          continue
        # Skip pairs far apart in the reference conformer.
        ref_dist = _calc_bond_length(self.conf_crds, i, j)
        if ref_dist >= dmax:
          continue
        rj = _get_vdw_radius(self.mol.GetAtomWithIdx(j).GetAtomicNum())
        r_min = scale * (ri + rj)
        self._vdws.append(dict(
            flat_i=int(self.flat_indices[i]),
            flat_j=int(self.flat_indices[j]),
            r_min=r_min,
            weight=weight,
        ))

  # ------------------------------------------------------------------
  # Build output arrays
  # ------------------------------------------------------------------

  def build(self) -> dict:
    """Build restraint arrays from the molecule.

    Returns:
      Dict with keys 'bond', 'angle', 'chiral'. Each value is a dict of
      numpy arrays with keys 'flat_indices_*', 'r0'/'th0'/'vol0', 'slack',
      'weight'. Returns None for each key if no restraints of that type exist.
      Also returns 'all_flat_indices': sorted array of all constrained flat indices.
    """
    self._extract_bonds()
    self._extract_angles()
    self._extract_chirals()
    if self.vdw_config:
      self._extract_vdw()

    if self.verbose:
      print(f'[ConformerFeaturizer] bonds={len(self._bonds)}, '
            f'angles={len(self._angles)}, chirals={len(self._chirals)}, '
            f'vdw={len(self._vdws)}')

    result = {}
    all_flat = set()

    if self._bonds:
      result['bond'] = dict(
          flat_i=np.array([b['flat_i'] for b in self._bonds], dtype=np.int32),
          flat_j=np.array([b['flat_j'] for b in self._bonds], dtype=np.int32),
          r0=np.array([b['r0'] for b in self._bonds], dtype=np.float32),
          slack=np.array([b['slack'] for b in self._bonds], dtype=np.float32),
          weight=np.array([b['weight'] for b in self._bonds], dtype=np.float32),
      )
      all_flat.update(b['flat_i'] for b in self._bonds)
      all_flat.update(b['flat_j'] for b in self._bonds)
    else:
      result['bond'] = None

    if self._angles:
      result['angle'] = dict(
          flat_i=np.array([a['flat_i'] for a in self._angles], dtype=np.int32),
          flat_j=np.array([a['flat_j'] for a in self._angles], dtype=np.int32),
          flat_k=np.array([a['flat_k'] for a in self._angles], dtype=np.int32),
          th0=np.array([a['th0'] for a in self._angles], dtype=np.float32),
          slack=np.array([a['slack'] for a in self._angles], dtype=np.float32),
          weight=np.array([a['weight'] for a in self._angles], dtype=np.float32),
      )
      all_flat.update(a['flat_i'] for a in self._angles)
      all_flat.update(a['flat_j'] for a in self._angles)
      all_flat.update(a['flat_k'] for a in self._angles)
    else:
      result['angle'] = None

    if self._chirals:
      result['chiral'] = dict(
          flat_center=np.array([c['flat_center'] for c in self._chirals], dtype=np.int32),
          flat_n1=np.array([c['flat_n1'] for c in self._chirals], dtype=np.int32),
          flat_n2=np.array([c['flat_n2'] for c in self._chirals], dtype=np.int32),
          flat_n3=np.array([c['flat_n3'] for c in self._chirals], dtype=np.int32),
          vol0=np.array([c['vol0'] for c in self._chirals], dtype=np.float32),
          slack=np.array([c['slack'] for c in self._chirals], dtype=np.float32),
          weight=np.array([c['weight'] for c in self._chirals], dtype=np.float32),
      )
      all_flat.update(c['flat_center'] for c in self._chirals)
      all_flat.update(c['flat_n1'] for c in self._chirals)
      all_flat.update(c['flat_n2'] for c in self._chirals)
      all_flat.update(c['flat_n3'] for c in self._chirals)
    else:
      result['chiral'] = None

    if self._vdws:
      result['vdw'] = dict(
          flat_i=np.array([v['flat_i'] for v in self._vdws], dtype=np.int32),
          flat_j=np.array([v['flat_j'] for v in self._vdws], dtype=np.int32),
          r_min=np.array([v['r_min'] for v in self._vdws], dtype=np.float32),
          weight=np.array([v['weight'] for v in self._vdws], dtype=np.float32),
      )
      all_flat.update(v['flat_i'] for v in self._vdws)
      all_flat.update(v['flat_j'] for v in self._vdws)
    else:
      result['vdw'] = None

    result['all_flat_indices'] = np.array(sorted(all_flat), dtype=np.int32)
    return result


def build_ligand_flat_indices(
    chain_id: str,
    token_asym_ids: np.ndarray,
    chain_id_to_asym_int: dict[str, int],
    max_atoms_per_token: int,
) -> np.ndarray:
  """Compute flat batch indices for atoms of a ligand chain.

  For a ligand, each atom occupies its own token at within_token_idx=0.
  flat_idx = token_idx * max_atoms_per_token + 0

  Args:
    chain_id: Chain letter (e.g. 'B').
    token_asym_ids: (num_tokens,) int array of asym_id per token.
    chain_id_to_asym_int: mapping from chain letter to asym_id integer.
    max_atoms_per_token: second dim of the positions layout.

  Returns:
    (n_ligand_atoms,) int32 array of flat indices, in token order.
  """
  if chain_id not in chain_id_to_asym_int:
    raise ValueError(f'Chain ID "{chain_id}" not found in mapping: {chain_id_to_asym_int}')
  asym_int = chain_id_to_asym_int[chain_id]
  token_indices = np.where(token_asym_ids == asym_int)[0]
  # Each ligand atom: flat_idx = token_idx * max_atoms_per_token (within_token=0)
  flat_indices = (token_indices * max_atoms_per_token).astype(np.int32)
  return flat_indices
