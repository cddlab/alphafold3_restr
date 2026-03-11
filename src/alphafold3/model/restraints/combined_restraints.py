# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""CombinedRestraints: manages conformer and distance restraints during diffusion.

Restraints are applied after each denoising step via a JAX minimizer,
compatible with hk.vmap + hk.scan and aligned with Protenix's step-wise flow.

Coordinate layout (AF3):
  positions: (num_tokens, max_atoms_per_token, 3) per sample inside the scan.
  flat_idx = token_idx * max_atoms_per_token + within_token_idx
  For ligands: within_token_idx = 0 always.

Usage:
  # Build at inference time, after featurisation:
  restraints = CombinedRestraints.from_config(restraint_config, fold_input, batch)
  # Pass to model for injection inside diffusion loop
  model_output = model(batch, restraints=restraints)
"""
from __future__ import annotations

import dataclasses
import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from alphafold3.model.restraints import jax_energy
from alphafold3.model.restraints.selection import AtomSelector


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class RestraintConfig:
  """Top-level restraint configuration.

  Attributes:
    use_gpu: Legacy compatibility flag retained in the input schema.
    start_sigma: Only apply restraints when noise level <= start_sigma.
    max_iter: Number of gradient descent steps.
    learning_rate: Step size for gradient descent.
    method: Legacy compatibility field.
    verbose: Print statistics.
    conformer_restraints_config: Sub-config for conformer restraints.
      Keys: 'enabled' (bool), 'bond' (dict), 'angle' (dict), 'chiral' (dict).
    distance_restraints_config: List of distance restraint specs.
      Each spec: 'atom_selection1', 'atom_selection2', 'harmonic'/'flat-bottomed'/etc.
  """
  use_gpu: bool = False
  start_sigma: float = 1.0
  max_iter: int = 100
  learning_rate: float = 0.01
  method: str = 'CG'
  verbose: bool = False
  conformer_restraints_config: dict = dataclasses.field(default_factory=dict)
  distance_restraints_config: list = dataclasses.field(default_factory=list)

  @classmethod
  def from_dict(cls, d: dict) -> 'RestraintConfig':
    """Parse from the restraints_config JSON dict.

    Expected format:
      {
        "gpu": bool,
        "start_sigma": float,
        "max_iter": int,
        "learning_rate": float,
        "method": str,
        "verbose": bool,
        "conformer_restraints_config": {...},
        "distance_restraints_config": [...],
      }
    """
    return cls(
        use_gpu=d.get('gpu', False),
        start_sigma=float(d.get('start_sigma', 1.0)),
        max_iter=int(d.get('max_iter', 100)),
        learning_rate=float(d.get('learning_rate', 0.01)),
        method=d.get('method', 'CG'),
        verbose=bool(d.get('verbose', False)),
        conformer_restraints_config=d.get('conformer_restraints_config', {}),
        distance_restraints_config=d.get('distance_restraints_config', []),
    )


# ---------------------------------------------------------------------------
# Distance restraint data
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class DistanceRestraintData:
  """A single distance restraint between two atom groups."""
  atom_selection1: str
  atom_selection2: str
  distance_type: str   # 'harmonic', 'flat-bottomed', 'flat-bottomed1', 'flat-bottomed2'
  target1: float       # lower bound or harmonic target
  target2: float       # upper bound (only for flat-bottomed)
  weight: float = 1.0
  global_sites1: list[int] = dataclasses.field(default_factory=list)
  global_sites2: list[int] = dataclasses.field(default_factory=list)

  @classmethod
  def from_dict(cls, d: dict) -> 'DistanceRestraintData':
    if 'harmonic' in d:
      target1 = float(d['harmonic']['target_distance'])
      target2 = target1
      dtype = 'harmonic'
    elif 'flat-bottomed' in d:
      target1 = float(d['flat-bottomed']['target_distance1'])
      target2 = float(d['flat-bottomed']['target_distance2'])
      dtype = 'flat-bottomed'
    elif 'flat-bottomed1' in d:
      target1 = float(d['flat-bottomed1']['target_distance1'])
      target2 = target1
      dtype = 'flat-bottomed1'
    elif 'flat-bottomed2' in d:
      target2 = float(d['flat-bottomed2']['target_distance2'])
      target1 = target2
      dtype = 'flat-bottomed2'
    else:
      raise ValueError(f'No valid distance type in {d}')
    return cls(
        atom_selection1=d['atom_selection1'],
        atom_selection2=d['atom_selection2'],
        distance_type=dtype,
        target1=target1,
        target2=target2,
        weight=float(d.get('weight', 1.0)),
    )

  @property
  def type_code(self) -> int:
    return {'harmonic': 0, 'flat-bottomed': 1, 'flat-bottomed1': 2, 'flat-bottomed2': 3}[
        self.distance_type
    ]


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CombinedRestraints:
  """Manages conformer and distance restraints for guided diffusion."""

  def __init__(
      self,
      config: RestraintConfig,
      active_sites: np.ndarray,
      conformer_raw: dict | None,
      distance_restraints: list[DistanceRestraintData],
      max_atoms_per_token: int,
  ):
    """Do not call directly. Use from_config()."""
    self.config = config
    self.active_sites = active_sites            # (n_active,) global flat indices, sorted
    self.conformer_raw = conformer_raw          # raw flat-index arrays from ConformerFeaturizer
    self.distance_restraints = distance_restraints
    self.max_atoms_per_token = max_atoms_per_token

    # Derived: local index remapping (global_flat → local position in active_sites)
    self._global_to_local: dict[int, int] = {
        int(g): l for l, g in enumerate(active_sites)
    }
    self.n_active = len(active_sites)
    # JAX arrays used by the in-scan minimizer.
    self._jax_conformer: dict | None = None
    self._jax_distance: dict | None = None

    if self.n_active > 0:
      self._build_jax_arrays()

  # ------------------------------------------------------------------
  # Construction
  # ------------------------------------------------------------------

  @classmethod
  def from_config(
      cls,
      config: RestraintConfig,
      batch_dict: dict[str, Any],
      conformer_raw_data: dict | None = None,
      distance_raw_data: list[DistanceRestraintData] | None = None,
  ) -> 'CombinedRestraints':
    """Build CombinedRestraints from config and batch information.

    Args:
      config: RestraintConfig.
      batch_dict: The features.BatchDict (numpy arrays). Must contain
        'ref_mask' with shape (num_tokens, max_atoms_per_token).
      conformer_raw_data: Output from ConformerFeaturizer.build(). Contains
        'bond', 'angle', 'chiral' dicts with 'flat_i', 'flat_j', etc.
        Also 'all_flat_indices'.
      distance_raw_data: List of DistanceRestraintData with global_sites resolved.

    Returns:
      CombinedRestraints instance.
    """
    ref_mask = batch_dict['ref_mask']  # (num_tokens, max_atoms_per_token)
    max_atoms_per_token = ref_mask.shape[-1]

    # Collect all active flat indices
    all_flat = set()

    if conformer_raw_data is not None:
      flat_idx_arr = conformer_raw_data.get('all_flat_indices')
      if flat_idx_arr is not None:
        all_flat.update(int(x) for x in flat_idx_arr)

    distance_restraints = distance_raw_data or []
    for dr in distance_restraints:
      all_flat.update(dr.global_sites1)
      all_flat.update(dr.global_sites2)

    active_sites = np.array(sorted(all_flat), dtype=np.int32)

    if config.verbose:
      def _count(d, key):
        return 0 if d is None or d.get(key) is None else len(next(iter(d[key].values())))
      n_bonds = _count(conformer_raw_data, 'bond')
      n_angles = _count(conformer_raw_data, 'angle')
      n_chirals = _count(conformer_raw_data, 'chiral')
      n_vdws = _count(conformer_raw_data, 'vdw')
      print(f'[CombinedRestraints] active_sites={len(active_sites)}, '
            f'bonds={n_bonds}, angles={n_angles}, chirals={n_chirals}, '
            f'vdw={n_vdws}, distance_restraints={len(distance_restraints)}')

    return cls(
        config=config,
        active_sites=active_sites,
        conformer_raw=conformer_raw_data,
        distance_restraints=distance_restraints,
        max_atoms_per_token=max_atoms_per_token,
    )

  def is_active(self) -> bool:
    return self.n_active > 0

  # ------------------------------------------------------------------
  # Global-to-local index conversion
  # ------------------------------------------------------------------

  def _to_local(self, global_idx: int) -> int:
    return self._global_to_local[global_idx]

  def _scatter_active_positions_numpy(
      self,
      pos_flat: np.ndarray,
      optimized_active: np.ndarray,
  ) -> np.ndarray:
    """Scatter optimized active coordinates back to the full flat layout.

    Protenix applies restraints directly to the selected active atoms.
    """
    pos_flat_new = pos_flat.copy()
    pos_flat_new[self.active_sites] = optimized_active.astype(pos_flat.dtype)
    return pos_flat_new

  def _scatter_active_positions_jax(
      self,
      pos_flat: jnp.ndarray,
      optimized_active: jnp.ndarray,
  ) -> jnp.ndarray:
    """JAX equivalent of `_scatter_active_positions_numpy`."""
    pos_flat_new = pos_flat.at[jnp.array(self.active_sites, dtype=jnp.int32)].set(
        optimized_active
    )
    return pos_flat_new

  def _flat_pair_to_local(self, flat_i: np.ndarray, flat_j: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    li = np.array([self._global_to_local[int(x)] for x in flat_i], dtype=np.int32)
    lj = np.array([self._global_to_local[int(x)] for x in flat_j], dtype=np.int32)
    return li, lj

  def _flat_triple_to_local(
      self, fi: np.ndarray, fj: np.ndarray, fk: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    li = np.array([self._global_to_local[int(x)] for x in fi], dtype=np.int32)
    lj = np.array([self._global_to_local[int(x)] for x in fj], dtype=np.int32)
    lk = np.array([self._global_to_local[int(x)] for x in fk], dtype=np.int32)
    return li, lj, lk

  def _flat_quad_to_local(
      self, fc: np.ndarray, f1: np.ndarray, f2: np.ndarray, f3: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    lc = np.array([self._global_to_local[int(x)] for x in fc], dtype=np.int32)
    l1 = np.array([self._global_to_local[int(x)] for x in f1], dtype=np.int32)
    l2 = np.array([self._global_to_local[int(x)] for x in f2], dtype=np.int32)
    l3 = np.array([self._global_to_local[int(x)] for x in f3], dtype=np.int32)
    return lc, l1, l2, l3

  # ------------------------------------------------------------------
  # JAX array construction
  # ------------------------------------------------------------------

  def _build_jax_arrays(self) -> None:
    """Convert raw flat-index arrays to local-indexed JAX arrays."""
    jax_conf = {}

    if self.conformer_raw is not None:
      b = self.conformer_raw.get('bond')
      if b is not None:
        li, lj = self._flat_pair_to_local(b['flat_i'], b['flat_j'])
        jax_conf['bond'] = dict(
            idx=jnp.array(np.stack([li, lj], axis=1), dtype=jnp.int32),
            r0=jnp.array(b['r0']),
            slack=jnp.array(b['slack']),
            weight=jnp.array(b['weight']),
            mask=jnp.ones(len(li), dtype=jnp.float32),
        )

      a = self.conformer_raw.get('angle')
      if a is not None:
        li, lj, lk = self._flat_triple_to_local(a['flat_i'], a['flat_j'], a['flat_k'])
        jax_conf['angle'] = dict(
            idx=jnp.array(np.stack([li, lj, lk], axis=1), dtype=jnp.int32),
            th0=jnp.array(a['th0']),
            slack=jnp.array(a['slack']),
            weight=jnp.array(a['weight']),
            mask=jnp.ones(len(li), dtype=jnp.float32),
        )

      c = self.conformer_raw.get('chiral')
      if c is not None:
        lc, l1, l2, l3 = self._flat_quad_to_local(
            c['flat_center'], c['flat_n1'], c['flat_n2'], c['flat_n3']
        )
        jax_conf['chiral'] = dict(
            idx=jnp.array(np.stack([lc, l1, l2, l3], axis=1), dtype=jnp.int32),
            vol0=jnp.array(c['vol0']),
            slack=jnp.array(c['slack']),
            weight=jnp.array(c['weight']),
            mask=jnp.ones(len(lc), dtype=jnp.float32),
        )

      v = self.conformer_raw.get('vdw')
      if v is not None:
        li, lj = self._flat_pair_to_local(v['flat_i'], v['flat_j'])
        jax_conf['vdw'] = dict(
            idx=jnp.array(np.stack([li, lj], axis=1), dtype=jnp.int32),
            r_min=jnp.array(v['r_min']),
            weight=jnp.array(v['weight']),
            mask=jnp.ones(len(li), dtype=jnp.float32),
        )

    self._jax_conformer = jax_conf if jax_conf else None

    # Distance restraints to JAX arrays
    if self.distance_restraints:
      max_grp = max(
          max(len(dr.global_sites1), len(dr.global_sites2))
          for dr in self.distance_restraints
      )
      n_dist = len(self.distance_restraints)
      grp1_idx = np.zeros((n_dist, max_grp), dtype=np.int32)
      grp2_idx = np.zeros((n_dist, max_grp), dtype=np.int32)
      grp1_mask = np.zeros((n_dist, max_grp), dtype=np.float32)
      grp2_mask = np.zeros((n_dist, max_grp), dtype=np.float32)
      target1 = np.zeros(n_dist, dtype=np.float32)
      target2 = np.zeros(n_dist, dtype=np.float32)
      dist_type = np.zeros(n_dist, dtype=np.int32)
      weight = np.zeros(n_dist, dtype=np.float32)

      for i, dr in enumerate(self.distance_restraints):
        for j, g in enumerate(dr.global_sites1):
          grp1_idx[i, j] = self._global_to_local[g]
          grp1_mask[i, j] = 1.0
        for j, g in enumerate(dr.global_sites2):
          grp2_idx[i, j] = self._global_to_local[g]
          grp2_mask[i, j] = 1.0
        target1[i] = dr.target1
        target2[i] = dr.target2
        dist_type[i] = dr.type_code
        weight[i] = dr.weight

      self._jax_distance = dict(
          grp1_idx=jnp.array(grp1_idx),
          grp2_idx=jnp.array(grp2_idx),
          grp1_mask=jnp.array(grp1_mask),
          grp2_mask=jnp.array(grp2_mask),
          target1=jnp.array(target1),
          target2=jnp.array(target2),
          dist_type=jnp.array(dist_type),
          weight=jnp.array(weight),
          mask=jnp.ones(n_dist, dtype=jnp.float32),
      )
    else:
      self._jax_distance = None

  # ------------------------------------------------------------------
  # In-scan minimization
  # ------------------------------------------------------------------

  def minimize_gpu(
      self,
      positions: jnp.ndarray,
      sigma_t: jnp.ndarray,
  ) -> jnp.ndarray:
    """Apply JAX gradient descent on active atoms.

    Called per-sample inside apply_denoising_step (after hk.vmap).

    Args:
      positions: (num_tokens, max_atoms_per_token, 3) single-sample positions.
      sigma_t: scalar noise level at current step.

    Returns:
      Updated positions with same shape.
    """
    if not self.is_active():
      return positions

    active_sites_jax = jnp.array(self.active_sites, dtype=jnp.int32)
    n_active = self.n_active
    max_per = self.max_atoms_per_token

    def do_minimize(pos):
      # Flatten to (N_flat, 3)
      pos_flat = pos.reshape(-1, 3)
      # Extract active sites
      active_pos = pos_flat[active_sites_jax]  # (n_active, 3)
      x0 = active_pos.reshape(-1)

      # Build energy function capturing JAX arrays in closure
      jax_conf = self._jax_conformer
      jax_dist = self._jax_distance

      energy_fn = functools.partial(
          jax_energy.total_energy,
          n_active=n_active,
          conformer_data=jax_conf,
          distance_data=jax_dist,
      )

      # Gradient descent
      x_opt = jax_energy.minimize_gradient_descent(
          x0, energy_fn, self.config.max_iter, self.config.learning_rate
      )

      # Scatter back
      optimized_active = x_opt.reshape(n_active, 3)
      pos_flat_new = self._scatter_active_positions_jax(pos_flat, optimized_active)
      return pos_flat_new.reshape(pos.shape)

    # Gate on sigma_t: only minimize when noise level is low enough
    return jax.lax.cond(
        sigma_t <= jnp.array(self.config.start_sigma, dtype=sigma_t.dtype),
        do_minimize,
        lambda pos: pos,
        positions,
    )

  # ------------------------------------------------------------------
  # CPU mode: called after the full diffusion loop
  # ------------------------------------------------------------------

  def _numpy_energy(self, crds_flat: np.ndarray) -> float:
    """CPU numpy energy function for scipy."""
    crds = crds_flat.reshape(self.n_active, 3)
    ene = 0.0

    if self.conformer_raw is not None:
      b = self.conformer_raw.get('bond')
      if b is not None:
        li = np.array([self._global_to_local[int(x)] for x in b['flat_i']])
        lj = np.array([self._global_to_local[int(x)] for x in b['flat_j']])
        diff = crds[li] - crds[lj]
        dist = np.linalg.norm(diff, axis=-1)
        r2 = b['r0'] + b['slack']
        r1 = b['r0'] - b['slack']
        delta = np.where(dist > r2, dist - r2, np.where(dist < r1, dist - r1, 0.0))
        ene += np.sum(b['weight'] * delta ** 2)

      a = self.conformer_raw.get('angle')
      if a is not None:
        li = np.array([self._global_to_local[int(x)] for x in a['flat_i']])
        lj = np.array([self._global_to_local[int(x)] for x in a['flat_j']])
        lk = np.array([self._global_to_local[int(x)] for x in a['flat_k']])
        rij = crds[li] - crds[lj]
        rkj = crds[lk] - crds[lj]
        n_ij = np.linalg.norm(rij, axis=-1, keepdims=True)
        n_kj = np.linalg.norm(rkj, axis=-1, keepdims=True)
        cos_th = np.sum(rij * rkj, axis=-1) / (n_ij.squeeze() * n_kj.squeeze() + 1e-12)
        cos_th = np.clip(cos_th, -1.0, 1.0)
        theta = np.arccos(cos_th)
        th2 = a['th0'] + a['slack']
        th1 = a['th0'] - a['slack']
        delta = np.where(theta > th2, theta - th2, np.where(theta < th1, theta - th1, 0.0))
        ene += np.sum(a['weight'] * delta ** 2)

      c = self.conformer_raw.get('chiral')
      if c is not None:
        lc = np.array([self._global_to_local[int(x)] for x in c['flat_center']])
        l1 = np.array([self._global_to_local[int(x)] for x in c['flat_n1']])
        l2 = np.array([self._global_to_local[int(x)] for x in c['flat_n2']])
        l3 = np.array([self._global_to_local[int(x)] for x in c['flat_n3']])
        for i in range(len(lc)):
          a0, a1, a2, a3 = crds[lc[i]], crds[l1[i]], crds[l2[i]], crds[l3[i]]
          v1 = a1 - a0; v2 = a2 - a0; v3 = a3 - a0
          vol = np.dot(v1, np.cross(v2, v3))
          thr = c['vol0'][i] - c['slack'][i] if c['vol0'][i] > 0 else c['vol0'][i] + c['slack'][i]
          delta = vol - thr
          ene += c['weight'][i] * delta ** 2

      vw = self.conformer_raw.get('vdw')
      if vw is not None:
        li = np.array([self._global_to_local[int(x)] for x in vw['flat_i']])
        lj = np.array([self._global_to_local[int(x)] for x in vw['flat_j']])
        diff = crds[li] - crds[lj]
        dist = np.linalg.norm(diff, axis=-1)
        delta = np.minimum(0.0, dist - vw['r_min'])
        ene += np.sum(vw['weight'] * delta ** 2)

    for dr in self.distance_restraints:
      ls1 = [self._global_to_local[g] for g in dr.global_sites1]
      ls2 = [self._global_to_local[g] for g in dr.global_sites2]
      com1 = np.mean(crds[ls1], axis=0)
      com2 = np.mean(crds[ls2], axis=0)
      d = np.linalg.norm(com2 - com1)
      if dr.distance_type == 'harmonic':
        delta = d - dr.target1
      elif dr.distance_type in ('flat-bottomed', 'flat-bottomed1') and d < dr.target1:
        delta = d - dr.target1
      elif dr.distance_type in ('flat-bottomed', 'flat-bottomed2') and d > dr.target2:
        delta = d - dr.target2
      else:
        delta = 0.0
      ene += delta ** 2

    return ene

  def _numpy_grad(self, crds_flat: np.ndarray) -> np.ndarray:
    """CPU numpy gradient for scipy (finite differences)."""
    eps = 1e-4
    grad = np.zeros_like(crds_flat)
    for i in range(len(crds_flat)):
      x_plus = crds_flat.copy(); x_plus[i] += eps
      x_minus = crds_flat.copy(); x_minus[i] -= eps
      grad[i] = (self._numpy_energy(x_plus) - self._numpy_energy(x_minus)) / (2 * eps)
    return grad

  def minimize_cpu(self, positions_np: np.ndarray) -> np.ndarray:
    """Apply scipy CG minimization to a single sample's positions.

    Args:
      positions_np: (num_tokens, max_atoms_per_token, 3) numpy array.

    Returns:
      Optimized positions with same shape.
    """
    from scipy import optimize

    if not self.is_active():
      return positions_np

    active_sites = self.active_sites
    pos_flat = positions_np.reshape(-1, 3)
    active_pos = pos_flat[active_sites]
    x0 = active_pos.reshape(-1).astype(np.float64)

    if self.config.verbose:
      e0 = self._numpy_energy(x0)
      print(f'[CombinedRestraints CPU] initial energy: {e0:.5f}')

    opt = optimize.minimize(
        self._numpy_energy,
        x0,
        jac=self._numpy_grad,
        method=self.config.method,
        options={'maxiter': self.config.max_iter},
    )

    if self.config.verbose:
      print(f'[CombinedRestraints CPU] final energy: {opt.fun:.5f}, '
            f'success={opt.success}')

    optimized = opt.x.reshape(len(active_sites), 3)
    pos_flat_new = self._scatter_active_positions_numpy(pos_flat, optimized)
    return pos_flat_new.reshape(positions_np.shape)

  def apply_cpu_postprocess(
      self, samples: dict[str, Any]
  ) -> dict[str, Any]:
    """Apply CPU scipy refinement to all diffusion samples.

    Called after sample() returns, outside the JAX JIT context.

    Args:
      samples: dict with 'atom_positions' of shape
        (num_samples, num_tokens, max_atoms_per_token, 3).

    Returns:
      Updated samples dict with refined atom_positions.
    """
    if not self.is_active():
      return samples

    atom_positions = np.array(samples['atom_positions'])
    num_samples = atom_positions.shape[0]

    for s in range(num_samples):
      atom_positions[s] = self.minimize_cpu(atom_positions[s])

    # Return numpy array so downstream structure_tables code can set flags.writeable.
    return {**samples, 'atom_positions': atom_positions}

  # ------------------------------------------------------------------
  # Resolve distance restraint selections against batch layout
  # ------------------------------------------------------------------

  @staticmethod
  def resolve_distance_restraints(
      distance_configs: list[dict],
      token_asym_ids: np.ndarray,
      ref_mask: np.ndarray,
      chain_id_to_asym_int: dict[str, int],
      max_atoms_per_token: int,
  ) -> list[DistanceRestraintData]:
    """Resolve atom selection strings to global flat indices.

    Args:
      distance_configs: list of distance restraint spec dicts.
      token_asym_ids: (num_tokens,) int array, asym_id per token.
      chain_id_to_asym_int: mapping chain letter → asym_id int.
      max_atoms_per_token: positions layout second dim.

    Returns:
      List of DistanceRestraintData with global_sites resolved.
    """
    results = []
    for spec in distance_configs:
      dr = DistanceRestraintData.from_dict(spec)
      sel1 = AtomSelector(dr.atom_selection1)
      sel2 = AtomSelector(dr.atom_selection2)

      sites1, sites2 = [], []
      # Build chain letter for each token from asym_id int
      asym_int_to_chain = {v: k for k, v in chain_id_to_asym_int.items()}
      for token_idx, asym_int in enumerate(token_asym_ids):
        chain_letter = asym_int_to_chain.get(int(asym_int), '')
        resid = token_idx + 1
        for within_token_idx in range(max_atoms_per_token):
          if not bool(ref_mask[token_idx, within_token_idx]):
            continue
          flat_idx = int(token_idx) * max_atoms_per_token + within_token_idx
          candidate = {
              'chain': chain_letter,
              'resid': resid,
              'index': flat_idx,
          }
          if sel1.matches(candidate):
            sites1.append(flat_idx)
          if sel2.matches(candidate):
            sites2.append(flat_idx)

      if not sites1:
        raise ValueError(f'atom_selection1 "{dr.atom_selection1}" matched no atoms')
      if not sites2:
        raise ValueError(f'atom_selection2 "{dr.atom_selection2}" matched no atoms')

      dr.global_sites1 = sites1
      dr.global_sites2 = sites2
      results.append(dr)
      if True:  # always print for now
        print(f'[DistanceRestraint] group1={len(sites1)} atoms, group2={len(sites2)} atoms')

    return results
