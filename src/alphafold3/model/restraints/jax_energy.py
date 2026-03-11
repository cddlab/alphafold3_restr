# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Pure JAX energy and gradient functions for geometric restraints.

All functions operate on positions arrays of shape (n_active, 3),
where n_active is the number of restrained atoms (active sites).
Indices in bond/angle/chiral/distance arrays refer to positions WITHIN
the active sites array (local indices), not global flat indices.

Used by the in-scan JAX gradient-descent minimizer.
All functions are differentiable via jax.grad / jax.value_and_grad.
"""

from __future__ import annotations

import functools

import jax
import jax.numpy as jnp


# ---------------------------------------------------------------------------
# Flat-bottomed bond length energy
# ---------------------------------------------------------------------------

def bond_energy(
    positions: jnp.ndarray,
    idx: jnp.ndarray,
    r0: jnp.ndarray,
    slack: jnp.ndarray,
    weight: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Flat-bottomed bond length restraint energy.

  Args:
    positions: (n_active, 3) active atom coordinates.
    idx: (n_bonds, 2) int32 local indices of bonded atom pairs.
    r0: (n_bonds,) reference bond lengths.
    slack: (n_bonds,) allowed slack around r0 (flat bottom half-width).
    weight: (n_bonds,) energy weight per bond.
    mask: (n_bonds,) bool, 1 for valid bonds, 0 for padding.

  Returns:
    Scalar total bond energy.
  """
  ai, aj = idx[:, 0], idx[:, 1]
  diff = positions[ai] - positions[aj]  # (n_bonds, 3)
  dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)  # (n_bonds,)

  r_upper = r0 + slack
  r_lower = r0 - slack

  # Flat-bottomed: penalize outside [r_lower, r_upper]
  delta = jnp.where(dist > r_upper, dist - r_upper,
                    jnp.where(dist < r_lower, dist - r_lower, 0.0))
  energy_per_bond = weight * delta ** 2 * mask
  return jnp.sum(energy_per_bond)


# ---------------------------------------------------------------------------
# Flat-bottomed bond angle energy
# ---------------------------------------------------------------------------

def angle_energy(
    positions: jnp.ndarray,
    idx: jnp.ndarray,
    th0: jnp.ndarray,
    slack: jnp.ndarray,
    weight: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Flat-bottomed bond angle restraint energy.

  Args:
    positions: (n_active, 3) active atom coordinates.
    idx: (n_angles, 3) int32 local indices [ai, aj(vertex), ak].
    th0: (n_angles,) reference angles in radians.
    slack: (n_angles,) allowed deviation from th0 in radians.
    weight: (n_angles,) energy weight per angle.
    mask: (n_angles,) bool, 1 for valid angles.

  Returns:
    Scalar total angle energy.
  """
  ai, aj, ak = idx[:, 0], idx[:, 1], idx[:, 2]
  rij = positions[ai] - positions[aj]  # (n_angles, 3)
  rkj = positions[ak] - positions[aj]

  # Compute angle via dot product
  norm_ij = jnp.sqrt(jnp.sum(rij ** 2, axis=-1) + 1e-12)
  norm_kj = jnp.sqrt(jnp.sum(rkj ** 2, axis=-1) + 1e-12)
  cos_th = jnp.sum(rij * rkj, axis=-1) / (norm_ij * norm_kj)
  cos_th = jnp.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
  theta = jnp.arccos(cos_th)  # (n_angles,)

  th_upper = th0 + slack
  th_lower = th0 - slack

  delta = jnp.where(theta > th_upper, theta - th_upper,
                    jnp.where(theta < th_lower, theta - th_lower, 0.0))
  energy_per_angle = weight * delta ** 2 * mask
  return jnp.sum(energy_per_angle)


# ---------------------------------------------------------------------------
# Chiral volume energy
# ---------------------------------------------------------------------------

def chiral_energy(
    positions: jnp.ndarray,
    idx: jnp.ndarray,
    vol0: jnp.ndarray,
    slack: jnp.ndarray,
    weight: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """Chiral volume restraint energy (scalar triple product).

  For a chiral center a0 with neighbors a1, a2, a3:
    chiral_vol = dot(a1-a0, cross(a2-a0, a3-a0))

  The target threshold is:
    thr = vol0 - slack  if vol0 > 0
    thr = vol0 + slack  if vol0 <= 0

  Penalizes departure from the correct chirality.

  Args:
    positions: (n_active, 3) active atom coordinates.
    idx: (n_chirals, 4) int32 local indices [center, n1, n2, n3].
    vol0: (n_chirals,) reference chiral volumes.
    slack: (n_chirals,) allowed deviation.
    weight: (n_chirals,) energy weight.
    mask: (n_chirals,) bool, 1 for valid chirals.

  Returns:
    Scalar total chiral energy.
  """
  a0 = positions[idx[:, 0]]  # (n_chirals, 3)
  a1 = positions[idx[:, 1]]
  a2 = positions[idx[:, 2]]
  a3 = positions[idx[:, 3]]

  v1 = a1 - a0
  v2 = a2 - a0
  v3 = a3 - a0

  # Scalar triple product: dot(v1, cross(v2, v3))
  cross_v2_v3 = jnp.cross(v2, v3)  # (n_chirals, 3)
  vol = jnp.sum(v1 * cross_v2_v3, axis=-1)  # (n_chirals,)

  # Threshold: penalize if chirality is inverted
  thr = jnp.where(vol0 > 0, vol0 - slack, vol0 + slack)
  delta = vol - thr
  energy_per_chiral = weight * delta ** 2 * mask
  return jnp.sum(energy_per_chiral)


# ---------------------------------------------------------------------------
# VDW repulsion energy (non-bonded atom pair lower-bound)
# ---------------------------------------------------------------------------

def vdw_energy(
    positions: jnp.ndarray,
    idx: jnp.ndarray,
    r_min: jnp.ndarray,
    weight: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
  """VDW repulsion energy for non-bonded atom pairs.

  Penalizes only when the interatomic distance falls below r_min
  (lower-bound only, unlike the symmetric flat-bottomed bond potential).

    delta = min(0, d - r_min)
    E = weight * delta^2

  Args:
    positions: (n_active, 3) active atom coordinates.
    idx: (n_vdw, 2) int32 local indices of non-bonded pairs.
    r_min: (n_vdw,) minimum allowed distances (scale * (r_vdw_i + r_vdw_j)).
    weight: (n_vdw,) energy weight per pair.
    mask: (n_vdw,) bool, 1 for valid pairs.

  Returns:
    Scalar total VDW repulsion energy.
  """
  ai, aj = idx[:, 0], idx[:, 1]
  diff = positions[ai] - positions[aj]  # (n_vdw, 3)
  dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)  # (n_vdw,)

  # Penalize only when too close: delta <= 0 when d >= r_min.
  delta = jnp.minimum(0.0, dist - r_min)
  energy_per_pair = weight * delta ** 2 * mask
  return jnp.sum(energy_per_pair)


# ---------------------------------------------------------------------------
# Distance restraint energy (COM distance between two atom groups)
# ---------------------------------------------------------------------------

def distance_energy(
    positions: jnp.ndarray,
    grp1_idx: jnp.ndarray,
    grp2_idx: jnp.ndarray,
    grp1_mask: jnp.ndarray,
    grp2_mask: jnp.ndarray,
    target1: jnp.ndarray,
    target2: jnp.ndarray,
    dist_type: jnp.ndarray,
    weight: jnp.ndarray,
    restr_mask: jnp.ndarray,
) -> jnp.ndarray:
  """Distance restraint energy between centers of mass of two atom groups.

  Restraint types (encoded in dist_type):
    0 = harmonic:     penalize (d - target1)^2
    1 = flat-bottomed: penalize d < target1 or d > target2
    2 = lower-bound:  penalize d < target1
    3 = upper-bound:  penalize d > target2

  Args:
    positions: (n_active, 3) active atom coordinates.
    grp1_idx: (n_dist, max_grp) int32 local indices for group 1.
    grp2_idx: (n_dist, max_grp) int32 local indices for group 2.
    grp1_mask: (n_dist, max_grp) float, 1 for valid atoms in group 1.
    grp2_mask: (n_dist, max_grp) float, 1 for valid atoms in group 2.
    target1: (n_dist,) lower target distances (or harmonic target).
    target2: (n_dist,) upper target distances.
    dist_type: (n_dist,) int32 restraint type code.
    weight: (n_dist,) unused compatibility field.
    restr_mask: (n_dist,) bool, 1 for valid distance restraints.

  Returns:
    Scalar total distance energy.
  """
  # Compute COM for each group using masked mean
  # grp1_idx: (n_dist, max_grp)
  # positions: (n_active, 3)
  # grp_pos: (n_dist, max_grp, 3)
  grp1_pos = positions[grp1_idx]  # (n_dist, max_grp, 3)
  grp2_pos = positions[grp2_idx]

  # Masked mean: sum(pos * mask) / sum(mask)
  m1 = grp1_mask[..., None]  # (n_dist, max_grp, 1)
  m2 = grp2_mask[..., None]
  com1 = jnp.sum(grp1_pos * m1, axis=1) / (jnp.sum(grp1_mask, axis=1, keepdims=True) + 1e-12)
  com2 = jnp.sum(grp2_pos * m2, axis=1) / (jnp.sum(grp2_mask, axis=1, keepdims=True) + 1e-12)

  diff = com2 - com1  # (n_dist, 3)
  dist = jnp.sqrt(jnp.sum(diff ** 2, axis=-1) + 1e-12)  # (n_dist,)

  # Compute delta based on restraint type
  delta_harmonic = dist - target1
  delta_lower = jnp.minimum(0.0, dist - target1)  # penalize d < target1
  delta_upper = jnp.maximum(0.0, dist - target2)  # penalize d > target2
  # For flat-bottomed: penalize both below target1 AND above target2
  delta_flat = jnp.where(dist < target1, dist - target1,
                         jnp.where(dist > target2, dist - target2, 0.0))

  delta = jnp.where(dist_type == 0, delta_harmonic,
                    jnp.where(dist_type == 1, delta_flat,
                              jnp.where(dist_type == 2, jnp.minimum(0.0, dist - target1),
                                        delta_upper)))

  del weight
  energy_per_dist = delta ** 2 * restr_mask
  return jnp.sum(energy_per_dist)


# ---------------------------------------------------------------------------
# Combined energy function
# ---------------------------------------------------------------------------

def total_energy(
    x: jnp.ndarray,
    n_active: int,
    conformer_data: dict,
    distance_data: dict,
) -> jnp.ndarray:
  """Combined restraint energy for optimization.

  Args:
    x: (n_active * 3,) flattened active atom coordinates.
    n_active: number of active atoms (static).
    conformer_data: dict with keys 'bond', 'angle', 'chiral', each containing
      arrays (idx, r0/th0/vol0, slack, weight, mask).
    distance_data: dict with keys for distance restraints.

  Returns:
    Scalar total energy.
  """
  positions = x.reshape(n_active, 3)
  ene = jnp.array(0.0, dtype=x.dtype)

  if conformer_data is not None:
    if conformer_data.get('bond') is not None:
      b = conformer_data['bond']
      ene = ene + bond_energy(
          positions, b['idx'], b['r0'], b['slack'], b['weight'], b['mask']
      )

    if conformer_data.get('angle') is not None:
      a = conformer_data['angle']
      ene = ene + angle_energy(
          positions, a['idx'], a['th0'], a['slack'], a['weight'], a['mask']
      )

    if conformer_data.get('chiral') is not None:
      c = conformer_data['chiral']
      ene = ene + chiral_energy(
          positions, c['idx'], c['vol0'], c['slack'], c['weight'], c['mask']
      )

    if conformer_data.get('vdw') is not None:
      v = conformer_data['vdw']
      ene = ene + vdw_energy(
          positions, v['idx'], v['r_min'], v['weight'], v['mask']
      )

  if distance_data is not None:
    d = distance_data
    ene = ene + distance_energy(
        positions,
        d['grp1_idx'], d['grp2_idx'],
        d['grp1_mask'], d['grp2_mask'],
        d['target1'], d['target2'],
        d['dist_type'], d['weight'], d['mask'],
    )

  return ene


# ---------------------------------------------------------------------------
# GPU gradient descent minimizer
# ---------------------------------------------------------------------------

def minimize_gradient_descent(
    x0: jnp.ndarray,
    energy_fn,
    n_steps: int,
    learning_rate: float = 0.01,
) -> jnp.ndarray:
  """Simple gradient descent using jax.lax.fori_loop.

  Compatible with jax.vmap (unlike jax.lax.while_loop).

  Args:
    x0: (n_active * 3,) initial coordinates.
    energy_fn: callable taking x → scalar energy.
    n_steps: number of gradient descent steps (static).
    learning_rate: step size.

  Returns:
    Optimized coordinates (n_active * 3,).
  """
  grad_fn = jax.grad(energy_fn)

  def step(_, x):
    g = grad_fn(x)
    return x - learning_rate * g

  return jax.lax.fori_loop(0, n_steps, step, x0)
