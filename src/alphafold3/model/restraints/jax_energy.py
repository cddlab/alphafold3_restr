"""JAX-traceable energy functions for GPU restraint optimization.

Used by CombinedRestraints.minimize_single_sample_jax() when gpu=True.
All functions operate on the active-atom subset (local coordinates).
"""
from __future__ import annotations
import jax.numpy as jnp


def jax_bond_energy(
    positions: jnp.ndarray,
    aid0_arr: jnp.ndarray,
    aid1_arr: jnp.ndarray,
    r0_arr: jnp.ndarray,
    slack_arr: jnp.ndarray,
    w_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Flat-bottomed bond energy. positions: (N, 3), aids: (B,)"""
    v = positions[aid0_arr] - positions[aid1_arr]  # (B, 3)
    d = jnp.linalg.norm(v, axis=-1)  # (B,)
    r1 = r0_arr - slack_arr
    r2 = r0_arr + slack_arr
    below = jnp.where(d < r1, w_arr * (d - r1) ** 2, 0.0)
    above = jnp.where(d > r2, w_arr * (d - r2) ** 2, 0.0)
    return jnp.sum(below + above)


def jax_angle_energy(
    positions: jnp.ndarray,
    aid0_arr: jnp.ndarray,
    aid1_arr: jnp.ndarray,
    aid2_arr: jnp.ndarray,
    th0_arr: jnp.ndarray,
    slack_arr: jnp.ndarray,
    w_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Flat-bottomed angle energy. positions: (N, 3), aids: (B,)"""
    v_ij = positions[aid0_arr] - positions[aid1_arr]  # (B, 3)
    v_kj = positions[aid2_arr] - positions[aid1_arr]  # (B, 3)
    norm_ij = jnp.linalg.norm(v_ij, axis=-1, keepdims=True) + 1e-8
    norm_kj = jnp.linalg.norm(v_kj, axis=-1, keepdims=True) + 1e-8
    cos_th = jnp.sum(v_ij / norm_ij * v_kj / norm_kj, axis=-1)
    cos_th = jnp.clip(cos_th, -1.0 + 1e-7, 1.0 - 1e-7)
    theta = jnp.arccos(cos_th)  # (B,)
    th1 = th0_arr - slack_arr
    th2 = th0_arr + slack_arr
    below = jnp.where(theta < th1, w_arr * (theta - th1) ** 2, 0.0)
    above = jnp.where(theta > th2, w_arr * (theta - th2) ** 2, 0.0)
    return jnp.sum(below + above)


def jax_chiral_energy(
    positions: jnp.ndarray,
    aid0_arr: jnp.ndarray,
    aid1_arr: jnp.ndarray,
    aid2_arr: jnp.ndarray,
    aid3_arr: jnp.ndarray,
    chiral_vol_arr: jnp.ndarray,
    slack_arr: jnp.ndarray,
    w_arr: jnp.ndarray,
) -> jnp.ndarray:
    """Chiral volume energy. positions: (N, 3), aids: (B,)"""
    a0 = positions[aid0_arr]  # (B, 3)
    a1 = positions[aid1_arr]
    a2 = positions[aid2_arr]
    a3 = positions[aid3_arr]
    v1 = a1 - a0
    v2 = a2 - a0
    v3 = a3 - a0
    vol = jnp.sum(v1 * jnp.cross(v2, v3), axis=-1)  # (B,)
    # threshold: same sign as chiral_vol, slackened
    thr = jnp.where(chiral_vol_arr > 0, chiral_vol_arr - slack_arr, chiral_vol_arr + slack_arr)
    delta = vol - thr
    return jnp.sum(w_arr * delta ** 2)


def jax_distance_energy(
    positions: jnp.ndarray,
    sites1_arr: jnp.ndarray,
    sites2_arr: jnp.ndarray,
    target_distance: float | None,
    target_distance1: float | None,
    target_distance2: float | None,
    distance_restraint_type: str,
) -> jnp.ndarray:
    """COM distance energy for one DistanceData entry. positions: (N, 3)"""
    com1 = jnp.mean(positions[sites1_arr], axis=0)
    com2 = jnp.mean(positions[sites2_arr], axis=0)
    dist = jnp.linalg.norm(com2 - com1)
    zero = jnp.zeros(())
    rtype = distance_restraint_type
    if rtype == "harmonic":
        return (dist - target_distance) ** 2
    elif rtype == "flat-bottomed":
        below = jnp.where(dist < target_distance1, (dist - target_distance1) ** 2, zero)
        above = jnp.where(dist > target_distance2, (dist - target_distance2) ** 2, zero)
        return below + above
    elif rtype == "flat-bottomed1":
        return jnp.where(dist < target_distance1, (dist - target_distance1) ** 2, zero)
    elif rtype == "flat-bottomed2":
        return jnp.where(dist > target_distance2, (dist - target_distance2) ** 2, zero)
    return zero
