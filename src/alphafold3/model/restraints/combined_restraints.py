# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""AF3 restraint glue around the shared rgi_utils engine.

Spec construction (conformer bond/angle/chiral + intramolecular VdW + distance
restraints) now lives entirely in rgi_utils: ``AF3RestraintAdapter`` exposes the
AF3 batch through the rgi_utils adapter protocols and ``featurizer.build_spec``
builds the ``RestraintSpec``. This module keeps only the AF3-specific glue:

  - inject ``backend='jax'`` (AF3 runs the pure-JAX minimizer in the scan),
  - reshape positions ``(num_tokens, max_atoms_per_token, 3) <-> (-1, 3)`` around
    the pure-JAX minimizer applied after each denoising step.

The minimizer itself (jaxopt CG/LBFGS over an analytic energy, gated on the
noise level) is ``rgi_utils.optim.jax_optim.make_minimizer`` — no pure_callback,
no scipy — so the whole step stays JIT-compiled inside hk.scan/hk.vmap.
"""

from __future__ import annotations

import logging

from alphafold3.model.restraints.adapter import AF3RestraintAdapter

logger = logging.getLogger(__name__)


class AF3Restraints:
  """Applies the rgi_utils pure-JAX minimizer after each denoising step.

  Network code (model.py / diffusion_head.py) consumes this object purely by
  duck typing: ``is_active()`` and ``minimize_gpu(positions, sigma)``.
  """

  def __init__(self, rgi, minimizer):
    self._rgi = rgi  # rgi_utils CombinedRestraints (for is_active / stats)
    self._minimizer = minimizer  # pure (flat_coords, sigma) -> flat_coords | None

  def is_active(self) -> bool:
    return self._rgi.is_active() and self._minimizer is not None

  @property
  def n_active(self) -> int:
    """Number of optimised atoms (used by run_alphafold logging); 0 if none."""
    spec = getattr(self._rgi, 'spec', None)
    return int(spec.n_active) if spec is not None else 0

  def minimize_gpu(self, positions, sigma_t):
    """One restraint-minimization step on active atoms inside the diffusion scan.

    positions: ``(num_tokens, max_atoms_per_token, 3)``. The rgi_utils minimizer
    is pure JAX and gated on the noise level, so this stays JIT-compiled inside
    hk.scan / hk.vmap. Reshaping to/from the flat ``(-1, 3)`` atom layout is the
    only AF3-specific step.
    """
    if self._minimizer is None:
      return positions
    shape = positions.shape
    flat = positions.reshape(-1, 3)
    flat_opt = self._minimizer(flat, sigma_t)
    return flat_opt.reshape(shape)


def build_restraints(fold_input, example) -> 'AF3Restraints | None':
  """Build ``AF3Restraints`` from ``fold_input.restraints_config`` + a batch.

  Args:
    fold_input: Folding input with an optional ``.restraints_config`` dict.
    example: A featurised batch (numpy arrays); supplies atom layout, reference
      coordinates, elements and atom names used to resolve restraints.

  Returns:
    An ``AF3Restraints`` instance, or ``None`` when no restraints_config is set.
  """
  if fold_input.restraints_config is None:
    return None
  from rgi_utils.combined import CombinedRestraints

  # AF3 runs the pure-JAX minimizer inside the diffusion scan.
  config = dict(fold_input.restraints_config)
  config.setdefault('backend', 'jax')

  adapter = AF3RestraintAdapter(fold_input, example)
  rgi = CombinedRestraints()
  rgi.set_config(config)
  rgi.setup(adapter, nbatch=1)
  minimizer = rgi.get_minimizer() if rgi.is_active() else None
  return AF3Restraints(rgi, minimizer)
