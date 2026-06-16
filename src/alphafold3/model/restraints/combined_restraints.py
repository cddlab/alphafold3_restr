# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""AF3 restraint glue around the shared rgi_utils engine.

Spec construction (conformer bond/angle/chiral + intramolecular VdW + distance
restraints) lives entirely in rgi_utils: the in-tool ``build_af3_adapter`` shim
exposes the AF3 batch through the rgi_utils adapter protocols and
``featurizer.build_spec`` builds the ``RestraintSpec``. This module keeps only the
AF3-specific glue: read ``fold_input.restraints_config`` and inject ``backend='jax'``
(AF3 runs the pure-JAX minimizer in the scan).

The scan-time wrapper — flatten ``(num_tokens, max_atoms_per_token, 3) <-> (-1, 3)``
around the pure minimizer + the duck-typed ``is_active``/``minimize_gpu``/``finalize``
interface — now lives in ``rgi_utils.optim.scan_runner.ScanMinimizer`` (imported here
as ``AF3Restraints`` for back-compat). The minimizer itself (CG/LBFGS over an analytic
energy, gated on the noise level) is ``rgi_utils.optim.jax_optim.make_minimizer`` — no
pure_callback, no scipy — so the whole step stays JIT-compiled inside hk.scan/hk.vmap.
"""

from __future__ import annotations

import logging

from alphafold3.model.restraints.adapter import build_af3_adapter
# The scan-time reshape wrapper + duck-typed (is_active/minimize_gpu/finalize)
# interface is the framework-free rgi_utils ScanMinimizer; alias it to the historical
# AF3Restraints name so model.py / diffusion_head.py + restraints/__init__.py are
# unchanged.
from rgi_utils.optim.scan_runner import ScanMinimizer as AF3Restraints

logger = logging.getLogger(__name__)


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

  # AF3 runs the pure-JAX minimizer inside the diffusion scan, so it supports ONLY
  # the jax backend. Force it (do not setdefault): an explicit backend:numpy/torch in
  # the fold-input JSON would otherwise build an unused torch/numpy optimizer, leave
  # get_minimizer()=None, and silently apply NO restraints (distance + conformer both
  # vanish with no error). Warn if a different backend was requested.
  config = dict(fold_input.restraints_config)
  if config.get('backend') not in (None, 'jax'):
    logger.warning(
        'AF3 supports only the jax backend; ignoring restraints_config '
        'backend=%r and forcing jax.',
        config['backend'],
    )
  config['backend'] = 'jax'
  # NOTE: the dynamic ligand-protein VdW now runs on jax too (rgi_utils jax_optim
  # ports the torch term), so AF3 no longer overrides vdw.mode. An unset mode takes
  # the rgi_utils default ("both" = intramolecular + ligand-protein), both of which
  # work under the jax backend; set mode explicitly to pick one.

  adapter = build_af3_adapter(fold_input, example)
  rgi = CombinedRestraints()
  rgi.set_config(config)
  rgi.setup(adapter, nbatch=1)
  minimizer = rgi.get_minimizer() if rgi.is_active() else None
  return AF3Restraints(rgi, minimizer)
