# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""AF3 restraint glue around the shared rgi_toolkit engine.

Spec construction (conformer bond/angle/chiral + intramolecular VdW + distance
restraints) lives entirely in rgi_toolkit: the in-tool ``build_af3_adapter`` shim
exposes the AF3 batch through the rgi_toolkit adapter protocols and
``featurizer.build_spec`` builds the ``RestraintSpec``. This module keeps only the
AF3-specific glue: read ``fold_input.restraints_config`` and hand it to the engine.
The jax backend is selected implicitly by AF3 calling ``get_minimizer()`` (the pure-JAX
minimizer it runs inside the scan) — backend is no longer a config key, it is inferred
from the invocation.

The scan-time wrapper — flatten ``(num_tokens, max_atoms_per_token, 3) <-> (-1, 3)``
around the pure minimizer + the duck-typed ``is_active``/``minimize_gpu``/``finalize``
interface — now lives in ``rgi_toolkit.optim.scan_runner.ScanMinimizer`` (imported here
as ``AF3Restraints`` for back-compat). The minimizer itself (CG/LBFGS over an analytic
energy, gated on the noise level) is ``rgi_toolkit.optim.jax_optim.make_minimizer`` — no
pure_callback, no scipy — so the whole step stays JIT-compiled inside hk.scan/hk.vmap.
"""

from __future__ import annotations

from alphafold3.model.restraints.adapter import build_af3_adapter
# The scan-time reshape wrapper + duck-typed (is_active/minimize_gpu/finalize)
# interface is the framework-free rgi_toolkit ScanMinimizer; alias it to the historical
# AF3Restraints name so model.py / diffusion_head.py + restraints/__init__.py are
# unchanged.
from rgi_toolkit.optim.scan_runner import ScanMinimizer as AF3Restraints


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
  from rgi_toolkit.combined import CombinedRestraints

  # AF3 runs the pure-JAX minimizer inside the diffusion scan; the jax backend is
  # selected implicitly when get_minimizer() is called below (backend is no longer a
  # config key — it is inferred from the invocation, and get_minimizer() => jax).
  # NOTE: the dynamic ligand-protein VdW runs on jax too (rgi_toolkit jax_optim ports the
  # torch term), so AF3 no longer overrides vdw.mode. An unset mode takes the rgi_toolkit
  # default ("both" = intramolecular + ligand-protein), both of which work under jax;
  # set mode explicitly to pick one.
  adapter = build_af3_adapter(fold_input, example)
  rgi = CombinedRestraints()
  # Single-call lifecycle (config= folds in the old set_config); matches the other
  # five tools. set_config()+setup() is the deprecated two-call shim.
  rgi.setup(adapter, nbatch=1, config=fold_input.restraints_config)
  minimizer = rgi.get_minimizer() if rgi.is_active() else None
  return AF3Restraints(rgi, minimizer)
