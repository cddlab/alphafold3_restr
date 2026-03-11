# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Restraint system for guided diffusion sampling in AlphaFold 3.

Supports conformer restraints (bond/angle/chiral volumes for ligands)
and distance restraints (COM distance between atom groups).

GPU mode: JAX gradient descent (jax.lax.fori_loop) inside the diffusion scan.
CPU mode: scipy CG minimization applied once after the full diffusion loop.
"""

from alphafold3.model.restraints.combined_restraints import CombinedRestraints
from alphafold3.model.restraints.combined_restraints import RestraintConfig

__all__ = ['CombinedRestraints', 'RestraintConfig']
