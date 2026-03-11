# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Restraint system for guided diffusion sampling in AlphaFold 3.

Supports conformer restraints (bond/angle/chiral volumes for ligands)
and distance restraints (COM distance between atom groups).

Restraints are applied step-wise inside the diffusion scan via a JAX minimizer.
"""

from alphafold3.model.restraints.combined_restraints import CombinedRestraints
from alphafold3.model.restraints.combined_restraints import RestraintConfig

__all__ = ['CombinedRestraints', 'RestraintConfig']
