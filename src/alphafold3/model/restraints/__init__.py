# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Restraint system for guided diffusion sampling in AlphaFold 3.

Conformer restraints (bond/angle/chiral + intramolecular VdW for ligands) and
distance restraints (centroid distance between atom groups) are built by rgi_utils
from the AF3 batch via ``AF3RestraintAdapter`` and applied step-wise inside the
diffusion scan by a pure-JAX minimizer.
"""

from alphafold3.model.restraints.combined_restraints import AF3Restraints
from alphafold3.model.restraints.combined_restraints import build_restraints

__all__ = ['AF3Restraints', 'build_restraints']
