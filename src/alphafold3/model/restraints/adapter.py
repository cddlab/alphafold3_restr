# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Adapter from an AF3 fold_input + featurised batch to the rgi_utils protocols.

AF3 atom layout: positions are ``(num_tokens, max_atoms_per_token, 3)`` and
``flat_idx = token_idx * max_atoms_per_token + within_token_idx`` (ligand atoms
each occupy their own token with ``within_token_idx = 0``).

This adapter implements the rgi_utils adapter protocols so
``rgi_utils.featurizer.build_spec`` can build the ``RestraintSpec`` directly,
replacing AF3's bespoke ``ConformerFeaturizer`` + JAX-array construction:

  - FrameworkAdapter.iter_atoms      (distance restraint selection)
  - ConformerAdapter.num_atoms / get_elements / iter_ligand_confs

The AF3-specific pieces deliberately kept here (they have no rgi_utils
equivalent) are: CCD-by-name mol lookup, the leaving-atom subset (a CCD mol may
contain atoms — e.g. glucose O1 — that AF3 tokenisation drops), the
``ref_atom_name_chars`` decode, and the flat-index / per-chain-resid mapping.
"""

from __future__ import annotations

import logging
from typing import Iterator

import numpy as np

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data.tools import rdkit_utils
from rgi_utils.atom_context import AtomRecord, LigandConf

logger = logging.getLogger(__name__)


def _decode_atom_name_chars(chars: np.ndarray) -> str:
  """Decode AF3's ord(c)-32 encoded atom name back to a string."""
  return ''.join(chr(int(x) + 32) for x in chars if int(x) != 0).strip()


class AF3RestraintAdapter:
  """rgi_utils adapter over an AF3 fold_input + featurised batch (numpy arrays)."""

  def __init__(self, fold_input: folding_input.Input, example: dict):
    self.fold_input = fold_input
    self.token_asym_ids = np.asarray(example['asym_id'])  # (num_tokens,)
    self.ref_mask = np.asarray(example['ref_mask'])  # (num_tokens, max)
    self.ref_pos = np.asarray(example['ref_pos'])  # (num_tokens, max, 3)
    self.ref_atom_name_chars = np.asarray(example['ref_atom_name_chars'])
    self.ref_element = np.asarray(example['ref_element'])  # (num_tokens, max)
    self.max_atoms_per_token = self.ref_pos.shape[1]
    # chain.id -> asym int (1-based, in fold_input chain order)
    self.chain_id_to_asym_int = {
        c.id: i + 1 for i, c in enumerate(fold_input.chains)
    }
    self.asym_int_to_chain = {v: k for k, v in self.chain_id_to_asym_int.items()}
    # The chain<->asym mapping assumes fold_input.chains order matches the batch
    # asym_id assignment (both 1-based by appearance), which holds for standard
    # inference (no cropping). Warn if the assumed asym ids aren't all present in
    # the batch, so a misalignment is visible rather than silently restraining the
    # wrong atoms (e.g. a chain dropped by structure cleaning).
    batch_asyms = {int(a) for a in np.unique(self.token_asym_ids)}
    if not set(self.asym_int_to_chain) <= batch_asyms:
      logger.warning(
          'chain<->asym mapping may be misaligned: fold_input asym ids %s are '
          'not all present in the batch (batch asym ids: %s).',
          sorted(set(self.asym_int_to_chain) - batch_asyms),
          sorted(batch_asyms),
      )
    self._ccd = None

  @property
  def ccd(self):
    if self._ccd is None:
      self._ccd = chemical_components.Ccd(user_ccd=self.fold_input.user_ccd)
    return self._ccd

  # --- FrameworkAdapter ------------------------------------------------------
  def num_atoms(self) -> int:
    return int(self.ref_pos.shape[0] * self.max_atoms_per_token)

  def get_elements(self) -> np.ndarray:
    """(num_tokens*max,) atomic numbers; padding atoms (ref_mask 0) -> 0."""
    elem = self.ref_element.reshape(-1).astype(np.int64)
    mask = self.ref_mask.reshape(-1)
    return np.where(mask > 0, elem, 0)

  def iter_atoms(self) -> Iterator[AtomRecord]:
    """Yield AtomRecord(chain, resid, index) for every real atom.

    ``resid`` is the 1-based residue/token ordinal WITHIN the chain (it resets
    at each chain) and ``index`` is the global flat index ``token*max+within`` —
    consistent with boltz/protenix, so a selection like "chain B and resid 5"
    means residue 5 of chain B in every framework.
    """
    asym = self.token_asym_ids
    # per-chain 1-based token ordinal
    per_chain_resid = np.zeros(len(asym), dtype=int)
    counts: dict[int, int] = {}
    for ti, aint in enumerate(asym):
      aint = int(aint)
      counts[aint] = counts.get(aint, 0) + 1
      per_chain_resid[ti] = counts[aint]
    for token_idx, aint in enumerate(asym):
      chain = self.asym_int_to_chain.get(int(aint), '')
      resid = int(per_chain_resid[token_idx])
      for within in range(self.max_atoms_per_token):
        if not bool(self.ref_mask[token_idx, within]):
          continue
        flat = int(token_idx) * self.max_atoms_per_token + within
        name = _decode_atom_name_chars(
            self.ref_atom_name_chars[token_idx, within]
        ) or None
        yield AtomRecord(chain=chain, resid=resid, index=flat, name=name)

  # --- ConformerAdapter ------------------------------------------------------
  def iter_ligand_confs(self) -> Iterator[LigandConf]:
    """Yield one LigandConf per ligand chain with conformer_restraints=True."""
    try:
      from rdkit import Chem  # pylint: disable=g-import-not-at-top
    except ImportError:
      return
    pos_flat = self.ref_pos.reshape(-1, 3)
    for chain in self.fold_input.chains:
      if not isinstance(chain, folding_input.Ligand):
        continue
      if not chain.conformer_restraints:  # ligand opted out
        continue
      mol, flat_indices = self._ligand_mol(chain, Chem)
      if mol is None or len(flat_indices) == 0:
        continue
      conf_crds = pos_flat[flat_indices]  # (n_atoms, 3) reference coords
      # A SMILES mol carries no 3D geometry, so chiral tags exist only if the SMILES
      # annotated them (@/@@); the featurizer keys chiral restraints on GetChiralTag.
      # Attach the reference conformer and perceive stereo from it (matching the CCD
      # path, which assigns stereo from the ideal conformer) so an unannotated SMILES
      # stereocentre still gets chiral restraints. (MolFromSmiles keeps implicit-H on,
      # so the chai SetNoImplicit dance is unnecessary here.)
      # NOTE: an ETKDG ideal-conformer target (like boltz/protenix) is NOT used here
      # because af3's ligand atoms are in TOKEN order, which differs from the SMILES mol's
      # RDKit-canonical order, and for SYMMETRIC ligands (e.g. fumarate/maleate) a
      # connectivity-only substructure match can't pick the correct 1:1 atom mapping
      # (it has several automorphisms). So af3 keeps ref_pos as the dihedral target =>
      # cis/trans is only partially corrected on af3 (documented limitation).
      if mol.GetNumConformers() == 0 and mol.GetNumAtoms() == len(conf_crds):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i in range(len(conf_crds)):
          conf.SetAtomPosition(
              i,
              (float(conf_crds[i, 0]), float(conf_crds[i, 1]), float(conf_crds[i, 2])),
          )
        mol.AddConformer(conf, assignId=True)
        try:
          Chem.AssignStereochemistryFrom3D(mol)
        except Exception:  # geometry-only restraints don't need a clean valence model
          pass
      yield LigandConf(
          mol=mol,
          conf_coords=conf_crds,
          global_indices=np.asarray(flat_indices, dtype=np.int64),
          # opted-in: ligands that set conformer_restraints=False are skipped above
          # (line ~128). Pass True explicitly because LigandConf defaults to False.
          conformer_restraints=True,
      )

  # --- AF3-specific ligand mol construction ----------------------------------
  def _ligand_mol(self, chain, Chem):
    """Return (mol, flat_indices) for a ligand chain, or (None, []) to skip.

    Only SMILES- or single-CCD-based ligands have a mol graph for bond/angle/
    chiral extraction. For CCD ligands we keep only the atoms that survive AF3
    tokenisation (leaving atoms such as glucose O1 are dropped).
    """
    if chain.smiles is not None:
      mol = Chem.MolFromSmiles(chain.smiles)
      if mol is None:
        return None, []
      return mol, self._smiles_flat_indices(chain.id)
    if chain.ccd_ids is not None and len(chain.ccd_ids) == 1:
      ccd_cif = self.ccd.get(chain.ccd_ids[0])
      if ccd_cif is None:
        return None, []
      try:
        mol = rdkit_utils.mol_from_ccd_cif(
            ccd_cif, sort_alphabetically=False, remove_hydrogens=True
        )
      except rdkit_utils.MolFromMmcifError:
        return None, []
      names = [a.GetProp('atom_name').strip() for a in mol.GetAtoms()]
      flat, kept = self._ccd_flat_indices(chain.id, names)
      # Restrain only atoms present in the structure (drop CCD-only atoms).
      if 0 < len(kept) < mol.GetNumAtoms():
        mol = self._subset_mol(mol, kept, Chem)
      return mol, flat
    return None, []

  def _smiles_flat_indices(self, chain_id: str) -> np.ndarray:
    asym_int = self.chain_id_to_asym_int[chain_id]
    token_indices = np.where(self.token_asym_ids == asym_int)[0]
    # ligand atom: flat_idx = token_idx * max_atoms_per_token (within = 0)
    return (token_indices * self.max_atoms_per_token).astype(np.int32)

  def _ccd_flat_indices(self, chain_id: str, atom_names):
    """Map CCD mol atom names to flat indices; return (flat, kept mol indices)."""
    asym_int = self.chain_id_to_asym_int[chain_id]
    token_indices = np.where(self.token_asym_ids == asym_int)[0]
    name_to_flat: dict[str, int] = {}
    for token_idx in token_indices:
      for within in range(self.max_atoms_per_token):
        if not bool(self.ref_mask[token_idx, within]):
          continue
        name = _decode_atom_name_chars(
            self.ref_atom_name_chars[token_idx, within]
        )
        if name:
          name_to_flat.setdefault(
              name, int(token_idx) * self.max_atoms_per_token + within
          )
    kept = [i for i, nm in enumerate(atom_names) if nm in name_to_flat]
    missing = [nm for nm in atom_names if nm not in name_to_flat]
    if missing:
      logger.info(
          'chain %s: dropping %d ligand atom(s) absent from the structure: %s',
          chain_id,
          len(missing),
          missing,
      )
    flat = np.array([name_to_flat[atom_names[i]] for i in kept], dtype=np.int32)
    return flat, kept

  @staticmethod
  def _subset_mol(mol, kept, Chem):
    """Copy ``mol`` keeping only atoms ``kept`` (elements, atom_name, chiral
    tags, bonds among kept atoms and the conformer)."""
    rw = Chem.RWMol()
    old2new = {}
    for new_i, old_i in enumerate(kept):
      a = mol.GetAtomWithIdx(int(old_i))
      na = Chem.Atom(a.GetAtomicNum())
      if a.HasProp('atom_name'):
        na.SetProp('atom_name', a.GetProp('atom_name'))
      na.SetChiralTag(a.GetChiralTag())
      rw.AddAtom(na)
      old2new[int(old_i)] = new_i
    for b in mol.GetBonds():
      i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
      if i in old2new and j in old2new:
        rw.AddBond(old2new[i], old2new[j], b.GetBondType())
    out = rw.GetMol()
    if mol.GetNumConformers() > 0:
      conf = mol.GetConformer()
      newconf = Chem.Conformer(len(kept))
      for new_i, old_i in enumerate(kept):
        newconf.SetAtomPosition(new_i, conf.GetAtomPosition(int(old_i)))
      out.AddConformer(newconf, assignId=True)
    try:
      Chem.SanitizeMol(out)
    except Exception:  # geometry-only restraints don't need a clean valence model
      pass
    return out
