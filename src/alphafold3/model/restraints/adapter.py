# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""In-tool shim: resolve AF3 ligand mols + build the rgi_utils AF3 adapter.

All alphafold3 coupling lives HERE (folding_input access + CCD/SMILES -> RDKit mol).
The framework-free adapter logic — flat-index / per-chain-resid mapping, the
leaving-atom subset, atom-name decode, ``iter_atoms`` / ``iter_ligand_confs`` —
moved to ``rgi_utils.alphafold3.adapter`` so AF3 follows the same pattern as the
torch tools (a framework-free adapter in rgi_utils fed by tool-extracted plain data).
"""

from __future__ import annotations

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.constants import residue_names
from alphafold3.data.tools import rdkit_utils
from rgi_utils.alphafold3.adapter import AF3RestraintAdapter


def _resolve_ligand_mols(fold_input: folding_input.Input):
    """Resolve ``[(chain_id, mol, is_smiles)]`` for ligand chains opted into conformer
    restraints — the ONLY alphafold3-coupled step.

    SMILES ligands -> ``Chem.MolFromSmiles``; single-CCD ligands -> the AF3 CCD
    machinery (``Ccd`` + ``mol_from_ccd_cif``). The Ccd is built lazily (only when a
    CCD ligand is actually present), matching the original property. Mols keep all CCD
    atoms (incl. leaving atoms); the rgi_utils adapter maps atom names to flat indices
    and drops CCD-only atoms.
    """
    try:
        from rdkit import Chem  # pylint: disable=g-import-not-at-top
    except ImportError:
        return []
    out = []
    ccd = None
    for chain in fold_input.chains:
        if not isinstance(chain, folding_input.Ligand):
            continue
        if not chain.conformer_restraints:  # ligand opted out
            continue
        if chain.smiles is not None:
            mol = Chem.MolFromSmiles(chain.smiles)
            if mol is not None:
                out.append((chain.id, mol, True))
            continue
        if chain.ccd_ids is not None and len(chain.ccd_ids) == 1:
            if ccd is None:
                ccd = chemical_components.Ccd(user_ccd=fold_input.user_ccd)
            ccd_cif = ccd.get(chain.ccd_ids[0])
            if ccd_cif is None:
                continue
            try:
                mol = rdkit_utils.mol_from_ccd_cif(
                    ccd_cif, sort_alphabetically=False, remove_hydrogens=True
                )
            except rdkit_utils.MolFromMmcifError:
                continue
            out.append((chain.id, mol, False))
    return out


def build_af3_adapter(
    fold_input: folding_input.Input, example: dict
) -> AF3RestraintAdapter:
    """Build the framework-free rgi_utils AF3 adapter from a fold_input + batch.

    The chain<->asym mapping assumes ``fold_input.chains`` order matches the batch
    asym_id assignment (both 1-based by appearance), which holds for standard
    inference (no cropping); the rgi_utils adapter warns if it looks misaligned.
    """
    chain_id_to_asym = {c.id: i + 1 for i, c in enumerate(fold_input.chains)}
    conformer_restraints_by_asym = {
        chain_id_to_asym[chain.id]: bool(
            getattr(chain, "conformer_restraints", False)
        )
        for chain in fold_input.chains
    }
    return AF3RestraintAdapter(
        batch=example,
        chain_id_to_asym=chain_id_to_asym,
        polymer_residue_names=residue_names.POLYMER_TYPES,
        ligand_mols=_resolve_ligand_mols(fold_input),
        conformer_restraints_by_asym=conformer_restraints_by_asym,
    )
