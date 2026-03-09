"""Helper functions for setting up restraints from fold_input and atom layout."""
from __future__ import annotations

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

from alphafold3.common import folding_input
from alphafold3.constants import chemical_components
from alphafold3.data.tools import rdkit_utils

from .combined_restraints import CombinedRestraints


def setup_restraints(
    fold_input: folding_input.Input,
    all_token_atoms_layout,
    max_atoms_per_token: int,
    ccd: chemical_components.Ccd,
    restraints_config: dict,
) -> CombinedRestraints | None:
    """Set up combined restraints from fold_input and AlphaFold3 atom layout.

    Call this AFTER featurisation (so all_token_atoms_layout is available).
    The resulting CombinedRestraints instance is passed to run_inference_with_restraints().

    Args:
        fold_input: AlphaFold3 folding input (chains + restraints field).
        all_token_atoms_layout: AtomLayout shape (num_tokens, max_atoms_per_token).
        max_atoms_per_token: number of atom slots per token.
        ccd: chemical components dictionary.
        restraints_config: dict parsed from restraint YAML.

    Returns:
        CombinedRestraints instance with restraints configured, or None if inactive.
    """
    CombinedRestraints._instance = None  # reset singleton for new job
    combined = CombinedRestraints.get_instance()
    combined.set_config(restraints_config)

    # ---- Conformer restraints ----
    # Build atom name -> flat global index mapping for ALL atoms
    num_tokens = all_token_atoms_layout.shape[0]
    # (chain_id, atom_name_in_residue, token_idx) uniqueness assumption:
    # For ligands: each token = 1 residue (the ligand molecule is one res per chain)
    # We build (chain_id, atom_name) -> flat_idx
    chain_atom_to_flat: dict[tuple[str, str], int] = {}
    for token_idx in range(num_tokens):
        for atom_in_token in range(max_atoms_per_token):
            atom_name = all_token_atoms_layout.atom_name[token_idx, atom_in_token]
            if not atom_name:
                continue
            chain_id = str(all_token_atoms_layout.chain_id[token_idx, atom_in_token])
            flat_idx = token_idx * max_atoms_per_token + atom_in_token
            chain_atom_to_flat[(chain_id, str(atom_name))] = flat_idx

    for chain in fold_input.chains:
        if not isinstance(chain, folding_input.Ligand):
            continue
        # Apply conformer restraints only to ligands with conformer_restraint=True
        if not chain.conformer_restraint:
            continue
        chain_id = chain.id

        # Get RDKit mol for this ligand
        mol = _get_rdkit_mol(chain, ccd)
        if mol is None:
            print(f"[Restraints] Warning: could not build RDKit mol for chain {chain_id}, skipping conformer restraints.")
            continue

        mol_noH = Chem.RemoveHs(mol)
        try:
            AllChem.EmbedMolecule(mol_noH, AllChem.ETKDGv3())
            conf = mol_noH.GetConformer()
        except Exception as e:
            print(f"[Restraints] Warning: conformer generation failed for chain {chain_id}: {e}")
            continue

        # Map mol atom idx -> flat global index
        global_indices = np.full(mol_noH.GetNumAtoms(), -1, dtype=np.int64)
        for i, atom in enumerate(mol_noH.GetAtoms()):
            atom_name = atom.GetProp('atom_name') if atom.HasProp('atom_name') else atom.GetSymbol()
            flat_idx = chain_atom_to_flat.get((chain_id, atom_name))
            if flat_idx is not None:
                global_indices[i] = flat_idx

        # Register conformer restraints
        for bond in mol_noH.GetBonds():
            ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if global_indices[ai] >= 0 and global_indices[aj] >= 0:
                combined.make_bond(ai, aj, conf, global_indices)

        combined.make_angle_restraints(mol_noH, conf, global_indices)

        for atom in mol_noH.GetAtoms():
            if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
                if global_indices[atom.GetIdx()] >= 0:
                    combined.make_chiral(atom.GetIdx(), mol_noH, conf, global_indices)

    # ---- Distance restraints ----
    combined.set_feats_af3(all_token_atoms_layout, max_atoms_per_token)

    # ---- Build feature array ----
    N_flat = num_tokens * max_atoms_per_token
    conformer_arr = combined.build_conformer_restraint_array(N_flat)

    # ---- Setup sites ----
    combined.setup_site(conformer_arr, max_atoms_per_token)

    if not combined.is_active():
        print("[Restraints] No active restraints found.")
        return None

    return combined


def _get_rdkit_mol(chain: folding_input.Ligand, ccd: chemical_components.Ccd) -> Chem.Mol | None:
    """Get an RDKit mol (with H) for a ligand chain."""
    if chain.smiles:
        mol = Chem.MolFromSmiles(chain.smiles)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        mol = rdkit_utils.assign_atom_names_from_graph(mol)
        return mol
    elif chain.ccd_ids:
        for ccd_id in chain.ccd_ids:
            try:
                ccd_data = ccd[ccd_id]
                mol = rdkit_utils.mol_from_ccd_cif(ccd_data, remove_hydrogens=False)
                if mol is not None:
                    return mol
            except Exception:
                continue
    return None
