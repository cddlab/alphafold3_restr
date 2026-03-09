# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaFold 3 inference pipeline implementation by Google DeepMind.
License: CC BY-NC-SA 4.0 (code) / separate terms for model weights.
The current branch (`add_restraint`) adds restraint-guided inference ported from the Protenix project.

## Build & Install

```bash
# Install with uv (Linux only; requires CUDA 12 for GPU deps)
uv sync

# Build C++ extensions (pybind11 via scikit-build-core / CMake)
uv build

# Fetch genetic databases (~1 TB)
bash fetch_databases.sh <DB_DIR>

# Build chemical component data
uv run build_data
```

## Running Inference

```bash
# Single input JSON
python run_alphafold.py \
  --json_path=fold_input.json \
  --model_dir=~/models \
  --output_dir=~/af_output

# Directory of JSON files
python run_alphafold.py \
  --input_dir=~/af_input \
  --model_dir=~/models \
  --output_dir=~/af_output

# Data pipeline only (CPU, no GPU needed)
python run_alphafold.py --run_inference=false ...

# Inference only (skip MSA/template search)
python run_alphafold.py --run_data_pipeline=false ...
```

## Running Tests

```bash
# End-to-end tests
uv run pytest run_alphafold_test.py -v

# Data pipeline tests
uv run pytest run_alphafold_data_test.py -v

# Unit tests within the package
uv run pytest src/ -v
```

## Architecture

### Key Entry Points
- `run_alphafold.py` — main prediction script (absl flags interface)
- `src/alphafold3/common/folding_input.py` — `Input` dataclass parsing JSON (dialect `alphafold3` v1–4 and `alphafoldserver` v1)

### Data Pipeline (`src/alphafold3/data/`)
- `pipeline.py` — orchestrates MSA search and template search
- `msa.py`, `templates.py` — MSA/template building
- `featurisation.py` — converts `folding_input.Input` → feature tensors
- `tools/` — wrappers for external binaries (jackhmmer, hmmer, etc.)

### Model (`src/alphafold3/model/`)
- `model.py` — `ModelRunner`; top-level inference with `InferenceResult`
- `model_config.py` — `Config` dataclass hierarchy
- `features.py`, `feat_batch.py` — batch feature assembly
- `network/` — JAX/Haiku neural network modules:
  - `evoformer.py` — Evoformer trunk
  - `diffusion_head.py` — diffusion sampling head (key integration point for restraints)
  - `confidence_head.py`, `distogram_head.py` — auxiliary prediction heads
  - `atom_cross_attention.py`, `template_modules.py`

### Restraints (`src/alphafold3/model/restraints/`) — add_restraint branch
- `restraint_setup.py` — `setup_restraints()`: converts `folding_input.Input.restraints_config` dict + atom layout into `CombinedRestraints`
- `combined_restraints.py` — Singleton managing conformer + distance restraints; optimizer (JAX BFGS on GPU, scipy CG on CPU)
- `bond_restr_data.py`, `angle_restr_data.py`, `chiral_data.py` — conformer geometry restraints
- `distance_restr_data.py` — inter-chain distance restraints with flat-bottomed potential
- `selection.py` — VMD-style atom selection language for distance restraints
- `jax_energy.py` — JAX energy functions for GPU optimization

### Structure Representation (`src/alphafold3/structure/`)
- `structure.py` — `Structure` class (mmCIF-based)
- `mmcif.py` — mmCIF parsing/writing
- `bonds.py`, `sterics.py`

### C++ Extensions
- `src/alphafold3/cpp.cc` — pybind11 entry point; exposes CIF dict parsing, DSSP, etc.
- `CMakeLists.txt` — build config for C++ via scikit-build-core

## Input JSON Format

Current version: `dialect: "alphafold3"`, `version: 4`.
Supports: protein/RNA/DNA chains, ligands (CCD codes or SMILES), covalent bonds, custom MSA, structural templates.

### Restraints (add_restraint branch)

Restraints are specified in two places:
- **Per-ligand**: `"conformer_restraint": true` on the ligand entity in `sequences[]`
- **Global config**: `"restraints_config": {...}` at the JSON root

Key `restraints_config` fields:
- `conformer_restraints_config`: per-type (bond/angle/chiral) slack and weight
- `distance_restraints_config`: list of inter-chain distance restraint specs with atom selections
- `max_iter`, `start_sigma`, `gpu`, `verbose`

Alternatively, pass restraints config via YAML file (takes precedence over JSON-embedded config):
```bash
python run_alphafold.py --restraint_config=restraints.yaml ...
```

See `examples/restraint_example.json` for a full working example.

## Contributing Notes

- AI-generated code must be labeled in PR descriptions
- All PRs require manual testing (folding a specific input to verify correctness)
- CLA required at https://cla.developers.google.com/
