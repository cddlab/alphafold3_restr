# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaFold 3 (Google DeepMind) is a biomolecular structure prediction model. This repository contains the inference pipeline. The framework is **JAX + Haiku (hk)** — not PyTorch.

## Common Commands

```bash
# Install (development)
uv sync

# Build C++ extensions
pip install -e . --no-build-isolation

# Run inference (Docker)
python run_alphafold.py \
    --json_path=fold_input.json \
    --model_dir=$HOME/models \
    --output_dir=./output

# Run without data pipeline (inference only)
python run_alphafold.py --json_path=... --run_data_pipeline=false

# Run tests (absl-based)
python -m pytest run_alphafold_test.py
python -m pytest run_alphafold_data_test.py

# Build genetic databases
build_data --jackhmmer_binary_path=$(which jackhmmer) [other flags]
```

## Architecture Overview

### Main Forward Pass

`run_alphafold.py` → `model.Model.__call__()` in `src/alphafold3/model/model.py`:

1. **`create_target_feat_embedding()`** — atom-level features → token embeddings via `AtomCrossAttEncoder`
2. **`Evoformer.__call__()`** (recycling loop, `hk.fori_loop`) — produces `embeddings['single']` and `embeddings['pair']`
3. **`Model._sample_diffusion()`** → **`diffusion_head.sample()`** — iterative denoising via `hk.scan` + `hk.vmap`
4. **`ConfidenceHead`** — pLDDT, PAE, pTM computation over all samples

### Key Model Files

| File | Role |
|------|------|
| `model/model.py` | Top-level `Model` class; orchestrates Evoformer + diffusion + confidence |
| `model/network/evoformer.py` | `Evoformer`: MSA stack + Pairformer; outputs `single`/`pair` embeddings |
| `model/network/diffusion_head.py` | `DiffusionHead.__call__()` (one denoising step); `sample()` (full loop) |
| `model/network/modules.py` | `PairFormerIteration`, `EvoformerIteration` building blocks |
| `model/network/atom_cross_attention.py` | Atom↔token cross-attention encoder/decoder |
| `model/feat_batch.py` | `Batch` dataclass — typed wrapper around all feature arrays |
| `model/features.py` | Individual feature group dataclasses (`TokenFeatures`, `MSA`, etc.) |
| `model/model_config.py` | `GlobalConfig` (bfloat16, final init, etc.) |
| `common/folding_input.py` | Input JSON parsing → `FoldInput` dataclass (all sequence/entity definitions) |
| `data/featurisation.py` | `FoldInput` → `BatchDict` (numpy arrays) |
| `data/pipeline.py` | MSA/template search pipeline |

### Diffusion Sampling Loop

`diffusion_head.sample()` (lines 297–369):
- Uses `hk.vmap` (over samples) + `hk.scan` (over noise levels)
- `apply_denoising_step(carry, noise_level)` is the inner step function:
  - Adds noise → calls `denoising_step(positions_noisy, t_hat)` (= `DiffusionHead.__call__`)
  - Computes `positions_denoised` → Euler update → returns next positions
- **Restraint injection point**: after `positions_denoised` is computed, before the Euler step

### Batch Data Flow

```
FoldInput (JSON)
  → featurisation.py: np arrays (BatchDict)
  → features.py: typed dataclasses
  → feat_batch.Batch.from_data_dict()
  → Model.__call__(batch)
```

`feat_batch.Batch` fields: `msa`, `templates`, `token_features`, `ref_structure`, `predicted_structure_info`, `polymer_ligand_bond_info`, `ligand_ligand_bond_info`, `pseudo_beta_info`, `atom_cross_att`, `convert_model_output`, `frames`

### Configuration

No separate config files — configs are `base_config.BaseConfig` dataclass instances defined inline in each `hk.Module` as nested `Config` classes. Override via constructor arguments or `dataclasses.replace()`.

### C++ Extensions

`src/alphafold3/cpp.cc` exposes `cif_dict`, `msa_profile`, and `mkdssp` via pybind11. Built with CMake via scikit-build-core.

## Input JSON Format

`alphafold3` dialect (version 1–4):
```json
{
  "name": "...",
  "sequences": [
    {"protein": {"id": "A", "sequence": "..."}},
    {"ligand": {"id": "B", "ccdCodes": ["ATP"]}}
  ],
  "modelSeeds": [42],
  "dialect": "alphafold3",
  "version": 4
}
```
Also accepts `alphafoldserver` dialect (auto-detected and converted). See `docs/input.md`.

## Restraint Implementation Plan (Porting from Protenix)

Protenix (`protenix/model/restraints/`) implements conformer-restraints (bond/angle/chiral) and distance-restraints applied via CG minimization after each denoising step. In AlphaFold 3, the equivalent injection point is inside `diffusion_head.sample()` → `apply_denoising_step()`.

**Critical constraint**: The inner loop runs inside `hk.scan` (XLA-compiled). Python-level callbacks (scipy, PyTorch) **cannot** be called. All restraint energy/gradient functions must be pure JAX.

**Implementation approach**:
1. Restraint energy/gradient → JAX-differentiable functions (`jnp` operations)
2. CG minimization → `jax.scipy.optimize.minimize` (L-BFGS-B) or manual gradient steps inside `jax.lax.while_loop`
3. Restraint data (atom indices, target distances) → pass as extra arrays in `Batch` or as static closure variables
4. The `sample()` function signature needs a `restraints` argument propagated into `apply_denoising_step`

**Key files to create/modify**:
- `src/alphafold3/model/restraints/` — new package (conformer_restraints.py, distance_restraints.py, combined_restraints.py)
- `src/alphafold3/model/network/diffusion_head.py` — modify `sample()` to accept and apply restraints after `positions_denoised`
- `src/alphafold3/model/model.py` — pass restraints through `_sample_diffusion()`
- `src/alphafold3/common/folding_input.py` — add restraint fields to input parsing
- `src/alphafold3/data/featurisation.py` — convert restraint specs to feature arrays
