# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repository is

This is a **fork of Google DeepMind's AlphaFold 3** (`run_alphafold.py` + `src/alphafold3/`) extended with **Restraint-Guided Inference (RGI)**. The branch of interest is `rgi-integration`; `main` tracks upstream AF3. The RGI work adds the ability to bias the diffusion sampler with user-supplied geometric restraints (ligand conformer geometry, intramolecular VdW, and inter-residue distance restraints) applied as an in-scan JAX minimization after each denoising step.

The heavy lifting (restraint spec construction, the analytic energy + CG/LBFGS minimizer, atom-name/index adapter logic, the DSL for atom selections) lives in an **external dependency, `rgi_utils`** (`github.com/cddlab/rgi_utils`, pinned to the `rgi-integration` rev in `pyproject.toml`). Code in this repo is intentionally a thin "glue" layer. When debugging restraint *behavior* (energy terms, selection DSL, minimizer), the source is in `rgi_utils`, not here.

## Commands

```bash
# Build/install locally (compiles C++ extensions via scikit-build-core + CMake).
uv sync                       # or: pip install --no-build-isolation .

# Build the C++ chemical-components data file (required once after install)
build_data                    # entry point -> alphafold3.build_data:build_data

# Run inference (see README / docs/installation.md for the Docker form)
python run_alphafold.py --json_path=input_af3_restr.json \
    --model_dir=<MODEL_DIR> --output_dir=<OUT> --db_dir=<DB_DIR>

# Skip the CPU genetic-search pipeline when MSAs are pre-supplied in the JSON
python run_alphafold.py ... --norun_data_pipeline
```

There is no project-wide lint config beyond a `.ruff_cache` (ruff is used ad hoc). `requires-python >= 3.12`.

## Restraint architecture (the RGI extension)

The end-to-end flow, from JSON to biased coordinates:

1. **Input parsing** (`src/alphafold3/common/folding_input.py`): two RGI-specific additions to the AF3 input schema —
   - top-level `restraints_config` dict on `Input` (distance + conformer config, `method`, `max_iter`, `start_sigma`, `gpu`, etc.).
   - per-ligand `conformer_restraints: true` flag on `Ligand` chains (opt-in; ligands default to off).
   - Also adds `unpairedMsaPath`/`pairedMsaPath` to load MSAs from a file instead of inline.

2. **Glue layer** (`src/alphafold3/model/restraints/`):
   - `adapter.py` — the ONLY place AF3 internals are coupled to restraints. `_resolve_ligand_mols` turns CCD/SMILES ligands into RDKit mols, then hands plain data to `rgi_utils.alphafold3.adapter.AF3RestraintAdapter`.
   - `combined_restraints.py` — `build_restraints(fold_input, example)` reads `restraints_config`, **forces `backend='jax'`** (AF3 runs the minimizer inside `hk.scan`/`hk.vmap`, so numpy/torch backends are silently inert — it warns and overrides), and builds a `CombinedRestraints` via `rgi_utils`. The returned object is aliased from `rgi_utils.optim.scan_runner.ScanMinimizer` under the historical name `AF3Restraints`.

3. **Inference wiring** (`run_alphafold.py`): builds restraints **once** from the first featurised example (they are seed-invariant), then `_build_model_with_restraints` JIT-compiles a forward pass with the restraints object captured in closure (cached by restraints identity, since restraint JAX arrays must be constant-folded). `restraints.finalize(...)` runs after inference.

4. **In-scan application** (`src/alphafold3/model/network/diffusion_head.py`, `sample()`): after each `denoising_step`, if `restraints` is active, `restraints.minimize_gpu(positions_denoised, noise_level_prev, istep)` is applied to `x0` before the Euler step. The scan carries `(noise_level, istep)` so restraints can gate on a diffusion-step window (`start_step`/`stop_step`) and on the noise level (`start_sigma`/`stop_sigma`). `model.py` only passes restraints through when `restraints.is_active()`.

The whole minimization stays JIT-compiled (no `pure_callback`, no scipy), which is why the jax backend is mandatory and why the reshape `(num_tokens, max_atoms_per_token, 3) <-> (-1, 3)` wrapper lives in `rgi_utils`' `ScanMinimizer`.

### Example inputs

`input_af3_restr.json` and the `bench_in_af3_*.json` files are RGI example inputs (conformer-only, distance-only, and combined). `out_restr_example/qbp_rgi_example/` holds a reference output. Note `.gitignore` excludes most `input_*.json`/`bench_in_*.json`/`out_*/` as scratch — the committed examples are deliberate exceptions.

## Working in this codebase

- The base AF3 source under `src/alphafold3/` is large; most of it (data pipeline, model network, structure parsing, C++ extensions in `*/cpp/`) is upstream and rarely needs changing for RGI work. Restraint changes concentrate in the four files listed above plus `rgi_utils`.
- Restraint *config semantics* (selection DSL like `"chain A and (resid 5 to 84)"`, harmonic/flat-bottom modes, `move` mode, weights, sigma windows) are documented in `rgi_utils`' docs, not here.
- License headers in `src/` still say CC BY-NC-SA 4.0, but the repo relicensed to Apache 2.0 (commit `7b197fe`); `CMakeLists.txt` and `LICENSE` reflect Apache 2.0.
- This is a fork: upstream merges land on `main` then merge into `rgi-integration`. Keep RGI changes localized to the glue layer to minimize merge conflicts with upstream AF3.
