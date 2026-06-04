#!/bin/bash
#SBATCH -J af3_ex
#SBATCH -o run_restr_example.out
#SBATCH -e run_restr_example.err
#SBATCH -p af3
#SBATCH --gres=gpu:1
# alphafold3 RGI example runner (JAX backend). GPU work must go through sbatch.
# Submit from THIS repo directory:  cd alphafold3_restr && sbatch run_restr_example.sh
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate

# AF3 enables NO persistent XLA cache by default, so every process recompiles the graph
# (~2 min). Enabling it (node-local /tmp, off NFS) lets repeated runs reuse the compile.
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/${USER}_jax_cache}"

rm -rf out_restr_example   # else AF3 skip-existing early-returns on the prior output
# restr_example.json has an inline single-sequence MSA (self-contained) and carries
# `restraints_config` (backend forced to jax). RGI runs inside the diffusion hk.scan.
# NOTE: conformer vdw MUST be mode=intramolecular under JAX.
python run_alphafold.py \
    --run_data_pipeline=False \
    --db_dir=/mnt/database/public_databases \
    --pdb_database_path=/home/apps/alphafold3/database/pdb_mmcif/mmcif_files \
    --model_dir=/home/apps/alphafold3/models \
    --json_path=restr_example.json \
    --output_dir=out_restr_example \
    2>&1 | grep -iE "rgi_utils|built spec|restraint|Error|Traceback|finalize|dropping" || true
# (COM check: run check_dist.py with a gemmi-enabled venv; AF3's venv lacks gemmi.)
echo done
