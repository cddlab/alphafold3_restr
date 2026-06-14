#!/bin/bash
#SBATCH -J af3_grp
#SBATCH -o run_grp_test.out
#SBATCH -e run_grp_test.err
#SBATCH -p af3
#SBATCH --gres=gpu:1
set -e
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source .venv/bin/activate
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-/tmp/${USER}_jax_cache}"
rm -rf out_grp_test
if ! python run_alphafold.py --run_data_pipeline=False --db_dir=/mnt/database/public_databases --pdb_database_path=/home/apps/alphafold3/database/pdb_mmcif/mmcif_files --model_dir=/home/apps/alphafold3/models --json_path=grp_test.json --output_dir=out_grp_test > run_grp_test.log 2>&1; then
  echo "af3 FAILED:"; tail -n 40 run_grp_test.log; exit 1
fi
grep -iE "built spec|setup:|finalize|restraint" run_grp_test.log || true
CIF=$(find out_grp_test -name '*.cif' | head -1); echo "CIF: $CIF"
GP=../chai-lab_restr/.venv/bin/python
"$GP" ../check_angle.py "$CIF" 5-84 90-180 186-224 || true
"$GP" ../check_dihedral.py "$CIF" 5-50 51-100 101-150 151-224 || true
echo done
