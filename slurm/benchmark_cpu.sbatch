#!/usr/bin/env bash
#SBATCH --job-name=imts-cpu
#SBATCH --account=prj0000000262
#SBATCH --partition=cpup
#SBATCH --qos=highpriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
# This account's cpup/highpriority request passed Orion --test-only validation.
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; check sbatch --chdir and bootstrap.sh" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"

srun .venv/bin/python -m imts_benchmark --threads "${SLURM_CPUS_PER_TASK:-4}" run \
  --model "${IMTS_MODEL:-rocket}" --dataset "${IMTS_DATASET:-JapaneseVowels}" \
  --seed "${IMTS_SEED:-0}" --device cpu \
  --max-dense-gib "${IMTS_MAX_DENSE_GIB:-8}" --output-dir "${IMTS_OUTPUT_DIR:-results}"
