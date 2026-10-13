#!/usr/bin/env bash
#SBATCH --job-name=imts-gpu
#SBATCH --account=prj0000000262
#SBATCH --partition=h200p
#SBATCH --qos=highpriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
# This account's h200p/highpriority request passed Orion --test-only validation.
# The one-hour limit is an initial budget; adjust it using measured full runs.
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
  --model "${IMTS_MODEL:-timesnet}" --dataset "${IMTS_DATASET:-JapaneseVowels}" \
  --seed "${IMTS_SEED:-0}" --device cuda \
  --max-dense-gib "${IMTS_MAX_DENSE_GIB:-8}" --output-dir "${IMTS_OUTPUT_DIR:-results}"
