#!/usr/bin/env bash
#SBATCH --job-name=imts-smoke
#SBATCH --account=prj0000000262
#SBATCH --partition=testqueue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
# Submit from the repository root, or pass --chdir=/absolute/path/to/IMTS.
if [[ ! -x .venv/bin/python ]]; then
  echo "Missing .venv/bin/python; check sbatch --chdir and bootstrap.sh" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export NUMBA_NUM_THREADS="$OMP_NUM_THREADS"

model="${IMTS_MODEL:-timesnet}"
device=cpu
if [[ "$model" == timesnet || "$model" == brits ]]; then
  device=cuda
fi
.venv/bin/python -m imts_benchmark --threads "${SLURM_CPUS_PER_TASK:-4}" doctor
srun .venv/bin/python -m imts_benchmark --threads "${SLURM_CPUS_PER_TASK:-4}" smoke \
  --model "$model" --dataset "${IMTS_DATASET:-synthetic}" \
  --device "$device" --max-dense-gib "${IMTS_MAX_DENSE_GIB:-8}" \
  --output-dir "${IMTS_OUTPUT_DIR:-results/smoke}"
