#!/usr/bin/env bash
#SBATCH --job-name=imts-method
#SBATCH --account=prj0000000262
#SBATCH --partition=h200p
#SBATCH --qos=highpriority
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00

# Header/body template: submit_method_pilot.py appends the frozen worker command.
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMBA_NUM_THREADS=4
