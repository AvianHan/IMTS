#!/usr/bin/env bash
# Usage: bash scripts/smoke_all.sh [synthetic|DATASET] [cpu|cuda] [OUTPUT_DIR]
set -euo pipefail

if [[ $# -gt 3 ]]; then
  echo "Usage: $0 [synthetic|DATASET] [cpu|cuda] [OUTPUT_DIR]" >&2
  exit 2
fi
dataset="${1:-synthetic}"
neural_device="${2:-cpu}"
output_dir="${3:-results/smoke}"
case "$neural_device" in
  cpu|cuda) ;;
  *) echo "Neural device must be cpu or cuda" >&2; exit 2 ;;
esac
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv first with scripts/bootstrap.sh cpu or cu124" >&2
  exit 1
fi
export PYTHONUNBUFFERED=1
for model in rocket borf lgbm rifc timesnet brits; do
  device=cpu
  if [[ "$model" == timesnet || "$model" == brits ]]; then
    device="$neural_device"
  fi
  echo "Smoke check: model=$model dataset=$dataset device=$device"
  .venv/bin/python -m imts_benchmark --threads "${IMTS_THREADS:-1}" smoke \
    --model "$model" --dataset "$dataset" --device "$device" \
    --max-dense-gib "${IMTS_MAX_DENSE_GIB:-8}" --output-dir "$output_dir"
done
