#!/usr/bin/env bash
# Usage: PYTHON_BIN=python3.11 bash scripts/bootstrap.sh base|cpu|cu124
set -euo pipefail

profile="${1:-base}"
case "$profile" in
  base|cpu|cu124) ;;
  *) echo "Usage: $0 [base|cpu|cu124]" >&2; exit 2 ;;
esac
if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [base|cpu|cu124]" >&2
  exit 2
fi

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3}"
"$python_bin" -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Python 3.10-3.12 is required for the pinned dependencies"'
if [[ ! -x .venv/bin/python ]]; then
  "$python_bin" -m venv .venv
fi
.venv/bin/python -c 'import sys; assert (3, 10) <= sys.version_info[:2] < (3, 13), "Existing .venv needs Python 3.10-3.12"'
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt

if [[ "$profile" != base ]]; then
  # Explicit user selection: the CUDA version printed by nvidia-smi is not
  # automatically used as a wheel index.
  # Force reinstall also replaces a same-version wheel from another profile.
  if [[ "$(uname -s)" == Darwin ]]; then
    if [[ "$profile" != cpu ]]; then
      echo "macOS does not support the CUDA profile; use cpu." >&2
      exit 2
    fi
    .venv/bin/python -m pip install --force-reinstall "torch==2.6.0"
  else
    .venv/bin/python -m pip install --force-reinstall "torch==2.6.0" \
      --index-url "https://download.pytorch.org/whl/$profile"
  fi
  .venv/bin/python -m pip install -r requirements-neural.txt
fi
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -m imts_benchmark --threads "${IMTS_THREADS:-1}" doctor
echo "Environment ready: $repo_dir/.venv (profile: $profile)"
