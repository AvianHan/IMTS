#!/usr/bin/env python3
"""Run this checkout's upgraded launcher against an unchanged baseline checkout.

Only the launcher implementation comes from this checkout. Training source,
configuration, Python environment, result directory, templates, and accounting
all come from --baseline-root. This wrapper neither stops an existing watcher
nor changes baseline files. Submission still requires the explicit --submit flag.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
from pathlib import Path
import subprocess
import sys


LAUNCHER_ROOT = Path(__file__).resolve().parents[1]


def load_launcher():
    path = LAUNCHER_ROOT / "scripts/submit_benchmarks.py"
    spec = importlib.util.spec_from_file_location("imts_parallel_baseline_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load the upgraded baseline launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_baseline_root(value):
    root = Path(value).expanduser().resolve()
    if root == LAUNCHER_ROOT.resolve():
        raise ValueError("--baseline-root must be a separate unchanged baseline checkout, "
                         "not this method checkout.")
    required = (".venv/bin/python", "configs/datasets.json", "configs/baselines.json",
                "src/imts_benchmark/provenance.py", "slurm/benchmark_cpu.sbatch", "slurm/benchmark_gpu.sbatch")
    missing = [str(root / name) for name in required if not (root / name).is_file()]
    if missing:
        raise ValueError("Incomplete baseline checkout; required files are missing: " + ", ".join(missing))
    return root


@contextlib.contextmanager
def forwarded_arguments(path, arguments):
    previous = sys.argv
    sys.argv = [str(path), *arguments]
    try:
        yield
    finally:
        sys.argv = previous


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False, allow_abbrev=False)
    parser.add_argument("--baseline-root", default="~/IMTS", help="unchanged baseline checkout (default: ~/IMTS)")
    parser.add_argument("-h", "--help", action="store_true", help="show wrapper and forwarded launcher options")
    args, remaining = parser.parse_known_args(argv)
    launcher = load_launcher()
    if args.help:
        parser.print_help()
        print("\nForwarded baseline launcher options:")
        with forwarded_arguments(launcher.__file__, ["--help"]):
            return launcher.main()
    root = validate_baseline_root(args.baseline_root)
    launcher.ROOT = root
    print(f"Baseline training checkout: {root}", flush=True)
    print(f"Launcher implementation: {Path(launcher.__file__).resolve()}", flush=True)
    with forwarded_arguments(launcher.__file__, remaining):
        return launcher.main()


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
