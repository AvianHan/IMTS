#!/usr/bin/env python3
"""Run a named pilot group locally, sequentially; default is validation only.

The Slurm submitter handles parallel scheduling.  This script intentionally does
not launch unbounded local GPU processes and stops at the first failed fit.
"""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main(argv=None):
    from imts_agents.variants import GROUPS, variants_for_group
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=tuple(GROUPS), default="core")
    parser.add_argument("--dataset", default="JapaneseVowels")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", default="results/method-pilot")
    parser.add_argument("--cache-dir")
    parser.add_argument("--smoke", action="store_true")
    # All other runner options are forwarded unchanged, including --epochs and
    # explicit --evaluate-test. A caller cannot override group task identities.
    args, extra = parser.parse_known_args(argv)
    if any(item.split("=", 1)[0] in {"--variant", "--dataset", "--seed", "--device", "--output-dir", "--cache-dir"} for item in extra):
        parser.error("Task identity options must be passed once through this script")
    for spec in variants_for_group(args.group):
        command = [sys.executable, "-m", "imts_agents", "run", "--dataset", args.dataset,
                   "--variant", spec.name, "--seed", str(args.seed), "--device", args.device,
                   "--output-dir", args.output_dir, *extra]
        if args.cache_dir:
            command += ["--cache-dir", args.cache_dir]
        if args.smoke:
            command += ["--smoke"]
        print(f"Starting {spec.name}: {args.dataset}, seed {args.seed}", flush=True)
        import os
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        result = subprocess.run(command, cwd=ROOT, env=env)
        if result.returncode:
            return result.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
