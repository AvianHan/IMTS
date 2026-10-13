"""Small CLI; set CPU thread budgets before importing numerical libraries."""
import argparse
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(description="Pyrregular six-baseline experiments")
    parser.add_argument("--threads", type=int, default=1, help="CPU thread budget (before subcommand)")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor", help="Report installed libraries, CUDA and source revision")
    inspect = sub.add_parser("inspect", help="Load real data and check the stored split")
    inspect.add_argument("--dataset", default="JapaneseVowels")
    inspect.add_argument("--cache-dir")
    inspect.add_argument("--max-dense-gib", type=float, default=8.0)
    for command in ("smoke", "run"):
        p = sub.add_parser(command)
        p.add_argument("--model", required=True)
        p.add_argument("--dataset", default="synthetic" if command == "smoke" else "JapaneseVowels")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
        p.add_argument("--output-dir", default="results")
        p.add_argument("--cache-dir")
        p.add_argument("--max-dense-gib", type=float, default=8.0)
    plan = sub.add_parser("plan", help="Write the 252 intended benchmark tasks; does not submit jobs")
    plan.add_argument("--output", default="experiment-plan.csv")
    summary = sub.add_parser("summarize", help="Aggregate successful benchmark runs only")
    summary.add_argument("--results-dir", default="results")
    summary.add_argument("--output", default="benchmark-summary.csv")
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be positive")
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        os.environ[key] = str(args.threads)
    try:
        if args.command == "doctor":
            return doctor()
        if args.command == "plan":
            return write_plan(args.output)
        if args.command == "summarize":
            return summarize(args.results_dir, args.output)
        from .data import load_data
        if args.command == "inspect":
            d = load_data(args.dataset, args.cache_dir, args.max_dense_gib)
            metadata = {k: v for k, v in d.metadata.items() if not k.endswith("source_indices")}
            print(json.dumps({"dataset": d.name, "train_shape": list(d.X_train.shape),
                              "test_shape": list(d.X_test.shape), "metadata": metadata}, indent=2, default=str))
            return 0
        from .runner import execute
        execute(model_name=args.model, dataset=args.dataset, seed=args.seed,
                smoke=args.command == "smoke", device=args.device, output_dir=args.output_dir,
                cache_dir=args.cache_dir, max_dense_gib=args.max_dense_gib)
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


def doctor():
    from .provenance import environment, UPSTREAM_REVISION
    out = environment()
    required = ("pyrregular", "sktime", "aeon", "lightgbm")
    missing = [name for name in required if importlib.util.find_spec(name) is None]
    out["missing_classical_libraries"] = missing
    out["neural_installed"] = all(importlib.util.find_spec(name) is not None for name in ("torch", "pypots"))
    if importlib.util.find_spec("torch"):
        import torch
        out["torch"] = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
                        "cuda_available": torch.cuda.is_available()}
        if torch.cuda.is_available():
            out["torch"]["device"] = torch.cuda.get_device_name(0)
    if importlib.util.find_spec("pyrregular"):
        from pyrregular.repository import DATA_REVISION
        out["data_revision"] = DATA_REVISION
        # Source file hashes are checked independently of install origin (wheel/git).
        import pyrregular
        import hashlib
        from .provenance import ROOT
        expected = json.loads((ROOT / "references/upstream_hashes.json").read_text())
        base = Path(pyrregular.__file__).parent
        mismatches = [path for path, digest in expected.items()
                      if not (base / path).exists() or hashlib.sha256((base / path).read_bytes()).hexdigest() != digest]
        out["upstream_source_mismatches"] = mismatches
        if mismatches:
            missing.append("pyrregular source mismatch")
    print(json.dumps(out, indent=2))
    return 1 if missing else 0


def write_plan(output):
    import csv
    from .data import DATASETS
    from .provenance import ROOT
    config = json.loads((ROOT / "configs/baselines.json").read_text())
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["task_id", "dataset", "model", "seed", "device"])
        writer.writeheader()
        task_id = 0
        for dataset in DATASETS:
            for model, settings in config["models"].items():
                for seed in settings["seeds"]:
                    writer.writerow(dict(task_id=task_id, dataset=dataset, model=model, seed=seed,
                                         device="cuda" if model in {"TIMESNET", "BRITS"} else "cpu"))
                    task_id += 1
    print(f"Wrote {task_id} tasks to {path}. No jobs submitted.")
    return 0


def summarize(results_dir, output):
    import hashlib
    import pandas as pd
    from .data import DATASETS
    from .provenance import ROOT
    config = json.loads((ROOT / "configs/baselines.json").read_text())
    records, seen = [], set()
    for path in sorted(Path(results_dir).rglob("result.json")):
        r = json.loads(path.read_text())
        if r["profile"] != "benchmark":
            continue
        key = (r["dataset"], r["model"], r["seed"])
        if r["dataset"] not in DATASETS or r["model"] not in config["models"]:
            raise ValueError(f"Benchmark result outside the configured experiment: {key}")
        if r["status"] not in {"success", "failed", "running"}:
            raise ValueError(f"Unknown benchmark run status in {path}: {r['status']}")
        if r["status"] == "success":
            if key in seen:
                raise ValueError(f"Duplicate successful seed {key}; select one experiment folder before summarizing")
            seen.add(key)
        records.append(r)
    if not records:
        raise ValueError("No benchmark runs; smoke scores are intentionally excluded")

    # Failed attempts are coverage records, never score/environment evidence.
    # In particular, retrying a failed installation does not contaminate a
    # subsequent successful experiment's environment comparison.
    successes = [r for r in records if r["status"] == "success"]
    if len({r["environment"]["source_sha256"] for r in successes}) > 1:
        raise ValueError("Mixed code versions; summarize each experiment separately")
    model_environments = {}
    shared_dependencies = {
        "numpy", "pandas", "scipy", "scikit-learn", "numba", "pyrregular",
        "sktime", "aeon", "lightgbm", "pypots",
    }
    shared_versions = {}
    for r in successes:
        model = r["model"]
        packages = {
            name.lower().replace("_", "-").replace(".", "-"): value
            for name, value in r["environment"]["packages"].items()
        }
        serialized = json.dumps(packages, sort_keys=True)
        previous = model_environments.setdefault(model, serialized)
        if previous != serialized:
            raise ValueError(f"Mixed environments within model {model}; summarize each experiment separately")
        for name in shared_dependencies & packages.keys():
            previous_version = shared_versions.setdefault(name, packages[name])
            if previous_version != packages[name]:
                raise ValueError(f"Mixed shared dependency {name}: {previous_version} vs {packages[name]}")
    # CPU and CUDA builds may differ between models, but every model's repeats
    # must use one identical package inventory. Shared scientific libraries
    # remain pinned across models, wherever both environments contain them.
    refs = pd.read_csv(ROOT / "references/pyrregular_f1_mean.csv").set_index("dataset")
    groups = []
    for dataset in DATASETS:
        for model, settings in config["models"].items():
            attempts = [r for r in records if (r["dataset"], r["model"]) == (dataset, model)]
            completed = [r for r in attempts if r["status"] == "success"]
            failed = [r for r in attempts if r["status"] == "failed"]
            actual = sorted(r["seed"] for r in completed)
            expected = settings["seeds"]
            reference = float(refs.loc[dataset, model]) if dataset in refs.index else float("nan")
            values = pd.Series([r["metrics"]["macro_f1"] for r in completed], dtype=float)
            avg = values.mean()
            package_record = model_environments.get(model)
            groups.append(dict(
                dataset=dataset, model=model, completed_runs=len(completed), expected_runs=len(expected),
                failed_runs=len(failed), running_runs=sum(r["status"] == "running" for r in attempts),
                expected_seeds=json.dumps(expected), actual_seeds=json.dumps(actual),
                failed_seeds=json.dumps(sorted({r["seed"] for r in failed} - set(actual))),
                missing_seeds=json.dumps(sorted(set(expected) - set(actual))),
                complete=set(actual) == set(expected), macro_f1_mean=avg,
                macro_f1_std=values.std(ddof=1) if len(values) > 1 else float("nan"),
                paper_macro_f1=reference, difference=avg-reference,
                mean_fit_seconds=pd.Series([r["fit_seconds"] for r in completed], dtype=float).mean(),
                environment_sha256=(hashlib.sha256(package_record.encode()).hexdigest()
                                    if package_record is not None and completed else ""),
            ))
    result = pd.DataFrame(groups)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    if Path(output).exists():
        raise FileExistsError(output)
    result.to_csv(output, index=False)
    reported = result[result.completed_runs > 0]
    if not reported.empty:
        print(reported[["dataset", "model", "completed_runs", "expected_runs", "failed_runs",
                        "complete", "macro_f1_mean", "macro_f1_std", "difference"]].to_string(index=False))
    print(f"Coverage: {int(result.complete.sum())}/{len(result)} dataset/model combinations complete; "
          f"{int(result.completed_runs.sum())}/{int(result.expected_runs.sum())} successful fits; "
          f"{int(result.failed_runs.sum())} failed attempts; {int(result.running_runs.sum())} unfinished records.")
    print(f"Full coverage CSV: {output}. Unrun/all-failed combinations have blank scores, not zero. "
          "failed_runs counts attempts; failed_seeds excludes seeds subsequently completed.")
    print("std uses ddof=1; historical paper ddof is not documented. Missing references remain empty.")
    return 0
