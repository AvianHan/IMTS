"""Compare held-out test results without selecting among repeated successes.

This module reads small JSON records only. Validation, smoke, failed and running
attempts never contribute scores. Incomplete seed groups remain visibly partial.
"""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

from .variants import GROUPS


ROOT = Path(__file__).resolve().parents[2]
METHOD_SEEDS = (0, 1, 2)
_HARDWARE_CONFIG = {"device", "threads", "evaluation_device"}


def _serialized(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_serialized(value).encode()).hexdigest()


def _require_same(store, key, value, description):
    signature = _serialized(value)
    if key in store and store[key] != signature:
        raise ValueError(f"Incompatible {description} for {key}; select a single experiment protocol")
    store[key] = signature


def _finite(value, field, path, *, maximum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
        raise ValueError(f"Invalid {field} in {path}")
    if maximum is not None and value > maximum:
        raise ValueError(f"Invalid {field} in {path}")
    return float(value)


def _packages(environment):
    packages = environment.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("Missing package provenance in a successful test record")
    return {key.lower().replace("_", "-").replace(".", "-"): value for key, value in packages.items()}


def _settings_without_seed(value):
    if isinstance(value, dict):
        return {key: _settings_without_seed(item) for key, item in value.items()
                if key.split("__")[-1] not in {"random_state", "seed", "device"}}
    if isinstance(value, list):
        return [_settings_without_seed(item) for item in value]
    return value


def compare_results(method_root, baseline_root, output=None):
    """Return JSON-safe coverage, dataset rows and per-run runtime/provenance.

    ``output`` optionally creates a new aggregate CSV (never overwrites one).
    Method and baseline code hashes may differ, but each family must internally
    use one source version. Method hyperparameters must agree across variants;
    dataset/seed-specific protocols must agree for the same dataset and seed.
    Hardware differences are recorded rather than silently treated as equal cost.
    """
    datasets_config = json.loads((ROOT / "configs/datasets.json").read_text())
    baseline_config = json.loads((ROOT / "configs/baselines.json").read_text())
    datasets = {item["name"]: item for item in datasets_config["datasets"]}
    expected = {("method", name): METHOD_SEEDS for name in GROUPS["ablation"]}
    expected.update({("baseline", name): tuple(item["seeds"])
                     for name, item in baseline_config["models"].items()})
    runs, seen, signatures, excluded, attempts = [], {}, {}, Counter(), Counter()
    for family, directory in (("method", method_root), ("baseline", baseline_root)):
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"Result directory does not exist: {directory}")
        for path in sorted(directory.rglob("result.json")):
            try:
                record = json.loads(path.read_text())
            except (OSError, ValueError) as exc:
                raise ValueError(f"Unreadable result record: {path}") from exc
            if family == "method":
                official = (record.get("experiment_family") == "imts_agents"
                            and record.get("profile") == "test"
                            and record.get("config", {}).get("smoke") is False)
                model = record.get("variant")
            else:
                official = record.get("profile") == "benchmark" and not record.get("config", {}).get("smoke", False)
                model = record.get("model")
            if not official:
                excluded[f"{family}:not_official_test"] += 1
                continue
            if (family, model) not in expected:
                excluded[f"{family}:outside_selected_models"] += 1
                continue
            dataset, seed = record.get("dataset"), record.get("seed")
            if dataset not in datasets or isinstance(seed, bool) or seed not in expected[family, model]:
                raise ValueError(f"Unexpected dataset/seed in {path}: {(dataset, seed)}")
            status = record.get("status")
            if status not in {"success", "failed", "running"}:
                raise ValueError(f"Unknown run status in {path}: {status}")
            # A planned test fit sets test_evaluated only after its evaluation
            # succeeds. Its failed/running attempts still belong to coverage.
            if (family == "method" and status == "success"
                    and (record.get("test_evaluated") is not True
                         or record.get("test_evaluation_status", "success") != "success")):
                excluded[f"{family}:not_official_test"] += 1
                continue
            attempts[family, dataset, model, status] += 1
            if status != "success":
                excluded[f"{family}:{status}"] += 1
                continue
            key = family, dataset, model, seed
            if key in seen:
                raise ValueError(f"Duplicate successful seed {key}: {seen[key]} and {path}; do not select the best score")
            seen[key] = str(path)
            env = record.get("environment", {})
            source = env.get("source_sha256")
            if not source:
                raise ValueError(f"Missing source provenance in {path}")
            _require_same(signatures, (family, "source"), source, "source versions")
            packages = _packages(env)
            _require_same(signatures, (family, "environment", model if family == "baseline" else "all"),
                          {"packages": packages, "python": env.get("python")}, "software environments")
            data = record.get("data", {})
            if (data.get("source") != "pyrregular" or data.get("sha256") != datasets[dataset]["sha256"]
                    or data.get("data_revision") != datasets_config["data_revision"]):
                raise ValueError(f"Unverified or incompatible dataset source in {path}")
            if family == "method":
                config = record["config"]
                if record.get("config_sha256") != _digest(config):
                    raise ValueError(f"Missing or modified configuration in {path}")
                if any(config.get(name) != record.get(name) for name in ("dataset", "seed", "variant")):
                    raise ValueError(f"Configuration identity disagrees with result in {path}")
                train_config = {key: value for key, value in config.items()
                                if key not in _HARDWARE_CONFIG | {"dataset", "seed", "variant", "evaluate_test", "smoke"}}
                _require_same(signatures, (family, "training"), train_config, "training hyperparameters")
                protocol = record.get("comparison_protocol")
                if not protocol or record.get("comparison_protocol_sha256") != _digest(protocol):
                    raise ValueError(f"Missing or modified comparison protocol in {path}")
                if protocol.get("source_sha256") != source:
                    raise ValueError(f"Comparison protocol source disagrees with environment in {path}")
                if any(protocol.get(name) != value for name, value in config.items()
                       if name not in {"variant", "evaluate_test"}):
                    raise ValueError(f"Configuration disagrees with comparison protocol in {path}")
                if protocol.get("evaluate_test") is not True:
                    raise ValueError(f"Comparison protocol is not an official test evaluation in {path}")
                comparable_protocol = {key: value for key, value in protocol.items()
                                       if key not in _HARDWARE_CONFIG | {"evaluate_test"}}
                _require_same(signatures, (family, "protocol", dataset, seed), comparable_protocol,
                              "data splits/preprocessing/protocols")
                if not record.get("variant_spec"):
                    raise ValueError(f"Missing variant specification in {path}")
                _require_same(signatures, (family, "variant", model), record["variant_spec"], "variant definitions")
                test = record.get("test", {})
                metrics = test.get("metrics", {})
                prediction_seconds = test.get("cost", {}).get("predict_seconds")
                test_ids = data.get("sample_ids_sha256", {}).get("test")
            else:
                if not isinstance(record.get("parameters"), dict):
                    raise ValueError(f"Missing baseline parameters in {path}")
                _require_same(signatures, (family, "settings", model), _settings_without_seed(record["parameters"]),
                              "baseline model hyperparameters")
                for package in ("numpy", "pandas", "scipy", "scikit-learn", "numba", "pyrregular", "sktime", "aeon", "lightgbm", "pypots"):
                    if package in packages:
                        _require_same(signatures, (family, "shared", package), packages[package], "shared package versions")
                metrics = record.get("metrics", {})
                prediction_seconds = record.get("predict_seconds")
                test_ids = data.get("test_ids_sha256")
            if not test_ids:
                raise ValueError(f"Missing official test identity in {path}")
            _require_same(signatures, ("test_ids", dataset), test_ids, "official test sample identities")
            total_seconds = record.get("total_seconds")
            if total_seconds is not None:
                total_seconds = _finite(total_seconds, "total_seconds", path)
            runs.append({"family": family, "dataset": dataset, "model": model, "seed": seed,
                         "macro_f1": _finite(metrics.get("macro_f1"), "test macro_f1", path, maximum=1),
                         "fit_seconds": _finite(record.get("fit_seconds"), "fit_seconds", path),
                         "test_predict_seconds": _finite(prediction_seconds, "test predict_seconds", path),
                         "total_seconds": total_seconds, "hardware": record.get("hardware", {"device": record.get("requested_device")}),
                         "memory": record.get("memory"), "source_sha256": source,
                         "environment_sha256": _digest(packages), "path": str(path)})
    rows = []
    for dataset in datasets:
        for (family, model), seeds in expected.items():
            completed = sorted((run for run in runs if (run["family"], run["dataset"], run["model"]) == (family, dataset, model)),
                               key=lambda run: run["seed"])
            actual = [run["seed"] for run in completed]
            scores = [run["macro_f1"] for run in completed]
            complete = set(actual) == set(seeds)
            rows.append({"dataset": dataset, "family": family, "model": model,
                         "successful_seeds": actual, "expected_seeds": list(seeds),
                         "successful_runs": len(actual), "expected_runs": len(seeds), "complete": complete,
                         "mean_scope": "complete" if complete else "partial" if actual else "unavailable",
                         "macro_f1_mean": statistics.mean(scores) if scores else None,
                         "macro_f1_std": statistics.stdev(scores) if len(scores) > 1 else None,
                         "mean_fit_seconds": statistics.mean(run["fit_seconds"] for run in completed) if completed else None,
                         "mean_test_predict_seconds": statistics.mean(run["test_predict_seconds"] for run in completed) if completed else None,
                         "failed_attempts": attempts[family, dataset, model, "failed"],
                         "unfinished_records": attempts[family, dataset, model, "running"],
                         "fit_seconds_by_seed": {run["seed"]: run["fit_seconds"] for run in completed},
                         "test_predict_seconds_by_seed": {run["seed"]: run["test_predict_seconds"] for run in completed},
                         "run_paths": [run["path"] for run in completed]})
    coverage = []
    for family, model in expected:
        selected = [row for row in rows if (row["family"], row["model"]) == (family, model)]
        coverage.append({"family": family, "model": model,
                         "successful_runs": sum(row["successful_runs"] for row in selected),
                         "expected_runs": sum(row["expected_runs"] for row in selected),
                         "complete_datasets": sum(row["complete"] for row in selected),
                         "expected_datasets": len(datasets)})
    report = {"rows": rows, "coverage": coverage, "runs": runs, "excluded": dict(excluded),
              "note": "Official test scores only. Partial means are explicitly marked and are not complete benchmark results. "
                      "No failed score is imputed as zero. Standard deviation uses ddof=1; one-seed std is unavailable. "
                      "Method and baseline implementations/environments may differ; runtime is descriptive and hardware is retained. "
                      "No cross-dataset ranking or automatic best-variant selection is performed."}
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _serialized(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        report["csv"] = str(output)
    return report


def format_coverage(report):
    """Small terminal-friendly table; full scores are in report['rows']/CSV."""
    lines = [f"{'Model / variant':<32} {'Successful fits':>17} {'Complete datasets':>19}"]
    for row in report["coverage"]:
        count = f"{row['successful_runs']}/{row['expected_runs']}"
        complete = f"{row['complete_datasets']}/{row['expected_datasets']}"
        lines.append(f"{row['model']:<32} {count:>17} {complete:>19}")
    lines.append(report["note"])
    return "\n".join(lines)
