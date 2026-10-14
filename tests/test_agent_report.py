from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from imts_agents import report
from imts_agents.variants import get_variant


@pytest.fixture
def folders(tmp_path):
    method, baseline = tmp_path / "method", tmp_path / "baseline"
    method.mkdir()
    baseline.mkdir()
    return method, baseline


def method_record(seed=0, variant="history_state", score=0.6):
    config = {"dataset": "JapaneseVowels", "variant": variant, "seed": seed,
              "device": "cuda", "threads": 4, "epochs": 100, "patience": 15,
              "smoke": False, "evaluate_test": True}
    datasets = json.loads((report.ROOT / "configs/datasets.json").read_text())
    item = next(item for item in datasets["datasets"] if item["name"] == "JapaneseVowels")
    protocol = {key: value for key, value in config.items() if key != "variant"}
    protocol.update(source_sha256="method-source", sample_ids_sha256={"test": "official-test"})
    return {"experiment_family": "imts_agents", "status": "success", "profile": "test",
            "test_evaluated": True, "dataset": "JapaneseVowels", "variant": variant, "seed": seed,
            "config": config, "config_sha256": report._digest(config),
            "comparison_protocol": protocol, "comparison_protocol_sha256": report._digest(protocol),
            "variant_spec": asdict(get_variant(variant)),
            "environment": {"source_sha256": "method-source", "packages": {"torch": "2.1", "numpy": "2.0"}},
            "data": {"source": "pyrregular", "sha256": item["sha256"], "data_revision": datasets["data_revision"],
                     "sample_ids_sha256": {"test": "official-test"}},
            "metrics": {"macro_f1": 0.99}, "validation": {"metrics": {"macro_f1": 0.99}},
            "test": {"metrics": {"macro_f1": score}, "cost": {"predict_seconds": 3}},
            "fit_seconds": 10 + seed, "predict_seconds": 1, "total_seconds": 15,
            "hardware": {"device": "cuda", "cuda_device_name": "H200"},
            "memory": {"cuda_peak_allocated_bytes": 1000}}


def baseline_record(model="BORF", seed=0, score=0.5):
    method = method_record(seed)
    return {"status": "success", "profile": "benchmark", "dataset": "JapaneseVowels", "model": model, "seed": seed,
            "environment": {"source_sha256": "baseline-source", "packages": {"numpy": "2.0", "pyrregular": "1.0"}},
            "data": {key: value for key, value in method["data"].items() if key != "sample_ids_sha256"} | {"test_ids_sha256": "official-test"},
            "parameters": {"random_state": seed, "model__random_state": seed, "epochs": 100},
            "metrics": {"macro_f1": score}, "fit_seconds": 20, "predict_seconds": 2, "requested_device": "cpu"}


def save(folder, name, record):
    path = folder / name / "result.json"
    path.parent.mkdir()
    path.write_text(json.dumps(record))


def row(result, model):
    return next(row for row in result["rows"] if row["dataset"] == "JapaneseVowels" and row["model"] == model)


def test_test_metrics_partial_seeds_cpu_counts_and_csv(folders, tmp_path):
    method, baseline = folders
    save(method, "one", method_record(score=0.4))
    save(method, "two", method_record(seed=1, score=0.8))
    save(baseline, "borf", baseline_record())
    save(baseline, "rocket", baseline_record("ROCKET"))
    result = report.compare_results(method, baseline, tmp_path / "scores.csv")
    partial = row(result, "history_state")
    assert partial["macro_f1_mean"] == pytest.approx(0.6)
    assert partial["macro_f1_std"] == pytest.approx(0.2828427)
    assert partial["successful_seeds"] == [0, 1]
    assert partial["expected_seeds"] == [0, 1, 2]
    assert partial["complete"] is False and partial["mean_scope"] == "partial"
    assert partial["mean_test_predict_seconds"] == 3
    assert row(result, "BORF")["complete"] is True
    assert row(result, "BORF")["expected_runs"] == 1
    assert row(result, "BORF")["macro_f1_std"] is None
    assert row(result, "ROCKET")["expected_runs"] == 3
    assert row(result, "BRITS")["macro_f1_mean"] is None
    assert len(result["rows"]) == 18 * 16
    assert sum(row["expected_runs"] for row in result["coverage"]) == 540 + 252
    assert "2/54" in report.format_coverage(result)
    assert json.loads(json.dumps(result, allow_nan=False))["runs"][0]["test_predict_seconds"] == 3
    assert "fit_seconds_by_seed" in (tmp_path / "scores.csv").read_text()
    with pytest.raises(FileExistsError):
        report.compare_results(method, baseline, tmp_path / "scores.csv")


def test_validation_smoke_and_unsuccessful_do_not_supply_scores(folders):
    method, baseline = folders
    for kind in ("validation", "smoke", "failed", "running", "test_failure"):
        record = method_record()
        if kind in {"validation", "smoke"}:
            record["profile"] = kind
            record["config"]["smoke"] = kind == "smoke"
        elif kind == "test_failure":
            record["test_evaluation_status"] = "failed"
        else:
            record["status"] = kind
            record["test_evaluated"] = False
        save(method, kind, record)
    smoke = baseline_record()
    smoke["profile"] = "smoke"
    save(baseline, "smoke", smoke)
    result = report.compare_results(method, baseline)
    assert result["runs"] == []
    assert row(result, "history_state")["macro_f1_mean"] is None
    assert row(result, "history_state")["failed_attempts"] == 1
    assert row(result, "history_state")["unfinished_records"] == 1
    assert result["excluded"]["method:failed"] == 1
    assert result["excluded"]["method:running"] == 1


@pytest.mark.parametrize("family", ["method", "baseline"])
def test_duplicate_success_refused_even_if_score_is_better(folders, family):
    folder = folders[0 if family == "method" else 1]
    record = method_record() if family == "method" else baseline_record()
    save(folder, "first", record)
    save(folder, "second", deepcopy(record))
    with pytest.raises(ValueError, match="Duplicate successful seed"):
        report.compare_results(*folders)


@pytest.mark.parametrize("change", ["source", "environment", "training", "split", "protocol", "variant"])
def test_incompatible_method_records_refused(folders, change):
    method, _ = folders
    save(method, "first", method_record())
    second = method_record(seed=1)
    if change == "source":
        second["environment"]["source_sha256"] = "different"
    elif change == "environment":
        second["environment"]["packages"]["torch"] = "different"
    elif change == "training":
        second["config"]["epochs"] = 10
        second["config_sha256"] = report._digest(second["config"])
    elif change == "split":
        second["data"]["sample_ids_sha256"]["test"] = "different"
    elif change == "variant":
        second["variant_spec"]["communication"] = False
    else:
        second = method_record(variant="write_content")
        second["comparison_protocol"]["preprocessing_sha256"] = "different"
        second["comparison_protocol_sha256"] = report._digest(second["comparison_protocol"])
    save(method, "second", second)
    with pytest.raises(ValueError, match="Incompatible"):
        report.compare_results(*folders)


def test_baseline_repeats_allow_seed_change_but_not_hyperparameter_change(folders):
    _, baseline = folders
    save(baseline, "seed0", baseline_record("ROCKET", 0))
    second = baseline_record("ROCKET", 1)
    save(baseline, "seed1", second)
    assert row(report.compare_results(*folders), "ROCKET")["successful_runs"] == 2
    second["parameters"]["epochs"] = 10
    (baseline / "seed1" / "result.json").write_text(json.dumps(second))
    with pytest.raises(ValueError, match="hyperparameters"):
        report.compare_results(*folders)


def test_official_test_identity_must_match_across_method_and_baselines(folders):
    method, baseline = folders
    save(method, "method", method_record())
    record = baseline_record()
    record["data"]["test_ids_sha256"] = "another-test-split"
    save(baseline, "baseline", record)
    with pytest.raises(ValueError, match="official test sample identities"):
        report.compare_results(*folders)


def test_hardware_change_is_reported_without_splitting_accuracy_protocol(folders):
    method, _ = folders
    save(method, "first", method_record())
    second = method_record(variant="write_static")
    second["config"]["device"] = "cpu"
    second["config"]["threads"] = 1
    second["config_sha256"] = report._digest(second["config"])
    second["comparison_protocol"].update(device="cpu", threads=1)
    second["comparison_protocol_sha256"] = report._digest(second["comparison_protocol"])
    second["hardware"] = {"device": "cpu"}
    save(method, "second", second)
    result = report.compare_results(*folders)
    assert {run["hardware"]["device"] for run in result["runs"]} == {"cpu", "cuda"}


def test_config_and_protocol_cannot_disagree_even_if_each_hash_is_valid(folders):
    method, _ = folders
    record = method_record()
    record["comparison_protocol"]["epochs"] = 1
    record["comparison_protocol_sha256"] = report._digest(record["comparison_protocol"])
    save(method, "record", record)
    with pytest.raises(ValueError, match="Configuration disagrees"):
        report.compare_results(*folders)
