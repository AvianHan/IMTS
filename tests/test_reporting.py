import json
import csv
import pytest
from imts_benchmark.cli import summarize, write_plan


def _record(tmp_path, name, *, model="BRITS", seed=0, status="success", packages=None,
            source="same-source", profile="benchmark"):
    directory = tmp_path / name
    directory.mkdir()
    result = {
        "profile": profile, "status": status, "dataset": "JapaneseVowels", "model": model,
        "seed": seed, "environment": {"source_sha256": source, "packages": packages or {
            "numpy": "1.26.4", "pandas": "2.2.3", "pyrregular": "0.0.test",
        }},
    }
    if status == "success":
        result.update(metrics={"macro_f1": 0.8}, fit_seconds=1.5)
    (directory / "result.json").write_text(json.dumps(result))


def _summary_rows(tmp_path):
    output = tmp_path / "summary.csv"
    assert summarize(tmp_path, output) == 0
    with output.open() as stream:
        return {(row["dataset"], row["model"]): row for row in csv.DictReader(stream)}


def test_plan_has_six_models_18_datasets_252_fits(tmp_path):
    import csv
    path = tmp_path / "plan.csv"
    assert write_plan(path) == 0
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 252
    assert len({r["dataset"] for r in rows}) == 18
    assert len({r["model"] for r in rows}) == 6
    assert len({(r["dataset"], r["model"], r["seed"]) for r in rows}) == 252


def test_smoke_cannot_be_reported_as_benchmark(tmp_path):
    (tmp_path / "result.json").write_text(json.dumps({"profile": "smoke", "status": "success"}))
    with pytest.raises(ValueError, match="smoke"):
        summarize(tmp_path, tmp_path / "summary.csv")


def test_failed_runs_not_reported_as_zero(tmp_path):
    for seed in range(3):
        _record(tmp_path, f"failed-{seed}", seed=seed, status="failed")
    rows = _summary_rows(tmp_path)
    assert len(rows) == 18 * 6
    row = rows["JapaneseVowels", "BRITS"]
    assert row["complete"] == "False"
    assert row["completed_runs"] == "0" and row["failed_runs"] == "3"
    assert row["macro_f1_mean"] == "" and row["difference"] == ""
    assert json.loads(row["expected_seeds"]) == [0, 1, 2]
    assert json.loads(row["actual_seeds"]) == []
    assert json.loads(row["failed_seeds"]) == [0, 1, 2]
    assert rows["JapaneseVowels", "ROCKET"]["macro_f1_mean"] == ""


def test_different_model_cpu_cuda_environments_can_be_combined(tmp_path):
    shared = {"numpy": "1.26.4", "pandas": "2.2.3", "pyrregular": "0.0.test"}
    _record(tmp_path, "cpu", model="LGBM", packages={**shared, "torch": "2.6.0+cpu"})
    _record(tmp_path, "gpu", packages={**shared, "torch": "2.6.0+cu124", "pypots": "1.5"})
    rows = _summary_rows(tmp_path)
    assert rows["JapaneseVowels", "LGBM"]["complete"] == "True"
    assert rows["JapaneseVowels", "BRITS"]["complete"] == "False"
    assert (rows["JapaneseVowels", "LGBM"]["environment_sha256"] !=
            rows["JapaneseVowels", "BRITS"]["environment_sha256"])


def test_same_model_repeats_require_one_environment(tmp_path):
    _record(tmp_path, "cpu", seed=0, packages={"torch": "2.6.0+cpu"})
    _record(tmp_path, "gpu", seed=1, packages={"torch": "2.6.0+cu124"})
    with pytest.raises(ValueError, match="within model BRITS"):
        summarize(tmp_path, tmp_path / "summary.csv")


def test_shared_scientific_dependencies_must_match_across_models(tmp_path):
    _record(tmp_path, "cpu", model="LGBM", packages={"scikit-learn": "1.5.2"})
    _record(tmp_path, "gpu", packages={"scikit_learn": "1.6.0"})
    with pytest.raises(ValueError, match="shared dependency scikit-learn"):
        summarize(tmp_path, tmp_path / "summary.csv")


def test_source_version_must_match_across_models(tmp_path):
    _record(tmp_path, "first", model="LGBM")
    _record(tmp_path, "second", source="changed-source")
    with pytest.raises(ValueError, match="Mixed code"):
        summarize(tmp_path, tmp_path / "summary.csv")


def test_duplicate_successful_seed_is_rejected(tmp_path):
    _record(tmp_path, "first")
    _record(tmp_path, "second")
    with pytest.raises(ValueError, match="Duplicate successful seed"):
        summarize(tmp_path, tmp_path / "summary.csv")


def test_failed_retry_and_smoke_do_not_contaminate_scores_or_environments(tmp_path):
    _record(tmp_path, "failed", status="failed", source="old-code", packages={"numpy": "old"})
    _record(tmp_path, "smoke", profile="smoke", source="smoke-code", packages={"numpy": "other"})
    _record(tmp_path, "success")
    rows = _summary_rows(tmp_path)
    row = rows["JapaneseVowels", "BRITS"]
    assert row["completed_runs"] == "1" and row["failed_runs"] == "1"
    assert row["macro_f1_mean"] == "0.8"
    assert json.loads(row["actual_seeds"]) == [0]
    assert json.loads(row["failed_seeds"]) == []
    assert json.loads(row["missing_seeds"]) == [1, 2]
