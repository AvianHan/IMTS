"""Offline launcher safety checks: no dataset downloads, ML imports or Slurm jobs."""

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "imts_batch_launcher_test", REPO / "scripts/submit_benchmarks.py"
)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)

ENVIRONMENT = {"python": "3.10.12", "packages": {"numpy": "1.26.4", "torch": "2.6.0+cu124"}}


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    root = tmp_path / "IMTS"
    for name in ("configs", "slurm"):
        shutil.copytree(REPO / name, root / name)
    source = root / "src/imts_benchmark/model.py"
    source.parent.mkdir(parents=True)
    source.write_text("# Synthetic source fixture; no model is imported.\n")
    output = root / "results/jv-first-full"
    config = launcher.read_json(root / "configs/datasets.json")
    monkeypatch.setattr(launcher, "ROOT", root)
    monkeypatch.setattr(launcher, "inventory", lambda _: ENVIRONMENT)
    return SimpleNamespace(
        root=root, output=output, ledger=root / "results/accounting/batches",
        source=launcher.source_digest(root), source_file=source,
        revision=config["data_revision"],
        datasets={row["name"]: row for row in config["datasets"]},
    )


def task(model="ROCKET", seed=0):
    return {"dataset": "JapaneseVowels", "model": model, "seed": seed,
            "device": "cuda" if model in {"TIMESNET", "BRITS"} else "cpu"}


def record(exp, selected, attempt="first", status="success", profile="benchmark", **updates):
    directory = (exp.output / "benchmark" / selected["dataset"] / selected["model"] /
                 f"seed-{selected['seed']}" / attempt)
    directory.mkdir(parents=True)
    value = {
        **selected, "status": status, "profile": profile,
        "requested_device": selected["device"],
        "environment": {**ENVIRONMENT, "source_sha256": exp.source},
        "data": {"smoke_subset": False, "data_revision": exp.revision},
        "thread_limits": {name: "4" for name in
                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }
    value.update(updates)
    (directory / "result.json").write_text(json.dumps(value))
    # The planner checks presence; numerical prediction validation belongs to the runner.
    if status == "success":
        (directory / "predictions.npz").write_bytes(b"fixture")
    return directory / "result.json"


def snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() if path.is_file() else None
            for path in root.rglob("*")}


def worker_arguments(exp, tasks, group="gpu"):
    manifest = exp.root / "worker-manifest.json"
    template = exp.root / "slurm" / f"benchmark_{group}.sbatch"
    launcher.write_new(manifest, {
        "batch_id": "offline-test", "worker_sha256": launcher.digest(launcher.__file__),
        "root": str(exp.root), "output_root": str(exp.output),
        "source_sha256": exp.source, "tasks": {group: tasks},
        "environment": ENVIRONMENT, "python": str(exp.root / ".venv/bin/python"),
        "data_revision": exp.revision, "datasets": exp.datasets, "max_dense_gib": 8,
        "templates": {group: {"path": str(template), "sha256": launcher.digest(template)}},
    })
    return SimpleNamespace(worker=str(manifest), manifest_sha256=launcher.digest(manifest), group=group)


def test_dry_run_reuses_exactly_six_successes_without_writing_or_submitting(experiment, monkeypatch, capsys):
    exp = experiment
    for model in ("ROCKET", "BORF", "LGBM", "RIFC", "TIMESNET", "BRITS"):
        record(exp, task(model))
    # Neither failure nor a smoke score may satisfy a benchmark task.
    record(exp, task("ROCKET", 1), status="failed")
    record(exp, task("ROCKET", 2), profile="smoke")
    before = snapshot(exp.root)
    monkeypatch.setattr(sys, "argv", ["submit_benchmarks.py"])
    def unexpected_call(*args, **kwargs):
        pytest.fail(f"Dry run attempted an external command: {args[0]}")
    monkeypatch.setattr(launcher.subprocess, "run", unexpected_call)

    assert launcher.main() == 0

    output = capsys.readouterr().out
    assert "Planned fits: 252 | Compatible successes: 6" in output
    assert "Remaining: 246 = CPU 140 + GPU 106" in output
    assert "DRY RUN: no jobs submitted" in output
    assert snapshot(exp.root) == before


@pytest.mark.parametrize("change,expected", [
    ({"source_sha256": "old-source"}, "source hash"),
    ({"packages": {"numpy": "2.0.0"}}, "package inventory"),
])
def test_incompatible_success_is_rejected(experiment, change, expected):
    exp = experiment
    record(exp, task(), environment={**ENVIRONMENT, "source_sha256": exp.source, **change})
    with pytest.raises(RuntimeError, match=expected):
        launcher.successful_tasks(exp.output, [task()], exp.source, ENVIRONMENT, exp.revision)


def test_duplicate_success_cannot_be_silently_selected(experiment):
    exp = experiment
    record(exp, task(), attempt="one")
    record(exp, task(), attempt="two")
    with pytest.raises(RuntimeError, match="Duplicate successful run"):
        launcher.successful_tasks(exp.output, [task()], exp.source, ENVIRONMENT, exp.revision)


@pytest.mark.parametrize("receipt_status,expected", [
    (None, "Submission outcome is unknown"),
    ("unconfirmed", "Unconfirmed submission"),
    ("submitted", "still active"),
])
def test_live_or_ambiguous_previous_submission_blocks_retry(experiment, monkeypatch, receipt_status, expected):
    exp = experiment
    folder = exp.ledger / "old-batch"
    folder.mkdir(parents=True)
    launcher.write_new(folder / "manifest.json", {
        "output_root": str(exp.output), "tasks": {"cpu": [task()], "gpu": []},
    })
    launcher.write_new(folder / "cpu.intent.json", {"job_name": "old-batch"})
    if receipt_status:
        launcher.write_new(folder / "cpu.receipt.json", {"status": receipt_status, "job_id": "123"})
    monkeypatch.setattr(launcher, "array_is_terminal", lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match=expected):
        launcher.check_previous_batches(exp.ledger, exp.output, [task()])


def test_queue_activity_or_missing_accounting_never_proves_completion(monkeypatch):
    calls = []
    def queued(command, **kwargs):
        calls.append(command[0])
        return subprocess.CompletedProcess(command, 0, "123_0|COMPLETING\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", queued)
    assert launcher.array_is_terminal("123", 2) is False
    assert calls == ["squeue"]

    monkeypatch.setattr(launcher.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 0, "", ""))
    with pytest.raises(RuntimeError, match="absent|accounting|evidence"):
        launcher.array_is_terminal("123", 2)


def test_partial_array_accounting_or_queue_error_cannot_authorize_retry(monkeypatch):
    def partial(command, **kwargs):
        output = "" if command[0] == "squeue" else "123_0|COMPLETED\n"
        return subprocess.CompletedProcess(command, 0, output, "")
    monkeypatch.setattr(launcher.subprocess, "run", partial)
    with pytest.raises(RuntimeError, match="accounting|missing|Missing|incomplete|Incomplete"):
        launcher.array_is_terminal("123", 2)

    def unavailable(command, **kwargs):
        assert command[0] == "squeue"
        return subprocess.CompletedProcess(command, 1, "", "controller unavailable")
    monkeypatch.setattr(launcher.subprocess, "run", unavailable)
    with pytest.raises(RuntimeError, match="queue|Slurm|squeue"):
        launcher.array_is_terminal("123", 2)


def test_all_expected_terminal_array_indices_allow_retry(monkeypatch):
    def complete(command, **kwargs):
        output = "999_0|RUNNING\n" if command[0] == "squeue" else "123_0|COMPLETED\n123_1|TIMEOUT\n"
        return subprocess.CompletedProcess(command, 0, output, "")
    monkeypatch.setattr(launcher.subprocess, "run", complete)
    assert launcher.array_is_terminal("123", 2) is True


def test_partial_submission_keeps_receipts_and_does_not_retry(experiment, monkeypatch):
    exp = experiment
    args = SimpleNamespace(max_dense_gib=8, cpu_concurrency=4, gpu_concurrency=2,
                           cpu_time="02:00:00", gpu_time="01:00:00")
    models = launcher.read_json(exp.root / "configs/baselines.json")["models"]
    tasks = [{**task(model, seed), "dataset": dataset}
             for dataset in exp.datasets for model, settings in models.items()
             for seed in settings["seeds"]
             if not (dataset == "JapaneseVowels" and seed == 0)]
    real_run = subprocess.run
    calls = []
    def fake_sbatch(command, **kwargs):
        calls.append(command)
        assert command[0] == "sbatch"
        assert "SBATCH_QOS" not in kwargs["env"]
        folder = next(exp.ledger.iterdir())
        if len(calls) == 1:
            assert (folder / "cpu.intent.json").exists()
            return subprocess.CompletedProcess(command, 0, "6001\n", "")
        # The first allocation must already be durably recorded before the second call.
        assert launcher.read_json(folder / "cpu.receipt.json")["job_id"] == "6001"
        assert (folder / "gpu.intent.json").exists()
        return subprocess.CompletedProcess(command, 1, "", "temporary submission failure")
    monkeypatch.setenv("SBATCH_QOS", "unrecorded-override")
    monkeypatch.setattr(launcher.subprocess, "run", fake_sbatch)
    with pytest.raises(RuntimeError, match="outcome requires review"):
        launcher.submit(args, exp.root, exp.output, exp.ledger, tasks,
                        exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})

    assert len(calls) == 2
    folder = next(exp.ledger.iterdir())
    assert launcher.read_json(folder / "cpu.receipt.json")["status"] == "submitted"
    assert launcher.read_json(folder / "gpu.receipt.json")["status"] == "unconfirmed"
    assert (folder / "manifest.json").is_file()
    assert "--array=0-139%4" in calls[0] and "--array=0-105%2" in calls[1]
    assert all("--export=ALL" in command and "--no-requeue" in command for command in calls)
    for group in ("cpu", "gpu"):
        # Syntax-check only: bash -n cannot execute the frozen worker or training.
        checked = real_run(["bash", "-n", str(folder / f"{group}.sbatch")],
                           capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr


def test_worker_source_change_stops_before_download_or_training(experiment, monkeypatch):
    exp = experiment
    manifest = exp.root / "queued-manifest.json"
    launcher.write_new(manifest, {
        "worker_sha256": launcher.digest(launcher.__file__),
        "root": str(exp.root), "output_root": str(exp.output),
        "source_sha256": exp.source, "tasks": {"cpu": [task()]},
    })
    args = SimpleNamespace(worker=str(manifest), manifest_sha256=launcher.digest(manifest), group="cpu")
    exp.source_file.write_text("# Changed after submission.\n")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    def unexpected_call(*args, **kwargs):
        pytest.fail("Worker must stop before external processes when source changed")
    monkeypatch.setattr(launcher.subprocess, "run", unexpected_call)

    with pytest.raises(RuntimeError, match="src/configs changed"):
        launcher.worker(args)


def test_worker_dispatches_selected_index_with_recorded_environment(experiment, monkeypatch, capsys):
    exp = experiment
    selected = task("BRITS", 2)
    args = worker_arguments(exp, [task("TIMESNET", 0), selected])
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "4")
    for name in ("IMTS_MODEL", "IMTS_DATASET", "IMTS_SEED", "IMTS_OUTPUT_DIR", "IMTS_MAX_DENSE_GIB",
                 "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
        monkeypatch.setenv(name, "unrelated-shell-value")
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        assert kwargs["cwd"] == exp.root
        environment = kwargs["env"]
        assert environment["IMTS_MODEL"] == "BRITS"
        assert environment["IMTS_DATASET"] == "JapaneseVowels"
        assert environment["IMTS_SEED"] == "2"
        assert environment["IMTS_OUTPUT_DIR"] == str(exp.output)
        assert environment["IMTS_MAX_DENSE_GIB"] == "8"
        assert all(environment[name] == "4" for name in
                   ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"))
        if len(calls) == 1:
            assert command[:3] == [str(exp.root / ".venv/bin/python"), "-c", launcher.CACHE_CODE]
            assert json.loads(command[3]) == exp.datasets["JapaneseVowels"]
            assert command[4] == exp.revision and kwargs["check"] is True
            return subprocess.CompletedProcess(command, 0)
        assert command == ["bash", str(exp.root / "slurm/benchmark_gpu.sbatch")]
        assert "--device cuda" in Path(command[1]).read_text()
        return subprocess.CompletedProcess(command, 17)
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)

    assert launcher.worker(args) == 17
    assert len(calls) == 2
    announced = json.loads(capsys.readouterr().out.splitlines()[0])
    assert announced["task"] == selected and announced["slurm_array_task_id"] == 1


def test_worker_rechecks_success_and_skips_cache_and_training(experiment, monkeypatch, capsys):
    exp = experiment
    selected = task("BRITS", 0)
    args = worker_arguments(exp, [selected])
    # A compatible result arrives after this worker's manifest was created.
    record(exp, selected)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    def unexpected_call(*args, **kwargs):
        pytest.fail("An already successful task must not download data or train again")
    monkeypatch.setattr(launcher.subprocess, "run", unexpected_call)

    assert launcher.worker(args) == 0
    assert "Already successful; no duplicate fit" in capsys.readouterr().out


def test_task_lock_excludes_a_second_worker_and_remains_reusable(tmp_path):
    lock = tmp_path / "task.lock"
    with launcher.file_lock(lock):
        with pytest.raises(RuntimeError, match="Another submission/worker"):
            with launcher.file_lock(lock):
                pytest.fail("Second worker unexpectedly acquired the same task lock")
    assert lock.is_file()
    with launcher.file_lock(lock):
        pass
