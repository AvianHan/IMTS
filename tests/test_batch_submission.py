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
QOS_REFUSAL = {
    "job_id": "",
    "manifest_sha256": "2e9e717bfcd2d6e0d412ffd7087643e0c5c9663f0692183a13e23776ed89b86f",
    "returncode": 1,
    "status": "unconfirmed",
    "stderr": "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
              "sbatch: error: Batch job submission failed: Job violates accounting/QOS policy "
              "(job submit limit, user's size and/or time limits)\n",
    "stdout": "",
}


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
    monkeypatch.setattr(launcher, "SHARED_SUBMISSION_LOCK", tmp_path / "shared/submission.lock")
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


def submit_arguments(**updates):
    return SimpleNamespace(**{
        "max_dense_gib": 8, "cpu_concurrency": 4, "gpu_concurrency": 2,
        "cpu_batch_size": 4, "gpu_batch_size": 2,
        "cpu_time": "02:00:00", "gpu_time": "01:00:00", **updates,
    })


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
    assert "Next batch: 6 = CPU 4 + GPU 2" in output
    assert "Deferred: 240" in output
    assert "DRY RUN: no jobs submitted" in output
    assert snapshot(exp.root) == before


def test_total_cap_cli_dry_run_stays_read_only_and_does_not_query_queue(experiment, monkeypatch, capsys):
    exp = experiment
    monkeypatch.setattr(sys, "argv", [
        "submit_benchmarks.py", "--cpu-batch-size", "56", "--gpu-batch-size", "56",
        "--total-batch-size", "56", "--max-user-jobs", "60",
        "--cpu-concurrency", "56", "--gpu-concurrency", "56",
    ])
    before = snapshot(exp.root)
    def unexpected(*args, **kwargs):
        pytest.fail("A quota-cap dry run must not query Slurm or submit anything")
    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    assert launcher.main() == 0
    assert "Next batch: 56 = CPU 28 + GPU 28" in capsys.readouterr().out
    assert snapshot(exp.root) == before
    assert not launcher.SHARED_SUBMISSION_LOCK.exists()


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


def test_exact_legacy_qos_refusal_is_ignored_without_mutating_ledger(experiment, monkeypatch):
    exp = experiment
    folder = exp.ledger / "20261013T093211Z-bc81dd88"
    folder.mkdir(parents=True)
    launcher.write_new(folder / "manifest.json", {
        "output_root": str(exp.output), "tasks": {"cpu": [task()], "gpu": []},
    })
    launcher.write_new(folder / "cpu.intent.json", {
        "job_name": "imts-cpu-20261013T093211Z-bc81dd88",
        "command": ["sbatch", "--parsable", "--array=0-139%4"],
        "manifest_sha256": QOS_REFUSAL["manifest_sha256"],
    })
    launcher.write_new(folder / "cpu.receipt.json", QOS_REFUSAL)
    before = snapshot(exp.ledger)

    def unexpected_call(*args, **kwargs):
        pytest.fail("An explicit refusal must not be treated as a submitted Slurm job")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected_call)
    monkeypatch.setattr(launcher, "array_is_terminal", unexpected_call)
    assert launcher.known_submit_rejection(QOS_REFUSAL) is True
    assert launcher.check_previous_batches(exp.ledger, exp.output, [task()]) == set()
    assert snapshot(exp.ledger) == before


@pytest.mark.parametrize("updates", [
    {"returncode": 0},
    {"returncode": None},
    {"status": "submitted"},
    {"job_id": "6001"},
    {"stdout": "6001\n"},
    {"stdout": "Submitted batch job 6001\n"},
    {"stdout": "unrecognized scheduler response\n"},
    {"stderr": "connection closed before receiving response"},
    {"stderr": "temporary submission failure"},
    {"stderr": ""},
])
def test_ambiguous_receipt_cannot_be_classified_as_a_known_refusal(updates):
    assert launcher.known_submit_rejection({**QOS_REFUSAL, **updates}) is False


@pytest.mark.parametrize("updates", [
    {"returncode": 0}, {"job_id": "6001"}, {"stdout": "6001\n"},
])
def test_qos_text_does_not_override_ambiguous_submission_evidence(experiment, monkeypatch, updates):
    exp = experiment
    folder = exp.ledger / "ambiguous-refusal"
    folder.mkdir(parents=True)
    launcher.write_new(folder / "manifest.json", {
        "output_root": str(exp.output), "tasks": {"cpu": [task()]},
    })
    launcher.write_new(folder / "cpu.intent.json", {"job_name": "ambiguous-refusal"})
    launcher.write_new(folder / "cpu.receipt.json", {**QOS_REFUSAL, **updates})
    monkeypatch.setattr(launcher, "array_is_terminal", lambda *_: True)
    with pytest.raises(RuntimeError, match="Unconfirmed submission"):
        launcher.check_previous_batches(exp.ledger, exp.output, [task()])


def test_batch_selection_preserves_per_device_order_and_caps_total_submission():
    tasks = [selected for seed in range(6)
             for selected in (task("BRITS", seed), task("ROCKET", seed))]
    before = list(tasks)
    selected = launcher.select_batch(tasks, submit_arguments())
    assert selected == [entry for entry in tasks
                        if entry["seed"] < (2 if entry["device"] == "cuda" else 4)]
    assert len(selected) == 6
    assert tasks == before
    # Batch size bounds queued + running tasks even if concurrency is larger.
    selected = launcher.select_batch(tasks, submit_arguments(
        cpu_batch_size=2, gpu_batch_size=1, cpu_concurrency=8, gpu_concurrency=4))
    assert selected == [task("BRITS", 0), task("ROCKET", 0), task("ROCKET", 1)]


def test_total_cap_interleaves_pools_and_preserves_their_order():
    tasks = [task("ROCKET", seed) for seed in range(40)] + [task("BRITS", seed) for seed in range(40)]
    selected = launcher.select_batch(tasks, submit_arguments(
        cpu_batch_size=56, gpu_batch_size=56, total_batch_size=56))
    assert len(selected) == 56
    assert selected == [entry for seed in range(28) for entry in (task("ROCKET", seed), task("BRITS", seed))]


@pytest.mark.parametrize("cpu_count,gpu_count,cpu_cap,gpu_cap,expected", [
    (3, 80, 56, 56, (3, 53)),
    (80, 0, 56, 56, (56, 0)),
    (0, 80, 56, 56, (0, 56)),
    (80, 80, 4, 56, (4, 52)),
    (80, 80, 4, 2, (4, 2)),
])
def test_total_cap_fills_available_pool_but_honors_per_pool_caps(cpu_count, gpu_count, cpu_cap, gpu_cap, expected):
    tasks = [task("ROCKET", seed) for seed in range(cpu_count)] + [task("BRITS", seed) for seed in range(gpu_count)]
    selected = launcher.select_batch(tasks, submit_arguments(
        cpu_batch_size=cpu_cap, gpu_batch_size=gpu_cap, total_batch_size=56))
    cpu = [entry for entry in selected if entry["device"] == "cpu"]
    gpu = [entry for entry in selected if entry["device"] == "cuda"]
    assert (len(cpu), len(gpu)) == expected
    assert cpu == [task("ROCKET", seed) for seed in range(expected[0])]
    assert gpu == [task("BRITS", seed) for seed in range(expected[1])]


def test_expanded_user_queue_counts_unique_array_elements(monkeypatch):
    def fake_squeue(command, **kwargs):
        assert command == ["squeue", "-h", "-r", "-u", launcher.getpass.getuser(), "-o", "%i"]
        return subprocess.CompletedProcess(command, 0, "123_0\n123_1\n456\n123_0\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", fake_squeue)
    assert launcher.active_user_jobs() == {"123_0", "123_1", "456"}


@pytest.mark.parametrize("returncode,stdout,stderr", [
    (1, "", "controller unavailable"),
    (0, "123_[0-55]\n", ""),
    (0, "JOBID\n123\n", ""),
    (0, "123\n\n", ""),
    (0, "unexpected response\n", ""),
])
def test_quota_query_failure_or_malformed_output_blocks_all_submission(experiment, monkeypatch,
                                                                      returncode, stdout, stderr):
    exp, calls = experiment, []
    def fake_squeue(command, **kwargs):
        calls.append(command)
        assert command[0] == "squeue", "A failed quota check must not reach sbatch"
        return subprocess.CompletedProcess(command, returncode, stdout, stderr)
    monkeypatch.setattr(launcher.subprocess, "run", fake_squeue)
    with pytest.raises(RuntimeError, match="Cannot check user job quota"):
        launcher.submit(submit_arguments(max_user_jobs=60), exp.root, exp.output, exp.ledger,
                        [task(), task("BRITS")], exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})
    assert len(calls) == 1
    assert not exp.ledger.exists()


def test_entire_wave_must_fit_before_any_sbatch(experiment, monkeypatch):
    exp = experiment
    def fake_squeue(command, **kwargs):
        assert command[0] == "squeue", "Must reject the whole wave, not submit CPU then discover GPU overflow"
        return subprocess.CompletedProcess(command, 0, "\n".join(str(i) for i in range(59)) + "\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", fake_squeue)
    with pytest.raises(RuntimeError, match="59 active/reserved.*requires 2"):
        launcher.submit(submit_arguments(max_user_jobs=60), exp.root, exp.output, exp.ledger,
                        [task(), task("BRITS")], exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})
    assert not exp.ledger.exists()


def test_capped_submission_preserves_concurrency_and_rechecks_each_group(experiment, monkeypatch):
    exp, calls = experiment, []
    selected = [task("ROCKET", seed) for seed in range(60)] + [task("BRITS", seed) for seed in range(60)]
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            # Four method jobs occupy the reserved places. An accepted CPU
            # array is intentionally absent, simulating brief queue lag.
            return subprocess.CompletedProcess(command, 0, "10\n11\n12\n13\n", "")
        assert command[0] == "sbatch"
        return subprocess.CompletedProcess(command, 0, "6001\n" if command[-1].endswith("cpu.sbatch") else "6002\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    folder, jobs = launcher.submit(submit_arguments(
        max_user_jobs=60, cpu_batch_size=56, gpu_batch_size=56, total_batch_size=56,
        cpu_concurrency=9, gpu_concurrency=7), exp.root, exp.output, exp.ledger,
        selected, exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})
    assert jobs == [("6001", 28), ("6002", 28)]
    assert [command[0] for command in calls] == ["squeue", "squeue", "sbatch", "squeue", "sbatch"]
    assert "--array=0-27%9" in calls[2]
    assert "--array=0-27%7" in calls[4]
    assert len(launcher.read_json(folder / "manifest.json")["tasks"]["cpu"]) == 28


def test_queue_race_after_first_group_preserves_receipt_and_never_retries(experiment, monkeypatch):
    exp, calls = experiment, []
    def fake_run(command, **kwargs):
        calls.append(command)
        if command[0] == "squeue":
            # An unrelated non-cooperating sbatch consumes places after CPU
            # submission; our CPU array has not appeared in squeue yet.
            visible = "\n".join(str(i) for i in range(59)) + "\n" if len(calls) > 3 else ""
            return subprocess.CompletedProcess(command, 0, visible, "")
        assert len(calls) == 3 and command[0] == "sbatch"
        return subprocess.CompletedProcess(command, 0, "6001\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="Earlier confirmed arrays: 6001"):
        launcher.submit(submit_arguments(max_user_jobs=60), exp.root, exp.output, exp.ledger,
                        [task(), task("BRITS")], exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})
    folder = next(exp.ledger.iterdir())
    assert launcher.read_json(folder / "cpu.receipt.json")["status"] == "submitted"
    assert not (folder / "gpu.intent.json").exists()
    assert len(calls) == 4


def test_shared_submission_lock_blocks_another_launcher(experiment, monkeypatch):
    exp = experiment
    def unexpected(*args, **kwargs):
        pytest.fail("Shared lock must exclude all queue queries and submissions")
    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    with launcher.file_lock(launcher.SHARED_SUBMISSION_LOCK):
        with pytest.raises(RuntimeError, match="Another submission/worker"):
            launcher.submit(submit_arguments(max_user_jobs=60), exp.root, exp.output, exp.ledger,
                            [task()], exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})
    assert not exp.ledger.exists()


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


def test_watch_waits_for_transient_accounting_lag_before_proceeding(monkeypatch):
    calls, sleeps = [], []

    def delayed_accounting(job_id, task_count):
        calls.append((job_id, task_count))
        if len(calls) <= 3:
            raise launcher.AccountingPending("Per-task accounting not available yet")
        return True

    monkeypatch.setattr(launcher, "array_is_terminal", delayed_accounting)
    monkeypatch.setattr(launcher.time, "sleep", sleeps.append)
    launcher.wait_for_arrays([("9001", 4)], 30)
    assert calls == [("9001", 4)] * 4
    assert sleeps == [30] * 3


def test_watch_missing_accounting_has_bounded_wait_and_query_errors_stop_immediately(monkeypatch):
    calls, sleeps = [], []

    def missing_accounting(job_id, task_count):
        calls.append((job_id, task_count))
        raise launcher.AccountingPending("Per-task accounting remains incomplete")

    monkeypatch.setattr(launcher, "array_is_terminal", missing_accounting)
    monkeypatch.setattr(launcher.time, "sleep", sleeps.append)
    with pytest.raises(launcher.AccountingPending):
        launcher.wait_for_arrays([("9002", 2)], 30)
    assert calls == [("9002", 2)] * 5
    assert sleeps == [30] * 4

    def query_failure(*args):
        raise RuntimeError("Cannot check live Slurm jobs: controller unavailable")

    monkeypatch.setattr(launcher, "array_is_terminal", query_failure)
    sleeps.clear()
    with pytest.raises(RuntimeError, match="controller unavailable"):
        launcher.wait_for_arrays([("9003", 2)], 30)
    assert sleeps == []


def test_partial_submission_keeps_receipts_and_does_not_retry(experiment, monkeypatch):
    exp = experiment
    args = submit_arguments()
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
    with pytest.raises(RuntimeError, match="unconfirmed outcome"):
        launcher.submit(args, exp.root, exp.output, exp.ledger, tasks,
                        exp.source, ENVIRONMENT, exp.datasets, exp.revision, {})

    assert len(calls) == 2
    folder = next(exp.ledger.iterdir())
    assert launcher.read_json(folder / "cpu.receipt.json")["status"] == "submitted"
    assert launcher.read_json(folder / "gpu.receipt.json")["status"] == "unconfirmed"
    assert (folder / "manifest.json").is_file()
    assert "--array=0-3%4" in calls[0] and "--array=0-1%2" in calls[1]
    manifest = launcher.read_json(folder / "manifest.json")
    assert len(manifest["tasks"]["cpu"]) == 4
    assert len(manifest["tasks"]["gpu"]) == 2
    assert all("--export=ALL" in command and "--no-requeue" in command for command in calls)
    for group in ("cpu", "gpu"):
        # Syntax-check only: bash -n cannot execute the frozen worker or training.
        checked = real_run(["bash", "-n", str(folder / f"{group}.sbatch")],
                           capture_output=True, text=True)
        assert checked.returncode == 0, checked.stderr


@pytest.mark.parametrize("overrides,expected", [
    ({}, {"cpu": "32G", "gpu": "32G"}),
    ({"cpu_mem": "128g", "gpu_mem": "96G"}, {"cpu": "128G", "gpu": "96G"}),
    ({"gpu_mem": "65536"}, {"cpu": "32G", "gpu": "65536"}),
])
def test_memory_requests_match_manifest_and_preserve_templates(experiment, monkeypatch, overrides, expected):
    exp = experiment
    calls = []
    monkeypatch.setenv("SBATCH_MEM_PER_NODE", "999G")

    def submit(command, **kwargs):
        assert "SBATCH_MEM_PER_NODE" not in kwargs["env"]
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, f"{6000 + len(calls)}\n", "")

    monkeypatch.setattr(launcher.subprocess, "run", submit)
    folder, _ = launcher.submit(submit_arguments(**overrides), exp.root, exp.output, exp.ledger,
                                [task(), task("BRITS")], exp.source, ENVIRONMENT,
                                exp.datasets, exp.revision, {})
    manifest = launcher.read_json(folder / "manifest.json")
    for group, command in zip(("cpu", "gpu"), calls):
        requested = overrides.get(f"{group}_mem")
        resource = manifest["resources"][group]
        assert resource["mem"] == expected[group]
        assert resource["mem_source"] == ("command line" if requested else "template")
        actual_flags = [value for value in command if value.startswith("--mem=")]
        assert actual_flags == ([f"--mem={expected[group]}"] if requested else [])
        assert launcher.read_json(folder / f"{group}.intent.json")["command"] == command
        assert (folder / f"{group}.template.sbatch").read_bytes() == (
            exp.root / "slurm" / f"benchmark_{group}.sbatch"
        ).read_bytes()
        assert "#SBATCH --mem=32G\n" in (folder / f"{group}.sbatch").read_text()


@pytest.mark.parametrize("option", ["--cpu-mem", "--gpu-mem"])
@pytest.mark.parametrize("value", ["0", "0G", "-1G", "1.5G", "128GB", "128G --qos=other", "", "unlimited"])
def test_invalid_memory_is_rejected_before_any_query_or_submission(monkeypatch, option, value):
    monkeypatch.setattr(sys, "argv", ["submit_benchmarks.py", f"{option}={value}", "--submit"])

    def unexpected(*args, **kwargs):
        pytest.fail("Invalid memory reached an external command")

    monkeypatch.setattr(launcher.subprocess, "run", unexpected)
    with pytest.raises(SystemExit) as error:
        launcher.main()
    assert error.value.code == 2


def test_new_qos_refusal_records_rejection_and_stops_before_gpu(experiment, monkeypatch):
    exp = experiment
    calls = []

    def refused(command, **kwargs):
        calls.append(command)
        assert command[0] == "sbatch"
        return subprocess.CompletedProcess(command, 1, "", QOS_REFUSAL["stderr"])

    monkeypatch.setattr(launcher.subprocess, "run", refused)
    with pytest.raises(RuntimeError, match="QOS|QoS|rejected|limit"):
        launcher.submit(submit_arguments(), exp.root, exp.output, exp.ledger,
                        [task(), task("BRITS")], exp.source, ENVIRONMENT,
                        exp.datasets, exp.revision, {})

    assert len(calls) == 1
    folder = next(exp.ledger.iterdir())
    receipt = launcher.read_json(folder / "cpu.receipt.json")
    assert receipt["status"] == "rejected"
    assert receipt["stderr"] == QOS_REFUSAL["stderr"]
    assert not (folder / "gpu.intent.json").exists()
    before = snapshot(folder)
    assert launcher.check_previous_batches(exp.ledger, exp.output, [task()]) == set()
    assert snapshot(folder) == before


def test_watch_runs_bounded_waves_without_repeating_failed_fits(experiment, monkeypatch, capsys):
    exp = experiment
    monkeypatch.setattr(sys, "argv", [
        "submit_benchmarks.py", "--datasets", "JapaneseVowels", "--submit", "--watch",
        "--poll-seconds", "30",
    ])
    submissions, polls, sleeps = [], {}, []

    def fake_sbatch(command, **kwargs):
        assert command[0] == "sbatch", "All real Slurm commands must be mocked"
        folder = Path(command[-1]).parent
        group = Path(command[-1]).stem
        manifest = launcher.read_json(folder / "manifest.json")
        selected = manifest["tasks"][group]
        assert 0 < len(selected) <= (4 if group == "cpu" else 2)
        # Each previous wave must be confirmed terminal before any new wave.
        earlier = [(job_id, previous_folder) for job_id, previous_folder, _, _ in submissions
                   if previous_folder != folder]
        assert all(polls.get(job_id, 0) >= 2 for job_id, _ in earlier)
        job_id = str(7000 + len(submissions))
        submissions.append((job_id, folder, group, selected))
        return subprocess.CompletedProcess(command, 0, job_id + "\n", "")

    def terminal(job_id, task_count):
        assert any(item[0] == job_id and len(item[3]) == task_count for item in submissions)
        polls[job_id] = polls.get(job_id, 0) + 1
        return polls[job_id] >= 2

    monkeypatch.setattr(launcher.subprocess, "run", fake_sbatch)
    monkeypatch.setattr(launcher, "array_is_terminal", terminal)
    monkeypatch.setattr(launcher.time, "sleep", sleeps.append)

    # No result files are produced: terminal training failures must not loop forever.
    assert launcher.main() == 1

    models = launcher.read_json(exp.root / "configs/baselines.json")["models"]
    expected = {launcher.task_key(task(model, seed)) for model, settings in models.items()
                for seed in settings["seeds"]}
    submitted_keys = [launcher.task_key(entry)
                      for _, _, _, selected in submissions for entry in selected]
    assert len(submitted_keys) == len(set(submitted_keys)) == len(expected) == 14
    assert set(submitted_keys) == expected
    assert len({folder for _, folder, _, _ in submissions}) == 3
    assert sleeps and all(0 < seconds <= 60 for seconds in sleeps)
    assert all(polls.get(job_id, 0) >= 2 for job_id, _, _, _ in submissions)
    assert len(list(exp.ledger.glob("*/*.receipt.json"))) == len(submissions)
    assert "14" in capsys.readouterr().out


def test_watch_partial_submission_stops_without_sleep_or_next_wave(experiment, monkeypatch):
    exp = experiment
    monkeypatch.setattr(sys, "argv", [
        "submit_benchmarks.py", "--datasets", "JapaneseVowels", "--submit", "--watch",
    ])
    calls = []

    def fake_sbatch(command, **kwargs):
        assert command[0] == "sbatch"
        calls.append(command)
        if len(calls) == 1:
            return subprocess.CompletedProcess(command, 0, "8001\n", "")
        return subprocess.CompletedProcess(command, 1, "", "controller connection lost")

    def unexpected_wait(*args, **kwargs):
        pytest.fail("Partial submission must stop for review before waiting or trying a new wave")

    monkeypatch.setattr(launcher.subprocess, "run", fake_sbatch)
    monkeypatch.setattr(launcher, "array_is_terminal", unexpected_wait)
    monkeypatch.setattr(launcher.time, "sleep", unexpected_wait)
    with pytest.raises(RuntimeError, match="unconfirmed outcome"):
        launcher.main()

    assert len(calls) == 2
    folders = list(exp.ledger.iterdir())
    assert len(folders) == 1
    assert launcher.read_json(folders[0] / "cpu.receipt.json")["job_id"] == "8001"
    assert launcher.read_json(folders[0] / "gpu.receipt.json")["status"] == "unconfirmed"


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
