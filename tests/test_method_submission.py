"""Offline quota, rolling refill, provenance and stage-gate tests; never call Slurm."""
import argparse
import contextlib
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/submit_method_pilot.py"
SPEC = importlib.util.spec_from_file_location("method_submit", SCRIPT)
submit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(submit)


def result(stdout="", stderr="", returncode=0):
    return subprocess.CompletedProcess([], returncode, stdout, stderr)


@pytest.fixture
def pilot(tmp_path, monkeypatch):
    root = tmp_path / "method"
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    (root / "src").mkdir()
    (root / "src/model.py").write_text("VERSION = 1\n")
    (root / "slurm").mkdir()
    (root / "slurm/method_gpu.sbatch").write_text("#!/bin/bash\nset -euo pipefail\n")
    args = argparse.Namespace(output_dir="results/method-pilot", time="03:00:00", retry_failed=False,
                              group="core", max_user_jobs=60, staged=False, evaluate_test=False,
                              watch=False, poll_seconds=30, reserve_other_jobs=0, baseline_root=None)
    tasks = [{"dataset": "JapaneseVowels", "variant": name, "seed": 0} for name in
             ("core_none", "core_outer", "core_inner", "core_both")]
    monkeypatch.setattr(submit, "submission_lock", contextlib.nullcontext)
    monkeypatch.setattr(submit, "queue_ids", lambda: set())
    monkeypatch.setattr(submit, "qos_limits", lambda: {"effective_submit_cap": 60})
    monkeypatch.setattr(submit, "inventory", lambda _: {"python": "3.11", "packages": {}})
    return root, args, tasks


def snapshot(root):
    return {"source_sha256": submit.source_digest(root), "environment": {"python": "3.11", "packages": {}}}


def make_ledger(root, tasks, status="submitted", job="123", intent_only=False, mode="validation_only"):
    folder = root / "results/accounting/method-pilots/prior"
    folder.mkdir(parents=True)
    submit.write_new(folder / "manifest.json", {"tasks": tasks, "mode": mode})
    digest = submit.sha256(folder / "manifest.json")
    submit.write_new(folder / "intent.json", {"manifest_sha256": digest})
    if not intent_only:
        submit.write_new(folder / "receipt.json", {"manifest_sha256": digest, "status": status, "job_id": job})
    return folder.parent


def test_atomic_ledger_never_replaces_existing_file(tmp_path):
    path = tmp_path / "record.json"
    submit.write_new(path, {"first": True})
    with pytest.raises(FileExistsError):
        submit.write_new(path, {"second": True})
    assert json.loads(path.read_text()) == {"first": True}
    assert not list(tmp_path.glob("*.tmp-*"))


@pytest.mark.parametrize("response", [result(stderr="controller unavailable", returncode=1), result("100_[0-53]\n")])
def test_queue_failure_or_compressed_arrays_fail_closed(monkeypatch, response):
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: response)
    with pytest.raises(RuntimeError):
        submit.queue_ids()


def test_queue_counts_each_array_task_once(monkeypatch):
    def run(command, **kwargs):
        assert "-r" in command
        return result("100_0\n100_1\n100_1\n101\n")
    monkeypatch.setattr(submit.subprocess, "run", run)
    assert submit.queue_ids() == {"100_0", "100_1", "101"}


def test_capacity_counts_pending_running_and_invisible_accepted_tasks():
    limits = {"effective_submit_cap": 60}
    submit.check_capacity({str(i) for i in range(56)}, 4, limits)
    with pytest.raises(RuntimeError, match="57 existing"):
        submit.check_capacity({str(i) for i in range(57)}, 4, limits)
    assert submit.available_slots({str(i) for i in range(57)}, {"new_0", "new_1"}, 2, limits) == 1


def test_baseline_reservation_is_not_subtracted_twice():
    ids = {str(i) for i in range(59)}  # 57 method + 2 baseline
    assert submit.available_slots(ids, set(), 57, {"effective_submit_cap": 60}, reserve=2, reserved_active=2) == 1
    assert submit.available_slots(set(), set(), 0, {"effective_submit_cap": 60}, reserve=2) == 58
    assert submit.available_slots(set(), set(), 0, {"effective_submit_cap": 60}, reserve=0) == 60


def test_lower_live_qos_limit_is_respected(monkeypatch):
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: result("highpriority|6||||||\n"))
    limits = submit.qos_limits()
    assert limits["effective_submit_cap"] == 6
    with pytest.raises(RuntimeError, match="> 6"):
        submit.check_capacity({"1", "2", "3"}, 4, limits)


@pytest.mark.parametrize("response", [result(stderr="denied", returncode=1), result(""),
                                       result("highpriority|unexpected||||||\n")])
def test_missing_or_invalid_qos_fails_closed(monkeypatch, response):
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: response)
    with pytest.raises(RuntimeError):
        submit.qos_limits()


def test_intent_without_receipt_blocks_retry(pilot):
    root, _, tasks = pilot
    ledger = make_ledger(root, tasks, intent_only=True)
    with pytest.raises(RuntimeError, match="Ambiguous"):
        submit.ledger_status(ledger, set())


def test_unconfirmed_receipt_blocks_even_explicit_retry(pilot):
    root, _, tasks = pilot
    ledger = make_ledger(root, tasks, status="unconfirmed", job="")
    with pytest.raises(RuntimeError, match="Unconfirmed"):
        submit.check_previous(ledger, set(), tasks, retry_failed=True)


def test_active_array_blocks_duplicate_only_not_unrelated_task(pilot, monkeypatch):
    root, _, tasks = pilot
    ledger = make_ledger(root, tasks)
    monkeypatch.setattr(submit, "accounting_rows", lambda _: {})
    with pytest.raises(RuntimeError, match="remains active"):
        submit.check_previous(ledger, {"123_0"}, tasks)
    submit.check_previous(ledger, {"123_0"}, [{"dataset": "JapaneseVowels", "variant": "other", "seed": 0}])
    _, invisible, active = submit.ledger_status(ledger, {"123_0"})
    assert active == 4 and invisible == {"123_1", "123_2", "123_3"}


def test_per_task_accounting_releases_completed_slots_in_active_array(pilot, monkeypatch):
    root, _, tasks = pilot
    ledger = make_ledger(root, tasks)
    monkeypatch.setattr(submit, "accounting_rows", lambda _: {"123_0": ("COMPLETED", "0:0"), "123_1": ("TIMEOUT", "0:0")})
    entries, invisible, active = submit.ledger_status(ledger, {"123_2", "123_3"})
    assert active == 2 and not invisible
    assert not entries[submit.execution_key(tasks[0], "validation_only")][0]["active"]


def test_incomplete_accounting_is_not_a_finished_array(monkeypatch):
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: result("123_0|COMPLETED|0:0|\n"))
    with pytest.raises(RuntimeError, match="incomplete"):
        submit.accounting_states("123", 4)


def test_terminal_failure_requires_intentional_retry(pilot, monkeypatch):
    root, _, tasks = pilot
    ledger = make_ledger(root, tasks)
    monkeypatch.setattr(submit, "accounting_rows", lambda _: {f"123_{i}": ("TIMEOUT", "0:0") for i in range(4)})
    with pytest.raises(RuntimeError, match="retry-failed"):
        submit.check_previous(ledger, set(), tasks)
    submit.check_previous(ledger, set(), tasks, retry_failed=True)


def test_submit_uses_available_three_slots_instead_of_rejecting_four_task_plan(pilot, monkeypatch):
    root, args, tasks = pilot
    commands = []
    monkeypatch.setattr(submit, "queue_ids", lambda: {str(i) for i in range(57)})
    monkeypatch.setattr(submit, "successful_results", lambda *a: {})
    def run(command, **kwargs):
        commands.append(command)
        assert "--array=0-2%3" in command
        assert "--gres=gpu:1" in command and "--time=03:00:00" in command
        assert not any(key.startswith("SBATCH_") for key in kwargs["env"])
        return result("987;orion\n")
    monkeypatch.setenv("SBATCH_GRES", "gpu:8")
    monkeypatch.setattr(submit.subprocess, "run", run)
    submit.submit(args, root, tasks)
    assert len(commands) == 1
    folder = next((root / "results/accounting/method-pilots").glob("*/manifest.json")).parent
    manifest = json.loads((folder / "manifest.json").read_text())
    receipt = json.loads((folder / "receipt.json").read_text())
    assert manifest["tasks"] == tasks[:3]
    assert manifest["mode"] == "validation_only"
    assert manifest["resources"]["global_task_cap"] == 60
    assert receipt["job_id"] == "987"
    assert receipt["manifest_sha256"] == submit.sha256(folder / "manifest.json")


def test_sixty_task_array_has_no_four_task_cap(pilot, monkeypatch):
    root, args, tasks = pilot
    many = [dict(tasks[0], seed=i) for i in range(60)]
    monkeypatch.setattr(submit.subprocess, "run", lambda command, **kw: result("987") if "--array=0-59%60" in command else pytest.fail(str(command)))
    submit.submit_batch(args, root, many, "test", set(), {"effective_submit_cap": 60}, snapshot(root))


@pytest.mark.parametrize("response,expected", [
    (result("unexpected scheduler response"), "unconfirmed"),
    (result("", "sbatch: error: QOSMaxSubmitJobPerUserLimit\nsbatch: error: Batch job submission failed: policy", 1), "rejected")])
def test_sbatch_ambiguous_or_known_rejection_is_recorded_without_retry(pilot, monkeypatch, response, expected):
    root, args, tasks = pilot
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return response
    monkeypatch.setattr(submit.subprocess, "run", run)
    with pytest.raises(RuntimeError, match=expected):
        submit.submit_batch(args, root, tasks, "validation_only", set(), {"effective_submit_cap": 60}, snapshot(root))
    assert len(calls) == 1
    receipt = next((root / "results/accounting/method-pilots").glob("*/receipt.json"))
    assert json.loads(receipt.read_text())["status"] == expected


def test_preflight_does_not_submit_or_write_ledger(pilot, monkeypatch):
    root, args, tasks = pilot
    monkeypatch.setattr(submit.subprocess, "run", lambda command, **kw: result() if "--test-only" in command and "--wrap=true" in command else pytest.fail(str(command)))
    submit.test_only(args, root, tasks)
    assert not (root / "results/accounting/method-pilots").exists()


def test_rolling_refill_does_not_wait_for_whole_array(pilot, monkeypatch):
    root, args, tasks = pilot
    args.watch = True
    args.reserve_other_jobs = 2
    monkeypatch.setattr(submit, "reserved_jobs_active", lambda *a: 2)
    tasks = [dict(tasks[0], seed=i) for i in range(70)]
    submitted, complete, chunks, waits = [], set(), [], []
    def states(*_):
        entries = {submit.execution_key(task, "validation_only"): [{"active": task["seed"] not in complete}]
                   for task in submitted}
        return entries, set(), len(submitted) - len(complete)
    def successes(*_):
        return {submit.execution_key(task, "validation_only"): Path("result.json") for task in submitted if task["seed"] in complete}
    def batch(_args, _root, chunk, mode, ids, limits, snap):
        assert len(ids) + len(chunk) <= 60
        assert len(submitted) - len(complete) + len(chunk) <= 58
        chunks.append(len(chunk))
        submitted.extend(chunk)
    def sleep(seconds):
        waits.append(seconds)
        if len(waits) == 1:
            complete.update(task["seed"] for task in submitted[:3])
        else:
            complete.update(task["seed"] for task in submitted)
    monkeypatch.setattr(submit, "queue_ids", lambda: {"baseline_0", "baseline_1"} | {f"m_{task['seed']}" for task in submitted if task["seed"] not in complete})
    monkeypatch.setattr(submit, "ledger_status", states)
    monkeypatch.setattr(submit, "successful_results", successes)
    monkeypatch.setattr(submit, "submit_batch", batch)
    monkeypatch.setattr(submit.time, "sleep", sleep)
    outcome = submit.phase(args, root, tasks, "validation_only", snapshot(root), {"effective_submit_cap": 60})
    assert outcome["complete"]
    assert chunks == [58, 3, 9]
    assert len(submitted) == len({task["seed"] for task in submitted}) == 70


def test_full_queue_waits_then_submits_without_failing(pilot, monkeypatch):
    root, args, tasks = pilot
    args.watch = True
    state = {"poll": 0, "submitted": False}
    monkeypatch.setattr(submit, "queue_ids", lambda: {str(i) for i in range(60)} if state["poll"] == 0 else set())
    monkeypatch.setattr(submit, "ledger_status", lambda *a: ({}, set(), 0))
    monkeypatch.setattr(submit, "successful_results", lambda *a: {submit.execution_key(t, "validation_only"): Path("ok") for t in tasks} if state["submitted"] else {})
    def batch(*_):
        assert state["poll"] > 0
        state["submitted"] = True
    monkeypatch.setattr(submit, "submit_batch", batch)
    monkeypatch.setattr(submit.time, "sleep", lambda _: state.update(poll=state["poll"] + 1))
    assert submit.phase(args, root, tasks, "validation_only", snapshot(root), {"effective_submit_cap": 60})["complete"]


def test_failed_fit_is_not_retried_within_same_invocation(pilot, monkeypatch):
    root, args, tasks = pilot
    args.watch = args.retry_failed = True
    calls = []
    monkeypatch.setattr(submit, "successful_results", lambda *a: {})
    monkeypatch.setattr(submit, "submit_batch", lambda *a: calls.append(a[2]))
    monkeypatch.setattr(submit.time, "sleep", lambda _: None)
    outcome = submit.phase(args, root, tasks, "validation_only", snapshot(root), {"effective_submit_cap": 60})
    assert len(calls) == 1 and len(outcome["failed"]) == 4 and not outcome["complete"]


def test_stage_gate_failure_never_submits_full_plan(pilot, monkeypatch):
    root, args, tasks = pilot
    args.staged = args.watch = args.evaluate_test = True
    modes = []
    monkeypatch.setattr(submit, "select_tasks", lambda *a, **kw: (tasks, 4))
    def phase(*arguments):
        modes.append(arguments[3])
        return {"successes": {}, "failed": tasks, "complete": False}
    monkeypatch.setattr(submit, "phase", phase)
    with pytest.raises(RuntimeError, match="gate CLOSED"):
        submit.submit(args, root, tasks)
    assert modes == ["validation_only"]
    assert not list((root / "results/accounting/method-pilots").glob("stage-gate-*.json"))


def test_stage_gate_promotes_only_after_all_expected_results_and_records_evidence(pilot, monkeypatch):
    root, args, tasks = pilot
    args.staged = args.watch = args.evaluate_test = True
    modes = []
    paths = {}
    for i, task in enumerate(tasks):
        path = root / f"result-{i}.json"
        path.write_text('{}')
        paths[submit.execution_key(task, "validation_only")] = path
    monkeypatch.setattr(submit, "select_tasks", lambda *a, **kw: (tasks, 4))
    def phase(*arguments):
        modes.append(arguments[3])
        return {"successes": paths, "failed": [], "complete": True}
    monkeypatch.setattr(submit, "phase", phase)
    submit.submit(args, root, tasks)
    assert modes == ["validation_only", "test"]
    gate = json.loads(next((root / "results/accounting/method-pilots").glob("stage-gate-*.json")).read_text())
    assert len(gate["pilot_results"]) == 4 and gate["score_threshold"] is None


def test_worker_stops_if_method_source_changes(pilot, monkeypatch):
    root, _, tasks = pilot
    path = root / "manifest.json"
    submit.write_new(path, {"root": str(root), "python": str(root / ".venv/bin/python"),
                           "tasks": tasks, "worker_sha256": submit.sha256(SCRIPT),
                           "source_sha256": submit.source_digest(root)})
    args = argparse.Namespace(worker=str(path), manifest_sha256=submit.sha256(path))
    (root / "src/model.py").write_text("VERSION = 2\n")
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: pytest.fail("must not train"))
    with pytest.raises(RuntimeError, match="changed after submission"):
        submit.worker(args)


def test_worker_stops_if_environment_changes(pilot, monkeypatch):
    root, _, tasks = pilot
    path = root / "manifest.json"
    submit.write_new(path, {"root": str(root), "python": str(root / ".venv/bin/python"),
                           "tasks": tasks, "worker_sha256": submit.sha256(SCRIPT),
                           "source_sha256": submit.source_digest(root), "environment": {"old": True}})
    with pytest.raises(RuntimeError, match="inventory changed"):
        submit.worker(argparse.Namespace(worker=str(path), manifest_sha256=submit.sha256(path)))


def valid_record(tmp_path, task, snap, mode="validation_only"):
    from imts_agents.runner import RunConfig, _digest
    from imts_agents.variants import get_variant
    folder = tmp_path / "run"
    folder.mkdir()
    config = asdict(RunConfig(**task, device="cuda", threads=4, evaluate_test=mode == "test"))
    record = {"status": "success", "experiment_family": "imts_agents", "profile": "test" if mode == "test" else "validation",
              "config": config, "config_sha256": _digest(config), "variant_spec": asdict(get_variant(task["variant"])),
              "environment": {**snap["environment"], "source_sha256": snap["source_sha256"]},
              "test_evaluated": mode == "test", "comparison_protocol": {"source": snap["source_sha256"]},
              "validation_checks": {key: True for key in ("finite_metrics", "predictions_reconstructed", "action_paths_cover_nonempty_patches")}}
    record["comparison_protocol_sha256"] = _digest(record["comparison_protocol"])
    for name in ("best.pt", "diagnostics.json", "val_predictions.npz", "test_predictions.npz"):
        (folder / name).write_bytes(name.encode())
    record.update(checkpoint_sha256=submit.sha256(folder / "best.pt"), diagnostics_file="diagnostics.json",
                  diagnostics_sha256=submit.sha256(folder / "diagnostics.json"))
    record["validation"] = {"metrics": {"macro_f1": .5}, "predictions_file": "val_predictions.npz",
                            "predictions_sha256": submit.sha256(folder / "val_predictions.npz")}
    if mode == "test":
        record["test"] = {"metrics": {"macro_f1": .4}, "predictions_file": "test_predictions.npz",
                          "predictions_sha256": submit.sha256(folder / "test_predictions.npz")}
    path = folder / "result.json"
    path.write_text(json.dumps(record))
    return path, record


@pytest.mark.parametrize("change", ["nan", "artifact", "source", "environment", "check", "config", "test_scope"])
def test_stage_result_validation_rejects_nonfinite_damaged_or_incompatible_record(tmp_path, change):
    task = {"dataset": "JapaneseVowels", "variant": "core_both", "seed": 0}
    snap = {"source_sha256": "source", "environment": {"python": "3.11", "packages": {}}}
    path, record = valid_record(tmp_path, task, snap)
    assert submit.valid_success(path, task, "validation_only", snap)
    if change == "nan":
        record["validation"]["metrics"]["macro_f1"] = float("nan")
    elif change == "artifact":
        (path.parent / "best.pt").write_bytes(b"changed")
    elif change == "source":
        record["environment"]["source_sha256"] = "different"
    elif change == "environment":
        record["environment"]["packages"] = {"torch": "different"}
    elif change == "check":
        record["validation_checks"]["action_paths_cover_nonempty_patches"] = False
    elif change == "config":
        record["config"]["epochs"] = 1
    else:
        record["test_evaluated"] = True
    path.write_text(json.dumps(record))
    assert not submit.valid_success(path, task, "validation_only", snap)


def test_validation_success_cannot_count_as_test_success(tmp_path):
    task = {"dataset": "JapaneseVowels", "variant": "core_both", "seed": 0}
    snap = {"source_sha256": "source", "environment": {"python": "3.11", "packages": {}}}
    path, _ = valid_record(tmp_path, task, snap)
    assert not submit.valid_success(path, task, "test", snap)


def test_walltime_bounds_and_no_four_fit_cap():
    assert submit.parse_walltime("3-00:00:00") == "3-00:00:00"
    assert submit.parse_walltime("72:00:00") == "72:00:00"
    for value in ("00:00:00", "72:00:01", "3-00:00:01", "1-25:00:00", "0-25:00:00", "bad"):
        with pytest.raises(argparse.ArgumentTypeError):
            submit.parse_walltime(value)
    assert submit.pilot_count("60") == 60


def test_default_is_only_core_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(submit, "queue_ids", lambda: pytest.fail("dry run must not query Slurm"))
    assert submit.main([]) == 0
    output = capsys.readouterr().out
    assert "4 fits; 4 variants in group core" in output
    assert "core_none" in output and "core_both" in output
    assert "DRY RUN" in output and "dynamic_feedback" not in output


def test_staged_plan_is_30_plus_540_and_entirely_offline(monkeypatch, capsys):
    monkeypatch.setattr(submit, "queue_ids", lambda: pytest.fail("dry run must not query Slurm"))
    assert submit.main(["--group", "ablation", "--staged", "--evaluate-test", "--watch"]) == 0
    output = capsys.readouterr().out
    assert "30 validation-only" in output and "540 fresh test fits" in output and "570" in output


def test_multiple_datasets_and_seeds_are_cartesian_product(monkeypatch, capsys):
    monkeypatch.setattr(submit, "queue_ids", lambda: pytest.fail("dry run must not query Slurm"))
    submit.main(["--group", "ablation", "--datasets", "JapaneseVowels", "Physionet2012", "--seeds", "0", "1"])
    assert "40 fits; 10 variants" in capsys.readouterr().out


@pytest.mark.parametrize("argv", [["--staged", "--group", "ablation"],
                                   ["--staged", "--group", "ablation", "--evaluate-test", "--submit"],
                                   ["--max-user-jobs", "61"]])
def test_invalid_stage_or_quota_never_queries_slurm(monkeypatch, argv):
    monkeypatch.setattr(submit, "queue_ids", lambda: pytest.fail("must not query Slurm"))
    with pytest.raises(SystemExit):
        submit.main(argv)


def test_bad_dataset_is_rejected_before_any_slurm_call(monkeypatch):
    monkeypatch.setattr(submit, "queue_ids", lambda: pytest.fail("must not query Slurm"))
    with pytest.raises(ValueError, match="Unknown configured dataset"):
        submit.main(["--submit", "--dataset", "typo"])


def test_baseline_watcher_detection_requires_matching_root_and_submit_watch(tmp_path):
    proc = tmp_path / "proc"
    folder = proc / "123"
    folder.mkdir(parents=True)
    baseline = tmp_path / "baseline"
    baseline.mkdir()
    (folder / "cwd").symlink_to(baseline)
    (folder / "cmdline").write_bytes(b"python\0scripts/submit_benchmarks.py\0--submit\0--watch\0")
    assert submit.baseline_watcher_active(baseline, proc)
    assert not submit.baseline_watcher_active(tmp_path / "other", proc)
    (folder / "cmdline").write_bytes(b"python\0scripts/submit_benchmarks.py\0--dry-run\0")
    assert not submit.baseline_watcher_active(baseline, proc)


def test_unrelated_jobs_cannot_take_baseline_reserved_slots():
    limits = {"effective_submit_cap": 60}
    # 53 method, five unrelated, and the baseline two have just finished.
    ids = {str(i) for i in range(58)}
    assert submit.available_slots(ids, set(), 53, limits, reserve=2, reserved_active=0) == 0
    # Once its next two arrive, no reservation is subtracted a second time.
    assert submit.available_slots(ids | {"base_0", "base_1"}, set(), 53, limits,
                                  reserve=2, reserved_active=2) == 0


def test_only_verified_baseline_gpu_receipts_release_reserved_slots(tmp_path):
    folder = tmp_path / "results/accounting/batches/batch"
    folder.mkdir(parents=True)
    (folder / "gpu.receipt.json").write_text(json.dumps({"status": "submitted", "job_id": "123"}))
    args = argparse.Namespace(baseline_root=tmp_path)
    assert submit.reserved_jobs_active(args, {"123_0", "123_1", "456_0"}) == 2


def worker_manifest(root, task, mode="validation_only"):
    folder = root / "batch"
    folder.mkdir()
    manifest = {"root": str(root), "python": str(root / ".venv/bin/python"), "tasks": [task],
                "worker_sha256": submit.sha256(SCRIPT), **snapshot(root), "mode": mode,
                "batch_id": "batch", "output_dir": str(root / "output")}
    path = folder / "manifest.json"
    submit.write_new(path, manifest)
    return path, manifest


def test_worker_reuses_verified_success_without_training_again(pilot, monkeypatch):
    root, _, tasks = pilot
    path, manifest = worker_manifest(root, tasks[0])
    record = root / "verified.json"
    record.write_text('{}')
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(submit, "successful_results", lambda *a: {submit.execution_key(tasks[0], "validation_only"): record})
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: pytest.fail("must reuse success"))
    assert submit.worker(argparse.Namespace(worker=str(path), manifest_sha256=submit.sha256(path))) == 0
    assert json.loads((path.parent / "task-0.json").read_text())["result_sha256"] == submit.sha256(record)


def test_worker_duplicate_task_lock_blocks_training(pilot, monkeypatch):
    root, _, tasks = pilot
    path, manifest = worker_manifest(root, tasks[0])
    key = submit.execution_key(tasks[0], "validation_only")
    name = submit.hashlib.sha256(json.dumps([key, manifest["source_sha256"]]).encode()).hexdigest()
    lock_dir = Path(manifest["output_dir"]) / ".task-locks"
    lock_dir.mkdir(parents=True)
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(submit, "successful_results", lambda *a: pytest.fail("must stop before training/reuse"))
    with (lock_dir / name).open("a") as stream:
        submit.fcntl.flock(stream, submit.fcntl.LOCK_EX | submit.fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Duplicate worker"):
            submit.worker(argparse.Namespace(worker=str(path), manifest_sha256=submit.sha256(path)))


def test_test_stage_worker_explicitly_passes_evaluate_test(pilot, monkeypatch):
    root, _, tasks = pilot
    path, _ = worker_manifest(root, tasks[0], "test")
    record = root / "verified.json"
    record.write_text('{}')
    queries = []
    def successes(*args):
        queries.append(True)
        return {} if len(queries) == 1 else {submit.execution_key(tasks[0], "test"): record}
    def run(command, **kwargs):
        assert command[0] == "srun" and "--evaluate-test" in command
        assert kwargs["env"]["PYTHONPATH"] == str(root / "src")
        return result()
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    monkeypatch.setattr(submit, "successful_results", successes)
    monkeypatch.setattr(submit.subprocess, "run", run)
    assert submit.worker(argparse.Namespace(worker=str(path), manifest_sha256=submit.sha256(path))) == 0


def test_plan_lock_rejects_second_watcher(pilot):
    root, _, _ = pilot
    with submit.plan_lock(root):
        with pytest.raises(RuntimeError, match="already active"):
            with submit.plan_lock(root):
                pytest.fail("second watch acquired lock")


def test_recovery_reuses_verified_old_success_and_preserves_environment_guard(tmp_path):
    policy = json.loads((SCRIPT.parents[1] / "docs/geolife-recovery-policy.json").read_text())
    task = {"dataset": "JapaneseVowels", "variant": "history_state", "seed": 0}
    old = {"source_sha256": policy["legacy_source_sha256"], "environment": {"python": "3.11", "packages": {}}}
    path, record = valid_record(tmp_path, task, old, mode="test")
    snap = {**old, "source_sha256": policy["corrected_source_sha256"]}
    assert not submit.valid_success(path, task, "test", snap)
    snap["recovery_policy"] = policy
    assert submit.valid_success(path, task, "test", snap)
    record["environment"]["packages"] = {"torch": "changed"}
    path.write_text(json.dumps(record))
    assert not submit.valid_success(path, task, "test", snap)
    assert not submit.compatible_source({"environment": {"source_sha256": "unknown"}}, task, snap)
    assert not submit.compatible_source({"environment": {"source_sha256": old["source_sha256"]}},
                                        {**task, "dataset": "GeolifeSupervised"}, snap)


def test_recovery_plan_skips_500_successes_and_submits_only_40(pilot, monkeypatch):
    root, args, _ = pilot
    args.retry_failed = True
    tasks = [{"dataset": f"dataset-{i // 30}", "variant": f"variant-{i % 10}", "seed": i % 3}
             for i in range(540)]
    successful = {submit.execution_key(task, "test"): "verified.json" for task in tasks[:500]}
    assert len(successful) == 500
    monkeypatch.setattr(submit, "successful_results", lambda *_: successful)
    monkeypatch.setattr(submit, "ledger_status", lambda *_: ({}, set(), 0))
    submitted = []
    monkeypatch.setattr(submit, "submit_batch", lambda _a, _r, chunk, *_: submitted.extend(chunk))
    submit.phase(args, root, tasks, "test", snapshot(root), {"effective_submit_cap": 60})
    assert submitted == tasks[500:]


def test_recovery_submission_freezes_helper_and_policy(pilot, monkeypatch):
    root, args, tasks = pilot
    policy = json.loads((SCRIPT.parents[1] / "docs/geolife-recovery-policy.json").read_text())
    snap = {**snapshot(root), "recovery_policy": policy,
            "recovery_policy_sha256": submit.recovery_helper().policy_digest(policy)}
    monkeypatch.setattr(submit.subprocess, "run", lambda *a, **kw: result("991\n"))
    _, folder = submit.submit_batch(args, root, tasks, "test", set(), {"effective_submit_cap": 60}, snap)
    manifest = json.loads((folder / "manifest.json").read_text())
    assert manifest["recovery_policy"] == policy
    assert manifest["recovery_helper_sha256"] == submit.sha256(folder / "recovery_policy.py")
    assert (folder / "recovery_policy.py").read_bytes() == SCRIPT.with_name("recovery_policy.py").read_bytes()


def test_worker_rejects_changed_frozen_recovery_helper_before_reuse(pilot, monkeypatch):
    root, _, tasks = pilot
    path, manifest = worker_manifest(root, tasks[0])
    manifest["recovery_policy"] = {"kind": "geolife-label-vocabulary-v1"}
    manifest["recovery_helper_sha256"] = "wrong"
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(submit, "successful_results", lambda *_: pytest.fail("must stop before reuse"))
    with pytest.raises(RuntimeError, match="Frozen recovery helper changed"):
        submit.worker(argparse.Namespace(worker=str(path), manifest_sha256=submit.sha256(path)))
