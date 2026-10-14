"""Focused safety checks for a launcher-only handoff; never contact Slurm."""
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("resume_max_test", REPO / "scripts/resume_max_baselines.py")
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


@pytest.fixture
def flow(tmp_path, monkeypatch):
    root = tmp_path / "baseline"
    root.mkdir()
    seen = []
    launcher = SimpleNamespace(
        file_lock=lambda path: contextlib.nullcontext(),
        source_digest=lambda root: "source",
        inventory=lambda python: {"python": "3.11"},
        wait_for_arrays=lambda jobs, interval: seen.append(("wait", jobs, interval)),
    )
    wrapper = SimpleNamespace(validate_baseline_root=lambda value: root,
                              load_launcher=lambda: launcher)
    monkeypatch.setattr(resume, "load_wrapper", lambda: wrapper)
    monkeypatch.setattr(resume, "find_watchers", lambda root: [])
    monkeypatch.setattr(resume, "preflight", lambda root, launcher: seen.append("preflight"))
    monkeypatch.setattr(resume, "managed_arrays", lambda root, launcher: [("162140", 2)])
    monkeypatch.setattr(resume, "stop_verified_watcher", lambda info: seen.append(("stop", info["pid"])))
    monkeypatch.setattr(resume.os, "execv", lambda executable, args: seen.append(("exec", executable, args)))
    return root, launcher, seen


def test_default_is_read_only_plan(flow, capsys):
    root, launcher, seen = flow
    assert resume.main(["--baseline-root", str(root)]) == 0
    assert seen == []
    assert "DRY RUN" in capsys.readouterr().out


def test_submit_refuses_old_watcher_without_handoff(flow, monkeypatch):
    root, launcher, seen = flow
    monkeypatch.setattr(resume, "find_watchers", lambda root: [{"pid": 26722}])
    with pytest.raises(RuntimeError, match="--handoff is required"):
        resume.main(["--submit"])
    assert seen == []


def test_preflight_failure_never_stops_existing_watcher(flow, monkeypatch):
    root, launcher, seen = flow
    monkeypatch.setattr(resume, "find_watchers", lambda root: [{"pid": 26722}])
    def fail(*args):
        raise RuntimeError("GPU preflight failed")
    monkeypatch.setattr(resume, "preflight", fail)
    with pytest.raises(RuntimeError, match="GPU preflight failed"):
        resume.main(["--submit", "--handoff"])
    assert seen == []


def test_handoff_waits_then_executes_original_python_and_combined_cap(flow, monkeypatch):
    root, launcher, seen = flow
    monkeypatch.setattr(resume, "find_watchers", lambda root: [{"pid": 26722}])
    assert resume.main(["--submit", "--handoff"]) == 0
    assert seen[:3] == ["preflight", ("stop", 26722), ("wait", [("162140", 2)], 60)]
    _, executable, args = seen[3]
    assert executable == str(root / ".venv/bin/python")
    assert args[args.index("--baseline-root") + 1] == str(root)
    for flag in ("--cpu-batch-size", "--gpu-batch-size", "--total-batch-size",
                 "--cpu-concurrency", "--gpu-concurrency", "--max-user-jobs"):
        assert args[args.index(flag) + 1] == "60"
    assert args[args.index("--gpu-time") + 1] == "03:00:00"
    assert args[-2:] == ["--submit", "--watch"]


def test_ambiguous_receipt_or_accounting_failure_never_executes(flow, monkeypatch):
    root, launcher, seen = flow
    def fail(*args):
        raise RuntimeError("Unknown submission outcome")
    monkeypatch.setattr(resume, "managed_arrays", fail)
    with pytest.raises(RuntimeError, match="Unknown submission"):
        resume.main(["--submit"])
    assert seen == ["preflight"]


def test_training_source_change_while_waiting_refuses_resume(flow, monkeypatch):
    root, launcher, seen = flow
    sources = iter(["original", "changed"])
    launcher.source_digest = lambda root: next(sources)
    with pytest.raises(RuntimeError, match="changed during handoff"):
        resume.main(["--submit"])
    assert not any(isinstance(item, tuple) and item[0] == "exec" for item in seen)


def process_fixture(proc_root, pid, cwd, argv, start="100"):
    proc = proc_root / str(pid)
    proc.mkdir(parents=True)
    (proc / "cwd").symlink_to(cwd, target_is_directory=True)
    (proc / "cmdline").write_bytes("\0".join(argv).encode() + b"\0")
    # starttime is field 22; the suffix begins with state (field 3).
    (proc / "stat").write_text(f"{pid} (python worker) S " + "0 " * 18 + start + " 0\n")
    return proc


def test_only_uid_cwd_exact_script_and_flags_qualify(tmp_path, monkeypatch):
    root = tmp_path / "baseline"
    root.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    procs = tmp_path / "proc"
    process_fixture(procs, 990001, root, ["python", "scripts/submit_benchmarks.py", "--submit", "--watch"])
    process_fixture(procs, 990002, other, ["python", "scripts/submit_benchmarks.py", "--submit", "--watch"])
    process_fixture(procs, 990003, root, ["python", "scripts/submit_benchmarks.py", "--submit"])
    watchers = resume.find_watchers(root, procs)
    assert [item["pid"] for item in watchers] == [990001]
    assert watchers[0]["start"] == "100"
    monkeypatch.setattr(resume.os, "getuid", lambda: os.stat(root).st_uid + 1)
    assert resume.find_watchers(root, procs) == []


def test_multiple_or_upgraded_watchers_are_not_stopped(tmp_path):
    root = tmp_path / "baseline"
    root.mkdir()
    procs = tmp_path / "proc"
    process_fixture(procs, 990001, root, ["python", "scripts/submit_benchmarks.py", "--submit", "--watch"])
    process_fixture(procs, 990002, root, ["python", "scripts/submit_benchmarks.py", "--submit", "--watch"])
    with pytest.raises(RuntimeError, match="Multiple or already-upgraded"):
        resume.find_watchers(root, procs)
    (procs / "990002/cmdline").write_bytes(
        f"python\0/tmp/tool/scripts/submit_parallel_baselines.py\0--submit\0--watch\0--baseline-root={root}\0".encode())
    with pytest.raises(RuntimeError, match="990002"):
        resume.find_watchers(root, procs)


def test_expected_script_with_wrong_cwd_is_conflict(tmp_path):
    root = tmp_path / "baseline"
    root.mkdir()
    procs = tmp_path / "proc"
    process_fixture(procs, 990001, tmp_path,
                    ["python", str(root / "scripts/submit_benchmarks.py"), "--submit", "--watch"])
    with pytest.raises(RuntimeError, match="no process was stopped"):
        resume.find_watchers(root, procs)


def test_pidfd_only_signals_verified_watcher_and_never_slurm(monkeypatch):
    info = {"pid": 26722, "start": "123", "argv": ("python",), "cwd": Path("/baseline"), "uid": os.getuid()}
    readings = iter([info, info, None])
    seen = []
    monkeypatch.setattr(resume, "read_process", lambda pid: next(readings))
    monkeypatch.setattr(resume.os, "pidfd_open", lambda pid: 123)
    monkeypatch.setattr(resume.signal, "pidfd_send_signal", lambda fd, sig: seen.append((fd, sig)))
    monkeypatch.setattr(resume.os, "close", lambda fd: seen.append(("close", fd)))
    resume.stop_verified_watcher(info)
    assert seen == [(123, signal.SIGTERM), ("close", 123)]


def test_reused_pid_is_never_signalled(monkeypatch):
    info = {"pid": 26722, "start": "123"}
    monkeypatch.setattr(resume, "read_process", lambda pid: {**info, "start": "456"})
    with pytest.raises(RuntimeError, match="identity changed"):
        resume.stop_verified_watcher(info)


def test_qos_and_both_templates_preflight_without_real_submission(tmp_path, monkeypatch):
    calls = []
    launcher = SimpleNamespace(active_user_jobs=lambda: set(),
                               template_text=lambda root, group: "#SBATCH --qos=highpriority\n")
    def run(command, **kwargs):
        calls.append(command)
        if command[0] == "sacctmgr":
            return subprocess.CompletedProcess(command, 0, "highpriority|60|\n", "")
        assert command[0:2] == ["sbatch", "--test-only"]
        assert "SBATCH_TIME" not in kwargs["env"]
        return subprocess.CompletedProcess(command, 0, "", "Job would run")
    monkeypatch.setattr(resume.subprocess, "run", run)
    monkeypatch.setenv("SBATCH_TIME", "00:01:00")
    resume.preflight(tmp_path, launcher)
    assert len(calls) == 3
    assert "--time=02:00:00" in calls[1]
    assert "--time=03:00:00" in calls[2]


@pytest.mark.parametrize("response", ["highpriority|30|\n", "highpriority||\n", "", "highpriority|60|\nhighpriority|60|\n"])
def test_unconfirmed_qos_cap_refuses_preflight(tmp_path, monkeypatch, response):
    monkeypatch.setattr(resume.subprocess, "run", lambda command, **kw: subprocess.CompletedProcess(command, 0, response, ""))
    with pytest.raises(RuntimeError, match="Expected confirmed"):
        resume.preflight(tmp_path, SimpleNamespace())


def write_manifest(root, name, receipt=None, intent=True):
    folder = root / "results/accounting/batches" / name
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"root": str(root), "output_root": str(root / "results/jv-first-full"),
                                                     "tasks": {"gpu": [{"model": "BRITS"}]}}))
    if intent:
        (folder / "gpu.intent.json").write_text("{}")
    if receipt is not None:
        (folder / "gpu.receipt.json").write_text(json.dumps(receipt))


def test_ledger_only_confirms_receipts_and_skips_known_refusal(tmp_path):
    launcher = resume.load_wrapper().load_launcher()
    write_manifest(tmp_path, "first", {"status": "submitted", "job_id": "162140"})
    write_manifest(tmp_path, "second", {"status": "unconfirmed", "job_id": "", "returncode": 1,
                                       "stdout": "", "stderr": "sbatch: error: QOSMaxSubmitJobPerUserLimit\n"
                                       "sbatch: error: Batch job submission failed: Job violates policy\n"})
    assert resume.managed_arrays(tmp_path, launcher) == [("162140", 1)]


@pytest.mark.parametrize("receipt", [None, {"status": "unconfirmed", "job_id": ""},
                                      {"status": "submitted", "job_id": "bad"}])
def test_ambiguous_ledger_outcomes_never_become_retry(tmp_path, receipt):
    launcher = resume.load_wrapper().load_launcher()
    write_manifest(tmp_path, "first", receipt)
    with pytest.raises(RuntimeError, match="Unknown submission|Unconfirmed receipt"):
        resume.managed_arrays(tmp_path, launcher)
