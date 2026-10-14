#!/usr/bin/env python3
"""Review or hand off the unchanged baseline experiment to a 60-task watcher.

Copy this file, submit_parallel_baselines.py, and submit_benchmarks.py into a
separate scripts/ directory. The baseline Python, source, configs, templates,
results, and accounting remain in --baseline-root. No training is performed on
the login node. Default is a read-only plan; --submit --handoff explicitly stops
one verified old watcher, preserves its Slurm jobs, and waits before resuming.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time


SCRIPTS = Path(__file__).resolve().parent
CAP = 60
TIMES = {"cpu": "02:00:00", "gpu": "03:00:00"}


def load_wrapper():
    spec = importlib.util.spec_from_file_location(
        "imts_resume_wrapper", SCRIPTS / "submit_parallel_baselines.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Missing submit_parallel_baselines.py beside this script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_process(pid, proc_root=Path("/proc")):
    """Capture identity, including start time so a recycled PID is not targeted."""
    proc = proc_root / str(pid)
    try:
        uid = proc.stat().st_uid
        if uid != os.getuid():
            return None
        argv = tuple(arg for arg in (proc / "cmdline").read_bytes().decode().split("\0") if arg)
        relevant = {"submit_benchmarks.py", "submit_parallel_baselines.py", "resume_max_baselines.py"}
        if "--submit" not in argv or not any(Path(arg).name in relevant for arg in argv):
            return None
        cwd = (proc / "cwd").resolve(strict=True)
        # /proc/stat's comm field can itself contain spaces or parentheses.
        start = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
        return {"pid": pid, "uid": uid, "cwd": cwd, "argv": argv, "start": start}
    except (FileNotFoundError, ProcessLookupError):
        return None


def flag_value(argv, flag, default):
    for index, arg in enumerate(argv):
        if arg == flag:
            if index + 1 == len(argv):
                raise RuntimeError(f"Malformed running launcher argument: {flag}")
            return argv[index + 1]
        if arg.startswith(flag + "="):
            return arg.split("=", 1)[1]
    return default


def find_watchers(root, proc_root=Path("/proc")):
    old, conflicts = [], []
    expected = root / "scripts/submit_benchmarks.py"
    for proc in proc_root.iterdir():
        if not proc.name.isdecimal() or int(proc.name) == os.getpid():
            continue
        info = read_process(int(proc.name), proc_root)
        if info is None or "--submit" not in info["argv"]:
            continue
        argv, cwd = info["argv"], info["cwd"]
        script_paths = [(Path(arg) if Path(arg).is_absolute() else cwd / arg).resolve()
                        for arg in argv if arg.endswith(".py")]
        if expected in script_paths and "--watch" in argv:
            if cwd == root:
                old.append(info)
            else:
                conflicts.append(info)  # Identified, but outside the permitted stop guard.
        if any(path.name in {"submit_parallel_baselines.py", "resume_max_baselines.py"}
               for path in script_paths):
            baseline = Path(flag_value(argv, "--baseline-root", "~/IMTS")).expanduser()
            baseline = baseline if baseline.is_absolute() else cwd / baseline
            if baseline.resolve() == root:
                conflicts.append(info)
    if len(old) > 1 or conflicts:
        raise RuntimeError("Multiple or already-upgraded baseline launchers need review; "
                           f"no process was stopped. PIDs: {[p['pid'] for p in old + conflicts]}")
    return old


def preflight(root, launcher):
    """All scheduler checks happen before the old watcher is stopped."""
    command = ["sacctmgr", "-n", "-P", "show", "qos", "where", "name=highpriority",
               "format=Name,MaxSubmitJobsPU"]
    result = subprocess.run(command, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Cannot confirm QoS before handoff: {result.stderr.strip()}")
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == "highpriority"]
    if len(rows) != 1 or len(rows[0]) < 2 or rows[0][1].strip() != str(CAP):
        raise RuntimeError(f"Expected confirmed highpriority MaxSubmitJobsPU={CAP}; "
                           f"received {rows!r}. Old watcher was not stopped.")
    launcher.active_user_jobs()  # Require a valid, expanded user queue query.
    clean_env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
    for group, limit in TIMES.items():
        template = root / "slurm" / f"benchmark_{group}.sbatch"
        text = launcher.template_text(root, group)
        if "#SBATCH --qos=highpriority" not in text:
            raise RuntimeError(f"Unexpected QoS in {template}; old watcher was not stopped")
        command = ["sbatch", "--test-only", f"--time={limit}", f"--chdir={root}", str(template)]
        result = subprocess.run(command, text=True, capture_output=True, env=clean_env)
        if result.returncode:
            raise RuntimeError(f"{group.upper()} preflight failed; old watcher was not stopped: "
                               f"{result.stderr.strip() or result.stdout.strip()}")
        print(f"Preflight passed: {group.upper()}, time={limit}", flush=True)


def stop_verified_watcher(info, timeout=30):
    current = read_process(info["pid"])
    if current is None:
        print("Old watcher already exited; no signal sent.", flush=True)
        return
    if current != info:
        raise RuntimeError("Watcher identity changed during preflight; no signal sent")
    # A pidfd keeps the signal tied to this process rather than a reusable PID.
    if not hasattr(os, "pidfd_open") or not hasattr(signal, "pidfd_send_signal"):
        raise RuntimeError("This Python/Linux lacks pidfd signaling; stop the verified watcher manually")
    fd = os.pidfd_open(info["pid"])
    try:
        if read_process(info["pid"]) != info:
            raise RuntimeError("Watcher identity changed before signal; no signal sent")
        signal.pidfd_send_signal(fd, signal.SIGTERM)
    finally:
        os.close(fd)
    print(f"Sent SIGTERM to old watcher PID {info['pid']} only; submitted Slurm jobs are preserved.", flush=True)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        current = read_process(info["pid"])
        if current is None or current["start"] != info["start"]:
            return
        time.sleep(0.5)
    raise RuntimeError("Old watcher has not exited; no new jobs submitted and no force-kill attempted")


def managed_arrays(root, launcher):
    """Read immutable receipts; an absent or uncertain outcome is never retried."""
    output = (root / "results/jv-first-full").resolve()
    arrays = {}
    for path in sorted((root / "results/accounting/batches").glob("*/manifest.json")):
        manifest = launcher.read_json(path)
        if Path(manifest["output_root"]).resolve() != output:
            continue
        if Path(manifest["root"]).resolve() != root:
            raise RuntimeError(f"Unexpected baseline root in {path}")
        for group, tasks in manifest["tasks"].items():
            if not tasks:
                continue
            intent = path.parent / f"{group}.intent.json"
            receipt = path.parent / f"{group}.receipt.json"
            if not intent.exists():
                if receipt.exists():
                    raise RuntimeError(f"Receipt without intent requires review: {receipt}")
                continue
            if not receipt.exists():
                raise RuntimeError(f"Unknown submission outcome; inspect Slurm before retrying: {intent}")
            saved = launcher.read_json(receipt)
            if launcher.known_submit_rejection(saved):
                continue
            job = saved.get("job_id", "")
            if saved.get("status") != "submitted" or not re.fullmatch(r"[0-9]+", job):
                raise RuntimeError(f"Unconfirmed receipt requires review; no retry: {receipt}")
            if job in arrays and arrays[job] != len(tasks):
                raise RuntimeError(f"Conflicting task counts for Slurm array {job}")
            arrays[job] = len(tasks)
    return sorted(arrays.items(), key=lambda item: int(item[0]))


def resume_command(root):
    return [str(root / ".venv/bin/python"), "-u", str(SCRIPTS / "submit_parallel_baselines.py"),
            "--baseline-root", str(root), "--cpu-batch-size", str(CAP),
            "--gpu-batch-size", str(CAP), "--total-batch-size", str(CAP),
            "--cpu-concurrency", str(CAP), "--gpu-concurrency", str(CAP),
            "--max-user-jobs", str(CAP), "--cpu-time", TIMES["cpu"],
            "--gpu-time", TIMES["gpu"], "--submit", "--watch"]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default="~/IMTS")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--handoff", action="store_true",
                        help="with --submit, stop one verified old watcher before waiting")
    args = parser.parse_args(argv)
    if args.handoff and not args.submit:
        parser.error("--handoff requires --submit")
    wrapper = load_wrapper()
    root = wrapper.validate_baseline_root(args.baseline_root)
    launcher = wrapper.load_launcher()
    watchers = find_watchers(root)
    print(f"Unchanged baseline: {root}", flush=True)
    print("Plan: at most 60 total CPU + GPU tasks per wave; no reserved method slots.", flush=True)
    print("GPU time: 3 hours; CPU time: 2 hours. Scheduler allocation is not guaranteed.", flush=True)
    print(f"Verified old watcher PIDs: {[item['pid'] for item in watchers]}", flush=True)
    command = resume_command(root)
    if not args.submit:
        print("DRY RUN: no signals, writes, or submissions. Use --submit --handoff to proceed.", flush=True)
        print("Planned command after safe handoff and accounting checks:\n" + shlex.join(command), flush=True)
        return 0
    if watchers and not args.handoff:
        raise RuntimeError("Old watcher is active; --handoff is required to stop it. No action taken.")
    ledger_parent = root / "results/accounting"
    with launcher.file_lock(ledger_parent / ".resume-max.lock"):
        preflight(root, launcher)
        source, environment = launcher.source_digest(root), launcher.inventory(root / ".venv/bin/python")
        # Re-check after preflight, including a second handoff launched meanwhile.
        watchers = find_watchers(root)
        if watchers and not args.handoff:
            raise RuntimeError("An old watcher appeared during preflight; --handoff is required")
        for watcher in watchers:
            stop_verified_watcher(watcher)
        with launcher.file_lock(ledger_parent / ".submit.lock"):
            jobs = managed_arrays(root, launcher)
            print(f"Checking {len(jobs)} recorded arrays; waiting for complete terminal accounting.", flush=True)
            launcher.wait_for_arrays(jobs, 60)
            if launcher.source_digest(root) != source or launcher.inventory(root / ".venv/bin/python") != environment:
                raise RuntimeError("Baseline source or environment changed during handoff; no jobs submitted")
        print("Earlier managed arrays are terminal. Resuming all still-unsuccessful fits.", flush=True)
        print("Completed successes are reused; prior failures/timeouts may be attempted once again.", flush=True)
        print("Each wave finishes before the next wave starts. Other user jobs count toward 60.", flush=True)
        os.execv(command[0], command)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
