#!/usr/bin/env python3
"""Submit method experiments with rolling refill within the shared 60-task quota.

Dry run is entirely offline. --staged --evaluate-test --submit --watch first
checks 30 validation fits, then runs the frozen 540-fit full comparison. The
second stage starts only when every pilot result passes the recorded checks.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import asdict
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
HARD_CAP = 60
QOS = "highpriority"
PILOT_DATASETS = ("JapaneseVowels", "CharacterTrajectories", "Physionet2012")
TERMINAL = {"BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED",
            "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT"}
INVENTORY_CODE = """
import importlib.metadata, json, platform
packages = {d.metadata['Name'].lower().replace('_', '-').replace('.', '-'): d.version
            for d in importlib.metadata.distributions() if d.metadata['Name']}
print(json.dumps({'python': platform.python_version(), 'packages': packages}, sort_keys=True))
"""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root):
    digest = hashlib.sha256()
    for directory in (root / "src", root / "configs"):
        for path in sorted(directory.rglob("*")):
            if path.is_file() and path.suffix in {".py", ".json"}:
                digest.update(str(path.relative_to(root)).encode())
                digest.update(path.read_bytes())
    return digest.hexdigest()


def recovery_helper():
    path = Path(__file__).with_name("recovery_policy.py")
    spec = importlib.util.spec_from_file_location("imts_recovery_policy", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def compatible_source(record, task, snapshot):
    source = record.get("environment", {}).get("source_sha256")
    policy = snapshot.get("recovery_policy")
    if policy is None:
        return source == snapshot["source_sha256"]
    return (snapshot["source_sha256"] == policy["corrected_source_sha256"]
            and recovery_helper().source_allowed(policy, source, task["dataset"], task["seed"]))


def inventory(python):
    result = subprocess.run([str(python), "-c", INVENTORY_CODE], text=True,
                            capture_output=True, check=True)
    return json.loads(result.stdout)


def write_new(path, value):
    """Publish a complete immutable JSON file; never expose a half-written ledger."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    try:
        with temporary.open("x") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # atomic publication, and refuses replacement
    finally:
        temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def submission_lock():
    path = Path.home() / ".cache/imts/submission.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        # Only held for a queue snapshot and one sbatch, never while waiting for jobs.
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


@contextlib.contextmanager
def plan_lock(root):
    path = root / "results/accounting/method-pilots/.watch.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another method submitter/watch is already active in this checkout.") from error
        yield


def queue_ids():
    result = subprocess.run(["squeue", "-h", "-r", "-u", getpass.getuser(), "-o", "%i"],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Cannot confirm the expanded user queue: {result.stderr.strip()}")
    ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if any(not re.fullmatch(r"[0-9]+(?:_[0-9]+)?", value) for value in ids):
        raise RuntimeError("Unexpected or unexpanded squeue job IDs; no submission made.")
    return set(ids)


def qos_limits():
    fields = "Name,MaxSubmitJobsPU,MaxJobsPU,MaxTRESPU,MaxTRESPA,GrpTRES,MaxWall"
    result = subprocess.run(["sacctmgr", "-n", "-P", "show", "qos", "where", f"name={QOS}",
                             f"format={fields}"], text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Cannot confirm QoS limits: {result.stderr.strip()}")
    rows = [line.strip().split("|") for line in result.stdout.splitlines() if line.strip()]
    rows = [row for row in rows if row[0] == QOS]
    if len(rows) != 1 or len(rows[0]) < 7:
        raise RuntimeError("Missing or unexpected highpriority QoS record; no submission made.")
    record = dict(zip(fields.split(","), rows[0]))
    raw = record["MaxSubmitJobsPU"].strip()
    if raw and not raw.isdecimal():
        raise RuntimeError(f"Unrecognized MaxSubmitJobsPU value: {raw!r}; no submission made.")
    record["effective_submit_cap"] = min(HARD_CAP, int(raw)) if raw else HARD_CAP
    return record


def check_capacity(ids, count, limits):
    cap = limits["effective_submit_cap"]
    if len(ids) + count > cap:
        raise RuntimeError(f"Submission would exceed the cap: {len(ids)} existing + {count} method "
                           f"> {cap}. Running and pending tasks both count; no jobs submitted.")


def accounting_rows(job_id):
    result = subprocess.run(["sacct", "--noheader", "--array", "--allocations", "--parsable2",
                             "--jobs", job_id, "--format=JobID%80,State%40,ExitCode"],
                            text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(f"Cannot confirm prior method accounting: {result.stderr.strip()}")
    states = {}
    for line in result.stdout.splitlines():
        fields = line.strip().split("|")
        if len(fields) >= 3 and fields[0]:
            states[fields[0]] = (fields[1].split()[0].rstrip("+") if fields[1].split() else "", fields[2])
    return states


def accounting_states(job_id, count):
    states = accounting_rows(job_id)
    expected = [f"{job_id}_{index}" for index in range(count)]
    if any(key not in states or states[key][0] not in TERMINAL for key in expected):
        raise RuntimeError(f"Method array {job_id} has incomplete/nonterminal accounting.")
    return [states[key] for key in expected]


def task_key(task):
    return task["dataset"], task["variant"], task["seed"]


def execution_key(task, mode):
    return (*task_key(task), mode)


def ledger_status(ledger, ids):
    """Track tasks separately: an active array does not block unrelated refills.

    Unknown accounting is still outstanding and consumes quota conservatively.
    Missing/ambiguous receipts always stop submission rather than risk duplicates.
    """
    entries, invisible, active_count = {}, set(), 0
    for folder in sorted(ledger.glob("*")) if ledger.exists() else []:
        if not folder.is_dir() or not (folder / "intent.json").exists():
            continue
        manifest_path, receipt_path = folder / "manifest.json", folder / "receipt.json"
        if not manifest_path.exists() or not receipt_path.exists():
            raise RuntimeError(f"Ambiguous earlier method submission: {folder}. Review it; no automatic retry.")
        manifest = json.loads(manifest_path.read_text())
        intent = json.loads((folder / "intent.json").read_text())
        receipt = json.loads(receipt_path.read_text())
        digest = sha256(manifest_path)
        if intent.get("manifest_sha256") != digest or receipt.get("manifest_sha256") != digest:
            raise RuntimeError(f"Prior method ledger hash mismatch: {folder}")
        if receipt.get("status") == "rejected":
            continue
        job_id = receipt.get("job_id", "")
        if receipt.get("status") != "submitted" or not re.fullmatch(r"[0-9]+", job_id):
            raise RuntimeError(f"Unconfirmed earlier method submission: {receipt_path}. Review before retrying.")
        expected = [f"{job_id}_{index}" for index in range(len(manifest["tasks"]))]
        states = accounting_rows(job_id) if any(key not in ids for key in expected) else {}
        for index, (task, job_task) in enumerate(zip(manifest["tasks"], expected)):
            state, exit_code = states.get(job_task, ("", ""))
            active = job_task in ids or state not in TERMINAL
            key = execution_key(task, manifest.get("mode", "validation_only"))
            entry = {"task": task, "active": active, "state": state, "exit_code": exit_code,
                     "folder": str(folder), "job_task": job_task, "manifest": manifest}
            entries.setdefault(key, []).append(entry)
            if active:
                active_count += 1
                if job_task not in ids:
                    invisible.add(job_task)
    return entries, invisible, active_count


def check_previous(ledger, ids, tasks, retry_failed=False, mode="validation_only"):
    entries, _, _ = ledger_status(ledger, ids)
    for task in tasks:
        for entry in entries.get(execution_key(task, mode), []):
            if entry["active"]:
                raise RuntimeError(f"Method task {task_key(task)} remains active; it cannot be duplicated.")
            if entry["state"] == "COMPLETED" and entry["exit_code"] == "0:0":
                raise RuntimeError(f"Method task {task_key(task)} already completed; inspect its result.")
            if not retry_failed:
                raise RuntimeError(f"Method task {task_key(task)} previously failed; use --retry-failed after log review.")


def parse_walltime(value):
    match = re.fullmatch(r"(?:(\d+)-)?(\d{1,3}):([0-5]\d):([0-5]\d)", value)
    if not match:
        raise argparse.ArgumentTypeError("Use HH:MM:SS or D-HH:MM:SS")
    days, hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if match.group(1) is not None and hours > 23:
        raise argparse.ArgumentTypeError("With D-HH:MM:SS, HH must be 00–23")
    total = days * 86400 + hours * 3600 + minutes * 60 + seconds
    if not 0 < total <= 3 * 86400:
        raise argparse.ArgumentTypeError("Walltime must be positive and at most 3 days")
    return value


def positive_int(value):
    count = int(value)
    if count < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return count


def pilot_count(value):
    return positive_int(value)  # retained CLI compatibility, no arbitrary four-fit limit


def configured_datasets(root):
    return [item["name"] for item in json.loads((root / "configs/datasets.json").read_text())["datasets"]]


def select_tasks(args, root, *, staged_pilot=False, staged_full=False):
    configured = configured_datasets(root)
    datasets = (list(PILOT_DATASETS) if staged_pilot else configured if staged_full or args.all_datasets
                else args.datasets or [args.dataset])
    seeds = [0] if staged_pilot else [0, 1, 2] if staged_full else args.seeds or [args.seed]
    if len(set(datasets)) != len(datasets) or len(set(seeds)) != len(seeds):
        raise ValueError("Repeated datasets/seeds are not allowed")
    for dataset in datasets:
        if dataset not in configured:
            raise ValueError(f"Unknown configured dataset {dataset!r}; no method submitted")
    if any(not 0 <= seed < 2**32 for seed in seeds):
        raise ValueError("Seeds must be in [0, 2**32)")
    sys.path.insert(0, str(root / "src"))
    registry = importlib.import_module("imts_agents.variants")
    group = "motivation" if args.group == "motif" else args.group
    variants = registry.variant_names(group)
    if not 0 <= args.start_index < len(variants):
        raise ValueError(f"--start-index must be between 0 and {len(variants) - 1} for {group}")
    chosen = variants[args.start_index:args.start_index + args.count if args.count else None]
    return [{"variant": name, "dataset": dataset, "seed": seed}
            for dataset in datasets for seed in seeds for name in chosen], len(variants)


def sbatch_base(root, walltime):
    return ["sbatch", "--parsable", "--export=ALL", "--no-requeue", f"--chdir={root}",
            "--account=prj0000000262", "--partition=h200p", f"--qos={QOS}", "--nodes=1",
            "--ntasks=1", "--cpus-per-task=4", "--mem=32G", "--gres=gpu:1", f"--time={walltime}"]


def clean_submit_env():
    return {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}


def output_path(args, root):
    output = Path(args.output_dir).expanduser()
    return (root / output).resolve() if not output.is_absolute() else output.resolve()


def baseline_watcher_active(root, proc=Path("/proc")):
    """Read only: reserve two slots while the user's old two-GPU watch is alive."""
    if not root:
        return False
    root = Path(root).expanduser().resolve()
    for folder in proc.iterdir():
        if not folder.name.isdecimal():
            continue
        try:
            if folder.stat().st_uid != os.getuid():
                continue
            argv = [x.decode(errors="replace") for x in (folder / "cmdline").read_bytes().split(b"\0") if x]
            if "--submit" not in argv or "--watch" not in argv:
                continue
            scripts = {Path(value).name for value in argv[:3]}
            if not scripts & {"submit_benchmarks.py", "submit_parallel_baselines.py"}:
                continue
            actual_root = (folder / "cwd").resolve()
            if "--baseline-root" in argv:
                actual_root = Path(argv[argv.index("--baseline-root") + 1]).expanduser().resolve()
            if actual_root == root:
                return True
        except (OSError, IndexError):
            continue  # process may exit during the snapshot
    return False


def reserve_count(args):
    return max(getattr(args, "reserve_other_jobs", 0),
               2 if baseline_watcher_active(getattr(args, "baseline_root", None)) else 0)


def reserved_jobs_active(args, ids):
    """Count the legacy baseline's own GPU jobs, not unrelated jobs in its slots."""
    root = getattr(args, "baseline_root", None)
    if not root:
        return 0
    job_ids = set()
    for path in (Path(root).expanduser() / "results/accounting/batches").glob("*/gpu.receipt.json"):
        try:
            receipt = json.loads(path.read_text())
            job_id = receipt.get("job_id", "")
            if receipt.get("status") == "submitted" and re.fullmatch(r"[0-9]+", job_id):
                job_ids.add(job_id)
        except (OSError, ValueError, TypeError):
            continue  # unknown records cannot be used to release the reservation
    return sum(value.split("_", 1)[0] in job_ids for value in ids)


def available_slots(ids, invisible, active_method, limits, reserve=0, reserved_active=0):
    cap = limits["effective_submit_cap"]
    protected_empty = max(0, reserve - reserved_active)
    global_free = cap - len(ids | invisible) - protected_empty
    method_free = cap - reserve - active_method
    return max(0, min(global_free, method_free))


def test_only(args, root, tasks):
    with submission_lock():
        ids, limits = queue_ids(), qos_limits()
        limits["effective_submit_cap"] = min(limits["effective_submit_cap"], getattr(args, "max_user_jobs", HARD_CAP))
        count = min(len(tasks), max(0, limits["effective_submit_cap"] - len(ids)))
        if not count:
            raise RuntimeError("No current submission headroom; --test-only submitted nothing.")
        command = sbatch_base(root, args.time) + ["--test-only", f"--array=0-{count - 1}%{count}",
                                                 "--job-name=imts-method-preflight", "--wrap=true"]
        result = subprocess.run(command, text=True, capture_output=True, env=clean_submit_env())
    print(result.stdout.strip(), flush=True)
    if result.stderr.strip():
        print(result.stderr.strip(), file=sys.stderr, flush=True)
    if result.returncode:
        raise RuntimeError("Slurm --test-only preflight failed; no jobs were submitted.")
    print("Preflight accepted. No jobs submitted; this does not reserve resources or guarantee a start time.")


def submit_batch(args, root, tasks, mode, ids, limits, snapshot):
    """Caller holds submission_lock and already subtracts outstanding invisible jobs."""
    if not 1 <= len(tasks) <= limits["effective_submit_cap"]:
        raise ValueError("A submission must fit inside the global task quota")
    check_capacity(ids, len(tasks), limits)
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"Missing isolated method environment: {python}")
    if source_digest(root) != snapshot["source_sha256"] or inventory(python) != snapshot["environment"]:
        raise RuntimeError("Source/environment changed during this plan; no more jobs submitted.")
    output = output_path(args, root)
    ledger = root / "results/accounting/method-pilots"
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = ledger / batch_id
    folder.mkdir(parents=True, exist_ok=False)
    worker_path = folder / "worker.py"
    worker_path.write_bytes(Path(__file__).read_bytes())
    if snapshot.get("recovery_policy") is not None:
        helper_path = folder / "recovery_policy.py"
        helper_path.write_bytes(Path(__file__).with_name("recovery_policy.py").read_bytes())
    template_path = root / "slurm/method_gpu.sbatch"
    template = template_path.read_text()
    manifest = {"schema_version": 2, "batch_id": batch_id,
                "created_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
                "python": str(python), "output_dir": str(output), "tasks": tasks,
                **snapshot, "worker_sha256": sha256(worker_path), "template_sha256": sha256(template_path),
                "mode": mode, "group": args.group,
                "resources": {"walltime": args.time, "gpu_per_task": 1, "cpus_per_task": 4,
                              "memory_per_task": "32G", "global_task_cap": limits["effective_submit_cap"]},
                "quota_snapshot": {"all_user_tasks": len(ids), "limits": limits}}
    if snapshot.get("recovery_policy") is not None:
        manifest["recovery_helper_sha256"] = sha256(helper_path)
    manifest_path = folder / "manifest.json"
    write_new(manifest_path, manifest)
    manifest_hash = sha256(manifest_path)
    wrapper = folder / "job.sbatch"
    invocation = [str(python), str(worker_path), "--worker", str(manifest_path),
                  "--manifest-sha256", manifest_hash]
    wrapper.write_text(template.rstrip() + "\nexec " + shlex.join(invocation) + "\n")
    command = sbatch_base(root, args.time) + [f"--array=0-{len(tasks) - 1}%{len(tasks)}",
               f"--job-name=imts-method-{batch_id}", f"--output={folder}/slurm-%A_%a.out", str(wrapper)]
    write_new(folder / "intent.json", {"command": command, "manifest_sha256": manifest_hash,
                                       "wrapper_sha256": sha256(wrapper)})
    try:
        result = subprocess.run(command, text=True, capture_output=True,
                                env=clean_submit_env(), timeout=120)
        stdout, stderr, returncode = result.stdout, result.stderr, result.returncode
    except (OSError, subprocess.TimeoutExpired) as error:
        write_new(folder / "receipt.json", {"status": "unconfirmed", "job_id": "", "returncode": None,
                  "stdout": "", "stderr": str(error), "manifest_sha256": manifest_hash})
        raise RuntimeError(f"sbatch outcome requires review: {folder / 'receipt.json'}. "
                           "No automatic retry was made.") from error
    match = re.fullmatch(r"([0-9]+)(?:;[A-Za-z0-9_.-]+)?", stdout.strip())
    confirmed = returncode == 0 and match is not None
    job_id = match.group(1) if confirmed else ""
    rejected = (returncode != 0 and not stdout.strip()
                and "sbatch: error: QOSMaxSubmitJobPerUserLimit" in stderr
                and "sbatch: error: Batch job submission failed:" in stderr)
    receipt = {"status": "submitted" if confirmed else "rejected" if rejected else "unconfirmed",
               "job_id": job_id, "returncode": returncode, "stdout": stdout, "stderr": stderr,
               "manifest_sha256": manifest_hash}
    write_new(folder / "receipt.json", receipt)
    if not confirmed:
        raise RuntimeError(f"Method submission {receipt['status']}: {stderr.strip()}\n"
                           f"Review {folder / 'receipt.json'}; no automatic retry was made.")
    print(f"Submitted method array {job_id}: {len(tasks)} {mode} fits; 1 GPU per running fit.", flush=True)
    print(f"Manifest, receipt, and logs: {folder}", flush=True)
    return job_id, folder


def normalized_environment(value):
    return {"python": value.get("python"),
            "packages": {k.lower().replace("_", "-").replace(".", "-"): v
                         for k, v in value.get("packages", {}).items()}}


def valid_success(path, task, mode, snapshot, require_diagnostics=True):
    """No score thresholds: gate on completion, finite metrics and intact artifacts."""
    from imts_agents.runner import RunConfig, _digest
    from imts_agents.variants import get_variant
    try:
        record = json.loads(Path(path).read_text())
        expected = asdict(RunConfig(**task, device="cuda", threads=4, evaluate_test=mode == "test"))
        if (record.get("status") != "success" or record.get("experiment_family") != "imts_agents"
                or record.get("profile") != ("test" if mode == "test" else "validation")
                or record.get("config") != expected or record.get("config_sha256") != _digest(expected)
                or record.get("variant_spec") != asdict(get_variant(task["variant"]))
                or not compatible_source(record, task, snapshot)
                or normalized_environment(record.get("environment", {})) != snapshot["environment"]
                or record.get("test_evaluated") != (mode == "test")):
            return False
        if record.get("comparison_protocol_sha256") != _digest(record["comparison_protocol"]):
            return False
        artifacts = [("best.pt", record["checkpoint_sha256"])]
        for split in ("validation", "test") if mode == "test" else ("validation",):
            section = record[split]
            metrics = section["metrics"]
            if not metrics or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in metrics.values()):
                return False
            artifacts.append((section["predictions_file"], section["predictions_sha256"]))
        if require_diagnostics:
            checks = record.get("validation_checks", {})
            if any(checks.get(key) is not True for key in
                   ("finite_metrics", "predictions_reconstructed", "action_paths_cover_nonempty_patches")):
                return False
            artifacts.append((record["diagnostics_file"], record["diagnostics_sha256"]))
        for name, digest in artifacts:
            artifact = Path(path).parent / name
            if artifact.resolve().parent != Path(path).parent.resolve() or sha256(artifact) != digest:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False


def successful_results(output, tasks, mode, snapshot):
    profile = "test" if mode == "test" else "validation"
    found = {}
    for task in tasks:
        parent = output / profile / task["dataset"] / task["variant"] / f"seed-{task['seed']}"
        for path in sorted(parent.glob("*/result.json")):
            if valid_success(path, task, mode, snapshot):
                found[execution_key(task, mode)] = path
                break  # earliest compatible success; no best-score selection
    return found


def phase(args, root, tasks, mode, snapshot, limits):
    """Each fit is attempted at most once in this invocation, rolling on free slots."""
    output, ledger = output_path(args, root), root / "results/accounting/method-pilots"
    attempted = set()
    while True:
        if source_digest(root) != snapshot["source_sha256"]:
            raise RuntimeError("Source changed during the plan; refusing further submission or stage promotion.")
        with submission_lock():
            ids = queue_ids()
            entries, invisible, active_method = ledger_status(ledger, ids)
            successes = successful_results(output, tasks, mode, snapshot)
            ready, active, failed = [], [], []
            for task in tasks:
                key = execution_key(task, mode)
                # A result may be written before Slurm releases its allocation.
                prior = entries.get(key, [])
                if key in successes:
                    continue
                if any(entry["active"] for entry in prior):
                    active.append(task)
                elif key in attempted or (prior and not args.retry_failed):
                    failed.append(task)
                else:
                    ready.append(task)
            slots = available_slots(ids, invisible, active_method, limits, reserve_count(args),
                                    reserved_jobs_active(args, ids))
            chunk = ready[:slots]
            if chunk:
                submit_batch(args, root, chunk, mode, ids | invisible, limits, snapshot)
                attempted.update(execution_key(task, mode) for task in chunk)
                active.extend(chunk)
                ready = ready[len(chunk):]
        print(f"{mode}: successful {len(successes)}/{len(tasks)}, outstanding {len(active)}, "
              f"not yet submitted {len(ready)}, unsuccessful {len(failed)}; "
              f"global occupied {len(ids | invisible) + len(chunk)}/{limits['effective_submit_cap']}.", flush=True)
        if not ready and not active:
            return {"successes": successes, "failed": failed, "complete": len(successes) == len(tasks)}
        if not args.watch:
            print("Stopped after the current submission. Use --watch to refill freed slots automatically.", flush=True)
            return {"successes": successes, "failed": failed, "complete": False}
        print(f"Watching; checking for free slots again in {args.poll_seconds} seconds.", flush=True)
        time.sleep(args.poll_seconds)


def submit(args, root, tasks):
    python = root / ".venv/bin/python"
    if not python.is_file():
        raise RuntimeError(f"Missing isolated method environment: {python}")
    snapshot = {"source_sha256": source_digest(root), "environment": inventory(python)}
    if getattr(args, "recovery_policy", None):
        policy = recovery_helper().load_policy(args.recovery_policy, root)
        snapshot.update(recovery_policy=policy,
                        recovery_policy_sha256=recovery_helper().policy_digest(policy))
        print("Explicit recovery approval: " + json.dumps(recovery_helper().summary(policy)), flush=True)
    limits = qos_limits()
    limits["effective_submit_cap"] = min(limits["effective_submit_cap"], args.max_user_jobs)
    with plan_lock(root):
        if args.staged:
            pilot_tasks, _ = select_tasks(args, root, staged_pilot=True)
            gate = phase(args, root, pilot_tasks, "validation_only", snapshot, limits)
            if not gate["complete"]:
                raise RuntimeError(f"Stage gate CLOSED: {len(gate['successes'])}/{len(pilot_tasks)} compatible "
                                   "pilot successes. Full comparison was NOT submitted; inspect pilot logs.")
            # Recheck bytes and environment at the actual promotion boundary.
            if source_digest(root) != snapshot["source_sha256"] or inventory(python) != snapshot["environment"]:
                raise RuntimeError("Source/environment changed at stage gate; full comparison was NOT submitted.")
            evidence = {"created_at": datetime.now(timezone.utc).isoformat(), **snapshot,
                        "pilot_results": [{"task": list(key), "path": str(path), "sha256": sha256(path)}
                                          for key, path in gate["successes"].items()],
                        "gate": "all expected pilot tasks complete; finite validation metrics; verified artifacts",
                        "score_threshold": None, "full_fits": len(tasks),
                        "full_training": "fresh fits, including pilot seed0; official test evaluated once per fit"}
            gate_path = root / "results/accounting/method-pilots" / ("stage-gate-" + uuid.uuid4().hex + ".json")
            write_new(gate_path, evidence)
            print(f"Stage gate OPEN: all {len(pilot_tasks)} pilot checks passed. Evidence: {gate_path}", flush=True)
        outcome = phase(args, root, tasks, "test" if args.evaluate_test else "validation_only", snapshot, limits)
    if args.watch and not outcome["complete"]:
        raise RuntimeError(f"Plan finished with {len(outcome['failed'])} unsuccessful fits; inspect logs, "
                           "then use --retry-failed only if a new attempt is appropriate.")
    if outcome["complete"]:
        print(f"METHOD PLAN COMPLETE: {len(tasks)}/{len(tasks)} verified successful fits; "
              "no further method fits remain in this plan.", flush=True)
    return outcome


def worker(args):
    manifest_path = Path(args.worker).resolve()
    if sha256(manifest_path) != args.manifest_sha256:
        raise RuntimeError("Method manifest changed after submission")
    manifest = json.loads(manifest_path.read_text())
    if sha256(__file__) != manifest["worker_sha256"]:
        raise RuntimeError("Frozen method worker changed after submission")
    root, python = Path(manifest["root"]), Path(manifest["python"])
    if source_digest(root) != manifest["source_sha256"]:
        raise RuntimeError("Method src/configs changed after submission; refusing to train different code")
    if inventory(python) != manifest["environment"]:
        raise RuntimeError("Method Python/package inventory changed after submission")
    if manifest.get("recovery_policy") is not None:
        if sha256(Path(__file__).with_name("recovery_policy.py")) != manifest.get("recovery_helper_sha256"):
            raise RuntimeError("Frozen recovery helper changed after submission")
        helper = recovery_helper()
        policy = helper.validate_policy(manifest["recovery_policy"], root)
        if helper.policy_digest(policy) != manifest.get("recovery_policy_sha256"):
            raise RuntimeError("Frozen recovery policy digest mismatch")
    mode = manifest.get("mode")
    if mode not in {"validation_only", "test"}:
        raise RuntimeError("Unexpected method evaluation mode")
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    if not 0 <= index < len(manifest["tasks"]):
        raise RuntimeError("Invalid method array index")
    task = manifest["tasks"][index]
    sys.path.insert(0, str(root / "src"))
    print(json.dumps({"batch_id": manifest["batch_id"], "task": task, "mode": mode}), flush=True)
    output = Path(manifest["output_dir"])
    key = execution_key(task, mode)
    lock_dir = output / ".task-locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_name = hashlib.sha256(json.dumps([key, manifest["source_sha256"]]).encode()).hexdigest()
    with (lock_dir / lock_name).open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Duplicate worker for {key}; training was not started.") from error
        found = successful_results(output, [task], mode, manifest)
        returncode = 0
        if key not in found:
            command = [str(python), "-m", "imts_agents", "run", "--variant", task["variant"],
                       "--dataset", task["dataset"], "--seed", str(task["seed"]), "--device", "cuda",
                       "--output-dir", manifest["output_dir"], "--threads", "4"]
            if mode == "test":
                command.append("--evaluate-test")
            env = dict(os.environ)
            env["PYTHONPATH"] = str(root / "src")
            returncode = subprocess.run(["srun", *command], cwd=root, env=env).returncode
            found = successful_results(output, [task], mode, manifest)
        success = returncode == 0 and key in found
        receipt = {"status": "success" if success else "failed", "task": task, "mode": mode,
                   "returncode": returncode, "source_sha256": manifest["source_sha256"]}
        if success:
            receipt.update(result_path=str(found[key]), result_sha256=sha256(found[key]))
        write_new(manifest_path.parent / f"task-{index}.json", receipt)
        if returncode == 0 and not success:
            print("ERROR: training exited zero but no compatible verified success record was found.", file=sys.stderr)
            return 1
        return returncode


def main(argv=None):
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    groups = None
    if "--worker" not in raw_argv:
        sys.path.insert(0, str(ROOT / "src"))
        groups = tuple(importlib.import_module("imts_agents.variants").GROUPS) + ("motif",)
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="show the plan without Slurm queries (default)")
    mode.add_argument("--submit", action="store_true", help="submit all tasks that fit current headroom")
    mode.add_argument("--test-only", action="store_true", help="Slurm preflight without submitting or writing a ledger")
    parser.add_argument("--group", choices=groups, default="core")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--count", type=pilot_count, help="optional variant slice; default all variants in the group")
    datasets = parser.add_mutually_exclusive_group()
    datasets.add_argument("--dataset", default="JapaneseVowels")
    datasets.add_argument("--datasets", nargs="+")
    datasets.add_argument("--all-datasets", action="store_true")
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int, default=0)
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--output-dir", default="results/method-pilot")
    parser.add_argument("--time", type=parse_walltime, default="03:00:00")
    parser.add_argument("--retry-failed", action="store_true", help="explicitly retry old unsuccessful tasks once")
    parser.add_argument("--recovery-policy", type=Path,
                        help="explicit audited Geolife source-compatibility policy; default remains strict")
    parser.add_argument("--evaluate-test", action="store_true", help="evaluate official test using fixed configurations")
    parser.add_argument("--watch", action="store_true", help="refill slots as tasks finish, without a whole-array barrier")
    parser.add_argument("--poll-seconds", type=positive_int, default=30)
    parser.add_argument("--max-user-jobs", type=positive_int, default=HARD_CAP)
    parser.add_argument("--reserve-other-jobs", type=int, default=0,
                        help="share quota with another submitter; default no reservation")
    parser.add_argument("--baseline-root", type=Path,
                        help="reserve two slots only while the old two-GPU baseline watcher here is alive")
    parser.add_argument("--staged", action="store_true",
                        help="3 datasets x seed0 validation gate, then all18 x seeds0,1,2; needs --evaluate-test")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        return worker(args)
    if not 1 <= args.max_user_jobs <= HARD_CAP:
        parser.error("--max-user-jobs must be between 1 and the authorized cap of 60")
    if not 0 <= args.reserve_other_jobs < args.max_user_jobs:
        parser.error("--reserve-other-jobs must be nonnegative and below --max-user-jobs")
    if args.poll_seconds > 60:
        parser.error("--poll-seconds must be at most 60")
    if args.staged and (args.group != "ablation" or not args.evaluate_test or args.count is not None or args.start_index):
        parser.error("--staged requires --group ablation --evaluate-test and the entire variant group")
    if args.staged and args.submit and not args.watch:
        parser.error("--staged --submit requires --watch for automatic stage progression")
    tasks, total = select_tasks(args, ROOT, staged_full=args.staged)
    if args.staged:
        pilot_tasks, _ = select_tasks(args, ROOT, staged_pilot=True)
        print(f"Stage 1: {len(pilot_tasks)} validation-only pilot fits ({', '.join(PILOT_DATASETS)}; seed0).")
        print(f"Stage 2: {len(tasks)} fresh test fits, all 18 datasets x 3 seeds x {total} variants, only after all pilot checks pass.")
        print(f"Total new fits if nothing is reusable: {len(pilot_tasks) + len(tasks)}. No score-based pilot selection.")
    else:
        print(f"Method plan: {len(tasks)} fits; {total} variants in group {args.group}; "
              f"{'official test' if args.evaluate_test else 'validation only'}.")
    print("Variants: " + ", ".join(dict.fromkeys(task["variant"] for task in tasks)))
    print(f"Per task: 1 GPU, 4 CPU cores, 32 GiB; time {args.time}; global running+pending cap {args.max_user_jobs}.")
    print("Rolling refill: " + ("enabled" if args.watch else "disabled; one available chunk only"))
    if args.baseline_root:
        print("Baseline coordination: reserve its 2 slots while its verified old watcher is alive; release automatically afterward.")
    if args.test_only:
        test_only(args, ROOT, tasks)
    elif args.submit:
        submit(args, ROOT, tasks)
    else:
        print("DRY RUN: no Slurm queries or jobs. --submit submits current headroom; --watch refills automatically.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
