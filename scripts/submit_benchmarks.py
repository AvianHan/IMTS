#!/usr/bin/env python3
"""Plan or submit bounded Slurm arrays, reusing compatible successful runs.

The planner uses only Python's standard library. A worker prepares the pinned
dataset cache, then delegates training to the existing benchmark sbatch script.
No jobs are submitted unless --submit is supplied. See docs/orion.md.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import fcntl
import getpass
import hashlib
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
GPU_MODELS = {"TIMESNET", "BRITS"}
MAX_ACCOUNTING_POLLS = 5
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "NODE_FAIL",
            "OUT_OF_MEMORY", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}
INVENTORY_CODE = """
import importlib.metadata, json, platform
packages = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
            if d.metadata['Name']}
print(json.dumps({'python': platform.python_version(), 'packages': packages}))
"""
CACHE_CODE = """
import fcntl, json, pathlib, sys
from pyrregular import repository
spec, revision = json.loads(sys.argv[1]), sys.argv[2]
if repository.DATA_REVISION != revision:
    raise RuntimeError('Dataset revision differs from the batch manifest')
registered = repository.REPOSITORY.registry[spec['file']].removeprefix('sha256:')
if registered != spec['sha256']:
    raise RuntimeError('Dataset registry hash differs from the batch manifest')
cache = pathlib.Path(repository.REPOSITORY.path)
cache.mkdir(parents=True, exist_ok=True)
with (cache / ('.imts-download-' + spec['file'] + '.lock')).open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    repository.download_dataset_from_huggingface(spec['file'])
"""


class AccountingPending(RuntimeError):
    """An array left the queue, but complete accounting is not visible yet."""


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Ledger records are append-only: never overwrite an earlier receipt."""
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_digest(root):
    # Keep this identical to src/imts_benchmark/provenance.py, without importing ML.
    h = hashlib.sha256()
    for folder in (root / "src", root / "configs"):
        for path in sorted(folder.rglob("*")):
            if path.suffix in {".py", ".json"}:
                h.update(str(path.relative_to(root)).encode())
                h.update(path.read_bytes())
    return h.hexdigest()


def normalized_packages(packages):
    return {name.lower().replace("_", "-").replace(".", "-"): version
            for name, version in packages.items()}


def inventory(python):
    result = subprocess.run([str(python), "-c", INVENTORY_CODE], text=True,
                            capture_output=True, check=True)
    info = json.loads(result.stdout)
    info["packages"] = normalized_packages(info["packages"])
    return info


def task_key(task):
    return task["dataset"], task["model"], task["seed"]


def select_batch(tasks, args):
    """Limit submitted tasks, independently of the arrays' running limits."""
    limits = {group: getattr(args, f"{group}_batch_size", default)
              for group, default in (("cpu", 4), ("gpu", 2))}
    if any(not isinstance(value, int) or isinstance(value, bool) or value < 1
           for value in limits.values()):
        raise ValueError("Batch sizes must be positive integers")
    counts = {"cpu": 0, "gpu": 0}
    selected = []
    for task in tasks:
        group = "gpu" if task["device"] == "cuda" else "cpu"
        if counts[group] < limits[group]:
            selected.append(task)
            counts[group] += 1
    return selected


def known_submit_rejection(receipt):
    """Recognize only the observed, unequivocal QoS submit-limit refusal.

    Older launchers saved this response as ``unconfirmed``. Its immutable
    receipt can be interpreted safely without changing or deleting the record.
    A timeout, malformed success response, or any job ID remains uncertain.
    """
    code = receipt.get("returncode")
    if (not isinstance(code, (int, float)) or isinstance(code, bool)
            or not math.isfinite(code) or code == 0
            or receipt.get("status") == "submitted"):
        return False
    if any(not isinstance(receipt.get(name), str) or receipt[name].strip()
           for name in ("stdout", "job_id")):
        return False
    error = receipt.get("stderr")
    return (isinstance(error, str)
            and re.search(r"^sbatch: error: QOSMaxSubmitJobPerUserLimit\s*$", error, re.M) is not None
            and re.search(r"^sbatch: error: Batch job submission failed: .+$", error, re.M) is not None)


@contextlib.contextmanager
def file_lock(path, blocking=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise RuntimeError(f"Another submission/worker holds {path}; try later") from exc
        yield


def successful_tasks(output, tasks, source, env, revision):
    """Never silently reuse a score from another experiment or hide duplicates."""
    selected = {task_key(task): task for task in tasks}
    successes, running = {}, {}
    for path in sorted((output / "benchmark").glob("*/*/seed-*/*/result.json")):
        record = read_json(path)
        if record.get("profile") != "benchmark":
            continue
        key = task_key(record)
        if key not in selected:
            continue
        if record.get("status") == "running":
            running[key] = str(path)
        if record.get("status") != "success":
            continue
        if key in successes:
            raise RuntimeError(f"Duplicate successful run for {key}: {successes[key]} and {path}")
        previous = record.get("environment", {})
        expected_device = selected[key]["device"]
        checks = {
            "source hash": previous.get("source_sha256") == source,
            "Python version": previous.get("python") == env["python"],
            "package inventory": normalized_packages(previous.get("packages", {})) == env["packages"],
            "device": record.get("requested_device") == expected_device,
            "full dataset": record.get("data", {}).get("smoke_subset") is False,
            "data revision": record.get("data", {}).get("data_revision") == revision,
            "predictions file": (path.parent / "predictions.npz").is_file(),
            "thread limits": all(record.get("thread_limits", {}).get(name) == "4" for name in
                                 ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")),
        }
        mismatches = [name for name, matches in checks.items() if not matches]
        if mismatches:
            raise RuntimeError(f"Cannot reuse {path}: incompatible {', '.join(mismatches)}. "
                               "Keep this experiment unchanged or choose a separate output root.")
        successes[key] = str(path)
    return successes, running


def array_is_terminal(job_id, task_count):
    """Accounting can lag; an absent squeue entry alone never proves completion."""
    # Listing this user's queue avoids squeue's invalid-ID error for finished jobs.
    queue = subprocess.run(["squeue", "--noheader", "--user", getpass.getuser(), "--format=%i|%T"],
                           text=True, capture_output=True)
    if queue.returncode:
        raise RuntimeError(f"Cannot check live Slurm jobs: {queue.stderr.strip()}")
    queue_ids = [line.split("|")[0].strip() for line in queue.stdout.splitlines()]
    if any(value == job_id or value.startswith(job_id + "_") for value in queue_ids):
        return False
    accounting = subprocess.run(["sacct", "--noheader", "--array", "--allocations", "--parsable2",
                                 "--jobs", job_id, "--format=JobID,State"],
                                text=True, capture_output=True)
    if accounting.returncode:
        raise RuntimeError(f"Cannot check Slurm job {job_id}: {accounting.stderr.strip()}")
    rows = [line.split("|") for line in accounting.stdout.splitlines() if line.strip()]
    states = {row[0].strip(): row[1].split()[0].rstrip("+") for row in rows if len(row) >= 2}
    expected = {f"{job_id}_{index}" for index in range(task_count)}
    if not expected <= states.keys():
        raise AccountingPending(f"Job {job_id} lacks complete per-task accounting evidence; "
                                "refusing duplicate submission. Check its receipt and Slurm manually.")
    return all(states[key] in TERMINAL for key in expected)


def check_previous_batches(ledger, output, pending):
    """Return keys covered by finished managed arrays; reject live/uncertain ones."""
    wanted = {task_key(task) for task in pending}
    finished = set()
    for manifest_path in sorted(ledger.glob("*/manifest.json")):
        manifest = read_json(manifest_path)
        if Path(manifest["output_root"]) != output:
            continue
        for group, tasks in manifest["tasks"].items():
            overlap = wanted & {task_key(task) for task in tasks}
            if not overlap:
                continue
            folder = manifest_path.parent
            intent, receipt = folder / f"{group}.intent.json", folder / f"{group}.receipt.json"
            if not intent.exists():
                continue  # This group's sbatch was never attempted.
            if not receipt.exists():
                raise RuntimeError(f"Submission outcome is unknown: {intent}. "
                                   "Inspect Slurm using the recorded job name before any retry.")
            saved = read_json(receipt)
            if known_submit_rejection(saved):
                continue  # A recorded refusal did not create a Slurm array.
            if saved["status"] != "submitted":
                raise RuntimeError(f"Unconfirmed submission: {receipt}. Inspect Slurm before retrying.")
            if not array_is_terminal(saved["job_id"], len(tasks)):
                raise RuntimeError(f"Array {saved['job_id']} is still active and covers pending tasks. "
                                   "Wait for it to finish; do not resubmit the same batch.")
            finished.update(overlap)
    return finished


def template_text(root, group):
    text = (root / "slurm" / f"benchmark_{group}.sbatch").read_text()
    if "#SBATCH --cpus-per-task=4" not in text:
        raise RuntimeError("This launcher's experiment requires the existing four-CPU templates")
    return text


def submit(args, root, output, ledger, tasks, source, env, datasets, revision, skipped):
    tasks = select_batch(tasks, args)
    if not tasks:
        raise ValueError("Cannot submit an empty batch")
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = ledger / batch_id
    folder.mkdir(parents=True)
    groups = {group: [t for t in tasks if (t["device"] == "cuda") == (group == "gpu")]
              for group in ("cpu", "gpu")}
    # Frozen worker/template files make queued tasks independent of later script edits.
    worker_path = folder / "worker.py"
    worker_path.write_bytes(Path(__file__).read_bytes())
    templates = {}
    for group in groups:
        path = folder / f"{group}.template.sbatch"
        path.write_text(template_text(root, group))
        templates[group] = {"path": str(path), "sha256": digest(path)}
    manifest = {
        "schema_version": 1, "batch_id": batch_id,
        "created_at": datetime.now(timezone.utc).isoformat(), "root": str(root),
        "output_root": str(output), "python": str(root / ".venv/bin/python"),
        "source_sha256": source, "environment": env, "data_revision": revision,
        "datasets": datasets, "tasks": groups, "skipped_successes": list(skipped.values()),
        "worker_sha256": digest(worker_path), "templates": templates,
        "max_dense_gib": args.max_dense_gib,
        "resources": {group: {"concurrency": getattr(args, f"{group}_concurrency"),
                              "time_limit": getattr(args, f"{group}_time"),
                              "cpus_per_task": 4} for group in groups},
    }
    manifest_path = folder / "manifest.json"
    write_new(manifest_path, manifest)
    manifest_hash = digest(manifest_path)
    submitted = []
    for group, group_tasks in groups.items():
        if not group_tasks:
            continue
        headers = [line for line in Path(templates[group]["path"]).read_text().splitlines()
                   if line.startswith("#SBATCH")]
        command = [str(root / ".venv/bin/python"), str(worker_path), "--worker",
                   str(manifest_path), "--manifest-sha256", manifest_hash, "--group", group]
        wrapper = folder / f"{group}.sbatch"
        wrapper.write_text("#!/usr/bin/env bash\n" + "\n".join(headers) +
                           "\nset -euo pipefail\nexec " + shlex.join(command) + "\n")
        job_name = f"imts-{group}-{batch_id}"
        request = ["sbatch", "--parsable", "--export=ALL", "--no-requeue", f"--chdir={root}", f"--job-name={job_name}",
                   f"--array=0-{len(group_tasks) - 1}%{getattr(args, group + '_concurrency')}",
                   f"--time={getattr(args, group + '_time')}",
                   f"--output={folder}/slurm-%A_%a.out", str(wrapper)]
        write_new(folder / f"{group}.intent.json", {
            "command": request, "job_name": job_name, "manifest_sha256": manifest_hash,
        })
        # SBATCH_* environment overrides script directives; use exactly the recorded request.
        clean_env = {key: value for key, value in os.environ.items() if not key.startswith("SBATCH_")}
        result = subprocess.run(request, text=True, capture_output=True, env=clean_env)
        job_id = result.stdout.strip().split(";")[0]
        confirmed = result.returncode == 0 and re.fullmatch(r"[0-9]+", job_id)
        receipt = {"status": "submitted" if confirmed else "unconfirmed", "job_id": job_id,
                   "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
                   "manifest_sha256": manifest_hash}
        rejected = known_submit_rejection(receipt)
        if rejected:
            receipt["status"] = "rejected"
        write_new(folder / f"{group}.receipt.json", receipt)
        if not confirmed:
            outcome = "was rejected by Slurm" if rejected else "has an unconfirmed outcome"
            earlier = (f"Earlier confirmed arrays: {', '.join(job for job, _ in submitted)}. "
                       "They were not cancelled; check their current state before another submission."
                       if submitted else "No earlier array was confirmed in this batch.")
            raise RuntimeError(f"{group.upper()} submission {outcome}. "
                               f"Receipt: {folder / (group + '.receipt.json')}\n"
                               f"Slurm stderr: {result.stderr.strip() or '(empty)'}\n"
                               f"{earlier} No automatic retry was made.")
        submitted.append((job_id, len(group_tasks)))
        print(f"Submitted {group.upper()} array {job_id}: {len(group_tasks)} tasks, "
              f"at most {getattr(args, group + '_concurrency')} running", flush=True)
    print(f"Manifest and receipts: {folder}", flush=True)
    return folder, submitted


def wait_for_arrays(jobs, poll_seconds):
    """Wait for arrays, tolerating brief accounting lag but never query errors."""
    waiting = list(jobs)
    accounting_polls = {}
    while waiting:
        next_waiting = []
        for job, count in waiting:
            try:
                terminal = array_is_terminal(job, count)
                accounting_polls.pop(job, None)
            except AccountingPending as exc:
                accounting_polls[job] = accounting_polls.get(job, 0) + 1
                if accounting_polls[job] >= MAX_ACCOUNTING_POLLS:
                    raise AccountingPending(f"Job {job} still lacks complete accounting after "
                                            f"{MAX_ACCOUNTING_POLLS} polls; --watch stopped. "
                                            "Check its receipt and Slurm before another submission.") from exc
                terminal = False
                print(f"Waiting for accounting for array {job} "
                      f"({accounting_polls[job]}/{MAX_ACCOUNTING_POLLS} checks)", flush=True)
            if not terminal:
                next_waiting.append((job, count))
        waiting = next_waiting
        if not waiting:
            break
        print(f"Waiting for arrays: {', '.join(job for job, _ in waiting)}; "
              f"checking again in {poll_seconds} seconds", flush=True)
        remaining_delay = poll_seconds
        while remaining_delay > 0:
            delay = min(remaining_delay, 60)
            time.sleep(delay)
            remaining_delay -= delay


def run_batches(args, root, output, ledger, tasks, source, env, datasets,
                revision, skipped, pending):
    """Submit one bounded batch, or watch sequential batches until attempted.

    The caller holds the submission lock throughout. Failed training is never
    retried within this invocation; fresh invocations may explicitly retry it.
    """
    if not args.watch:
        submit(args, root, output, ledger, pending, source, env, datasets, revision, skipped)
        return 0
    attempted = set()
    wave = 0
    while True:
        if source_digest(root) != source:
            raise RuntimeError("src/configs changed during --watch; stopped before the next batch. "
                               "Restore the recorded version before continuing this experiment.")
        if inventory(root / ".venv/bin/python") != env:
            raise RuntimeError("Python/package inventory changed during --watch; "
                               "stopped before the next batch.")
        skipped, running = successful_tasks(output, tasks, source, env, revision)
        pending = [task for task in tasks if task_key(task) not in skipped
                   and task_key(task) not in attempted]
        if not pending:
            unsuccessful = attempted - skipped.keys()
            print(f"Watch finished: {len(skipped)}/{len(tasks)} compatible successes; "
                  f"{len(unsuccessful)} attempted tasks without a compatible success.", flush=True)
            if unsuccessful:
                print("These tasks were not retried; inspect their logs before a new submission:", flush=True)
                for dataset, model, seed in sorted(unsuccessful):
                    print(f"  {dataset} / {model} / seed {seed}", flush=True)
            return 1 if unsuccessful else 0
        finished_keys = check_previous_batches(ledger, output, pending)
        unmanaged = ({task_key(task) for task in pending} & running.keys()) - finished_keys
        if unmanaged:
            raise RuntimeError(f"Unresolved running records: {[running[key] for key in sorted(unmanaged)]}. "
                               "Check their jobs before resubmitting; --watch stopped safely.")
        batch = select_batch(pending, args)
        wave += 1
        print(f"Wave {wave}: submitting {len(batch)} tasks; "
              f"{len(pending) - len(batch)} still deferred", flush=True)
        _, jobs = submit(args, root, output, ledger, batch, source, env, datasets, revision, skipped)
        attempted.update(task_key(task) for task in batch)
        wait_for_arrays(jobs, args.poll_seconds)


def worker(args):
    path = Path(args.worker).resolve()
    if digest(path) != args.manifest_sha256:
        raise RuntimeError("Batch manifest was modified after submission")
    manifest = read_json(path)
    if digest(__file__) != manifest["worker_sha256"]:
        raise RuntimeError("Frozen worker script was modified after submission")
    root, output = Path(manifest["root"]), Path(manifest["output_root"])
    index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    tasks = manifest["tasks"][args.group]
    if not 0 <= index < len(tasks):
        raise RuntimeError(f"Invalid array task index {index}")
    task = tasks[index]
    key_id = hashlib.sha256(json.dumps(task_key(task)).encode()).hexdigest()
    lock_path = output / "accounting/task-locks" / (key_id + ".lock")
    with file_lock(lock_path, blocking=True):
        if source_digest(root) != manifest["source_sha256"]:
            raise RuntimeError("src/configs changed while jobs were queued; restore the recorded version")
        env = inventory(manifest["python"])
        if env != manifest["environment"]:
            raise RuntimeError("Python/package inventory changed while jobs were queued")
        successes, _ = successful_tasks(output, [task], manifest["source_sha256"], env,
                                        manifest["data_revision"])
        if task_key(task) in successes:
            print(f"Already successful; no duplicate fit: {task_key(task)}", flush=True)
            return 0
        template = manifest["templates"][args.group]
        if digest(template["path"]) != template["sha256"]:
            raise RuntimeError("Frozen benchmark template was modified")
        if os.environ.get("SLURM_CPUS_PER_TASK") != "4":
            raise RuntimeError("Expected the recorded allocation of four CPUs per task")
        print(json.dumps({"batch_id": manifest["batch_id"], "task": task,
                          "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                          "slurm_array_task_id": index}), flush=True)
        started = time.perf_counter()
        run_env = dict(os.environ, IMTS_MODEL=task["model"], IMTS_DATASET=task["dataset"],
                       IMTS_SEED=str(task["seed"]), IMTS_OUTPUT_DIR=str(output),
                       IMTS_MAX_DENSE_GIB=str(manifest["max_dense_gib"]),
                       OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", OPENBLAS_NUM_THREADS="4", NUMBA_NUM_THREADS="4")
        subprocess.run([manifest["python"], "-c", CACHE_CODE,
                        json.dumps(manifest["datasets"][task["dataset"]]), manifest["data_revision"]],
                       cwd=root, env=run_env, check=True)
        print(f"Cache preparation (including lock wait): {time.perf_counter() - started:.3f} seconds", flush=True)
        return subprocess.run(["bash", template["path"]], cwd=root, env=run_env).returncode


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return number


def poll_interval(value):
    number = int(value)
    if number < 30:
        raise argparse.ArgumentTypeError("must be at least 30 seconds")
    return number


def time_limit(value):
    if not re.fullmatch(r"(?:[0-9]+-)?[0-9]{1,3}:[0-5][0-9]:[0-5][0-9]", value):
        raise argparse.ArgumentTypeError("use HH:MM:SS or D-HH:MM:SS")
    if not any(int(part) for part in re.split(r"[-:]", value)):
        raise argparse.ArgumentTypeError("time limit must be positive")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit", action="store_true", help="actually submit; default is a read-only dry run")
    parser.add_argument("--watch", action="store_true",
                        help="with --submit, wait for each bounded batch then submit the next; keep this process running")
    parser.add_argument("--poll-seconds", type=poll_interval, default=60,
                        help="Slurm polling interval in watch mode (minimum 30; default 60)")
    parser.add_argument("--datasets", nargs="+", help="optional configured dataset names")
    parser.add_argument("--output-root", default="results/jv-first-full")
    parser.add_argument("--cpu-concurrency", type=positive_int, default=4)
    parser.add_argument("--gpu-concurrency", type=positive_int, default=2)
    parser.add_argument("--cpu-batch-size", type=positive_int, default=4,
                        help="maximum CPU tasks submitted per wave (default 4)")
    parser.add_argument("--gpu-batch-size", type=positive_int, default=2,
                        help="maximum GPU tasks submitted per wave (default 2)")
    parser.add_argument("--cpu-time", type=time_limit, default="02:00:00")
    parser.add_argument("--gpu-time", type=time_limit, default="01:00:00")
    parser.add_argument("--max-dense-gib", type=float, default=8.0)
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--manifest-sha256", help=argparse.SUPPRESS)
    parser.add_argument("--group", choices=("cpu", "gpu"), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    if args.watch and not args.submit:
        parser.error("--watch requires --submit")
    if not math.isfinite(args.max_dense_gib) or args.max_dense_gib <= 0:
        parser.error("--max-dense-gib must be finite and positive")
    root = ROOT
    output = Path(args.output_root).expanduser()
    output = (root / output).resolve() if not output.is_absolute() else output.resolve()
    ledger = root / "results/accounting/batches"
    data_config = read_json(root / "configs/datasets.json")
    model_config = read_json(root / "configs/baselines.json")
    datasets = {item["name"]: item for item in data_config["datasets"]}
    if args.datasets and set(args.datasets) - datasets.keys():
        parser.error(f"Unknown datasets: {sorted(set(args.datasets) - datasets.keys())}")
    chosen = [name for name in datasets if not args.datasets or name in args.datasets]
    tasks = [{"dataset": name, "model": model, "seed": seed,
              "device": "cuda" if model in GPU_MODELS else "cpu"}
             for name in chosen for model, config in model_config["models"].items() for seed in config["seeds"]]
    lock = file_lock(ledger.parent / ".submit.lock") if args.submit else contextlib.nullcontext()
    with lock:
        source = source_digest(root)
        env = inventory(root / ".venv/bin/python")
        skipped, running = successful_tasks(output, tasks, source, env, data_config["data_revision"])
        pending = [task for task in tasks if task_key(task) not in skipped]
        finished_keys = check_previous_batches(ledger, output, pending)
        unmanaged = {task_key(task) for task in pending} & running.keys() - finished_keys
        if unmanaged:
            raise RuntimeError(f"Unresolved running records: {[running[key] for key in sorted(unmanaged)]}. "
                               "Check their jobs before resubmitting; no automatic retry was made.")
        counts = {group: sum((task["device"] == "cuda") == (group == "gpu") for task in pending)
                  for group in ("cpu", "gpu")}
        batch = select_batch(pending, args)
        batch_counts = {group: sum((task["device"] == "cuda") == (group == "gpu") for task in batch)
                        for group in ("cpu", "gpu")}
        print(f"Datasets: {len(chosen)} | Planned fits: {len(tasks)} | Compatible successes: {len(skipped)}", flush=True)
        print(f"Remaining: {len(pending)} = CPU {counts['cpu']} + GPU {counts['gpu']}", flush=True)
        print(f"Next batch: {len(batch)} = CPU {batch_counts['cpu']} + GPU {batch_counts['gpu']}", flush=True)
        print(f"Deferred: {len(pending) - len(batch)}", flush=True)
        print(f"Concurrency: CPU {args.cpu_concurrency}, GPU {args.gpu_concurrency}; "
              f"time limits: {args.cpu_time}, {args.gpu_time}; dense-load guard: {args.max_dense_gib:g} GiB", flush=True)
        print(f"Results: {output}", flush=True)
        if not pending:
            print("All selected tasks already succeeded. No jobs submitted.", flush=True)
        elif not args.submit:
            print("DRY RUN: no jobs submitted. Add --submit for the next batch only, "
                  "or --submit --watch to continue through deferred tasks.", flush=True)
        else:
            return run_batches(args, root, output, ledger, tasks, source, env, datasets,
                               data_config["data_revision"], skipped, pending)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
