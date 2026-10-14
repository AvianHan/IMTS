"""Verify upgraded scheduling never moves baseline training into the method tree."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("parallel_wrapper_test", REPO / "scripts/submit_parallel_baselines.py")
wrapper = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wrapper)
ENVIRONMENT = {"python": "3.11", "packages": {"numpy": "1.26.4"}}


@pytest.fixture
def baseline(tmp_path):
    root = tmp_path / "baseline"
    for name in ("configs", "slurm"):
        shutil.copytree(REPO / name, root / name)
    (root / "src/imts_benchmark").mkdir(parents=True)
    (root / "src/imts_benchmark/provenance.py").write_text("# Unchanged baseline source fixture\n")
    (root / ".venv/bin").mkdir(parents=True)
    (root / ".venv/bin/python").symlink_to(sys.executable)
    return root


def test_wrapper_targets_baseline_and_forwards_args_without_side_effects(baseline, monkeypatch):
    previous = sys.argv
    seen = {}
    fake = SimpleNamespace(__file__=str(REPO / "scripts/submit_benchmarks.py"), ROOT=REPO)
    def main():
        seen["root"] = fake.ROOT
        seen["args"] = sys.argv[1:]
        return 7
    fake.main = main
    monkeypatch.setattr(wrapper, "load_launcher", lambda: fake)
    assert wrapper.main(["--baseline-root", str(baseline), "--total-batch-size", "56", "--max-user-jobs", "60"]) == 7
    assert seen["root"] == baseline.resolve()
    assert seen["args"] == ["--total-batch-size", "56", "--max-user-jobs", "60"]
    assert sys.argv is previous
    assert not (baseline / "results").exists()


def test_same_checkout_and_missing_baseline_are_rejected(baseline, monkeypatch):
    monkeypatch.setattr(wrapper, "LAUNCHER_ROOT", baseline)
    with pytest.raises(ValueError, match="separate unchanged"):
        wrapper.validate_baseline_root(baseline)
    with pytest.raises(ValueError, match="Incomplete baseline"):
        wrapper.validate_baseline_root(baseline / "missing")


def test_forwarded_argv_is_restored_on_error():
    previous = sys.argv
    with pytest.raises(RuntimeError):
        with wrapper.forwarded_arguments("launcher.py", ["--submit"]):
            raise RuntimeError("submission failed")
    assert sys.argv is previous


def test_real_launcher_freezes_new_worker_but_baseline_training_and_guards(baseline, tmp_path, monkeypatch):
    launcher = wrapper.load_launcher()
    monkeypatch.setattr(wrapper, "load_launcher", lambda: launcher)
    monkeypatch.setattr(launcher, "SHARED_SUBMISSION_LOCK", tmp_path / "submission.lock")
    inventory_paths = []
    def inventory(python):
        inventory_paths.append(Path(python))
        return ENVIRONMENT
    monkeypatch.setattr(launcher, "inventory", inventory)
    monkeypatch.setattr(launcher, "active_user_jobs", lambda: set())
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        assert command[0] == "sbatch"
        assert f"--chdir={baseline}" in command
        return subprocess.CompletedProcess(command, 0, f"{800 + len(calls)}\n", "")
    monkeypatch.setattr(launcher.subprocess, "run", run)
    before = launcher.source_digest(baseline)
    wrapper.main(["--baseline-root", str(baseline), "--submit", "--datasets", "JapaneseVowels",
                  "--cpu-batch-size", "1", "--gpu-batch-size", "1", "--total-batch-size", "2",
                  "--max-user-jobs", "60", "--gpu-time", "03:00:00"])
    folder = next((baseline / "results/accounting/batches").iterdir())
    manifest_path = folder / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert Path(manifest["root"]) == baseline
    assert Path(manifest["python"]) == baseline / ".venv/bin/python"
    assert Path(manifest["output_root"]) == baseline / "results/jv-first-full"
    assert manifest["source_sha256"] == before == launcher.source_digest(baseline)
    assert manifest["environment"] == ENVIRONMENT
    assert set(inventory_paths) == {baseline / ".venv/bin/python"}
    assert (folder / "worker.py").read_bytes() == (REPO / "scripts/submit_benchmarks.py").read_bytes()
    assert Path(manifest["templates"]["gpu"]["path"]).read_bytes() == (baseline / "slurm/benchmark_gpu.sbatch").read_bytes()
    assert "--time=03:00:00" in calls[1]

    # A changed baseline source must be rejected even though the scheduling script
    # and its default ROOT came from a different checkout.
    (baseline / "src/imts_benchmark/provenance.py").write_text("# Accidental baseline edit\n")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "0")
    arguments = SimpleNamespace(worker=str(manifest_path), manifest_sha256=launcher.digest(manifest_path), group="gpu")
    with pytest.raises(RuntimeError, match="src/configs changed"):
        launcher.worker(arguments)
    assert len(calls) == 2  # No cache preparation or training subprocess ran.
