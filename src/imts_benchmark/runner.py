"""One fresh fit per run, isolated outputs, explicit success/failure records."""
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from .data import load_data, make_synthetic, subset_for_smoke
from .metrics import evaluate
from .models import build_model, model_settings, model_source, normalize_name
from .provenance import environment, write_json, verify_upstream
from .training import checked_training


def execute(*, model_name, dataset, seed=0, smoke=False, device="cpu", output_dir="results",
            cache_dir=None, max_dense_gib=8.0):
    model_name = normalize_name(model_name)
    if dataset == "synthetic" and not smoke:
        raise ValueError("Synthetic data are only allowed in smoke mode")
    profile = "smoke" if smoke else "benchmark"
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    run_dir = Path(output_dir) / profile / dataset / model_name / f"seed-{seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1, "run_id": run_id, "status": "running", "profile": profile,
        "dataset": dataset, "model": model_name, "seed": seed, "requested_device": device,
        "started_at": datetime.now(timezone.utc).isoformat(), "model_source": model_source(model_name),
        "environment": environment(),
        "thread_limits": {key: os.environ.get(key) for key in
                          ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS")},
    }
    write_json(run_dir / "result.json", result)
    print(f"Run output: {run_dir}", flush=True)
    wall_start = time.perf_counter()
    try:
        result["upstream_verification"] = verify_upstream()
        data = make_synthetic() if dataset == "synthetic" else load_data(dataset, cache_dir, max_dense_gib)
        if smoke:
            data = subset_for_smoke(data, seed=seed)
        result["data"] = data.metadata
        result["shapes"] = {"train": list(data.X_train.shape), "test": list(data.X_test.shape)}
        result["class_counts"] = {
            split: {str(k): int(n) for k, n in zip(*np.unique(labels, return_counts=True))}
            for split, labels in (("train", data.y_train), ("test", data.y_test))
        }
        write_json(run_dir / "split_ids.json", {
            "train": np.asarray(data.train_ids).astype(str).tolist(),
            "test": np.asarray(data.test_ids).astype(str).tolist(),
        })
        # Each call constructs a new estimator; no weights or fitted transforms reused.
        model = build_model(model_name, seed=seed, smoke=smoke, device=device)
        if model_name in {"TIMESNET", "BRITS"}:
            # Log the deterministic internal split made by the official wrapper.
            # Stratifying encoded labels yields the same partition as original labels.
            from sklearn.model_selection import train_test_split
            try:
                fitting_ids, validation_ids = train_test_split(data.train_ids, stratify=data.y_train, random_state=seed)
                stratified = True
            except ValueError:
                fitting_ids, validation_ids = train_test_split(data.train_ids, random_state=seed)
                stratified = False
            write_json(run_dir / "validation_ids.json", {
                "training": np.asarray(fitting_ids).astype(str).tolist(),
                "validation": np.asarray(validation_ids).astype(str).tolist(),
                "stratified": stratified,
                "origin": "reconstructed using the official wrapper's deterministic split and seed",
            })
        result["parameters"] = model_settings(model)
        write_json(run_dir / "result.json", result)
        with (run_dir / "training.log").open("w", buffering=1) as log:
            # Python prints are captured here; library stderr is retained by Slurm/stdout.
            with contextlib.redirect_stdout(log):
                t = time.perf_counter()
                with checked_training(model_name in {"TIMESNET", "BRITS"}, log):
                    model.fit(data.X_train, data.y_train)
                result["fit_seconds"] = time.perf_counter() - t
                t = time.perf_counter()
                probabilities = np.asarray(model.predict_proba(data.X_test))
                result["predict_seconds"] = time.perf_counter() - t
        classes = np.asarray(model.classes_)
        scores, predictions = evaluate(data.y_test, probabilities, classes)
        # Store string labels safely, without an object array requiring allow_pickle.
        np.savez_compressed(run_dir / "predictions.npz", y_true=data.y_test.astype(str),
                            y_pred=predictions.astype(str), classes=classes.astype(str),
                            probabilities=probabilities, sample_ids=np.asarray(data.test_ids).astype(str))
        # Re-read persisted predictions and check that score reconstruction is identical.
        with np.load(run_dir / "predictions.npz", allow_pickle=False) as saved:
            restored, _ = evaluate(saved["y_true"], saved["probabilities"], saved["classes"])
        if any(not np.isclose(restored[key], value, rtol=0, atol=1e-12) for key, value in scores.items()):
            raise RuntimeError("Persisted predictions did not reproduce the metrics")
        result.update(status="success", metrics=scores)
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        result["total_seconds"] = time.perf_counter() - wall_start
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(run_dir / "result.json", result)
    print(json.dumps({"status": result["status"], "profile": profile, "metrics": result["metrics"],
                      "fit_seconds": result["fit_seconds"], "output": str(run_dir)}, ensure_ascii=False), flush=True)
    return result, run_dir
