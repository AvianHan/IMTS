"""Fresh, auditable pilot fits with a validation-only default.

The official test split is never used for training, early stopping, or policy
rewards.  Evaluating it requires an explicit flag after configuration selection.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import traceback
import uuid

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from imts_benchmark.metrics import evaluate
from imts_benchmark.provenance import environment
from .data import prepare_dataset


@dataclass(frozen=True)
class RunConfig:
    variant: str = "core_both"
    dataset: str = "JapaneseVowels"
    seed: int = 0
    device: str = "cpu"
    epochs: int = 100
    patience: int = 15
    batch_size: int = 32
    hidden: int = 32
    n_patches: int = 8
    max_inner_steps: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    cost_weight: float = 1e-3
    policy_weight: float = 1.0
    entropy_weight: float = 1e-3
    gradient_clip: float = 5.0
    threads: int = 4
    evaluate_test: bool = False
    smoke: bool = False

    def validate(self):
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        for key in ("epochs", "patience", "batch_size", "hidden", "n_patches", "max_inner_steps", "threads"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key in ("learning_rate", "weight_decay", "cost_weight", "policy_weight", "entropy_weight", "gradient_clip"):
            value = getattr(self, key)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        if self.learning_rate == 0 or self.gradient_clip == 0:
            raise ValueError("learning_rate and gradient_clip must be positive")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.dataset == "synthetic" and not self.smoke:
            raise ValueError("Synthetic data may only be used with --smoke")
        if self.smoke and self.evaluate_test:
            raise ValueError("Smoke diagnostics must not evaluate the official test split")


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _json(value, **kwargs):
    return json.dumps(value, default=_json_default, allow_nan=False, **kwargs)


def _write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(_json(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _digest(value):
    return hashlib.sha256(_json(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _array_digest(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str((array.shape, str(array.dtype))).encode())
        digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _seed(seed, threads, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(threads)
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable; refusing a silent CPU fallback")
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.reset_peak_memory_stats()
    torch.use_deterministic_algorithms(True)


def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def _loader(data, split, config, shuffle=False):
    X, mask, labels = (np.asarray(getattr(data, f"{prefix}_{split}")) for prefix in ("X", "mask", "y"))
    if X.ndim != 4 or mask.shape != X.shape[:-1] or labels.shape != (len(X),) or len(X) == 0:
        raise ValueError(f"Invalid or empty {split} tensors")
    if not np.isfinite(X).all():
        raise ValueError(f"Non-finite features in {split}")
    if labels.dtype.kind not in "iu" or np.any(labels < 0) or np.any(labels >= len(data.classes)):
        raise ValueError(f"Invalid encoded labels in {split}")
    tensors = TensorDataset(torch.from_numpy(X.astype(np.float32, copy=False)),
                            torch.from_numpy(mask.astype(bool, copy=False)),
                            torch.from_numpy(labels.astype(np.int64, copy=False)))
    return DataLoader(tensors, batch_size=config.batch_size, shuffle=shuffle, num_workers=0,
                      generator=torch.Generator().manual_seed(config.seed))


def _role_labels(model):
    scopes = model.variant.role_scopes
    return [f"{scope}_{index}" if scopes.count(scope) > 1 else scope
            for index, scope in enumerate(scopes)]


def _gate_statistics(sums, counts, squared_sums, role_order):
    """Use valid-token weights, not a mean of per-batch or per-sample means."""
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    second_moments = np.divide(squared_sums, counts, out=np.zeros_like(sums), where=counts > 0)
    deviations = np.sqrt(np.maximum(second_moments - means ** 2, 0))
    return {"role_order": role_order, "valid_token_count": counts.astype(np.int64).tolist(),
            "mean": [float(value) if count else None for value, count in zip(means, counts)],
            "population_std": [float(value) if count else None for value, count in zip(deviations, counts)],
            "scope": "initial memory write, over valid current-patch channel tokens",
            "interpretation": "Soft write gates do not block later attention or communication and do not imply compute savings."}


def _predict(model, loader, device, n_classes):
    model.eval()
    probabilities, labels, updates, actions = [], [], [], []
    role_order = _role_labels(model)
    gate_totals = np.zeros((3, len(role_order)), dtype=np.float64)
    gates_available = None
    _sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for X, mask, y in loader:
            output = model(X.to(device), mask.to(device))
            if not torch.isfinite(output["logits"]).all():
                raise FloatingPointError("Non-finite evaluation logits")
            probabilities.append(output["logits"].softmax(-1).cpu().numpy())
            labels.append(y.numpy())
            updates.append(output["updates"].detach().cpu().numpy())
            actions.append(output["action_counts"].detach().cpu().numpy())
            gate_keys = ("input_gate_sum", "input_gate_count", "input_gate_squared_sum")
            available = all(key in output for key in gate_keys)
            if any(key in output for key in gate_keys) != available:
                raise ValueError("Incomplete input-write gate diagnostics")
            if gates_available is not None and available != gates_available:
                raise ValueError("Input-write gate diagnostics changed between batches")
            gates_available = available
            if available:
                for index, key in enumerate(gate_keys):
                    values = output[key].detach().cpu().numpy().astype(np.float64)
                    if values.shape != (len(y), len(role_order)) or not np.isfinite(values).all():
                        raise ValueError(f"Invalid {key} diagnostics")
                    if np.any(values < 0):
                        raise ValueError(f"Negative {key} diagnostics")
                    gate_totals[index] += values.sum(axis=0)
    _sync(device)
    elapsed = time.perf_counter() - started
    p, y = np.concatenate(probabilities), np.concatenate(labels)
    metrics, predictions = evaluate(y, p, np.arange(n_classes))
    action_labels = role_order.copy()
    action_labels += [f"unused_{index}" for index in range(len(action_labels), 3)]
    action_labels += ["COMMIT"]
    cost = {"mean_role_updates_per_sample": float(np.concatenate(updates).mean()),
            "mean_action_counts_per_sample": np.concatenate(actions).mean(axis=0).tolist(),
            "action_order": action_labels,
            "predict_seconds": elapsed, "samples": len(y),
            "seconds_per_sample": elapsed / len(y),
            "cost_scope": "Role-update counts exclude input writes, encoding, policy, messages and readout; they are not FLOPs."}
    if gates_available:
        cost["input_write_gate"] = _gate_statistics(*gate_totals, role_order)
    return metrics, predictions, p, y, cost


def _action_path_diagnostics(model, loader, device):
    """One best-checkpoint validation pass; retain counts rather than state traces."""
    model.eval()
    labels = _role_labels(model)
    labels += [f"unused_{index}" for index in range(len(labels), 3)]
    labels += ["COMMIT"]
    counts = Counter()
    nonempty_patches = empty_patches = committed_patches = 0
    _sync(device)
    started = time.perf_counter()
    with torch.no_grad():
        for X, mask, _ in loader:
            output = model(X.to(device), mask.to(device), return_actions=True)
            actions = output["actions"].detach().cpu().numpy()
            nonempty = mask.any(-1).numpy()
            if actions.ndim != 3 or actions.shape[:2] != nonempty.shape:
                raise ValueError("Action path diagnostics do not match patch masks")
            for patch_actions, valid in zip(actions.reshape(-1, actions.shape[-1]), nonempty.reshape(-1)):
                if not valid:
                    if np.any(patch_actions != -1):
                        raise ValueError("An empty patch executed a recorded action")
                    empty_patches += 1
                    continue
                executed = patch_actions[patch_actions != -1]
                if len(executed) == 0 or np.any((executed < 0) | (executed > 3)):
                    raise ValueError("A nonempty patch has invalid recorded actions")
                # After COMMIT, or after trace padding begins, no action may follow.
                if not np.array_equal(patch_actions[:len(executed)], executed):
                    raise ValueError("Action trace resumes after an inactive step")
                if np.any(executed[:-1] == 3):
                    raise ValueError("Action trace continues after COMMIT")
                path = tuple(labels[int(action)] for action in executed)
                counts[path] += 1
                nonempty_patches += 1
                committed_patches += int(executed[-1] == 3)
    _sync(device)
    return {"nonempty_patches": nonempty_patches, "empty_patches": empty_patches,
            "unique_paths": len(counts), "committed_patches": committed_patches,
            "commit_patch_fraction": committed_patches / nonempty_patches if nonempty_patches else None,
            "distribution": [{"actions": list(path), "count": count,
                              "fraction": count / nonempty_patches}
                             for path, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))],
            "diagnostic_seconds": time.perf_counter() - started,
            "scope": "deterministic best-checkpoint validation paths; diagnostic pass excluded from predict_seconds"}


def _save_predictions(run_dir, split, predicted, data):
    metrics, predictions, probabilities, labels, cost = predicted
    sample_ids = np.asarray(data.metadata["sample_ids"][split]).astype(str)
    if len(sample_ids) != len(labels):
        raise ValueError("Prediction sample IDs do not match the evaluated split")
    path = run_dir / f"{split}_predictions.npz"
    np.savez_compressed(path, y_true=labels, y_pred=predictions, probabilities=probabilities,
                        classes=np.arange(len(data.classes)),
                        original_classes=np.asarray(data.classes).astype(str), sample_ids=sample_ids)
    with np.load(path, allow_pickle=False) as saved:
        restored_metrics, restored_predictions = evaluate(saved["y_true"], saved["probabilities"], saved["classes"])
        if not np.array_equal(restored_predictions, saved["y_pred"]):
            raise RuntimeError("Persisted predictions failed label reconstruction")
    if any(not np.isclose(metrics[key], restored_metrics[key], atol=1e-12, rtol=0) for key in metrics):
        raise RuntimeError("Persisted probabilities failed metric reconstruction")
    return {"metrics": metrics, "cost": cost, "predictions_file": path.name,
            "predictions_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def execute(*, output_dir="results/method-pilot", cache_dir=None, **kwargs):
    """Run exactly one fresh dataset/variant/seed fit and return (result, folder)."""
    from .model import AgentClassifier
    from .variants import get_variant

    config = RunConfig(**kwargs)
    config.validate()
    spec = get_variant(config.variant)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    profile = "smoke" if config.smoke else "test" if config.evaluate_test else "validation"
    run_dir = Path(output_dir) / profile / config.dataset / config.variant / f"seed-{config.seed}" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    config_dict = asdict(config)
    result = {"schema_version": 1, "experiment_family": "imts_agents", "run_id": run_id,
              "status": "running", "profile": profile, "dataset": config.dataset,
              "variant": config.variant, "seed": config.seed, "config": config_dict,
              "config_sha256": _digest(config_dict), "variant_spec": asdict(spec),
              "started_at": datetime.now(timezone.utc).isoformat(),
              "test_evaluated": False, "formal_benchmark": False,
              "evaluation_scope": "validation used for early stopping; not an unbiased test result",
              "output": str(run_dir), "environment": environment()}
    _write_json(run_dir / "result.json", result)
    print(f"Run output: {run_dir}", flush=True)
    wall_started = time.perf_counter()
    try:
        _seed(config.seed, config.threads, config.device)
        data = prepare_dataset(config.dataset, seed=config.seed, n_patches=config.n_patches,
                               time_features=spec.time_features, cache_dir=cache_dir,
                               synthetic=config.dataset == "synthetic")
        result["data"] = data.metadata
        result["prepared_data_sha256"] = {
            split: _array_digest(*(getattr(data, f"{prefix}_{split}") for prefix in ("X", "mask", "y")))
            for split in ("train", "val", "test")}
        # A comparison protocol intentionally omits variant-specific representation
        # choices. Source, splits, labels, hyperparameters, and evaluation scope must
        # agree; variant and all feature hashes remain recorded separately.
        protocol = {key: value for key, value in config_dict.items() if key != "variant"}
        protocol["source_sha256"] = result["environment"]["source_sha256"]
        protocol["software_versions"] = {
            name: result["environment"].get("packages", {}).get(name)
            for name in ("torch", "numpy", "scikit-learn", "pyrregular")}
        protocol["data_source"] = {name: data.metadata.get(name)
                                   for name in ("source", "data_revision", "sha256", "generator_seed")}
        protocol["preprocessing_sha256"] = _digest(data.metadata.get("preprocessing", {}))
        protocol["sample_ids_sha256"] = data.metadata["sample_ids_sha256"]
        protocol["label_sha256"] = {split: _array_digest(getattr(data, f"y_{split}"))
                                      for split in ("train", "val", "test")}
        result["comparison_protocol"] = protocol
        result["comparison_protocol_sha256"] = _digest(protocol)
        _write_json(run_dir / "data_metadata.json", data.metadata)
        train_loader = _loader(data, "train", config, shuffle=True)
        val_loader = _loader(data, "val", config)
        model = AgentClassifier(n_features=data.X_train.shape[-1], n_channels=data.X_train.shape[-2],
                                n_classes=len(data.classes), variant=spec, hidden=config.hidden,
                                max_inner_steps=config.max_inner_steps).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        result["parameters"] = {"total": sum(p.numel() for p in model.parameters()),
                                 "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
        policy_names = ("local_policies.", "router_cell.", "router_head.", "step_embedding.")
        result["parameters"]["policy"] = sum(p.numel() for name, p in model.named_parameters()
                                                 if name.startswith(policy_names))
        result["hardware"] = {"device": config.device, "torch_threads": torch.get_num_threads()}
        if config.device == "cuda":
            result["hardware"].update(cuda_device_name=torch.cuda.get_device_name(),
                                        cuda_capability=list(torch.cuda.get_device_capability()))
        result["training"] = {"optimizer": "AdamW", "early_stopping_metric": "validation macro_f1",
                               "early_stopping_direction": "maximize", "selection_tie": "earliest epoch",
                               "policy_reward": "-cross_entropy.detach() - cost_weight * role_updates",
                               "policy_baseline": "previous EMA reward, decay=0.9, initial=0",
                               "validation_policy": "deterministic evaluation mode",
                               "effective_epoch_limit": 1 if config.smoke else config.epochs}
        _write_json(run_dir / "result.json", result)
        best_score, stale, reward_baseline, best_epoch = -math.inf, 0, 0.0, 0
        epoch_limit = 1 if config.smoke else config.epochs
        _sync(config.device)
        fit_started = time.perf_counter()
        with (run_dir / "history.jsonl").open("w", buffering=1) as history:
            for epoch in range(1, epoch_limit + 1):
                model.train()
                sums = {"cross_entropy": 0.0, "objective": 0.0, "policy_loss": 0.0,
                        "policy_entropy": 0.0, "role_updates": 0.0}
                sample_count = 0
                for X, mask, labels in train_loader:
                    X, mask, labels = X.to(config.device), mask.to(config.device), labels.to(config.device)
                    optimizer.zero_grad(set_to_none=True)
                    output = model(X, mask)
                    ce = nn.functional.cross_entropy(output["logits"], labels, reduction="none")
                    policy_loss = ce.new_zeros(())
                    entropy = ce.new_zeros(())
                    if spec.train_policy:
                        reward = -ce.detach() - config.cost_weight * output["updates"].detach()
                        advantage = reward - reward_baseline
                        policy_loss = -(advantage * output["policy_log_prob"]).mean()
                        entropy = output["policy_entropy"].mean()
                        reward_baseline = 0.9 * reward_baseline + 0.1 * reward.mean().item()
                    loss = ce.mean() + config.policy_weight * policy_loss - config.entropy_weight * entropy
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite training loss at epoch {epoch}")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip, error_if_nonfinite=True)
                    optimizer.step()
                    count = len(labels)
                    sample_count += count
                    for key, value in (("cross_entropy", ce.mean()), ("objective", loss),
                                       ("policy_loss", policy_loss), ("policy_entropy", entropy),
                                       ("role_updates", output["updates"].float().mean())):
                        sums[key] += value.detach().item() * count
                validation = _predict(model, val_loader, config.device, len(data.classes))
                val_metrics = validation[0]
                improved = val_metrics["macro_f1"] > best_score
                if improved:
                    best_score, best_epoch, stale = val_metrics["macro_f1"], epoch, 0
                    temporary = run_dir / "best.pt.tmp"
                    torch.save({"model_state_dict": model.state_dict(), "config": config_dict,
                                "epoch": epoch, "validation_macro_f1": best_score,
                                "source_sha256": result["environment"]["source_sha256"],
                                "comparison_protocol_sha256": result["comparison_protocol_sha256"]}, temporary)
                    temporary.replace(run_dir / "best.pt")
                else:
                    stale += 1
                row = {"epoch": epoch, "train": {k: v / sample_count for k, v in sums.items()},
                       "validation": val_metrics, "validation_cost": validation[4],
                       "checkpoint_improved": improved, "stale_epochs": stale}
                history.write(_json(row) + "\n")
                print(f"Epoch {epoch:03d}: train CE={row['train']['cross_entropy']:.5f}; "
                      f"validation macro_f1={val_metrics['macro_f1']:.5f}; best epoch={best_epoch}", flush=True)
                if stale >= config.patience:
                    break
        _sync(config.device)
        result["fit_seconds"] = time.perf_counter() - fit_started
        result["epochs_completed"] = epoch
        result["best_epoch"] = best_epoch
        result["stop_reason"] = "smoke_epoch_limit" if config.smoke else "early_stopping" if stale >= config.patience else "epoch_limit"
        checkpoint = torch.load(run_dir / "best.pt", map_location=config.device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        validation = _save_predictions(run_dir, "val", _predict(model, val_loader, config.device, len(data.classes)), data)
        result["validation"] = validation
        result["metrics"] = validation["metrics"]
        result["predict_seconds"] = validation["cost"]["predict_seconds"]
        action_paths = _action_path_diagnostics(model, val_loader, config.device)
        expected_patches = int(np.asarray(data.mask_val).any(-1).sum())
        if (action_paths["nonempty_patches"] != expected_patches
                or sum(item["count"] for item in action_paths["distribution"]) != expected_patches):
            raise RuntimeError("Action path diagnostics do not cover the validation patches")
        diagnostics = {"split": "validation", "checkpoint_epoch": best_epoch,
                       "history_access": spec.history_access,
                       "outer_recurrent": spec.outer_recurrent, "input_write": spec.input_write,
                       "routing": spec.routing, "communication": spec.communication,
                       "input_write_gate": validation["cost"].get("input_write_gate"),
                       "action_paths": action_paths,
                       "interpretation": [
                           "Validation selected this checkpoint; diagnostic patterns are exploratory, not held-out test evidence.",
                           "A repeated path does not support input-dependent routing; inspect the actual path distribution.",
                           "Initial-write gates do not restrict later attention or messages and do not imply reduced computation.",
                           "Role-update counts are not FLOPs; compare measured latency and allocated memory as well."]}
        diagnostics_path = run_dir / "diagnostics.json"
        _write_json(diagnostics_path, diagnostics)
        result["diagnostics_file"] = diagnostics_path.name
        result["diagnostics_sha256"] = hashlib.sha256(diagnostics_path.read_bytes()).hexdigest()
        result["validation_checks"] = {
            "finite_metrics": all(math.isfinite(value) for value in validation["metrics"].values()),
            "predictions_reconstructed": True, "action_paths_cover_nonempty_patches": True}
        if not result["validation_checks"]["finite_metrics"]:
            raise FloatingPointError("Non-finite validation metrics")
        if config.evaluate_test:
            test = _save_predictions(run_dir, "test", _predict(model, _loader(data, "test", config), config.device, len(data.classes)), data)
            result["test"] = test
            result["metrics"] = test["metrics"]
            result["predict_seconds"] = test["cost"]["predict_seconds"]
            result["test_evaluated"] = True
            result["evaluation_scope"] = "official test evaluated once after validation checkpoint selection"
        result["checkpoint_sha256"] = hashlib.sha256((run_dir / "best.pt").read_bytes()).hexdigest()
        result["memory"] = ({"cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                             "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved()}
                            if config.device == "cuda" else {"cpu_peak_rss_bytes": None, "note": "CPU peak RSS not measured"})
        result["status"] = "success"
    except (Exception, KeyboardInterrupt) as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        (run_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        result["total_seconds"] = time.perf_counter() - wall_started
        result["finished_at"] = datetime.now(timezone.utc).isoformat()
        _write_json(run_dir / "result.json", result)
    print(_json({"status": result["status"], "profile": profile, "variant": config.variant,
                 "metrics": result["metrics"], "output": str(run_dir)}), flush=True)
    return result, run_dir


def summarize(output_dir):
    """Keep incompatible protocols separate; preserve failures and duplicate runs."""
    groups, incomplete, invalid = {}, [], []
    for path in sorted(Path(output_dir).rglob("result.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            invalid.append(str(path))
            continue
        if record.get("experiment_family") != "imts_agents":
            continue
        row = {key: record.get(key) for key in ("variant", "dataset", "seed", "profile", "status", "metrics", "parameters",
                                               "fit_seconds", "predict_seconds", "best_epoch", "epochs_completed", "error")}
        row["path"] = str(path)
        if record.get("status") != "success" or not record.get("comparison_protocol_sha256"):
            row["metrics"] = None
            incomplete.append(row)
        else:
            key = record["comparison_protocol_sha256"]
            groups.setdefault(key, {"protocol": record["comparison_protocol"], "runs": []})["runs"].append(row)
    return {"note": "Compare within a protocol group only. Validation and smoke scores are not official test results. "
                    "Repeated attempts are listed separately; no best-seed or best-test cherry picking.",
            "protocol_groups": groups, "incomplete": incomplete, "invalid_result_files": invalid}


def evaluate_checkpoint(run_dir, *, device=None, cache_dir=None):
    """Evaluate a locked validation checkpoint on the official test exactly once.

    No fit occurs. Source, prepared tensors, split identities, configuration and
    checkpoint bytes must still match the original run. A failed test attempt
    remains recorded and requires review instead of an automatic retry.
    """
    from .model import AgentClassifier
    from .variants import get_variant

    run_dir = Path(run_dir)
    path = run_dir / "result.json"
    result = json.loads(path.read_text())
    if result.get("experiment_family") != "imts_agents" or result.get("status") != "success":
        raise ValueError("Expected a successful imts_agents validation run")
    if result.get("profile") != "validation" or result.get("config", {}).get("smoke"):
        raise ValueError("Only a complete validation pilot may be evaluated on test")
    if result.get("test_evaluated") or result.get("test_evaluation_attempted_at") or (run_dir / "test_predictions.npz").exists():
        raise ValueError("This run already evaluated or attempted test; refusing repeated test evaluation")
    if result["config_sha256"] != _digest(result["config"]):
        raise ValueError("Original run configuration was modified")
    if result["environment"]["source_sha256"] != environment()["source_sha256"]:
        raise ValueError("Source differs from training; restore the exact source before test evaluation")
    checkpoint_path = run_dir / "best.pt"
    if hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() != result["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the recorded best checkpoint")
    config = RunConfig(**result["config"])
    config.validate()
    evaluation_device = device or config.device
    if evaluation_device not in {"cpu", "cuda"}:
        raise ValueError("Evaluation device must be cpu or cuda")
    spec = get_variant(config.variant)
    _seed(config.seed, config.threads, evaluation_device)
    data = prepare_dataset(config.dataset, seed=config.seed, n_patches=config.n_patches,
                           time_features=spec.time_features, cache_dir=cache_dir)
    hashes = {split: _array_digest(*(getattr(data, f"{prefix}_{split}") for prefix in ("X", "mask", "y")))
              for split in ("train", "val", "test")}
    if hashes != result["prepared_data_sha256"] or data.metadata["sample_ids_sha256"] != result["data"]["sample_ids_sha256"]:
        raise ValueError("Prepared data or sample identities differ from training")
    lock_path = run_dir / ".test-evaluation.lock"
    lock_fd = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(lock_fd)
    try:
        # Check the record again after acquiring the lock against parallel calls.
        current = json.loads(path.read_text())
        if current.get("test_evaluated") or current.get("test_evaluation_attempted_at"):
            raise ValueError("Test evaluation was started by another process")
        result["test_evaluation_attempted_at"] = datetime.now(timezone.utc).isoformat()
        result["test_evaluation_status"] = "running"
        _write_json(path, result)
        checkpoint = torch.load(checkpoint_path, map_location=evaluation_device, weights_only=True)
        if checkpoint["config"] != result["config"] or checkpoint["comparison_protocol_sha256"] != result["comparison_protocol_sha256"]:
            raise ValueError("Checkpoint configuration or comparison protocol does not match the run")
        model = AgentClassifier(n_features=data.X_train.shape[-1], n_channels=data.X_train.shape[-2],
                                n_classes=len(data.classes), variant=spec, hidden=config.hidden,
                                max_inner_steps=config.max_inner_steps).to(evaluation_device)
        model.load_state_dict(checkpoint["model_state_dict"])
        test = _save_predictions(run_dir, "test", _predict(model, _loader(data, "test", config), evaluation_device, len(data.classes)), data)
        result["test"] = test
        result["test_evaluated"] = True
        result["test_evaluation_status"] = "success"
        result["test_evaluation_device"] = evaluation_device
        result["test_evaluated_at"] = datetime.now(timezone.utc).isoformat()
        result["profile"] = "test"
        result["metrics"] = test["metrics"]
        result["predict_seconds"] = test["cost"]["predict_seconds"]
        result["evaluation_scope"] = "official test evaluated once from a locked validation-selected checkpoint; exploratory pilot"
        result["comparison_protocol"]["evaluate_test"] = True
        result["comparison_protocol"]["evaluation_device"] = evaluation_device
        result["comparison_protocol_sha256"] = _digest(result["comparison_protocol"])
        _write_json(path, result)
    except (Exception, KeyboardInterrupt) as exc:
        if result.get("test_evaluation_attempted_at"):
            result["test_evaluation_status"] = "failed"
            result["test_evaluation_error"] = f"{type(exc).__name__}: {exc}"
            _write_json(path, result)
        raise
    finally:
        lock_path.unlink(missing_ok=True)
    print(_json({"status": "success", "profile": "test", "variant": config.variant,
                 "metrics": test["metrics"], "output": str(run_dir)}), flush=True)
    return result, run_dir
