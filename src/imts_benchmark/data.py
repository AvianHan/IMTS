"""Official Pyrregular splits, with explicit memory and leakage safeguards.

The dense representation is exactly ``df.irr.to_dense()``: (sample, channel,
position), NaNs retained, no appended time channel. Preprocessing belongs to
the upstream model pipeline. Full runs never shorten sequences or subsample.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Any

import numpy as np


_MANIFEST_PATH = Path(__file__).resolve().parents[2] / "configs" / "datasets.json"
MANIFEST = json.loads(_MANIFEST_PATH.read_text(encoding="utf-8"))
DATASET_CHANNELS = {item["name"]: item["channels"] for item in MANIFEST["datasets"]}
DATASETS = tuple(DATASET_CHANNELS)
_DATASET_SPECS = {item["name"]: item for item in MANIFEST["datasets"]}
_CACHE_LOCK = threading.RLock()
_VALID_SPLITS = frozenset({"train", "val", "validation", "test"})


@dataclass
class DatasetSplit:
    name: str
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_ids: np.ndarray
    test_ids: np.ndarray
    metadata: dict[str, Any]


def _digest(values: np.ndarray) -> str:
    payload = json.dumps(np.asarray(values).tolist(), ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_split(y: np.ndarray, split: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Reject unknown split names *before* interpreting non-test as training."""
    if y.ndim != 1 or split.ndim != 1 or ids.ndim != 1:
        raise ValueError("Targets, official split labels, and sample IDs must be 1D.")
    if not len(y) or len(y) != len(split) or len(y) != len(ids):
        raise ValueError("Targets, split labels, and sample IDs must have equal nonzero lengths.")
    unknown = set(split.tolist()) - _VALID_SPLITS
    if unknown:
        raise ValueError(f"Unknown official split labels: {sorted(unknown)}; refusing to train on them.")
    if len(set(ids.tolist())) != len(ids):
        raise ValueError("Sample IDs must be unique; duplicate IDs could leak across splits.")
    if any(value is None or value != value for value in y):
        raise ValueError("Default classification target contains missing labels.")
    train = split != "test"
    if not train.any() or train.all():
        raise ValueError("Official default task must contain both fitting and held-out test samples.")
    train_classes = set(y[train].tolist())
    if len(train_classes) < 2:
        raise ValueError("Official fitting split contains fewer than two classes.")
    unseen = set(y[~train].tolist()) - train_classes
    if unseen:
        raise ValueError(f"Test labels absent from the fitting split: {unseen}. Do not move test samples into training.")
    return train


@contextmanager
def _repository_cache(repository: Any, cache_dir: str | Path | None):
    # Pyrregular uses a shared Pooch object. Restore its path even if loading fails.
    with _CACHE_LOCK:
        previous = repository.REPOSITORY.path
        try:
            if cache_dir is not None:
                destination = Path(cache_dir).expanduser().resolve()
                destination.mkdir(parents=True, exist_ok=True)
                repository.REPOSITORY.path = destination
            yield
        finally:
            repository.REPOSITORY.path = previous


def _dense_with_guard(df: Any, max_dense_gib: float) -> tuple[np.ndarray, dict[str, Any]]:
    if not math.isfinite(max_dense_gib) or max_dense_gib <= 0:
        raise ValueError("max_dense_gib must be finite and greater than zero.")
    if tuple(df.dims) != ("ts_id", "signal_id", "time_id"):
        raise ValueError(f"Unexpected Pyrregular dimension order: {df.dims!r}")
    # This official operation returns sparse arrays. Their local time axis is
    # the longest *sample*, never the potentially enormous global timestamp grid.
    sparse_X, sparse_T = df.irr.reset_time_index()
    shape = tuple(int(v) for v in sparse_X.shape)
    x_bytes = math.prod(shape) * np.dtype(sparse_X.dtype).itemsize
    t_bytes = math.prod(sparse_T.shape) * np.dtype(sparse_T.dtype).itemsize
    # Account conservatively for dense X, dense T, and train/test indexing copies.
    # Model-specific allocations and sparse source storage are additional.
    estimated_peak = 2 * x_bytes + t_bytes
    del sparse_X, sparse_T
    if estimated_peak > max_dense_gib * 1024**3:
        raise MemoryError(
            f"Dense load for shape {shape} needs about {estimated_peak / 1024**3:.3f} GiB "
            f"for X/T/split copies, above --max-dense-gib={max_dense_gib}. "
            "No samples or timesteps were truncated. Use a larger-memory job and explicitly raise the limit."
        )
    X, T = df.irr.to_dense()
    del T  # The six selected official pipelines do not append timestamp channels.
    X = np.asarray(X)
    if X.shape != shape:
        raise RuntimeError("Pyrregular sparse preflight and dense conversion returned different shapes.")
    return X, {
        "dense_shape": list(shape),
        "dense_dtype": str(X.dtype),
        "estimated_dense_load_peak_bytes": estimated_peak,
        "max_dense_gib": max_dense_gib,
        "memory_estimate_excludes": "sparse source arrays and model allocations",
    }


def load_data(name: str, cache_dir: str | Path | None = None,
              max_dense_gib: float = 8.0) -> DatasetSplit:
    """Load one of the 18 multivariate datasets without changing its official split."""
    name = name.removesuffix(".h5")
    if name not in _DATASET_SPECS:
        raise ValueError(f"Unknown configured dataset {name!r}. Choose from: {', '.join(DATASETS)}")
    from pyrregular import load_dataset, repository

    spec = _DATASET_SPECS[name]
    actual_revision = getattr(repository, "DATA_REVISION", None)
    if actual_revision != MANIFEST["data_revision"]:
        raise RuntimeError(
            f"Pyrregular data revision mismatch: {actual_revision!r}; expected {MANIFEST['data_revision']}. "
            "Install the pinned upstream version before reproducing the benchmark."
        )
    registry_hash = repository.REPOSITORY.registry.get(spec["file"], "")
    if registry_hash.removeprefix("sha256:") != spec["sha256"]:
        raise RuntimeError(f"Pyrregular registry hash mismatch for {spec['file']}.")
    with _repository_cache(repository, cache_dir):
        df = load_dataset(spec["file"])["data"]
        source_path = str(Path(repository.REPOSITORY.path) / spec["file"])
    if int(df.sizes["signal_id"]) != spec["channels"]:
        raise ValueError(
            f"{name}: expected {spec['channels']} channels, received {df.sizes['signal_id']}. "
            "Check dataset revision before continuing."
        )
    task = df.irr.get_task()
    if task.get("task") != "classification":
        raise ValueError(f"{name}: default task is not classification: {task!r}")
    y, split = df.irr.get_task_target_and_split()
    y = np.asarray(y)
    split = np.asarray(split).astype(str)
    ids = np.asarray(df["ts_id"].values).astype(str)
    train = _validate_split(y, split, ids)
    X, memory = _dense_with_guard(df, max_dense_gib)
    if X.shape[0] != len(y):
        raise ValueError("Dense sample order length does not match target and ID coordinates.")
    labels, counts = np.unique(split, return_counts=True)
    metadata = {
        "source": "pyrregular",
        "data_revision": actual_revision,
        "data_version": getattr(repository, "DATA_VERSION", None),
        "file": spec["file"],
        "source_path": source_path,
        "sha256": spec["sha256"],
        "hash_verification": "Pooch REPOSITORY.fetch validates against the pinned registry",
        "task": dict(task),
        "official_split_counts": dict(zip(labels.tolist(), counts.tolist())),
        "split_policy": "fit: split != test; evaluate: split == test",
        "sample_order_sha256": _digest(ids),
        "train_ids_sha256": _digest(ids[train]),
        "test_ids_sha256": _digest(ids[~train]),
        "train_source_indices": np.flatnonzero(train).tolist(),
        "test_source_indices": np.flatnonzero(~train).tolist(),
        "preprocessing": "official irr.to_dense defaults; NaNs retained; no normalization/imputation/time concatenation",
        "smoke_subset": False,
        **memory,
    }
    return DatasetSplit(name, X[train], X[~train], y[train], y[~train],
                        ids[train], ids[~train], metadata)


def make_synthetic() -> DatasetSplit:
    """Deterministic plumbing data, explicitly unrelated to benchmark scores.

    Both splits are balanced (40 fitting, 16 held-out); each sample has three
    channels, 24--32 valid positions before ragged padding, plus internal NaNs.
    """
    rng = np.random.default_rng(712)
    n, channels, width = 56, 3, 32
    y = np.arange(n) % 2
    X = np.full((n, channels, width), np.nan, dtype=np.float64)
    for i in range(n):
        length = int(rng.integers(24, width + 1))
        times = np.cumsum(rng.uniform(0.5, 1.5, length))
        for channel in range(channels):
            values = np.sin(times / (3 + channel) + y[i] * 1.7)
            values += 0.6 * y[i] + rng.normal(0, 0.08, length)
            X[i, channel, :length] = values
            X[i, channel, rng.integers(2, length - 2, size=3)] = np.nan
    ids = np.asarray([f"synthetic-{i:03d}" for i in range(n)])
    return DatasetSplit(
        "synthetic", X[:40], X[40:], y[:40], y[40:], ids[:40], ids[40:],
        {"source": "synthetic", "purpose": "plumbing smoke test only; not a benchmark result",
         "generator_seed": 712, "dense_shape": list(X.shape), "smoke_subset": False,
         "train_ids_sha256": _digest(ids[:40]), "test_ids_sha256": _digest(ids[40:])},
    )


def _stratified_indices(y: np.ndarray, maximum: int, rng: np.random.Generator,
                        require_all_classes: bool) -> np.ndarray:
    if isinstance(maximum, bool) or not isinstance(maximum, (int, np.integer)) or maximum < 1:
        raise ValueError("Smoke sample limits must be positive integers.")
    if len(y) <= maximum:
        return np.arange(len(y))
    classes, inverse, counts = np.unique(y, return_inverse=True, return_counts=True)
    if maximum < len(classes):
        if require_all_classes:
            raise ValueError(
                f"Smoke max_train={maximum} cannot retain all {len(classes)} training classes; "
                "increase --max-train. Test data will never be used to fill training classes."
            )
        return np.sort(rng.choice(len(y), maximum, replace=False))
    ideal = counts * maximum / len(y)
    quota = np.minimum(counts, np.maximum(1, np.floor(ideal).astype(int)))
    while quota.sum() > maximum:
        candidates = np.flatnonzero(quota > 1)
        quota[candidates[np.argmax((quota - ideal)[candidates])]] -= 1
    while quota.sum() < maximum:
        candidates = np.flatnonzero(quota < counts)
        quota[candidates[np.argmax((ideal - quota)[candidates])]] += 1
    selected = [rng.choice(np.flatnonzero(inverse == i), int(k), replace=False)
                for i, k in enumerate(quota)]
    return np.sort(np.concatenate(selected))


def subset_for_smoke(data: DatasetSplit, max_train: int = 64,
                     max_test: int = 32, seed: int = 0) -> DatasetSplit:
    """Subsample each existing split independently, retaining every fitting class."""
    if not len(data.y_train) or not len(data.y_test):
        raise ValueError("Cannot smoke-test an empty fitting or test split.")
    if set(data.train_ids.tolist()) & set(data.test_ids.tolist()):
        raise ValueError("Fitting and test sample IDs overlap.")
    if set(data.y_test.tolist()) - set(data.y_train.tolist()):
        raise ValueError("Test labels absent from fitting split; refusing to repair using test data.")
    train_rng, test_rng = [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(2)]
    train = _stratified_indices(data.y_train, max_train, train_rng, True)
    test = _stratified_indices(data.y_test, max_test, test_rng, False)
    metadata = {
        **data.metadata,
        "smoke_subset": True,
        "smoke_seed": seed,
        "full_split_sizes": {"train": len(data.y_train), "test": len(data.y_test)},
        "smoke_train_indices_within_original_train": train.tolist(),
        "smoke_test_indices_within_original_test": test.tolist(),
        "train_ids_sha256": _digest(data.train_ids[train]),
        "test_ids_sha256": _digest(data.test_ids[test]),
        "purpose": "plumbing smoke test only; not comparable to full benchmark scores",
    }
    return replace(data, X_train=data.X_train[train], X_test=data.X_test[test],
                   y_train=data.y_train[train], y_test=data.y_test[test],
                   train_ids=data.train_ids[train], test_ids=data.test_ids[test],
                   metadata=metadata)
