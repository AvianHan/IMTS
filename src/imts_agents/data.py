"""Timestamped-record preprocessing for the first agent-method comparison.

All learned preprocessing uses the *inner training* partition.  Each sample has
one common origin (its first stored record timestamp), preserving channel offsets.
The first P-1 bins have a fixed training-derived width; the final bin is open
ended so long validation/test samples are retained rather than truncated.
Neither sample duration normalization nor global sparse-to-dense conversion is
used.  This is an explicit pilot representation, not an optimal patching claim.
"""
from __future__ import annotations

from dataclasses import dataclass
import fcntl
import gc
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from imts_benchmark.data import MANIFEST, _digest, _repository_cache, _validate_split


FEATURE_NAMES = (
    "log1p_count", "mean", "std", "minimum", "maximum", "first", "last",
    "first_offset_widths", "last_offset_widths", "log1p_previous_gap_widths",
    "log1p_record_count", "observed_fraction",
)
TIME_FEATURE_INDICES = (7, 8, 9)
EventSample = tuple[np.ndarray, np.ndarray, np.ndarray]
PREPARATION_MEMORY_BUDGET_BYTES = 16 * 1024**3
MOMENT_WORKSPACE_BYTES = 64 * 1024**2


class _MappedEvents(Sequence[EventSample]):
    """Transform one sample at a time, without another full event-data copy."""

    def __init__(self, source: Sequence[EventSample], transform: Any):
        self.source, self.transform = source, transform

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        return self.transform(self.source[index])


def _preparation_memory_estimate(events: Sequence[EventSample], n_channels: int,
                                 n_patches: int) -> dict[str, int]:
    """Conservative managed-array estimate, not a promise about process RSS.

    The largest dense phase retains float64 features while casting float32;
    final split copies have a lower peak (two float32 tensors). Raw event
    arrays remain live. Reserve six largest-sample array equivalents for
    cleaning/normalising/group sorting, a bounded moments workspace, and a
    512 MiB allowance for Python/runtime objects. Upstream sparse loading has
    a separate earlier peak and is released before this preparation phase.
    """
    sample_bytes = [sum(np.asarray(a).nbytes for a in sample) for sample in events]
    cells = len(events) * n_patches * n_channels
    dense = cells * (len(FEATURE_NAMES) * (8 + 4) + 2)
    raw = sum(sample_bytes)
    sample_workspace = 6 * max(sample_bytes, default=0)
    allowance = 512 * 1024**2
    return {
        "dense_peak_bytes": dense, "retained_event_bytes": raw,
        "largest_sample_workspace_bytes": sample_workspace,
        "moments_workspace_bytes": MOMENT_WORKSPACE_BYTES,
        "runtime_allowance_bytes": allowance,
        "estimated_peak_bytes": dense + raw + sample_workspace + MOMENT_WORKSPACE_BYTES + allowance,
        "budget_bytes": PREPARATION_MEMORY_BUDGET_BYTES,
    }


def _feature_moments(features: np.ndarray, masks: np.ndarray,
                     train_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Training-only population moments with a bounded gather workspace."""
    bytes_per_sample = int(np.prod(features.shape[1:])) * features.dtype.itemsize * 2
    chunk_samples = max(1, MOMENT_WORKSPACE_BYTES // max(bytes_per_sample, 1))
    count = 0
    center = np.zeros(features.shape[-1], dtype=np.float64)
    m2 = np.zeros_like(center)
    for start in range(0, len(train_indices), chunk_samples):
        idx = train_indices[start:start + chunk_samples]
        observed = features[idx][masks[idx]]
        if not len(observed):
            continue
        local_center = observed.mean(axis=0)
        # Reuse the bounded gather rather than materialising its square.
        observed -= local_center
        np.square(observed, out=observed)
        local_m2 = observed.sum(axis=0)
        n = len(observed)
        total = count + n
        delta = local_center - center
        m2 += local_m2 + delta**2 * count * n / total
        center += delta * n / total
        count = total
    scale = np.sqrt(m2 / max(count, 1))
    return center, np.where(scale > 1e-12, scale, 1.0)


@dataclass
class PreparedData:
    X_train: np.ndarray
    X_val: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    mask_train: np.ndarray
    mask_val: np.ndarray
    mask_test: np.ndarray
    metadata: dict[str, Any]
    classes: np.ndarray


def _clean_events(sample: EventSample, n_channels: int) -> EventSample:
    times, channels, values = (np.asarray(x) for x in sample)
    if any(x.ndim != 1 for x in (times, channels, values)) or not (len(times) == len(channels) == len(values)):
        raise ValueError("Event times, channel indices and values must be equal-length 1D arrays.")
    if channels.dtype.kind not in "iu" or np.any(channels < 0) or np.any(channels >= n_channels):
        raise ValueError("Event channel indices must be integers within the configured channel range.")
    times, values = times.astype(np.float64), values.astype(np.float64)
    if np.isinf(values).any():
        raise ValueError("Infinite observed values are invalid; they are not missing-value markers.")
    if not np.isfinite(times).all():
        raise ValueError("Stored records, including explicit missing values, require finite timestamps.")
    if len(times):
        times = times - times.min()
        order = np.argsort(times, kind="stable")
        times, channels, values = times[order], channels[order], values[order]
    return times, channels.astype(np.int64), values


def _patch_features(events: Sequence[EventSample], n_channels: int, n_patches: int,
                    width: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    shape = (len(events), n_patches, n_channels)
    # Include the float64 working tensor plus float32 split copies and masks.
    estimated_bytes = int(np.prod(shape)) * (len(FEATURE_NAMES) * 12 + 2)
    if estimated_bytes > PREPARATION_MEMORY_BUDGET_BYTES:
        raise MemoryError(f"Patch feature preparation estimates {estimated_bytes / 1024**3:.2f} GiB; "
                          "the 16 GiB preparation guard was exceeded. No events were truncated.")
    X = np.zeros((*shape, len(FEATURE_NAMES)), dtype=np.float64)
    mask = np.zeros(shape, dtype=bool)
    overflow_counts = np.zeros(len(events), dtype=np.int64)
    observed_counts = np.zeros(len(events), dtype=np.int64)
    for sample_i, (times, channels, values) in enumerate(events):
        observed_counts[sample_i] = int(np.isfinite(values).sum())
        quotient = times / width
        nearest = np.rint(quotient)
        quotient = np.where(np.isclose(quotient, nearest, rtol=1e-12, atol=1e-12), nearest, quotient)
        overflow_counts[sample_i] = int(np.sum(quotient > n_patches))
        # Clip the floating quotient before the integer conversion: an extreme
        # held-out time must not overflow int64 or silently discard an event.
        bins = np.minimum(np.floor(quotient), n_patches - 1).astype(np.int64)
        if not len(times):
            continue
        keys = bins * n_channels + channels
        order = np.argsort(keys, kind="stable")
        keys = keys[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(keys)) + 1, len(keys)]
        previous_times = np.zeros(n_channels, dtype=np.float64)
        # One pass over observed patch/channel groups; avoid repeatedly scanning
        # every event for all C*P combinations on large medical datasets.
        for left, right in zip(boundaries[:-1], boundaries[1:]):
            patch, channel = divmod(int(keys[left]), n_channels)
            selected = order[left:right]
            t, v = times[selected], values[selected]
            observed = np.isfinite(v)
            observed_values = v[observed]
            mask[sample_i, patch, channel] = True
            start = patch * width
            stats = ((np.mean(observed_values), np.std(observed_values, ddof=0),
                      np.min(observed_values), np.max(observed_values),
                      observed_values[0], observed_values[-1]) if observed.any() else (0.,) * 6)
            X[sample_i, patch, channel] = (
                np.log1p(len(observed_values)), *stats, (t[0] - start) / width,
                (t[-1] - start) / width, np.log1p((t[0] - previous_times[channel]) / width),
                np.log1p(len(v)), float(observed.mean()),
            )
            if observed.any():
                previous_times[channel] = float(t[observed][-1])
    for sample in X:
        if not np.isfinite(sample).all():
            raise ValueError("Patch features overflowed; inspect input magnitudes and timestamp units.")
    return X, mask, overflow_counts, observed_counts


def _value_scaler(events: Sequence[EventSample], train_indices: np.ndarray,
                  n_channels: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stable per-channel moments of finite inner-training values only."""
    counts = np.zeros(n_channels, dtype=np.int64)
    center = np.zeros(n_channels, dtype=np.float64)
    m2 = np.zeros(n_channels, dtype=np.float64)
    for sample_i in train_indices:
        _, channels, values = events[sample_i]
        finite = np.isfinite(values)
        for channel in np.unique(channels[finite]):
            v = values[finite & (channels == channel)]
            n = len(v)
            local_mean = float(v.mean())
            local_m2 = float(np.sum((v - local_mean) ** 2))
            total = counts[channel] + n
            delta = local_mean - center[channel]
            m2[channel] += local_m2 + delta**2 * counts[channel] * n / total
            center[channel] += delta * n / total
            counts[channel] = total
    scale = np.sqrt(m2 / np.maximum(counts, 1))
    scale = np.where(scale > 1e-12, scale, 1.0)
    if not np.isfinite(center).all() or not np.isfinite(scale).all():
        raise ValueError("Inner-training channel moments overflowed; inspect raw value magnitudes.")
    return center, scale, counts


def prepare_events(events: Sequence[EventSample], labels: np.ndarray,
                   official_split: np.ndarray, sample_ids: np.ndarray,
                   n_channels: int, *, dataset: str = "synthetic", seed: int = 0,
                   n_patches: int = 8, time_features: bool = True,
                   source_metadata: dict[str, Any] | None = None) -> PreparedData:
    """Prepare event samples expressed in a common time-coordinate unit.

    The official non-test pool is split using the baseline neural wrapper's
    sklearn 25% holdout and seed, with the same stratification fallback.  Labels
    are encoded using inner training only.  If fallback leaves an unseen class
    in validation/test, fail explicitly rather than repair using held-out data.
    """
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32).")
    if isinstance(n_patches, bool) or not isinstance(n_patches, int) or n_patches < 1:
        raise ValueError("n_patches must be a positive integer.")
    if isinstance(n_channels, bool) or not isinstance(n_channels, int) or n_channels < 1:
        raise ValueError("n_channels must be a positive integer.")
    labels = np.asarray(labels)
    official_split = np.asarray(official_split).astype(str)
    sample_ids = np.asarray(sample_ids).astype(str)
    fitting = _validate_split(labels, official_split, sample_ids)
    if len(events) != len(labels):
        raise ValueError("Number of event samples must match labels and official sample IDs.")
    fit_indices, test_indices = np.flatnonzero(fitting), np.flatnonzero(~fitting)
    try:
        train_indices, val_indices = train_test_split(
            fit_indices, stratify=labels[fit_indices], random_state=seed,
        )
        stratified = True
    except ValueError:
        train_indices, val_indices = train_test_split(fit_indices, random_state=seed)
        stratified = False
    encoder = LabelEncoder().fit(labels[train_indices])
    if len(encoder.classes_) < 2:
        raise ValueError("Inner training split has fewer than two classes.")
    unseen = set(labels.tolist()) - set(encoder.classes_.tolist())
    if unseen:
        raise ValueError(f"Held-out labels absent from inner training: {unseen}. "
                         "Refusing to move held-out samples or fit their labels.")
    memory_estimate = _preparation_memory_estimate(events, n_channels, n_patches)
    if memory_estimate["estimated_peak_bytes"] > PREPARATION_MEMORY_BUDGET_BYTES:
        raise MemoryError(
            f"Event/patch preparation estimates {memory_estimate['estimated_peak_bytes'] / 1024**3:.2f} GiB; "
            "the 16 GiB preparation guard was exceeded. No events were truncated."
        )
    cleaned = _MappedEvents(events, lambda sample: _clean_events(sample, n_channels))
    durations = np.asarray([float(t[-1]) if len(t) else 0.0 for t, _, _ in cleaned])
    training_duration = float(durations[train_indices].max())
    width = training_duration / n_patches if training_duration > 0 else 1.0
    value_center, value_scale, value_counts = _value_scaler(cleaned, train_indices, n_channels)
    normalized = _MappedEvents(cleaned, lambda sample: (
        sample[0], sample[1], (sample[2] - value_center[sample[1]]) / value_scale[sample[1]],
    ))
    features, masks, overflows, observed_counts = _patch_features(
        normalized, n_channels, n_patches, width,
    )
    center, scale = _feature_moments(features, masks, train_indices)
    np.subtract(features, center, out=features)
    np.divide(features, scale, out=features)
    features[~masks] = 0.0
    if not time_features:
        features[..., list(TIME_FEATURE_INDICES)] = 0.0
    with np.errstate(over="ignore", invalid="ignore"):
        features = features.astype(np.float32)
    for sample in features:
        if not np.isfinite(sample).all():
            raise ValueError("Scaled patch features cannot be represented as finite float32.")
    splits = {"train": train_indices, "val": val_indices, "test": test_indices}
    metadata: dict[str, Any] = {
        **(source_metadata or {}), "dataset": dataset, "seed": seed,
        "n_channels": n_channels, "n_patches": n_patches,
        "feature_names": list(FEATURE_NAMES), "time_features": bool(time_features),
        "disabled_feature_indices": [] if time_features else list(TIME_FEATURE_INDICES),
        "split_policy": "official non-test pool; sklearn 25% internal validation; official test untouched",
        "validation_stratified": stratified,
        "sample_ids": {name: sample_ids[idx].tolist() for name, idx in splits.items()},
        "source_indices": {name: idx.tolist() for name, idx in splits.items()},
        "sample_ids_sha256": {name: _digest(sample_ids[idx]) for name, idx in splits.items()},
        "official_split_counts": {str(s): int(np.sum(official_split == s)) for s in np.unique(official_split)},
        "classes": encoder.classes_.tolist(),
        "class_mapping": [{"label": label, "index": i} for i, label in enumerate(encoder.classes_.tolist())],
        "preprocessing": {
            "fit_partition": "inner training only", "origin": "first stored record across all channels per sample, including explicit NaN records",
            "memory_preparation": {
                **memory_estimate,
                "strategy": "lazy per-sample event transforms; bounded training-only feature moments; in-place float64 scaling; float32 split arrays",
                "estimate_scope": "managed preparation arrays plus allowance; upstream sparse-load peak is separate; not a process RSS guarantee",
            },
            "timestamp_units": (source_metadata or {}).get("timestamp_units", "input event time-coordinate units"),
            "patch_width_time_units": width,
            "inner_training_max_duration_time_units": training_duration,
            "nominal_patch_horizon_time_units": width * n_patches,
            "zero_duration_width_fallback_time_units": 1.0,
            "patch_boundaries_time_units": [width * i for i in range(n_patches)],
            "boundary_policy": "left-closed bins; boundary event belongs to the next bin; last bin is [last start, infinity)",
            "boundary_float_tolerance": "time/width within rtol=1e-12,atol=1e-12 of an integer is snapped to that boundary",
            "overflow_definition": "stored record timestamp strictly greater than P * training-fitted patch width; missing records included",
            "overflow_event_counts": {name: int(overflows[idx].sum()) for name, idx in splits.items()},
            "overflow_sample_counts": {name: int((overflows[idx] > 0).sum()) for name, idx in splits.items()},
            "observed_event_counts": {name: int(observed_counts[idx].sum()) for name, idx in splits.items()},
            "record_counts": {name: sum(len(events[i][0]) for i in idx) for name, idx in splits.items()},
            "empty_sample_counts": {name: sum(not len(events[i][0]) for i in idx) for name, idx in splits.items()},
            "no_observed_value_sample_counts": {name: int((observed_counts[idx] == 0).sum()) for name, idx in splits.items()},
            "raw_value_scaler": "per-channel finite inner-training values; population std; constant or unobserved scale=1",
            "raw_value_center_by_channel": value_center.tolist(),
            "raw_value_scale_by_channel": value_scale.tolist(),
            "raw_observed_count_by_channel": value_counts.tolist(),
            "scaler": "pooled record-present inner-training patch/channel features; population std; constant scale=1",
            "feature_center": center.tolist(), "feature_scale": scale.tolist(),
            "gap_policy": "first stored record in current patch minus previous observed value time for that channel; common sample origin if none; NaNs do not update last-observed time",
            "missing_policy": "explicit NaN records retained with mask=true; finite values alone determine value statistics; all-missing value stats=0 before feature scaling; no imputation",
            "padding_policy": "mask=false only for no stored record in channel/patch; all features then zero; stored NaNs distinct from absent records; physical missing events absent from source cannot be reconstructed",
            "time_ablation": "same shape, width, scaler, values and mask; standardized feature columns 7,8,9 zeroed",
        },
    }
    return PreparedData(
        features[train_indices], features[val_indices], features[test_indices],
        encoder.transform(labels[train_indices]).astype(np.int64),
        encoder.transform(labels[val_indices]).astype(np.int64),
        encoder.transform(labels[test_indices]).astype(np.int64),
        masks[train_indices], masks[val_indices], masks[test_indices], metadata, encoder.classes_,
    )


def _events_from_frame(df: Any) -> tuple[list[EventSample], list[str | None]]:
    if tuple(df.dims) != ("ts_id", "signal_id", "time_id"):
        raise ValueError(f"Unexpected Pyrregular dimension order: {df.dims!r}")
    arr = df.data
    if not hasattr(arr, "coords") or not hasattr(arr, "data"):
        raise ValueError("Expected Pyrregular sparse COO observations; refusing global densification.")
    coords, values = np.asarray(arr.coords), np.asarray(arr.data)
    if coords.shape != (3, len(values)):
        raise ValueError("Sparse observation coordinates do not match values.")
    if not np.isnan(arr.fill_value):
        raise ValueError("Expected NaN sparse fill; implicit observed zeros cannot be dropped.")
    timeline = np.asarray(df["time_id"].values)
    # Match upstream's default coordinate scale (1e-9). This converts datetime
    # nanoseconds to seconds, but numeric dataset coordinates can instead be
    # ordinal indices; do not infer real-world units from the conversion alone.
    # Subtract before float conversion to preserve small calendar-time gaps.
    if timeline.dtype.kind in "mM":
        unit_type = "datetime64[ns]" if timeline.dtype.kind == "M" else "timedelta64[ns]"
        timeline = timeline.astype(unit_type).astype(np.int64)
    order = np.argsort(coords[0], kind="stable")
    sample_coords = coords[0, order]
    boundaries = np.searchsorted(sample_coords, np.arange(int(df.sizes["ts_id"]) + 1))
    events, origins = [], []
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        selected = order[start:end]
        t, c, v = timeline[coords[2, selected]], coords[1, selected], values[selected]
        if len(t):
            if not np.isfinite(t).all() or np.any(t == np.iinfo(np.int64).min):
                raise ValueError("Stored record has an invalid timestamp.")
            origin = t.min()
            # Unsigned subtraction yields the exact nonnegative difference for
            # ordered int64 coordinates, even if the signed difference would
            # overflow. Convert only after subtracting; portable beyond Linux.
            delta = (t.astype(np.uint64) - np.asarray(origin).astype(np.uint64)
                     if t.dtype.kind in "iu" else t - origin)
            seconds = np.asarray(delta, dtype=np.float64) * 1e-9
            origins.append(str(origin))
        else:
            seconds = np.empty(0, dtype=np.float64)
            origins.append(None)
        events.append((seconds, c.astype(np.int64), v.astype(np.float64)))
    return events, origins


def synthetic_events(seed: int = 712) -> tuple[list[EventSample], np.ndarray, np.ndarray, np.ndarray, int]:
    """Small irregular plumbing dataset; its scores are never benchmark scores."""
    rng = np.random.default_rng(seed)
    events = []
    labels = np.arange(64) % 2
    for sample in range(len(labels)):
        times, channels, values = [], [], []
        for channel in range(3):
            t = np.cumsum(rng.uniform(0.1, 0.8, size=20 - channel * 3)) + channel * 0.3
            v = np.sin(t / (channel + 1)) + labels[sample] * 0.8 + rng.normal(0, 0.05, len(t))
            keep = rng.random(len(t)) > 0.1
            times.extend(t[keep]); channels.extend([channel] * int(keep.sum())); values.extend(v[keep])
        events.append((np.asarray(times), np.asarray(channels, dtype=np.int64), np.asarray(values)))
    split = np.asarray(["train"] * 48 + ["test"] * 16)
    ids = np.asarray([f"synthetic-{i:03d}" for i in range(len(labels))])
    return events, labels, split, ids, 3


def prepare_dataset(dataset: str, seed: int = 0, n_patches: int = 8,
                    time_features: bool = True, cache_dir: str | Path | None = None,
                    synthetic: bool = False) -> PreparedData:
    if synthetic or dataset == "synthetic":
        events, labels, split, ids, channels = synthetic_events()
        return prepare_events(events, labels, split, ids, channels, dataset="synthetic",
                              seed=seed, n_patches=n_patches, time_features=time_features,
                              source_metadata={"source": "synthetic", "purpose": "plumbing only; not benchmark scores", "generator_seed": 712,
                                               "timestamp_units": "synthetic time units"})
    from pyrregular import load_dataset, repository
    from imts_benchmark.provenance import verify_upstream

    verify_upstream()
    name = dataset.removesuffix(".h5")
    specs = {item["name"]: item for item in MANIFEST["datasets"]}
    if name not in specs:
        raise ValueError(f"Unknown configured dataset {name!r}.")
    spec = specs[name]
    if getattr(repository, "DATA_REVISION", None) != MANIFEST["data_revision"]:
        raise RuntimeError("Pyrregular data revision mismatch.")
    actual_hash = repository.REPOSITORY.registry.get(spec["file"], "").removeprefix("sha256:")
    if actual_hash != spec["sha256"]:
        raise RuntimeError("Pyrregular dataset registry hash mismatch.")
    with _repository_cache(repository, cache_dir):
        cache = Path(repository.REPOSITORY.path)
        cache.mkdir(parents=True, exist_ok=True)
        # Use the baseline launcher's exact lock name so independently launched
        # baseline/method workers serialize the first verified download. Release
        # the lock before HDF5 parsing and training.
        with (cache / (".imts-download-" + spec["file"] + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            repository.download_dataset_from_huggingface(spec["file"])
        df = load_dataset(spec["file"])["data"]
        source_path = str(Path(repository.REPOSITORY.path) / spec["file"])
    if int(df.sizes["signal_id"]) != spec["channels"]:
        raise ValueError("Dataset channel count differs from the pinned specification.")
    task = df.irr.get_task()
    if task.get("task") != "classification":
        raise ValueError("Default task is not classification.")
    labels, split = df.irr.get_task_target_and_split()
    ids = np.asarray(df["ts_id"].values).astype(str)
    _validate_split(np.asarray(labels), np.asarray(split).astype(str), ids)
    events, origins = _events_from_frame(df)
    source_metadata = {
        "source": "pyrregular", "data_revision": MANIFEST["data_revision"],
        "file": spec["file"], "sha256": spec["sha256"], "source_path": source_path,
        "source_timestamp_origins_native_units": origins,
        "source_time_coordinate_dtype": str(df["time_id"].values.dtype),
        "source_time_coordinate_units_attribute": str(df["time_id"].attrs.get("units", "not specified")),
        "timestamp_units": "source coordinate multiplied by 1e-9 (upstream default); physical seconds only when source units establish nanoseconds",
        "source_time_coordinate_scale": 1e-9,
        "hash_verification": "Pooch fetch validates pinned registry hash", "task": dict(task),
        "sparse_load_memory": {
            "stored_records": int(df.data.data.size),
            "sparse_array_bytes": int(df.data.data.nbytes + df.data.coords.nbytes),
            "note": "upstream sparse frame is released before patch preparation; loading/extraction has its own transient peak",
        },
    }
    # The extracted events own their data. Keeping the complete sparse frame
    # through patch scaling unnecessarily adds several GiB on InsectWingbeat.
    del df
    # xarray caches the .irr accessor, which references its DataArray back.
    # Collect that cycle before allocating the potentially multi-GiB patches.
    gc.collect()
    return prepare_events(
        events, labels, split, ids, spec["channels"], dataset=name, seed=seed,
        n_patches=n_patches, time_features=time_features,
        source_metadata=source_metadata,
    )
