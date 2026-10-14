"""No-leakage and event-time semantics of the agent pilot's data adapter."""
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.model_selection import train_test_split

from imts_agents.data import (
    FEATURE_NAMES, TIME_FEATURE_INDICES, _events_from_frame, _patch_features,
    prepare_dataset, prepare_events,
)
import imts_agents.data as data_module


def example():
    events = []
    for i in range(20):
        events.append((np.array([100., 101., 102., 103., 104., 107.]),
                       np.array([0, 0, 1, 0, 1, 1]),
                       np.array([i, i + 1., i + 2., i + 3., i + 4., i + 5.])))
    labels = np.array(["a", "b"] * 10)
    split = np.array(["train"] * 16 + ["test"] * 4)
    ids = np.array([f"id-{i}" for i in range(20)])
    return events, labels, split, ids


def test_holdout_matches_baseline_and_records_disjoint_ids():
    events, labels, split, ids = example()
    data = prepare_events(events, labels, split, ids, 2, seed=9)
    train, val = train_test_split(np.arange(16), stratify=labels[:16], random_state=9)
    assert data.metadata["source_indices"] == {"train": train.tolist(), "val": val.tolist(), "test": [16, 17, 18, 19]}
    assert data.classes.tolist() == ["a", "b"]
    assert data.X_train.shape == (12, 8, 2, len(FEATURE_NAMES))
    assert data.y_train.dtype == np.int64
    assert data.X_train.dtype == np.float32
    assert data.mask_train.dtype == bool
    id_sets = [set(v) for v in data.metadata["sample_ids"].values()]
    assert all(not (id_sets[a] & id_sets[b]) for a, b in ((0, 1), (0, 2), (1, 2)))


def test_validation_and_test_cannot_fit_width_scaler_or_training_features():
    events, labels, split, ids = example()
    original = prepare_events(events, labels, split, ids, 2, seed=0)
    heldout = original.metadata["source_indices"]["val"] + original.metadata["source_indices"]["test"]
    for i in heldout:
        t, c, v = events[i]
        events[i] = (t * 1000., c, v * 10000.)
    changed = prepare_events(events, labels, split, ids, 2, seed=0)
    p0, p1 = original.metadata["preprocessing"], changed.metadata["preprocessing"]
    assert p0["patch_width_time_units"] == p1["patch_width_time_units"]
    assert p0["feature_center"] == p1["feature_center"]
    assert p0["feature_scale"] == p1["feature_scale"]
    assert p0["raw_value_center_by_channel"] == p1["raw_value_center_by_channel"]
    assert p0["raw_value_scale_by_channel"] == p1["raw_value_scale_by_channel"]
    np.testing.assert_array_equal(original.X_train, changed.X_train)
    assert p1["overflow_event_counts"]["test"] > 0


def test_time_ablation_preserves_dimensions_masks_and_value_features():
    events, labels, split, ids = example()
    full = prepare_events(events, labels, split, ids, 2)
    no_time = prepare_events(events, labels, split, ids, 2, time_features=False)
    for name in ("train", "val", "test"):
        actual, ablated = getattr(full, "X_" + name), getattr(no_time, "X_" + name)
        assert actual.shape == ablated.shape
        np.testing.assert_array_equal(actual[..., :7], ablated[..., :7])
        assert not ablated[..., list(TIME_FEATURE_INDICES)].any()
        np.testing.assert_array_equal(getattr(full, "mask_" + name), getattr(no_time, "mask_" + name))


def test_overflow_is_retained_and_channel_offsets_are_not_reset():
    sample = (np.array([0., 2., 3., 5., 20.]), np.array([0, 1, 0, 1, 1]), np.arange(5.))
    X, mask, overflow, counts = _patch_features([sample], 2, 4, 2.)
    assert counts.tolist() == [5]
    assert overflow.tolist() == [1]
    assert not mask[0, 0, 1]  # channel 1 begins in the second patch, not patch 0
    assert mask[0, 1, 1]
    assert X[0, 3, 1, 6] == 4.  # the overflow event remains in the final patch
    assert np.isclose(np.expm1(X[..., 0]).sum(), 5)
    assert X[0, 3, 1, 9] == np.log1p((20. - 5.) / 2.)


def test_later_events_do_not_change_earlier_patch_features():
    sample = (np.array([0., 1., 4., 9.]), np.array([0, 1, 0, 1]), np.array([1., 2., 3., 4.]))
    X0, m0, _, _ = _patch_features([sample], 2, 4, 2.)
    changed = (np.array([0., 1., 4., 90.]), sample[1], np.array([1., 2., 3., 400.]))
    X1, m1, _, _ = _patch_features([changed], 2, 4, 2.)
    np.testing.assert_array_equal(X0[:, :3], X1[:, :3])
    np.testing.assert_array_equal(m0[:, :3], m1[:, :3])


def test_empty_constant_and_nan_observations_remain_finite():
    events, labels, split, ids = example()
    events = [(np.array([5., 5.]), np.array([0, 1]), np.array([2., np.nan])) for _ in events]
    events[0] = (np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=float))
    data = prepare_events(events, labels, split, ids, 2)
    assert data.metadata["preprocessing"]["patch_width_time_units"] == 1.
    for name in ("train", "val", "test"):
        X, mask = getattr(data, "X_" + name), getattr(data, "mask_" + name)
        assert np.isfinite(X).all()
        assert not X[~mask].any()
        # Channel 1 has explicit NaN records: present token, zero observed count.
        assert mask[:, 0, 1].any()
        assert not mask[:, 1:, 1].any()


def test_invalid_split_and_infinite_values_are_rejected():
    events, labels, split, ids = example()
    split[0] = "weird"
    with pytest.raises(ValueError, match="Unknown official split"):
        prepare_events(events, labels, split, ids, 2)
    split[0] = "train"
    events[0][2][0] = np.inf
    with pytest.raises(ValueError, match="Infinite observed"):
        prepare_events(events, labels, split, ids, 2)


def test_sparse_event_extraction_preserves_nanosecond_offsets_without_dense_grid():
    epoch = 1700000000000000000
    frame = SimpleNamespace(
        dims=("ts_id", "signal_id", "time_id"), sizes={"ts_id": 2, "signal_id": 2, "time_id": 10**12},
        data=SimpleNamespace(coords=np.array([[0, 0, 1], [0, 1, 1], [0, 1, 2]]),
                             data=np.array([1., 2., 3.]), fill_value=np.nan),
    )
    class Frame:
        dims, sizes, data = frame.dims, frame.sizes, frame.data
        def __getitem__(self, key):
            assert key == "time_id"
            return SimpleNamespace(values=np.array([epoch, epoch + 1, epoch + 9], dtype=np.int64))
    events, origins = _events_from_frame(Frame())
    np.testing.assert_allclose(events[0][0], [0., 1e-9], rtol=0, atol=1e-20)
    np.testing.assert_array_equal(events[0][1], [0, 1])
    assert origins == [str(epoch), str(epoch + 9)]


def test_synthetic_helper_uses_fixed_generator_and_seeded_holdout():
    first = prepare_dataset("synthetic", seed=0)
    again = prepare_dataset("synthetic", seed=0)
    other = prepare_dataset("synthetic", seed=1)
    np.testing.assert_array_equal(first.X_train, again.X_train)
    assert first.metadata["purpose"] == "plumbing only; not benchmark scores"
    assert first.metadata["sample_ids"]["train"] != other.metadata["sample_ids"]["train"]


def test_explicit_missing_records_differ_from_absence_and_do_not_advance_observed_clock():
    sample = (np.array([0., 2., 4.]), np.array([0, 0, 0]), np.array([3., np.nan, 5.]))
    X, mask, _, count = _patch_features([sample], 2, 4, 2.)
    assert count.tolist() == [2]
    assert mask[0, 1, 0]  # stored missing record, active token
    assert not mask[0, 1, 1]  # absent channel record, padding token
    assert X[0, 1, 0, 0] == 0  # zero observed count
    assert X[0, 1, 0, 10] == np.log1p(1)  # one stored record
    assert X[0, 1, 0, 11] == 0  # observed fraction zero
    assert X[0, 2, 0, 9] == np.log1p(4. / 2.)  # NaN at t=2 did not reset clock
    assert not X[0, 1, 1].any()


def test_missing_record_anchors_common_origin_and_survives_sparse_extraction():
    class Frame:
        dims = ("ts_id", "signal_id", "time_id")
        sizes = {"ts_id": 1, "signal_id": 2, "time_id": 2}
        data = SimpleNamespace(coords=np.array([[0, 0], [0, 1], [0, 1]]),
                               data=np.array([np.nan, 4.]), fill_value=np.nan)
        def __getitem__(self, key):
            return SimpleNamespace(values=np.array([100, 103], dtype=np.int64))
    events, origins = _events_from_frame(Frame())
    assert origins == ["100"]
    np.testing.assert_allclose(events[0][0], [0, 3e-9], rtol=0, atol=1e-20)
    assert np.isnan(events[0][2][0])
    assert len(events[0][0]) == 2


def test_raw_channel_scaling_fits_distinct_units_independently():
    events, labels, split, ids = example()
    original = prepare_events(events, labels, split, ids, 2)
    for i, (t, c, v) in enumerate(events):
        changed = v.copy()
        changed[c == 1] = changed[c == 1] * 1000. + 3000.
        events[i] = t, c, changed
    rescaled = prepare_events(events, labels, split, ids, 2)
    np.testing.assert_allclose(original.X_train, rescaled.X_train, atol=1e-6, rtol=1e-6)
    p0, p1 = original.metadata["preprocessing"], rescaled.metadata["preprocessing"]
    assert np.isclose(p1["raw_value_center_by_channel"][1], p0["raw_value_center_by_channel"][1] * 1000. + 3000.)
    assert np.isclose(p1["raw_value_scale_by_channel"][1], p0["raw_value_scale_by_channel"][1] * 1000.)


def test_patch_boundaries_are_stable_under_upstream_timestamp_unit_scaling():
    t = np.array([0., 3., 6., 9., 12.])
    sample = (t, np.zeros(5, dtype=int), np.arange(5.))
    raw, mask, overflow, _ = _patch_features([sample], 1, 8, 3.)
    converted = (t * 1e-9, sample[1], sample[2])
    scaled, scaled_mask, scaled_overflow, _ = _patch_features([converted], 1, 8, 3. * 1e-9)
    np.testing.assert_array_equal(mask, scaled_mask)
    np.testing.assert_array_equal(overflow, scaled_overflow)
    np.testing.assert_allclose(raw, scaled, rtol=1e-12, atol=1e-12)


def test_chunked_feature_moments_match_population_reference_without_mutation(monkeypatch):
    rng = np.random.default_rng(3)
    features = rng.normal(size=(19, 8, 5, len(FEATURE_NAMES)))
    features[..., 0] = 2.0  # constant-feature fallback
    masks = rng.random(features.shape[:-1]) > .3
    idx = np.array([1, 3, 4, 6, 7, 10, 12, 15])
    before = features.copy()
    observed = features[idx][masks[idx]]
    expected_scale = observed.std(axis=0)
    expected_scale = np.where(expected_scale > 1e-12, expected_scale, 1.0)
    # Force multiple gathers, including a last partial chunk.
    monkeypatch.setattr(data_module, "MOMENT_WORKSPACE_BYTES", 2 * features[0].nbytes * 3)
    center, scale = data_module._feature_moments(features, masks, idx)
    np.testing.assert_allclose(center, observed.mean(axis=0), rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(scale, expected_scale, rtol=1e-13, atol=1e-13)
    np.testing.assert_array_equal(features, before)
    masks[:] = False
    center, scale = data_module._feature_moments(features, masks, idx)
    np.testing.assert_array_equal(center, np.zeros(len(FEATURE_NAMES)))
    np.testing.assert_array_equal(scale, np.ones(len(FEATURE_NAMES)))


def test_lazy_event_preparation_does_not_retain_normalized_dataset(monkeypatch):
    events, labels, split, ids = example()
    copies = [tuple(a.copy() for a in sample) for sample in events]
    original = data_module._patch_features

    def checked(events_for_patches, *args):
        assert isinstance(events_for_patches, data_module._MappedEvents)
        assert isinstance(events_for_patches.source, data_module._MappedEvents)
        assert events_for_patches.source.source is events
        return original(events_for_patches, *args)

    monkeypatch.setattr(data_module, "_patch_features", checked)
    result = prepare_events(events, labels, split, ids, 2)
    for actual_sample, before_sample in zip(events, copies):
        for actual, before in zip(actual_sample, before_sample):
            np.testing.assert_array_equal(actual, before)
    estimate = result.metadata["preprocessing"]["memory_preparation"]
    assert estimate["retained_event_bytes"] == sum(a.nbytes for sample in events for a in sample)
    assert estimate["estimated_peak_bytes"] < estimate["budget_bytes"]


def test_insect_size_memory_estimate_is_bounded_without_large_allocation():
    # Pinned HDF5 metadata: N=50,000, C=200, 67,106,800 stored records.
    # Zero-stride views represent this byte accounting without allocating data.
    class InsectEvents:
        def __len__(self):
            return 50_000

        def __iter__(self):
            for i in range(len(self)):
                records = 1342 + (i < 6800)
                yield tuple(np.broadcast_to(np.zeros((), dtype=dtype), (records,))
                            for dtype in (np.float64, np.int64, np.float64))

    estimate = data_module._preparation_memory_estimate(InsectEvents(), 200, 8)
    assert estimate["retained_event_bytes"] == 67_106_800 * 24
    assert 12 * 1024**3 < estimate["estimated_peak_bytes"] < 16 * 1024**3
    assert estimate["budget_bytes"] == 16 * 1024**3


def test_preparation_guard_rejects_before_allocating_patch_tensor(monkeypatch):
    events, labels, split, ids = example()
    monkeypatch.setattr(data_module, "PREPARATION_MEMORY_BUDGET_BYTES", 1)

    def forbidden(*args):
        raise AssertionError("guard must precede patch tensor allocation")

    monkeypatch.setattr(data_module, "_patch_features", forbidden)
    with pytest.raises(MemoryError, match="No events were truncated"):
        prepare_events(events, labels, split, ids, 2)
