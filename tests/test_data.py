"""Offline data safety tests; no downloaded datasets or model dependencies."""

from dataclasses import replace
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from imts_benchmark.data import (
    DATASETS, MANIFEST, _dense_with_guard, load_data, make_synthetic,
    subset_for_smoke,
)


class FakeFrame:
    dims = ("ts_id", "signal_id", "time_id")

    def __init__(self):
        self.X = np.arange(6 * 12 * 24, dtype=float).reshape(6, 12, 24)
        self.X[0, 0, 3] = np.nan
        self.y = np.array([0, 1, 0, 1, 0, 1])
        self.split = np.array(["train", "val", "validation", "train", "test", "test"])
        self.ids = np.array(["a", "b", "c", "d", "e", "f"])
        # Deliberately huge global timestamp axis, which must never be densified.
        self.sizes = {"ts_id": 6, "signal_id": 12, "time_id": 10**12}
        self.dense_calls = 0
        self.irr = self

    def get_task(self):
        return {"task": "classification", "target": "label", "split": "split_default"}

    def get_task_target_and_split(self):
        return self.y, self.split

    def reset_time_index(self):
        return (SimpleNamespace(shape=self.X.shape, dtype=self.X.dtype),
                SimpleNamespace(shape=(6, 1, 24), dtype=np.dtype("float64")))

    def to_dense(self):
        self.dense_calls += 1
        return self.X.copy(), np.zeros((6, 1, 24))

    def __getitem__(self, key):
        if key != "ts_id":
            raise KeyError(key)
        return SimpleNamespace(values=self.ids)


def fake_pyrregular(frame):
    spec = next(d for d in MANIFEST["datasets"] if d["name"] == "JapaneseVowels")
    repository = SimpleNamespace(
        DATA_REVISION=MANIFEST["data_revision"], DATA_VERSION="data-v1",
        REPOSITORY=SimpleNamespace(path=Path("/original-cache"),
                                   registry={spec["file"]: spec["sha256"]}),
    )
    return SimpleNamespace(repository=repository, load_dataset=lambda _: {"data": frame})


class OfficialDataSafetyTests(unittest.TestCase):
    def test_official_non_test_split_and_order_preserved(self):
        frame = FakeFrame()
        with patch.dict(sys.modules, {"pyrregular": fake_pyrregular(frame)}):
            data = load_data("JapaneseVowels")
        self.assertEqual(data.train_ids.tolist(), ["a", "b", "c", "d"])
        self.assertEqual(data.test_ids.tolist(), ["e", "f"])
        np.testing.assert_array_equal(data.X_train, frame.X[:4])
        np.testing.assert_array_equal(data.X_test, frame.X[4:])
        self.assertTrue(np.isnan(data.X_train[0, 0, 3]))
        self.assertEqual(data.metadata["train_source_indices"], [0, 1, 2, 3])
        self.assertFalse(data.metadata["smoke_subset"])

    def test_unknown_split_rejected_before_dense_allocation(self):
        frame = FakeFrame()
        frame.split[0] = "mystery"
        with patch.dict(sys.modules, {"pyrregular": fake_pyrregular(frame)}):
            with self.assertRaisesRegex(ValueError, "Unknown official split"):
                load_data("JapaneseVowels")
        self.assertEqual(frame.dense_calls, 0)

    def test_no_test_sample_moved_to_cover_unseen_label(self):
        frame = FakeFrame()
        frame.y[-1] = 9
        with patch.dict(sys.modules, {"pyrregular": fake_pyrregular(frame)}):
            with self.assertRaisesRegex(ValueError, "Do not move test"):
                load_data("JapaneseVowels")
        self.assertEqual(frame.dense_calls, 0)

    def test_empty_test_split_rejected(self):
        frame = FakeFrame()
        frame.split[:] = "train"
        with patch.dict(sys.modules, {"pyrregular": fake_pyrregular(frame)}):
            with self.assertRaisesRegex(ValueError, "held-out test"):
                load_data("JapaneseVowels")

    def test_duplicate_ids_across_splits_rejected(self):
        frame = FakeFrame()
        frame.ids[-1] = frame.ids[0]
        with patch.dict(sys.modules, {"pyrregular": fake_pyrregular(frame)}):
            with self.assertRaisesRegex(ValueError, "duplicate IDs"):
                load_data("JapaneseVowels")

    def test_memory_limit_rejects_before_to_dense(self):
        frame = FakeFrame()
        with self.assertRaisesRegex(MemoryError, "No samples or timesteps were truncated"):
            _dense_with_guard(frame, max_dense_gib=1e-9)
        self.assertEqual(frame.dense_calls, 0)

    def test_memory_preflight_uses_local_sample_axis(self):
        frame = FakeFrame()
        X, metadata = _dense_with_guard(frame, max_dense_gib=0.001)
        self.assertEqual(X.shape, (6, 12, 24))
        self.assertEqual(metadata["estimated_dense_load_peak_bytes"], 2 * X.nbytes + 6 * 24 * 8)
        self.assertEqual(frame.dense_calls, 1)

    def test_cache_override_restored_after_failure(self):
        frame = FakeFrame()
        module = fake_pyrregular(frame)
        def fail_load(_):
            self.assertNotEqual(module.repository.REPOSITORY.path, Path("/original-cache"))
            raise OSError("simulated failed download")
        module.load_dataset = fail_load
        with TemporaryDirectory() as folder:
            with patch.dict(sys.modules, {"pyrregular": module}):
                with self.assertRaises(OSError):
                    load_data("JapaneseVowels", cache_dir=folder)
        self.assertEqual(module.repository.REPOSITORY.path, Path("/original-cache"))

    def test_manifest_has_exactly_18_multivariate_datasets(self):
        self.assertEqual(len(DATASETS), 18)
        self.assertNotIn("Abf", DATASETS)
        self.assertTrue(all(d["channels"] > 1 and len(d["sha256"]) == 64
                            for d in MANIFEST["datasets"]))


class SmokeSplitSafetyTests(unittest.TestCase):
    def test_synthetic_splits_are_balanced_and_disjoint(self):
        data = make_synthetic()
        self.assertEqual(data.X_train.shape, (40, 3, 32))
        self.assertEqual(data.X_test.shape, (16, 3, 32))
        self.assertEqual(np.bincount(data.y_train).tolist(), [20, 20])
        self.assertEqual(np.bincount(data.y_test).tolist(), [8, 8])
        self.assertFalse(set(data.train_ids) & set(data.test_ids))
        self.assertTrue(np.isnan(data.X_train).any())

    def test_smoke_subsets_stay_in_existing_splits_and_preserve_order(self):
        data = make_synthetic()
        smoke = subset_for_smoke(data, max_train=9, max_test=5, seed=7)
        again = subset_for_smoke(data, max_train=9, max_test=5, seed=7)
        self.assertEqual(len(smoke.y_train), 9)
        self.assertEqual(len(smoke.y_test), 5)
        self.assertEqual(set(smoke.y_train), set(data.y_train))
        self.assertLessEqual(set(smoke.train_ids), set(data.train_ids))
        self.assertLessEqual(set(smoke.test_ids), set(data.test_ids))
        self.assertFalse(set(smoke.train_ids) & set(smoke.test_ids))
        self.assertEqual(smoke.train_ids.tolist(), sorted(smoke.train_ids.tolist()))
        np.testing.assert_array_equal(smoke.train_ids, again.train_ids)
        np.testing.assert_array_equal(smoke.test_ids, again.test_ids)
        self.assertFalse(data.metadata["smoke_subset"])
        self.assertTrue(smoke.metadata["smoke_subset"])

    def test_rare_training_class_retained(self):
        data = make_synthetic()
        labels = np.zeros(40, dtype=int)
        labels[-1] = 1
        data = replace(data, y_train=labels)
        smoke = subset_for_smoke(data, max_train=2)
        self.assertEqual(set(smoke.y_train), {0, 1})

    def test_cannot_fit_classes_in_smoke_budget_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "cannot retain all 2 training classes"):
            subset_for_smoke(make_synthetic(), max_train=1)


if __name__ == "__main__":
    unittest.main()
