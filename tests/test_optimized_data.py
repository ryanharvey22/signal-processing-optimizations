from __future__ import annotations
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gcfcr.optimized.data import (cap_indices, load_experiment_data, split_indices,
                                  synthetic_radar_fixture)


class SplitTests(unittest.TestCase):
    def test_three_way_caps_do_not_change_membership(self):
        partitions = split_indices(1000, seed=3)
        self.assertEqual([len(partitions[s]) for s in ("train", "val", "test")], [800, 100, 100])
        self.assertEqual(len(np.unique(np.concatenate(list(partitions.values())))), 1000)
        for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
            self.assertFalse(np.intersect1d(partitions[left], partitions[right]).size)
        small = cap_indices(partitions["train"], 40, seed=9)
        large = cap_indices(partitions["train"], 80, seed=9)
        self.assertTrue(set(small).issubset(large))
        self.assertGreater(int(small.max()), 500)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                cap_indices(partitions["train"], invalid)
        for args in ({"train_fraction": 0.9}, {"val_fraction": 0}, {"groups": np.arange(2)}):
            with self.assertRaises(ValueError):
                split_indices(100, **args)

    def test_replicas_of_known_group_stay_together(self):
        groups = np.repeat(np.arange(100), 3)
        partitions = split_indices(len(groups), groups=groups)
        sets = [set(groups[rows]) for rows in partitions.values()]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(sum(len(rows) for rows in partitions.values()), len(groups))

    def test_fixture_is_reproducible_and_manifest_is_not_radchar(self):
        kwargs = dict(seed=9, synthetic_total=300, train_cap=100, val_cap=20, test_cap=20)
        data = load_experiment_data(**kwargs)
        again = load_experiment_data(**kwargs)
        self.assertEqual(data.manifest, again.manifest)
        self.assertTrue(data.manifest["synthetic"])
        self.assertEqual(data.train.iq.dtype, np.complex64)
        self.assertEqual(data.train.iq.shape, (100, 512))
        self.assertEqual(set(data.train.y), set(range(5)))
        self.assertFalse(set(data.train.row_ids) & set(data.test.row_ids))
        self.assertFalse(set(data.val.group_ids) & set(data.test.group_ids))
        # An altered cap selects nested members without changing the holdout.
        changed = load_experiment_data(**dict(kwargs, train_cap=200))
        np.testing.assert_array_equal(changed.test.iq, data.test.iq)
        self.assertTrue(set(data.train.row_ids).issubset(changed.train.row_ids))
        altered = load_experiment_data(**dict(kwargs, seed=10))
        self.assertNotEqual(altered.manifest["source_sha256"], data.manifest["source_sha256"])


@unittest.skipUnless(importlib.util.find_spec("h5py"), "optional HDF5 dependency unavailable")
class Hdf5Tests(unittest.TestCase):
    def make_file(self, path):
        import h5py
        labels = np.zeros(200, dtype=[("index", "i8"), ("signal_type", "i8"),
            ("signal_to_noise_ratio", "i8")])
        labels["index"] = np.arange(200)
        labels["signal_type"] = np.repeat(np.arange(5), 40)
        with h5py.File(path, "w") as f:
            f["labels"] = labels
            f["iq"] = np.broadcast_to(np.arange(200)[:, None], (200, 512)).astype(np.complex64)

    def test_real_file_caps_read_correct_rows_and_distinct_holdouts(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.h5"
            self.make_file(path)
            data = load_experiment_data(path, train_cap=70, val_cap=10, test_cap=12)
            self.assertFalse(data.manifest["synthetic"])
            self.assertIn("row-disjoint only", data.manifest["independence"])
            for split in (data.train, data.val, data.test):
                np.testing.assert_array_equal(split.iq[:, 0].real, split.row_ids)
                self.assertIsNone(split.group_ids)
            self.assertEqual(set(data.train.y), set(range(5)))
            if importlib.util.find_spec("torch"):
                from gcfcr.data.radchar import RadCharDataset
                train = RadCharDataset(path, split="train", max_samples=70)
                val = RadCharDataset(path, split="val", max_samples=10)
                test = RadCharDataset(path, split="test", max_samples=12)
                np.testing.assert_array_equal(train._indices, data.train.row_ids)
                self.assertFalse(set(val._indices) & set(test._indices))
                self.assertEqual(train._iq.shape, (70, 512))
                tensor, meta = train[0]
                self.assertEqual(float(tensor[0].real), meta["row_id"])
                self.assertEqual(meta["index"], meta["row_id"])

    def test_known_hdf5_groups_and_invalid_schema(self):
        import h5py
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sample.h5"
            self.make_file(path)
            with h5py.File(path, "a") as f:
                f["group_ids"] = np.repeat(np.arange(100), 2)
            data = load_experiment_data(path, train_cap=None, val_cap=None, test_cap=None)
            self.assertFalse(set(data.train.group_ids) & set(data.test.group_ids))
            self.assertEqual(data.manifest["independence"], "recorded-group-disjoint")
            with h5py.File(path, "a") as f:
                f["iq"][0, 0] = np.inf
            with self.assertRaises(ValueError):
                load_experiment_data(path, train_cap=None, val_cap=None, test_cap=None)


if __name__ == "__main__":
    unittest.main()
