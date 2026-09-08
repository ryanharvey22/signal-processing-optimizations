from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import numpy as np
from gcfcr.optimized.baselines import (FeaturePrototypeMatcher, WaveformMatchedFilter,
                                      select_bank_indices)


class RawMatchedFilterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(21)
        self.iq = (rng.normal(size=(24, 32)) + 1j * rng.normal(size=(24, 32))).astype(np.complex64)
        self.y = np.repeat(np.arange(3), 8)
        self.indices = select_bank_indices(self.y, 3, seed=17)

    def test_quality_selection_uses_only_training_metadata(self):
        quality = np.tile(np.arange(8), 3)
        selected = select_bank_indices(self.y, 2, quality=quality)
        np.testing.assert_array_equal(selected, [6, 7, 14, 15, 22, 23])
        np.testing.assert_array_equal(selected, select_bank_indices(self.y, 2, quality=quality))
        for invalid in (quality[:-1], np.full(24, np.nan)):
            with self.assertRaises(ValueError):
                select_bank_indices(self.y, 2, quality=invalid)

    def test_aligned_blas_agrees_with_scalar_correlation(self):
        model = WaveformMatchedFilter(bank_indices=self.indices, query_chunk=2, template_chunk=2).fit(self.iq, self.y)
        expected = np.zeros((5, 3))
        for row, query in enumerate(self.iq[:5]):
            for template, label in zip(model.bank, model.bank_labels):
                expected[row, label] = max(expected[row, label], abs(np.vdot(template, query)) ** 2)
        np.testing.assert_allclose(model.scores(self.iq[:5]), expected, atol=2e-5, rtol=1e-5)
        np.testing.assert_array_equal(model.predict(self.iq[:5]), model.predict(self.iq[:5] * np.complex64(3j)))

    def test_circular_fft_agrees_with_exhaustive_delay_search(self):
        model = WaveformMatchedFilter(mode="fft", bank_indices=self.indices, query_chunk=2, template_chunk=2).fit(self.iq, self.y)
        expected = np.zeros((5, 3))
        for row, query in enumerate(self.iq[:5]):
            for template, label in zip(model.bank, model.bank_labels):
                score = max(abs(np.vdot(np.roll(template, shift), query)) ** 2 for shift in range(32))
                expected[row, label] = max(expected[row, label], score)
        np.testing.assert_allclose(model.scores(self.iq[:5]), expected, atol=3e-5, rtol=1e-5)
        np.testing.assert_allclose(model.scores(np.roll(self.iq[:5], 11, axis=1)), expected, atol=3e-5, rtol=1e-5)
        huge_chunks = WaveformMatchedFilter(mode="fft", bank_indices=self.indices, query_chunk=100, template_chunk=100).fit(self.iq, self.y)
        np.testing.assert_allclose(huge_chunks.scores(self.iq), model.scores(self.iq), atol=3e-5)

    def test_large_finite_inputs_fail_explicitly_and_non_radix_cost_is_unavailable(self):
        for mode in ("aligned", "fft"):
            model = WaveformMatchedFilter(mode=mode).fit(self.iq, self.y)
            with self.assertRaises(ValueError):
                model.predict(self.iq * np.float32(1e20))
            with self.assertRaises(ValueError):
                WaveformMatchedFilter(mode=mode).fit(self.iq * np.float32(1e20), self.y)
        odd = WaveformMatchedFilter(mode="fft").fit(self.iq[:, :30], self.y)
        self.assertTrue(np.isfinite(odd.scores(self.iq[:, :30])).all())
        with self.assertRaises(ValueError):
            odd.operation_counts(30)

    def test_roundtrip_cost_and_validation(self):
        for mode in ("aligned", "fft"):
            model = WaveformMatchedFilter(mode=mode, bank_indices=self.indices).fit(self.iq, self.y)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "raw.npz"
                model.save(path)
                loaded = WaveformMatchedFilter.load(path)
                np.testing.assert_array_equal(loaded.predict(self.iq), model.predict(self.iq))
                self.assertEqual(loaded.storage_bytes, model.storage_bytes)
            self.assertGreater(model.operation_counts(32)["real_flops_estimate"], 0)
            self.assertEqual(model.predict(np.empty((0, 32), np.complex64)).shape, (0,))
            with self.assertRaises(ValueError):
                model.predict(np.ones((2, 16), np.complex64))
        for invalid in (np.array([1, 1]), np.array([-1, 2]), np.array([1.0, 2.0]), np.array([100])):
            with self.assertRaises(ValueError):
                WaveformMatchedFilter(bank_indices=invalid).fit(self.iq, self.y)
        with self.assertRaises(ValueError):
            WaveformMatchedFilter().fit(self.iq, self.y + 1)


class FeatureControlTests(unittest.TestCase):
    def test_pca_and_spectral_frozen_roundtrip(self):
        rng = np.random.default_rng(8)
        iq = (rng.normal(size=(30, 128)) + 1j * rng.normal(size=(30, 128))).astype(np.complex64)
        y = np.repeat(np.arange(3), 10)
        for pca_dim, standardize in ((None, False), (8, False), (None, True), (8, True)):
            model = FeaturePrototypeMatcher(bins=32, prototypes_per_class=2, pca_dim=pca_dim,
                                            query_chunk=3, standardize=standardize).fit(iq, y)
            predictions = model.predict(iq)
            np.testing.assert_array_equal(model.predict(np.roll(iq * np.complex64(2j), 17, axis=1)), predictions)
            self.assertTrue(np.isfinite(model.scores(np.zeros((2, 128), np.complex64))).all())
            self.assertEqual(model.predict(np.empty((0, 128), np.complex64)).shape, (0,))
            self.assertGreater(model.operation_counts(128)["real_flops_estimate"], 0)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "feature.npz"
                model.save(path)
                loaded = FeaturePrototypeMatcher.load(path)
                np.testing.assert_array_equal(loaded.predict(iq), predictions)
                self.assertEqual(loaded.storage_bytes, model.storage_bytes)
                np.testing.assert_array_equal(loaded.components, model.components)
                np.testing.assert_array_equal(loaded.scale, model.scale)
                self.assertEqual(loaded.standardize, standardize)
        with self.assertRaises(ValueError):
            FeaturePrototypeMatcher(bins=32, pca_dim=31).fit(iq, y)


if __name__ == "__main__":
    unittest.main()
