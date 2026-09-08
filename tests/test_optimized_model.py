"""Math, export, bounded bank, and optional training checks for the new model."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gcfcr.optimized.autoencoder import LatentAutoencoder, build_prototype_bank, fit
from gcfcr.optimized.features import feature_operation_counts, spectral_features


class SpectralFeaturesTests(unittest.TestCase):
    def test_invariances_and_energy(self):
        rng = np.random.default_rng(91)
        iq = (rng.normal(size=(7, 512)) + 1j * rng.normal(size=(7, 512))).astype(np.complex64)
        reference = spectral_features(iq)
        transformed = np.roll(iq * np.complex64(3 * np.exp(0.73j)), 79, axis=1)
        np.testing.assert_allclose(spectral_features(transformed), reference, atol=3e-6, rtol=3e-6)
        self.assertEqual(reference.dtype, np.float32)
        np.testing.assert_array_equal(spectral_features(np.zeros((2, 512), np.complex64)), np.zeros((2, 128)))
        self.assertEqual(spectral_features(np.empty((0, 512), np.complex64)).shape, (0, 128))

    def test_validation(self):
        for array in (np.zeros((2, 512)), np.zeros(512, np.complex64), np.full((1, 512), np.nan + 0j), np.zeros((1, 0), np.complex64)):
            with self.assertRaises(ValueError):
                spectral_features(array)
        for bins in (0, True, 3, 1.5):
            with self.assertRaises(ValueError):
                spectral_features(np.zeros((1, 512), np.complex64), bins)
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
            spectral_features(np.full((1, 512), 1e36, np.complex64))
        self.assertEqual(feature_operation_counts(512, 128)["fft_real_ops_estimate"], 23040)
        self.assertEqual(feature_operation_counts(96, 32)["fft_real_ops_estimate"], "unavailable")


class PrototypeTests(unittest.TestCase):
    def test_deterministic_multimodal_and_tiled(self):
        x = np.array([[1, 0], [0.95, 0.05], [-1, 0], [-0.95, 0.05], [0, 1], [0.05, 0.95]], np.float32)
        y = np.array([2, 2, 2, 2, 8, 8])
        a, labels = build_prototype_bank(x, y, 2, chunk_size=1)
        b, other = build_prototype_bank(x, y, 2, chunk_size=1024)
        np.testing.assert_allclose(a, b, atol=1e-6)
        np.testing.assert_array_equal(labels, other)
        self.assertEqual(a.shape, (4, 2))
        self.assertTrue(np.allclose(np.linalg.norm(a, axis=1), 1))
        self.assertLess(float(a[0] @ a[1]), -0.9)

    def test_zero_identical_and_invalid(self):
        bank, labels = build_prototype_bank(np.zeros((3, 4), np.float32), np.array([0, 0, 1]), 4)
        self.assertEqual(bank.shape, (3, 4))
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
            build_prototype_bank(np.full((3, 4), 1e38, np.float32), np.array([0, 0, 1]))
        self.assertTrue(np.isfinite(bank).all())
        for x, y in ((np.empty((0, 4)), np.empty(0, np.int64)), (np.ones((2, 4)), np.array([1.5, 2.5])), (np.ones((2, 4)), np.array([-1, 1]))):
            with self.assertRaises(ValueError):
                build_prototype_bank(x, y)


class DeploymentTests(unittest.TestCase):
    def model(self):
        return LatentAutoencoder((np.eye(4, dtype=np.float32),), (np.zeros(4, np.float32),), np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32), np.array([3, 9]))

    def test_predict_label_preservation_and_save(self):
        model = self.model()
        features = np.array([[2, 0, 0, 0], [0, 4, 0, 0]], np.float32)
        np.testing.assert_array_equal(model.predict_features(features), [3, 9])
        self.assertEqual(model.match_codes(model.encode_features(features)).shape, (2, 2))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.npz"
            model.save(path)
            with np.load(path, allow_pickle=False) as data:
                self.assertFalse(any(data[key].dtype.hasobject for key in data.files))
            restored = LatentAutoencoder.load(path)
            np.testing.assert_array_equal(restored.predict_features(features), [3, 9])
            for a, b in zip(model.weights, restored.weights):
                np.testing.assert_array_equal(a, b)
        self.assertEqual(model.operation_counts(512)["decoder_runtime_ops"], 0)
        self.assertGreater(model.storage_bytes, 0)

    def test_relu_and_ownership(self):
        weight = np.array([[1, 0], [0, 1]], np.float32)
        model = LatentAutoencoder((weight, weight), (np.zeros(2), np.zeros(2)), np.eye(2), np.array([0, 1]))
        weight[:] = 0
        np.testing.assert_allclose(model.encode_features(np.array([[-1, 4]], np.float32)), [[0, 1]])
        self.assertEqual(model.predict_features(np.empty((0, 2), np.float32)).shape, (0,))
        with self.assertRaises(ValueError):
            model.encode_features(np.ones((1, 3), np.float32))
        with self.assertRaises(ValueError):
            model.encode_features(np.full((1, 2), np.nan, np.float32))

    def test_pre_relu_overflow_is_not_masked(self):
        for sign in (-1, 1):
            model = LatentAutoencoder((np.full((1, 2), sign * 1e38, np.float32), np.ones((2, 1), np.float32)), (np.zeros(1), np.ones(2)), np.eye(2), np.array([0, 1]))
            with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
                model.encode_features(np.full((1, 2), 10, np.float32))

    def test_invalid_weights_bank(self):
        with self.assertRaises(ValueError):
            LatentAutoencoder((np.eye(2),), (np.zeros(3),), np.eye(2), np.array([0, 1]))
        with self.assertRaises(ValueError):
            LatentAutoencoder((np.eye(2),), (np.zeros(2),), np.empty((0, 2)), np.empty(0, np.int64))


@unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is optional for deployment")
class TrainingTests(unittest.TestCase):
    def test_reproducible_training_validation_selection_and_no_validation_fit(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(4)
        y = np.repeat([0, 1], 24)
        x = rng.normal(0, 0.2, (48, 8)).astype(np.float32)
        x[:, 0] += np.where(y == 0, -2, 2)
        vx, vy = x[::3].copy(), y[::3].copy()
        updates = []
        model, history = fit(x, y, vx, vy, epochs=16, hidden_dim=0, latent_dim=4, seed=9, batch_size=16, learning_rate=0.02, progress=updates.append)
        self.assertEqual(len(history), 16)
        self.assertEqual(len(updates), 16)
        self.assertGreaterEqual(max(row["validation_accuracy"] for row in history), 0.9)
        self.assertEqual(float(np.mean(model.predict_features(vx) == vy)), max(row["validation_accuracy"] for row in history))
        original, _ = fit(x, y, vx, vy, epochs=1, hidden_dim=0, latent_dim=4, seed=9, batch_size=16)
        changed, _ = fit(x, y, vx + 200, vy, epochs=1, hidden_dim=0, latent_dim=4, seed=9, batch_size=16)
        np.testing.assert_array_equal(original.weights[0], changed.weights[0])
        np.testing.assert_array_equal(original.prototypes, changed.prototypes)
        for kwargs in ({"epochs": 0}, {"reconstruction_weight": 0}, {"augmentation_std": -1}, {"learning_rate": float("nan")}):
            with self.assertRaises(ValueError):
                fit(x, y, vx, vy, **kwargs)


if __name__ == "__main__":
    unittest.main()
