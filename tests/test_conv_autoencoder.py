"""Temporal frontend, boundary-safe BN folding, deployment and training checks."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gcfcr.optimized.conv_autoencoder import ConvLatentAutoencoder, _conv_numpy, fit, temporal_features, iq_features


def example_model(layers: int = 4, chunk_size: int = 3) -> ConvLatentAutoencoder:
    rng = np.random.default_rng(633)
    channels = [3, 4, 6, 6, 6, 6, 6, 6][:layers + 1]
    weights = tuple(rng.normal(0, 0.1, (out, inp, 5)).astype(np.float32) for inp, out in zip(channels, channels[1:]))
    biases = tuple(rng.normal(0, 0.1, out).astype(np.float32) for out in channels[1:])
    return ConvLatentAutoencoder(weights, biases, rng.normal(0, 0.1, (4, 2 * channels[-1])), rng.normal(0, 0.1, 4), np.eye(4)[:2], np.array([2, 9]), chunk_size)


class TemporalTests(unittest.TestCase):
    def test_phase_amplitude_and_circular_shift(self):
        rng = np.random.default_rng(17)
        iq = (rng.normal(size=(6, 512)) + 1j * rng.normal(size=(6, 512))).astype(np.complex64)
        features = temporal_features(iq)
        np.testing.assert_allclose(temporal_features(iq * np.complex64(3 * np.exp(0.3j))), features, rtol=3e-6, atol=4e-6)
        np.testing.assert_allclose(temporal_features(np.roll(iq, 47, axis=1)), np.roll(features, 47, axis=2), rtol=3e-6, atol=4e-6)
        np.testing.assert_allclose(features[:, 0].mean(axis=1), 1, atol=2e-7)
        np.testing.assert_array_equal(temporal_features(np.zeros((2, 512), np.complex64)), np.zeros((2, 3, 512)))
        self.assertEqual(temporal_features(np.empty((0, 512), np.complex64)).shape, (0, 3, 512))
        for invalid in (np.zeros((1, 512)), np.zeros((1, 511), np.complex64), np.full((1, 512), complex(np.nan, 0))):
            with self.assertRaises(ValueError):
                temporal_features(invalid)
        with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
            temporal_features(np.full((1, 512), 1e36, np.complex64))

    def test_pre_relu_overflow_is_not_masked(self):
        for sign in (-1, 1):
            with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
                _conv_numpy(np.ones((1, 3, 512), np.float32), np.full((1, 3, 5), sign * 1e38, np.float32), np.zeros(1, np.float32))

    def test_coherent_frontend_and_invalid_mutated_chunk(self):
        rng = np.random.default_rng(53)
        iq = (rng.normal(size=(2, 512)) + 1j * rng.normal(size=(2, 512))).astype(np.complex64)
        features = iq_features(iq)
        np.testing.assert_allclose(features[:, 0] ** 2 + features[:, 1] ** 2, features[:, 2], atol=2e-6)
        np.testing.assert_allclose(iq_features(iq * 4), features, atol=2e-6)
        np.testing.assert_array_equal(iq_features(iq * 0), np.zeros_like(features))
        model = example_model()
        model.frontend = "iq"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "iq.npz"
            model.save(path)
            restored = ConvLatentAutoencoder.load(path)
            self.assertEqual(restored.frontend, "iq")
            np.testing.assert_allclose(restored.encode(iq), model.encode(iq), atol=2e-6)
        model.chunk_size = -1
        with self.assertRaises(ValueError):
            model.encode(iq)
        with self.assertRaises(ValueError):
            model.encode_features(features)

    def test_chunking_export_and_stride_shift_invariance(self):
        rng = np.random.default_rng(18)
        iq = (rng.normal(size=(7, 512)) + 1j * rng.normal(size=(7, 512))).astype(np.complex64)
        for layers in (3, 4, 6):
            model = example_model(layers)
            a = model.encode(iq)
            model.chunk_size = 1
            np.testing.assert_allclose(model.encode(iq), a, rtol=3e-6, atol=3e-6)
            np.testing.assert_allclose(model.encode(np.roll(iq, 2 ** layers, axis=1)), a, rtol=4e-6, atol=4e-6)
            self.assertEqual(model.predict(iq).shape, (7,))
            self.assertEqual(model.scores(iq).shape, (7, 2))
            self.assertEqual(model.encode(np.empty((0, 512), np.complex64)).shape, (0, 4))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "conv.npz"
                model.save(path)
                restored = ConvLatentAutoencoder.load(path)
                np.testing.assert_allclose(restored.encode(iq), a, atol=3e-6)
                with np.load(path, allow_pickle=False) as data:
                    self.assertFalse(any(data[name].dtype.hasobject for name in data.files))
            counts = model.operation_counts()
            self.assertLess(counts["real_flops_estimate"], 1_000_000)
            self.assertEqual(counts["fft_runtime_ops"], 0)
            self.assertEqual(counts["decoder_runtime_ops"], 0)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is optional for deployment")
class TorchParityTests(unittest.TestCase):
    def test_bn_fold_matches_eval_at_all_circular_boundaries(self):
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(61)
        rng = np.random.default_rng(21)
        x = rng.normal(size=(3, 3, 512)).astype(np.float32)
        x[0] = 0
        conv = torch.nn.Conv1d(3, 5, 5, stride=2, padding=2, padding_mode="circular")
        bn = torch.nn.BatchNorm1d(5)
        with torch.no_grad():
            bn.running_mean.copy_(torch.linspace(-1, 1, 5))
            bn.running_var.copy_(torch.linspace(0.4, 2, 5))
            bn.weight.copy_(torch.linspace(0.7, 1.8, 5))
            bn.bias.copy_(torch.linspace(-0.2, 0.3, 5))
        conv.eval()
        bn.eval()
        with torch.inference_mode():
            expected = torch.relu(bn(conv(torch.from_numpy(x)))).numpy()
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            weight = (conv.weight * scale[:, None, None]).numpy()
            bias = ((conv.bias - bn.running_mean) * scale + bn.bias).numpy()
        actual = _conv_numpy(x, weight, bias)
        np.testing.assert_allclose(actual, expected, rtol=3e-6, atol=2e-6)
        np.testing.assert_allclose(actual[:, :, [0, -1]], expected[:, :, [0, -1]], rtol=3e-6, atol=2e-6)

    def test_train_selects_actual_deployed_bank_without_test_data(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(713)

        def signals(per_class):
            labels = np.repeat([0, 1], per_class)
            frequency = np.where(labels == 0, 0.05, 0.22)
            phase = rng.uniform(-np.pi, np.pi, len(labels))
            x = np.exp(1j * (2 * np.pi * frequency[:, None] * np.arange(512) + phase[:, None]))
            x += 0.15 * (rng.normal(size=x.shape) + 1j * rng.normal(size=x.shape))
            return x.astype(np.complex64), labels

        train, labels = signals(20)
        val, val_labels = signals(6)
        updates = []
        model, history = fit(train, labels, val, val_labels, epochs=4, channels=(4, 6, 6), latent_dim=4, learning_rate=0.01, batch_size=10, progress=updates.append)
        self.assertEqual(len(history), 4)
        self.assertEqual(len(updates), 4)
        accuracy = float(np.mean(model.predict(val) == val_labels))
        self.assertEqual(accuracy, max(row["validation_accuracy"] for row in history))
        self.assertGreaterEqual(accuracy, 0.9)
        with self.assertRaises(ValueError):
            fit(train, labels, val, val_labels, epochs=0)


if __name__ == "__main__":
    unittest.main()
