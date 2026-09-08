"""Coherent accumulation, phase invariance, signed BN gain and safe export tests."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

from gcfcr.optimized.coherent_autoencoder import CoherentAutoencoder, _complex_block, _power_numpy, coherent_features, fit


def example_model():
    rng = np.random.default_rng(381)
    channels = [3, 4, 5, 5]
    return CoherentAutoencoder(rng.normal(0, 0.1, (3, 17)), rng.normal(0, 0.1, (3, 17)), np.array([1, -0.7, 0.9]), np.array([0.1, 0.4, -0.1]), tuple(rng.normal(0, 0.1, (out, inp, 5)) for inp, out in zip(channels, channels[1:])), tuple(rng.normal(0, 0.1, width) for width in channels[1:]), rng.normal(0, 0.1, (4, 10)), rng.normal(0, 0.1, 4), np.eye(4)[:2], np.array([2, 9]), chunk_size=3)


class CoherentTests(unittest.TestCase):
    def test_global_phase_and_stride_shift_invariance(self):
        rng = np.random.default_rng(9)
        iq = (rng.normal(size=(7, 512)) + 1j * rng.normal(size=(7, 512))).astype(np.complex64)
        model = example_model()
        expected = model.encode(iq)
        np.testing.assert_allclose(model.encode(iq * np.complex64(2 * np.exp(1.37j))), expected, atol=5e-6, rtol=5e-6)
        np.testing.assert_allclose(model.encode(np.roll(iq, 16, axis=1)), expected, atol=5e-6, rtol=5e-6)
        model.chunk_size = 1
        np.testing.assert_allclose(model.encode(iq), expected, atol=5e-6, rtol=5e-6)
        model.chunk_size = -1
        with self.assertRaises(ValueError):
            model.encode(iq)
        self.assertEqual(coherent_features(np.empty((0, 512), np.complex64)).shape, (0, 2, 512))
        np.testing.assert_array_equal(coherent_features(np.zeros((2, 512), np.complex64)), np.zeros((2, 2, 512)))

    def test_cached_complex_filters_are_immutable(self):
        model = example_model()
        with self.assertRaises(ValueError):
            model.complex_real[0, 0] = 0
        with self.assertRaises(ValueError):
            model.complex_imag.fill(0)
        with self.assertRaises(AttributeError):
            model.complex_real = np.zeros_like(model.complex_real)

    def test_serialization_accounting_and_overflow(self):
        model = example_model()
        x = np.ones((2, 512), np.complex64)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coherent.npz"
            model.save(path)
            restored = CoherentAutoencoder.load(path)
            np.testing.assert_array_equal(restored.predict(x), model.predict(x))
            np.testing.assert_allclose(restored.encode(x), model.encode(x), atol=1e-7)
            with np.load(path, allow_pickle=False) as arrays:
                self.assertFalse(any(arrays[name].dtype.hasobject for name in arrays.files))
        counts = model.operation_counts()
        self.assertEqual(counts["fft_runtime_ops"], 0)
        self.assertEqual(counts["decoder_runtime_ops"], 0)
        self.assertGreater(model.storage_bytes, 0)
        for gain in (-1, 1):
            with np.errstate(over="ignore", invalid="ignore"), self.assertRaises(ValueError):
                _power_numpy(np.full((1, 2, 512), 10, np.float32), np.full((2, 2, 17), 1e38, np.float32), np.array([gain], np.float32), np.zeros(1, np.float32))


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch is optional for deployment")
class TorchParityTests(unittest.TestCase):
    def test_signed_power_bn_matches_torch_at_boundaries(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(15)
        x = rng.normal(size=(3, 2, 512)).astype(np.float32)
        x[0] = 0
        real, imag = rng.normal(0, 0.1, (3, 17)).astype(np.float32), rng.normal(0, 0.1, (3, 17)).astype(np.float32)
        block = _complex_block(real, imag)
        bn = torch.nn.BatchNorm1d(3)
        with torch.no_grad():
            bn.weight.copy_(torch.tensor([1.2, -0.8, 0.4]))
            bn.bias.copy_(torch.tensor([0.1, 0.7, -0.2]))
            bn.running_mean.copy_(torch.tensor([0.5, 0.2, -0.3]))
            bn.running_var.copy_(torch.tensor([0.9, 0.4, 1.2]))
        bn.eval()
        with torch.inference_mode():
            response = torch.nn.functional.conv1d(torch.nn.functional.pad(torch.from_numpy(x), (8, 8), mode="circular"), torch.from_numpy(block), stride=2)
            expected = torch.relu(bn(response[:, :3].square() + response[:, 3:].square())).numpy()
            gain = (bn.weight / torch.sqrt(bn.running_var + bn.eps)).numpy()
            bias = (bn.bias - bn.weight / torch.sqrt(bn.running_var + bn.eps) * bn.running_mean).numpy()
        actual = _power_numpy(x, block, gain, bias)
        np.testing.assert_allclose(actual, expected, atol=3e-6, rtol=3e-6)
        np.testing.assert_allclose(actual[:, :, [0, -1]], expected[:, :, [0, -1]], atol=3e-6, rtol=3e-6)

    def test_train_and_actual_deployed_validation_selection(self):
        import torch
        torch.set_num_threads(1)
        rng = np.random.default_rng(233)
        def signals(count):
            labels = np.repeat([0, 1], count)
            frequencies = np.where(labels == 0, 0.06, 0.24)
            phase = rng.uniform(-np.pi, np.pi, len(labels))
            iq = np.exp(1j * (2 * np.pi * frequencies[:, None] * np.arange(512) + phase[:, None]))
            iq += 0.2 * (rng.normal(size=iq.shape) + 1j * rng.normal(size=iq.shape))
            return iq.astype(np.complex64), labels
        train, labels = signals(20)
        val, vy = signals(6)
        model, history = fit(train, labels, val, vy, epochs=4, complex_channels=3, kernel_size=17, channels=(4, 5, 5), latent_dim=4, batch_size=10, learning_rate=0.01)
        accuracy = float(np.mean(model.predict(val) == vy))
        self.assertEqual(accuracy, max(row["validation_accuracy"] for row in history))
        self.assertGreaterEqual(accuracy, 0.9)
        with self.assertRaises(ValueError):
            fit(train, labels, val, vy, kernel_size=32)


if __name__ == "__main__":
    unittest.main()
