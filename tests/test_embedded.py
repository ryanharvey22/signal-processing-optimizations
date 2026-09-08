"""Embedded export contracts and native C99 parity when a C compiler is present."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np

from gcfcr.optimized.autoencoder import LatentAutoencoder

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("export_embedded", ROOT / "scripts/export_embedded.py")
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)
COMPILER = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")


def model(hidden: int = 48) -> LatentAutoencoder:
    rng = np.random.default_rng(621)
    dimensions = [128, *([hidden] if hidden else []), 16]
    weights = tuple(rng.normal(0, 0.12, (target, source)).astype(np.float32) for source, target in zip(dimensions, dimensions[1:]))
    biases = tuple(rng.normal(0, 0.05, target).astype(np.float32) for target in dimensions[1:])
    codes = rng.normal(size=(5, 16)).astype(np.float32)
    return LatentAutoencoder(weights, biases, codes, np.array([0, 1, 2, 3, 4]))


class EmbeddedTests(unittest.TestCase):
    def test_export_memory_and_immutable_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "ogae_model.h"
            report = EXPORT.export_model(model(), target)
            text = target.read_text()
            self.assertEqual(report["workspace_bytes"], 4 * (1024 + 128 + 48 + 16))
            self.assertLess(report["model_numeric_flash_bytes"], 32 * 1024)
            self.assertIn("static const ogae_model ogae_exported_model", text)
            self.assertIn("static const float ogae_exported_twiddle_real[256]", text)
            self.assertNotIn("malloc", text)
            golden = Path(directory) / "ogae_golden.h"
            EXPORT.export_golden(model(), EXPORT.diagnostic_iq(8), golden)
            self.assertIn("#define OGAE_EXPORTED_GOLDEN_CASES 8u", golden.read_text())
            with self.assertRaises(ValueError):
                EXPORT.export_model(model(), target, symbol="bad-name")
            with self.assertRaises(ValueError):
                EXPORT.export_model(model(), target, samples=256)

    @unittest.skipUnless(COMPILER, "native C compiler is required for embedded kernel parity")
    def test_c99_matches_numpy_for_linear_and_hidden_encoders(self):
        for hidden in (0, 48):
            with self.subTest(hidden=hidden), tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                candidate = model(hidden)
                refs = EXPORT.diagnostic_iq(5)
                labels = np.arange(5)
                EXPORT.export_model(candidate, directory / "ogae_model.h", aligned_iq=refs, aligned_labels=labels)
                EXPORT.export_golden(candidate, EXPORT.diagnostic_iq(16), directory / "ogae_golden.h", aligned_iq=refs, aligned_labels=labels)
                executable = directory / "parity"
                subprocess.run([COMPILER, "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-pedantic", "-I", str(ROOT / "embedded"), "-I", str(directory), str(ROOT / "embedded/ogae.c"), str(ROOT / "embedded/test_inference.c"), "-lm", "-o", str(executable)], check=True, capture_output=True, text=True)
                result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
                self.assertIn("embedded parity passed", result.stdout)


if __name__ == "__main__":
    unittest.main()
