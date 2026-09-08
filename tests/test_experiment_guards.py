"""Interrupted evaluation must not permit further model or baseline selection."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from optimized_filter import ensure_tunable

class GuardTests(unittest.TestCase):
    def test_interrupted_evaluation_locks_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "experiment.json").write_text(json.dumps({"test_used_for_selection": False}))
            ensure_tunable(directory)
            (directory / "test_evaluation_started.json").write_text("{}")
            with self.assertRaises(ValueError):
                ensure_tunable(directory)

    def test_inherited_test_exposure_locks_derived_selection(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source, derived = directory / "source", directory / "derived"
            source.mkdir()
            derived.mkdir()
            (source / "test_evaluation_started.json").write_text("{}")
            manifest = {"test_used_for_selection": False, "inherited_validation_experiment": str(source)}
            with self.assertRaises(ValueError):
                ensure_tunable(derived, manifest)
            with self.assertRaises(ValueError):
                ensure_tunable(derived, {"test_used_for_selection": True})
