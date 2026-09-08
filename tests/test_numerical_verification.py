"""Near-tie reporting must never weaken primary or full-score checks."""
import sys
from pathlib import Path
import numpy as np
import pytest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from numerical_verification import verify_scores


def test_auxiliary_near_tie_is_explicit_and_primary_fails():
    original = np.array([[.6, .6000001], [.9, .1]], dtype=np.float32)
    native = original.copy()
    native[0] = .6
    classes, expected, rows = np.array([1, 3]), np.array([3, 1]), np.array([37180, 12])
    prediction, report = verify_scores("aux", native, classes, expected, rows, (original, classes), strict=False)
    assert prediction.tolist() == [1, 1]
    assert report["golden_predictions_equal"] is False
    assert report["parity_status"] == "auxiliary_near_tie_disagreement"
    assert report["disagreements"][0]["source_row_id"] == 37180
    assert report["disagreements"][0]["reference_margin"] > 0
    assert report["disagreements"][0]["native_margin"] == 0
    with pytest.raises(ValueError, match="native predictions differ"):
        verify_scores("latent", native, classes, expected, rows, (original, classes), strict=True)
    with pytest.raises(ValueError, match="native predictions differ"):
        verify_scores("aux", native, classes, expected, rows, strict=False)


def test_semantic_drift_cannot_hide_behind_ties_or_unchanged_labels():
    original = np.array([[.1, .8], [.9, .1]], dtype=np.float32)
    classes, expected, rows = np.array([1, 3]), np.array([3, 1]), np.array([1, 2])
    for native in (np.array([[.5, .5], [.9, .1]]), original + .1):
        with pytest.raises(AssertionError):
            verify_scores("aux", native, classes, expected, rows, (original, classes), strict=False)
    with pytest.raises(ValueError, match="nonfinite"):
        verify_scores("aux", original * np.nan, classes, expected, rows, (original, classes), strict=False)
    with pytest.raises(AssertionError):
        verify_scores("aux", original, classes, expected, rows, (original, classes[::-1]), strict=False)


def test_supplement_must_reproduce_original_labels():
    scores = np.array([[.9, .1]], dtype=np.float32)
    classes = np.array([1, 3])
    with pytest.raises(AssertionError):
        verify_scores("aux", scores, classes, np.array([3]), np.array([1]), (scores, classes), strict=False)


@pytest.mark.parametrize("classes,expected,rows", [
    ([1, 1], [1], [10]), ([1., 3.], [1], [10]), ([1, 3], [[1]], [10]),
    ([1, 3], [7], [10]), ([1, 3], [1], [10.]), ([1, 3], [1], [10, 20]),
])
def test_invalid_label_and_identity_schemas(classes, expected, rows):
    with pytest.raises(ValueError):
        verify_scores("aux", np.array([[.9, .1]]), np.asarray(classes),
                      np.asarray(expected), np.asarray(rows), strict=False)
