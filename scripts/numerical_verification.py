"""Strict primary parity and explicit post-freeze auxiliary numerical diagnostics."""
import hashlib
import json
from pathlib import Path
import numpy as np

RTOL, ATOL = 3e-4, 3e-5


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classes_for(model):
    return model.classes if hasattr(model, "classes") else np.arange(model.num_classes)


def scores_chunked(model, iq, chunk=128):
    return np.concatenate([model.scores(iq[i:i + chunk]) for i in range(0, len(iq), chunk)])


def load_replay(directory, manifest, row_ids):
    """Optional supplement never changes the immutable original evaluation."""
    directory = Path(directory)
    meta_path = directory / "auxiliary_replay.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta["format"] != "post-freeze-auxiliary-score-replay-v1":
        raise ValueError("unsupported replay format")
    for filename, key in (("auxiliary_replay.npz", "scores_sha256"),
                          ("frozen_goldens.npz", "frozen_goldens_sha256"),
                          ("spec.json", "frozen_spec_sha256")):
        if digest(directory / filename) != meta[key]:
            raise ValueError(f"post-freeze replay checksum mismatch: {filename}")
    frozen_spec = json.loads((directory / "spec.json").read_text(encoding="utf-8"))
    source_sha = manifest.get("source_sha256", manifest.get("dataset", {}).get("source_sha256"))
    if source_sha != frozen_spec["dataset"]["source_sha256"]:
        raise ValueError("replay source dataset identity changed")
    if "dataset" in manifest and manifest["dataset"] != frozen_spec["dataset"]:
        raise ValueError("replay split manifest changed")
    expected_names = {name for name, row in manifest["models"].items()
                      if name != "latent" and row["kind"] != "waveform"}
    if set(meta["model_sha256"]) != expected_names:
        raise ValueError("replay auxiliary model coverage changed")
    if meta["rtol"] != RTOL or meta["atol"] != ATOL:
        raise ValueError("replay numerical tolerances changed")
    result = {}
    with np.load(directory / "auxiliary_replay.npz", allow_pickle=False) as arrays:
        validate_integer_vector(arrays["row_ids"], "replay row_ids", unique=True)
        validate_integer_vector(np.asarray(row_ids), "query row_ids", unique=True)
        np.testing.assert_array_equal(arrays["row_ids"], row_ids)
        for name in expected_names:
            if manifest["models"][name]["sha256"] != meta["model_sha256"][name]:
                raise ValueError(f"{name}: replay model checksum mismatch")
            scores, classes = arrays[name + "_scores"], arrays[name + "_classes"]
            validate_integer_vector(classes, "replay classes", unique=True)
            if scores.shape != (len(row_ids), len(classes)) or not np.isfinite(scores).all():
                raise ValueError("invalid replay score array")
            result[name] = (scores, classes)
    return result


def validate_integer_vector(value, name, unique=False):
    if value.ndim != 1 or value.dtype.kind not in "iu" or not len(value):
        raise ValueError(f"{name} must be a nonempty integer vector")
    if unique and len(np.unique(value)) != len(value):
        raise ValueError(f"{name} must contain unique values")


def verify_scores(name, scores, classes, expected, row_ids, reference=None, strict=True):
    """Report actual labels; permit auxiliary flips only with full-score evidence."""
    scores, classes, expected = np.asarray(scores), np.asarray(classes), np.asarray(expected)
    row_ids = np.asarray(row_ids)
    validate_integer_vector(classes, "classes", unique=True)
    validate_integer_vector(expected, "expected")
    validate_integer_vector(row_ids, "row_ids", unique=True)
    if len(row_ids) != len(expected) or not np.isin(expected, classes).all():
        raise ValueError("query identity or expected class membership mismatch")
    if scores.shape != (len(expected), len(classes)) or not np.isfinite(scores).all():
        raise ValueError(f"{name}: invalid score shape or nonfinite scores")
    prediction = classes[scores.argmax(axis=1)]
    maximum_residual = None
    reference_scores = None
    if reference is not None:
        reference_scores, reference_classes = reference
        validate_integer_vector(np.asarray(reference_classes), "reference classes", unique=True)
        np.testing.assert_array_equal(classes, reference_classes)
        if reference_scores.shape != scores.shape or not np.isfinite(reference_scores).all():
            raise ValueError(f"{name}: invalid full-score replay")
        np.testing.assert_array_equal(classes[reference_scores.argmax(axis=1)], expected)
        np.testing.assert_allclose(scores, reference_scores, rtol=RTOL, atol=ATOL)
        maximum_residual = float(np.max(np.abs(scores - reference_scores)))
    mismatch = np.flatnonzero(prediction != expected)
    if len(mismatch) and (strict or reference_scores is None):
        raise ValueError(f"{name}: native predictions differ from frozen golden results")
    disagreements = []
    for i in mismatch:
        old_column = int(np.flatnonzero(classes == expected[i])[0])
        new_column = int(np.flatnonzero(classes == prediction[i])[0])
        old_scores = reference_scores[i]
        bound = 2 * ATOL + RTOL * (abs(float(old_scores[old_column])) + abs(float(old_scores[new_column])))
        reference_margin = float(old_scores[old_column] - old_scores[new_column])
        native_margin = float(scores[i, new_column] - scores[i, old_column])
        if not 0 <= reference_margin <= bound or not 0 <= native_margin <= bound:
            raise ValueError(f"{name}: unexplained decision disagreement at query {i}")
        disagreements.append({"query_index": int(i), "source_row_id": int(row_ids[i]),
            "frozen_label": int(expected[i]), "native_label": int(prediction[i]),
            "reference_margin": reference_margin, "native_margin": native_margin,
            "allowed_two_score_perturbation": bound})
    return prediction, {"golden_predictions_equal": not len(mismatch),
        "parity_status": "exact_predictions" if not len(mismatch) else "auxiliary_near_tie_disagreement",
        "post_freeze_full_score_replay_checked": reference is not None,
        "full_score_max_absolute_residual": maximum_residual,
        "disagreements": disagreements}
