import numpy as np
import pytest
from gcfcr.optimized.metrics import classification_metrics, paired_accuracy_interval
from gcfcr.optimized.classifier import FeatureClassifier, fit_classifier

def test_metrics_and_paired_group_interval():
    y = np.array([0, 0, 1, 1])
    result = classification_metrics(y, [0, 1, 1, 0], snr_db=[0, 0, 5, 5])
    assert result["accuracy"] == result["macro_f1"] == 0.5
    assert result["confusion"] == [[1, 1], [1, 1]]
    interval = paired_accuracy_interval(y, y, y, groups=[1, 1, 2, 2], repetitions=100)
    assert interval["ci95"] == [0, 0] and interval["independent_units"] == 2
    interval = paired_accuracy_interval(y, y, 1 - y, repetitions=100)
    assert interval["ci95"] == [1, 1]

def test_classifier_roundtrip_and_validation(tmp_path):
    classifier = FeatureClassifier((np.eye(2, dtype=np.float32),), (np.zeros(2),), np.array([3, 8]))
    path = tmp_path / "classifier.npz"
    classifier.save(path)
    loaded = FeatureClassifier.load(path)
    np.testing.assert_array_equal(loaded.scores_features([[1, 2]]), [[1, 2]])
    with pytest.raises(ValueError):
        loaded.scores_features([[np.nan, 0]])
    with pytest.raises(ValueError):
        FeatureClassifier((np.eye(2),), (np.zeros(3),), [0, 1])

def test_direct_classifier_train_only():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    rng = np.random.default_rng(123)
    train = rng.normal(size=(120, 4)).astype(np.float32)
    val = rng.normal(size=(40, 4)).astype(np.float32)
    y, vy = (train[:, 0] > 0).astype(int), (val[:, 0] > 0).astype(int)
    model, history = fit_classifier(train, y, val, vy, hidden_dim=8, epochs=12, learning_rate=0.02)
    assert np.mean(model.classes[model.scores_features(val).argmax(1)] == vy) >= 0.8
    assert len(history) == 12

def test_critical_review_regressions():
    with pytest.raises(ValueError):
        FeatureClassifier((np.eye(2),), (np.zeros(2),), [0.9, 1.9])
    with pytest.raises(ValueError):
        FeatureClassifier((np.eye(2),), (np.zeros(2),), np.array([0, 2**64-1],dtype=np.uint64))
    huge = FeatureClassifier((np.full((2, 2), 1e38, dtype=np.float32),), (np.zeros(2),), [0, 1])
    with np.errstate(over="ignore"), pytest.raises(ValueError):
        huge.scores_features([[1e38, 1e38]])
    small = FeatureClassifier((np.zeros((2, 3)),), (np.zeros(2),), [0, 1])
    assert small.operation_counts(96)["real_flops_estimate"] == "unavailable"
    with pytest.raises(ValueError):
        paired_accuracy_interval([0, 1], [0, 1], [1, 0], groups=[7, 7], repetitions=100)
