"""Direct compact classifier control: same features and training budget as the AE."""
from __future__ import annotations
from dataclasses import dataclass
import json
from pathlib import Path
import numpy as np
from .features import spectral_features, feature_operation_counts

@dataclass
class FeatureClassifier:
    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    classes: np.ndarray

    def __post_init__(self):
        self.weights = tuple(np.array(w, dtype=np.float32, order="C", copy=True) for w in self.weights)
        self.biases = tuple(np.array(b, dtype=np.float32, copy=True) for b in self.biases)
        labels = np.asarray(self.classes)
        if (labels.ndim != 1 or labels.dtype.kind not in "iu" or
            np.any(labels < 0) or np.any(labels > np.iinfo(np.int64).max)):
            raise ValueError("classes must be nonnegative int64-compatible integers")
        self.classes = labels.astype(np.int64, copy=True)
        if len(self.weights) not in (1, 2) or len(self.biases) != len(self.weights):
            raise ValueError("one or two affine layers required")
        previous = self.weights[0].shape[1] if self.weights[0].ndim == 2 else 0
        for w, b in zip(self.weights, self.biases):
            if w.ndim != 2 or not w.shape[0] or w.shape[1] != previous or b.shape != (w.shape[0],):
                raise ValueError("invalid affine dimensions")
            if not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("weights must be finite")
            previous = len(b)
        if self.classes.ndim != 1 or len(self.classes) != previous or len(np.unique(self.classes)) != previous:
            raise ValueError("one unique label per output required")

    def scores_features(self, features):
        x = np.asarray(features, dtype=np.float32)
        if x.ndim != 2 or x.shape[1] != self.weights[0].shape[1] or not np.isfinite(x).all():
            raise ValueError("invalid feature matrix")
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w.T + b
            if i + 1 < len(self.weights):
                np.maximum(x, 0, out=x)
        if not np.isfinite(x).all():
            raise ValueError("classifier overflowed; check features and weights")
        return x

    def scores(self, iq):
        return self.scores_features(spectral_features(iq, self.weights[0].shape[1]))

    def predict(self, iq):
        return self.classes[self.scores(iq).argmax(axis=1)]

    @property
    def storage_bytes(self):
        return sum(x.nbytes for x in (*self.weights, *self.biases, self.classes))

    def operation_counts(self, n_samples=512):
        result = feature_operation_counts(n_samples, self.weights[0].shape[1])
        macs = sum(w.size for w in self.weights)
        result.update(encoder_macs=macs, affine_real_ops=2 * macs,
                      class_argmax_comparisons=len(self.classes) - 1,
                      relu_comparisons=sum(len(x) for x in self.biases[:-1]),
                      real_flops_estimate=(result["fft_real_ops_estimate"] + result["frontend_add_multiply_ops"] +
                      result["frontend_divisions"] + 2 * macs)
                      if isinstance(result["fft_real_ops_estimate"], int) else "unavailable",
                      special_function_evaluations=result["frontend_log1p"])
        return result

    def save(self, path):
        arrays = {"metadata": np.asarray(json.dumps({"format": "feature-classifier-v1", "layers": len(self.weights)})),
                  "classes": self.classes}
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            arrays[f"weight_{i}"], arrays[f"bias_{i}"] = w, b
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as d:
            meta = json.loads(str(d["metadata"].item()))
            if meta.get("format") != "feature-classifier-v1" or meta.get("layers") not in (1, 2):
                raise ValueError("unsupported classifier artifact")
            return cls(tuple(d[f"weight_{i}"] for i in range(meta["layers"])),
                       tuple(d[f"bias_{i}"] for i in range(meta["layers"])), d["classes"])

def fit_classifier(train, y, val, vy, *, hidden_dim=48, epochs=40, seed=42,
                   learning_rate=0.002, batch_size=256, progress=None):
    """Select by held-out validation accuracy, never final test labels."""
    import torch
    from torch import nn
    train, val = np.asarray(train, np.float32), np.asarray(val, np.float32)
    y, vy = np.asarray(y), np.asarray(vy)
    if (train.ndim != 2 or val.ndim != 2 or not len(train) or not len(val) or
        train.shape[1] != val.shape[1] or not np.isfinite(train).all() or not np.isfinite(val).all() or
        y.shape != (len(train),) or vy.shape != (len(val),) or y.dtype.kind not in "iu" or
        vy.dtype.kind not in "iu" or not np.isin(vy, y).all()):
        raise ValueError("invalid training/validation data")
    if hidden_dim < 0 or epochs < 1 or batch_size < 1 or learning_rate <= 0 or not np.isfinite(learning_rate):
        raise ValueError("invalid training settings")
    classes, encoded = np.unique(y, return_inverse=True)
    if len(classes) < 2:
        raise ValueError("at least two training classes required")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean, scale = train.mean(0), train.std(0).clip(min=0.05)
    x = torch.from_numpy((train - mean) / scale)
    targets = torch.from_numpy(encoded)
    dims = [train.shape[1], *([hidden_dim] if hidden_dim else []), len(classes)]
    layers = []
    for i, (a, b) in enumerate(zip(dims, dims[1:])):
        layers.append(nn.Linear(a, b))
        if i + 2 < len(dims):
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    best, best_accuracy, history = None, -1, []
    for epoch in range(1, epochs + 1):
        order = rng.permutation(len(x))
        loss_sum = 0.0
        for start in range(0, len(x), batch_size):
            ids = torch.from_numpy(order[start:start + batch_size])
            loss = nn.functional.cross_entropy(model(x[ids]), targets[ids])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(ids)
        affines = [layer for layer in model if isinstance(layer, nn.Linear)]
        weights = [layer.weight.detach().numpy().copy() for layer in affines]
        biases = [layer.bias.detach().numpy().copy() for layer in affines]
        weights[0] /= scale[None, :]
        biases[0] -= weights[0] @ mean
        candidate = FeatureClassifier(tuple(weights), tuple(biases), classes)
        accuracy = float(np.mean(classes[candidate.scores_features(val).argmax(1)] == vy))
        if accuracy > best_accuracy:
            best, best_accuracy = candidate, accuracy
        row = {"epoch": epoch, "validation_accuracy": accuracy, "train_loss": loss_sum / len(x)}
        history.append(row)
        if progress:
            progress(row)
    return best, history
