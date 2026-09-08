"""Discriminative feature autoencoder and compact, decoder-free deployment.

Training reconstructs standardized log-power features, not raw IQ. A supervised
cosine objective shapes the latent geometry; model selection always measures the
actual train-built prototype bank on validation data. The classifier used by the
loss and the decoder are discarded at export. Inference imports only NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from gcfcr.optimized.features import feature_operation_counts, spectral_features


def _positive_int(value: int, name: str, *, zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < (0 if zero else 1):
        raise ValueError(f"{name} must be {'nonnegative' if zero else 'positive'} integer")
    return int(value)


def _matrix(x: np.ndarray, name: str) -> np.ndarray:
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] == 0 or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite matrix with nonzero width")
    return a


def _labels(y: np.ndarray, rows: int) -> np.ndarray:
    labels = np.asarray(y)
    if labels.ndim != 1 or len(labels) != rows or not np.issubdtype(labels.dtype, np.integer):
        raise ValueError("labels must be an integer vector matching the number of rows")
    if (labels < 0).any() or (labels > np.iinfo(np.int64).max).any():
        raise ValueError("labels must fit a nonnegative int64")
    return labels.astype(np.int64, copy=False)


def normalize_rows(x: np.ndarray) -> np.ndarray:
    """Normalize float32 rows, retaining zero vectors as zero."""
    norm = np.sqrt(np.sum(x * x, axis=1, keepdims=True))
    if not np.isfinite(norm).all():
        raise ValueError("row norm overflowed; scale input values before normalization")
    return np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12)


def build_prototype_bank(
    codes: np.ndarray,
    labels: np.ndarray,
    prototypes_per_class: int = 1,
    *,
    iterations: int = 12,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Build deterministic per-class spherical k-means prototypes from train only.

    Farthest-first initialization and stable argmax make repeat runs reproducible.
    Scores are tiled to at most ``chunk_size * prototypes_per_class`` floats;
    no pairwise train-by-train distance matrix is formed. Empty clusters retain
    their previous representative. A zero-mean class uses its first code, avoiding
    inventing a nonzero direction when every reference has zero energy.
    """
    x = _matrix(codes, "codes")
    y = _labels(labels, len(x))
    if not len(x):
        raise ValueError("prototype bank requires at least one training reference")
    k = _positive_int(prototypes_per_class, "prototypes_per_class")
    iterations = _positive_int(iterations, "iterations")
    chunk_size = _positive_int(chunk_size, "chunk_size")
    x = normalize_rows(x)
    banks, bank_labels = [], []
    for label in np.unique(y):
        members = x[y == label]
        count = min(k, len(members))
        mean = normalize_rows(members.mean(axis=0, keepdims=True))[0]
        if count == 1:
            centers = mean[None, :]
            if not np.any(mean):
                centers = members[:1].copy()
        else:
            first = int(np.argmin(members @ mean))
            chosen = [first]
            best = members @ members[first]
            for _ in range(1, count):
                best[chosen] = np.inf
                index = int(np.argmin(best))
                chosen.append(index)
                best = np.maximum(best, members @ members[index])
            centers = members[chosen].copy()
            for _ in range(iterations):
                sums = np.zeros_like(centers)
                populations = np.zeros(count, dtype=np.int64)
                for start in range(0, len(members), chunk_size):
                    chunk = members[start:start + chunk_size]
                    assignment = np.argmax(chunk @ centers.T, axis=1)
                    np.add.at(sums, assignment, chunk)
                    populations += np.bincount(assignment, minlength=count)
                updated = normalize_rows(sums)
                valid = (populations > 0) & np.any(updated != 0, axis=1)
                updated[~valid] = centers[~valid]
                if np.allclose(updated, centers, rtol=0, atol=1e-6):
                    centers = updated
                    break
                centers = updated
        banks.append(centers)
        bank_labels.append(np.full(count, label, dtype=np.int64))
    return np.ascontiguousarray(np.concatenate(banks)), np.concatenate(bank_labels)


@dataclass
class LatentAutoencoder:
    """A folded affine encoder plus normalized reference codes.

    ``weights`` use output-by-input orientation. ReLU follows each layer except
    the last. Input feature standardization is already folded into the first
    affine transform. ``scores`` returns one maximum cosine score per sorted
    class label; ``predict`` returns those original integer labels.
    """

    weights: tuple[np.ndarray, ...]
    biases: tuple[np.ndarray, ...]
    prototypes: np.ndarray
    prototype_labels: np.ndarray

    def __post_init__(self) -> None:
        if not 1 <= len(self.weights) <= 2 or len(self.weights) != len(self.biases):
            raise ValueError("encoder must contain one or two affine layers")
        weights, biases = [], []
        previous = None
        for weight, bias in zip(self.weights, self.biases):
            w = _matrix(weight, "weights").copy(order="C")
            b = np.asarray(bias, dtype=np.float32)
            if w.shape[0] == 0 or b.shape != (w.shape[0],) or not np.isfinite(b).all():
                raise ValueError("invalid affine bias or empty output dimension")
            if previous is not None and w.shape[1] != previous:
                raise ValueError("adjacent encoder dimensions differ")
            weights.append(w)
            biases.append(b.copy())
            previous = w.shape[0]
        bank = _matrix(self.prototypes, "prototypes")
        labels = _labels(self.prototype_labels, len(bank))
        if len(bank) == 0 or bank.shape[1] != previous:
            raise ValueError("prototype bank must be nonempty and match latent dimension")
        self.weights = tuple(weights)
        self.biases = tuple(biases)
        self.prototypes = normalize_rows(bank).copy(order="C")
        self.prototype_labels = labels.copy()
        self.classes = np.unique(labels)
        self._class_rows = tuple(np.flatnonzero(labels == label) for label in self.classes)

    @property
    def input_dim(self) -> int:
        return self.weights[0].shape[1]

    @property
    def latent_dim(self) -> int:
        return self.weights[-1].shape[0]

    @property
    def storage_bytes(self) -> int:
        """Persistent numeric payload, excluding Python/BLAS object overhead."""
        return sum(a.nbytes for a in (*self.weights, *self.biases, self.prototypes, self.prototype_labels, self.classes, *self._class_rows))

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        x = _matrix(features, "features")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"features must have width {self.input_dim}")
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = x @ weight.T + bias
            if index + 1 < len(self.weights):
                np.maximum(x, 0, out=x)
        if not np.isfinite(x).all():
            raise ValueError("encoder overflowed; check feature scaling and model weights")
        return normalize_rows(x)

    def encode(self, iq: np.ndarray) -> np.ndarray:
        return self.encode_features(spectral_features(iq, self.input_dim))

    def match_codes(self, codes: np.ndarray) -> np.ndarray:
        """Score already unit-normalized codes, using a bounded prototype bank."""
        z = _matrix(codes, "codes")
        if z.shape[1] != self.latent_dim:
            raise ValueError("codes have wrong latent dimension")
        similarity = z @ self.prototypes.T
        return np.column_stack([similarity[:, rows].max(axis=1) for rows in self._class_rows])

    def scores(self, iq: np.ndarray) -> np.ndarray:
        return self.match_codes(self.encode(iq))

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        return self.classes[self.match_codes(self.encode_features(features)).argmax(axis=1)]

    def predict(self, iq: np.ndarray) -> np.ndarray:
        return self.classes[self.scores(iq).argmax(axis=1)]

    def operation_counts(self, n_samples: int = 512) -> dict[str, int | str]:
        """Per-query accounting; MAC=one multiply+add (two real arithmetic ops).

        Transcendentals, division, comparisons, and FFT estimates stay separate;
        divisions count as one arithmetic operation in real_flops_estimate;
        square roots and logarithms are separate special_function_evaluations.
        These are algorithmic estimates, not CPU instructions or peak FLOPS.
        """
        encoder_macs = sum(weight.size for weight in self.weights)
        match_macs = self.prototypes.size
        frontend = feature_operation_counts(n_samples, self.input_dim)
        fft_ops = frontend["fft_real_ops_estimate"]
        arithmetic = frontend["frontend_add_multiply_ops"] + frontend["frontend_divisions"] + 2 * (encoder_macs + match_macs) + 3 * self.latent_dim - 1
        return {
            **frontend,
            "real_flops_estimate": arithmetic + fft_ops if isinstance(fft_ops, int) else "unavailable",
            "special_function_evaluations": self.input_dim + 1,
            "encoder_macs": encoder_macs,
            "matching_macs": match_macs,
            "affine_and_matching_real_ops": 2 * (encoder_macs + match_macs),
            "relu_comparisons": sum(b.size for b in self.biases[:-1]),
            "latent_norm_add_multiply_ops": 2 * self.latent_dim - 1,
            "latent_norm_sqrt": 1,
            "latent_norm_divisions": self.latent_dim,
            "class_max_comparisons": len(self.prototypes) - len(self.classes),
            "class_argmax_comparisons": len(self.classes) - 1,
            "decoder_runtime_ops": 0,
        }

    operation_count = operation_counts

    def save(self, path: str | Path) -> None:
        """Write numeric arrays and a small Unicode JSON manifest, never pickle."""
        arrays = {
            "metadata": np.asarray(json.dumps({"format": "gcfcr-latent-ae", "version": 1, "layers": len(self.weights)})),
            "prototypes": self.prototypes,
            "prototype_labels": self.prototype_labels,
        }
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            arrays[f"weight_{index}"] = weight
            arrays[f"bias_{index}"] = bias
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "LatentAutoencoder":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            layers = metadata.get("layers")
            if metadata.get("format") != "gcfcr-latent-ae" or metadata.get("version") != 1 or layers not in (1, 2):
                raise ValueError("unsupported autoencoder artifact format")
            return cls(
                tuple(data[f"weight_{index}"] for index in range(layers)),
                tuple(data[f"bias_{index}"] for index in range(layers)),
                data["prototypes"], data["prototype_labels"],
            )


def fit(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    *,
    seed: int = 42,
    epochs: int = 40,
    hidden_dim: int = 48,
    latent_dim: int = 16,
    prototypes_per_class: int = 1,
    batch_size: int = 128,
    learning_rate: float = 0.002,
    reconstruction_weight: float = 0.1,
    augmentation_std: float = 0.05,
    temperature: float = 0.1,
    device: str = "cpu",
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[LatentAutoencoder, list[dict[str, Any]]]:
    """Train and select using validation prototype accuracy, then reconstruction.

    The validation set may guide epoch selection and must be independent of the
    final test set. The API deliberately takes no test data. Standardization and
    all reference prototypes use training features exclusively. Gaussian feature
    corruption trains denoising in feature space; it is not an AWGN IQ channel.
    """
    train = _matrix(train_features, "train_features")
    val = _matrix(val_features, "val_features")
    y = _labels(train_labels, len(train))
    vy = _labels(val_labels, len(val))
    if not len(train) or not len(val) or train.shape[1] != val.shape[1]:
        raise ValueError("nonempty train and validation matrices must have equal width")
    classes, targets = np.unique(y, return_inverse=True)
    if len(classes) < 2 or not np.isin(vy, classes).all():
        raise ValueError("training requires at least two classes and must cover validation labels")
    seed = _positive_int(seed, "seed", zero=True)
    epochs = _positive_int(epochs, "epochs")
    hidden_dim = _positive_int(hidden_dim, "hidden_dim", zero=True)
    latent_dim = _positive_int(latent_dim, "latent_dim")
    prototypes_per_class = _positive_int(prototypes_per_class, "prototypes_per_class")
    batch_size = _positive_int(batch_size, "batch_size")
    for value, name, allow_zero in ((learning_rate, "learning_rate", False), (reconstruction_weight, "reconstruction_weight", False), (augmentation_std, "augmentation_std", True), (temperature, "temperature", False)):
        if not np.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
            raise ValueError(f"{name} must be finite and {'nonnegative' if allow_zero else 'positive'}")

    # Import only for training: a deployed artifact needs neither Torch nor a decoder.
    import torch
    from torch import nn
    from torch.nn import functional as functional

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    mean = train.mean(axis=0)
    scale = train.std(axis=0).clip(min=0.05)
    normalized = ((train - mean) / scale).astype(np.float32)
    validation = ((val - mean) / scale).astype(np.float32)
    tensor_train = torch.from_numpy(normalized).to(device)
    tensor_val = torch.from_numpy(validation).to(device)
    tensor_y = torch.from_numpy(targets.astype(np.int64)).to(device)
    dimensions = [train.shape[1], *([hidden_dim] if hidden_dim else []), latent_dim]

    def network(sizes: list[int]) -> nn.Sequential:
        layers = []
        for index, (source, target) in enumerate(zip(sizes, sizes[1:])):
            layers.append(nn.Linear(source, target))
            if index + 2 < len(sizes):
                layers.append(nn.ReLU())
        return nn.Sequential(*layers).to(device)

    encoder = network(dimensions)
    decoder = network(list(reversed(dimensions)))
    classifier = nn.Parameter(torch.randn(len(classes), latent_dim, device=device))
    optimizer = torch.optim.AdamW([*encoder.parameters(), *decoder.parameters(), classifier], lr=learning_rate, weight_decay=1e-4)
    best_key, best_model, history = None, None, []

    def deployment() -> LatentAutoencoder:
        affine = [layer for layer in encoder if isinstance(layer, nn.Linear)]
        weights = [layer.weight.detach().cpu().numpy().copy() for layer in affine]
        biases = [layer.bias.detach().cpu().numpy().copy() for layer in affine]
        weights[0] /= scale[None, :]
        biases[0] -= weights[0] @ mean
        provisional = LatentAutoencoder(tuple(weights), tuple(biases), np.zeros((1, latent_dim), np.float32), np.zeros(1, np.int64))
        codes = provisional.encode_features(train)
        bank, labels = build_prototype_bank(codes, y, prototypes_per_class)
        return LatentAutoencoder(tuple(weights), tuple(biases), bank, labels)

    for epoch in range(1, epochs + 1):
        encoder.train()
        decoder.train()
        total_loss = 0.0
        for start in range(0, len(train), batch_size):
            if start == 0:
                order = rng.permutation(len(train))
            index = torch.from_numpy(order[start:start + batch_size]).to(device)
            clean = tensor_train[index]
            augmented = clean + augmentation_std * torch.randn_like(clean) if augmentation_std else clean
            latent = encoder(augmented)
            reconstruction = functional.mse_loss(decoder(latent), clean)
            logits = functional.normalize(latent, dim=1) @ functional.normalize(classifier, dim=1).T / temperature
            loss = functional.cross_entropy(logits, tensor_y[index]) + reconstruction_weight * reconstruction
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(index)
        encoder.eval()
        decoder.eval()
        with torch.inference_mode():
            val_reconstruction = float(functional.mse_loss(decoder(encoder(tensor_val)), tensor_val))
        candidate = deployment()
        correct = int(np.count_nonzero(candidate.predict_features(val) == vy))
        key = (correct, -val_reconstruction)
        selected = best_key is None or key > best_key
        if selected:
            best_key, best_model = key, candidate
        history.append({"epoch": epoch, "train_loss": total_loss / len(train), "validation_accuracy": correct / len(val), "validation_reconstruction_mse": val_reconstruction, "selected": selected})
        if progress is not None:
            progress(dict(history[-1]))
    assert best_model is not None
    return best_model, history
