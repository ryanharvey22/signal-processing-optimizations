"""Compact temporal feature autoencoder: local IQ statistics, no FFT or logarithm.

Both frontends normalize amplitude and are equivariant to circular shifts.
The temporal-product frontend is analytically invariant to global phase; the
coherent-IQ frontend learns phase robustness through training augmentation.
Neither provides analytic carrier-frequency-offset invariance. The default training-only
decoder reconstructs shift-invariant spectral features, not original complex IQ.
Global pooling discards absolute time origin; no waveform reconstruction is claimed.
NumPy deployment imports no training framework and omits the decoder/classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from gcfcr.optimized.autoencoder import _labels, _matrix, _positive_int, build_prototype_bank, normalize_rows


SAMPLES = 512


def temporal_features(iq: np.ndarray) -> np.ndarray:
    """Return ``(batch, 3, 512)`` normalized power and adjacent complex products.

    Channels are N*|x[n]|^2/E, N*Re(x[n]*conj(x[n-1]))/E, and its imaginary
    component. The first product wraps to x[N-1]. Every channel is divided by
    frame energy before multiplying by N, avoiding a large reciprocal. Exact
    zero frames map to zero; nonfinite input or arithmetic overflow is rejected.
    """
    x = np.asarray(iq)
    if x.ndim != 2 or x.shape[1] != SAMPLES or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("iq must be finite complex frames shaped (batch, 512)")
    x = x.astype(np.complex64, copy=False)
    real, imag = x.real, x.imag
    previous_real, previous_imag = np.roll(real, 1, axis=1), np.roll(imag, 1, axis=1)
    power = real * real + imag * imag
    energy = power.sum(axis=1, keepdims=True)
    if not np.isfinite(energy).all():
        raise ValueError("IQ energy overflowed; scale input amplitudes")
    features = np.stack((power, real * previous_real + imag * previous_imag,
                         imag * previous_real - real * previous_imag), axis=1)
    features = np.divide(features, energy[:, None, :], out=np.zeros_like(features), where=energy[:, None, :] > 0)
    features *= SAMPLES
    if not np.isfinite(features).all():
        raise ValueError("temporal feature calculation overflowed")
    return np.ascontiguousarray(features)


def iq_features(iq: np.ndarray) -> np.ndarray:
    """RMS-normalized I, Q and power; preserves coherent phase relationships."""
    x = np.asarray(iq)
    if x.ndim != 2 or x.shape[1] != SAMPLES or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("iq must be finite complex frames shaped (batch, 512)")
    x = x.astype(np.complex64, copy=False)
    power = x.real * x.real + x.imag * x.imag
    energy = power.sum(axis=1, keepdims=True)
    if not np.isfinite(energy).all():
        raise ValueError("IQ energy overflowed; scale input amplitudes")
    rms = np.sqrt(energy / SAMPLES)
    real = np.divide(x.real, rms, out=np.zeros_like(power), where=rms > 0)
    imag = np.divide(x.imag, rms, out=np.zeros_like(power), where=rms > 0)
    normalized_power = np.divide(power, energy, out=np.zeros_like(power), where=energy > 0) * SAMPLES
    result = np.stack((real, imag, normalized_power), axis=1)
    if not np.isfinite(result).all():
        raise ValueError("IQ normalization overflowed")
    return np.ascontiguousarray(result)


def _features(array: np.ndarray) -> np.ndarray:
    x = np.asarray(array, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (3, SAMPLES) or not np.isfinite(x).all():
        raise ValueError("features must be finite and shaped (batch, 3, 512)")
    return x


def _conv_numpy(x: np.ndarray, weight: np.ndarray, bias: np.ndarray) -> np.ndarray:
    """Circular pad=2, kernel=5, stride=2; same order as Torch Conv1d."""
    padded = np.pad(x, ((0, 0), (0, 0), (2, 2)), mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, 5, axis=2)[:, :, ::2, :]
    batch, channels, length, kernel = windows.shape
    patches = windows.transpose(0, 2, 1, 3).reshape(batch * length, channels * kernel)
    result = patches @ weight.reshape(len(weight), -1).T + bias
    if not np.isfinite(result).all():
        raise ValueError("convolution overflowed before activation")
    result = result.reshape(batch, length, len(weight)).transpose(0, 2, 1)
    return np.maximum(result, 0)


@dataclass
class ConvLatentAutoencoder:
    """Three to seven folded convolutions, pooled linear code, and reference bank.
    Circular shift invariance is exact for multiples of 2**layers, not every
    sample shift, because each convolution subsamples by two.
    """
    conv_weights: tuple[np.ndarray, ...]
    conv_biases: tuple[np.ndarray, ...]
    projection_weight: np.ndarray
    projection_bias: np.ndarray
    prototypes: np.ndarray
    prototype_labels: np.ndarray
    chunk_size: int = 32
    frontend: str = "temporal"

    def __post_init__(self) -> None:
        if self.frontend not in ("temporal", "iq"):
            raise ValueError("frontend must be temporal or iq")
        if len(self.conv_weights) not in range(3, 8) or len(self.conv_biases) != len(self.conv_weights):
            raise ValueError("three to seven convolution layers are required")
        weights, biases, channels = [], [], 3
        for weight, bias in zip(self.conv_weights, self.conv_biases):
            w, b = np.asarray(weight, np.float32), np.asarray(bias, np.float32)
            if w.ndim != 3 or w.shape[1:] != (channels, 5) or w.shape[0] < 1 or b.shape != (w.shape[0],) or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("invalid convolution weights, biases or channel dimensions")
            weights.append(w.copy(order="C"))
            biases.append(b.copy())
            channels = w.shape[0]
        projection = _matrix(self.projection_weight, "projection_weight")
        projection_bias = np.asarray(self.projection_bias, np.float32)
        if not len(projection) or projection.shape[1] != 2 * channels or projection_bias.shape != (len(projection),) or not np.isfinite(projection_bias).all():
            raise ValueError("projection dimensions must match mean/max pooling")
        bank = _matrix(self.prototypes, "prototypes")
        labels = _labels(self.prototype_labels, len(bank))
        if not len(bank) or bank.shape[1] != len(projection):
            raise ValueError("nonempty prototype bank must match latent dimension")
        self.conv_weights, self.conv_biases = tuple(weights), tuple(biases)
        self.projection_weight, self.projection_bias = projection.copy(order="C"), projection_bias.copy()
        self.prototypes, self.prototype_labels = normalize_rows(bank), labels.copy()
        self.classes = np.unique(labels)
        self._class_rows = tuple(np.flatnonzero(labels == label) for label in self.classes)
        self.chunk_size = _positive_int(self.chunk_size, "chunk_size")

    @property
    def latent_dim(self) -> int:
        return len(self.projection_weight)

    @property
    def storage_bytes(self) -> int:
        return sum(array.nbytes for array in (*self.conv_weights, *self.conv_biases, self.projection_weight, self.projection_bias, self.prototypes, self.prototype_labels, self.classes, *self._class_rows))

    def extract_features(self, iq: np.ndarray) -> np.ndarray:
        """Apply the saved model frontend; useful for deployment parity checks."""
        if self.frontend not in ("temporal", "iq"):
            raise ValueError("frontend must be temporal or iq")
        return (temporal_features if self.frontend == "temporal" else iq_features)(iq)

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        _positive_int(self.chunk_size, "chunk_size")
        x = _features(features)
        output = np.empty((len(x), self.latent_dim), np.float32)
        for start in range(0, len(x), self.chunk_size):
            hidden = x[start:start + self.chunk_size]
            for weight, bias in zip(self.conv_weights, self.conv_biases):
                hidden = _conv_numpy(hidden, weight, bias)
            pooled = np.concatenate((hidden.mean(axis=2), hidden.max(axis=2)), axis=1)
            code = pooled @ self.projection_weight.T + self.projection_bias
            if not np.isfinite(code).all():
                raise ValueError("convolution encoder overflowed")
            output[start:start + len(code)] = normalize_rows(code)
        return output

    def encode(self, iq: np.ndarray) -> np.ndarray:
        _positive_int(self.chunk_size, "chunk_size")
        # Chunk before the frontend too: scratch stays bounded for a large caller batch.
        x = np.asarray(iq)
        if x.ndim != 2 or x.shape[1] != SAMPLES or not np.iscomplexobj(x):
            raise ValueError("iq must be complex frames shaped (batch, 512)")
        output = np.empty((len(x), self.latent_dim), np.float32)
        for start in range(0, len(x), self.chunk_size):
            chunk = self.extract_features(x[start:start + self.chunk_size])
            output[start:start + len(chunk)] = self.encode_features(chunk)
        return output

    def match_codes(self, codes: np.ndarray) -> np.ndarray:
        z = _matrix(codes, "codes")
        if z.shape[1] != self.latent_dim:
            raise ValueError("code dimension differs from model")
        similarity = z @ self.prototypes.T
        return np.column_stack([similarity[:, rows].max(axis=1) for rows in self._class_rows])

    def scores(self, iq: np.ndarray) -> np.ndarray:
        return self.match_codes(self.encode(iq))

    def predict(self, iq: np.ndarray) -> np.ndarray:
        return self.classes[self.scores(iq).argmax(axis=1)]

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        return self.classes[self.match_codes(self.encode_features(features)).argmax(axis=1)]

    def operation_counts(self, n_samples: int = SAMPLES) -> dict[str, int | str]:
        if n_samples != SAMPLES:
            raise ValueError("this model uses 512-sample frames")
        length, macs, relu = SAMPLES, 0, 0
        stage_sizes = [3 * length]
        patch_sizes = []
        for weight in self.conv_weights:
            length //= 2
            macs += length * weight.size
            relu += length * len(weight)
            stage_sizes.append(length * len(weight))
            patch_sizes.append(length * weight.shape[1] * 5)
        channels = len(self.conv_weights[-1])
        macs += self.projection_weight.size
        matching = self.prototypes.size
        frontend = (16 * SAMPLES - 1) if self.frontend == "temporal" else (8 * SAMPLES)  # ordinary mul/add/div
        pooling = channels * length  # mean reduction and division
        normalization = 3 * self.latent_dim - 1
        return {"real_flops_estimate": frontend + 2 * (macs + matching) + pooling + normalization,
                "special_function_evaluations": 1 + int(self.frontend == "iq"), "frontend_real_ops": frontend,
                "encoder_macs": macs, "matching_macs": matching, "relu_comparisons": relu,
                "pool_max_comparisons": channels * (length - 1),
                "class_max_comparisons": len(self.prototypes) - len(self.classes),
                "class_argmax_comparisons": len(self.classes) - 1,
                "decoder_runtime_ops": 0, "fft_runtime_ops": 0,
                "numpy_chunk_frames": self.chunk_size,
                "largest_numpy_patch_payload_bytes": self.chunk_size * max(patch_sizes) * 4,
                "direct_c_activation_payload_bytes_estimate": 4 * max(a + b for a, b in zip(stage_sizes, stage_sizes[1:])),
                "scratch_note": "NumPy also allocates padding, outputs and BLAS workspace; C estimate excludes input/output and stack"}

    operation_count = operation_counts

    def save(self, path: str | Path) -> None:
        arrays = {"metadata": np.asarray(json.dumps({"format": "gcfcr-conv-latent-ae", "version": 1, "chunk_size": self.chunk_size, "layers": len(self.conv_weights), "frontend": self.frontend})),
                  "projection_weight": self.projection_weight, "projection_bias": self.projection_bias,
                  "prototypes": self.prototypes, "prototype_labels": self.prototype_labels}
        for index, (weight, bias) in enumerate(zip(self.conv_weights, self.conv_biases)):
            arrays[f"conv_weight_{index}"], arrays[f"conv_bias_{index}"] = weight, bias
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "ConvLatentAutoencoder":
        with np.load(path, allow_pickle=False) as arrays:
            metadata = json.loads(str(arrays["metadata"].item()))
            if metadata.get("format") != "gcfcr-conv-latent-ae" or metadata.get("version") != 1:
                raise ValueError("unsupported convolutional autoencoder artifact")
            layers = metadata.get("layers", 3)
            if layers not in range(3, 8):
                raise ValueError("invalid layer count")
            return cls(tuple(arrays[f"conv_weight_{index}"] for index in range(layers)),
                       tuple(arrays[f"conv_bias_{index}"] for index in range(layers)),
                       arrays["projection_weight"], arrays["projection_bias"], arrays["prototypes"], arrays["prototype_labels"],
                       chunk_size=metadata.get("chunk_size", 32), frontend=metadata.get("frontend", "temporal"))


def fit(train_iq: np.ndarray, train_labels: np.ndarray, val_iq: np.ndarray, val_labels: np.ndarray, *,
        seed: int = 42, epochs: int = 40, channels: Sequence[int] = (12, 16, 16, 16), latent_dim: int = 16,
        prototypes_per_class: int = 1, batch_size: int = 128, learning_rate: float = 0.002,
        reconstruction_weight: float = 0.01, reconstruction_target: str = "spectral", augmentation_std: float = 0.01,
        shift_augmentation: bool = True, frontend: str = "temporal", temperature: float = 0.1, device: str = "cpu",
        progress: Callable[[dict[str, Any]], None] | None = None) -> tuple[ConvLatentAutoencoder, list[dict[str, Any]]]:
    """Train on IQ; select actual folded NumPy reference-bank validation accuracy.

    No test argument exists. Convolution BatchNorm running statistics use training
    batches only and fold into output weights/biases, including boundary taps.
    Training-reference codes are computed by the equivalent frozen Torch encoder;
    deployed NumPy validation is evaluated with those very same stored prototypes.
    """
    if frontend not in ("temporal", "iq"):
        raise ValueError("frontend must be temporal or iq")
    frontend_fn = temporal_features if frontend == "temporal" else iq_features
    train, val = frontend_fn(train_iq), frontend_fn(val_iq)
    y, vy = _labels(train_labels, len(train)), _labels(val_labels, len(val))
    if not len(train) or not len(val):
        raise ValueError("train and validation sets must be nonempty")
    classes, targets = np.unique(y, return_inverse=True)
    if len(classes) < 2 or not np.isin(vy, classes).all():
        raise ValueError("training needs at least two classes and must cover validation labels")
    channels = tuple(_positive_int(c, "channels") for c in channels)
    if len(channels) not in range(3, 8):
        raise ValueError("three to seven convolution channel counts are required")
    seed, epochs = _positive_int(seed, "seed", zero=True), _positive_int(epochs, "epochs")
    latent_dim, batch_size = _positive_int(latent_dim, "latent_dim"), _positive_int(batch_size, "batch_size")
    prototypes_per_class = _positive_int(prototypes_per_class, "prototypes_per_class")
    for value, name, zero in ((learning_rate, "learning_rate", False), (reconstruction_weight, "reconstruction_weight", True), (augmentation_std, "augmentation_std", True), (temperature, "temperature", False)):
        if not np.isfinite(value) or value < 0 or (value == 0 and not zero):
            raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")

    if reconstruction_target not in ("spectral", "temporal"):
        raise ValueError("reconstruction_target must be spectral or temporal")
    if reconstruction_target == "spectral":
        from gcfcr.optimized.features import spectral_features
        reconstruction_train = spectral_features(train_iq, 128)
        reconstruction_val = spectral_features(val_iq, 128)
    else:
        reconstruction_train = train.reshape(len(train), -1)
        reconstruction_val = val.reshape(len(val), -1)
    reconstruction_dim = reconstruction_train.shape[1]

    import torch
    from torch import nn
    from torch.nn import functional as functional

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    layers, previous = [], 3
    for width in channels:
        layers.extend((nn.Conv1d(previous, width, 5, stride=2, padding=2, padding_mode="circular"), nn.BatchNorm1d(width), nn.ReLU()))
        previous = width
    convolution = nn.Sequential(*layers).to(device)
    projection = nn.Linear(2 * channels[-1], latent_dim).to(device)
    decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, reconstruction_dim)).to(device)
    classifier = nn.Parameter(torch.randn(len(classes), latent_dim, device=device))
    parameters = [*convolution.parameters(), *projection.parameters(), *decoder.parameters(), classifier]
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=1e-4)
    tensor_train, tensor_val = torch.from_numpy(train).to(device), torch.from_numpy(val).to(device)
    tensor_y = torch.from_numpy(targets.astype(np.int64)).to(device)
    target_train = torch.from_numpy(reconstruction_train).to(device)
    target_val = torch.from_numpy(reconstruction_val).to(device)

    def encode(x):
        hidden = convolution(x)
        return projection(torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=1))

    def deployment() -> ConvLatentAutoencoder:
        weights, biases = [], []
        for index in range(0, len(convolution), 3):
            conv, bn = convolution[index], convolution[index + 1]
            # BatchNorm acts after convolution, so its affine fold applies equally
            # at every circular boundary. No input-mean padding correction exists.
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            weights.append((conv.weight.detach() * scale[:, None, None]).cpu().numpy())
            biases.append(((conv.bias.detach() - bn.running_mean.detach()) * scale + bn.bias.detach()).cpu().numpy())
        codes = np.empty((len(train), latent_dim), np.float32)
        with torch.inference_mode():
            for start in range(0, len(train), batch_size):
                code = normalize_rows(encode(tensor_train[start:start + batch_size]).cpu().numpy())
                codes[start:start + len(code)] = code
        bank, labels = build_prototype_bank(codes, y, prototypes_per_class)
        return ConvLatentAutoencoder(tuple(weights), tuple(biases), projection.weight.detach().cpu().numpy(), projection.bias.detach().cpu().numpy(), bank, labels, frontend=frontend)

    history, best_key, best_model = [], None, None
    for epoch in range(1, epochs + 1):
        convolution.train()
        projection.train()
        decoder.train()
        total_loss = 0.0
        order = rng.permutation(len(train))
        for start in range(0, len(train), batch_size):
            indices = torch.from_numpy(order[start:start + batch_size]).to(device)
            clean = tensor_train[indices]
            if shift_augmentation:
                clean = torch.roll(clean, int(rng.integers(SAMPLES)), dims=2)
            if frontend == "iq":
                # A per-frame random global phase leaves the spectral target unchanged.
                phase = torch.as_tensor(rng.uniform(-np.pi, np.pi, len(clean)), dtype=clean.dtype, device=device)
                cosine, sine = phase.cos()[:, None], phase.sin()[:, None]
                real, imag = clean[:, 0], clean[:, 1]
                clean = torch.stack((real * cosine - imag * sine, real * sine + imag * cosine, clean[:, 2]), dim=1)
            augmented = clean + augmentation_std * torch.randn_like(clean) if augmentation_std else clean
            code = encode(augmented)
            reconstruction_target_batch = (clean.flatten(1) if reconstruction_target == "temporal" else target_train[indices])
            reconstruction = functional.mse_loss(decoder(code), reconstruction_target_batch)
            logits = functional.normalize(code, dim=1) @ functional.normalize(classifier, dim=1).T / temperature
            loss = functional.cross_entropy(logits, tensor_y[indices]) + reconstruction_weight * reconstruction
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(indices)
        convolution.eval()
        projection.eval()
        decoder.eval()
        reconstruction_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(val), batch_size):
                chunk = tensor_val[start:start + batch_size]
                reconstruction_sum += float(functional.mse_loss(decoder(encode(chunk)), target_val[start:start + len(chunk)])) * len(chunk)
        reconstruction_mse = reconstruction_sum / len(val)
        candidate = deployment()
        correct = int(np.count_nonzero(candidate.predict_features(val) == vy))
        key = (correct, 0)  # Same first-best-epoch rule for reconstruction and ablation.
        selected = best_key is None or key > best_key
        if selected:
            best_key, best_model = key, candidate
        row = {"epoch": epoch, "train_loss": total_loss / len(train), "validation_accuracy": correct / len(val), "validation_reconstruction_mse": reconstruction_mse, "selected": selected, "reconstruction_target": reconstruction_target, "reconstruction_weight": reconstruction_weight}
        history.append(row)
        if progress:
            progress(dict(row))
    assert best_model is not None
    return best_model, history
