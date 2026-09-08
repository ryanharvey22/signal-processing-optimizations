"""Phase-invariant coherent-filter autoencoder for compact signal discrimination.

A learned complex kernel integrates IQ before magnitude-squaring. This preserves
coherent gain inside the first receptive field and removes global phase exactly
in real arithmetic. Small real convolutions then encode the response envelopes.
The optional training-only decoder reconstructs 128 spectral features, not IQ.
No test data is accepted by training; inference imports only NumPy.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from gcfcr.optimized.autoencoder import _labels, _matrix, _positive_int, build_prototype_bank, normalize_rows
from gcfcr.optimized.conv_autoencoder import _conv_numpy
from gcfcr.optimized.features import spectral_features

SAMPLES = 512


def coherent_features(iq: np.ndarray) -> np.ndarray:
    """RMS-normalized I/Q; zero frames map to zero, overflow is rejected."""
    x = np.asarray(iq)
    if x.ndim != 2 or x.shape[1] != SAMPLES or not np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("iq must be finite complex frames shaped (batch, 512)")
    x = x.astype(np.complex64, copy=False)
    energy = (x.real * x.real + x.imag * x.imag).sum(axis=1, keepdims=True)
    if not np.isfinite(energy).all():
        raise ValueError("IQ energy overflowed; scale input amplitudes")
    rms = np.sqrt(energy / SAMPLES)
    real = np.divide(x.real, rms, out=np.zeros(x.shape, np.float32), where=rms > 0)
    imag = np.divide(x.imag, rms, out=np.zeros(x.shape, np.float32), where=rms > 0)
    features = np.stack((real, imag), axis=1)
    if not np.isfinite(features).all():
        raise ValueError("IQ normalization overflowed")
    return features


def _features(x: np.ndarray) -> np.ndarray:
    value = np.asarray(x, np.float32)
    if value.ndim != 3 or value.shape[1:] != (2, SAMPLES) or not np.isfinite(value).all():
        raise ValueError("features must be finite normalized IQ shaped (batch, 2, 512)")
    return value


def _complex_block(real: np.ndarray, imag: np.ndarray) -> np.ndarray:
    # [A,-B; B,A] multiplies I+jQ by A+jB. Conjugating the learned kernel is
    # unnecessary: its real/imaginary coefficients are both freely trainable.
    return np.ascontiguousarray(np.concatenate((np.stack((real, -imag), axis=1), np.stack((imag, real), axis=1)), axis=0))


def _power_numpy(x: np.ndarray, block: np.ndarray, gain: np.ndarray, bias: np.ndarray) -> np.ndarray:
    kernel = block.shape[2]
    padding = kernel // 2
    padded = np.pad(x, ((0, 0), (0, 0), (padding, padding)), mode="wrap")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel, axis=2)[:, :, ::2, :]
    batch, _, length, _ = windows.shape
    patches = windows.transpose(0, 2, 1, 3).reshape(batch * length, 2 * kernel)
    response = patches @ block.reshape(len(block), -1).T
    if not np.isfinite(response).all():
        raise ValueError("coherent accumulation overflowed")
    response = response.reshape(batch, length, len(block)).transpose(0, 2, 1)
    channels = len(block) // 2
    power = response[:, :channels] ** 2 + response[:, channels:] ** 2
    if not np.isfinite(power).all():
        raise ValueError("coherent response power overflowed")
    scaled = power * gain[None, :, None] + bias[None, :, None]
    if not np.isfinite(scaled).all():
        raise ValueError("power normalization overflowed before activation")
    return np.maximum(scaled, 0)


@dataclass
class CoherentAutoencoder:
    complex_real: np.ndarray
    complex_imag: np.ndarray
    power_gain: np.ndarray
    power_bias: np.ndarray
    conv_weights: tuple[np.ndarray, ...]
    conv_biases: tuple[np.ndarray, ...]
    projection_weight: np.ndarray
    projection_bias: np.ndarray
    prototypes: np.ndarray
    prototype_labels: np.ndarray
    chunk_size: int = 32

    def __setattr__(self, name, value):
        # The expanded GEMM cache is derived once. Coefficients must therefore
        # remain immutable; construct a new model to change learned filters.
        if name in ("complex_real", "complex_imag") and "_block" in self.__dict__:
            raise AttributeError("complex filters are immutable; construct a replacement model")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        real, imag = _matrix(self.complex_real, "complex_real"), _matrix(self.complex_imag, "complex_imag")
        if real.shape != imag.shape or len(real) < 1 or real.shape[1] < 3 or real.shape[1] > 65 or real.shape[1] % 2 != 1:
            raise ValueError("complex weights require matching nonempty odd kernels of length 3..65")
        gain, bias = np.asarray(self.power_gain, np.float32), np.asarray(self.power_bias, np.float32)
        if gain.shape != (len(real),) or bias.shape != gain.shape or not np.isfinite(gain).all() or not np.isfinite(bias).all():
            raise ValueError("power gain and bias must match complex channels")
        if len(self.conv_weights) not in (3, 4, 5) or len(self.conv_biases) != len(self.conv_weights):
            raise ValueError("three through five real convolution stages are required")
        channels, weights, biases = len(real), [], []
        for weight, bias_vector in zip(self.conv_weights, self.conv_biases):
            w, b = np.asarray(weight, np.float32), np.asarray(bias_vector, np.float32)
            if w.ndim != 3 or w.shape[1:] != (channels, 5) or not len(w) or b.shape != (len(w),) or not np.isfinite(w).all() or not np.isfinite(b).all():
                raise ValueError("invalid real convolution dimensions or coefficients")
            weights.append(w.copy(order="C"))
            biases.append(b.copy())
            channels = len(w)
        projection = _matrix(self.projection_weight, "projection_weight")
        projection_bias = np.asarray(self.projection_bias, np.float32)
        if not len(projection) or projection.shape[1] != 2 * channels or projection_bias.shape != (len(projection),) or not np.isfinite(projection_bias).all():
            raise ValueError("projection must match pooled final channels")
        bank = _matrix(self.prototypes, "prototypes")
        labels = _labels(self.prototype_labels, len(bank))
        if not len(bank) or bank.shape[1] != len(projection):
            raise ValueError("nonempty reference bank must match latent dimension")
        self.complex_real, self.complex_imag = real.copy(order="C"), imag.copy(order="C")
        self.power_gain, self.power_bias = gain.copy(), bias.copy()
        self.conv_weights, self.conv_biases = tuple(weights), tuple(biases)
        self.projection_weight, self.projection_bias = projection.copy(order="C"), projection_bias.copy()
        self.prototypes, self.prototype_labels = normalize_rows(bank), labels.copy()
        self.classes = np.unique(labels)
        self._class_rows = tuple(np.flatnonzero(labels == label) for label in self.classes)
        self.complex_real.setflags(write=False)
        self.complex_imag.setflags(write=False)
        self._block = _complex_block(self.complex_real, self.complex_imag)
        self.chunk_size = _positive_int(self.chunk_size, "chunk_size")

    @property
    def latent_dim(self) -> int:
        return len(self.projection_weight)

    @property
    def storage_bytes(self) -> int:
        # Includes the expanded NumPy GEMM kernel cache. Portable C can use only
        # original real/imaginary arrays, so its flash report is smaller.
        return sum(x.nbytes for x in (self.complex_real, self.complex_imag, self.power_gain, self.power_bias, *self.conv_weights, *self.conv_biases, self.projection_weight, self.projection_bias, self.prototypes, self.prototype_labels, self.classes, self._block, *self._class_rows))

    def extract_features(self, iq: np.ndarray) -> np.ndarray:
        return coherent_features(iq)

    def encode_features(self, features: np.ndarray) -> np.ndarray:
        chunk_size = _positive_int(self.chunk_size, "chunk_size")
        x = _features(features)
        output = np.empty((len(x), self.latent_dim), np.float32)
        for start in range(0, len(x), chunk_size):
            hidden = _power_numpy(x[start:start + chunk_size], self._block, self.power_gain, self.power_bias)
            for weight, bias in zip(self.conv_weights, self.conv_biases):
                hidden = _conv_numpy(hidden, weight, bias)
            pooled = np.concatenate((hidden.mean(axis=2), hidden.max(axis=2)), axis=1)
            code = pooled @ self.projection_weight.T + self.projection_bias
            if not np.isfinite(code).all():
                raise ValueError("coherent encoder projection overflowed")
            output[start:start + len(code)] = normalize_rows(code)
        return output

    def encode(self, iq: np.ndarray) -> np.ndarray:
        chunk_size = _positive_int(self.chunk_size, "chunk_size")
        x = np.asarray(iq)
        if x.ndim != 2 or x.shape[1] != SAMPLES or not np.iscomplexobj(x):
            raise ValueError("iq must be complex frames shaped (batch, 512)")
        output = np.empty((len(x), self.latent_dim), np.float32)
        for start in range(0, len(x), chunk_size):
            features = self.extract_features(x[start:start + chunk_size])
            output[start:start + len(features)] = self.encode_features(features)
        return output

    def match_codes(self, codes: np.ndarray) -> np.ndarray:
        z = _matrix(codes, "codes")
        if z.shape[1] != self.latent_dim:
            raise ValueError("latent dimension differs from bank")
        scores = z @ self.prototypes.T
        return np.column_stack([scores[:, rows].max(axis=1) for rows in self._class_rows])

    def scores(self, iq: np.ndarray) -> np.ndarray:
        return self.match_codes(self.encode(iq))

    def predict(self, iq: np.ndarray) -> np.ndarray:
        return self.classes[self.scores(iq).argmax(axis=1)]

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        return self.classes[self.match_codes(self.encode_features(features)).argmax(axis=1)]

    def operation_counts(self, n_samples: int = SAMPLES) -> dict[str, int | str]:
        if n_samples != SAMPLES:
            raise ValueError("coherent model requires 512-sample frames")
        length, complex_channels, kernel = SAMPLES // 2, len(self.complex_real), self.complex_real.shape[1]
        complex_macs = length * complex_channels * kernel
        power_elements = length * complex_channels
        real_macs = self.projection_weight.size
        stage_sizes = [power_elements]
        for weight in self.conv_weights:
            length //= 2
            real_macs += length * weight.size
            stage_sizes.append(length * len(weight))
        matching = self.prototypes.size
        channels = len(self.conv_weights[-1])
        frontend = 6 * SAMPLES + 1  # power reduction, RMS scale and two channel divisions
        arithmetic = frontend + 8 * complex_macs + 5 * power_elements + 2 * (real_macs + matching) + channels * length + 3 * self.latent_dim - 1
        return {"real_flops_estimate": arithmetic, "special_function_evaluations": 2,
                "coherent_complex_macs": complex_macs, "tail_projection_real_macs": real_macs,
                "matching_macs": matching, "first_kernel_samples": kernel,
                "receptive_field_samples": kernel + 4 * sum(2 ** index for index in range(1, len(self.conv_weights) + 1)),
                "fft_runtime_ops": 0, "decoder_runtime_ops": 0, "numpy_chunk_frames": self.chunk_size,
                "direct_c_activation_payload_bytes_estimate": 4 * (max(stage_sizes[::2]) + max(stage_sizes[1::2])),
                "scratch_note": "direct activation estimate excludes retained IQ features, pooling, latent, input/output and stack; NumPy GEMM allocates patches"}

    operation_count = operation_counts

    def save(self, path: str | Path) -> None:
        arrays = {"metadata": np.asarray(json.dumps({"format": "gcfcr-coherent-latent-ae", "version": 1, "tail_layers": len(self.conv_weights), "chunk_size": self.chunk_size})),
                  "complex_real": self.complex_real, "complex_imag": self.complex_imag, "power_gain": self.power_gain, "power_bias": self.power_bias,
                  "projection_weight": self.projection_weight, "projection_bias": self.projection_bias, "prototypes": self.prototypes, "prototype_labels": self.prototype_labels}
        for index, (weight, bias) in enumerate(zip(self.conv_weights, self.conv_biases)):
            arrays[f"conv_weight_{index}"], arrays[f"conv_bias_{index}"] = weight, bias
        with Path(path).open("wb") as stream:
            np.savez_compressed(stream, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> "CoherentAutoencoder":
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            layers = metadata.get("tail_layers")
            if metadata.get("format") != "gcfcr-coherent-latent-ae" or metadata.get("version") != 1 or layers not in (3, 4, 5):
                raise ValueError("unsupported coherent autoencoder artifact")
            return cls(data["complex_real"], data["complex_imag"], data["power_gain"], data["power_bias"], tuple(data[f"conv_weight_{i}"] for i in range(layers)), tuple(data[f"conv_bias_{i}"] for i in range(layers)), data["projection_weight"], data["projection_bias"], data["prototypes"], data["prototype_labels"], metadata.get("chunk_size", 32))


def fit(train_iq: np.ndarray, train_labels: np.ndarray, val_iq: np.ndarray, val_labels: np.ndarray, *,
        seed: int = 42, epochs: int = 60, complex_channels: int = 8, kernel_size: int = 33,
        channels: Sequence[int] = (12, 16, 16, 16, 16), latent_dim: int = 16,
        prototypes_per_class: int = 1, batch_size: int = 128, learning_rate: float = 0.002,
        reconstruction_weight: float = 0.01, augmentation_std: float = 0.01,
        shift_augmentation: bool = True, temperature: float = 0.1, device: str = "cpu",
        progress: Callable[[dict[str, Any]], None] | None = None) -> tuple[CoherentAutoencoder, list[dict[str, Any]]]:
    """Train only on train, retaining the first best deployed validation epoch."""
    train, val = coherent_features(train_iq), coherent_features(val_iq)
    y, vy = _labels(train_labels, len(train)), _labels(val_labels, len(val))
    if not len(train) or not len(val):
        raise ValueError("training and validation cannot be empty")
    classes, targets = np.unique(y, return_inverse=True)
    if len(classes) < 2 or not np.isin(vy, classes).all():
        raise ValueError("training requires two classes and must cover validation labels")
    seed, epochs = _positive_int(seed, "seed", zero=True), _positive_int(epochs, "epochs")
    complex_channels, kernel_size = _positive_int(complex_channels, "complex_channels"), _positive_int(kernel_size, "kernel_size")
    if kernel_size < 3 or kernel_size > 65 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be odd within 3..65")
    channels = tuple(_positive_int(c, "channels") for c in channels)
    if len(channels) not in (3, 4, 5):
        raise ValueError("three to five real convolution stages are required")
    latent_dim, batch_size = _positive_int(latent_dim, "latent_dim"), _positive_int(batch_size, "batch_size")
    prototypes_per_class = _positive_int(prototypes_per_class, "prototypes_per_class")
    for value, name, zero in ((learning_rate, "learning_rate", False), (reconstruction_weight, "reconstruction_weight", True), (augmentation_std, "augmentation_std", True), (temperature, "temperature", False)):
        if not np.isfinite(value) or value < 0 or (value == 0 and not zero):
            raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")

    import torch
    from torch import nn
    from torch.nn import functional as functional
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    real = nn.Parameter(torch.randn(complex_channels, kernel_size, device=device) / np.sqrt(2 * kernel_size))
    imag = nn.Parameter(torch.randn(complex_channels, kernel_size, device=device) / np.sqrt(2 * kernel_size))
    power_bn = nn.BatchNorm1d(complex_channels).to(device)
    layers, previous = [], complex_channels
    for width in channels:
        layers += [nn.Conv1d(previous, width, 5, stride=2, padding=2, padding_mode="circular"), nn.BatchNorm1d(width), nn.ReLU()]
        previous = width
    convolution = nn.Sequential(*layers).to(device)
    projection = nn.Linear(2 * channels[-1], latent_dim).to(device)
    decoder = nn.Sequential(nn.Linear(latent_dim, 64), nn.ReLU(), nn.Linear(64, 128)).to(device)
    classifier = nn.Parameter(torch.randn(len(classes), latent_dim, device=device))
    optimizer = torch.optim.AdamW([real, imag, *power_bn.parameters(), *convolution.parameters(), *projection.parameters(), *decoder.parameters(), classifier], lr=learning_rate, weight_decay=1e-4)
    tensor_train, tensor_val = torch.from_numpy(train).to(device), torch.from_numpy(val).to(device)
    tensor_y = torch.from_numpy(targets.astype(np.int64)).to(device)
    target_train = torch.from_numpy(spectral_features(train_iq, 128)).to(device)
    target_val = torch.from_numpy(spectral_features(val_iq, 128)).to(device)

    def encode(x):
        block = torch.cat((torch.stack((real, -imag), dim=1), torch.stack((imag, real), dim=1)), dim=0)
        response = functional.conv1d(functional.pad(x, (kernel_size // 2, kernel_size // 2), mode="circular"), block, stride=2)
        power = response[:, :complex_channels].square() + response[:, complex_channels:].square()
        hidden = convolution(functional.relu(power_bn(power)))
        return projection(torch.cat((hidden.mean(dim=2), hidden.amax(dim=2)), dim=1))

    def deployment():
        gain = power_bn.weight.detach() / torch.sqrt(power_bn.running_var.detach() + power_bn.eps)
        bias = power_bn.bias.detach() - gain * power_bn.running_mean.detach()
        weights, biases = [], []
        for index in range(0, len(convolution), 3):
            conv, bn = convolution[index], convolution[index + 1]
            scale = bn.weight.detach() / torch.sqrt(bn.running_var.detach() + bn.eps)
            weights.append((conv.weight.detach() * scale[:, None, None]).cpu().numpy())
            biases.append(((conv.bias.detach() - bn.running_mean.detach()) * scale + bn.bias.detach()).cpu().numpy())
        codes = np.empty((len(train), latent_dim), np.float32)
        with torch.inference_mode():
            for start in range(0, len(train), batch_size):
                code = encode(tensor_train[start:start + batch_size]).cpu().numpy()
                codes[start:start + len(code)] = normalize_rows(code)
        bank, labels = build_prototype_bank(codes, y, prototypes_per_class)
        return CoherentAutoencoder(real.detach().cpu().numpy(), imag.detach().cpu().numpy(), gain.cpu().numpy(), bias.cpu().numpy(), tuple(weights), tuple(biases), projection.weight.detach().cpu().numpy(), projection.bias.detach().cpu().numpy(), bank, labels)

    history, best_accuracy, best_model = [], -1, None
    for epoch in range(1, epochs + 1):
        power_bn.train(); convolution.train(); projection.train(); decoder.train()
        order, total_loss = rng.permutation(len(train)), 0.0
        for start in range(0, len(train), batch_size):
            index = torch.from_numpy(order[start:start + batch_size]).to(device)
            clean = tensor_train[index]
            if shift_augmentation:
                clean = torch.roll(clean, int(rng.integers(SAMPLES)), dims=2)
            augmented = clean + augmentation_std * torch.randn_like(clean) if augmentation_std else clean
            code = encode(augmented)
            logits = functional.normalize(code, dim=1) @ functional.normalize(classifier, dim=1).T / temperature
            reconstruction = functional.mse_loss(decoder(code), target_train[index])
            loss = functional.cross_entropy(logits, tensor_y[index]) + reconstruction_weight * reconstruction
            optimizer.zero_grad(set_to_none=True)
            loss.backward(); optimizer.step()
            total_loss += float(loss.detach()) * len(index)
        power_bn.eval(); convolution.eval(); projection.eval(); decoder.eval()
        reconstruction_sum = 0.0
        with torch.inference_mode():
            for start in range(0, len(val), batch_size):
                chunk = tensor_val[start:start + batch_size]
                reconstruction_sum += float(functional.mse_loss(decoder(encode(chunk)), target_val[start:start + batch_size])) * len(chunk)
        candidate = deployment()
        correct = int(np.count_nonzero(candidate.predict_features(val) == vy))
        selected = correct > best_accuracy
        if selected:
            best_accuracy, best_model = correct, candidate
        row = {"epoch": epoch, "train_loss": total_loss / len(train), "validation_accuracy": correct / len(val), "validation_reconstruction_mse": reconstruction_sum / len(val), "selected": selected}
        history.append(row)
        if progress:
            progress(dict(row))
    assert best_model is not None
    return best_model, history
