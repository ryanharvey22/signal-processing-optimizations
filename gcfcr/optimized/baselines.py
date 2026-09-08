"""Train-only, bounded-memory controls for learned latent waveform matching.

The raw bank uses actual training exemplars rather than phase-cancelled complex
class averages. The aligned and circular-delay controls can share bank_indices.
Costs count the complete prediction path, not merely the final inner products.
"""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from gcfcr.optimized.features import feature_operation_counts, spectral_features
from gcfcr.optimized.autoencoder import build_prototype_bank


def _iq_array(iq: np.ndarray) -> np.ndarray:
    iq = np.asarray(iq)
    if iq.ndim != 2 or iq.shape[1] < 1 or not np.iscomplexobj(iq):
        raise ValueError("iq must have shape (batch, samples) and complex dtype")
    with np.errstate(over="ignore", invalid="ignore"):
        iq = np.ascontiguousarray(iq, dtype=np.complex64)
    if not np.isfinite(iq).all():
        raise ValueError("iq samples exceed finite complex64 range")
    # Squared correlation is returned as float32. Reject out-of-range input
    # explicitly rather than silently reducing an infinite norm to a zero bank.
    limit = np.sqrt(np.finfo(np.float32).max / (8 * iq.shape[1]))
    if np.any(np.abs(iq.real) > limit) or np.any(np.abs(iq.imag) > limit):
        raise ValueError("IQ amplitude exceeds the safe float32 correlation range; rescale the input")
    return iq


def _labels(y: np.ndarray, n: int) -> np.ndarray:
    y = np.asarray(y)
    if y.shape != (n,) or y.dtype.kind not in "iu" or n < 1:
        raise ValueError("nonempty integer labels must match training rows")
    unique = np.unique(y)
    if not np.array_equal(unique, np.arange(len(unique))):
        raise ValueError("training labels must cover contiguous classes starting at zero")
    return y.astype(np.int64)


def select_bank_indices(y: np.ndarray, templates_per_class: int = 16,
                        *, seed: int = 42, quality: np.ndarray | None = None) -> np.ndarray:
    """Class-balanced exemplars; optional quality ranks TRAIN metadata only.

    Higher quality wins and seeded shuffling breaks ties. No query metadata is
    consulted. Reuse returned IDs in aligned and circular-delay controls.
    """
    y = _labels(y, len(y))
    if templates_per_class < 1 or int(templates_per_class) != templates_per_class:
        raise ValueError("templates_per_class must be a positive integer")
    if quality is not None:
        quality = np.asarray(quality)
        if quality.shape != y.shape or quality.dtype.kind not in "fiu" or not np.isfinite(quality).all():
            raise ValueError("quality must provide one finite numeric training value per row")
    rng = np.random.default_rng(seed)
    selected = []
    for label in np.unique(y):
        candidates = rng.permutation(np.flatnonzero(y == label))
        if quality is not None:
            candidates = candidates[np.argsort(-quality[candidates].astype(np.float64), kind="stable")]
        selected.append(candidates[:templates_per_class])
    return np.sort(np.concatenate(selected))


def _unit_rows(x: np.ndarray) -> np.ndarray:
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _metadata(data) -> dict:
    return json.loads(str(data["metadata"].item()))


class WaveformMatchedFilter:
    """Noncoherent aligned or all-circular-delay matching against training examples.

    Unknown phase is removed by squared magnitude; no noisy complex averaging is
    used. Circular delay deliberately models periodic frame translation, not a
    zero-padded linear-delay or Doppler search. Query energy normalization is
    unnecessary for argmax because it scales every class equally.
    """
    def __init__(self, *, mode: str = "aligned", templates_per_class: int = 16,
                 bank_indices: np.ndarray | None = None, seed: int = 42,
                 query_chunk: int = 8, template_chunk: int = 16) -> None:
        if mode not in ("aligned", "fft"):
            raise ValueError("mode must be aligned or fft")
        if min(query_chunk, template_chunk, templates_per_class) < 1:
            raise ValueError("chunk sizes and template count must be positive")
        if any(int(v) != v for v in (query_chunk, template_chunk, templates_per_class)):
            raise ValueError("chunk sizes and template count must be integers")
        self.mode = mode
        self.templates_per_class = int(templates_per_class)
        self.bank_indices = None if bank_indices is None else np.asarray(bank_indices).copy()
        self.seed = seed
        self.query_chunk = int(query_chunk)
        self.template_chunk = int(template_chunk)

    def fit(self, iq: np.ndarray, y: np.ndarray) -> "WaveformMatchedFilter":
        iq = _iq_array(iq)
        y = _labels(y, len(iq))
        indices = self.bank_indices
        if indices is None:
            indices = select_bank_indices(y, self.templates_per_class, seed=self.seed)
        if (indices.ndim != 1 or indices.dtype.kind not in "iu" or len(indices) < 1
                or np.any(indices < 0) or np.any(indices >= len(y))
                or len(np.unique(indices)) != len(indices)):
            raise ValueError("bank_indices must be distinct in-range integer training row IDs")
        if not np.array_equal(np.unique(y[indices]), np.unique(y)):
            raise ValueError("the raw bank must cover every training class")
        self.bank_indices = indices.astype(np.int64)
        self.bank = np.ascontiguousarray(_unit_rows(iq[indices]), dtype=np.complex64)
        self.bank_labels = y[indices]
        self.num_classes = int(y.max()) + 1
        self.n_samples = iq.shape[1]
        self._prepare()
        return self

    def _prepare(self) -> None:
        self._bank_conj = np.ascontiguousarray(self.bank.conj().T)
        self._fft_conj = (np.fft.fft(self.bank, axis=-1).conj()
                          if self.mode == "fft" else None)

    def scores(self, iq: np.ndarray) -> np.ndarray:
        if not hasattr(self, "bank"):
            raise RuntimeError("fit or load the baseline before predicting")
        iq = _iq_array(iq)
        if iq.shape[1] != self.n_samples:
            raise ValueError("query sample count does not match the fitted bank")
        out = np.zeros((len(iq), self.num_classes), dtype=np.float32)
        if self.mode == "aligned":
            # BLAS consumes BxN and NxK; there is no broadcast BxKxN temporary.
            for start in range(0, len(iq), self.query_chunk):
                stop = min(start + self.query_chunk, len(iq))
                for bank_start in range(0, len(self.bank), self.template_chunk):
                    bank_stop = min(bank_start + self.template_chunk, len(self.bank))
                    inner = iq[start:stop] @ self._bank_conj[:, bank_start:bank_stop]
                    power = inner.real * inner.real + inner.imag * inner.imag
                    self._merge(out[start:stop], power, self.bank_labels[bank_start:bank_stop])
        else:
            for start in range(0, len(iq), self.query_chunk):
                stop = min(start + self.query_chunk, len(iq))
                query_fft = np.fft.fft(iq[start:stop], axis=-1)
                for bank_start in range(0, len(self.bank), self.template_chunk):
                    bank_stop = min(bank_start + self.template_chunk, len(self.bank))
                    corr = np.fft.ifft(query_fft[:, None, :] *
                        self._fft_conj[None, bank_start:bank_stop, :], axis=-1)
                    power = (corr.real * corr.real + corr.imag * corr.imag).max(axis=-1)
                    self._merge(out[start:stop], power, self.bank_labels[bank_start:bank_stop])
        return out

    @staticmethod
    def _merge(out: np.ndarray, power: np.ndarray, labels: np.ndarray) -> None:
        for label in np.unique(labels):
            out[:, label] = np.maximum(out[:, label], power[:, labels == label].max(axis=1))

    def predict(self, iq: np.ndarray) -> np.ndarray:
        return self.scores(iq).argmax(axis=1)

    @property
    def storage_bytes(self) -> int:
        return (self.bank.nbytes + self.bank_labels.nbytes + self.bank_indices.nbytes
                + self._bank_conj.nbytes + (0 if self._fft_conj is None else self._fft_conj.nbytes))

    def operation_counts(self, n_samples: int = 512) -> dict:
        n, k = self.n_samples, len(self.bank)
        if n_samples != n:
            raise ValueError("operation counts require the fitted sample count")
        if self.mode == "aligned":
            flops = k * (8 * n - 2 + 3)
            comparisons = k + self.num_classes - 1
            fft_flops = 0
        else:
            if n & (n - 1):
                raise ValueError("FFT operation estimates require a power-of-two frame length")
            fft_flops = (k + 1) * 5 * n * np.log2(n)
            flops = fft_flops + k * n * 9
            comparisons = k * (n - 1) + k + self.num_classes - 1
        return {"real_flops_estimate": int(flops), "fft_flops_estimate": int(fft_flops),
                "comparisons": int(comparisons), "special_function_evaluations": 0,
                "convention": "multiply and add each 1 FLOP; complex dot 8N-2; complex FFT estimated 5Nlog2N; magnitude squared 3"}

    def save(self, path: str | Path) -> None:
        metadata = dict(format_version=1, kind="waveform", mode=self.mode,
            templates_per_class=self.templates_per_class, seed=self.seed,
            query_chunk=self.query_chunk, template_chunk=self.template_chunk,
            num_classes=self.num_classes, n_samples=self.n_samples)
        np.savez_compressed(path, metadata=json.dumps(metadata), bank=self.bank,
                            labels=self.bank_labels, indices=self.bank_indices)

    @classmethod
    def load(cls, path: str | Path) -> "WaveformMatchedFilter":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata(data)
            if metadata.get("format_version") != 1 or metadata.get("kind") != "waveform":
                raise ValueError("unsupported waveform artifact")
            obj = cls(**{key: metadata[key] for key in ("mode", "templates_per_class", "seed",
                                                        "query_chunk", "template_chunk")})
            obj.bank = _iq_array(data["bank"])
            obj.bank_labels = _labels(data["labels"], len(obj.bank))
            obj.bank_indices = np.asarray(data["indices"], dtype=np.int64).copy()
            if obj.bank_indices.shape != (len(obj.bank),) or np.any(obj.bank_indices < 0):
                raise ValueError("invalid artifact bank indices")
            obj.num_classes = int(obj.bank_labels.max()) + 1
            obj.n_samples = obj.bank.shape[1]
            if obj.n_samples != metadata["n_samples"] or obj.num_classes != metadata["num_classes"]:
                raise ValueError("artifact metadata does not match bank dimensions")
            obj._prepare()
        return obj


class FeaturePrototypeMatcher:
    """Same spectral front end as the AE, with optional train-fitted PCA.

    Spherical prototypes are built by the same helper as the learned method.
    The PCA control tests whether learned nonlinear encoding improves on a
    conventional linear bottleneck. Neither control accesses validation/test.
    """
    def __init__(self, *, prototypes_per_class: int = 1, pca_dim: int | None = None,
                 bins: int = 128, query_chunk: int = 1024, standardize: bool = False) -> None:
        if bins < 1 or prototypes_per_class < 1 or query_chunk < 1:
            raise ValueError("dimensions, prototype count and chunk size must be positive")
        if any(int(v) != v for v in (bins, prototypes_per_class, query_chunk)):
            raise ValueError("dimensions, prototype count and chunk size must be integers")
        if pca_dim is not None and (pca_dim < 1 or int(pca_dim) != pca_dim):
            raise ValueError("PCA dimension must be a positive integer")
        self.prototypes_per_class = int(prototypes_per_class)
        self.pca_dim = None if pca_dim is None else int(pca_dim)
        self.bins = int(bins)
        self.query_chunk = int(query_chunk)
        self.standardize = bool(standardize)

    def fit(self, iq: np.ndarray, y: np.ndarray) -> "FeaturePrototypeMatcher":
        iq = np.asarray(iq)
        features = spectral_features(iq, bins=self.bins)
        y = _labels(y, len(iq))
        self.n_samples = iq.shape[1]
        self.num_classes = int(y.max()) + 1
        self.mean = np.zeros(self.bins, dtype=np.float32)
        self.scale = np.ones(self.bins, dtype=np.float32)
        self.components = np.empty((0, self.bins), dtype=np.float32)
        if self.standardize or self.pca_dim is not None:
            self.mean = features.mean(axis=0)
            features = features - self.mean
        if self.standardize:
            self.scale = np.maximum(features.std(axis=0), 1e-6)
            features = features / self.scale
        if self.pca_dim is not None:
            if self.pca_dim > min(features.shape):
                raise ValueError("PCA dimension exceeds training rank bound")
            _, _, vh = np.linalg.svd(features, full_matrices=False)
            self.components = np.ascontiguousarray(vh[:self.pca_dim], dtype=np.float32)
            features = features @ self.components.T
        self.bank, self.bank_labels = build_prototype_bank(features, y,
            prototypes_per_class=self.prototypes_per_class)
        self.bank = np.ascontiguousarray(self.bank, dtype=np.float32)
        self.bank_labels = np.asarray(self.bank_labels, dtype=np.int64)
        return self

    def scores(self, iq: np.ndarray) -> np.ndarray:
        if not hasattr(self, "bank"):
            raise RuntimeError("fit or load the baseline before predicting")
        iq = np.asarray(iq)
        if iq.ndim != 2 or not np.iscomplexobj(iq) or iq.shape[1] != self.n_samples:
            raise ValueError("query sample count does not match the fitted features")
        out = np.full((len(iq), self.num_classes), -np.inf, dtype=np.float32)
        for start in range(0, len(iq), self.query_chunk):
            stop = min(start + self.query_chunk, len(iq))
            features = spectral_features(iq[start:stop], bins=self.bins)
            if self.standardize or self.pca_dim is not None:
                features = features - self.mean
            if self.standardize:
                features = features / self.scale
            if self.pca_dim is not None:
                features = features @ self.components.T
            features = _unit_rows(features)
            scores = features @ self.bank.T
            for label in np.unique(self.bank_labels):
                out[start:stop, label] = scores[:, self.bank_labels == label].max(axis=1)
        return out

    def predict(self, iq: np.ndarray) -> np.ndarray:
        return self.scores(iq).argmax(axis=1)

    @property
    def storage_bytes(self) -> int:
        return self.bank.nbytes + self.bank_labels.nbytes + self.mean.nbytes + self.scale.nbytes + self.components.nbytes

    def operation_counts(self, n_samples: int = 512) -> dict:
        n, d, k = self.n_samples, self.bank.shape[1], len(self.bank)
        if n_samples != n:
            raise ValueError("operation counts require the fitted sample count")
        # FFT + |X|^2 + pooled reductions + energy reduction/normalization/log1p;
        # logarithms and square roots are reported separately from arithmetic.
        front = feature_operation_counts(n, self.bins)
        fft_flops = front["fft_real_ops_estimate"]
        if fft_flops == "unavailable":
            raise ValueError("FFT operation estimates require a power-of-two frame length")
        front_end = fft_flops + front["frontend_add_multiply_ops"] + front["frontend_divisions"]
        projection = 0 if self.pca_dim is None else d * (2 * self.bins - 1)
        if self.standardize or self.pca_dim is not None:
            projection += self.bins
        if self.standardize:
            projection += self.bins
        normalization = 3 * d - 1
        matching = k * (2 * d - 1)
        return {"real_flops_estimate": int(front_end + projection + normalization + matching),
                "fft_flops_estimate": int(fft_flops), "comparisons": int(k + self.num_classes - 1),
                "special_function_evaluations": self.bins + 1,
                "convention": "multiply/add/divide each 1 FLOP; complex FFT estimated 5Nlog2N; log1p and sqrt counted separately"}

    def save(self, path: str | Path) -> None:
        metadata = dict(format_version=1, kind="feature", prototypes_per_class=self.prototypes_per_class,
                        pca_dim=self.pca_dim, bins=self.bins, query_chunk=self.query_chunk, standardize=self.standardize,
                        n_samples=self.n_samples, num_classes=self.num_classes)
        np.savez_compressed(path, metadata=json.dumps(metadata), bank=self.bank,
                            labels=self.bank_labels, mean=self.mean, scale=self.scale, components=self.components)

    @classmethod
    def load(cls, path: str | Path) -> "FeaturePrototypeMatcher":
        with np.load(path, allow_pickle=False) as data:
            metadata = _metadata(data)
            if metadata.get("format_version") != 1 or metadata.get("kind") != "feature":
                raise ValueError("unsupported feature artifact")
            obj = cls(**{key: metadata[key] for key in ("prototypes_per_class", "pca_dim", "bins", "query_chunk")},
                      standardize=metadata.get("standardize", False))
            obj.bank = np.asarray(data["bank"], dtype=np.float32).copy()
            obj.bank_labels = _labels(data["labels"], len(obj.bank))
            obj.mean = np.asarray(data["mean"], dtype=np.float32).copy()
            obj.scale = (np.asarray(data["scale"], dtype=np.float32).copy() if "scale" in data
                         else np.ones(obj.bins, dtype=np.float32))
            obj.components = np.asarray(data["components"], dtype=np.float32).copy()
            dim = obj.bins if obj.pca_dim is None else obj.pca_dim
            if (obj.bank.ndim != 2 or obj.bank.shape[1] != dim or obj.mean.shape != (obj.bins,)
                    or obj.components.shape != (0 if obj.pca_dim is None else dim, obj.bins)
                    or obj.scale.shape != (obj.bins,) or np.any(obj.scale <= 0)
                    or not all(np.isfinite(a).all() for a in (obj.bank, obj.mean, obj.scale, obj.components))):
                raise ValueError("invalid feature artifact dimensions or non-finite weights")
            obj.num_classes = int(obj.bank_labels.max()) + 1
            obj.n_samples = int(metadata["n_samples"])
            if obj.n_samples < obj.bins or obj.n_samples % obj.bins or obj.num_classes != metadata["num_classes"]:
                raise ValueError("invalid feature artifact metadata")
        return obj
