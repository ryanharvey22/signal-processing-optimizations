"""Reproducible three-way radar data splits and a clearly labelled CI fixture.

No data downloads occur here. RadChar variants overlap; choose ONE release file.
Caps are applied after assigning the full population to splits, never before.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

SPLIT_VERSION = "full-population-80-10-10-v1"


@dataclass(frozen=True)
class SignalSplit:
    iq: np.ndarray
    y: np.ndarray
    snr_db: np.ndarray
    row_ids: np.ndarray
    group_ids: np.ndarray | None = None

    def __len__(self) -> int:
        return len(self.y)


@dataclass(frozen=True)
class ExperimentData:
    train: SignalSplit
    val: SignalSplit
    test: SignalSplit
    manifest: dict[str, Any]


def split_indices(n: int, *, seed: int = 42, train_fraction: float = 0.8,
                  val_fraction: float = 0.1,
                  groups: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Assign rows (or all rows of each known group) before any sample cap.

    Grouped fractions describe counts of groups, not necessarily counts of rows.
    An absent group identifier guarantees row separation only; it does not prove
    independence of unrecorded waveform realizations or collection sessions.
    """
    if n < 0 or not (0 < train_fraction < 1) or not (0 < val_fraction < 1):
        raise ValueError("positive fractions and nonnegative population required")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be below 1")
    if groups is not None:
        groups = np.asarray(groups)
        if groups.shape != (n,):
            raise ValueError("groups must contain one identifier per row")
        _, inverse = np.unique(groups, return_inverse=True)
        units = int(inverse.max()) + 1 if n else 0
    else:
        inverse = None
        units = n
    order = np.random.default_rng(seed).permutation(units)
    end_train = int(units * train_fraction)
    end_val = int(units * (train_fraction + val_fraction))
    result = {}
    for name, assignment in zip(("train", "val", "test"),
                                 (order[:end_train], order[end_train:end_val], order[end_val:])):
        result[name] = (np.flatnonzero(np.isin(inverse, assignment)) if inverse is not None
                        else np.sort(assignment)).astype(np.int64, copy=False)
    return result


def cap_indices(indices: np.ndarray, cap: int | None, *, seed: int = 42) -> np.ndarray:
    """Deterministic nested subsamples, chosen across the split rather than a prefix."""
    if cap is None:
        return indices.copy()
    if isinstance(cap, bool) or int(cap) != cap or cap < 1:
        raise ValueError("sample caps must be positive integers or None")
    if cap >= len(indices):
        return indices.copy()
    chosen = np.random.default_rng(seed).permutation(len(indices))[:int(cap)]
    return np.sort(indices[chosen])


def _array_hash(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(str(array.shape).encode())
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_radchar_metadata(handle) -> tuple[np.ndarray, np.ndarray | None]:
    """Validate release schema without loading the large waveform dataset."""
    if "iq" not in handle or "labels" not in handle:
        raise ValueError("RadChar requires iq and labels datasets")
    iq = handle["iq"]
    if iq.ndim != 2 or iq.shape[1] < 1 or iq.dtype.kind != "c":
        raise ValueError("iq must have shape (N, samples) and a complex dtype")
    labels = np.asarray(handle["labels"][:])
    if labels.ndim != 1 or len(labels) != len(iq):
        raise ValueError("labels must have one structured row per waveform")
    required = {"signal_type", "signal_to_noise_ratio", "index"}
    if not required.issubset(labels.dtype.names or ()):
        raise ValueError("labels are missing signal_type, signal_to_noise_ratio or index")
    y = labels["signal_type"]
    if y.dtype.kind not in "iu" or np.any((y < 0) | (y >= 5)):
        raise ValueError("RadChar signal_type must be an integer in 0..4")
    if not np.isfinite(labels["signal_to_noise_ratio"]).all():
        raise ValueError("SNR metadata must be finite")
    groups = None
    for key in ("group_id", "group"):
        if key in (labels.dtype.names or ()):
            groups = np.asarray(labels[key])
            break
    if groups is None and "group_ids" in handle:
        groups = np.asarray(handle["group_ids"][:])
    if groups is not None and (groups.shape != (len(labels),) or groups.dtype.kind not in "iu"):
        raise ValueError("group identifiers must be a length-N integer array")
    return labels, groups


def read_iq_rows(dataset, indices: np.ndarray, *, block_rows: int = 1024) -> np.ndarray:
    """Read only selected rows, with bounded HDF5 selections and float32 output."""
    out = np.empty((len(indices), dataset.shape[1]), dtype=np.complex64)
    for start in range(0, len(indices), block_rows):
        stop = min(start + block_rows, len(indices))
        out[start:stop] = dataset[indices[start:stop]]
    if not np.isfinite(out).all():
        raise ValueError("waveforms contain non-finite samples")
    return out


def synthetic_radar_fixture(n: int = 6000, *, n_samples: int = 512,
                            seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent pulse realizations for CI; this is NOT the RadChar dataset.

    Five families use unmodulated, binary Barker, quadriphase-coded, Frank and
    chirped pulses. Phase, circular delay, carrier offset, amplitude, pulse count,
    width and noise are sampled independently of class. This simplified generator
    is only a reproducible engineering test, never evidence of state of the art.
    """
    if n < 30 or n_samples < 128:
        raise ValueError("fixture requires at least 30 waveforms of length >=128")
    rng = np.random.default_rng(seed)
    y = np.arange(n, dtype=np.int64) % 5
    rng.shuffle(y)
    snr_db = rng.choice(np.arange(-12, 19, 6, dtype=np.float32), n)
    iq = np.empty((n, n_samples), dtype=np.complex64)
    barker = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    frank = np.exp(2j * np.pi * np.outer(np.arange(4), np.arange(4)).ravel() / 4)
    axis = np.arange(n_samples)
    for row, label in enumerate(y):
        width = int(rng.integers(max(16, n_samples // 16), max(17, n_samples // 10)))
        pri = int(rng.integers(width + max(4, n_samples // 32), width + max(8, n_samples // 16)))
        pulses = int(rng.integers(2, 7))
        phase = rng.uniform(-np.pi, np.pi)
        shift = int(rng.integers(n_samples))
        carrier = rng.uniform(-0.005, 0.005)
        amplitude = np.exp(rng.uniform(-0.7, 0.7))
        chirp_rate = rng.uniform(3, 8)
        wave = np.zeros(n_samples, dtype=np.complex64)
        u = np.arange(width) / width
        if label == 0:
            pulse = np.ones(width)
        elif label == 1:
            pulse = barker[np.minimum((u * len(barker)).astype(int), len(barker) - 1)]
        elif label == 2:
            # Distinct quadriphase code, named generically rather than claiming
            # an exact implementation of RadChar's polyphase Barker generator.
            code = np.exp(0.5j * np.pi * np.array([0, 0, 1, 3, 2, 0, 3, 3, 1, 2, 1, 0, 2]))
            pulse = code[np.minimum((u * len(code)).astype(int), len(code) - 1)]
        elif label == 3:
            pulse = frank[np.minimum((u * len(frank)).astype(int), len(frank) - 1)]
        else:
            pulse = np.exp(1j * np.pi * chirp_rate * (u - 0.5) ** 2)
        for number in range(pulses):
            start = number * pri
            if start >= n_samples:
                break
            count = min(width, n_samples - start)
            wave[start:start + count] = pulse[:count]
        wave = amplitude * np.roll(wave, shift) * np.exp(1j * (phase + 2 * np.pi * carrier * axis))
        power = float(np.mean(np.abs(wave) ** 2))
        noise_std = np.sqrt(power / (2 * 10 ** (float(snr_db[row]) / 10)))
        noise = noise_std * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        iq[row] = wave + noise
    return iq, y, snr_db


def load_experiment_data(h5_path: str | Path | None = None, *, seed: int = 42,
                         train_cap: int | None = 3000, val_cap: int | None = 600,
                         test_cap: int | None = 1000, n_samples: int = 512,
                         synthetic_total: int = 6000) -> ExperimentData:
    """Load one real release or generate a reproducible synthetic CI fixture."""
    caps = dict(train=train_cap, val=val_cap, test=test_cap)
    if h5_path is None:
        iq, y, snr = synthetic_radar_fixture(synthetic_total, n_samples=n_samples, seed=seed)
        rows = np.arange(len(y), dtype=np.int64)
        groups = rows.copy()  # each fixture row is a freshly generated realization
        assignments = split_indices(len(y), seed=seed, groups=groups)
        splits = {}
        for name, indices in assignments.items():
            ids = cap_indices(indices, caps[name], seed=seed)
            splits[name] = SignalSplit(iq[ids], y[ids], snr[ids], rows[ids], groups[ids])
        manifest = {"source": "synthetic-radar-fixture-v1", "synthetic": True,
                    "source_sha256": _array_hash(iq), "population": len(y),
                    "snr_levels_db": [-12, -6, 0, 6, 12, 18],
                    "independence": "each row is independently generated; no replicated clean realization"}
    else:
        import h5py
        path = Path(h5_path).resolve(strict=True)
        with h5py.File(path, "r") as handle:
            labels, groups = read_radchar_metadata(handle)
            if handle["iq"].shape[1] != n_samples:
                raise ValueError(f"expected {n_samples} IQ samples per row")
            assignments = split_indices(len(labels), seed=seed, groups=groups)
            splits = {}
            for name, indices in assignments.items():
                ids = cap_indices(indices, caps[name], seed=seed)
                splits[name] = SignalSplit(read_iq_rows(handle["iq"], ids),
                    labels["signal_type"][ids].astype(np.int64),
                    labels["signal_to_noise_ratio"][ids].astype(np.float32), ids,
                    groups[ids] if groups is not None else None)
        manifest = {"source": path.name, "synthetic": False, "source_sha256": _file_hash(path),
                    "population": len(labels), "independence": "recorded-group-disjoint" if groups is not None
                    else "row-disjoint only; release has no independent-realization/group identifier"}
    manifest.update(split_version=SPLIT_VERSION, seed=seed, n_samples=n_samples,
                    train_fraction=0.8, val_fraction=0.1, test_fraction=0.1)
    manifest["splits"] = {}
    for name, split in splits.items():
        if not len(split):
            raise ValueError(f"{name} split is empty; increase the population/group count")
        manifest["splits"][name] = {"samples": len(split), "rows_sha256": _array_hash(split.row_ids),
            "iq_sha256": _array_hash(split.iq), "labels_sha256": _array_hash(split.y),
            "classes": {str(int(c)): int(np.sum(split.y == c)) for c in np.unique(split.y)},
            "group_count": len(np.unique(split.group_ids)) if split.group_ids is not None else None}
    return ExperimentData(**splits, manifest=manifest)
