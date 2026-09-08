#!/usr/bin/env python3
"""
Print empirical dataset statistics (class counts, SNR histograms for RadChar).

Run from repo root with data available, e.g.:
  python3 scripts/dataset_stats.py
  python3 scripts/dataset_stats.py --data-dir /path/to/data
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.data.radchar import SIGNAL_TYPE_NAMES
from gcfcr.optimized.data import read_radchar_metadata, split_indices


def _radchar_h5_path(data_dir: Path) -> Path | None:
    p = os.environ.get("RADCHAR_H5")
    if p:
        path = Path(p)
        return path if path.is_file() else None
    path = data_dir / "radchar" / "RadChar-Tiny.h5"
    return path if path.is_file() else None


def radchar_stats(h5_path: Path) -> None:
    import h5py

    with h5py.File(h5_path, "r") as f:
        labels, groups = read_radchar_metadata(f)
        n = labels.shape[0]

    st = labels["signal_type"].astype(np.int64)
    snr = labels["signal_to_noise_ratio"]

    print(f"RadChar file: {h5_path}")
    print(f"  total samples (in file): {n}")
    print("  per-class counts (full file):")
    for c in range(5):
        cnt = int((st == c).sum())
        pct = 100.0 * cnt / max(n, 1)
        print(f"    {c} {SIGNAL_TYPE_NAMES[c]}: {cnt} ({pct:.2f}%)")

    snr_vals, snr_cnt = np.unique(snr, return_counts=True)
    print(f"  unique SNR values: {len(snr_vals)}")
    print("  SNR (dB) histogram (value: count):")
    for v, c in zip(snr_vals, snr_cnt):
        print(f"    {int(v)}: {int(c)}")

    # Shared full-population assignments; metadata only, no waveform reads.
    for name, idx in split_indices(n, seed=42, groups=groups).items():
        st_s = st[idx]
        print(f"  {name} split size: {len(idx)} (train/val/test=0.8/0.1/0.1, seed=42)")
        for c in range(5):
            cnt = int((st_s == c).sum())
            print(f"    class {c}: {cnt}")


def mnist_stats(data_dir: Path) -> None:
    from gcfcr.data.pipeline import build_dataset

    for split in ("train", "val", "test"):
        ds = build_dataset("mnist", data_dir=data_dir, split=split)
        counts = np.zeros(10, dtype=np.int64)
        for i in range(len(ds)):
            _, y = ds[i]
            counts[int(y)] += 1
        print(f"MNIST split={split}  n={len(ds)}")
        for d in range(10):
            print(f"  digit {d}: {int(counts[d])}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--radchar-only", action="store_true")
    p.add_argument("--mnist-only", action="store_true")
    args = p.parse_args()

    if not args.radchar_only:
        print("=== MNIST ===")
        try:
            mnist_stats(args.data_dir)
        except Exception as e:
            print(f"  skip: {e}")

    if not args.mnist_only:
        print("\n=== RadChar ===")
        h5 = _radchar_h5_path(args.data_dir)
        if h5 is None:
            print(
                "  skip: no HDF5 at data/radchar/RadChar-Tiny.h5 or RADCHAR_H5"
            )
        else:
            radchar_stats(h5)


if __name__ == "__main__":
    main()
