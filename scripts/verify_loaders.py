#!/usr/bin/env python3
"""Smoke-test RadChar and MNIST loaders (run from repo root)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Repo root on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data",
        help="Cache / dataset root",
    )
    parser.add_argument("--radchar-only", action="store_true")
    parser.add_argument("--mnist-only", action="store_true")
    args = parser.parse_args()

    from gcfcr.data.pipeline import build_dataset

    if not args.radchar_only:
        print("MNIST train[0] …")
        ds = build_dataset("mnist", data_dir=args.data_dir, split="train")
        x, y = ds[0]
        print(f"  image shape={tuple(x.shape)} dtype={x.dtype} target={y}")

    if not args.mnist_only:
        print("RadChar …")
        try:
            ds = build_dataset(
                "radchar",
                data_dir=args.data_dir,
                split="train",
                radchar_max_samples=1000,
            )
        except FileNotFoundError as e:
            print(f"  skip: {e}")
            return
        iq, meta = ds[0]
        print(f"  iq shape={tuple(iq.shape)} dtype={iq.dtype}")
        print(f"  signal_type={meta['signal_type']} name={meta['signal_type_name']} snr={meta['snr_db']}")


if __name__ == "__main__":
    main()
