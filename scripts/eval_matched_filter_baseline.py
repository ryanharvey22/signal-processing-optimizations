#!/usr/bin/env python3
"""
CLI for experiment matrix row: **classical matched filter** (class-mean templates).

Implementation: ``gcfcr.methods.classical_matched_filter``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.methods.classical_matched_filter import evaluate_accuracy


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("mnist", "radchar"), default="mnist")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    acc, correct, total = evaluate_accuracy(
        dataset=args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        device=device,
    )
    print(
        f"matched_filter_baseline  dataset={args.dataset}  split={args.split}  "
        f"accuracy={acc:.4f}  ({correct}/{total})"
    )


if __name__ == "__main__":
    main()
