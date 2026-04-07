#!/usr/bin/env python3
"""
CLI for experiment matrix row: **latent-template retrieval** (k-NN in z).

Implementation: ``gcfcr.methods.latent_template_retrieval``.

Example:
  python scripts/eval_nn_retrieval.py \\
    --checkpoint checkpoints/ae_mnist_L64_best.pt \\
    --npz data/latent/mnist_train_L64.npz \\
    --k 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.methods.latent_template_retrieval import evaluate_knn_retrieval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--npz", type=Path, required=True, help="Latent bank from train split")
    p.add_argument("--dataset", choices=("mnist", "radchar"), default="mnist")
    p.add_argument("--split", choices=("val", "test"), default="val")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument(
        "--normalize-bank",
        action="store_true",
        help="L2-normalize bank and queries (cosine geometry via Euclidean on sphere)",
    )
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    acc, correct, total = evaluate_knn_retrieval(
        checkpoint=args.checkpoint,
        npz_path=args.npz,
        dataset=args.dataset,
        split=args.split,
        data_dir=args.data_dir,
        k=args.k,
        batch_size=args.batch_size,
        normalize_bank=args.normalize_bank,
        device=device,
    )
    print(
        f"dataset={args.dataset} split={args.split}  k={args.k}  "
        f"normalize_bank={args.normalize_bank}  accuracy={acc:.4f}  ({correct}/{total})"
    )


if __name__ == "__main__":
    main()
