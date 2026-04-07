#!/usr/bin/env python3
"""
Evaluate k-NN classification: encode val (or test) segments, query a latent bank (.npz).

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
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.data.pipeline import build_dataset
from gcfcr.models.autoencoder import RadCharIQAutoencoder, iq_to_real_stacked
from gcfcr.models.reference_bank import LatentReferenceBank


def _collate_radchar(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    iq_list, metas = zip(*batch)
    iq = torch.stack(iq_list, dim=0)
    y = torch.tensor([m["signal_type"] for m in metas], dtype=torch.long)
    x = iq_to_real_stacked(iq)
    return x, y


def _majority_vote(ref_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ref_labels: (B, k) long."""
    b, _ = ref_labels.shape
    out = torch.empty(b, dtype=torch.long)
    for i in range(b):
        c = torch.bincount(ref_labels[i], minlength=num_classes)
        out[i] = int(c.argmax())
    return out


@torch.no_grad()
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

    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt["config"]
    if cfg["dataset"] != args.dataset:
        print(
            f"Warning: checkpoint dataset={cfg['dataset']} but you passed --dataset {args.dataset}"
        )

    ae = RadCharIQAutoencoder(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    ae.load_state_dict(ckpt["ae_state"])
    ae.eval()

    num_classes = int(cfg["num_classes"])

    data = np.load(args.npz)
    bank = LatentReferenceBank(
        torch.from_numpy(np.asarray(data["z"], dtype=np.float32)),
        torch.from_numpy(np.asarray(data["label"], dtype=np.int64)),
        normalize=args.normalize_bank,
    )

    if args.dataset == "mnist":

        def collate(batch):
            imgs, y = zip(*batch)
            x = torch.stack(imgs, dim=0).view(len(imgs), -1)
            return x, torch.tensor(y, dtype=torch.long)

        ds = build_dataset("mnist", data_dir=args.data_dir, split=args.split)
    else:
        collate = _collate_radchar
        ds = build_dataset("radchar", data_dir=args.data_dir, split=args.split)

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        num_workers=0,
    )

    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        z = ae.encode(x)
        if args.normalize_bank:
            z = torch.nn.functional.normalize(z, dim=1)
        _, _, ref_lbl = bank.nearest(z.cpu(), k=args.k)
        pred = _majority_vote(ref_lbl, num_classes)
        pred = pred.to(y.device)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    acc = correct / max(total, 1)
    print(
        f"dataset={args.dataset} split={args.split}  k={args.k}  "
        f"normalize_bank={args.normalize_bank}  accuracy={acc:.4f}  ({correct}/{total})"
    )


if __name__ == "__main__":
    main()
