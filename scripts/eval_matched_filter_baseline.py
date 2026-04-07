#!/usr/bin/env python3
"""
Matched-filter-style baseline using **class mean templates** from the train split.

- **MNIST (real):** For each class, mean flattened image on train; score = cosine
  similarity to each template; **argmax** → classification. This is a **linear template
  bank** (same spirit as correlation-based matching; not literal radar pulse compression).

- **RadChar (complex IQ):** Per-class **mean complex waveform** on train; score =
  **|⟨μ_c, x⟩|** with Hermitian inner product (magnitude handles unknown carrier phase
  roughly like a **noncoherent** correlation per template); **argmax** → class.

Uses the **same** `build_dataset` splits as the autoencoder pipeline (templates from
`split=train` only). Compare printed accuracy to ``eval_nn_retrieval.py``.

Caveat: classical matched-filter **optimality** is for a **known** signal in **AWGN**.
Here templates are **estimated** means; MNIST is not a radar IQ model. For RadChar this
is a much closer apples-to-apples competitor to latent k-NN.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.data.pipeline import build_dataset


def _collate_mnist(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, y = zip(*batch)
    x = torch.stack(imgs, dim=0).view(len(imgs), -1)
    return x, torch.tensor(y, dtype=torch.long)


def _collate_radchar_iq(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    iq_list, metas = zip(*batch)
    iq = torch.stack(iq_list, dim=0)
    y = torch.tensor([m["signal_type"] for m in metas], dtype=torch.long)
    return iq, y


@torch.no_grad()
def _fit_mnist_templates(
    loader: DataLoader, num_classes: int, dim: int, device: torch.device
) -> torch.Tensor:
    sums = torch.zeros(num_classes, dim, device=device)
    counts = torch.zeros(num_classes, device=device)
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        for c in range(num_classes):
            m = y == c
            if m.any():
                sums[c] += x[m].sum(dim=0)
                counts[c] += m.sum().float()
    mu = sums / counts.clamp(min=1).unsqueeze(1)
    return torch.nn.functional.normalize(mu, dim=1)


@torch.no_grad()
def _fit_radchar_templates(
    loader: DataLoader, num_classes: int, n_samples: int, device: torch.device
) -> torch.Tensor:
    """Complex templates (num_classes, n_samples)."""
    sums = torch.zeros(num_classes, n_samples, dtype=torch.complex64, device=device)
    counts = torch.zeros(num_classes, device=device)
    for iq, y in loader:
        iq = iq.to(device)
        y = y.to(device)
        for c in range(num_classes):
            m = y == c
            if m.any():
                sums[c] += iq[m].sum(dim=0).to(torch.complex64)
                counts[c] += m.sum().float()
    mu = sums / counts.clamp(min=1).unsqueeze(1).to(torch.complex64)
    # Unit-energy templates for fair comparison across classes
    denom = mu.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
    return mu / denom


@torch.no_grad()
def _mnist_scores(x: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    """Cosine sim: x (B,D) real, templates (C,D) row-normalized."""
    xn = torch.nn.functional.normalize(x, dim=1)
    return xn @ templates.T


@torch.no_grad()
def _radchar_scores(iq: torch.Tensor, templates: torch.Tensor) -> torch.Tensor:
    """|⟨t_c, x⟩| with t Hermitian: (B,C). iq (B,N) complex, templates (C,N)."""
    # (B,1,N) * (1,C,N).conj() -> sum N
    inner = (iq.unsqueeze(1) * templates.conj().unsqueeze(0)).sum(dim=-1)
    return inner.abs()


@torch.no_grad()
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

    if args.dataset == "mnist":
        num_classes = 10
        dim = 784
        collate = _collate_mnist
        train_ld = DataLoader(
            build_dataset("mnist", data_dir=args.data_dir, split="train"),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        templates = _fit_mnist_templates(train_ld, num_classes, dim, device)
        eval_ld = DataLoader(
            build_dataset("mnist", data_dir=args.data_dir, split=args.split),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        correct, total = 0, 0
        for x, y in eval_ld:
            x = x.to(device)
            y = y.to(device)
            scores = _mnist_scores(x, templates)
            pred = scores.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)
    else:
        num_classes = 5
        n_samples = 512
        collate = _collate_radchar_iq
        train_ld = DataLoader(
            build_dataset("radchar", data_dir=args.data_dir, split="train"),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        templates = _fit_radchar_templates(train_ld, num_classes, n_samples, device)
        eval_ld = DataLoader(
            build_dataset("radchar", data_dir=args.data_dir, split=args.split),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
        correct, total = 0, 0
        for iq, y in eval_ld:
            iq = iq.to(device)
            y = y.to(device)
            iq_u = iq / iq.abs().pow(2).sum(dim=1, keepdim=True).sqrt().clamp(min=1e-8)
            scores = _radchar_scores(iq_u, templates)
            pred = scores.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += y.size(0)

    acc = correct / max(total, 1)
    print(
        f"matched_filter_baseline  dataset={args.dataset}  split={args.split}  "
        f"accuracy={acc:.4f}  ({correct}/{total})"
    )


if __name__ == "__main__":
    main()
