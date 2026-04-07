"""
Experiment matrix: **Latent-template retrieval** (encode once, k-NN in latent space).

Uses ``RadCharIQAutoencoder`` + ``LatentReferenceBank``. Train/export via
``scripts/train_autoencoder.py``; evaluation entry point here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Literal, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from gcfcr.data.pipeline import build_dataset
from gcfcr.models.autoencoder import RadCharIQAutoencoder, iq_to_real_stacked
from gcfcr.models.reference_bank import LatentReferenceBank


def collate_radchar_real_stacked(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    iq_list, metas = zip(*batch)
    iq = torch.stack(iq_list, dim=0)
    y = torch.tensor([m["signal_type"] for m in metas], dtype=torch.long)
    x = iq_to_real_stacked(iq)
    return x, y


def majority_vote(ref_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """ref_labels: (B, k) long."""
    b, _ = ref_labels.shape
    out = torch.empty(b, dtype=torch.long)
    for i in range(b):
        c = torch.bincount(ref_labels[i], minlength=num_classes)
        out[i] = int(c.argmax())
    return out


@torch.no_grad()
def evaluate_knn_retrieval(
    *,
    checkpoint: Path,
    npz_path: Path,
    dataset: Literal["mnist", "radchar"],
    split: Literal["val", "test"],
    data_dir: Path,
    k: int,
    batch_size: int,
    normalize_bank: bool,
    device: torch.device,
) -> Tuple[float, int, int]:
    """Returns (accuracy, correct, total)."""
    ckpt = torch.load(checkpoint, map_location=device)
    cfg = ckpt["config"]
    if cfg["dataset"] != dataset:
        print(
            f"Warning: checkpoint dataset={cfg['dataset']} but evaluate_knn_retrieval "
            f"dataset={dataset}"
        )

    ae = RadCharIQAutoencoder(
        input_dim=cfg["input_dim"],
        latent_dim=cfg["latent_dim"],
        hidden_dims=tuple(cfg["hidden_dims"]),
    ).to(device)
    ae.load_state_dict(ckpt["ae_state"])
    ae.eval()

    num_classes = int(cfg["num_classes"])

    data = np.load(npz_path)
    bank = LatentReferenceBank(
        torch.from_numpy(np.asarray(data["z"], dtype=np.float32)),
        torch.from_numpy(np.asarray(data["label"], dtype=np.int64)),
        normalize=normalize_bank,
    )

    if dataset == "mnist":

        def collate(batch):
            imgs, y = zip(*batch)
            x = torch.stack(imgs, dim=0).view(len(imgs), -1)
            return x, torch.tensor(y, dtype=torch.long)

        ds = build_dataset("mnist", data_dir=data_dir, split=split)
    else:
        collate = collate_radchar_real_stacked
        ds = build_dataset("radchar", data_dir=data_dir, split=split)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
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
        if normalize_bank:
            z = torch.nn.functional.normalize(z, dim=1)
        _, _, ref_lbl = bank.nearest(z.cpu(), k=k)
        pred = majority_vote(ref_lbl, num_classes)
        pred = pred.to(y.device)
        correct += int((pred == y).sum().item())
        total += y.size(0)

    acc = correct / max(total, 1)
    return acc, correct, total
