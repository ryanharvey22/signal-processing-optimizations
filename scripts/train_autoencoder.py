#!/usr/bin/env python3
"""
Train MLP autoencoder on MNIST (flattened 784-D) or RadChar (1024-D stacked IQ).

Example (MNIST):
  python scripts/train_autoencoder.py --dataset mnist --epochs 30 --export-npz data/latent/mnist_train.npz

Progress: tqdm bars on epochs and each train/val pass. Disable with --no-progress.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gcfcr.data.pipeline import build_dataset
from gcfcr.models.autoencoder import RadCharIQAutoencoder, iq_to_real_stacked


def _collate_radchar(batch: List[Tuple[torch.Tensor, Dict[str, Any]]]) -> Tuple[torch.Tensor, torch.Tensor]:
    iq_list, metas = zip(*batch)
    iq = torch.stack(iq_list, dim=0)
    y = torch.tensor([m["signal_type"] for m in metas], dtype=torch.long)
    x = iq_to_real_stacked(iq)
    return x, y


def _run_epoch(
    ae: RadCharIQAutoencoder,
    cls_head: nn.Module | None,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    joint_w: float,
    train: bool,
    *,
    pbar_desc: str = "",
    show_progress: bool = True,
) -> Tuple[float, float]:
    if train:
        ae.train()
        if cls_head is not None:
            cls_head.train()
    else:
        ae.eval()
        if cls_head is not None:
            cls_head.eval()

    total_mse = 0.0
    total_ce = 0.0
    n = 0
    it = loader
    if show_progress:
        it = tqdm(loader, desc=pbar_desc, leave=False, unit="batch")
    for x, y in it:
        x = x.to(device)
        y = y.to(device)
        if optimizer:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(train):
            x_hat, z = ae(x)
            mse = F.mse_loss(x_hat, x)
            loss = mse
            if joint_w > 0 and cls_head is not None:
                ce = F.cross_entropy(cls_head(z), y)
                loss = loss + joint_w * ce
                total_ce += float(ce.item()) * x.size(0)
            if train:
                loss.backward()
                optimizer.step()
        total_mse += float(mse.item()) * x.size(0)
        n += x.size(0)
        if show_progress:
            it.set_postfix(mse=f"{total_mse / max(n, 1):.5f}")
    return total_mse / n, (total_ce / n if joint_w > 0 else 0.0)


@torch.no_grad()
def _export_latent_npz(
    ae: RadCharIQAutoencoder,
    loader: DataLoader,
    device: torch.device,
    out_path: Path,
    manifest: Dict[str, Any],
    *,
    show_progress: bool = True,
) -> None:
    ae.eval()
    zs: List[torch.Tensor] = []
    ys: List[torch.Tensor] = []
    it = tqdm(loader, desc="export latents", unit="batch") if show_progress else loader
    for x, y in it:
        x = x.to(device)
        z = ae.encode(x)
        zs.append(z.cpu())
        ys.append(y.long())
    z_cat = torch.cat(zs, dim=0).numpy().astype(np.float32)
    y_cat = torch.cat(ys, dim=0).numpy().astype(np.int64)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, z=z_cat, label=y_cat)
    manifest_path = out_path.with_suffix(out_path.suffix + ".manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Wrote {out_path} (z shape {z_cat.shape}) and {manifest_path}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=("mnist", "radchar"), default="mnist")
    p.add_argument("--data-dir", type=Path, default=ROOT / "data")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=64)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--joint-cls-weight", type=float, default=0.0, help="If >0, add CE on latent with linear head")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir", type=Path, default=ROOT / "checkpoints")
    p.add_argument("--export-npz", type=Path, default=None, help="After training, encode train split and save .npz")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable tqdm bars (e.g. for logging to file)",
    )
    args = p.parse_args()
    show_pbar = not args.no_progress

    torch.manual_seed(args.seed)
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )

    if args.dataset == "mnist":
        input_dim = 784
        num_classes = 10

        def collate(batch):
            imgs, y = zip(*batch)
            x = torch.stack(imgs, dim=0).view(len(imgs), -1)
            yt = torch.tensor(y, dtype=torch.long)
            return x, yt

        train_ds = build_dataset("mnist", data_dir=args.data_dir, split="train")
        val_ds = build_dataset("mnist", data_dir=args.data_dir, split="val")
        train_ld = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate,
            num_workers=0,
        )
        val_ld = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate,
            num_workers=0,
        )
    else:
        input_dim = 1024
        num_classes = 5
        train_ds = build_dataset("radchar", data_dir=args.data_dir, split="train")
        val_ds = build_dataset("radchar", data_dir=args.data_dir, split="val")
        train_ld = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=_collate_radchar,
            num_workers=0,
        )
        val_ld = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=_collate_radchar,
            num_workers=0,
        )

    ae = RadCharIQAutoencoder(
        input_dim=input_dim,
        latent_dim=args.latent_dim,
        hidden_dims=tuple(args.hidden_dims),
    ).to(device)

    cls_head: nn.Module | None = None
    if args.joint_cls_weight > 0:
        cls_head = nn.Linear(args.latent_dim, num_classes).to(device)

    params = list(ae.parameters())
    if cls_head is not None:
        params += list(cls_head.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.dataset}_L{args.latent_dim}"
    best_path = args.checkpoint_dir / f"ae_{tag}_best.pt"

    best_val = float("inf")
    config = {
        "dataset": args.dataset,
        "input_dim": input_dim,
        "latent_dim": args.latent_dim,
        "hidden_dims": list(args.hidden_dims),
        "num_classes": num_classes,
        "joint_cls_weight": args.joint_cls_weight,
        "epochs": args.epochs,
        "seed": args.seed,
    }

    epoch_bar = tqdm(
        range(1, args.epochs + 1),
        desc="epochs",
        unit="epoch",
        disable=not show_pbar,
    )
    for epoch in epoch_bar:
        tr_mse, tr_ce = _run_epoch(
            ae,
            cls_head,
            train_ld,
            device,
            optimizer,
            args.joint_cls_weight,
            True,
            pbar_desc=f"{epoch}/{args.epochs} train",
            show_progress=show_pbar,
        )
        va_mse, va_ce = _run_epoch(
            ae,
            cls_head,
            val_ld,
            device,
            None,
            args.joint_cls_weight,
            False,
            pbar_desc=f"{epoch}/{args.epochs} val",
            show_progress=show_pbar,
        )
        if show_pbar:
            epoch_bar.set_postfix(train_mse=f"{tr_mse:.5f}", val_mse=f"{va_mse:.5f}")
        ce_note = f" train_ce={tr_ce:.4f} val_ce={va_ce:.4f}" if args.joint_cls_weight > 0 else ""
        print(
            f"epoch {epoch:03d}  train_mse={tr_mse:.6f}  val_mse={va_mse:.6f}{ce_note}"
        )
        if va_mse < best_val:
            best_val = va_mse
            payload = {
                "ae_state": ae.state_dict(),
                "cls_state": cls_head.state_dict() if cls_head else None,
                "config": config,
                "val_mse": va_mse,
            }
            torch.save(payload, best_path)
            print(f"  saved {best_path}")

    print(f"Best val MSE: {best_val:.6f}")

    if args.export_npz:
        ckpt = torch.load(best_path, map_location=device)
        ae.load_state_dict(ckpt["ae_state"])
        manifest = dict(ckpt["config"])
        manifest["checkpoint"] = str(best_path.resolve())
        manifest["val_mse_at_save"] = ckpt["val_mse"]
        export_ld = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=train_ld.collate_fn,
            num_workers=0,
        )
        _export_latent_npz(
            ae, export_ld, device, args.export_npz, manifest, show_progress=show_pbar
        )


if __name__ == "__main__":
    main()
