"""Shared batch collate helpers for method modules (not an experiment method itself)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import torch


def collate_mnist_flat(
    batch: List[Tuple[torch.Tensor, int]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    imgs, y = zip(*batch)
    x = torch.stack(imgs, dim=0).view(len(imgs), -1)
    return x, torch.tensor(y, dtype=torch.long)


def collate_radchar_iq(
    batch: List[Tuple[torch.Tensor, Dict[str, Any]]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    iq_list, metas = zip(*batch)
    iq = torch.stack(iq_list, dim=0)
    y = torch.tensor([m["signal_type"] for m in metas], dtype=torch.long)
    return iq, y
