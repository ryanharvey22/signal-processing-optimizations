"""
Store encoded template/reference vectors and query live codes by nearest neighbor.

Pairs with a trained encoder (see ``gcfcr.models.autoencoder.RadCharIQAutoencoder``):
offline fill the
bank from the dataset; at runtime encode the live segment and call ``nearest``.
"""

from __future__ import annotations

from typing import Tuple

import torch


class LatentReferenceBank:
    """
    Reference codes shape ``(N, L)``, labels shape ``(N,)`` (e.g. class id per template).
    """

    def __init__(
        self,
        codes: torch.Tensor,
        labels: torch.Tensor,
        *,
        normalize: bool = False,
    ) -> None:
        if codes.dim() != 2:
            raise ValueError(f"codes must be (N, L), got {tuple(codes.shape)}")
        if labels.dim() != 1 or labels.shape[0] != codes.shape[0]:
            raise ValueError("labels must be 1D with same length as codes")
        self.codes = codes.detach().float()
        self.labels = labels.detach().long()
        self.normalize = normalize
        if normalize:
            self.codes = torch.nn.functional.normalize(self.codes, dim=1)

    @torch.no_grad()
    def nearest(
        self,
        z: torch.Tensor,
        *,
        k: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        z
            Query codes, shape ``(batch, L)``.
        k
            Number of neighbors per query.

        Returns
        -------
        distances
            ``(batch, k)`` — ascending L2 (or cosine distance if ``normalize=True``: use
            ``2 - 2 * inner`` for unit vectors; here we still use ``cdist`` on normalized
            vectors = chord length related to cosine).
        indices
            ``(batch, k)`` row indices into the reference set.
        ref_labels
            ``(batch, k)`` label of each neighbor.
        """
        if z.dim() != 2 or z.shape[1] != self.codes.shape[1]:
            raise ValueError(
                f"z must be (batch, {self.codes.shape[1]}), got {tuple(z.shape)}"
            )
        q = z.detach().float()
        if self.normalize:
            q = torch.nn.functional.normalize(q, dim=1)
        dist = torch.cdist(q, self.codes, p=2)
        k = min(k, self.codes.shape[0])
        dist_k, idx = dist.topk(k, largest=False, dim=1)
        ref_labels = self.labels[idx]
        return dist_k, idx, ref_labels

    @classmethod
    def from_tensors(
        cls,
        codes: torch.Tensor,
        labels: torch.Tensor,
        *,
        normalize: bool = False,
    ) -> "LatentReferenceBank":
        return cls(codes, labels, normalize=normalize)
