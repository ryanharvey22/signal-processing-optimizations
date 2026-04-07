"""
Primary model path: MLP autoencoder for building **latent reference codes**.

Train on RadChar (stacked Re/Im IQ) or MNIST (``input_dim=784`` flattened images).
Encoder output + ``LatentReferenceBank`` implements live **encode → nearest-neighbor /
threshold** discrimination without FFT or a classical matched filter.

See ``PROJECT.md`` (primary method).
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def iq_to_real_stacked(iq: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    iq
        Complex tensor, shape ``(batch, 512)`` for RadChar.

    Returns
    -------
    Real tensor ``(batch, 1024)`` = concat(Re, Im) along last dim.
    """
    if iq.dtype not in (torch.complex64, torch.complex128):
        raise TypeError(f"expected complex iq, got {iq.dtype}")
    re = iq.real
    im = iq.imag
    return torch.cat([re, im], dim=-1)


class RadCharIQAutoencoder(nn.Module):
    """MLP autoencoder: default ``input_dim=1024`` (RadChar IQ); use 784 for MNIST flat."""

    def __init__(
        self,
        input_dim: int = 1024,
        latent_dim: int = 64,
        hidden_dims: Sequence[int] = (512, 256),
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        enc_dims: List[int] = [input_dim, *hidden_dims, latent_dim]
        dec_dims: List[int] = [latent_dim, *reversed(hidden_dims), input_dim]

        def mlp(sizes: List[int], end_with_act: bool) -> nn.Sequential:
            layers: List[nn.Module] = []
            for i in range(len(sizes) - 1):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                if end_with_act or i < len(sizes) - 2:
                    layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.encoder = mlp(enc_dims, end_with_act=False)
        self.decoder = mlp(dec_dims, end_with_act=False)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def forward_from_iq(self, iq: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = iq_to_real_stacked(iq)
        return self.forward(x)

    def reconstruction_loss(self, x: torch.Tensor) -> torch.Tensor:
        x_hat, _ = self.forward(x)
        return F.mse_loss(x_hat, x)

    def reconstruction_loss_from_iq(self, iq: torch.Tensor) -> torch.Tensor:
        x = iq_to_real_stacked(iq)
        return self.reconstruction_loss(x)
