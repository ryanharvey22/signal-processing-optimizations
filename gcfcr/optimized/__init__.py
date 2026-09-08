"""Compact autoencoder matching; NumPy inference, optional PyTorch training."""
from gcfcr.optimized.features import spectral_features
from gcfcr.optimized.autoencoder import LatentAutoencoder, build_prototype_bank, fit

__all__ = ["LatentAutoencoder", "build_prototype_bank", "fit", "spectral_features"]
