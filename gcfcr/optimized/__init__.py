"""Compact autoencoder matching; NumPy inference, optional PyTorch training."""
from gcfcr.optimized.features import spectral_features
from gcfcr.optimized.autoencoder import LatentAutoencoder, build_prototype_bank, fit

__all__ = ["LatentAutoencoder", "build_prototype_bank", "fit", "spectral_features"]

from gcfcr.optimized.conv_autoencoder import ConvLatentAutoencoder, temporal_features
__all__ += ["ConvLatentAutoencoder", "temporal_features"]

from gcfcr.optimized.api import load_model, with_reference_codes
__all__ += ["load_model", "with_reference_codes"]

from gcfcr.optimized.coherent_autoencoder import CoherentAutoencoder
__all__ += ["CoherentAutoencoder"]
