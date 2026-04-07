"""
Latent autoencoder reference codes for waveform discrimination (RadChar / MNIST).

Flow: train AE → build ``LatentReferenceBank`` → live encode + distance / threshold.
"""

__version__ = "0.2.0"
