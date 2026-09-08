"""Small deployment API; importing this module never loads Torch."""
from dataclasses import replace
import json
from pathlib import Path
import numpy as np
from .autoencoder import LatentAutoencoder, build_prototype_bank
from .conv_autoencoder import ConvLatentAutoencoder
from .coherent_autoencoder import CoherentAutoencoder

def load_model(path: str | Path) -> LatentAutoencoder | ConvLatentAutoencoder | CoherentAutoencoder:
    """Load a numeric NPZ model, dispatching by its explicit format marker."""
    with np.load(path, allow_pickle=False) as arrays:
        metadata = json.loads(str(arrays["metadata"].item()))
    if metadata.get("format") == "gcfcr-coherent-latent-ae":
        return CoherentAutoencoder.load(path)
    if metadata.get("format") == "gcfcr-conv-latent-ae":
        return ConvLatentAutoencoder.load(path)
    return LatentAutoencoder.load(path)  # This loader rejects unknown formats.

def with_reference_codes(model, codes, labels, *, prototypes_per_class=1):
    """Return a new model with references from this encoder's normalized codes.

    Codes must come from the same frozen encoder/frontend; equal dimensions alone
    do not establish that independently trained latent spaces are compatible.
    All original arrays remain owned by their original model.
    """
    codes = np.asarray(codes)
    if codes.ndim != 2 or codes.shape[1] != model.latent_dim:
        raise ValueError("reference codes must match the encoder latent dimension")
    bank, bank_labels = build_prototype_bank(codes, labels, prototypes_per_class)
    return replace(model, prototypes=bank, prototype_labels=bank_labels)
