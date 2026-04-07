"""
One package per row of the experiment matrix in ``PROJECT.md``.

Implemented
-----------
- ``classical_matched_filter`` — class-mean templates, waveform correlation/cosine.
- ``latent_template_retrieval`` — encoder + ``LatentReferenceBank`` k-NN.

Stubs (implementations planned)
-------------------------------
- ``waveform_nearest_neighbor``
- ``learned_template_matched_filter``
- ``neural_classifier_baseline``

Shared helpers (not a method)
-----------------------------
- ``_collate`` — batch collate functions used by multiple methods.
"""

from gcfcr.methods import classical_matched_filter, latent_template_retrieval

__all__ = [
    "classical_matched_filter",
    "latent_template_retrieval",
]
