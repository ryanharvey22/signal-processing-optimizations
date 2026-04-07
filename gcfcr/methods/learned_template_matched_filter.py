"""
Experiment matrix: **Learned-template matched filter**.

Not implemented yet: learn references (e.g. AE latent -> decode to waveform),
then run classical correlation against those decoded templates for apples-to-apples
comparison with ``classical_matched_filter``.

Planned API sketch::

    build_decoded_template_bank(...) -> Tensor
    evaluate_decoded_mf(...)-> Tuple[float, int, int]
"""

from __future__ import annotations

__all__: list[str] = []
