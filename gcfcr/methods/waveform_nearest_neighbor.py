"""
Experiment matrix: **Nearest-neighbor waveform matcher** (train-bank references, raw IQ).

Not implemented yet: correlate each query against many stored waveforms (full or
subsampled train bank), then classify by nearest template or k-NN vote.

Planned API sketch::

    evaluate_waveform_nn(...)-> Tuple[float, int, int]  # accuracy, correct, total
"""

from __future__ import annotations

__all__: list[str] = []
