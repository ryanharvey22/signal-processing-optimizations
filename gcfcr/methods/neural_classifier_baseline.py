"""
Experiment matrix: **Neural classifier baseline** (direct logits / no explicit matching).

Not implemented yet: e.g. small MLP or linear head on encoder features with softmax CE,
for speed/accuracy comparison without retrieval.

Planned API sketch::

    train_classifier(...)
    evaluate_classifier(...)-> Tuple[float, int, int]
"""

from __future__ import annotations

__all__: list[str] = []
