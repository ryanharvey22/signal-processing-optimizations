"""Held-out classification metrics and paired, group-aware uncertainty."""
import numpy as np

def classification_metrics(y, prediction, *, snr_db=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y):
        raise ValueError("nonempty matching label vectors required")
    classes = np.unique(np.concatenate((y, prediction)))
    conf = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(conf, (np.searchsorted(classes, y), np.searchsorted(classes, prediction)), 1)
    tp = conf.diagonal()
    denominator = conf.sum(0) + conf.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(len(classes), dtype=float), where=denominator > 0)
    result = {"samples": len(y), "correct": int(np.sum(y == prediction)),
              "accuracy": float(np.mean(y == prediction)), "macro_f1": float(f1.mean()),
              "classes": classes.tolist(), "confusion": conf.tolist(),
              "per_class_recall": {str(c): float(tp[i] / conf[i].sum()) if conf[i].sum() else None
                                   for i, c in enumerate(classes)}}
    if snr_db is not None:
        snr = np.asarray(snr_db)
        if snr.shape != y.shape or not np.isfinite(snr).all():
            raise ValueError("one finite SNR value per prediction required")
        result["per_snr"] = {str(float(s)): {"samples": int(np.sum(snr == s)),
            "accuracy": float(np.mean(prediction[snr == s] == y[snr == s]))} for s in np.unique(snr)}
    return result

def paired_accuracy_interval(y, candidate, reference, *, groups=None, seed=42, repetitions=2000):
    y, candidate, reference = map(np.asarray, (y, candidate, reference))
    if y.ndim != 1 or not len(y) or candidate.shape != y.shape or reference.shape != y.shape or repetitions < 100:
        raise ValueError("nonempty aligned predictions and at least 100 bootstrap draws required")
    difference = (candidate == y).astype(float) - (reference == y)
    if groups is None:
        inverse = np.arange(len(y))
    else:
        groups = np.asarray(groups)
        if groups.shape != y.shape:
            raise ValueError("one group per row required")
        _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    if len(counts) < 2:
        raise ValueError("at least two independent units required for an uncertainty interval")
    sums = np.bincount(inverse, weights=difference)
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for start in range(0, repetitions, 32):
        ids = rng.integers(len(counts), size=(min(32, repetitions - start), len(counts)))
        estimates[start:start + len(ids)] = sums[ids].sum(1) / counts[ids].sum(1)
    lo, hi = np.quantile(estimates, [0.025, 0.975])
    return {"difference": float(difference.mean()), "ci95": [float(lo), float(hi)],
            "independent_units": len(counts), "repetitions": repetitions,
            "method": "paired percentile bootstrap of groups" if groups is not None else
                      "paired percentile bootstrap of rows; unrecorded dependence is not modeled"}
