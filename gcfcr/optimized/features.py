"""Compact, phase-invariant signal features with no training dependency."""
from __future__ import annotations

import numpy as np


def spectral_features(iq: np.ndarray, bins: int = 128) -> np.ndarray:
    """Return pooled log-power spectra for complex frames shaped ``(batch, samples)``.

    Pool adjacent, unshifted FFT bins. Dividing by total power removes amplitude;
    power removes global phase and circular time-shift dependence. These features
    intentionally discard phase, so an autoencoder of them cannot reconstruct IQ.
    A zero-energy frame maps to zero. Nonfinite input is rejected before the FFT;
    finite inputs too large for the FFT/power dtype raise ValueError on overflow.
    FFT precision depends on the NumPy/backend version; runtime measurements
    include this transform and its allocations, not only the small encoder.
    """
    x = np.asarray(iq)
    if x.ndim != 2 or not np.iscomplexobj(x):
        raise ValueError("iq must be a two-dimensional complex array")
    if not isinstance(bins, (int, np.integer)) or isinstance(bins, bool) or bins < 1:
        raise ValueError("bins must be a positive integer")
    if x.shape[1] == 0 or x.shape[1] % bins:
        raise ValueError("frame length must be positive and divisible by bins")
    if not np.isfinite(x).all():
        raise ValueError("iq must contain only finite values")
    spectrum = np.fft.fft(x, axis=-1)
    power = spectrum.real * spectrum.real + spectrum.imag * spectrum.imag
    pooled = power.reshape(len(x), bins, x.shape[1] // bins).sum(axis=-1)
    energy = pooled.sum(axis=-1, keepdims=True)
    if not np.isfinite(energy).all():
        raise ValueError("IQ power overflowed; scale frame amplitudes before extraction")
    scaled = np.divide(pooled, energy, out=np.zeros_like(pooled), where=energy > 0)
    scaled *= bins
    return np.log1p(scaled).astype(np.float32)


def feature_operation_counts(samples: int, bins: int) -> dict[str, int | str]:
    """Conventional arithmetic estimates, not hardware instruction counts.

    Complex FFT estimate is 5*N*log2(N) real operations for radix-2 sizes. Library
    FFT implementations vary; non-power-of-two lengths intentionally report no
    estimate. Logarithms, divisions and comparisons are reported separately.
    """
    if samples < 1 or bins < 1 or samples % bins:
        raise ValueError("samples must be positive and divisible by bins")
    radix_two = samples & (samples - 1) == 0
    return {
        "fft_real_ops_estimate": 5 * samples * (samples.bit_length() - 1) if radix_two else "unavailable",
        "frontend_add_multiply_ops": 4 * samples + bins - 1,
        "frontend_divisions": bins,
        "frontend_log1p": bins,
        "input_finiteness_checks": 2 * samples,
    }
