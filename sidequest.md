# Sidequest: choose coefficients *before* a full FFT

**Context:** The **primary** pipeline is **latent autoencoder + reference bank** (no FFT at runtime). This note applies to the **alternate sparse-coefficient / Fourier** track only.

## Thought

In that **alternate** pipeline we may compute a **full** Fourier (and wavelet) transform, then learn or optimize a **sparse mask** over all bins. In a **deployed** system, that still costs **O(N log N)** (or full wavelet cost) every time, which partly fights the goal of cheap real-time discrimination.

**Idea:** After training, we already know **which frequency indices** (or which wavelet subbands) matter. In production, can we **avoid computing the entire FFT** and instead evaluate **only those modes**—or approximate them with cheaper structure?

## Why it might be feasible

- A **subset of DFT bins** can be computed **directly** as inner products with complex exponentials (cost **O(K·N)** for *K* bins—better than full FFT only when *K* is small enough; cross-over depends on *N* and constants).
- **Goertzel**-style algorithms target **single-bin** or **few-bin** DFT evaluation without a full FFT (again attractive when *K* ≪ *N* and *N* is not huge).
- **Sparse / sublinear FFT** ideas exist in theory and specialized hardware contexts; practicality depends on signal model and *K*.
- For **wavelets**, keeping only certain subbands can correspond to a **partial decomposition** or **iterated filter bank** that stops early—structure depends on which subbands were selected.

## Open questions

- For our learned set of *K* indices, is **K** small enough that **K** explicit projections beat **one** FFT in wall time on target hardware?
- Are the chosen bins **arbitrary** (worst case for structure) or do they cluster (suggesting **bandpass + decimate** or **short FFT** on a downconverted band)?
- Does the **same** mask apply to all conditions, or do we need **conditional** indices (complicates “one cheap front end”)?
- **Numerical parity:** deployment path must match (or bound error vs.) the **full-transform + mask** path used in training.

## Status

**Sidequest** for a possible **sparse Fourier** branch—not required for the **AE + latent bank** build. Revisit if we add explicit frequency-domain masking and want a cheaper deployment FFT.
