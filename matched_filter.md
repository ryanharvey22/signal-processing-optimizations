# How a matched filter works

## What problem it solves

You receive a noisy measurement and want to decide whether a **known waveform** s(t) (the “template” or “reference”) is present. A common model is **additive white Gaussian noise (AWGN)**:


x(t) = s(t) + n(t)


under hypothesis **signal present**, and x(t) = n(t) under **noise only**. The matched filter is the **linear** processing rule that **maximizes output signal-to-noise ratio (SNR)** at a chosen time instant—usually the moment you want to declare “pulse here” or sample the correlator output.

So: **same statistical goal as good detection**, restricted to **linear** filters and a known s(t).

## Time domain: correlation with the template

For discrete samples, let the (complex) signal be s[0],\ldots,s[N-1]. The **matched filter** output at lag k is (up to a scale) the **correlation** of the received data with the **time-reversed conjugate** of the template:


y[k] \propto \sum_{n} x[n] s^*[n - k]


Intuition: you are **sliding** the expected shape s along the data and asking, at each shift, “how much does this segment look like s?” Conjugation matters for **complex baseband (I/Q)** so phases line up correctly.

At the **correct alignment**, x \approx s + n, so the sum coherently adds the signal terms and grows like **energy of s**; noise terms add with random signs and grow more slowly (**\sqrt{N}** scaling). That is why SNR peaks at the match.

## Frequency domain

Convolution/correlation in time is **multiplication** in frequency (with appropriate conjugation for correlation). A matched filter can be implemented as **FFT → multiply by S^*(f) → IFFT** (again, up to scaling and conventions). The filter’s frequency response is **conjugate-matched** to the spectrum of the known signal.

## What you store vs. what you run

- **Stored reference:** the template s (or equivalently its spectrum / pulse compression kernel in radar).
- **Runtime:** filter the incoming data with the **matched** impulse response (correlation receiver). Peak output → candidate detection; threshold → binary decision.

## Limits (when it is “optimal”)

- **Assumes** you know s **exactly** (amplitude may be unknown; then an **energy** or **GLRT** variant is used).
- **AWGN:** matched filter is optimal **among linear filters** for SNR at one sample; under Gaussian noise and known signal in additive model, the **likelihood ratio** structure is closely related (correlation with s^*).
- **Colored noise:** you should use a **whitener** before matching, or match to a **noise-whitened** signal—otherwise “match s” is no longer SNR-optimal.
- **Unknown delay / Doppler:** you search over delays (and possibly Doppler) by **bank** of matched filters or FFT fast convolution over a grid.

## Link to this project (learned latent discrimination)

A classical matched filter uses **one** fixed linear projection tuned to **one** template and **one** noise model. This repo’s direction is an **autoencoder bottleneck** z \in \mathbb{R}^L plus a **reference bank** and **latent-distance** decisions—**multi-template**, learned compression, not the same optimality story as a matched filter under AWGN, but a practical retrieval-style detector when z is trained to separate hypotheses.

**Head-to-head in this repo:** Run `scripts/eval_matched_filter_baseline.py` (class-mean templates + correlation) against `scripts/eval_nn_retrieval.py` (latent k-NN) on the **same** train/val/test splits.

**Alternative in this repo:** Train an **autoencoder** on the reference corpus, store **encoder outputs** as a **latent reference bank**, and at runtime **encode** the live segment and **nearest-neighbor** (or prototype distance) in that space—**no FFT / explicit matched filter** in that branch. **Presence** can be declared when the live code is **close enough** to some reference (minimum distance below a threshold \tau); **far** from all references suggests absent or off-library. That is **learned embedding + retrieval**; it only behaves like good matching if training makes z discriminative. Details: `PROBLEM_AND_APPROACH.md` (primary latent pipeline); implementation: `gcfcr/models/autoencoder.py`, `reference_bank.py`.

---

## Further reading (standard names)

- **Correlation receiver**, **matched filter**, **pulse compression** (radar LFM + matched filter).
- Kay, *Fundamentals of Statistical Signal Processing* (detection theory).
- Van Trees, *Detection, Estimation, and Modulation Theory*.

