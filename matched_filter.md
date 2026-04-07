# How a matched filter works

*For the repo's comparison plan (classical vs learned-template vs latent discrimination), see [PROJECT.md](PROJECT.md).*

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

## What this repo implements

Theory above describes a **single known template** and optimal linear reception. The code implements **multi-class discrimination** with a **small template bank** (one template per class on RadChar/MNIST baselines), not pulse compression on a long continuous stream. That is still matched-filter *spirit*: score each hypothesis by correlation-like similarity to a reference waveform (or cosine similarity on MNIST).

*For how this baseline compares to latent retrieval and the overall experiment plan, see [PROJECT.md](PROJECT.md).*

Implementation module: `gcfcr/methods/classical_matched_filter.py` (CLI below is a thin wrapper).

### Script: `scripts/eval_matched_filter_baseline.py`

**Purpose:** Accuracy-only baseline on the same `build_dataset` train/val/test splits as the autoencoder pipeline.

| Dataset | Templates (from `split=train` only) | Query scoring | Decision |
|---------|-------------------------------------|---------------|----------|
| **RadChar** | Per-class **mean complex IQ** (512 samples), then **unit energy** per row | Hermitian inner product \(\langle \mu_c, x \rangle\); score \(= \lvert \cdot \rvert\) (magnitude ≈ **noncoherent** w.r.t. unknown phase) | `argmax` over 5 classes |
| **MNIST** | Per-class **mean flattened image** (784), **L2-normalized** per row | Cosine similarity to each template | `argmax` over 10 classes |

**RadChar eval detail:** Each query IQ is **unit-normalized** in energy before scoring (same as the benchmark script), so scores are comparable across amplitude.

**Caveat:** Templates are **estimated class means**, not a single known \(s(t)\) in AWGN. Optimality of the textbook matched filter does not strictly apply; this is a **lightweight classical competitor** for RadChar.

### Script: `scripts/benchmark_matched_filter_runtime.py` (RadChar only)

Same module as above: `gcfcr/methods/classical_matched_filter.py` (`run_radchar_cpu_runtime_benchmark`).

**Purpose:** Same template construction as RadChar in `eval_matched_filter_baseline.py`, plus **timed** discrimination (per-frame latency, split throughput, accuracy).

**Templates:** Again **5 class-mean** complex waveforms from train, **unit energy** per template.

**Modes:**

| Mode | Flag | What runs | Notes |
|------|------|-----------|--------|
| **Time-domain** | `--mode time` | For each query and class: \(\lvert \sum_n x[n]\,\mu_c^*[n] \rvert\) after unit-normalizing \(x\) | One inner product per class; **no explicit lag search** (aligned 512-sample frame). |
| **FFT-domain** | `--mode fft` (default) | `FFT(x) * conj(FFT(μ_c))` then `IFFT`; score \(= \max_k \lvert \text{corr}[k] \rvert\) over lag \(k\) | Implements **circular** correlation via FFT; **peak over lag** is the score (closer to “unknown delay within the frame” than a single dot product). |

**CPU:** Intended for `--device cpu` with `--cpu-threads` set for reproducible single-thread timing.

**Not implemented yet (planned baselines per [PROJECT.md](PROJECT.md)):** Multi-template or full-train banks, learned decoded templates, Doppler grids, overlap-add streaming, or CFAR-style detection thresholds.

---

## Further reading (standard names)

- **Correlation receiver**, **matched filter**, **pulse compression** (radar LFM + matched filter).
- Kay, *Fundamentals of Statistical Signal Processing* (detection theory).
- Van Trees, *Detection, Estimation, and Modulation Theory*.

