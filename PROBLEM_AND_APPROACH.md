# Latent autoencoder references for discrimination & detection

## Summary (primary approach)

We **compress waveforms into a small latent code** with an **autoencoder** trained on a reference corpus (**RadChar** radar IQ and/or **MNIST** as a prototype). **Templates** are **encoder outputs** \(z\) stored in a **reference bank**. For a **live** segment we use the **same encoder**: if \(z_{\text{live}}\) is **close** to the bank (e.g. minimum L2 distance \(d^\* < \tau\)), we infer **something from the library is present**; **identity** is **nearest neighbor** or **k-NN** among references. This is **learned embedding + retrieval**, not FFT + matched filter on the deployment path.

**Discrimination** remains the goal: separate confusable classes and control false alarms via **\(\tau\)** on latent distance (calibrated like a detector threshold).

---

## Motivation

Full correlation / FFT-based matching is costly and hand-engineered masks miss task-relevant structure. A **bottleneck autoencoder** learns **data-driven compression** into \(L \ll D\) dimensions; **inference** is one **encoder forward pass** plus a **small nearest-neighbor** search over stored codes.

---

## Primary method: autoencoder + reference bank

1. **Front-end (fixed):** Real vector per example — RadChar: \(x = [\Re(z),\Im(z)] \in \mathbb{R}^{1024}\); MNIST: **flatten** to \(\mathbb{R}^{784}\) (or a conv AE later).
2. **Train:** MLP encoder \(E_\theta: \mathbb{R}^D \to \mathbb{R}^L\), decoder reconstructs \(x\). Loss: MSE (optionally **joint** cross-entropy on \(z\), or **contrastive** loss so classes stay separated).
3. **Reference bank:** Encode all library / training templates → store \(z_i\) with labels (`signal_type`, digit, …). Optional: **class centroids** or **k** prototypes per class.
4. **Live:** Encode segment → \(d^\* = \min_i \|z_{\text{live}} - z_i\|\) (or distance to nearest centroid). **Present** if \(d^\* < \tau\); **class** from argmin or k-NN. **Absent / unknown** if far from all references on validation-style noise.
5. **Caveats:** Reconstruction-only training does **not** guarantee separability in \(z\); validate **confusion** and **SNR** sweeps. See [matched_filter.md](matched_filter.md) for classical baseline and when latent retrieval is a poor substitute.

**Code:** `gcfcr/models/autoencoder.py` (`RadCharIQAutoencoder`, `iq_to_real_stacked`), `gcfcr/models/reference_bank.py` (`LatentReferenceBank`).

**Operational plan:** [LATENT_DATASET_PLAN.md](LATENT_DATASET_PLAN.md) (train → encode train split → save `Z` + metadata → evaluate with val queries).

---

## Problem formulation (general)

- **Input:** Segments with labels defining hypotheses to separate (multi-class, binary vs. confuser, etc.).
- **Latent path:** Encoder \(E\), distance metric, threshold \(\tau\), optional k-NN.
- **Objective:** Good **discrimination** (accuracy, AUC, \(d'\), EER vs. \(\tau\)) at **small** \(L\) and **cheap** inference.

---

## Dataset plan

### Primary: RadChar

**[RadChar](https://github.com/abcxyzi/RadChar)** — **512 complex IQ** samples, **signal_type** (5 classes), **SNR**. [Kaggle download](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023). Loader expects `data/radchar/RadChar-Tiny.h5` or `RADCHAR_H5`.

### Backup: MNIST

Fast plumbing and **784-D** autoencoder baseline; weaker radar narrative.

### Optional: RadioML, RadSeg, others

- **RadioML** (RML2016.*) — comms modulations, AMC-style discrimination ([DeepSig](https://www.deepsig.ai/datasets), mirrors).
- **[RadSeg](https://github.com/abcxyzi/RadSeg)** — longer IQ, segmentation-style labels.
- **[PerceptionDataset](https://ieee-dataport.org/documents/perceptiondataset-synthetic-radar-perception-dataset-random-burst-targets)** — radar bursts, IQ / spectrogram.
- **[TSRD](https://huggingface.co/datasets/alan-turing-institute/turing-synthetic-radar-dataset)** — PDW / pulse-train style.

**Order:** RadChar first for results; MNIST for debugging; others as cross-checks.

---

## Robustness: interference and noise

Latent distance is reliable when **off-library** and **noise-only** segments map **far** from the bank in validation. **In-distribution** interference that mimics reference structure will still appear “close”—mitigate with **training augmentation**, **multi-condition** references, or **backend** checks.

---

## Design choices

| Topic | Decision |
|--------|----------|
| Task | Detection (\(d^\*<\tau\)) + **identity** (NN / k-NN); report operating points |
| Latent dim \(L\) | e.g. 32–128; trade reconstruction vs. capacity |
| Training | Reconstruction + optional **CE on \(z\)** or contrastive |
| Bank | All templates vs. **centroids** / k-means per class |
| Data | **Primary:** RadChar. **Backup:** MNIST. **Optional:** RadioML, etc. |

---

## Success criteria

- Reconstruction quality on held-out data; **NN / k-NN accuracy** vs. class baselines.
- **Detection:** ROC / EER for \(d^\*\) vs. noise-only or off-library segments.
- **Robustness:** SNR sweeps; failure modes when classes overlap in \(z\).

**Future (not in repo):** sparse selection in an explicit Fourier/wavelet basis; see [sidequest.md](sidequest.md) for skipping a full FFT if that direction returns.

---

## Status

Project is centered on the **latent autoencoder + reference bank** pipeline. See [LATENT_DATASET_PLAN.md](LATENT_DATASET_PLAN.md) for how we build the latent dataset; training/export scripts are the next implementation step.
