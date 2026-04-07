# Suggested reading

Curated pointers for **detection**, **matched filtering / correlation**, **waveform discrimination**, and **learned** approaches that sit next to this repo’s experiment matrix. Order is roughly “foundations first,” then radar and comms-style classification, then embeddings and surveys.

This is not a literature review: use it as a **seed list** and follow references forward from each item.

---

## Detection and discrimination (theory)

These are the standard languages for “is a known signal present in noise?” and “which hypothesis?”

| Resource | Why read it |
|----------|-------------|
| S. Kay, *Fundamentals of Statistical Signal Processing, Volume II: Detection Theory* | Likelihood ratios, matched filter as detector for known signal in AWGN, composite hypotheses, GLRT structure, performance (ROC, deflection). |
| H. L. Van Trees, *Detection, Estimation, and Modulation Theory, Part I* | Classical detection and estimation; matched filter and Karhun–Loève viewpoints; deeper than a single chapter. |
| H. V. Poor, *An Introduction to Signal Detection and Estimation* | Shorter alternative to Van Trees; good for quick rigor on detectors and bounds. |
| L. L. Scharf, *Statistical Signal Processing* | Geometry of detection, subspaces, CFAR-style ideas in a unified linear-algebra framing. |

**Keywords to chase from here:** Neyman–Pearson, GLRT, coherent vs noncoherent integration, unknown phase, colored noise matched filter, CFAR.

---

## Matched filtering, correlation, and pulse compression (practice)

| Resource | Why read it |
|----------|-------------|
| M. A. Richards, *Fundamentals of Radar Signal Processing* | Pulse compression, matched filtering in radar, ambiguity functions, Doppler, practical receiver chains. |
| N. Levanon & E. Mozeson, *Radar Signals* | Waveform families (LFM, phase codes, etc.) that show up as **classes** in datasets like RadChar; good for intuition about what a “template” means physically. |
| IEEE / radar textbooks on **stretch processing**, **stretch**/**IFFT** receivers | Connects time-domain correlation to FFT implementations (same spirit as your `fft` scoring mode). |

**Keywords:** ambiguity function, pulse compression, stretch processing, replica correlation, replica mismatch.

---

## Radar datasets and multi-class waveform characterisation (close to RadChar)

| Resource | Why read it |
|----------|-------------|
| [RadChar](https://github.com/abcxyzi/RadChar) paper: *Multi-task Learning for Radar Signal Characterisation* ([arXiv:2306.13105](https://arxiv.org/abs/2306.13105)) | Describes the RadChar setting: IQ segments, multiple waveform families, SNR, and baseline deep models. Good anchor for “what ‘discrimination’ means here.” |
| Z. Geng et al., “Deep-Learning for Radar: A Survey,” *IEEE Aerospace and Electronic Systems Magazine*, 2022 ([IEEE Xplore](https://ieeexplore.ieee.org/document/9568916/)) | Broad map of where DL is used in radar (detection, classification, clutter, etc.); use the bibliography to drill into **classification** and **ATR** lines. |

---

## Automatic modulation classification and IQ with neural nets (analogous pipelines)

Much of the “IQ in, class out” and “embedding + classifier” literature lives under **AMC** and **wireless ML**, even when your end application is radar.

| Resource | Why read it |
|----------|-------------|
| T. J. O’Shea et al., “Over-the-Air Deep Learning Based Radio Signal Classification,” *IEEE JSAC* / related RML2016 work | Early, widely cited demonstration of CNNs on IQ / spectrogram inputs; useful for thinking about **representation** vs **template matching**. |
| Surveys on **automatic modulation classification** (search IEEE Xplore / arXiv for “AMC survey deep learning”) | Systematic comparison of handcrafted features vs end-to-end nets; metrics and failure modes (SNR sweep, unseen radios). |

**Keywords:** RML2016, RadioML, constellation, cyclostationary features, contrastive / metric learning on IQ.

---

## Embeddings, prototypes, and retrieval (latent / k-NN angle)

Relevant when you ask whether **latent nearest-neighbor** can replace or complement **waveform correlation**.

| Resource | Why read it |
|----------|-------------|
| K. Q. Weinberger & L. K. Saul, “Distance Metric Learning for Large Margin Nearest Neighbor Classification,” *JMLR* 2009 | Classic “make k-NN work in a learned space” framing. |
| F. Schroff et al., “FaceNet: A Unified Embedding for Face Recognition and Clustering,” *CVPR* 2015 | Triplet loss and embedding geometry at scale; analogy: **same class = small distance** in \(z\). |
| A. W. van der Merwe et al., “Open Set Recognition” line of work (search *IEEE TPAMI* / BMVC) | When “none of the above” matters for **detection** as well as **ID**—ties to threshold \(\tau\) on distance in your project notes. |

**Keywords:** metric learning, prototypical networks, few-shot, open-set recognition, contrastive loss.

---

## How to use this list with your repo

1. **Kay + Richards** give you the vocabulary to say *why* a matched filter baseline is there and where it stops being optimal (unknown template, colored noise, multi-template banks).
2. **RadChar paper** grounds the **task** and **labels** you actually load from HDF5.
3. **AMC / wireless DL** papers give comparators for **architecture** and **evaluation habits** (SNR sweeps, confusion, calibration).
4. **Metric learning / open-set** literature connects **latent banks + thresholds** to something closer to operational detection specs than raw accuracy alone.

If you add one file next to this list, keep a short `papers_read.bib` or Zotero folder and tag entries by matrix row: `classical_mf`, `waveform_nn`, `latent_retrieval`, `neural_classifier`.
