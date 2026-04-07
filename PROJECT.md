# Project overview

## Core question

**Is learned latent matching better than waveform-domain matching for speed at similar accuracy?**

We want a discriminator that is **faster than classical waveform matching** while keeping accuracy close, by moving comparisons to a **small latent representation** when that actually works. This repo is organized around that test. If latent-space discrimination is not competitive on accuracy and runtime, we should not continue down that path.

You do **not** always need classical matched filtering on learned templates—but you **can** use it on decoded waveforms to get a very **apples-to-apples** study (same correlation operator, different where the templates come from).

## Experiment matrix (same RadChar splits)


| Method                                | Idea                                                                                                                                                          |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Classical matched filter**          | Handcrafted or **class-mean** templates in waveform space; correlation / FFT path (implemented today for class means).                                        |
| **Nearest-neighbor waveform matcher** | **Train-bank** references; direct correlation or similarity in **raw IQ** (not yet implemented at full-bank scale).                                           |
| **Learned-template matched filter**   | Train an AE (or other model), **decode** reference embeddings back to waveform, then run **classical correlation** against those decoded templates (planned). |
| **Neural classifier baseline**        | Direct model output—**no** explicit matching step (planned or via a simple head on features).                                                                 |
| **Target: latent-template retrieval** | **Encode** the query once, compare in **latent space** against **reduced** references or prototypes (`eval_nn_retrieval.py` + bank export).                   |


## What we measure (three axes)

Every serious candidate should be reported on:

1. **Accuracy** (same splits, same task definition).
2. **Latency / turnaround** (per frame and full split or batch).
3. **Memory / storage footprint** of references (templates, latent bank, model weights).

That table is the project: **speed vs accuracy vs storage** across the matrix above.

---

## Primary objective and constraints

- **Primary metric:** discrimination turnaround time / latency on CPU.
- **Accuracy constraint:** allow small drop initially (about <=2% absolute), then recover with training.
- **Storage constraint:** template/reference representation should be much smaller than full waveform banks when possible.
- **Decision rule:** if latent methods do not show a meaningful runtime advantage at acceptable accuracy, de-prioritize them.

---

## Code layout (one module per experiment row)

| Matrix row | Python module | CLI (thin wrapper) |
|------------|---------------|--------------------|
| Classical matched filter | `gcfcr/methods/classical_matched_filter.py` | `scripts/eval_matched_filter_baseline.py`, `scripts/benchmark_matched_filter_runtime.py` |
| Latent-template retrieval | `gcfcr/methods/latent_template_retrieval.py` | `scripts/eval_nn_retrieval.py` |
| Nearest-neighbor waveform | `gcfcr/methods/waveform_nearest_neighbor.py` | (stub) |
| Learned-template MF | `gcfcr/methods/learned_template_matched_filter.py` | (stub) |
| Neural classifier | `gcfcr/methods/neural_classifier_baseline.py` | (stub) |

Shared batch helpers live in `gcfcr/methods/_collate.py` (not a matrix row).

**Training / export (supports latent row):** `scripts/train_autoencoder.py` — encoder/decoder + latent `.npz` bank.

The classical runtime CLI reports per-frame latency (`single_frame_ms`), split turnaround (`dataset_discrimination_s`), `fps`, and accuracy (`time` / `fft` modes).

---

## Data and evaluation protocol

- **Primary dataset:** [RadChar](https://github.com/abcxyzi/RadChar) ([Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023)). Schema, splits, and stats: [DATASETS.md](DATASETS.md).
- **Input shape:** complex IQ length 512 (or real-stacked 1024 for MLP input paths).
- **Leakage rule:** reference/template banks are built from **train only**; val/test are query-only.
- **Report for every method:**
  - accuracy
  - per-frame latency
  - total split turnaround
  - storage footprint of templates/references

---

## Near-term roadmap

1. Strengthen classical baselines from single mean templates to multi-template banks.
2. Add learned-template waveform benchmark (decoded references + matched filtering).
3. Benchmark latent bank reduction strategies (centroids/prototypes/subsample).
4. Compare all methods in one table: speed, accuracy, memory.
5. Keep only methods that meet the runtime goal under the accuracy guardrail.

