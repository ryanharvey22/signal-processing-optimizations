# Datasets

This repo uses two datasets: **RadChar** (primary, synthetic radar IQ) and **MNIST** (debug / pipeline prototype). Both are loaded through [`gcfcr/data/pipeline.py`](gcfcr/data/pipeline.py) via `build_dataset("radchar" | "mnist", ...)`.

---

## RadChar (radar IQ)

### Role in the project

Main benchmark for waveform discrimination: complex baseband segments, multiple radar **signal types**, and **SNR** metadata. Matches the experiment matrix in [`PROJECT.md`](PROJECT.md).

### Source and files

- **Paper / release:** [RadChar](https://github.com/abcxyzi/RadChar) (ICASSP 2023); [Kaggle: radchar-icassp-2023](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023).
- **Loader path (default):** `data/radchar/RadChar-Tiny.h5`, or override with env var **`RADCHAR_H5`**, or `build_dataset(..., radchar_h5=...)`.
- **Variants (catalog):** RadChar-Tiny (~50k waveforms), Small, Baseline, Large (see upstream README). This codebase is tested most often against **RadChar-Tiny**.

### HDF5 layout (as read by [`gcfcr/data/radchar.py`](gcfcr/data/radchar.py))

| Dataset in file | Shape / type | Description |
|-----------------|----------------|-------------|
| `iq` | `(N, 512)`, `complex64` | Complex baseband IQ samples per example. |
| `labels` | structured array, length `N` | One row per example; field names below. |

**Structured label fields** (exposed in the `meta` dict returned by the dataset):

| Field | Type (typical) | Meaning |
|-------|----------------|---------|
| `signal_type` | int 0–4 | Class index for classification. |
| `signal_to_noise_ratio` | int | SNR in dB (exposed as `snr_db` in `meta`). |
| `number_of_pulses` | int | Pulse count parameter for the synthetic scene. |
| `pulse_width` | float | Pulse width (simulation parameter). |
| `time_delay` | float | Time delay parameter. |
| `pulse_repetition_interval` | float | PRI. |
| `index` | int | Original row / dataset index from the release. |

### Classes (5)

| ID | Name (in code) | Typical interpretation (radar literature) |
|----|----------------|-------------------------------------------|
| 0 | `coherent_pulse_train` | Coherent unmodulated pulse train |
| 1 | `barker_code` | Barker-coded pulse |
| 2 | `polyphase_barker_code` | Polyphase Barker |
| 3 | `frank_code` | Frank-coded waveform |
| 4 | `linear_frequency_modulated` | LFM (chirp) |

Constants: `gcfcr.data.radchar.SIGNAL_TYPE_NAMES`, `NUM_SIGNAL_TYPES = 5`.

### Sample shape and dtype

- One example: **`iq`** tensor shape **`(512,)`**, **`torch.complex64`**.
- For the MLP autoencoder path, IQ is stacked to real **`[Re, Im]`** → **`(1024,)`** float via `iq_to_real_stacked`.

### Published summary statistics (not file-specific)

From the RadChar release / paper (verify on your exact file if needed):

- **RadChar-Tiny:** on the order of **50,000** examples (upstream default).
- **SNR:** reported range roughly **−20 dB to +20 dB** across the dataset (exact discrete levels depend on the HDF5 contents).
- **Sampling:** baseband IQ at **512 samples** per segment; published context often cites **3.2 MHz** sampling for the synthetic generation (see upstream docs).

Class balance in Tiny is intended to support classification; **empirical** counts and SNR histograms depend on the file on disk.

### Empirical statistics (your machine)

With `RadChar-Tiny.h5` in place, run:

```bash
python3 scripts/dataset_stats.py --radchar-only --data-dir data
```

This prints:

- total `N` in the file;
- **per-class counts** on the full file;
- **SNR value histogram** (each discrete SNR and count);
- **train / val** sizes and per-class counts using the **same** `train_fraction=0.9` and **`seed=42`** as [`RadCharDataset`](gcfcr/data/radchar.py).

### Train / validation / test splits (this repo)

- **`train` / `val`:** single HDF5 is shuffled with `numpy.random.default_rng(seed)` (default **`seed=42`**), **`train_fraction=0.9`** (default). Train gets the first 90% of the permutation, val the rest. Indices are **sorted** for reproducible ordering.
- **`test`:** currently uses the **same index set as `val`** unless you point `build_dataset` at a **different HDF5** for held-out evaluation.
- **`all`:** every example in the loaded slice (after optional `max_samples` truncation).

`build_dataset(..., radchar_max_samples=N)` truncates to the **first N rows** of the file **before** the train/val split (same as `RadCharDataset`).

---

## MNIST (handwritten digits)

### Role in the project

Fast sanity checks for loaders, autoencoder plumbing (`input_dim=784`), and retrieval scripts without radar data.

### Source and files

- **Library:** `torchvision.datasets.MNIST`.
- **Root in this repo:** `data/mnist/` (created on first download).

### Official split sizes (canonical)

| Split | Size | Notes |
|-------|------|--------|
| Official **train** | **60,000** | 28×28 grayscale, 10 classes. |
| Official **test** | **10,000** | Held-out evaluation set. |

**Class balance (official train):** each digit **0–9** appears **6,000** times in the full 60k train set. **Test:** **1,000** images per digit.

### How this repo splits MNIST ([`gcfcr/data/mnist_data.py`](gcfcr/data/mnist_data.py))

- **`split="train"`:** 90% of official **train** → **54,000** examples. Subset chosen with `torch.randperm` and **`generator` seed 42** (reproducible).
- **`split="val"`:** remaining 10% of official train → **6,000** examples.
- **`split="test"`:** full official **test** → **10,000** examples.

Per-digit counts in **our** train/val are **approximately** 5,400 / 600 each but **not guaranteed** to be exact after the random split; use `dataset_stats.py` for exact counts.

### Sample shape and dtype

- **`image`:** `float32`, shape **`(1, 28, 28)`**, values in **`[0, 1]`** (`transforms.ToTensor()`).
- **`target`:** `int` in **`0 .. 9`**.
- For `RadCharIQAutoencoder(input_dim=784)`: flatten to **`(784,)`**.

### Empirical statistics (your machine)

```bash
python3 scripts/dataset_stats.py --mnist-only --data-dir data
```

Prints per-digit counts for **train**, **val**, and **test** as implemented by `MNISTDataset`.

---

## Side-by-side summary

| | RadChar | MNIST |
|---|---------|--------|
| **Modality** | Complex IQ (radar baseband) | Grayscale image |
| **Classes** | 5 (`signal_type`) | 10 (digits) |
| **Core input shape** | `(512,)` complex → `(1024,)` real stacked | `(1,28,28)` → `(784,)` flat |
| **Extra labels** | SNR, pulse params, PRI, etc. | None |
| **Default local path** | `data/radchar/RadChar-Tiny.h5` | `data/mnist/` |
| **Typical Tiny size** | ~50k (upstream) | 60k+10k official |

---

## Reproducibility checklist

- **RadChar:** fixed split depends on **`radchar_seed`** (default **42**) and **`radchar_train_fraction`** (default **0.9**).
- **MNIST:** fixed split depends on **`torch.Generator().manual_seed(42)`** inside `MNISTDataset`.
- For comparable benchmarks across methods, use the same **`data_dir`**, **`RADCHAR_H5`** (if any), and no accidental **`max_samples`** mismatch unless intentional.
