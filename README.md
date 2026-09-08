# Latent reference autoencoders for waveform discrimination

**Goal:** Determine whether **latent-space discrimination** can beat or match classical waveform matching in **runtime** while keeping accuracy high. If latent matching cannot deliver a better speed/accuracy tradeoff, we should not use it.

## Optimized embedded workflow

The new comparison pipeline lives in `gcfcr/optimized/`. It provides NumPy-only
inference, optional PyTorch training, train-only latent prototype banks, and an
allocation-free C99 export for x86-64, AArch64, Cortex-M4F, M7, and M33 builds.
No intrinsics are used. The decoder runs only during training.

Use [the comparison scope](docs/COMPARISON_SCOPE.md) and
[independent review protocol](REVIEW_PROTOCOL.md) when interpreting accuracy.
Cortex-M functional emulation and compilation do not measure physical-board
latency, energy, or interrupt-time behavior. A programmable sensor-host MCU is
different from an IMU's vendor-defined state machine or machine-learning block.

```bash
python -m pip install -r requirements-optimized.txt
# Training and tests only:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install pytest
python scripts/fetch_radchar_tiny.py
python scripts/optimized_filter.py fit --h5 data/radchar/RadChar-Tiny.h5 --out experiments/spectral --mf-bank-counts 1 4 16 64
python scripts/optimized_filter.py strengthen --experiment experiments/spectral --mf-bank-counts 256 1024
python scripts/optimized_filter.py fit-coherent --base-experiment experiments/spectral --out experiments/coherent --epochs 60
python scripts/assemble_optimized_comparison.py --experiments experiments/spectral experiments/coherent --out experiments/final
# Freeze all choices before opening test results:
python scripts/optimized_filter.py evaluate --experiment experiments/final
python scripts/optimized_filter.py native --experiment experiments/final --output experiments/native.json
python scripts/export_embedded.py --model experiments/final/latent.npz --output build/generated/ogae_model.h
```

`fit-conv` compares a spectral-reconstruction auxiliary loss with the same
encoder and matching bank trained with zero reconstruction loss. Its `iq`
frontend preserves coherent I/Q; the alternative `temporal` frontend uses
normalized power and adjacent complex products. Neither decoder reconstructs
the original IQ waveform. The spectral MLP remains available as a smaller,
cheaper alternative. Validation history retains all candidates and controls.

The coherent variant first applies learned complex filters, then forms response
power and runs a small real-valued encoder. This preserves coherent integration
within the first kernel and gives analytic global-phase invariance. The kernel
can still lose phase relationships beyond its length; only measurements establish
whether the resulting accuracy tradeoff is useful. Its complex coefficients are
immutable to keep the cached inference representation consistent with export.

The source dataset and split memberships are hashed. The full population is
split before caps are applied, so training, validation, and test rows remain
disjoint. No test data are used to choose the encoder, references, or epoch.
Reproducing a frozen benchmark checks saved predictions, scores, and latent
codes before creating an architecture benchmark bundle.

```python
from gcfcr.optimized import load_model, with_reference_codes

model = load_model("experiments/final/latent.npz")
codes = model.encode(iq_frames)       # complex64 array: (frames, 512)
scores = model.match_codes(codes)    # columns follow model.classes
labels = model.predict(iq_frames)

# Reference codes must come from this same frozen encoder/frontend.
custom = with_reference_codes(model, reference_codes, reference_labels,
                              prototypes_per_class=4)
custom.save("custom_signal_set.npz")
```

C inference uses caller-owned workspace and constant flash tables. See
[embedded integration](embedded/README.md) for sizing, compiler flags, numerical
parity, and physical-board cycle measurements. Python chunks intermediate
activations but retains the returned output array; the C path performs no heap
allocations.

The earlier experiment scripts remain below for continuity; the optimized
workflow above is the controlled comparison and deployment path.

## Docs

| File | Contents |
|------|----------|
| [PROJECT.md](PROJECT.md) | Comparison-first project objective, methods under test, evaluation protocol |
| [DATASETS.md](DATASETS.md) | RadChar and MNIST: schema, classes, splits, how to print empirical stats |
| [PAPERS.md](PAPERS.md) | Suggested reading: detection, matched filtering, radar/AMC ML, embeddings |
| [matched_filter.md](matched_filter.md) | Full matched-filter and FFT correlation story (classical baseline) |

## Code layout

- `gcfcr/data/` — load RadChar HDF5 or MNIST (`build_dataset`)
- `gcfcr/methods/` — **one module per experiment-matrix method** (classical MF, latent retrieval, stubs for the rest); see [PROJECT.md](PROJECT.md)
- `gcfcr/models/autoencoder.py` — `RadCharIQAutoencoder`, `iq_to_real_stacked`
- `gcfcr/models/reference_bank.py` — `LatentReferenceBank` (nearest-neighbor lookup)
- `scripts/*.py` — thin CLIs that call into `gcfcr.methods.*`

## Setup

```bash
pip install -r requirements.txt
python scripts/verify_loaders.py
```

Place `RadChar-Tiny.h5` under `data/radchar/` (from [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023)) or set `RADCHAR_H5`. Dataset details: [DATASETS.md](DATASETS.md). Empirical counts/SNR histograms: `python3 scripts/dataset_stats.py`.

## Main benchmark workflow (RadChar)

```bash
pip install -r requirements.txt
```

### 1) Classical matched-filter runtime baseline

```bash
# Per-frame latency + split timing (FFT/IFFT matched-filter path)
python3 scripts/benchmark_matched_filter_runtime.py \
  --split val --mode fft --batch-size 256 --cpu-threads 1

# Smaller quick subset run
python3 scripts/benchmark_matched_filter_runtime.py \
  --split val --mode fft --max-eval-samples 2000 --batch-size 256 --cpu-threads 1

# Full available holdout timing
python3 scripts/benchmark_matched_filter_runtime.py \
  --split all --mode fft --batch-size 256 --cpu-threads 1
```

The script prints:
- `single_frame_ms` (one-frame discrimination latency, averaged over repeats)
- `dataset_discrimination_s` (total turnaround for the selected split/subset)
- `mean_frame_ms` and `fps`
- `accuracy`

### 2) Latent retrieval baseline

Train/export a latent bank:

```bash
python3 scripts/train_autoencoder.py \
  --dataset radchar --epochs 30 \
  --export-npz data/latent/radchar_train_L64.npz
```

Evaluate latent k-NN against val/test:

```bash
python3 scripts/eval_nn_retrieval.py \
  --checkpoint checkpoints/ae_radchar_L64_best.pt \
  --npz data/latent/radchar_train_L64.npz \
  --split val --k 5
```

Load a latent bank manually:

```python
import numpy as np
import torch
from gcfcr.models import LatentReferenceBank

d = np.load("data/latent/radchar_train_L64.npz")
bank = LatentReferenceBank(torch.from_numpy(d["z"]), torch.from_numpy(d["label"]))
```
