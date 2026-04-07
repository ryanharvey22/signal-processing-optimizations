# Latent reference autoencoders for waveform discrimination

**Goal:** Determine whether **latent-space discrimination** can beat or match classical waveform matching in **runtime** while keeping accuracy high. If latent matching cannot deliver a better speed/accuracy tradeoff, we should not use it.

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
