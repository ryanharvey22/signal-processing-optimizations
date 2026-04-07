# Latent reference autoencoders for waveform discrimination

Train an **autoencoder** on **RadChar** (synthetic radar IQ) or **MNIST** (prototype). Encode library waveforms into a **small latent code**, store them as a **reference bank**, then at runtime **encode the live segment** and decide **presence** and **identity** by **distance** in latent space (threshold + nearest neighbors)—**no FFT or classical matched filter** on that path.

## Docs

| File | Contents |
|------|----------|
| [PROBLEM_AND_APPROACH.md](PROBLEM_AND_APPROACH.md) | Full problem statement, AE pipeline, dataset plan |
| [LATENT_DATASET_PLAN.md](LATENT_DATASET_PLAN.md) | **Plan:** train AE → export latent reference `.npz` + manifest (phases, leakage, checklist) |
| [matched_filter.md](matched_filter.md) | Classical matched filter (baseline intuition) |
| [sidequest.md](sidequest.md) | Optional: skip full FFT when only *K* bins matter (sparse-coefficient deployment idea) |

## Code layout

- `gcfcr/data/` — load RadChar HDF5 or MNIST (`build_dataset`)
- `gcfcr/models/autoencoder.py` — `RadCharIQAutoencoder`, `iq_to_real_stacked`
- `gcfcr/models/reference_bank.py` — `LatentReferenceBank` (nearest-neighbor lookup)

## Setup

```bash
pip install -r requirements.txt
python scripts/verify_loaders.py
```

Place `RadChar-Tiny.h5` under `data/radchar/` (from [Kaggle](https://www.kaggle.com/datasets/abcxyzi/radchar-icassp-2023)) or set `RADCHAR_H5`.

## Training (MNIST first)

```bash
pip install -r requirements.txt   # includes tqdm for progress bars
python scripts/train_autoencoder.py --dataset mnist --epochs 30 --export-npz data/latent/mnist_train_L64.npz
```

You get an **epoch** bar plus **per-batch** train/val bars (running MSE). Use `--no-progress` for plain logs.

Checkpoint: `checkpoints/ae_mnist_L64_best.pt`. Latent bank: `data/latent/mnist_train_L64.npz` plus `.manifest.json`. Optional: `--joint-cls-weight 0.1` for a linear classifier on \(z\) alongside MSE.

RadChar (requires `data/radchar/RadChar-Tiny.h5`): `--dataset radchar`.

**k-NN retrieval (val or test vs. train bank):**

```bash
python scripts/eval_nn_retrieval.py \
  --checkpoint checkpoints/ae_mnist_L64_best.pt \
  --npz data/latent/mnist_train_L64.npz \
  --split val --k 5
```

**Matched-filter-style baseline** (same splits: class **mean templates** on train, correlation / cosine on eval — see script docstring for caveats):

```bash
python scripts/eval_matched_filter_baseline.py --dataset mnist --split val
python scripts/eval_matched_filter_baseline.py --dataset mnist --split test
# RadChar (needs HDF5):
python scripts/eval_matched_filter_baseline.py --dataset radchar --split val
```

Compare the printed **accuracy** to k-NN. On MNIST the linear template bank is usually **weaker** than a trained latent system; on **RadChar** the complex inner-product baseline is a **closer** classical competitor.

Load the bank for retrieval:

```python
import numpy as np
import torch
from gcfcr.models import LatentReferenceBank

d = np.load("data/latent/mnist_train_L64.npz")
bank = LatentReferenceBank(torch.from_numpy(d["z"]), torch.from_numpy(d["label"]))
```
