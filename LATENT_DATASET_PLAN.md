# Plan: building the latent reference dataset

This document is the **step-by-step plan** to go from raw RadChar / MNIST waveforms to a **saved corpus of latent vectors** \(z\) plus metadata, ready to load into `LatentReferenceBank` or offline analysis.

---

## Outcome (what “done” looks like)

| Deliverable | Description |
|-------------|-------------|
| **Trained weights** | Checkpoint of `RadCharIQAutoencoder` after training (encoder + decoder, or encoder-only export for deployment). |
| **Latent table** | Matrix `Z` of shape `(N_ref, L)` float32, one row per reference template. |
| **Metadata** | Parallel arrays or columns: at minimum **class label** (`signal_type` or MNIST digit); for RadChar also **`snr_db`**, optional original **`index`**, **`split`** (`train`/`val`). |
| **Manifest** | Small JSON/YAML: `input_dim`, `latent_dim`, `dataset` (`radchar`/`mnist`), RadChar file hash or version, train split seed, AE training hyperparameters, path to checkpoint. |

**Leakage rule:** References that define “what is in the library” should come from the **training split only** (or a dedicated **library** split carved from train). **Validation / test** segments are **queried** against that bank—they must **not** be copied into the bank rows used to tune \(\tau\) or report retrieval accuracy, unless you explicitly want a transductive experiment (document it).

---

## Phase 0 — Fix conventions

1. **RadChar:** Input \(x \in \mathbb{R}^{1024}\) via `iq_to_real_stacked` (unchanged).
2. **MNIST:** `image.view(-1, 784)` for `RadCharIQAutoencoder(input_dim=784)`.
3. **Latent dim \(L\):** Start with **64** (or **32** if bank size is huge and memory matters).
4. **Normalization (optional):** Consider **per-channel** standardization of \(x\) on **train** statistics only; save `mean`/`std` in the manifest for live inference. If skipped, document that distances are in raw amplitude space.

---

## Phase 1 — Train the autoencoder

**Objective:** Minimize reconstruction error so the encoder is a sensible compressor; optionally add a **discriminative** term so \(z\) separates classes.

| Step | Action |
|------|--------|
| 1.1 | Load data with `build_dataset("radchar" \| "mnist", split="train", ...)`. Use a **deterministic** train/val split (already seeded in loaders where applicable). |
| 1.2 | Initialize `RadCharIQAutoencoder(input_dim=D, latent_dim=L, hidden_dims=(512, 256))`. |
| 1.3 | **Loss v1:** `MSE(x_hat, x)` only. **Loss v2 (recommended before freezing bank):** add `α * CrossEntropy(classifier(z), y)` with a small linear head on \(z\), or contrastive loss—tune \(\alpha\) so reconstruction does not collapse. |
| 1.4 | Optimizer Adam, batch size 256–1024 (memory permitting), train until val MSE plateaus. |
| 1.5 | Save **best val MSE** checkpoint (and optionally best **val classification** if using joint loss). |

**Deliverable:** `checkpoints/ae_radchar_L64.pt` (example path).

---

## Phase 2 — Encode the reference split

| Step | Action |
|------|--------|
| 2.1 | Load **best** checkpoint; set `model.eval()`. |
| 2.2 | Iterate **train** split (or chosen **library** subset): batch forward, `z = model.encode(x)` (no grad). |
| 2.3 | Accumulate `Z`, `y_class`, and for RadChar `snr_db`, `orig_index` from `meta`. |
| 2.4 | Optional: **L2-normalize** rows of `Z` if using cosine-style geometry (`LatentReferenceBank(..., normalize=True)`). |
| 2.5 | Save atomically, e.g. **`data/latent/radchar_train_refs_L64.npz`** with keys `z`, `signal_type`, `snr_db`, `row_index` (into RadChar HDF5 or dataloader order). |

**Size sanity:** RadChar-Tiny \(N=50k\), \(L=64\), float32 → \(50k \times 64 \times 4 \approx 12.8\) MB for `Z` alone.

---

## Phase 3 — Optional derived banks

| Variant | Use case |
|---------|----------|
| **Class centroids** | One row per class: \(\mu_c = \mathrm{mean}(z \mid y=c)\). Smaller bank, faster NN, less template diversity. |
| **Stratified subsample** | Cap references per `(signal_type, snr_bin)` to balance the bank. |
| **k-means per class** | \(k\) prototypes per class for multi-modal latents. |

Save as separate `.npz` files with the same manifest schema + note `variant: centroid | subsample | kmeans`.

---

## Phase 4 — Evaluation hooks

1. **Encode val/test** segments (no addition to bank): for each \(z_{\text{query}}\), run `LatentReferenceBank(Z_train, labels_train).nearest(z_query, k=5)`.
2. **Classification metric:** majority vote of neighbor labels vs. true class — script: `scripts/eval_nn_retrieval.py`.
3. **Detection metric (TODO):** distribution of \(d^\*\) on **noise-only** or **held-out off-library** segments vs. in-library; sweep \(\tau\) for ROC / EER.

---

## Phase 5 — Code to implement (checklist)

- [x] `scripts/train_autoencoder.py` — MNIST (default) and RadChar; `--export-npz` writes train-bank `.npz` + `.manifest.json`.
- [x] `scripts/eval_nn_retrieval.py` — k-NN accuracy on val/test against a train `.npz` bank.
- [ ] `scripts/export_latent_dataset.py` — optional standalone re-export from checkpoint without retraining.
- [ ] (Optional) `gcfcr/io/latent_npz.py` — `load_latent_npz(path) -> LatentReferenceBank` helper.

---

## Open decisions (fill in as you go)

| Decision | Options | Notes |
|----------|---------|--------|
| Bank source | Full train vs. subsample vs. centroids | Full train is simplest for v1. |
| Normalize \(z\)? | Yes (L2) vs. no | Affects \(\tau\) calibration; pick one and keep manifest honest. |
| Joint loss | MSE-only first vs. early CE | If NN accuracy is poor on MSE-only, add small CE on \(z\). |
| MNIST first? | Yes for pipeline debug | Same scripts, `input_dim=784`. |

---

## Related files

- Models: `gcfcr/models/autoencoder.py`, `gcfcr/models/reference_bank.py`
- Data: `gcfcr/data/pipeline.py`
- Spec: [PROBLEM_AND_APPROACH.md](PROBLEM_AND_APPROACH.md)
